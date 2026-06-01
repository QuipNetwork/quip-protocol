# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU miner using Metal/MPS with IsingFeeder streaming pipeline.

Mirrors GPUMiner (gpu_miner.py) architecture: IsingFeeder for
background model generation, MetalScheduler for core budget and
IOKit-based yielding, and batched streaming dispatch via
MetalSASampler.sample_ising_streaming().
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from GPU.metal_sa import MetalSASampler
from GPU.metal_scheduler import DutyCycleController, MetalScheduler


def get_gpu_core_count() -> int:
    """Detect Apple Silicon GPU core count via ioreg."""
    try:
        result = subprocess.run(
            "ioreg -l | grep gpu-core-count",
            shell=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.stdout:
            for line in result.stdout.splitlines():
                if 'gpu-core-count' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        return int(parts[1].strip())
    except Exception as e:
        raise RuntimeError(
            f"Failed to detect GPU core count: {e}",
        )

    raise RuntimeError(
        "Could not find gpu-core-count in ioreg output",
    )


# Pipeline stall timeout constants (match gpu_miner.py)
_PIPELINE_STALL_FLOOR = 60.0
_SEC_PER_SWEEP = 0.03
_STALL_SAFETY_FACTOR = 5.0


class MetalMiner(BaseMiner):
    """Metal GPU miner with IsingFeeder streaming pipeline.

    Architecture mirrors GPUMiner: background model generation
    via IsingFeeder, core budget via MetalScheduler, and batched
    multi-problem dispatch via sample_ising_streaming().
    """

    # Metal MPS strategy: fewer sweeps, more reads
    ADAPT_MIN_SWEEPS = 64
    ADAPT_MAX_SWEEPS = 512
    ADAPT_MIN_READS = 32
    ADAPT_MAX_READS = 1024

    def __init__(self, miner_id: str, topology=None, **cfg):
        gpu_util = cfg.pop('utilization', cfg.pop('gpu_utilization', 100))
        yielding = cfg.pop('yielding', True)
        # Remove CUDA-only keys that flow through common_cfg
        cfg.pop('sms_per_nonce', None)

        try:
            sampler = MetalSASampler(topology=topology)
            super().__init__(
                miner_id, sampler, miner_type="GPU-Metal",
            )
            sampler.logger = self.logger
        except Exception as e:
            from CPU.sa_sampler import (
                SimulatedAnnealingStructuredSampler,
            )
            sampler = SimulatedAnnealingStructuredSampler(
                topology=topology,
            )
            super().__init__(
                miner_id, sampler, miner_type="CPU-FALLBACK",
            )
            self.logger.warning(
                "Metal GPU init failed, falling back to "
                "CPU: %s", e,
            )
            return

        if not 0 < gpu_util <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {gpu_util}",
            )
        self.gpu_utilization = gpu_util

        self.gpu_core_count = get_gpu_core_count()
        self._scheduler = MetalScheduler(
            gpu_core_count=self.gpu_core_count,
            gpu_utilization_pct=gpu_util,
            yielding=yielding,
        )

        self.logger.info(
            "Metal miner %s: utilization=%d%%, "
            "core_budget=%d, cores=%d, yielding=%s",
            miner_id,
            gpu_util,
            self._scheduler.get_core_budget(),
            self.gpu_core_count,
            yielding,
        )

        # Duty-cycle controller: sleep proportionally to compute
        # time so actual GPU utilization matches the target.
        self._duty_cycle = DutyCycleController(
            target_pct=gpu_util,
        )

        # Pipeline state (reset per mine_block call)
        self._feeder: Optional[IsingFeeder] = None
        self._stream: Optional[Iterator] = None
        self._active_tg = self._scheduler.get_core_budget()
        self._max_tg = self._active_tg

        signal.signal(signal.SIGTERM, self._cleanup_handler)

    # ── BaseMiner hooks ──────────────────────────────────

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Create IsingFeeder for this block."""
        prev_block = args[0] if len(args) > 0 else None
        node_info = args[1] if len(args) > 1 else None
        if prev_block is None or node_info is None:
            self.logger.error(
                "Missing prev_block or node_info",
            )
            return False

        cur_index = prev_block.header.index + 1
        budget = self._scheduler.get_core_budget()

        self._feeder = IsingFeeder(
            prev_hash=prev_block.hash,
            miner_id=node_info.miner_id,
            cur_index=cur_index,
            nodes=self.sampler.nodes,
            edges=self.sampler.edges,
            buffer_size=budget * 2,
        )

        self._stream = None
        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _sample_batch(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> Optional[
        List[Tuple[int, bytes, dimod.SampleSet]]
    ]:
        """Stream one batch from the Metal pipeline.

        Lazily creates the streaming iterator on first call.
        Returns one (nonce, salt, sampleset) per call, or
        None to fall through to single-nonce path.
        """
        # No feeder means fallback mode (CPU sampler)
        if self._feeder is None:
            return None

        if self._scheduler.should_throttle():
            time.sleep(0.5)

        # Dynamic batch sizing: check if IOKit suggests resizing
        if self._stream is not None and self._scheduler.yielding:
            new_tg = self._scheduler.check_stable_target_threadgroups(
                self._max_tg, self._active_tg,
            )
            if new_tg is not None and new_tg != self._active_tg:
                self.logger.info(
                    "Resizing Metal batch: %d → %d threadgroups",
                    self._active_tg, new_tg,
                )
                if hasattr(self._stream, 'close'):
                    self._stream.close()
                self._stream = None
                self._active_tg = new_tg
                self._duty_cycle.reset()

        if self._stream is None:
            budget = self._active_tg
            self._stream = self.sampler.sample_ising_streaming(
                self._feeder,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                max_threadgroups=budget,
                duty_cycle=self._duty_cycle,
                scheduler=self._scheduler,
            )

        try:
            model, ss = next(self._stream)
        except StopIteration:
            return None

        return [(model.nonce, model.salt, ss)]

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> dimod.SampleSet:
        """Single-nonce fallback (synchronous)."""
        results = self.sampler.sample_ising(
            [h], [J],
            num_reads=num_reads,
            num_sweeps=num_sweeps,
        )
        return results[0]

    def _post_mine_cleanup(self) -> None:
        """Stop stream and feeder."""
        if self._stream is not None:
            if hasattr(self._stream, 'close'):
                self._stream.close()
            self._stream = None
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM: stop feeder, scheduler, exit."""
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

        if hasattr(self, '_scheduler'):
            self._scheduler.stop()

        if hasattr(self, 'top_attempts'):
            self.top_attempts.clear()

        if hasattr(self, 'logger'):
            self.logger.info(
                "Metal miner %s received SIGTERM, "
                "cleaning up...", self.miner_id,
            )

        os._exit(0)
