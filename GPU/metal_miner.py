# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU miner using Metal/MPS with RandomIsingFeeder streaming pipeline.

Mirrors GPUMiner (gpu_miner.py) architecture: RandomIsingFeeder for
background model generation, MetalScheduler for core budget and
IOKit-based yielding, and batched streaming dispatch via
MetalSASampler.sample_ising_streaming().
"""
from __future__ import annotations

import os
import signal
import subprocess
from typing import List, Tuple

from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements
from GPU.metal_sa import MetalSASampler
from GPU.metal_scheduler import MetalScheduler


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
    """Metal GPU miner with RandomIsingFeeder streaming pipeline.

    Architecture mirrors GPUMiner: background model generation
    via RandomIsingFeeder, core budget via MetalScheduler, and batched
    multi-problem dispatch via sample_ising_streaming().
    """

    # Keep the feeder large enough to keep Metal threadgroup dispatch fed
    # without starving on Python-side derivation. Matches the old default
    # of ``budget * 2`` for typical Apple Silicon core counts (~10).
    FEEDER_BUFFER_SIZE = 16

    # Dotted path to the stream-driver producer factory (GPU/metal_stream.py).
    # The driver process builds its own scheduler + feeder; the worker keeps
    # neither.
    STREAM_FACTORY_DOTTED = "GPU.metal_stream:build_persistent_context"

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

        self.topology = topology

        # Metal is required: a failed sampler init crashes rather than
        # silently falling back to CPU (the driver process owns the sampler).
        sampler = MetalSASampler(topology=topology)
        super().__init__(
            miner_id, sampler, miner_type="GPU-Metal",
        )
        sampler.logger = self.logger

        if not 0 < gpu_util <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {gpu_util}",
            )
        self.gpu_utilization = gpu_util

        self.gpu_core_count = get_gpu_core_count()
        scheduler = MetalScheduler(
            gpu_core_count=self.gpu_core_count,
            gpu_utilization_pct=gpu_util,
            yielding=yielding,
        )
        self.logger.info(
            "Metal miner %s: utilization=%d%%, "
            "core_budget=%d, cores=%d, yielding=%s",
            miner_id,
            gpu_util,
            scheduler.get_core_budget(),
            self.gpu_core_count,
            yielding,
        )

        signal.signal(signal.SIGTERM, self._cleanup_handler)

    # ── BaseMiner hooks ──────────────────────────────────

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Per-attempt setup validating the work context.

        Kept as an override (instead of using BaseMiner's default no-op)
        so the validation of ``prev_block`` / ``node_info`` runs early
        — the controller treats a ``False`` here as "skip this attempt"
        rather than letting the loop crash on missing context fields.
        """
        prev_block = args[0] if len(args) > 0 else None
        node_info = args[1] if len(args) > 1 else None
        if prev_block is None or node_info is None:
            self.logger.error(
                "Missing prev_block or node_info",
            )
            return False

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

    def _stream_factory_kwargs(self, sample_ctx, nodes):
        """Return kwargs forwarded to GPU.metal_stream:build_persistent_context.

        Called by BaseMiner._ensure_driver when spawning the stream driver.
        """
        return {
            "miner_id": self.miner_id,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
            "num_reads": sample_ctx["num_reads"],
            "num_sweeps": sample_ctx["num_sweeps"],
            "topology": getattr(self, "topology", None),
            "utilization": getattr(self, "gpu_utilization", 100),
        }

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM: clear cached state, exit.

        The sampler, scheduler and feeder live in the stream-driver process
        (reaped by ``BaseMiner._close_driver``), so this worker-side handler
        only drops cached candidates before a hard exit.
        """
        if hasattr(self, 'top_attempts'):
            self.top_attempts.clear()

        if hasattr(self, 'logger'):
            self.logger.info(
                "Metal miner %s received SIGTERM, "
                "cleaning up...", self.miner_id,
            )

        os._exit(0)
