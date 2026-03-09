# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Gibbs miner using chromatic parallel block Gibbs sampling."""
from __future__ import annotations

import signal
import sys
import time
from typing import Dict, List, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from GPU.cuda_gibbs_sa import CudaGibbsSampler
from GPU.gpu_utilization import GpuUtilizationMonitor

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaGibbsMiner(BaseMiner):
    """CUDA GPU miner using chromatic block Gibbs sampling.

    Uses CudaGibbsSampler with adaptive SM limiting via NVML.
    The monitor adjusts SM count based on external GPU load,
    never exceeding the configured gpu_utilization ceiling.
    """

    ADAPT_MIN_SWEEPS = 256
    ADAPT_MAX_SWEEPS = 2048
    ADAPT_MIN_READS = 64
    ADAPT_MAX_READS = 256
    ADAPT_EXTRA_PARAMS = {'num_sweeps_per_beta': 1}

    def __init__(
        self,
        miner_id: str,
        device: str = "0",
        topology=None,
        update_mode: str = "gibbs",
        **cfg,
    ):
        if cp is None:
            raise ImportError("cupy not available")

        cp.cuda.Device(int(device)).use()

        self.device = device
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )

        device_sms = cp.cuda.Device(
            int(device)
        ).attributes['MultiProcessorCount']
        max_sms = max(
            1, int(device_sms * self.gpu_utilization / 100)
        )

        self._monitor = GpuUtilizationMonitor(
            device_id=int(device),
            max_utilization_pct=self.gpu_utilization,
            device_sms=device_sms,
        )

        sampler = CudaGibbsSampler(
            topology=topology,
            update_mode=update_mode,
            max_sms=max_sms,
        )
        super().__init__(
            miner_id, sampler, miner_type="GPU-CUDA-Gibbs",
        )

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        self.logger.info(
            f"CUDA Gibbs miner initialized on device {device} "
            f"(update_mode={update_mode}, "
            f"gpu_utilization={self.gpu_utilization}%%)"
        )

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA resource cleanup."""
        if hasattr(self, '_monitor'):
            self._monitor.stop()

        if hasattr(self, 'logger'):
            self.logger.info(
                f"CUDA Gibbs miner {self.miner_id} received "
                f"SIGTERM, cleaning up..."
            )

        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(
                    f"Error during CUDA Gibbs cleanup: {e}"
                )

        sys.exit(0)

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Set CUDA device context before mining."""
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}"
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

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> dimod.SampleSet:
        # Adaptive SM limiting: query monitor before each sample
        dynamic_sms = self._monitor.get_max_sms()
        self.sampler.max_sms = dynamic_sms

        if self._monitor.should_throttle():
            time.sleep(0.5)

        results = self.sampler.sample_ising(
            [h], [J],
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )
        return results[0]
