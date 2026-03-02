"""GPU miner using Metal/MPS via GPUSampler('mps')."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import signal
import subprocess
import sys
from typing import Dict, List, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from GPU.metal_sa import MetalSASampler
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


def get_gpu_core_count() -> int:
    """Detect Apple Silicon GPU core count programmatically."""
    try:
        # Use grep to filter ioreg output - much faster and avoids Unicode issues
        result = subprocess.run(
            "ioreg -l | grep gpu-core-count",
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.stdout:
            # Parse line like: | |   |   |   "gpu-core-count" = 40
            for line in result.stdout.splitlines():
                if 'gpu-core-count' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        return int(parts[1].strip())
    except Exception as e:
        raise RuntimeError(f"Failed to detect GPU core count: {e}")

    raise RuntimeError("Could not find gpu-core-count in ioreg output")


class MetalMiner(BaseMiner):
    # Metal MPS strategy: fewer sweeps, more reads
    ADAPT_MIN_SWEEPS = 64
    ADAPT_MAX_SWEEPS = 512
    ADAPT_MIN_READS = 32
    ADAPT_MAX_READS = 1024

    def __init__(self, miner_id: str, topology=None, **cfg):
        try:
            # Initialize base miner first to get the logger
            sampler = MetalSASampler(topology=topology)
            super().__init__(miner_id, sampler, miner_type="GPU-Metal")
            # Now update sampler with our logger
            sampler.logger = self.logger
            self.miner_type = "GPU-Metal"

            self.logger.info(f"Using MetalSASampler (Simulated Annealing)")
        except Exception as e:
            # For fallback case, we can't use logger yet since super().__init__() wasn't called
            sampler = SimulatedAnnealingStructuredSampler(topology=topology)
            super().__init__(miner_id, sampler, miner_type="CPU-FALLBACK")
            self.miner_type = "CPU-FALLBACK"
            # Now we can use logger
            self.logger.warning(f"Metal GPU initialization failed, falling back to CPU: {e}")

        # GPU utilization control (0-100, default 100)
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(f"gpu_utilization must be between 1-100, got {self.gpu_utilization}")

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of Metal resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"Metal miner {self.miner_id} received SIGTERM, cleaning up Metal resources...")

        # Metal-specific cleanup
        try:
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()

            # Reset sampler state if possible
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during Metal miner cleanup: {e}")

        # Exit gracefully
        sys.exit(0)

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
        **kwargs,
    ) -> dimod.SampleSet:
        # MetalSASampler.sample_ising takes lists; wrap single problem
        results = self.sampler.sample_ising(
            [h], [J], num_reads=num_reads, num_sweeps=num_sweeps,
        )
        return results[0]

