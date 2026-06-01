"""CPU miner using SimulatedAnnealingStructuredSampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


class SimulatedAnnealingMiner(BaseMiner):
    # CPU SA calibration: sweeps 64–4096, reads from min_solutions factors
    ADAPT_MIN_SWEEPS = 64
    ADAPT_MAX_SWEEPS = 4096
    ADAPT_MIN_READS = 64
    ADAPT_MAX_READS = 512
    ADAPT_READS_SOLUTION_MIN_FACTOR = 4
    ADAPT_READS_SOLUTION_MAX_FACTOR = 8
    ADAPT_READS_SOLUTION_FLOOR_FACTOR = 0

    def __init__(self, miner_id: str, sampler=None, topology=None, **cfg):
        if sampler is None:
            sampler = SimulatedAnnealingStructuredSampler(topology=topology)
        self.nodes = sampler.nodes
        self.edges = sampler.edges
        super().__init__(miner_id, sampler)
        self.miner_type = "CPU"

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of CPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"CPU miner {self.miner_id} received SIGTERM, cleaning up...")

        # CPU-specific cleanup
        try:
            # Reset any persistent library state
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()

            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during CPU miner cleanup: {e}")

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
        return self.sampler.sample_ising(
            h=h, J=J, num_reads=num_reads, num_sweeps=num_sweeps,
        )

    def _on_sampling_error(
        self,
        error: Exception,
        stop_event: multiprocessing.synchronize.Event,
    ) -> bool:
        if stop_event.is_set():
            self.logger.info("Interrupted during sampling")
            return True
        self.logger.error(
            f"Sampling error: {error}\n"
            f"  Topology: nodes={len(self.nodes)}, edges={len(self.edges)}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return False


