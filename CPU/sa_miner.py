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
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


class SimulatedAnnealingMiner(BaseMiner):
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
        return adapt_parameters(
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


def adapt_parameters(
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    num_nodes: int = DEFAULT_NUM_NODES,
    num_edges: int = DEFAULT_NUM_EDGES
):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Uses GSE-based difficulty calculation with log-linear interpolation
    in sweep space, calibrated from experimental data.

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum solution diversity required (reserved)
        min_solutions: Minimum number of valid solutions required
        num_nodes: Number of nodes in topology (default: DEFAULT_TOPOLOGY)
        num_edges: Number of edges in topology (default: DEFAULT_TOPOLOGY)

    Returns:
        Dictionary with num_sweeps and num_reads parameters
    """
    # Get normalized difficulty [0, 1]
    difficulty = energy_to_difficulty(
        difficulty_energy,
        num_nodes=num_nodes,
        num_edges=num_edges
    )

    # CPU SA calibration ranges (from experimental data)
    min_sweeps = 64      # Easiest difficulty
    max_sweeps = 4096    # Hardest difficulty (avoid overfitting beyond this)

    # Direct linear scaling: difficulty × max_sweeps
    # Example: difficulty=0.5 → 0.5 × 4096 = 2048 sweeps
    num_sweeps = max(min_sweeps, int(difficulty * max_sweeps))

    # Reads scale with both difficulty and min_solutions requirement
    base_reads = max(int(min_solutions) * 4, 64)
    max_reads = max(int(min_solutions) * 8, 512)
    num_reads = max(base_reads, int(difficulty * max_reads))

    return {
        'num_sweeps': num_sweeps,
        'num_reads': num_reads,
    }
