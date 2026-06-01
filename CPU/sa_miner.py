"""CPU miner using SimulatedAnnealingStructuredSampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import signal
import traceback
from typing import List, Tuple

from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements
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

    STREAM_FACTORY_DOTTED = "CPU.sa_stream:build_persistent_context"

    def __init__(self, miner_id: str, sampler=None, topology=None, **cfg):
        if sampler is None:
            sampler = SimulatedAnnealingStructuredSampler(topology=topology)
        self.nodes = sampler.nodes
        self.edges = sampler.edges
        super().__init__(miner_id, sampler)
        self.topology = topology
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

        # Exit gracefully — guard against raising SystemExit during
        # interpreter finalization (would produce "Exception ignored" noise).
        self._graceful_exit()

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
        """Kwargs forwarded to CPU.sa_stream:build_persistent_context."""
        return {
            "miner_id": self.miner_id,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
            "num_reads": sample_ctx["num_reads"],
            "num_sweeps": sample_ctx["num_sweeps"],
            "topology": getattr(self, "topology", None),
        }

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


