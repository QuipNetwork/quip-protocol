"""CPU miner using SimulatedAnnealingStructuredSampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import traceback

from shared.base_miner import BaseMiner
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
        self._register_sigterm_cleanup("CPU")

    def _backend_cleanup(self) -> None:
        """Handle SIGTERM-time cleanup of CPU resources."""
        # Reset any persistent library state
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
            self.sampler.cleanup()

        # Clear any cached data
        if hasattr(self, 'top_attempts'):
            self.top_attempts.clear()

    def _stream_factory_kwargs(self, sample_ctx, nodes):
        """Kwargs forwarded to CPU.sa_stream:build_persistent_context."""
        return {
            "miner_id": self.miner_id,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "allowed_h": sample_ctx.get("allowed_h_values"),
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


