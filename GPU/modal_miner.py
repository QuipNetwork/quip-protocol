"""GPU miner using Modal via ModalSampler(gpu_type)."""
from __future__ import annotations

import signal
import sys
from typing import List, Tuple

from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements
from GPU.modal_sampler import ModalSampler


class ModalMiner(BaseMiner):
    # Modal cloud GPU calibration ranges
    ADAPT_MIN_SWEEPS = 128
    ADAPT_MAX_SWEEPS = 4096
    ADAPT_MIN_READS = 64
    ADAPT_MAX_READS = 256
    ADAPT_READS_SOLUTION_FLOOR_FACTOR = 3

    STREAM_FACTORY_DOTTED = "GPU.modal_stream:build_persistent_context"

    def __init__(self, miner_id: str, gpu_type: str = "t4", **cfg):
        sampler = ModalSampler(gpu_type)
        super().__init__(miner_id, sampler)
        self.miner_type = f"GPU-{gpu_type.upper()}"
        self.gpu_type = gpu_type

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of Modal cloud resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"Modal miner {self.miner_id} received SIGTERM, cleaning up cloud GPU resources ({self.gpu_type})...")

        # Modal-specific cleanup
        try:
            # Terminate any running Modal functions
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()
                if hasattr(self, 'logger'):
                    self.logger.info("Modal functions terminated")

            # Close Modal connections/sessions
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'close'):
                self.sampler.close()
                if hasattr(self, 'logger'):
                    self.logger.info("Modal connections closed")

            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during Modal miner cleanup: {e}")

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

    def _stream_factory_kwargs(self, sample_ctx, nodes):
        """Kwargs forwarded to GPU.modal_stream:build_persistent_context."""
        return {
            "miner_id": self.miner_id,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
            "num_reads": sample_ctx["num_reads"],
            "num_sweeps": sample_ctx["num_sweeps"],
            "gpu_type": self.gpu_type,
        }
