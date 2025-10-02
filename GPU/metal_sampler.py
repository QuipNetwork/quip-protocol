"""3D Edwards-Anderson Metal Parallel Tempering GPU sampler."""

import logging
from typing import Optional
import dimod
from .metal_kernel_sampler import UnifiedMetalSampler


class MetalSampler:
    """3D Edwards-Anderson Metal Parallel Tempering GPU sampler using native Metal kernels."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "metal_ea_3d"

        # Use the 3D Edwards-Anderson Metal kernel sampler
        self._kernel_sampler = UnifiedMetalSampler(logger=logger)

        # Expose properties for compatibility (if available)
        self.nodes = getattr(self._kernel_sampler, 'nodes', None)
        self.edges = getattr(self._kernel_sampler, 'edges', None)
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = getattr(self._kernel_sampler, 'properties', {})

        self.logger.info(f"[MetalSampler] Initialized Metal sampler")

    def sample_ising(self, h, J, num_reads=256, num_sweeps=1000, num_replicas=None,
                     sample_interval=15, T_min=0.1, T_max=5.0, **kwargs):
        """Run unified GPU-only Parallel Tempering sampling (single kernel dispatch) - DEFAULT."""
        self.logger.debug(f"[MetalSampler] Starting unified GPU sampling: reads={num_reads}, sweeps={num_sweeps}")

        # Use the unified GPU-only kernel sampler (now the default)
        return self._kernel_sampler.sample_ising(
            h, J, num_reads=num_reads, num_sweeps=num_sweeps, num_replicas=num_replicas,
            sample_interval=sample_interval, T_min=T_min, T_max=T_max, **kwargs
        )


    def close(self):
        """Clean up Metal resources."""
        if hasattr(self, '_kernel_sampler'):
            self._kernel_sampler.close()

    def __del__(self):
        self.close()