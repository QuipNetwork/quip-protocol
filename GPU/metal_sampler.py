"""3D Edwards-Anderson Metal Parallel Tempering GPU sampler."""

import logging
from typing import Optional
import dimod
from .metal_kernel_sampler import MetalKernelDimodSampler


class MetalSampler:
    """3D Edwards-Anderson Metal Parallel Tempering GPU sampler using native Metal kernels."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "metal_ea_3d"

        # Use the 3D Edwards-Anderson Metal kernel sampler
        self._kernel_sampler = MetalKernelDimodSampler(device, logger)

        # Expose properties for compatibility
        self.nodes = self._kernel_sampler.nodes
        self.edges = self._kernel_sampler.edges
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = getattr(self._kernel_sampler, 'properties', {})

        self.logger.info(f"[MetalSampler] Initialized 3D Edwards-Anderson Metal sampler with {len(self.nodes)} nodes")

    def sample_ising(self, h, J, num_reads=256, num_sweeps=100000, num_replicas=None,
                     swap_interval=15, cooling_interval=500, T_min=0.1, T_max=5.0,
                     cooling_factor=0.999, **kwargs):
        """Run 3D Edwards-Anderson Parallel Tempering sampling."""
        self.logger.debug(f"[MetalSampler] Starting EA PT sampling: reads={num_reads}, sweeps={num_sweeps}")

        # Use the 3D Edwards-Anderson kernel sampler
        return self._kernel_sampler.sample_ising(
            h, J, num_reads=num_reads, num_sweeps=num_sweeps, num_replicas=num_replicas,
            swap_interval=swap_interval, cooling_interval=cooling_interval,
            T_min=T_min, T_max=T_max, cooling_factor=cooling_factor, **kwargs
        )

    def close(self):
        """Clean up Metal resources."""
        if hasattr(self, '_kernel_sampler'):
            self._kernel_sampler.close()

    def __del__(self):
        self.close()