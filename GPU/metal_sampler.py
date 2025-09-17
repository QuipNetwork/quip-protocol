"""Metal P-bit GPU sampler for quantum blockchain mining."""

import logging
from typing import Optional
import dimod
from .metal_kernel_sampler import MetalKernelDimodSampler


class MetalSampler:
    """Metal P-bit GPU sampler using native Metal kernels."""
    
    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "metal_pbit"
        
        # Use the new P-bit Metal kernel sampler
        self._kernel_sampler = MetalKernelDimodSampler(device, logger)
        
        # Expose properties for compatibility
        self.nodes = self._kernel_sampler.nodes
        self.edges = self._kernel_sampler.edges
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = self._kernel_sampler.properties
        
        self.logger.info(f"[MetalSampler] Initialized P-bit Metal sampler with {len(self.nodes)} nodes")
    
    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
        """Run P-bit Metal kernel-based simulated annealing."""
        self.logger.debug(f"[MetalSampler] Starting P-bit sampling: reads={num_reads}, sweeps={num_sweeps}")
        
        # Use the P-bit kernel sampler
        return self._kernel_sampler.sample_ising(h, J, num_reads, num_sweeps, **kwargs)
    
    def close(self):
        """Clean up Metal resources."""
        if hasattr(self, '_kernel_sampler'):
            self._kernel_sampler.close()
    
    def __del__(self):
        self.close()