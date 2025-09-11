"""Metal kernel interface for PyTorch integration."""

import os
import logging
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.utils.cpp_extension import load
except ImportError:
    torch = None

try:
    import torch
    # Check if we can access Metal through PyTorch MPS
    METAL_AVAILABLE = (hasattr(torch.backends, 'mps') and 
                      torch.backends.mps.is_available())
    
    # If we want actual Metal framework access, try PyObjC
    METAL_FRAMEWORK_AVAILABLE = False
    if METAL_AVAILABLE:
        try:
            from Metal import MTLCreateSystemDefaultDevice
            from MetalKit import MTKView
            METAL_FRAMEWORK_AVAILABLE = True
        except ImportError:
            # Fall back to MPS-only mode - this is perfectly fine for our use case
            pass
            
except ImportError:
    METAL_AVAILABLE = False
    METAL_FRAMEWORK_AVAILABLE = False


class MetalKernelInterface:
    """Interface for custom Metal kernels with PyTorch."""
    
    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.device_str = device
        self.logger = logger or logging.getLogger(__name__)
        self._device = None
        self._metal_device = None
        self._command_queue = None
        self._kernels = {}
        self._library = None
        
        if not torch:
            raise RuntimeError("PyTorch not available")
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal/MPS not available on this system")
        
        # Log Metal framework availability status
        if METAL_FRAMEWORK_AVAILABLE:
            self.logger.info("[MetalKernelInterface] Direct Metal framework access available")
        else:
            self.logger.info("[MetalKernelInterface] Using MPS-only mode (PyTorch Metal backend)")
            
        self._initialize_metal()
        
    def _initialize_metal(self):
        """Initialize Metal device and compile kernels."""
        try:
            # Initialize PyTorch MPS device
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS not available")
            self._device = torch.device("mps")
            
            # Initialize Metal device (if direct framework access is available)
            if METAL_FRAMEWORK_AVAILABLE:
                from Metal import MTLCreateSystemDefaultDevice
                self._metal_device = MTLCreateSystemDefaultDevice()
                if not self._metal_device:
                    raise RuntimeError("Could not create Metal device")
                    
                # Create command queue
                self._command_queue = self._metal_device.newCommandQueue()
                if not self._command_queue:
                    raise RuntimeError("Could not create Metal command queue")
                    
                # Load and compile kernel library
                self._load_kernel_library()
            else:
                # MPS-only mode: no direct Metal device access needed
                self.logger.info("[MetalKernelInterface] Running in MPS-only mode - kernel operations will use PyTorch fallbacks")
                self._metal_device = None
                self._command_queue = None
                self._library = None
            
            if self._metal_device:
                self.logger.info(f"[MetalKernelInterface] Initialized Metal device: {self._metal_device.name()}")
            else:
                self.logger.info("[MetalKernelInterface] Initialized MPS-only mode (no direct Metal device)")
            
        except Exception as e:
            self.logger.error(f"[MetalKernelInterface] Metal initialization failed: {e}")
            raise
            
    def _load_kernel_library(self):
        """Load and compile Metal kernel library."""
        # Find the kernel file
        kernel_path = Path(__file__).parent / "metal_kernels.metal"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Metal kernel file not found: {kernel_path}")
            
        # Read kernel source
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()
            
        # Compile library
        try:
            self._library = self._metal_device.newLibraryWithSource_options_error_(
                kernel_source, None, None
            )[0]
            if not self._library:
                raise RuntimeError("Failed to compile Metal kernel library")
                
            # Load specific kernel functions
            kernel_names = [
                "fused_metropolis_update",
                "optimized_coupling_field", 
                "compute_energies",
                "initialize_random_spins",
                "compute_local_fields",
                "optimized_coupling_field_shared"
            ]
            
            for name in kernel_names:
                kernel_function = self._library.newFunctionWithName_(name)
                if kernel_function:
                    compute_pipeline = self._metal_device.newComputePipelineStateWithFunction_error_(
                        kernel_function, None
                    )[0]
                    if compute_pipeline:
                        self._kernels[name] = compute_pipeline
                        self.logger.debug(f"[MetalKernelInterface] Loaded kernel: {name}")
                    else:
                        self.logger.warning(f"[MetalKernelInterface] Failed to create pipeline for: {name}")
                else:
                    self.logger.warning(f"[MetalKernelInterface] Kernel function not found: {name}")
                    
            self.logger.info(f"[MetalKernelInterface] Loaded {len(self._kernels)} Metal kernels")
            
        except Exception as e:
            self.logger.error(f"[MetalKernelInterface] Kernel compilation failed: {e}")
            raise
            
    def _get_metal_buffer_from_tensor(self, tensor):
        """Get Metal buffer from PyTorch tensor."""
        # This is a simplified approach - in practice, you'd need to access 
        # the underlying MPS buffer from the PyTorch tensor
        # For now, we'll use a fallback approach
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor
        
    def fused_metropolis_update(self, spins, local_fields, random_values, beta, num_chains, chunk_size):
        """Execute fused Metropolis update kernel."""
        if "fused_metropolis_update" not in self._kernels:
            raise RuntimeError("fused_metropolis_update kernel not available")
            
        # For now, fall back to PyTorch implementation with optimization hints
        # In a full implementation, you would:
        # 1. Get Metal buffers from PyTorch tensors
        # 2. Set up compute command encoder  
        # 3. Set kernel arguments
        # 4. Dispatch threads
        # 5. Commit and wait
        
        self.logger.debug(f"[MetalKernelInterface] Executing fused_metropolis_update (fallback)")
        
        # Optimized PyTorch fallback that mimics kernel behavior
        flat_spins = spins.view(-1)
        flat_fields = local_fields.view(-1)
        flat_random = random_values.view(-1)
        
        # Vectorized Metropolis computation
        current_spins = flat_spins.to(torch.float32)
        delta_e = 2.0 * current_spins * flat_fields
        accept_mask = (delta_e > 0) | (flat_random < torch.exp(-beta * torch.abs(delta_e)))
        
        # Conditional spin flips
        flat_spins[accept_mask] *= -1
        
        return spins
        
    def optimized_coupling_field(self, neighbor_sum, spins, i_indices, j_indices, j_values, num_chains, num_couplings, num_spins):
        """Execute optimized coupling field computation kernel."""
        if "optimized_coupling_field" not in self._kernels:
            raise RuntimeError("optimized_coupling_field kernel not available")
            
        self.logger.debug(f"[MetalKernelInterface] Executing optimized_coupling_field (fallback)")
        
        # Optimized PyTorch fallback
        sp_f = spins.to(torch.float32)
        neighbor_sum.zero_()
        
        # Vectorized scatter-add operations
        R = num_chains
        neighbor_sum.scatter_add_(1, i_indices.unsqueeze(0).expand(R, -1), sp_f[:, j_indices] * j_values)
        neighbor_sum.scatter_add_(1, j_indices.unsqueeze(0).expand(R, -1), sp_f[:, i_indices] * j_values)
        
        return neighbor_sum
        
    def compute_energies(self, spins, h_fields, i_indices, j_indices, j_values, num_chains, num_spins, num_couplings):
        """Execute vectorized energy computation kernel."""
        if "compute_energies" not in self._kernels:
            raise RuntimeError("compute_energies kernel not available")
            
        self.logger.debug(f"[MetalKernelInterface] Executing compute_energies (fallback)")
        
        # Optimized PyTorch fallback
        h_energy = (spins.to(torch.float32) * h_fields).sum(dim=1)
        j_energy = (spins[:, i_indices] * spins[:, j_indices] * j_values).sum(dim=1)
        energies = h_energy + j_energy
        
        return energies
        
    def initialize_random_spins(self, spins, random_values, num_chains, num_spins):
        """Execute efficient random spin initialization kernel.""" 
        if "initialize_random_spins" not in self._kernels:
            raise RuntimeError("initialize_random_spins kernel not available")
            
        self.logger.debug(f"[MetalKernelInterface] Executing initialize_random_spins (fallback)")
        
        # Optimized PyTorch fallback
        spins.copy_((random_values > 0.5).to(torch.int8) * 2 - 1)
        return spins
        
    def compute_local_fields(self, local_fields, neighbor_sums, h_fields, num_chains, num_spins):
        """Execute optimized local field computation kernel."""
        if "compute_local_fields" not in self._kernels:
            raise RuntimeError("compute_local_fields kernel not available")
            
        self.logger.debug(f"[MetalKernelInterface] Executing compute_local_fields (fallback)")
        
        # Optimized PyTorch fallback  
        torch.add(neighbor_sums, h_fields, out=local_fields)
        return local_fields
        
    def is_kernel_available(self, kernel_name):
        """Check if a specific kernel is available."""
        return kernel_name in self._kernels
        
    def get_available_kernels(self):
        """Get list of available kernel names."""
        return list(self._kernels.keys())
        
    def close(self):
        """Clean up Metal resources."""
        self._kernels.clear()
        self._library = None
        self._command_queue = None
        self._metal_device = None