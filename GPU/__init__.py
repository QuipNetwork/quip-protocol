"""GPU mining components for quantum blockchain."""

from .sampler import LocalGPUSampler
from .modal_sampler import GPUSampler, gpu_app
from .worker import gpu_mine_block_process

# Check if GPU functionality is available
try:
    import modal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = ['LocalGPUSampler', 'GPUSampler', 'gpu_app', 'gpu_mine_block_process', 'GPU_AVAILABLE']