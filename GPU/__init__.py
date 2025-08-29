"""GPU mining components for quantum blockchain."""

from .sampler import GPUSampler, LocalGPUSampler
from .modal_sampler import ModalSampler, gpu_app
from .gpu_metal import MetalMiner
from .gpu_cuda import CudaMiner
from .gpu_modal import ModalMiner
from .worker import gpu_mine_block_process

# Check if GPU functionality is available
try:
    import modal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'LocalGPUSampler', 'GPUSampler', 'ModalSampler',
    'MetalMiner', 'CudaMiner', 'ModalMiner',
    'gpu_app', 'gpu_mine_block_process', 'GPU_AVAILABLE'
]