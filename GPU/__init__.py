"""GPU mining components for quantum blockchain."""

from .sampler import GPUSampler
from .metal_sa import MetalSASampler
from .modal_sampler import ModalSampler, gpu_app
from .metal_miner import MetalMiner
from .cuda_miner import CudaMiner
from .modal_miner import ModalMiner

# Check if GPU functionality is available
try:
    import modal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'GPUSampler', 'MetalSASampler', 'ModalSampler',
    'MetalMiner', 'CudaMiner', 'ModalMiner',
    'gpu_app', 'GPU_AVAILABLE'
]