"""GPU mining components for quantum blockchain."""

from .sampler import GPUSampler
from .cuda_sa import CudaSASampler
from .modal_sampler import ModalSampler, gpu_app
from .cuda_miner import CudaMiner
from .modal_miner import ModalMiner

# Try to import Metal components (only available on macOS)
try:
    from .metal_sa import MetalSASampler
    from .metal_miner import MetalMiner
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    MetalSASampler = None
    MetalMiner = None

# Check if GPU functionality is available
try:
    import modal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'GPUSampler', 'CudaSASampler', 'ModalSampler',
    'CudaMiner', 'ModalMiner',
    'gpu_app', 'GPU_AVAILABLE', 'METAL_AVAILABLE'
]

# Add Metal components if available
if METAL_AVAILABLE:
    __all__.extend(['MetalSASampler', 'MetalMiner'])