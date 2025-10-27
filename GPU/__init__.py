"""GPU mining components for quantum blockchain."""

# Try to import CUDA components (only available with cupy)
try:
    from .cuda_sa import CudaSASamplerAsync
    from .cuda_miner import CudaMiner
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaSASamplerAsync = None
    CudaMiner = None

# Try to import Modal components
try:
    from .modal_sampler import ModalSampler, gpu_app
    from .modal_miner import ModalMiner
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    ModalSampler = None
    ModalMiner = None
    gpu_app = None

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
GPU_AVAILABLE = CUDA_AVAILABLE or MODAL_AVAILABLE

__all__ = [
    'ModalSampler',
    'CudaMiner', 'ModalMiner',
    'gpu_app', 'GPU_AVAILABLE', 'METAL_AVAILABLE', 'MODAL_AVAILABLE'
]

# Add CUDA components if available
if CUDA_AVAILABLE:
    __all__.extend(['CudaSASamplerAsync', 'CudaMiner'])

# Add Modal components if available
if MODAL_AVAILABLE:
    __all__.extend(['ModalSampler', 'ModalMiner', 'gpu_app'])

# Add Metal components if available
if METAL_AVAILABLE:
    __all__.extend(['MetalSASampler', 'MetalMiner'])
