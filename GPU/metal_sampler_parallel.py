"""Unified Metal Sampler - Uses new unified Metal kernel implementation.

This provides the unified interface using the new Metal kernels that handle
all processing in Metal for maximum efficiency.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

import dimod
from dwave_topologies import DEFAULT_TOPOLOGY


class MetalKernelDimodSampler:
    """Unified Metal kernel sampler - uses new implementation."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None,
                 verbose: bool = False, default_sample_interval: Optional[int] = None):
        """Initialize Metal sampler - same interface as original."""
        self.logger = logger or logging.getLogger(__name__)
        self._verbose = verbose
        self._default_sample_interval = default_sample_interval
        self.sampler_type = "metal_parallel"

        if logger is None:
            self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            if verbose:
                self.logger.setLevel(logging.DEBUG)

        if not METAL_AVAILABLE:
            raise ImportError("Metal not available - requires macOS with Apple Silicon")

        # Use the new unified Metal implementation
        from GPU.metal_kernel_sampler_parallel import UnifiedMetalSampler
        self._unified_sampler = UnifiedMetalSampler(
            logger=self.logger,
            verbose=verbose
        )

        # Set up topology compatibility (same as original)
        topology_graph = DEFAULT_TOPOLOGY.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())

        self.logger.debug(f"[MetalParallel] Initialized using new unified Metal implementation")

    def sample_ising(self, h, J, num_reads: int = 256, num_sweeps: int = 1000,
                    num_replicas: int = None, swap_interval: int = 15,
                    T_min: float = 0.01, T_max: float = 1.0,
                    sample_interval: Optional[int] = None,
                    cooling_factor: float = 0.999, cooling_start_sweep: int = None,
                    **kwargs) -> dimod.SampleSet:
        """Unified sample_ising interface - uses new Metal kernels."""
        start_time = time.time()

        # Validate inputs
        if not h and not J:
            raise ValueError("Either h or J must be non-empty")

        if sample_interval is None:
            sample_interval = self._default_sample_interval
        self.logger.debug(
            f"[MetalParallel] Sampling: reads={num_reads}, sweeps={num_sweeps}, replicas={num_replicas}, interval={sample_interval}"
        )

        # Use the new unified Metal implementation (forward only supported args)
        sampleset = self._unified_sampler.sample_ising(
            h=h,
            J=J,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            num_replicas=num_replicas or max(8, num_reads // 8),  # Default replicas
            T_min=T_min,
            T_max=T_max,
            sample_interval=sample_interval,
        )

        runtime = time.time() - start_time
        self.logger.debug(f"[MetalParallel] Completed: {runtime:.2f}s")

        return sampleset

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_unified_sampler'):
            self._unified_sampler.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


if __name__ == "__main__":
    # Test the parallel sampler
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Metal Parallel Sampler")
    print("=" * 50)

    try:
        sampler = MetalKernelDimodSampler(verbose=True)

        # Test with simple 2-spin problem
        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): -1.0}

        sampleset = sampler.sample_ising(
            h=h, J=J,
            num_reads=10,
            num_sweeps=100,
            num_replicas=4
        )

        energies = list(sampleset.record.energy)
        print(f"\nResults for 2-spin ferromagnetic:")
        print(f"Min energy: {min(energies)} (expected: -1)")

        # Test with problem from basic_ising_problems.py
        from tools.basic_ising_problems import BASIC_ISING_PROBLEMS

        print(f"\nTesting Problem 6 (8-spin cube):")
        h, J, optimal_energy, description = BASIC_ISING_PROBLEMS[6]
        print(f"Description: {description}")
        print(f"Optimal energy: {optimal_energy}")

        sampleset = sampler.sample_ising(
            h=h, J=J,
            num_reads=32,
            num_sweeps=128,
            num_replicas=4,
            sample_interval=8
        )

        energies = list(sampleset.record.energy)
        min_energy = min(energies)
        print(f"Min energy found: {min_energy}")
        print(f"Success: {min_energy <= optimal_energy}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()