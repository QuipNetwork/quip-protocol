"""Modal Labs GPU-accelerated sampler for cloud GPU mining."""

import time
import numpy as np
from dwave.system.testing import MockDWaveSampler

# Optional imports
try:
    import modal
except ImportError:
    modal = None

try:
    from numba import jit
except ImportError:
    jit = None

# GPU availability check
GPU_AVAILABLE = modal is not None

# Define Modal app globally
gpu_app = None
if GPU_AVAILABLE:
    gpu_app = modal.App("quantum-blockchain-gpu-miner")

    # GPU container image - simplified without CuPy for faster startup
    gpu_image = modal.Image.debian_slim().pip_install(
        "numpy",
        "numba",
    )

    # Define GPU functions for each type
    @gpu_app.function(
        image=gpu_image,
        gpu="t4",
        timeout=300,
    )
    def gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on T4 using Numba acceleration."""
        import time
        import numpy as np
        from numba import jit

        start_time = time.time()

        # Convert to arrays
        num_vars = max(max(h_dict.keys()), max(max(j) for j in J_dict.keys())) + 1
        h = np.zeros(num_vars)
        for i, val in h_dict.items():
            h[i] = val

        # Create coupling matrix
        J_matrix = np.zeros((num_vars, num_vars))
        for (i, j), val in J_dict.items():
            J_matrix[i, j] = val
            J_matrix[j, i] = val

        # Numba-accelerated annealing
        @jit(nopython=True)
        def anneal(h, J_matrix, num_sweeps):
            state = np.random.choice(np.array([-1, 1]), size=num_vars)
            betas = np.linspace(0.1, 10.0, num_sweeps)

            for beta in betas:
                for _ in range(num_vars):
                    i = np.random.randint(0, num_vars)
                    neighbors_sum = np.dot(J_matrix[i], state)
                    delta_e = 2 * state[i] * (h[i] + neighbors_sum)
                    if delta_e < 0 or np.random.random() < np.exp(-beta * delta_e):
                        state[i] *= -1

            energy = -np.dot(state, h) - 0.5 * np.dot(state, np.dot(J_matrix, state))
            return state, energy

        # Run parallel simulated annealing
        samples = []
        energies = []

        for read in range(num_reads):
            state, energy = anneal(h, J_matrix, num_sweeps)
            samples.append(state.tolist())
            energies.append(float(energy))

        return {
            "samples": samples,
            "energies": energies,
            "timing": {"total": time.time() - start_time}
        }

    @gpu_app.function(
        image=gpu_image,
        gpu="a10g",
        timeout=300,
    )
    def gpu_sample_a10g(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on A10G - same implementation, different GPU."""
        # Reuse T4 implementation
        return gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps)

    @gpu_app.function(
        image=gpu_image,
        gpu="a100",
        timeout=300,
    )
    def gpu_sample_a100(h_dict, J_dict, num_reads, num_sweeps):
        """GPU sampling on A100 - same implementation, different GPU."""
        # Reuse T4 implementation
        return gpu_sample_t4(h_dict, J_dict, num_reads, num_sweeps)


class GPUSampler(MockDWaveSampler):
    """GPU-accelerated sampler using Modal Labs."""

    def __init__(self, gpu_type: str = "t4"):
        """
        Initialize GPU sampler.

        Args:
            gpu_type: GPU type to use ('t4', 'a10g', 'a100')
                     t4: ~$0.10/hour (budget option)
                     a10g: ~$0.30/hour (balanced)
                     a100: ~$1.00/hour (performance)
        """
        if not GPU_AVAILABLE:
            raise ImportError("Modal not installed. Run: pip install modal")

        self.gpu_type = gpu_type

        # Map GPU type to function
        self.gpu_functions = {
            "t4": gpu_sample_t4,
            "a10g": gpu_sample_a10g,
            "a100": gpu_sample_a100
        }

        if gpu_type not in self.gpu_functions:
            raise ValueError(f"Invalid GPU type: {gpu_type}. Choose from: t4, a10g, a100")

        self._gpu_sample_func = self.gpu_functions[gpu_type]

        # Use same topology as SimulatedAnnealingStructuredSampler
        qpu = MockDWaveSampler()
        super().__init__(
            nodelist=qpu.nodelist,
            edgelist=qpu.edgelist,
            properties=qpu.properties,
            substitute_sampler=self
        )

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs):
        """Sample from Ising model using GPU acceleration."""
        # Convert h and J to dictionaries if needed
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J

        # Run on GPU via Modal (without context manager to avoid nested app.run)
        result = self._gpu_sample_func.remote(h_dict, J_dict, num_reads, num_sweeps)

        # Format result to match D-Wave interface
        class SampleSet:
            def __init__(self, samples, energies):
                self.record = type('Record', (), {
                    'sample': np.array(samples),
                    'energy': np.array(energies)  # Convert to numpy array
                })()

        return SampleSet(result["samples"], result["energies"])