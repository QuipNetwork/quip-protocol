"""Modal Labs GPU-accelerated sampler for cloud GPU mining."""

import time
import numpy as np
import collections.abc
import dimod
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY

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

            # Use the correct energy calculation from quantum_proof_of_work
            # Convert state to solution format and calculate energy properly
            from shared.quantum_proof_of_work import energy_of_solution

            # Get the node ordering from the sampler (this should be passed in)
            # For now, assume sequential ordering [0, 1, 2, ..., n-1]
            nodes = list(range(len(state)))
            energy = energy_of_solution(state.tolist(), h_dict, J_dict, nodes)
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


class ModalSampler(MockDWaveSampler):
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

        # Use the default topology (Advantage2) from quantum_proof_of_work
        topology_graph = DEFAULT_TOPOLOGY.graph
        properties = DEFAULT_TOPOLOGY.properties

        super().__init__(
            nodelist=list(topology_graph.nodes()),
            edgelist=list(topology_graph.edges()),
            properties=properties,
            substitute_sampler=self
        )
        
        # Type conversions to match protocol expectations (nodes should be ints for quantum_proof_of_work functions)
        nodes = []
        for node in self.nodelist:
            if not isinstance(node, int):
                raise ValueError(f"Expected node index to be int, got {type(node)}")
            nodes.append(int(node))
        edges = []
        for edge in self.edgelist:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(f"Expected edge to be tuple of length 2, got {edge}")
            if not isinstance(edge[0], int) or not isinstance(edge[1], int):
                raise ValueError(f"Expected edge indices to be int, got {type(edge[0])} and {type(edge[1])}")
            edges.append((int(edge[0]), int(edge[1])))
        self.nodes = nodes
        self.edges = edges

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
        """Sample from Ising model using GPU acceleration."""
        # Convert h and J to dictionaries if needed
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J

        # Run on GPU via Modal (without context manager to avoid nested app.run)
        result = self._gpu_sample_func.remote(h_dict, J_dict, num_reads, num_sweeps)

        # Format result to match D-Wave interface
        samples = result["samples"]
        energies = result["energies"]
        
        # Convert samples to the format expected by dimod.SampleSet.from_samples
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)