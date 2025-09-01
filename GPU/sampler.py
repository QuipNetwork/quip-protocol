"""Local GPU sampler using PyTorch (CUDA or MPS) directly."""

import os
import math
from typing import Any, Dict, List, Tuple, Optional
import collections.abc

import dimod
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import create_topology_graph, get_topology_properties

Variable = collections.abc.Hashable

# Optional torch import
try:
    import torch
except ImportError:
    torch = None


class GPUSampler(MockDWaveSampler):
    """Local GPU sampler using PyTorch (CUDA or MPS) directly."""

    def __init__(self, device: str):
        self._device_str = str(device)
        self._debug = os.getenv("QUIP_DEBUG") == "1"
        self.sampler_type = "gpu"
        
        if torch is None:
            raise RuntimeError("PyTorch not available")
            
        # Initialize device
        self._device = self._init_device(self._device_str)
        
        if self._debug:
            print(f"[GPU sampler pid={os.getpid()}] initialized device={self._device}", flush=True)

        # Use the default topology (Pegasus) from quantum_proof_of_work
        topology_graph = create_topology_graph()  # Uses DEFAULT_TOPOLOGY (Pegasus)
        properties = get_topology_properties()

        super().__init__(
            nodelist=list(topology_graph.nodes()),
            edgelist=list(topology_graph.edges()),
            properties=properties,
            substitute_sampler=self
        )

        self.nodelist: List[Variable] = list(topology_graph.nodes())
        self.edgelist: List[Tuple[Variable, Variable]] = list(topology_graph.edges())
        self.properties: Dict[str, Any] = properties
        
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
        
    def _init_device(self, device_str: str):
        """Initialize PyTorch device."""
        if device_str.lower() == "mps":
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                raise RuntimeError("MPS not available on this system")
        else:
            # assume CUDA ordinal
            idx = int(device_str)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            if idx < 0 or idx >= torch.cuda.device_count():
                raise RuntimeError(f"Invalid CUDA device index {idx}")
            return torch.device(f"cuda:{idx}")

    def close(self):
        # No worker process to clean up anymore
        pass

    def __del__(self):
        self.close()
    
    def _gpu_simulated_annealing(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], 
                                num_reads: int, num_sweeps: int) -> Tuple[List[List[int]], List[float]]:
        """Run simulated annealing on GPU."""
        # Build tensors
        n = 0
        if h:
            n = max(n, max(h.keys()) + 1)
        if J:
            n = max(n, max(max(i, j) for (i, j) in J.keys()) + 1)
        if n <= 0:
            raise ValueError("Invalid problem size")
            
        h_vec = torch.zeros(n, device=self._device, dtype=torch.float32)
        for i, v in h.items():
            h_vec[i] = float(v)
            
        if J:
            i_idx = torch.tensor([ij[0] for ij in J.keys()], device=self._device, dtype=torch.long)
            j_idx = torch.tensor([ij[1] for ij in J.keys()], device=self._device, dtype=torch.long)
            j_vals = torch.tensor([float(v) for v in J.values()], device=self._device, dtype=torch.float32)
        else:
            i_idx = j_idx = j_vals = None
            
        # Generate random spins {-1,1}
        spins = (torch.rand((num_reads, n), device=self._device) > 0.5).to(torch.int8)
        spins = spins * 2 - 1  # {0,1} -> {-1,1}

        # Simulated annealing using edge list (no dense J)
        # Match CPU SA semantics: one sweep ≈ n spin updates per read.
        # Use geometric beta schedule to mimic D-Wave SA behavior.
        betas = torch.exp(torch.linspace(math.log(0.1), math.log(10.0), steps=num_sweeps, device=self._device, dtype=torch.float32))
        R = num_reads
        ar = torch.arange(R, device=self._device)
        # Optimize for Metal: reduce operations, larger batches
        updates_per_sweep = int(os.getenv("QUIP_GPU_UPDATES_PER_SWEEP", str(max(n // 4, 100))))
        recompute_interval = int(os.getenv("QUIP_GPU_RECOMPUTE_INTERVAL", str(min(updates_per_sweep, 256))))

        for beta in betas:
            # Initial local field at this temperature
            if i_idx is not None:
                sp_f = spins.to(torch.float32)
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                # Contributions from edges (i -> j) and (j -> i)
                neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
            else:
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
            local_field = neighbor_sum + h_vec  # broadcasts h_vec over reads

            # Perform ~n updates per sweep with periodic field recomputation
            t = 0
            while t < updates_per_sweep:
                chunk = min(recompute_interval, updates_per_sweep - t)
                for _ in range(chunk):
                    idx = torch.randint(0, n, (R,), device=self._device)
                    s_i = spins[ar, idx].to(torch.float32)
                    lf_i = local_field[ar, idx]
                    delta_e = 2.0 * s_i * lf_i
                    accept = (delta_e < 0) | (torch.rand(R, device=self._device) < torch.exp(-beta * delta_e))
                    # Flip accepted spins - optimize tensor creation
                    spins[ar[accept], idx[accept]] *= -1

                # Recompute local fields after chunk
                if i_idx is not None:
                    sp_f = spins.to(torch.float32)
                    neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                    neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                    neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
                    local_field = neighbor_sum + h_vec
                t += chunk

        # Compute final energies
        h_energy = (spins.to(torch.float32) * h_vec).sum(dim=1)
        if i_idx is not None:
            j_energy = (spins[:, i_idx] * spins[:, j_idx] * j_vals).sum(dim=1)
        else:
            j_energy = torch.zeros(R, device=self._device)
        energies = h_energy + j_energy

        # Convert to Python lists
        samples_cpu = spins.cpu().tolist()
        energies_cpu = energies.cpu().tolist()
        
        return samples_cpu, energies_cpu

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
        """Run simulated annealing on GPU directly."""
        if self._debug:
            print(f"[GPU sampler pid={os.getpid()}] sampling device={self._device} reads={num_reads} sweeps={num_sweeps}", flush=True)
        
        # Convert to dicts for processing
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        
        # Run GPU simulated annealing
        samples, energies = self._gpu_simulated_annealing(h_dict, J_dict, num_reads, num_sweeps)

        # Convert samples to the format expected by dimod.SampleSet.from_samples
        # samples should be a list of dicts mapping variables to values
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)

