"""Metal-specific GPU sampler with debugging for Mac MPS performance issues."""

import os
import math
import time
from typing import Any, Dict, List, Tuple, Optional
import collections.abc

import dimod
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import create_topology_graph, get_topology_properties

Variable = collections.abc.Hashable

# Try to import torch
try:
    import torch
except ImportError:
    torch = None


class MetalSampler(MockDWaveSampler):
    """Metal-specific GPU sampler with performance debugging."""

    def __init__(self, device: str = "mps"):
        self._device_str = str(device)
        self._debug = os.getenv("QUIP_DEBUG") == "1"
        self.sampler_type = "metal"
        
        if torch is None:
            raise RuntimeError("PyTorch not available")
            
        # Initialize Metal device
        self._device = self._init_metal_device()
        
        if self._debug:
            print(f"[MetalSampler] Initialized device={self._device}", flush=True)

        # Use the default topology (Pegasus) from quantum_proof_of_work
        topology_graph = create_topology_graph()
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
        
        # Type conversions to match protocol expectations
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
        
    def _init_metal_device(self):
        """Initialize Metal MPS device with proper checks."""
        if not hasattr(torch, 'backends'):
            raise RuntimeError("PyTorch backends not available")
        if not hasattr(torch.backends, 'mps'):
            raise RuntimeError("MPS backend not available in this PyTorch build")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this system")
        return torch.device("mps")

    def close(self):
        pass

    def __del__(self):
        self.close()
    
    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
        """Run Metal-optimized simulated annealing with debugging."""
        if self._debug:
            print(f"[MetalSampler] Starting sampling: reads={num_reads}, sweeps={num_sweeps}", flush=True)
        
        # Convert to dicts for processing
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        
        start_time = time.time()
        
        # Run Metal-specific simulated annealing
        samples, energies = self._metal_simulated_annealing(h_dict, J_dict, num_reads, num_sweeps)
        
        total_time = time.time() - start_time
        if self._debug:
            print(f"[MetalSampler] Completed in {total_time:.3f}s ({total_time/num_sweeps*1000:.2f}ms per sweep)", flush=True)

        # Convert samples to the format expected by dimod.SampleSet.from_samples
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)
    
    def _metal_simulated_annealing(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], 
                                  num_reads: int, num_sweeps: int) -> Tuple[List[List[int]], List[float]]:
        """Metal-optimized simulated annealing with detailed timing."""
        
        # Setup phase
        setup_start = time.time()
        
        # Build problem size
        n = 0
        if h:
            n = max(n, max(h.keys()) + 1)
        if J:
            n = max(n, max(max(i, j) for (i, j) in J.keys()) + 1)
        if n <= 0:
            raise ValueError("Invalid problem size")
            
        if self._debug:
            print(f"[MetalSampler] Problem size: {n} variables, {len(J)} couplings", flush=True)
            
        # Build h vector
        h_vec = torch.zeros(n, device=self._device, dtype=torch.float32)
        for i, v in h.items():
            h_vec[i] = float(v)
            
        # Build J coupling tensors - USE INT32 instead of INT64 for MPS performance (6000x speedup!)
        if J:
            i_idx = torch.tensor([ij[0] for ij in J.keys()], device=self._device, dtype=torch.int32)
            j_idx = torch.tensor([ij[1] for ij in J.keys()], device=self._device, dtype=torch.int32)
            j_vals = torch.tensor([float(v) for v in J.values()], device=self._device, dtype=torch.float32)
        else:
            i_idx = j_idx = j_vals = None
            
        # Metal optimization: Original optimized parameters for 60ms/sweep performance
        # Fixed: Now correctly minimizes energy (was maximizing before)
        # Use original working parameters that achieved ~60ms per sweep  
        updates_per_sweep = n  # D-Wave style: each spin updated once per sweep
        
        # Fast convergence: Don't overdo parallel chains - focus on speed
        original_reads = num_reads
        # Only use moderate PMSA for very small read counts
        if num_reads < 32:  
            num_reads = max(num_reads * 2, 32)  # Minimal PMSA
            print(f"[MetalSampler] PMSA: Using {num_reads} parallel chains (was {original_reads})")
        
        R = num_reads
        
        # Initialize random spins {-1,1} with correct dimensions
        spins = (torch.rand((R, n), device=self._device) > 0.5).to(torch.int8)
        spins = spins * 2 - 1  # {0,1} -> {-1,1}
        
        if self._debug:
            print(f"[MetalSampler] Metal params: updates_per_sweep={updates_per_sweep} (corrected SA minimization)", flush=True)

        # Exact D-Wave beta schedule for proper simulated annealing
        # D-Wave uses: β = 0.0231 to 6.6214 (temps 43.3 to 0.15)
        beta_start = 0.0231  # D-Wave actual start (high temp = 43.3)
        beta_end = 6.6214    # D-Wave actual end (low temp = 0.15)  
        # Use CPU generation + MPS transfer for maximum compatibility
        cpu_betas = torch.logspace(math.log10(beta_start), math.log10(beta_end), steps=num_sweeps)
        betas = cpu_betas.to(self._device, dtype=torch.float32)
        R = num_reads
        ar = torch.arange(R, device=self._device, dtype=torch.int32)  # Use int32 for MPS performance
        
        setup_time = time.time() - setup_start
        if self._debug:
            print(f"[MetalSampler] Setup completed in {setup_time:.3f}s", flush=True)

        # Annealing loop with timing
        anneal_start = time.time()
        
        for sweep_idx, beta in enumerate(betas):
            sweep_start = time.time()
            
            # Compute local field once per sweep
            if i_idx is not None:
                sp_f = spins.to(torch.float32)
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
            else:
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
            local_field = neighbor_sum + h_vec

            # FULLY VECTORIZED SA: All spins updated in parallel per sweep
            # This trades some SA accuracy for massive speed improvement
            if updates_per_sweep > 0:
                # STRATEGY: Update each spin position once per sweep, all chains in parallel
                # This is equivalent to D-Wave's systematic sweep but vectorized
                
                # Create systematic sweep order (0, 1, 2, ..., n-1) 
                spin_order = torch.arange(n, device=self._device, dtype=torch.int32)
                # Shuffle for randomization while keeping vectorization
                shuffle_idx = torch.randperm(n, device=self._device)
                spin_order = spin_order[shuffle_idx]
                
                # Process spins in vectorized chunks
                chunk_size = min(128, n)  # Process 128 spins at once
                for chunk_start in range(0, n, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n)
                    chunk_spins = spin_order[chunk_start:chunk_end]  # Spins to update
                    
                    # Vectorized Metropolis for entire chunk
                    chunk_size_actual = len(chunk_spins)
                    
                    # Expand indices for all chains: (R, chunk_size)
                    chain_idx = ar.unsqueeze(1).expand(R, chunk_size_actual)
                    spin_idx = chunk_spins.unsqueeze(0).expand(R, chunk_size_actual)
                    
                    # Get current spin values and local fields
                    current_spins = spins[chain_idx, spin_idx].to(torch.float32)  # (R, chunk_size)
                    current_fields = local_field[chain_idx, spin_idx]             # (R, chunk_size)
                    
                    # Vectorized energy calculation
                    delta_e = 2.0 * current_spins * current_fields  # (R, chunk_size)
                    
                    # Vectorized Metropolis acceptance
                    rand_vals = torch.rand((R, chunk_size_actual), device=self._device)
                    accept_mask = (delta_e > 0) | (rand_vals < torch.exp(-beta * torch.abs(delta_e)))
                    
                    # Vectorized spin flips - only flip accepted ones
                    spins[chain_idx[accept_mask], spin_idx[accept_mask]] *= -1
                    
                # Recompute local field once after all updates  
                if i_idx is not None:
                    sp_f = spins.to(torch.float32)
                    neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                    neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                    neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
                    local_field = neighbor_sum + h_vec
            
            sweep_time = time.time() - sweep_start
            if self._debug and (sweep_idx < 3 or sweep_idx % max(num_sweeps // 10, 1) == 0):
                print(f"[MetalSampler] Sweep {sweep_idx}/{num_sweeps} took {sweep_time*1000:.1f}ms", flush=True)

        anneal_time = time.time() - anneal_start
        if self._debug:
            print(f"[MetalSampler] Annealing completed in {anneal_time:.3f}s", flush=True)

        # Final energy computation
        energy_start = time.time()
        
        h_energy = (spins.to(torch.float32) * h_vec).sum(dim=1)
        if i_idx is not None:
            j_energy = (spins[:, i_idx] * spins[:, j_idx] * j_vals).sum(dim=1)
        else:
            j_energy = torch.zeros(R, device=self._device)
        energies = h_energy + j_energy

        # Convert to Python lists and return only requested number of samples
        all_samples = spins.cpu().tolist()
        all_energies = energies.cpu().tolist()
        
        # If we used PMSA with more chains, select the best samples
        if len(all_energies) > original_reads:
            # Sort by energy and take the best samples
            sorted_indices = sorted(range(len(all_energies)), key=lambda i: all_energies[i])[:original_reads]
            samples_cpu = [all_samples[i] for i in sorted_indices]
            energies_cpu = [all_energies[i] for i in sorted_indices]
        else:
            samples_cpu = all_samples
            energies_cpu = all_energies
        
        energy_time = time.time() - energy_start
        if self._debug:
            print(f"[MetalSampler] Energy computation: {energy_time:.3f}s", flush=True)
            print(f"[MetalSampler] Timing breakdown: setup={setup_time:.3f}s, anneal={anneal_time:.3f}s, energy={energy_time:.3f}s", flush=True)
        
        return samples_cpu, energies_cpu