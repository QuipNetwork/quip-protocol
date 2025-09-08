"""Local GPU sampler using PyTorch (CUDA or MPS) directly."""

import os
import math
import logging
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

    def __init__(self, device: str, logger: Optional[logging.Logger] = None):
        self._device_str = str(device)
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "gpu"
        
        if torch is None:
            raise RuntimeError("PyTorch not available")
            
        # Initialize device
        self._device = self._init_device(self._device_str)
        
        self.logger.debug(f"[GPU sampler pid={os.getpid()}] initialized device={self._device}")

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
        print(f"[GPU annealing] Starting with |h|={len(h)}, |J|={len(J)}", flush=True)
        
        # Build tensors
        print(f"[GPU annealing] Building tensors...", flush=True)
        # CRITICAL FIX: Use actual number of nodes, not max node ID + 1
        # Pegasus topology has 5640 nodes but max node ID is 5729 (gaps in numbering)
        # We need tensor size = number of actual nodes, not max ID + 1
        n = len(self.nodes)  # Use actual node count (5640) not max node ID + 1 (5730)
        print(f"[GPU annealing] Problem size n={n} (corrected from max node ID approach)", flush=True)
        
        # CRITICAL FIX: Create node_id → position mapping for non-sequential Pegasus topology  
        # Pegasus nodes are like [30,31,...,2849,2910,...,5729] with gaps, not [0,1,2,...,n-1]
        node_to_pos = {int(node_id): pos for pos, node_id in enumerate(self.nodes)}
        print(f"[GPU annealing] Created node mapping for {len(self.nodes)} Pegasus nodes", flush=True)
            
        print(f"[GPU annealing] Creating h_vec tensor with proper node mapping...", flush=True)
        h_vec = torch.zeros(n, device=self._device, dtype=torch.float32)
        for node_id, v in h.items():
            pos = node_to_pos.get(int(node_id))
            if pos is not None:
                h_vec[pos] = float(v)
        print(f"[GPU annealing] h_vec created with {len(h)} nonzero elements correctly mapped", flush=True)
            
        if J:
            print(f"[GPU annealing] Creating J tensors with proper node mapping...", flush=True)
            # Convert node IDs to tensor positions using the mapping
            i_pos = []
            j_pos = []
            j_vals_list = []
            unmapped_edges = 0
            for (node_i, node_j), val in J.items():
                pos_i = node_to_pos.get(int(node_i))
                pos_j = node_to_pos.get(int(node_j))
                if pos_i is not None and pos_j is not None:
                    i_pos.append(pos_i)
                    j_pos.append(pos_j)
                    j_vals_list.append(float(val))
                else:
                    unmapped_edges += 1
            
            i_idx = torch.tensor(i_pos, device=self._device, dtype=torch.long)
            j_idx = torch.tensor(j_pos, device=self._device, dtype=torch.long)
            j_vals = torch.tensor(j_vals_list, device=self._device, dtype=torch.float32)
            print(f"[GPU annealing] J tensors created with {len(j_vals_list)}/{len(J)} edges correctly mapped", flush=True)
        else:
            i_idx = j_idx = j_vals = None
            print(f"[GPU annealing] No J coupling terms", flush=True)
            
        # Generate random spins {-1,1}
        print(f"[GPU annealing] Generating initial spins ({num_reads} x {n})...", flush=True)
        spins = (torch.rand((num_reads, n), device=self._device) > 0.5).to(torch.int8)
        spins = spins * 2 - 1  # {0,1} -> {-1,1}
        print(f"[GPU annealing] Initial spins generated", flush=True)

        # Highly optimized simulated annealing for GPU
        print(f"[GPU annealing] Starting optimized annealing loop...", flush=True)
        
        # CPU-LIKE: Match working CPU SA parameters for correctness
        target_sweeps = num_sweeps  # Use full sweep count like CPU
        print(f"[GPU annealing] CPU-LIKE: Using full {target_sweeps} sweeps for SA correctness", flush=True)
        
        # OPTIMIZED cooling schedule - match DWave's proven SA approach  
        # Use the actual DWave temperature range that works for CPU
        beta_start = 0.1   # DWave standard start (T=10)
        beta_end = 10.0    # Higher end temp for better fine-tuning (T=0.1)
        betas = torch.logspace(math.log10(beta_start), math.log10(beta_end), steps=target_sweeps, device=self._device, dtype=torch.float32)
        
        # INCREASED parallelism: Use GPU's strength to run more chains in parallel
        R = max(num_reads, 64)  # More parallel chains for better solutions
        if R != num_reads:
            print(f"[GPU annealing] Using {R} parallel chains (was {num_reads}) to find best solutions", flush=True)
        
        ar = torch.arange(R, device=self._device)
        
        # BALANCED updates per sweep - not too many to avoid inefficiency
        updates_per_sweep = max(n // 8, 128)  # Moderate updates for efficiency
        print(f"[GPU annealing] Using {updates_per_sweep} updates per sweep (balanced approach)", flush=True)
        
        # Pre-allocate tensors with the optimized dimensions
        neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
        sp_f = torch.zeros((R, n), device=self._device, dtype=torch.float32)
        
        # Also need to regenerate spins for the new R dimension
        if R != num_reads:
            spins = (torch.rand((R, n), device=self._device) > 0.5).to(torch.int8)
            spins = spins * 2 - 1  # {0,1} -> {-1,1}
            ar = torch.arange(R, device=self._device)  # Update ar as well
        
        print(f"[GPU annealing] Pre-allocated tensors ({R}, {n})", flush=True)

        for sweep_idx, beta in enumerate(betas):
            if sweep_idx % max(target_sweeps // 5, 1) == 0:
                print(f"[GPU annealing] Sweep {sweep_idx}/{target_sweeps}", flush=True)
                
            # ULTRA-FAST: Use matrix multiplication instead of scatter operations
            if i_idx is not None:
                # Convert spins to float32 only once per sweep
                spins_float = spins.to(torch.float32)  # (R, n)
                
                # Create sparse adjacency-like computation using advanced indexing
                # This is MUCH faster than scatter operations for large tensors
                edge_values_i = spins_float[:, i_idx] * j_vals  # (R, num_edges)
                edge_values_j = spins_float[:, j_idx] * j_vals  # (R, num_edges)
                
                # Fast parallel reduction using optimized scatter operations
                neighbor_sum.zero_()
                # Use the much faster scatter_add with proper tensor shapes
                neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), edge_values_i)
                neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), edge_values_j)
            else:
                neighbor_sum.zero_()
            local_field = neighbor_sum + h_vec  # (R, n)

            # CORRECTED SA: Use proper incremental updates with accurate local fields  
            # Previous approach used stale local fields causing poor convergence
            
            # Balance batch size for GPU efficiency and SA accuracy
            batch_size = min(32, updates_per_sweep // 4)  # Moderate batches for efficiency
            num_batches = max(updates_per_sweep // batch_size, 1)  # Ensure at least 1 batch
            
            for batch_idx in range(num_batches):
                # Generate random spin indices for this batch
                spin_indices = torch.randint(0, n, (batch_size, R), device=self._device, dtype=torch.long)
                spin_indices_T = spin_indices.T  # (R, batch_size)
                ar_expanded = ar.unsqueeze(1).expand(-1, batch_size)  # (R, batch_size)
                
                # Get current spins and local fields for this batch
                current_spins = spins[ar_expanded, spin_indices_T].to(torch.float32)  # (R, batch_size)
                current_fields = local_field[ar_expanded, spin_indices_T]  # (R, batch_size)
                
                # Compute energy change for flipping each spin
                delta_energies = 2.0 * current_spins * current_fields  # (R, batch_size)
                
                # Metropolis acceptance criterion (CORRECTED: match Metal sampler for energy maximization)
                random_vals = torch.rand_like(delta_energies)
                accept_mask = (delta_energies > 0) | (random_vals < torch.exp(-beta * torch.abs(delta_energies)))
                
                # Apply accepted spin flips
                accepted_positions = torch.where(accept_mask)
                if len(accepted_positions[0]) > 0:
                    read_ids = accepted_positions[0]  # R dimension indices
                    batch_ids = accepted_positions[1]  # batch_size dimension indices
                    spin_positions = spin_indices_T[read_ids, batch_ids]
                    
                    # Flip the accepted spins
                    spins[read_ids, spin_positions] *= -1
                    
                    # UPDATE LOCAL FIELD after each batch to maintain SA accuracy
                    # This is crucial - stale local fields cause poor optimization
                    if i_idx is not None:
                        spins_float = spins.to(torch.float32)
                        neighbor_sum.zero_()
                        neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), spins_float[:, i_idx] * j_vals)
                        neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), spins_float[:, j_idx] * j_vals)
                        local_field = neighbor_sum + h_vec

        print(f"[GPU annealing] Annealing loop completed, computing final energies...", flush=True)
        
        # OPTIMIZED energy computation - minimize GPU->CPU transfers
        spins_float = spins.to(torch.float32)  # Convert once
        h_energy = torch.sum(spins_float * h_vec, dim=1)  # (R,)
        
        if i_idx is not None:
            # Efficient edge energy computation using int8 arithmetic where possible
            spin_products = spins[:, i_idx] * spins[:, j_idx]  # int8 * int8 = int8 (faster)
            j_energy = torch.sum(spin_products.to(torch.float32) * j_vals, dim=1)  # Convert only final result
        else:
            j_energy = torch.zeros(R, device=self._device, dtype=torch.float32)
        
        total_energies = h_energy + j_energy  # (R,)
        
        
        
        # Select best samples from the increased parallelism
        # If we increased R, return only the requested number of best samples
        if R > num_reads:
            _, best_indices = torch.topk(total_energies, num_reads, largest=False)  # Get lowest energies
            final_spins = spins[best_indices]
            final_energies = total_energies[best_indices]
        else:
            final_spins = spins
            final_energies = total_energies
        
        print(f"[GPU annealing] Returning {len(final_energies)} best samples from {R} parallel chains", flush=True)
        
        # Convert to Python lists (matches expected interface)
        samples_cpu = final_spins.cpu().tolist()
        energies_cpu = final_energies.cpu().tolist()
        
        return samples_cpu, energies_cpu

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs) -> dimod.SampleSet:
        """Run simulated annealing on GPU directly."""
        print(f"[GPU sampler] Starting sampling on device={self._device} reads={num_reads} sweeps={num_sweeps}", flush=True)
        
        self.logger.debug(f"[GPU sampler pid={os.getpid()}] sampling device={self._device} reads={num_reads} sweeps={num_sweeps}")
        
        # Convert to dicts for processing
        print(f"[GPU sampler] Converting h,J to dicts...", flush=True)
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        print(f"[GPU sampler] Converted to dicts: |h|={len(h_dict)}, |J|={len(J_dict)}", flush=True)
        
        # Run GPU simulated annealing
        print(f"[GPU sampler] Calling _gpu_simulated_annealing...", flush=True)
        samples, energies = self._gpu_simulated_annealing(h_dict, J_dict, num_reads, num_sweeps)
        print(f"[GPU sampler] GPU annealing completed, got {len(samples)} samples", flush=True)

        # Convert samples to the format expected by dimod.SampleSet.from_samples
        # samples should be a list of dicts mapping variables to values
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)

