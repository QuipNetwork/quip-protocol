"""Optimized Metal GPU sampler with large chunks and frequent field updates for maximum performance."""

import os
import math
import time
import logging
from typing import Any, Dict, List, Tuple, Optional
import collections.abc

import dimod
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY

Variable = collections.abc.Hashable

# Try to import torch
try:
    import torch
except ImportError:
    torch = None

# Try to import Metal kernels
try:
    from .metal_kernel_interface import MetalKernelInterface
    METAL_KERNELS_AVAILABLE = True
except ImportError:
    METAL_KERNELS_AVAILABLE = False


class OptimizedChunkMetalSampler(MockDWaveSampler):
    """Optimized Metal GPU sampler with large chunks and frequent field updates."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self._device_str = str(device)
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "optimized_chunk_metal"
        
        if torch is None:
            raise RuntimeError("PyTorch not available")
            
        # Initialize Metal device
        self._device = self._init_metal_device()
        
        # Try to initialize Metal kernels for better performance
        self._metal_kernels = None
        if METAL_KERNELS_AVAILABLE:
            try:
                self._metal_kernels = MetalKernelInterface(device, logger)
                self.logger.info(f"[OptimizedChunkMetalSampler] Metal kernels initialized successfully")
            except Exception as e:
                self.logger.warning(f"[OptimizedChunkMetalSampler] Metal kernels failed to initialize: {e}, falling back to PyTorch MPS")
                self._metal_kernels = None
        else:
            self.logger.info(f"[OptimizedChunkMetalSampler] Metal kernels not available, using PyTorch MPS")
        
        self.logger.debug(f"[OptimizedChunkMetalSampler] Initialized device={self._device}")

        # Use the default topology (Advantage2) from quantum_proof_of_work
        topology_graph = DEFAULT_TOPOLOGY.graph
        properties = DEFAULT_TOPOLOGY.properties

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
        
        # Pre-compute and cache node mapping
        self._node_to_pos = {int(node_id): pos for pos, node_id in enumerate(self.nodes)}

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
        """Run optimized chunk Metal simulated annealing."""
        self.logger.debug(f"[OptimizedChunkMetalSampler] Starting sampling: reads={num_reads}, sweeps={num_sweeps}")
        
        # Convert to dicts for processing
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        
        start_time = time.time()
        
        # Optimized chunk sampling approach
        samples, energies = self._optimized_chunk_sampling(h_dict, J_dict, num_reads, num_sweeps)
        
        total_time = time.time() - start_time
        self.logger.debug(f"[OptimizedChunkMetalSampler] Completed in {total_time:.3f}s ({total_time/num_sweeps*1000:.2f}ms per sweep)")

        # Convert samples to the format expected by dimod.SampleSet.from_samples
        sample_dicts = []
        for sample in samples:
            sample_dict = {i: sample[i] for i in range(len(sample))}
            sample_dicts.append(sample_dict)
        
        # Create proper dimod.SampleSet
        return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)
    
    def _optimized_chunk_sampling(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], 
                                  num_reads: int, num_sweeps: int) -> Tuple[List[List[int]], List[float]]:
        """Optimized chunk Metal sampling with large chunks and frequent field updates."""
        
        timing_data = {}
        setup_start = time.time()
        
        # Build problem size
        n = 0
        if h:
            n = max(n, max(h.keys()) + 1)
        if J:
            n = max(n, max(max(i, j) for (i, j) in J.keys()) + 1)
        if n <= 0:
            raise ValueError("Invalid problem size")
            
        self.logger.debug(f"[OptimizedChunkMetalSampler] Problem size: {n} variables, {len(J)} couplings")
            
        # Use cached node mapping
        node_to_pos = self._node_to_pos
        
        # Build h vector using position mapping
        h_build_start = time.time()
        h_vec = torch.zeros(n, device=self._device, dtype=torch.float32)
        for node_id, v in h.items():
            pos = node_to_pos.get(int(node_id))
            if pos is not None:
                h_vec[pos] = float(v)
        timing_data['h_vector_build'] = time.time() - h_build_start
            
        # Build J coupling tensors
        j_build_start = time.time()
        if J:
            i_pos = []
            j_pos = []
            j_vals_list = []
            for (node_i, node_j), val in J.items():
                pos_i = node_to_pos.get(int(node_i))
                pos_j = node_to_pos.get(int(node_j))
                if pos_i is not None and pos_j is not None:
                    i_pos.append(pos_i)
                    j_pos.append(pos_j) 
                    j_vals_list.append(float(val))
                    
            i_idx = torch.tensor(i_pos, device=self._device, dtype=torch.int32)
            j_idx = torch.tensor(j_pos, device=self._device, dtype=torch.int32)
            j_vals = torch.tensor(j_vals_list, device=self._device, dtype=torch.float32)
        else:
            i_idx = j_idx = j_vals = None
        timing_data['j_tensor_build'] = time.time() - j_build_start
            
        # Conservative PMSA + optimized chunk size
        updates_per_sweep = n
        original_reads = num_reads
        
        if num_reads < 32:
            num_reads = max(num_reads * 2, 32)
            self.logger.debug(f"[OptimizedChunkMetalSampler] Conservative PMSA: Using {num_reads} parallel chains (was {original_reads})")
        
        R = num_reads
        
        # Initialize spins
        init_start = time.time()
        spins = torch.randint(0, 2, (R, n), device=self._device, dtype=torch.int8) * 2 - 1
        timing_data['spin_initialization'] = time.time() - init_start
        
        # ENERGY IMPROVEMENT 1: Better annealing schedule  
        schedule_start = time.time()
        beta_start = 0.0231
        beta_end = 6.6214
        
        # Use a more gradual schedule that spends more time in critical temperature ranges
        # This improves energy quality with minimal performance impact
        schedule_points = torch.linspace(0, 1, steps=num_sweeps)
        # Apply power law to spend more time at moderate temperatures (better convergence)
        schedule_points = torch.pow(schedule_points, 0.8)  # Slightly slower schedule
        cpu_betas = beta_start * torch.pow(beta_end / beta_start, schedule_points)
        betas = cpu_betas.to(self._device, dtype=torch.float32)
        ar = torch.arange(R, device=self._device, dtype=torch.int32)
        timing_data['schedule_setup'] = time.time() - schedule_start
        
        setup_time = time.time() - setup_start
        self.logger.debug(f"[OptimizedChunkMetalSampler] Setup completed in {setup_time:.3f}s")

        # OPTIMIZED CHUNK ANNEALING: Large chunks with frequent field updates
        anneal_start = time.time()
        timing_data['sweep_times'] = []
        timing_data['field_computation_times'] = []
        timing_data['update_times'] = []
        
        # ENERGY IMPROVEMENT 3: Adaptive chunk sizing and batch-aware field updates
        base_chunk_size = min(1536, n)  # Based on experimental results
        
        # BATCH SIZE OPTIMIZATION: More frequent field updates for higher batch sizes
        # Higher batch sizes need more frequent field updates to maintain energy quality
        if R >= 400:      # Ultra high batch sizes (800+)
            field_update_freq = 1  # Update every chunk
        elif R >= 200:    # High batch sizes (200-400)  
            field_update_freq = 1  # Update every chunk
        else:             # Normal batch sizes (<200)
            field_update_freq = 1  # Update every chunk (consistent with current optimized performance)
        
        self.logger.debug(f"[OptimizedChunkMetalSampler] Using batch-optimized configuration: base_chunk_size={base_chunk_size}, batch_size={R}, field_freq={field_update_freq}")
        
        for sweep_idx, beta in enumerate(betas):
            sweep_start = time.time()
            
            # ENERGY IMPROVEMENT 3: Use smaller chunks during critical annealing phase for better convergence
            annealing_progress = sweep_idx / num_sweeps
            if 0.4 <= annealing_progress <= 0.8:  # Critical phase where fine-tuning happens
                chunk_size = min(base_chunk_size // 2, n)  # Smaller chunks for better local search
            else:
                chunk_size = base_chunk_size  # Full size for efficiency
            
            # Standard field computation
            field_start = time.time()
            if i_idx is not None:
                sp_f = spins.to(torch.float32)
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
            else:
                neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
            local_field = neighbor_sum + h_vec
            field_time = time.time() - field_start
            timing_data['field_computation_times'].append(field_time)

            # OPTIMIZED LARGE CHUNK UPDATES with frequent field updates
            update_start = time.time()
            if updates_per_sweep > 0:
                # ENERGY IMPROVEMENT 2: Better randomization for exploration
                # Alternate between random and structured orders for better energy landscape exploration
                if sweep_idx % 3 == 0:
                    # Standard random permutation
                    shuffle_idx_cpu = torch.randperm(n)
                elif sweep_idx % 3 == 1:
                    # Reverse order for systematic exploration
                    shuffle_idx_cpu = torch.arange(n - 1, -1, -1)
                else:
                    # Sequential order for local optimization
                    shuffle_idx_cpu = torch.arange(n)
                
                shuffle_idx = shuffle_idx_cpu.to(self._device, dtype=torch.int32, non_blocking=True)
                spin_order = torch.arange(n, device=self._device, dtype=torch.int32)[shuffle_idx]
                
                chunk_count = 0
                
                for chunk_start in range(0, n, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n)
                    chunk_spins = spin_order[chunk_start:chunk_end]
                    chunk_size_actual = len(chunk_spins)
                    
                    # Vectorized Metropolis
                    chain_idx = ar.unsqueeze(1).expand(R, chunk_size_actual)
                    spin_idx = chunk_spins.unsqueeze(0).expand(R, chunk_size_actual)
                    
                    current_spins = spins[chain_idx, spin_idx].to(torch.float32)
                    current_fields = local_field[chain_idx, spin_idx]
                    
                    delta_e = 2.0 * current_spins * current_fields
                    
                    rand_vals = torch.rand((R, chunk_size_actual), device=self._device)
                    accept_mask = (delta_e > 0) | (rand_vals < torch.exp(-beta * torch.abs(delta_e)))
                    
                    # Optimized spin flipping
                    if accept_mask.any():
                        accept_indices = torch.where(accept_mask)
                        chain_indices = chain_idx[accept_indices] 
                        spin_indices = spin_idx[accept_indices]
                        spins[chain_indices, spin_indices] *= -1
                    
                    chunk_count += 1
                    
                    # FREQUENT FIELD UPDATES: Update after every chunk for energy quality
                    if accept_mask.any() and chunk_end < n and chunk_count % field_update_freq == 0:
                        if i_idx is not None:
                            sp_f = spins.to(torch.float32)
                            neighbor_sum = torch.zeros((R, n), device=self._device, dtype=torch.float32)
                            neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                            neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
                            local_field = neighbor_sum + h_vec
            
            update_time = time.time() - update_start
            timing_data['update_times'].append(update_time)
            
            sweep_time = time.time() - sweep_start
            timing_data['sweep_times'].append(sweep_time)
            
            if sweep_idx < 3 or sweep_idx % max(num_sweeps // 10, 1) == 0:
                self.logger.debug(f"[OptimizedChunkMetalSampler] Sweep {sweep_idx}/{num_sweeps} took {sweep_time*1000:.1f}ms (chunk_size={chunk_size})")

        anneal_time = time.time() - anneal_start
        self.logger.debug(f"[OptimizedChunkMetalSampler] Annealing completed in {anneal_time:.3f}s")

        # GPU-optimized energy computation
        energy_start = time.time()
        h_energy = (spins.to(torch.float32) * h_vec).sum(dim=1)
        if i_idx is not None:
            j_energy = (spins[:, i_idx] * spins[:, j_idx] * j_vals).sum(dim=1)
        else:
            j_energy = torch.zeros(R, device=self._device)
        energies = h_energy + j_energy
        
        # Convert to Python lists 
        all_samples = spins.cpu().tolist()
        all_energies = energies.cpu().tolist()
        
        # Select best samples if using PMSA
        if len(all_energies) > original_reads:
            sorted_indices = sorted(range(len(all_energies)), key=lambda i: all_energies[i])[:original_reads]
            samples_cpu = [all_samples[i] for i in sorted_indices]
            energies_cpu = [all_energies[i] for i in sorted_indices]
        else:
            samples_cpu = all_samples
            energies_cpu = all_energies
        
        energy_time = time.time() - energy_start
        timing_data['energy_computation'] = energy_time
        
        # Performance analysis
        total_field_time = sum(timing_data['field_computation_times'])
        total_update_time = sum(timing_data['update_times'])
        avg_sweep_time = sum(timing_data['sweep_times']) / len(timing_data['sweep_times']) if timing_data['sweep_times'] else 0
        
        self.logger.debug(f"[OptimizedChunkMetalSampler] Energy-enhanced timing: setup={setup_time:.3f}s, anneal={anneal_time:.3f}s, energy={energy_time:.3f}s")
        self.logger.debug(f"[OptimizedChunkMetalSampler] Energy-enhanced sweep: avg={avg_sweep_time*1000:.2f}ms, field={total_field_time:.3f}s, update={total_update_time:.3f}s")
        self.logger.debug(f"[OptimizedChunkMetalSampler] Energy improvements: better_schedule, adaptive_chunks, improved_randomization")
        
        return samples_cpu, energies_cpu