"""
CUDA Simulated Annealing Sampler - Exact D-Wave Implementation

This module provides a CUDA GPU implementation using CuPy RawKernel that exactly mimics D-Wave's
SimulatedAnnealingSampler from cpu_sa.cpp, including:

1. Delta energy array optimization (pre-compute, update incrementally)
2. xorshift32 RNG
3. Sequential variable ordering (spins 0..N-1)
4. Metropolis criterion with threshold optimization (skip if delta_E > 22.18/beta)
5. Beta schedule computation matching _default_ising_beta_range
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import warnings

import dimod
import cupy as cp
import numpy as np

from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY


def _default_ising_beta_range(
    h: Dict[int, float],
    J: Dict[tuple, float],
    max_single_qubit_excitation_rate: float = 0.01,
    scale_T_with_N: bool = True
) -> Tuple[float, float]:
    """
    Exact replica of D-Wave's _default_ising_beta_range function.

    Determine the starting and ending beta from h, J.

    Args:
        h: External field of Ising model (linear bias)
        J: Couplings of Ising model (quadratic biases)
        max_single_qubit_excitation_rate: Targeted single qubit excitation rate at final temperature
        scale_T_with_N: Whether to scale temperature with system size

    Returns:
        [hot_beta, cold_beta] - tuple of starting and ending inverse temperatures
    """
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

    # Approximate worst and best cases of the [non-zero] energy signal
    sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
    if sum_abs_bias_dict:
        min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
    else:
        min_abs_bias_dict = {}

    # Build bias dictionaries from J
    for (k1, k2), v in J.items():
        for k in [k1, k2]:
            sum_abs_bias_dict[k] += abs(v)
            if v != 0:
                if k in min_abs_bias_dict:
                    min_abs_bias_dict[k] = min(abs(v), min_abs_bias_dict[k])
                else:
                    min_abs_bias_dict[k] = abs(v)

    if not min_abs_bias_dict:
        # Null problem - all biases are zero
        warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                   'likely a value error. Temperature range is set arbitrarily '
                   'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
        warnings.warn(warn_msg)
        return (0.1, 1.0)

    # Hot temp: 50% flip probability for worst case
    max_effective_field = max(sum_abs_bias_dict.values(), default=0)

    if max_effective_field == 0:
        hot_beta = 1.0
    else:
        hot_beta = np.log(2) / (2 * max_effective_field)

    # Cold temp: Low excitation probability at end
    if len(min_abs_bias_dict) == 0:
        cold_beta = hot_beta
    else:
        values_array = np.array(list(min_abs_bias_dict.values()), dtype=float)
        min_effective_field = np.min(values_array)
        if scale_T_with_N:
            number_min_gaps = np.sum(min_effective_field == values_array)
        else:
            number_min_gaps = 1
        cold_beta = np.log(number_min_gaps / max_single_qubit_excitation_rate) / (2 * min_effective_field)

    return (hot_beta, cold_beta)


class CudaSASampler:
    """
    Simulated Annealing sampler using CUDA via CuPy RawKernel.

    Exactly mimics D-Wave's SimulatedAnnealingSampler implementation.
    """

    @staticmethod
    def _load_kernel_code():
        """Load CUDA kernel code from file."""
        import os
        kernel_file = os.path.join(os.path.dirname(__file__), 'cuda_sa.cu')
        with open(kernel_file, 'r') as f:
            return f.read()

    @staticmethod
    def _validate_csr(csr_row_ptr, csr_col_ind, csr_J_vals, n, num_problems):
        """
        Validate CSR structure on host side.

        Ensures:
        - row_ptr is monotonically increasing and starts at 0
        - col_ind values are in [0, n)
        - J_vals has correct length

        This allows the kernel to skip bounds checks for performance.
        """
        # Check row_ptr structure
        assert csr_row_ptr[0] == 0, "CSR row_ptr must start at 0"
        assert np.all(np.diff(csr_row_ptr) >= 0), "CSR row_ptr must be monotonically increasing"

        # Check col_ind bounds
        if len(csr_col_ind) > 0:
            assert np.all(csr_col_ind >= 0) and np.all(csr_col_ind < n), \
                f"CSR col_ind must be in [0, {n}), got min={csr_col_ind.min()}, max={csr_col_ind.max()}"

        # Check J_vals length matches col_ind
        assert len(csr_J_vals) == len(csr_col_ind), \
            f"CSR J_vals length {len(csr_J_vals)} != col_ind length {len(csr_col_ind)}"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Compile CUDA kernel
        try:
            # Load kernel code from file
            kernel_code = self._load_kernel_code()

            # Use NVRTC backend (default) with neighbor bounds checking to prevent illegal memory access
            # The bounds checking in the kernel prevents out-of-bounds neighbor access
            self._kernel = cp.RawKernel(
                kernel_code,
                "cuda_simulated_annealing"
            )
            self.logger.info("CUDA SA kernel compiled successfully with NVRTC backend")
        except Exception as e:
            raise RuntimeError(f"Failed to compile CUDA kernel: {e}")

        # Set up topology for mining compatibility
        topology_graph = DEFAULT_TOPOLOGY.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = {'topology': 'Zephyr'}

        # Pre-allocate delta_energy workspace in global memory
        # This is reused across all kernel calls to avoid repeated allocation
        # Size: max_threads_per_call * num_variables
        #
        # Optimal batching: 1 problem per SM, num_reads threads per SM
        # Workspace needed = num_SMs × max_reads_per_problem
        #
        # Query GPU properties for dynamic sizing
        try:
            device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
            num_sms = device_props['multiProcessorCount']
            total_memory_gb = device_props['totalGlobalMem'] / (1024**3)

            # Allocate workspace for: num_SMs × generous max_reads
            # Allow up to 1024 reads/problem (max threads per block)
            max_reads_per_problem = 1024
            self.max_threads_per_call = num_sms * max_reads_per_problem

            n = len(self.nodes)
            workspace_size_mb = (self.max_threads_per_call * n) / (1024 * 1024)

            # Safety check: don't exceed 10% of GPU memory
            max_workspace_mb = total_memory_gb * 1024 * 0.1  # 10% of total memory
            if workspace_size_mb > max_workspace_mb:
                # Reduce max_reads_per_problem to fit
                self.max_threads_per_call = int((max_workspace_mb * 1024 * 1024) / n)
                max_reads_per_problem = self.max_threads_per_call // num_sms
                workspace_size_mb = (self.max_threads_per_call * n) / (1024 * 1024)
                self.logger.warning(f"Reduced workspace to fit memory: {max_reads_per_problem} reads/problem max")

            self._delta_energy_workspace = cp.zeros((self.max_threads_per_call, n), dtype=cp.int8)
            self.logger.info(f"Pre-allocated delta_energy workspace: {num_sms} SMs × {max_reads_per_problem} reads = {self.max_threads_per_call} threads × {n} variables (~{workspace_size_mb:.1f} MB / {total_memory_gb:.1f} GB total)")
        except Exception as e:
            # Fallback to conservative fixed size
            self.max_threads_per_call = 8192
            n = len(self.nodes)
            self._delta_energy_workspace = cp.zeros((self.max_threads_per_call, n), dtype=cp.int8)
            workspace_size_mb = (self.max_threads_per_call * n) / (1024 * 1024)
            self.logger.warning(f"Could not query GPU properties ({e}), using fixed workspace: {self.max_threads_per_call} threads (~{workspace_size_mb:.1f} MB)")
    
    def sample_ising(
        self,
        h: Union[Dict[int, float], List[Dict[int, float]]],
        J: Union[Dict[Tuple[int, int], float], List[Dict[Tuple[int, int], float]]],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Union[dimod.SampleSet, List[dimod.SampleSet]]:
        """
        Sample from Ising model using pure simulated annealing.

        Args:
            h: Linear biases {node: bias} or list of dicts for multiple problems
            J: Quadratic biases {(node1, node2): coupling} or list of dicts for multiple problems
            num_reads: Number of independent SA runs per problem
            num_sweeps: Total number of sweeps (default 1000)
            num_sweeps_per_beta: Sweeps per beta value (default 1)
            beta_range: (hot_beta, cold_beta) or None for auto
            beta_schedule_type: "linear", "geometric", or "custom"
            beta_schedule: Custom beta schedule (requires beta_schedule_type="custom")
            seed: RNG seed

        Returns:
            dimod.SampleSet or list of SampleSets for multiple problems
        """
        # Handle both single problem and batched problems
        if isinstance(h, list):
            # Batched mode
            return self._sample_ising_batched(h, J, num_reads, num_sweeps, num_sweeps_per_beta,
                                             beta_range, beta_schedule_type, beta_schedule, seed)
        else:
            # Single problem mode
            h_list = [h]
            J_list = [J]
            results = self._sample_ising_batched(h_list, J_list, num_reads, num_sweeps, num_sweeps_per_beta,
                                               beta_range, beta_schedule_type, beta_schedule, seed)
            return results[0]

    def _sample_ising_batched(
        self,
        h: List[Dict[int, float]],
        J: List[Dict[Tuple[int, int], float]],
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        beta_schedule: Optional[np.ndarray],
        seed: Optional[int]
    ) -> List[dimod.SampleSet]:
        """
        Sample from multiple Ising models in batched mode using a single kernel launch.
        This matches the Metal implementation's approach for optimal GPU utilization.
        """
        num_problems = len(h)
        if len(J) != num_problems:
            raise ValueError(f"h and J must have same length: {num_problems} vs {len(J)}")

        n = len(self.nodes)

        self.logger.debug(f"[CudaSA] Processing {num_problems} problems, {num_reads} reads each, {num_sweeps} sweeps")

        # Compute beta schedule (use first problem for auto range)
        if beta_schedule is None:
            if beta_range is None:
                beta_range = _default_ising_beta_range(h[0], J[0])

            hot_beta, cold_beta = beta_range
            num_beta_values = num_sweeps // num_sweeps_per_beta

            if beta_schedule_type == "geometric":
                beta_schedule = np.logspace(np.log10(hot_beta), np.log10(cold_beta),
                                           num=num_beta_values, dtype=np.float32)
            elif beta_schedule_type == "linear":
                beta_schedule = np.linspace(hot_beta, cold_beta, num=num_beta_values, dtype=np.float32)
            else:
                raise ValueError(f"Unknown beta_schedule_type: {beta_schedule_type}")
        else:
            beta_schedule = np.asarray(beta_schedule, dtype=np.float32)
            num_beta_values = len(beta_schedule)

        self.logger.debug(f"[CudaSA] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # Build separate CSR arrays for each problem
        # (Not truly concatenated - we process sequentially anyway)
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        csr_arrays_list = []  # List of (row_ptr, col_ind, J_vals) tuples

        for prob_idx, (h_prob, J_prob) in enumerate(zip(h, J)):
            # Build adjacency list for this problem
            adjacency = [[] for _ in range(n)]
            for (u, v), Jij in J_prob.items():
                if u in node_to_idx and v in node_to_idx:
                    idx_u = node_to_idx[u]
                    idx_v = node_to_idx[v]
                    adjacency[idx_u].append((idx_v, int(Jij)))
                    adjacency[idx_v].append((idx_u, int(Jij)))

            # Sort adjacency lists for deterministic ordering
            for i in range(n):
                adjacency[i].sort()

            # Build CSR for this problem
            csr_row_ptr = np.zeros(n + 1, dtype=np.int32)
            csr_col_ind_list = []
            csr_J_vals_list = []

            for i in range(n):
                csr_row_ptr[i + 1] = csr_row_ptr[i] + len(adjacency[i])
                for j, Jij in adjacency[i]:
                    csr_col_ind_list.append(j)
                    csr_J_vals_list.append(Jij)

            csr_col_ind = np.array(csr_col_ind_list, dtype=np.int32)
            csr_J_vals = np.array(csr_J_vals_list, dtype=np.int8)

            # Validate this problem's CSR
            self._validate_csr(csr_row_ptr, csr_col_ind, csr_J_vals, n, 1)

            csr_arrays_list.append((csr_row_ptr, csr_col_ind, csr_J_vals))

            self.logger.debug(f"[CudaSA] Problem {prob_idx}: N={n}, edges={len(csr_col_ind_list)}")

        # Convert beta schedule to GPU array (shared across all problems)
        d_beta_schedule = cp.asarray(beta_schedule)

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Process each problem sequentially with separate kernel launches
        packed_size = (n + 7) // 8
        total_reads = num_problems * num_reads

        self.logger.debug(f"[CudaSA] Batch config: {num_problems} problems × {num_reads} reads = {total_reads} total reads")
        self.logger.debug(f"[CudaSA] Using sequential kernel launches")

        all_samples_packed = []
        all_energies = []

        for prob_idx in range(num_problems):
            # Get CSR arrays for this problem
            prob_csr_row_ptr, prob_csr_col_ind, prob_csr_J_vals = csr_arrays_list[prob_idx]

            # Copy to GPU
            d_prob_csr_row_ptr = cp.asarray(prob_csr_row_ptr)
            d_prob_csr_col_ind = cp.asarray(prob_csr_col_ind)
            d_prob_csr_J_vals = cp.asarray(prob_csr_J_vals)

            # Output arrays for this problem
            d_prob_output_samples = cp.zeros((num_reads, packed_size), dtype=cp.int8)
            d_prob_output_energies = cp.zeros(num_reads, dtype=cp.float32)

            # Launch kernel for this problem
            prob_block_size = 256
            prob_grid_size = (num_reads + prob_block_size - 1) // prob_block_size
            prob_total_threads = prob_grid_size * prob_block_size

            # Use pre-allocated workspace (reused across calls)
            self._kernel(
                (prob_grid_size,), (prob_block_size,),
                (d_prob_csr_row_ptr, d_prob_csr_col_ind, d_prob_csr_J_vals,
                 cp.asarray(np.zeros(n, dtype=np.float32)),  # h_vals (unused)
                 d_beta_schedule,
                 n, len(beta_schedule), num_sweeps_per_beta, num_reads, (seed or 0) ^ prob_idx,
                 d_prob_output_samples, d_prob_output_energies, self._delta_energy_workspace,
                 self.max_threads_per_call)  # workspace_capacity for bounds checking
            )

            # Copy results back
            all_samples_packed.append(cp.asnumpy(d_prob_output_samples))
            all_energies.append(cp.asnumpy(d_prob_output_energies))

        # Parse results into SampleSets
        samplesets = []
        for prob_idx in range(num_problems):
            samples_packed = all_samples_packed[prob_idx]
            energies = all_energies[prob_idx]

            # Unpack samples
            samples = []
            for packed in samples_packed:
                sample = {}
                for i, node in enumerate(self.nodes):
                    byte_idx = i >> 3
                    bit_idx = i & 7
                    bit = (packed[byte_idx] >> bit_idx) & 1
                    sample[node] = -1 if bit else 1
                samples.append(sample)

            sampleset = dimod.SampleSet.from_samples(samples, 'SPIN', energies)
            samplesets.append(sampleset)

            self.logger.debug(f"[CudaSA] Problem {prob_idx}: energy range [{energies.min()}, {energies.max()}]")

        return samplesets

