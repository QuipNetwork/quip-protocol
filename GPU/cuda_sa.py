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
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

import dimod
import cupy as cp
import numpy as np

from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY


@dataclass
class IsingJob:
    """Represents a single Ising problem to be solved on GPU."""
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    num_reads: int
    num_sweeps: int
    num_sweeps_per_beta: int
    beta_schedule: Optional[np.ndarray] = None  # Temperature schedule for this job
    seed: Optional[int] = None
    job_id: Optional[int] = None  # Assigned by sampler


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
    Persistent CUDA kernel wrapper for simulated annealing.

    Maintains a resident kernel on GPU that continuously processes jobs
    from a ring buffer queue.
    """

    @staticmethod
    def _load_kernel_code():
        """Load persistent CUDA kernel code from file."""
        import os
        kernel_file = os.path.join(os.path.dirname(__file__), 'cuda_sa.cu')
        with open(kernel_file, 'r') as f:
            return f.read()

    def __init__(self, device: Optional[int] = None, ring_size: int = 16, max_threads_per_job: int = 256):
        """
        Initialize persistent CUDA kernel.

        Args:
            device: CUDA device ID (None = default)
            ring_size: Size of input/output ring buffers
            max_threads_per_job: Maximum threads per job
        """
        self.logger = logging.getLogger(__name__)

        # Set CUDA device
        if device is not None:
            try:
                cp.cuda.Device(device).use()
                self.device_id = device
                self.logger.info(f"Persistent CUDA sampler using device {device}")
            except Exception as e:
                self.logger.warning(f"Failed to set CUDA device {device}: {e}")
                self.device_id = cp.cuda.runtime.getDevice()
        else:
            self.device_id = cp.cuda.runtime.getDevice()

        # Compile kernels with fast math optimizations
        # Note: NVRTC doesn't support -O3, but --use_fast_math enables aggressive optimizations
        try:
            kernel_code = self._load_kernel_code()
            compile_opts = ('--use_fast_math', '--maxrregcount=64')
            self._kernel_persistent = cp.RawKernel(kernel_code, "cuda_sa_persistent", options=compile_opts)
            self._kernel_sa = cp.RawKernel(kernel_code, "cuda_simulated_annealing", options=compile_opts)
            self.logger.info("CUDA kernels compiled successfully (persistent + SA) with fast math")
        except Exception as e:
            raise RuntimeError(f"Failed to compile CUDA kernels: {e}")

        # Setup topology
        topology_graph = DEFAULT_TOPOLOGY.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.n = len(self.nodes)

        # Ring buffer configuration
        self.ring_size = ring_size
        self.max_threads_per_job = max_threads_per_job

        # Allocate ring buffers using zero-copy mapped memory
        self._allocate_ring_buffers()

        # Control flag is now part of the mapped memory (initialized in _allocate_ring_buffers)
        self.control_flag_value = 0  # CONTROL_RUNNING = 0

        # Kernel stream (for persistent kernel)
        self._stream = cp.cuda.Stream()

        # Data transfer stream (for allocations and transfers)
        # Use a separate stream to avoid blocking on the persistent kernel
        self._transfer_stream = cp.cuda.Stream()

        # Job tracking
        self.job_counter = 0
        self.lock = threading.Lock()
        self.job_buffers = {}  # Store per-job GPU buffers

        # Buffer pool for pre-allocated GPU buffers
        # This avoids GPU memory allocation while the persistent kernel is running
        self.buffer_pool_size = self.ring_size  # One buffer set per ring slot
        self.available_buffers = queue.Queue()  # Queue of available buffer indices
        self.buffer_sets = []  # List of (beta_schedule, output_samples, output_energies) tuples

        # Allocate CSR data and output buffers for persistent kernel
        print("[INIT] Allocating persistent kernel data...", flush=True)
        self._allocate_persistent_kernel_data()
        print("[INIT] Persistent kernel data allocated", flush=True)

        # Allocate buffer pool before launching kernel
        print("[INIT] Allocating buffer pool...", flush=True)
        self._allocate_buffer_pool()
        print("[INIT] Buffer pool allocated", flush=True)

        # Launch persistent kernel
        print("[INIT] Launching persistent kernel...", flush=True)
        self._launch_persistent_kernel()
        print("[INIT] Persistent kernel launched", flush=True)

        self.logger.info(f"Persistent kernel initialized: ring_size={ring_size}, max_threads={max_threads_per_job}")

    def _allocate_ring_buffers(self):
        """Allocate ring buffers using zero-copy mapped memory for persistent kernel."""
        # Input ring buffer (jobs)
        # IMPORTANT: Must match CUDA struct layout with padding for pointer alignment
        input_dtype = np.dtype([
            ('job_id', np.int32),
            ('num_reads', np.int32),
            ('num_sweeps', np.int32),
            ('num_sweeps_per_beta', np.int32),
            ('seed', np.uint32),
            ('csr_row_ptr_offset', np.int32),
            ('csr_col_ind_offset', np.int32),
            ('N', np.int32),
            ('beta_schedule', np.uint64),      # 8-byte pointer, aligned to 8
            ('num_betas', np.int32),
            ('_padding', np.int32),             # 4 bytes padding for pointer alignment
            ('output_samples', np.uint64),      # 8-byte pointer, aligned to 8
            ('output_energies', np.uint64),     # 8-byte pointer, aligned to 8
        ])

        # Output ring buffer (results)
        output_dtype = np.dtype([
            ('job_id', np.int32),
            ('num_reads_done', np.int32),
            ('min_energy', np.float32),
            ('avg_energy', np.float32),
        ])

        # CPU-side tracking to avoid GPU synchronization while kernel is running
        self.cpu_input_head = 0
        self.cpu_input_tail = 0
        self.cpu_output_head = 0
        self.cpu_output_tail = 0

        # Use zero-copy (mapped) memory for ring buffers so the persistent kernel
        # can directly access host memory without explicit transfers
        # This is the recommended approach for persistent kernels
        self.logger.info("Allocating zero-copy mapped memory for ring buffers")

        # Calculate sizes
        input_ring_size = self.ring_size * input_dtype.itemsize
        output_ring_size = self.ring_size * output_dtype.itemsize

        # Allocate host-side ring buffers with cudaHostAllocMapped flag (value = 2)
        # We use the raw CUDA runtime API since CuPy doesn't expose this flag
        cudaHostAllocMapped = 2

        # Allocate pinned+mapped memory using CUDA runtime API via ctypes
        import ctypes

        # Load CUDA runtime library
        try:
            cudart = ctypes.CDLL('libcudart.so')
        except:
            cudart = ctypes.CDLL('cudart64_110.dll')  # Windows fallback

        # Define function signatures
        cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        cudart.cudaHostAlloc.restype = ctypes.c_int
        cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

        # Allocate pinned+mapped memory
        h_input_ring_ptr = ctypes.c_void_p()
        h_output_ring_ptr = ctypes.c_void_p()
        h_input_head_ptr = ctypes.c_void_p()
        h_input_tail_ptr = ctypes.c_void_p()
        h_output_head_ptr = ctypes.c_void_p()
        h_output_tail_ptr = ctypes.c_void_p()
        h_control_flag_ptr = ctypes.c_void_p()

        err = cudart.cudaHostAlloc(ctypes.byref(h_input_ring_ptr), input_ring_size, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for input_ring: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_output_ring_ptr), output_ring_size, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for output_ring: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_input_head_ptr), 4, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for input_head: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_input_tail_ptr), 4, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for input_tail: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_output_head_ptr), 4, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for output_head: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_output_tail_ptr), 4, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for output_tail: error {err}")
        err = cudart.cudaHostAlloc(ctypes.byref(h_control_flag_ptr), 4, cudaHostAllocMapped)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed for control_flag: error {err}")

        # Get device pointers to the mapped host memory
        d_input_ring_ptr = ctypes.c_void_p()
        d_output_ring_ptr = ctypes.c_void_p()
        d_input_head_ptr = ctypes.c_void_p()
        d_input_tail_ptr = ctypes.c_void_p()
        d_output_head_ptr = ctypes.c_void_p()
        d_output_tail_ptr = ctypes.c_void_p()
        d_control_flag_ptr = ctypes.c_void_p()

        cudart.cudaHostGetDevicePointer(ctypes.byref(d_input_ring_ptr), h_input_ring_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_output_ring_ptr), h_output_ring_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_input_head_ptr), h_input_head_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_input_tail_ptr), h_input_tail_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_output_head_ptr), h_output_head_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_output_tail_ptr), h_output_tail_ptr, 0)
        cudart.cudaHostGetDevicePointer(ctypes.byref(d_control_flag_ptr), h_control_flag_ptr, 0)

        # Store device pointers as integers
        self.d_input_ring_ptr = d_input_ring_ptr.value
        self.d_output_ring_ptr = d_output_ring_ptr.value
        self.d_input_head_ptr = d_input_head_ptr.value
        self.d_input_tail_ptr = d_input_tail_ptr.value
        self.d_output_head_ptr = d_output_head_ptr.value
        self.d_output_tail_ptr = d_output_tail_ptr.value
        self.d_control_flag_ptr = d_control_flag_ptr.value

        self.logger.info(f"Mapped memory allocated:")
        self.logger.info(f"  d_input_ring_ptr: 0x{self.d_input_ring_ptr:x}")
        self.logger.info(f"  d_output_ring_ptr: 0x{self.d_output_ring_ptr:x}")
        self.logger.info(f"  d_input_head_ptr: 0x{self.d_input_head_ptr:x}")
        self.logger.info(f"  d_input_tail_ptr: 0x{self.d_input_tail_ptr:x}")

        # Create numpy arrays that view the pinned memory using ctypes
        # We need to cast the c_void_p to the appropriate pointer type
        self.h_input_ring_view = np.ctypeslib.as_array(
            ctypes.cast(h_input_ring_ptr.value, ctypes.POINTER(ctypes.c_byte)),
            shape=(input_ring_size,)
        ).view(input_dtype).reshape(self.ring_size)
        self.h_output_ring_view = np.ctypeslib.as_array(
            ctypes.cast(h_output_ring_ptr.value, ctypes.POINTER(ctypes.c_byte)),
            shape=(output_ring_size,)
        ).view(output_dtype).reshape(self.ring_size)
        self.h_input_head_view = np.ctypeslib.as_array(
            ctypes.cast(h_input_head_ptr.value, ctypes.POINTER(ctypes.c_int32)),
            shape=(1,)
        )
        self.h_input_tail_view = np.ctypeslib.as_array(
            ctypes.cast(h_input_tail_ptr.value, ctypes.POINTER(ctypes.c_int32)),
            shape=(1,)
        )
        self.h_output_head_view = np.ctypeslib.as_array(
            ctypes.cast(h_output_head_ptr.value, ctypes.POINTER(ctypes.c_int32)),
            shape=(1,)
        )
        self.h_output_tail_view = np.ctypeslib.as_array(
            ctypes.cast(h_output_tail_ptr.value, ctypes.POINTER(ctypes.c_int32)),
            shape=(1,)
        )
        self.h_control_flag_view = np.ctypeslib.as_array(
            ctypes.cast(h_control_flag_ptr.value, ctypes.POINTER(ctypes.c_int32)),
            shape=(1,)
        )

        # Store host pointers for cleanup
        self.h_input_ring_ptr = h_input_ring_ptr
        self.h_output_ring_ptr = h_output_ring_ptr
        self.h_input_head_ptr = h_input_head_ptr
        self.h_input_tail_ptr = h_input_tail_ptr
        self.h_output_head_ptr = h_output_head_ptr
        self.h_output_tail_ptr = h_output_tail_ptr
        self.h_control_flag_ptr = h_control_flag_ptr
        self.cudart = cudart  # Store for cleanup

        # Initialize values
        self.h_input_head_view[0] = 0
        self.h_input_tail_view[0] = 0
        self.h_output_head_view[0] = 0
        self.h_output_tail_view[0] = 0
        self.h_control_flag_view[0] = 0  # CONTROL_RUNNING

        # Create CuPy array views for compatibility with existing code
        # These point to the device-side pointers of the mapped memory
        self.d_input_ring = cp.ndarray(self.ring_size, dtype=input_dtype,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_ring_ptr, input_ring_size, self), 0))
        self.d_output_ring = cp.ndarray(self.ring_size, dtype=output_dtype,
                                        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_ring_ptr, output_ring_size, self), 0))
        self.d_input_head = cp.ndarray(1, dtype=cp.int32,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_head_ptr, 4, self), 0))
        self.d_input_tail = cp.ndarray(1, dtype=cp.int32,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_tail_ptr, 4, self), 0))
        self.d_output_head = cp.ndarray(1, dtype=cp.int32,
                                        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_head_ptr, 4, self), 0))
        self.d_output_tail = cp.ndarray(1, dtype=cp.int32,
                                        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_tail_ptr, 4, self), 0))
        self.d_control_flag = cp.ndarray(1, dtype=cp.int32,
                                         memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_control_flag_ptr, 4, self), 0))

    def _allocate_persistent_kernel_data(self):
        """Allocate CSR data and output buffers for persistent kernel."""
        # Build CSR format for the graph
        nodes = self.nodes
        edges = self.edges
        n = len(nodes)

        print(f"[ALLOC] Building CSR for {n} nodes, {len(edges)} edges", flush=True)

        # Create node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        print(f"[ALLOC] Node mapping created", flush=True)

        # Build CSR format - use adjacency list for efficiency
        print(f"[ALLOC] Building adjacency list...", flush=True)
        adjacency = [[] for _ in range(n)]
        for (u, v) in edges:
            if u in node_to_idx and v in node_to_idx:
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                adjacency[idx_u].append((idx_v, 1))
                adjacency[idx_v].append((idx_u, 1))

        print(f"[ALLOC] Adjacency list built, now building CSR...", flush=True)
        csr_row_ptr = [0] * (n + 1)
        csr_col_ind = []
        csr_J_vals = []

        for i in range(n):
            # Sort neighbors for deterministic ordering
            adjacency[i].sort()
            csr_row_ptr[i + 1] = csr_row_ptr[i] + len(adjacency[i])
            for neighbor_idx, weight in adjacency[i]:
                csr_col_ind.append(neighbor_idx)
                csr_J_vals.append(np.int8(weight))  # Store as-is (±1)

        # Upload to device
        self.d_csr_row_ptr = cp.asarray(csr_row_ptr, dtype=cp.int32)
        self.d_csr_col_ind = cp.asarray(csr_col_ind, dtype=cp.int32)
        self.d_csr_J_vals = cp.asarray(csr_J_vals, dtype=cp.int8)

        # Beta schedule (temperature schedule for SA)
        num_betas = 10
        beta_schedule = np.linspace(0.1, 10.0, num_betas, dtype=np.float32)
        self.d_beta_schedule = cp.asarray(beta_schedule, dtype=cp.float32)

        # Output buffers for samples and energies
        max_reads = 1000
        max_N = n
        packed_size = (max_N + 7) // 8
        self.d_output_samples = cp.zeros((max_reads, packed_size), dtype=cp.int8)
        self.d_output_energies = cp.zeros(max_reads, dtype=cp.float32)

        # Delta energy workspace
        workspace_capacity = max_N
        self.d_delta_energy_workspace = cp.zeros(workspace_capacity, dtype=cp.int8)

        self.logger.info(f"Persistent kernel data allocated: {n} nodes, {len(csr_col_ind)} edges")

    def _allocate_buffer_pool(self):
        """Pre-allocate a pool of GPU buffers to avoid allocation during kernel execution."""
        self.logger.info(f"Allocating buffer pool with {self.buffer_pool_size} buffer sets...")

        # Determine maximum buffer sizes based on the graph
        max_N = len(self.nodes)
        max_reads = 1000  # Maximum reads per job
        max_betas = 20  # Maximum beta schedule length
        packed_size = (max_N + 7) // 8

        self.logger.info(f"Buffer pool: max_N={max_N}, max_reads={max_reads}, max_betas={max_betas}, packed_size={packed_size}")

        # Pre-allocate buffer sets (do NOT use transfer stream context - allocate on default stream)
        for i in range(self.buffer_pool_size):
            self.logger.info(f"Allocating buffer set {i}/{self.buffer_pool_size}...")
            beta_schedule = cp.zeros(max_betas, dtype=cp.float32)
            output_samples = cp.zeros((max_reads, packed_size), dtype=cp.int8)
            output_energies = cp.zeros(max_reads, dtype=cp.float32)

            self.buffer_sets.append((beta_schedule, output_samples, output_energies))
            self.available_buffers.put(i)
            self.logger.info(f"Buffer set {i} allocated")

        self.logger.info(f"Buffer pool allocated: {self.buffer_pool_size} sets of (beta[{max_betas}], samples[{max_reads}x{packed_size}], energies[{max_reads}])")

    def _launch_persistent_kernel(self):
        """Launch the persistent kernel on the GPU."""
        try:
            # Launch kernel with 1 block, max threads per SM
            # Thread 0: coordinator, Threads 1+: process reads in parallel
            block_size = min(self.max_threads_per_job, 1024)

            # Create CuPy memory pointers from the raw device pointers
            # This is necessary for CuPy's RawKernel to accept them
            # IMPORTANT: Must match CUDA struct layout with padding for pointer alignment
            input_dtype = np.dtype([
                ('job_id', np.int32),
                ('num_reads', np.int32),
                ('num_sweeps', np.int32),
                ('num_sweeps_per_beta', np.int32),
                ('seed', np.uint32),
                ('csr_row_ptr_offset', np.int32),
                ('csr_col_ind_offset', np.int32),
                ('N', np.int32),
                ('beta_schedule', np.uint64),      # 8-byte pointer, aligned to 8
                ('num_betas', np.int32),
                ('_padding', np.int32),             # 4 bytes padding for pointer alignment
                ('output_samples', np.uint64),      # 8-byte pointer, aligned to 8
                ('output_energies', np.uint64),     # 8-byte pointer, aligned to 8
            ])
            output_dtype = np.dtype([
                ('job_id', np.int32),
                ('num_reads_done', np.int32),
                ('min_energy', np.float32),
                ('avg_energy', np.float32),
            ])

            input_ring_size = self.ring_size * input_dtype.itemsize
            output_ring_size = self.ring_size * output_dtype.itemsize

            d_input_ring = cp.ndarray(self.ring_size, dtype=input_dtype,
                                      memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_ring_ptr, input_ring_size, self), 0))
            d_output_ring = cp.ndarray(self.ring_size, dtype=output_dtype,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_ring_ptr, output_ring_size, self), 0))
            d_input_head = cp.ndarray(1, dtype=cp.int32,
                                      memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_head_ptr, 4, self), 0))
            d_input_tail = cp.ndarray(1, dtype=cp.int32,
                                      memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_tail_ptr, 4, self), 0))
            d_output_head = cp.ndarray(1, dtype=cp.int32,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_head_ptr, 4, self), 0))
            d_output_tail = cp.ndarray(1, dtype=cp.int32,
                                       memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_output_tail_ptr, 4, self), 0))
            d_control_flag = cp.ndarray(1, dtype=cp.int32,
                                        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_control_flag_ptr, 4, self), 0))

            # Pass CuPy arrays to the kernel
            self._kernel_persistent(
                (1,), (block_size,),  # 1 block, multiple threads
                (
                    d_input_ring,
                    self.ring_size,
                    d_input_head,
                    d_input_tail,
                    d_output_ring,
                    self.ring_size,
                    d_output_head,
                    d_output_tail,
                    d_control_flag,
                    self.d_csr_row_ptr,
                    self.d_csr_col_ind,
                    self.d_csr_J_vals,
                ),
                stream=self._stream
            )

            self.logger.info(f"Persistent kernel launched on GPU with {block_size} threads")

            # Give kernel a moment to start and print initial messages
            import time
            time.sleep(0.1)

            # Flush CUDA printf buffer by querying stream (non-blocking)
            try:
                query_result = cp.cuda.runtime.streamQuery(self._stream.ptr)
                if query_result == 0:
                    self.logger.info("Kernel completed immediately (unexpected)")
                elif query_result == 600:  # cudaErrorNotReady
                    self.logger.info("Kernel is running")
                else:
                    self.logger.warning(f"Stream query returned: {query_result}")
            except Exception as e:
                self.logger.info(f"Stream query: {e}")
        except Exception as e:
            self.logger.error(f"Failed to launch persistent kernel: {e}")
            import traceback
            traceback.print_exc()
            raise

    def enqueue_job(self, job_id: int, num_reads: int, num_sweeps: int,
                   num_sweeps_per_beta: int, seed: int,
                   csr_row_ptr_offset: int, csr_col_ind_offset: int, N: int,
                   beta_schedule: cp.ndarray, num_betas: int,
                   output_samples: cp.ndarray, output_energies: cp.ndarray):
        """Enqueue a job to the input ring buffer using zero-copy mapped memory."""
        # Use CPU-side tracking to avoid GPU synchronization
        tail = self.cpu_input_tail
        slot = tail % self.ring_size

        self.logger.info(f"[ENQUEUE] job_id={job_id}, tail={tail}, slot={slot}")

        # Write directly to the mapped host memory (no memcpy needed!)
        # The GPU kernel will see this data via the mapped device pointer
        self.h_input_ring_view[slot]['job_id'] = job_id
        self.h_input_ring_view[slot]['num_reads'] = num_reads
        self.h_input_ring_view[slot]['num_sweeps'] = num_sweeps
        self.h_input_ring_view[slot]['num_sweeps_per_beta'] = num_sweeps_per_beta
        self.h_input_ring_view[slot]['seed'] = seed
        self.h_input_ring_view[slot]['csr_row_ptr_offset'] = csr_row_ptr_offset
        self.h_input_ring_view[slot]['csr_col_ind_offset'] = csr_col_ind_offset
        self.h_input_ring_view[slot]['N'] = N
        self.h_input_ring_view[slot]['beta_schedule'] = beta_schedule.data.ptr
        self.h_input_ring_view[slot]['num_betas'] = num_betas
        self.h_input_ring_view[slot]['_padding'] = 0  # Padding for pointer alignment
        self.h_input_ring_view[slot]['output_samples'] = output_samples.data.ptr
        self.h_input_ring_view[slot]['output_energies'] = output_energies.data.ptr

        # Increment tail (both CPU and host memory)
        self.cpu_input_tail = (tail + 1) % (2**31)  # Prevent overflow

        # Update the tail in host memory (both via numpy view and ctypes)
        self.h_input_tail_view[0] = self.cpu_input_tail

        # Also update via ctypes for consistency
        import ctypes
        tail_ptr = ctypes.cast(self.h_input_tail_ptr.value, ctypes.POINTER(ctypes.c_int32))
        tail_ptr[0] = self.cpu_input_tail

        print(f"[ENQUEUE] Updated tail to {self.cpu_input_tail}, h_input_tail_view[0]={self.h_input_tail_view[0]}", flush=True)
        self.logger.info(f"[ENQUEUE] Updated tail to {self.cpu_input_tail}, h_input_tail_view[0]={self.h_input_tail_view[0]}")

        self.logger.debug(f"Enqueued job_id={job_id}, num_reads={num_reads}, N={N}, slot={slot}, tail={tail+1}")

    def try_dequeue_result(self) -> Optional[Dict]:
        """Try to dequeue a result from output ring buffer using zero-copy mapped memory."""
        with self.lock:
            # Read the current tail from host memory (written by GPU kernel)
            tail = int(self.h_output_tail_view[0])

            # Update CPU-side tracking
            self.cpu_output_tail = tail

            head = self.cpu_output_head

            self.logger.info(f"[TRY_DEQUEUE] head={head}, tail={tail}")

            if head == tail:
                return None

            slot = head % self.ring_size

            # Read directly from mapped host memory (no memcpy needed!)
            result = self.h_output_ring_view[slot]

            self.logger.info(f"[TRY_DEQUEUE] Dequeued result from slot {slot}: job_id={result['job_id']}")

            # Increment head (both CPU and host memory)
            self.cpu_output_head = (head + 1) % (2**31)  # Prevent overflow
            self.h_output_head_view[0] = self.cpu_output_head

            return {
                'job_id': int(result['job_id']),
                'num_reads_done': int(result['num_reads_done']),
                'min_energy': float(result['min_energy']),
                'avg_energy': float(result['avg_energy']),
            }

    def get_queue_depth(self) -> Tuple[int, int]:
        """Get current input and output queue depths (CPU-side tracking)."""
        # No lock needed - we're only reading CPU-side variables
        # Track queue depth on CPU side to avoid GPU synchronization
        # which would block while the persistent kernel is running
        input_depth = (self.cpu_input_tail - self.cpu_input_head) % self.ring_size
        output_depth = (self.cpu_output_tail - self.cpu_output_head) % self.ring_size
        return input_depth, output_depth

    def set_control_flag(self, control_value: int):
        """Set control flag (CONTROL_RUNNING=0, CONTROL_STOP=1, CONTROL_DRAIN=2)."""
        self.h_control_flag_view[0] = control_value
        self.control_flag_value = control_value
        self.logger.debug(f"Control flag set to {control_value}")

    def stop(self, drain: bool = True):
        """Stop the persistent kernel."""
        if drain:
            self.set_control_flag(2)  # CONTROL_DRAIN
            self.logger.info("Draining kernel queue...")
        else:
            self.set_control_flag(1)  # CONTROL_STOP
            self.logger.info("Stopping kernel immediately...")

        self._stream.synchronize()
        self.logger.info("Kernel stopped")

    def sample_ising_async(self, jobs: List[IsingJob]) -> List[int]:
        """
        Asynchronously enqueue Ising jobs to GPU.

        Args:
            jobs: List of IsingJob objects to process

        Returns:
            List of job_ids assigned to the jobs
        """
        print(f"[ASYNC] sample_ising_async called with {len(jobs)} jobs", flush=True)
        self.logger.info(f"[ASYNC] sample_ising_async called with {len(jobs)} jobs")
        job_ids = []
        nodes = self.nodes
        n = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        for job in jobs:
            print(f"[ASYNC] Processing job: num_reads={job.num_reads}, num_sweeps={job.num_sweeps}", flush=True)
            self.logger.info(f"[ASYNC] Processing job: num_reads={job.num_reads}, num_sweeps={job.num_sweeps}")

            # Build CSR for this specific job
            print(f"[ASYNC] Building adjacency list for {len(job.J)} edges...", flush=True)
            adjacency = [[] for _ in range(n)]
            for (u, v), Jij in job.J.items():
                if u in node_to_idx and v in node_to_idx:
                    idx_u = node_to_idx[u]
                    idx_v = node_to_idx[v]
                    adjacency[idx_u].append((idx_v, int(Jij)))
                    adjacency[idx_v].append((idx_u, int(Jij)))
            print(f"[ASYNC] Adjacency list built", flush=True)

            # Sort adjacency lists for deterministic ordering
            for i in range(n):
                adjacency[i].sort()

            # Build CSR
            csr_row_ptr = np.zeros(n + 1, dtype=np.int32)
            csr_col_ind_list = []
            csr_J_vals_list = []

            for i in range(n):
                csr_row_ptr[i + 1] = csr_row_ptr[i] + len(adjacency[i])
                for j, Jij in adjacency[i]:
                    csr_col_ind_list.append(j)
                    # Store J values as-is (should be ±1 for Ising problems)
                    csr_J_vals_list.append(Jij)

            csr_col_ind = np.array(csr_col_ind_list, dtype=np.int32)
            csr_J_vals = np.array(csr_J_vals_list, dtype=np.int8)

            # Debug: verify J values before upload
            print(f"[ASYNC] csr_J_vals_list (first 10): {csr_J_vals_list[:10]}", flush=True)
            print(f"[ASYNC] csr_J_vals dtype: {csr_J_vals.dtype}, first 10: {csr_J_vals[:10]}", flush=True)
            print(f"[ASYNC] csr_J_vals (min={csr_J_vals.min()}, max={csr_J_vals.max()})", flush=True)

            # Copy to device and update global CSR (since we process one job at a time)
            # NOTE: Don't synchronize here - the kernel is running on a separate stream
            # and synchronizing would cause a deadlock. The kernel will see the data
            # when it's ready via memory coherency.
            print(f"[ASYNC] Uploading CSR to device...", flush=True)
            self.d_csr_row_ptr = cp.asarray(csr_row_ptr, dtype=cp.int32)
            self.d_csr_col_ind = cp.asarray(csr_col_ind, dtype=cp.int32)
            self.d_csr_J_vals = cp.asarray(csr_J_vals, dtype=cp.int8)
            print(f"[ASYNC] CSR uploaded", flush=True)

            # Wait for backpressure: pause if queue too full
            print(f"[ASYNC] Checking queue depth...", flush=True)
            while True:
                input_depth, _ = self.get_queue_depth()
                print(f"[ASYNC] Queue depth: {input_depth}, ring_size: {self.ring_size}", flush=True)
                if input_depth < 2 * self.ring_size:
                    break
                time.sleep(0.001)  # 1ms backoff

            print(f"[ASYNC] Acquiring lock...", flush=True)
            with self.lock:
                job_id = self.job_counter
                self.job_counter += 1
                job.job_id = job_id

                # Generate beta schedule if not provided
                if job.beta_schedule is None:
                    beta_range = _default_ising_beta_range(job.h, job.J)
                    job.beta_schedule = np.linspace(beta_range[0], beta_range[1], 10, dtype=np.float32)

                # Acquire buffers from the pre-allocated pool
                # This avoids GPU memory allocation while the persistent kernel is running
                buffer_idx = self.available_buffers.get()  # Blocks if pool is empty
                d_beta_schedule, d_output_samples, d_output_energies = self.buffer_sets[buffer_idx]

                # Copy beta schedule to the buffer (use set() to avoid cp.asarray allocation)
                packed_size = (n + 7) // 8
                beta_np = np.asarray(job.beta_schedule, dtype=np.float32)
                d_beta_schedule[:len(beta_np)].set(beta_np)

                # Store for later retrieval
                self.job_buffers[job_id] = {
                    'buffer_idx': buffer_idx,
                    'beta_schedule': d_beta_schedule,
                    'output_samples': d_output_samples,
                    'output_energies': d_output_energies,
                    'num_reads': job.num_reads,
                    'packed_size': packed_size,
                }

                # Enqueue job to persistent kernel
                print(f"[ASYNC] About to call enqueue_job for job_id={job_id}", flush=True)
                self.enqueue_job(
                    job_id=job_id,
                    num_reads=job.num_reads,
                    num_sweeps=job.num_sweeps,
                    num_sweeps_per_beta=job.num_sweeps_per_beta,
                    seed=job.seed if job.seed is not None else np.random.randint(0, 2**31),
                    csr_row_ptr_offset=0,
                    csr_col_ind_offset=0,
                    N=n,
                    beta_schedule=d_beta_schedule,
                    num_betas=len(job.beta_schedule),
                    output_samples=d_output_samples,
                    output_energies=d_output_energies,
                )
                print(f"[ASYNC] enqueue_job returned for job_id={job_id}", flush=True)

                job_ids.append(job_id)

        return job_ids

    def dequeue_results(self, timeout: float = 0.1) -> List[Tuple[int, dimod.SampleSet]]:
        """
        Dequeue results from GPU and convert to SampleSets.

        Args:
            timeout: Maximum time to wait for results (seconds)

        Returns:
            List of (job_id, sampleset) tuples
        """
        results = []
        start_time = time.time()
        self.logger.info(f"[DEQUEUE_RESULTS] Waiting up to {timeout}s for results")

        while time.time() - start_time < timeout:
            print(f"[DEQUEUE_RESULTS] Calling try_dequeue_result...", flush=True)
            result = self.try_dequeue_result()
            print(f"[DEQUEUE_RESULTS] try_dequeue_result returned: {result is not None}", flush=True)
            if result is None:
                time.sleep(0.001)  # 1ms backoff
                continue

            job_id = result['job_id']
            num_reads = result['num_reads_done']
            min_energy = result['min_energy']

            print(f"[DEQUEUE_RESULTS] Processing result for job_id={job_id}", flush=True)

            # Retrieve job buffers
            if job_id not in self.job_buffers:
                self.logger.warning(f"No buffers for job_id {job_id}")
                continue

            print(f"[DEQUEUE_RESULTS] Found buffers for job_id={job_id}", flush=True)

            buffers = self.job_buffers.pop(job_id)
            buffer_idx = buffers['buffer_idx']
            output_samples = buffers['output_samples']
            output_energies = buffers['output_energies']
            packed_size = buffers['packed_size']
            nodes = self.nodes
            n = len(nodes)

            print(f"[DEQUEUE_RESULTS] Copying results from GPU...", flush=True)

            # Copy results from GPU
            # NOTE: For now, just use the GPU arrays directly to avoid synchronization issues
            # with the persistent kernel
            print(f"[DEQUEUE_RESULTS] Using GPU arrays directly (no copy)", flush=True)
            samples_gpu = output_samples
            energies_gpu = output_energies

            print(f"[DEQUEUE_RESULTS] Results ready", flush=True)

            # Release buffer back to pool
            print(f"[DEQUEUE_RESULTS] Releasing buffer {buffer_idx}", flush=True)
            self.available_buffers.put(buffer_idx)

            # Unpack samples from bit-packed format
            print(f"[DEQUEUE_RESULTS] Unpacking {num_reads} samples...", flush=True)
            samples = []
            for read_idx in range(num_reads):
                print(f"[DEQUEUE_RESULTS] Unpacking read {read_idx}...", flush=True)
                # For now, create a dummy sample to test the rest of the code
                # TODO: Fix GPU array access issue
                sample = {node: 1 for node in nodes}
                samples.append(sample)
            print(f"[DEQUEUE_RESULTS] Unpacking complete", flush=True)

            # Create sampleset
            # For now, use dummy energies since we can't access GPU arrays
            # TODO: Fix GPU array access
            dummy_energies = np.zeros(num_reads, dtype=np.float32)
            print(f"[DEQUEUE_RESULTS] Creating sampleset...", flush=True)
            sampleset = dimod.SampleSet.from_samples(samples, 'SPIN', dummy_energies)
            print(f"[DEQUEUE_RESULTS] Sampleset created", flush=True)
            results.append((job_id, sampleset))

        return results

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
        Synchronous sample from Ising model.

        Enqueues jobs asynchronously and blocks until results are available.
        """
        print(f"[SAMPLE_ISING] Called with num_reads={num_reads}, num_sweeps={num_sweeps}", flush=True)
        self.logger.info(f"[SAMPLE_ISING] Called with num_reads={num_reads}, num_sweeps={num_sweeps}")
        # Handle both single problem and batched problems
        if isinstance(h, list):
            print(f"[SAMPLE_ISING] Batched mode with {len(h)} problems", flush=True)
            self.logger.info(f"[SAMPLE_ISING] Batched mode with {len(h)} problems")
            # Batched mode
            jobs = [
                IsingJob(h=h_prob, J=J_prob, num_reads=num_reads,
                        num_sweeps=num_sweeps, num_sweeps_per_beta=num_sweeps_per_beta, seed=seed)
                for h_prob, J_prob in zip(h, J)
            ]
            job_ids = self.sample_ising_async(jobs)

            # Block until all results are available
            samplesets = {}
            timeout = time.time() + 300  # 5 minute timeout
            while len(samplesets) < len(job_ids) and time.time() < timeout:
                results = self.dequeue_results(timeout=0.1)
                for job_id, sampleset in results:
                    samplesets[job_id] = sampleset

            return [samplesets.get(jid) for jid in job_ids]
        else:
            # Single problem mode
            print(f"[SAMPLE_ISING] Creating IsingJob...", flush=True)
            job = IsingJob(h=h, J=J, num_reads=num_reads,
                          num_sweeps=num_sweeps, num_sweeps_per_beta=num_sweeps_per_beta, seed=seed)
            print(f"[SAMPLE_ISING] Calling sample_ising_async...", flush=True)
            job_ids = self.sample_ising_async([job])
            print(f"[SAMPLE_ISING] Got job_ids: {job_ids}", flush=True)

            # Block until result is available
            print(f"[SAMPLE_ISING] Waiting for results...", flush=True)
            timeout = time.time() + 300  # 5 minute timeout
            while time.time() < timeout:
                print(f"[SAMPLE_ISING] Calling dequeue_results...", flush=True)
                results = self.dequeue_results(timeout=0.1)
                print(f"[SAMPLE_ISING] Got {len(results)} results", flush=True)
                for job_id, sampleset in results:
                    if job_id == job_ids[0]:
                        print(f"[SAMPLE_ISING] Found result for job {job_id}", flush=True)
                        return sampleset

            raise TimeoutError(f"Job {job_ids[0]} did not complete within timeout")

    def _sample_ising_single(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        beta_schedule: Optional[np.ndarray],
        seed: Optional[int]
    ) -> dimod.SampleSet:
        """
        Sample a single Ising problem using the persistent kernel.
        """
        n = len(self.nodes)
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Compute beta schedule
        if beta_schedule is None:
            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)

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

        # Build CSR for this problem
        adjacency = [[] for _ in range(n)]
        for (u, v), Jij in J.items():
            if u in node_to_idx and v in node_to_idx:
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                adjacency[idx_u].append((idx_v, int(Jij)))
                adjacency[idx_v].append((idx_u, int(Jij)))

        # Sort adjacency lists for deterministic ordering
        for i in range(n):
            adjacency[i].sort()

        # Build CSR
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

        # Copy to GPU
        d_csr_row_ptr = cp.asarray(csr_row_ptr)
        d_csr_col_ind = cp.asarray(csr_col_ind)
        d_csr_J_vals = cp.asarray(csr_J_vals)
        d_beta_schedule = cp.asarray(beta_schedule)

        # Allocate output arrays
        packed_size = (n + 7) >> 3
        d_output_samples = cp.zeros((num_reads, packed_size), dtype=cp.int8)
        d_output_energies = cp.zeros(num_reads, dtype=cp.float32)

        # Allocate workspace
        d_delta_energy_workspace = cp.zeros((num_reads, n), dtype=cp.int8)

        # Launch kernel
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create offset arrays for single problem
        row_ptr_offsets = np.array([0, len(csr_row_ptr) - 1], dtype=np.int32)
        col_ind_offsets = np.array([0, len(csr_col_ind)], dtype=np.int32)

        d_row_ptr_offsets = cp.asarray(row_ptr_offsets)
        d_col_ind_offsets = cp.asarray(col_ind_offsets)

        block_size = min(num_reads, 256)
        grid_size = (num_reads + block_size - 1) // block_size

        self._kernel_sa(
            (grid_size,), (block_size,),
            (d_csr_row_ptr, d_csr_col_ind, d_csr_J_vals,
             d_row_ptr_offsets, d_col_ind_offsets,
             1, d_beta_schedule,
             n, len(beta_schedule), num_sweeps_per_beta, num_reads, seed,
             d_output_samples, d_output_energies, d_delta_energy_workspace,
             num_reads * n)
        )

        # Copy results back
        samples_packed = cp.asnumpy(d_output_samples)
        energies = cp.asnumpy(d_output_energies)

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

        return dimod.SampleSet.from_samples(samples, 'SPIN', energies)


# ============================================================================
# Level 2: Mock Kernel for Testing CudaSASampler
# ============================================================================

class CudaKernelMock:
    """
    Mock CUDA kernel for testing CudaSASampler logic without GPU.

    Simulates async job processing with configurable delays.
    Implements same interface as CudaKernel for seamless testing.
    """

    def __init__(self, processing_delay: float = 0.01):
        """
        Initialize mock kernel.

        Args:
            processing_delay: Simulated processing time per job (seconds)
        """
        self.processing_delay = processing_delay
        self.job_queue = []
        self.result_queue = []
        self.kernel_state = 1  # STATE_IDLE
        self.lock = threading.Lock()
        self.worker_thread = None
        self.running = False

    def start(self):
        """Start background worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="CudaKernelMock-Worker"
        )
        self.worker_thread.start()

    def stop(self, drain: bool = True):
        """
        Stop background worker.

        Args:
            drain: If True, wait for queue to empty before stopping
        """
        if drain:
            # Wait for queue to empty
            deadline = time.time() + 30.0  # 30 second timeout
            while len(self.job_queue) > 0 and time.time() < deadline:
                time.sleep(0.001)

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

    def enqueue_job(
        self,
        job_id: int,
        h: np.ndarray,
        J: np.ndarray,
        num_reads: int,
        num_betas: int,
        num_sweeps_per_beta: int,
        beta_schedule: Optional[np.ndarray] = None,
        N: int = 0,
        **kwargs
    ) -> None:
        """
        Enqueue a mock job.

        Args:
            job_id: Unique job identifier
            h: Linear bias array
            J: Coupling values array
            num_reads: Number of samples to generate
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature
            beta_schedule: Temperature schedule
            N: Number of variables
            **kwargs: Additional arguments (ignored)
        """
        with self.lock:
            self.job_queue.append({
                'job_id': job_id,
                'h': h,
                'J': J,
                'num_reads': num_reads,
                'num_betas': num_betas,
                'num_sweeps_per_beta': num_sweeps_per_beta,
                'beta_schedule': beta_schedule,
                'N': N if N > 0 else len(h),
            })

    def signal_batch_ready(self):
        """
        Signal that batch is ready to process.

        For mock kernel, this is a no-op since jobs are processed immediately.
        """
        pass

    def try_dequeue_result(self) -> Optional[Dict]:
        """
        Try to dequeue a mock result (non-blocking).

        Returns:
            Result dict or None if queue empty
        """
        with self.lock:
            if len(self.result_queue) == 0:
                return None
            return self.result_queue.pop(0)

    def get_kernel_state(self) -> int:
        """
        Get kernel state.

        Returns:
            0 = STATE_RUNNING, 1 = STATE_IDLE
        """
        with self.lock:
            # RUNNING if there are jobs in queue or results in queue
            if len(self.job_queue) > 0 or len(self.result_queue) > 0:
                return 0  # STATE_RUNNING
            return 1  # STATE_IDLE

    def get_samples(self, result: Dict) -> np.ndarray:
        """
        Extract samples from result.

        Args:
            result: Result dict from try_dequeue_result()

        Returns:
            Samples array of shape (num_reads, N)
        """
        return result['samples']

    def get_energies(self, result: Dict) -> np.ndarray:
        """
        Extract energies from result.

        Args:
            result: Result dict from try_dequeue_result()

        Returns:
            Energies array of shape (num_reads,)
        """
        return result['energies']

    def _worker_loop(self):
        """Background worker that processes jobs."""
        while self.running:
            job = None
            with self.lock:
                if len(self.job_queue) > 0:
                    job = self.job_queue.pop(0)

            if job is None:
                time.sleep(0.001)
                continue

            # Simulate processing
            time.sleep(self.processing_delay)

            # Generate mock result
            N = job['N']
            num_reads = job['num_reads']

            # Generate random samples: {-1, +1}
            samples = np.random.randint(0, 2, size=(num_reads, N), dtype=np.int8)
            samples = samples * 2 - 1  # Convert to {-1, +1}

            # Generate random energies (negative for this problem)
            energies = np.random.randn(num_reads).astype(np.float32) * 100 - 14000

            result = {
                'job_id': job['job_id'],
                'min_energy': float(energies.min()),
                'avg_energy': float(energies.mean()),
                'samples': samples,
                'energies': energies,
                'samples_size': samples.nbytes,
                'energies_size': energies.nbytes,
                'num_reads': num_reads,
                'N': N,
            }

            with self.lock:
                self.result_queue.append(result)


# ============================================================================
# Level 2: Kernel Adapters and CudaSASamplerAsync - High-Level Async Sampler API
# ============================================================================

class CudaKernelAdapter:
    """
    Adapter to make CudaKernelRealSA compatible with CudaKernelMock interface.

    Converts array-based h/J to dict-based format expected by CudaKernelRealSA.
    """

    def __init__(self, kernel):
        """
        Initialize adapter.

        Args:
            kernel: CudaKernelRealSA instance
        """
        self.kernel = kernel

    def enqueue_job(
        self,
        job_id: int,
        h: np.ndarray,
        J: np.ndarray,
        num_reads: int,
        num_betas: int,
        num_sweeps_per_beta: int,
        beta_schedule: Optional[np.ndarray] = None,
        N: int = 0,
        edges: Optional[List[Tuple[int, int]]] = None,
        **kwargs
    ) -> None:
        """
        Enqueue a job, converting arrays to dicts.

        Args:
            job_id: Unique job identifier
            h: Linear bias array
            J: Coupling values array (indexed by edges if edges provided, else upper triangular)
            num_reads: Number of samples to generate
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature
            beta_schedule: Temperature schedule (ignored for real kernel)
            N: Number of variables
            edges: List of (i, j) edge tuples (if provided, J is indexed by edges)
            **kwargs: Additional arguments (ignored)
        """
        if N == 0:
            N = len(h)

        # Convert h array to dict
        h_dict = {}
        for i, val in enumerate(h):
            if val != 0:
                h_dict[i] = float(val)

        # Convert J array to dict
        J_dict = {}

        if edges is not None:
            # J is indexed by edges (for production topology)
            for idx, (i, j) in enumerate(edges):
                if idx < len(J) and J[idx] != 0:
                    J_dict[(i, j)] = float(J[idx])
        else:
            # Assume J is flattened upper triangular (for small problems)
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if idx < len(J) and J[idx] != 0:
                        J_dict[(i, j)] = float(J[idx])
                    idx += 1

        # Enqueue to real kernel
        self.kernel.enqueue_job(
            job_id=job_id,
            h=h_dict,
            J=J_dict,
            num_reads=num_reads,
            num_betas=num_betas,
            num_sweeps_per_beta=num_sweeps_per_beta,
            N=N
        )

    def signal_batch_ready(self):
        """Signal that batch is ready to process."""
        self.kernel.signal_batch_ready()

    def get_num_sms(self) -> int:
        """Get number of streaming multiprocessors (SMs) available."""
        return self.kernel.num_blocks

    def try_dequeue_result(self) -> Optional[Dict]:
        """Try to dequeue a result."""
        return self.kernel.try_dequeue_result()

    def get_kernel_state(self) -> int:
        """Get kernel state."""
        return self.kernel.get_kernel_state()

    def get_samples(self, result: Dict) -> np.ndarray:
        """Extract samples from result."""
        return self.kernel.get_samples(result)

    def get_energies(self, result: Dict) -> np.ndarray:
        """Extract energies from result."""
        return self.kernel.get_energies(result)

    def stop_immediate(self) -> None:
        """Stop the kernel immediately."""
        self.kernel.stop_immediate()

    def stop_drain(self) -> None:
        """Stop the kernel after draining queue."""
        self.kernel.stop_drain()

    def stop(self, drain: bool = True) -> None:
        """Stop the kernel (deprecated - use stop_immediate or stop_drain)."""
        self.kernel.stop(drain=drain)


class CudaSASamplerAsync:
    """
    High-level async Ising sampler with dimod-compatible API.

    Wraps CudaKernel or CudaKernelMock and provides:
    - Async job submission (sample_ising_async)
    - Result collection with ordering (collect_samples)
    - Synchronous wrapper (sample_ising)
    - Job ordering guarantees (critical for blockchain)
    - Timeout handling
    - Dimod SampleSet conversion
    """

    def __init__(self, kernel):
        """
        Initialize sampler with kernel.

        Args:
            kernel: CudaKernel or CudaKernelMock instance
        """
        self.kernel = kernel
        self.next_job_id = 0
        self.pending_jobs = {}  # job_id -> metadata
        self.completed_jobs = {}  # job_id -> SampleSet
        self.lock = threading.Lock()

        # Start kernel if mock (real kernel is already running)
        if hasattr(kernel, 'start'):
            kernel.start()

    def sample_ising_async(
        self,
        h_list: List[np.ndarray],
        J_list: List[np.ndarray],
        num_reads: int = 100,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 100,
        beta_schedule: Optional[np.ndarray] = None,
        edges: Optional[List[Tuple[int, int]]] = None
    ) -> List[int]:
        """
        Submit multiple Ising models for sampling (non-blocking).

        Args:
            h_list: List of linear bias arrays
            J_list: List of coupling arrays
            num_reads: Number of samples per model
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature
            beta_schedule: Temperature schedule (auto-generated if None)
            edges: List of (i, j) edge tuples (if provided, J is indexed by edges)

        Returns:
            List of job_ids in submission order
        """
        assert len(h_list) == len(J_list), "h_list and J_list must have same length"

        if beta_schedule is None:
            # Use first problem to compute beta schedule (shared across batch)
            beta_schedule = self._generate_beta_schedule(
                h=h_list[0],
                J=J_list[0],
                edges=edges,
                num_betas=num_betas
            )

        job_ids = []
        with self.lock:
            for h, J in zip(h_list, J_list):
                job_id = self.next_job_id
                self.next_job_id += 1

                # Store metadata for later collection
                self.pending_jobs[job_id] = {
                    'h': h,
                    'J': J,
                    'num_reads': num_reads,
                    'num_betas': num_betas,
                    'submitted_at': time.time()
                }

                # Enqueue to kernel
                self.kernel.enqueue_job(
                    job_id=job_id,
                    h=h,
                    J=J,
                    num_reads=num_reads,
                    num_betas=num_betas,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    beta_schedule=beta_schedule,
                    N=len(h),
                    edges=edges
                )

                job_ids.append(job_id)

            # Signal batch ready AFTER all jobs are enqueued
            self.kernel.signal_batch_ready()

        return job_ids

    def collect_samples(
        self,
        job_ids: Optional[List[int]] = None,
        timeout: float = 10.0
    ) -> List[dimod.SampleSet]:
        """
        Collect completed samples (blocking until all specified jobs complete).

        Args:
            job_ids: Specific jobs to collect (None = all pending)
            timeout: Max wait time in seconds

        Returns:
            List of SampleSets in same order as job_ids

        Raises:
            TimeoutError: If timeout exceeded before all jobs complete
        """
        if job_ids is None:
            with self.lock:
                job_ids = list(self.pending_jobs.keys())

        if len(job_ids) == 0:
            return []

        start_time = time.time()
        remaining_jobs = set(job_ids)

        while len(remaining_jobs) > 0:
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for jobs: {remaining_jobs}"
                )

            # Try to dequeue results
            result = self.kernel.try_dequeue_result()
            if result is None:
                time.sleep(0.0001)  # 100µs backoff
                continue

            job_id = result['job_id']

            # Convert to SampleSet
            with self.lock:
                if job_id not in self.pending_jobs:
                    # Unexpected job (already collected or never submitted)
                    continue

                # Remove from pending before processing
                self.pending_jobs.pop(job_id)

            samples = self.kernel.get_samples(result)
            energies = self.kernel.get_energies(result)

            # Samples are already in SPIN format {-1, +1} as float32
            # Convert to int8 for compatibility
            samples_spin = samples.astype(np.int8)

            # Create SampleSet with SPIN vartype
            sampleset = dimod.SampleSet.from_samples(
                samples_spin,
                vartype='SPIN',
                energy=energies,
                info={
                    'job_id': job_id,
                    'min_energy': result['min_energy'],
                    'avg_energy': result['avg_energy'],
                    'num_reads': len(energies)
                }
            )

            with self.lock:
                self.completed_jobs[job_id] = sampleset

            if job_id in remaining_jobs:
                remaining_jobs.remove(job_id)

        # Return in original order
        result_list = []
        for job_id in job_ids:
            with self.lock:
                if job_id in self.completed_jobs:
                    result_list.append(self.completed_jobs.pop(job_id))
                else:
                    raise RuntimeError(f"Job {job_id} not found in completed jobs")

        return result_list

    def sample_ising(
        self,
        h_list: List[np.ndarray],
        J_list: List[np.ndarray],
        num_reads: int = 100,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 100,
        beta_schedule: Optional[np.ndarray] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        timeout: float = 300.0
    ) -> List[dimod.SampleSet]:
        """
        Synchronous sampling (convenience wrapper).

        Calls sample_ising_async then collect_samples.

        Args:
            h_list: List of linear bias arrays
            J_list: List of coupling arrays
            num_reads: Number of samples per model
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature
            beta_schedule: Temperature schedule (auto-generated if None)
            edges: List of (i, j) edge tuples (if provided, J is indexed by edges)
            timeout: Maximum time to wait for all jobs to complete (seconds, default: 300)

        Returns:
            List of SampleSets in same order as input
        """
        job_ids = self.sample_ising_async(
            h_list, J_list, num_reads, num_betas, num_sweeps_per_beta, beta_schedule, edges
        )
        return self.collect_samples(job_ids, timeout=timeout)

    def get_num_sms(self) -> int:
        """
        Get number of streaming multiprocessors (SMs) available on GPU.

        Returns:
            Number of SMs that can process jobs in parallel
        """
        return self.kernel.get_num_sms()

    def stop_immediate(self):
        """
        Stop the sampler and kernel immediately.

        Does not wait for queued jobs to complete.
        """
        self.kernel.stop_immediate()

    def stop_drain(self):
        """
        Stop the sampler and kernel after draining queue.

        Finishes all queued jobs before exiting.
        """
        self.kernel.stop_drain()

    def stop(self, drain: bool = True):
        """
        Stop the sampler and kernel (deprecated - use stop_immediate or stop_drain).

        Args:
            drain: If True, finish current jobs. If False, immediate shutdown.
        """
        self.kernel.stop(drain=drain)

    def _generate_beta_schedule(
        self,
        h: np.ndarray,
        J: np.ndarray,
        edges: Optional[List[Tuple[int, int]]],
        num_betas: int
    ) -> np.ndarray:
        """
        Generate beta schedule using D-Wave's algorithm.

        Args:
            h: Linear bias array
            J: Coupling array (indexed by edges)
            edges: List of (i, j) edge tuples
            num_betas: Number of temperature steps

        Returns:
            Beta schedule array (geometric progression from hot to cold)
        """
        # Convert arrays to dicts for beta range calculation
        h_dict = {}
        for i, val in enumerate(h):
            if val != 0:
                h_dict[i] = float(val)

        J_dict = {}
        if edges is not None and len(J) > 0:
            for idx, (i, j) in enumerate(edges):
                if idx < len(J) and J[idx] != 0:
                    J_dict[(i, j)] = float(J[idx])

        # Compute beta range using D-Wave's algorithm
        hot_beta, cold_beta = _default_ising_beta_range(h_dict, J_dict)

        # Generate geometric schedule (matching D-Wave/Metal)
        if num_betas == 1:
            return np.array([cold_beta], dtype=np.float32)
        else:
            return np.logspace(
                np.log10(hot_beta),
                np.log10(cold_beta),
                num=num_betas,
                dtype=np.float32
            )


