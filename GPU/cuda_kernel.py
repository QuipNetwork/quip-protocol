"""
CUDA Kernel Interface - Level 1 Implementation

Low-level interface to CUDA test kernel for verification of:
- Job enqueue/dequeue
- Thread memory isolation
- Ring buffer operations
- Control flags (STOP/DRAIN)
- Kernel state tracking (RUNNING/IDLE)
"""

import ctypes
import os
import random
import struct
import threading
import time
import warnings
from collections import defaultdict
from typing import Optional, Dict, Tuple

import cupy as cp
import numpy as np


def _load_cudart():
    """Load CUDA runtime library with fallbacks for versioned names.

    The runtime image only has versioned libraries like libcudart.so.11.0,
    not the unversioned libcudart.so symlink (which is in devel images).
    """
    # Library names to try in order of preference
    lib_names = [
        'libcudart.so',           # Unversioned (works if symlink exists)
        'libcudart.so.11.0',      # CUDA 11.x versioned
        'libcudart.so.11',        # CUDA 11 major version
        'libcudart.so.12.0',      # CUDA 12.x versioned
        'libcudart.so.12',        # CUDA 12 major version
        'cudart64_110.dll',       # Windows CUDA 11.x
        'cudart64_12.dll',        # Windows CUDA 12.x
    ]

    errors = []
    for lib_name in lib_names:
        try:
            return ctypes.CDLL(lib_name)
        except OSError as e:
            errors.append(f"{lib_name}: {e}")
            continue

    # If all failed, raise with details
    raise OSError(
        f"Could not load CUDA runtime library. Tried:\n" +
        "\n".join(f"  - {err}" for err in errors)
    )


def _default_ising_beta_range(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
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


class CudaKernel:
    """
    Low-level CUDA kernel interface.
    Handles compilation, buffer management, and kernel execution.
    """

    def __init__(
        self,
        ring_size: int = 8,
        max_threads_per_job: int = 256,
        max_h_size: int = 5000,
        max_J_size: int = 50000,
        max_N: int = 5000
    ):
        """
        Initialize CUDA kernel and buffers.

        Args:
            ring_size: Size of job/result ring buffers
            max_threads_per_job: Maximum worker threads per job
            max_h_size: Maximum size of h array
            max_J_size: Maximum size of J array
            max_N: Maximum number of variables
        """
        self.ring_size = ring_size
        self.max_threads_per_job = max_threads_per_job
        self.max_h_size = max_h_size
        self.max_J_size = max_J_size
        self.max_N = max_N

        # Query SM count
        self.num_blocks = self._get_sm_count()

        # Compile kernel with fast math optimizations
        # Note: NVRTC uses --optimization-level instead of -O3
        self.module = cp.RawModule(
            code=self._load_kernel_code(),
            options=('--use_fast_math',)
        )
        self.kernel = self.module.get_function('cuda_sa_persistent_test')

        # Compute JobDesc size using native alignment to match CUDA
        # Format: 7i + Q + i + Q + i + 3Q + I (seed)
        self._jobdesc_fmt = 'iiiiiiiQiQiQQQI'
        self._sizeof_jobdesc = struct.calcsize(self._jobdesc_fmt)  # Expected 92 on x86_64

        # OutputSlot: ready(4) + job_id(4) + min_energy(4) + avg_energy(4) +
        #             num_reads(4) + N(4) + samples_offset(4) + energies_offset(4)
        # Total: 8 * 4 = 32 bytes
        self._sizeof_outputslot = 32

        # Allocate input ring buffer as raw byte array (device memory)
        self.d_input_ring_bytes = cp.zeros(ring_size * self._sizeof_jobdesc, dtype=cp.uint8)

        # Output slots allocation deferred until after mapped alloc helpers are defined

        # Host buffer for input ring packing (unused directly, kept for clarity)
        self.h_input_ring = np.zeros(ring_size * self._sizeof_jobdesc, dtype=np.uint8)

        # Control/head/tail/state pointers
        # - input_head: device memory (kernel increments atomically)
        # - input_tail: mapped host memory (host writes, kernel reads)
        # - control_flag: mapped host memory (host writes, kernel reads)
        # - kernel_state: mapped host memory (kernel writes, host reads)

        # input_head in device memory
        self.d_input_head_arr = cp.zeros(1, dtype=cp.int32)

        # Allocate mapped host memory for tail/control/state using CUDA runtime
        cudart = _load_cudart()

        cudaHostAllocMapped = 2
        # Set arg/restype
        cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        cudart.cudaHostAlloc.restype = ctypes.c_int
        cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

        def _alloc_mapped_int32():
            h_ptr = ctypes.c_void_p()
            err = cudart.cudaHostAlloc(ctypes.byref(h_ptr), 4, cudaHostAllocMapped)
            if err != 0:
                raise RuntimeError(f"cudaHostAlloc failed: error {err}")
            d_ptr = ctypes.c_void_p()
            err = cudart.cudaHostGetDevicePointer(ctypes.byref(d_ptr), h_ptr, 0)
            if err != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer failed: error {err}")
            # Host numpy view
            h_view = np.ctypeslib.as_array(ctypes.cast(h_ptr.value, ctypes.POINTER(ctypes.c_int32)), shape=(1,))
            # Device CuPy view
            d_arr = cp.ndarray(1, dtype=cp.int32,
                               memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(d_ptr.value, 4, self), 0))
            return h_ptr, d_ptr, h_view, d_arr

        def _alloc_mapped_bytes(nbytes: int):
            h_ptr = ctypes.c_void_p()
            err = cudart.cudaHostAlloc(ctypes.byref(h_ptr), nbytes, cudaHostAllocMapped)
            if err != 0:
                raise RuntimeError(f"cudaHostAlloc(bytes) failed: error {err}")
            d_ptr = ctypes.c_void_p()
            err = cudart.cudaHostGetDevicePointer(ctypes.byref(d_ptr), h_ptr, 0)
            if err != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer(bytes) failed: error {err}")
            # Host numpy view over mapped memory
            h_view = np.ctypeslib.as_array(ctypes.cast(h_ptr.value, ctypes.POINTER(ctypes.c_ubyte)), shape=(nbytes,))
            # Device CuPy view
            d_arr = cp.ndarray(nbytes, dtype=cp.uint8,
                               memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(d_ptr.value, nbytes, self), 0))
            return h_ptr, d_ptr, h_view, d_arr

        self.h_input_tail_ptr, self.d_input_tail_ptr, self.h_input_tail, self.d_input_tail_arr = _alloc_mapped_int32()
        self.h_control_flag_ptr, self.d_control_flag_ptr, self.h_control_flag, self.d_control_flag_arr = _alloc_mapped_int32()
        self.h_kernel_state_ptr, self.d_kernel_state_ptr, self.h_kernel_state, self.d_kernel_state_arr = _alloc_mapped_int32()

        # Initialize values
        self.d_input_head_arr.fill(0)
        self.h_input_tail[0] = 0
        self.h_control_flag[0] = 0  # CONTROL_RUNNING
        self.h_kernel_state[0] = 1  # STATE_IDLE initially

        # Allocate buffer pools (outputs, created on init)
        # Size based on number of blocks since each block uses bid as its offset
        max_jobs = self.num_blocks  # One slot per block
        # Allocate output slots (one per block) in mapped host memory so host can read without D2H copies
        self.h_output_slots_ptr, self.d_output_slots_ptr, self.h_output_slots, self.d_output_slots_bytes = _alloc_mapped_bytes(self.num_blocks * self._sizeof_outputslot)

        max_samples_per_job = max_threads_per_job * max_N
        max_energies_per_job = max_threads_per_job

        self.max_samples_per_job = max_samples_per_job
        self.max_energies_per_job = max_energies_per_job

        # Allocate samples/energies pools in mapped host memory to avoid D2H copies that can deadlock with persistent kernels
        bytes_samples = max_jobs * max_samples_per_job * 4  # float32 bytes
        bytes_energies = max_jobs * max_energies_per_job * 4
        self.h_samples_pool_ptr, self.d_samples_pool_ptr, self.h_samples_pool_bytes, _ = _alloc_mapped_bytes(bytes_samples)
        self.h_energies_pool_ptr, self.d_energies_pool_ptr, self.h_energies_pool_bytes, _ = _alloc_mapped_bytes(bytes_energies)
        # Host float32 views
        self.h_samples_pool = self.h_samples_pool_bytes.view(np.float32)
        self.h_energies_pool = self.h_energies_pool_bytes.view(np.float32)
        # Device float32 views
        self.d_samples_pool = cp.ndarray(max_jobs * max_samples_per_job, dtype=cp.float32,
                                         memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_samples_pool_ptr.value, bytes_samples, self), 0))
        self.d_energies_pool = cp.ndarray(max_jobs * max_energies_per_job, dtype=cp.float32,
                                          memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_energies_pool_ptr.value, bytes_energies, self), 0))

        # Launch kernel (runs persistently in separate stream)
        self.stream = cp.cuda.Stream(non_blocking=True)

        # Launch with proper pointer types
        self.kernel(
            (self.num_blocks,), (max_threads_per_job,),
            (
                self.d_input_ring_bytes,
                self.ring_size,
                self.d_input_head_arr,            # device memory (int32[1])
                self.d_input_tail_arr,            # mapped host memory as device pointer
                self.d_output_slots_bytes,        # OutputSlot array (no ring buffer)
                self.d_control_flag_arr,          # mapped host memory as device pointer
                self.d_kernel_state_arr,          # mapped host memory as device pointer
                self.d_samples_pool,
                self.d_energies_pool,
                max_samples_per_job,
                max_energies_per_job
            ),
            stream=self.stream
        )

    def _get_sm_count(self) -> int:
        """Query number of SM cores on GPU."""
        device = cp.cuda.Device()
        return device.attributes['MultiProcessorCount']

    def _load_kernel_code(self) -> str:
        """Load CUDA kernel source code."""
        # Read from GPU/cuda_sa.cu relative to this file
        kernel_path = os.path.join(os.path.dirname(__file__), 'cuda_sa.cu')
        with open(kernel_path, 'r') as f:
            return f.read()

    def enqueue_job(
        self,
        job_id: int,
        num_reads: int,
        N: int,
        h: Optional[np.ndarray] = None,
        J: Optional[Dict] = None,
        num_betas: int = 1,
        num_sweeps_per_beta: int = 256,
        seed: Optional[int] = None
    ) -> None:
        """
        Enqueue a job to the input ring.

        Args:
            job_id: Unique job identifier
            num_reads: Number of threads to use
            N: Number of variables
            h: Linear bias array (optional)
            J: Coupling dictionary (optional)
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature step
            seed: RNG seed for reproducibility (if None, uses random seed)
        """
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Get current tail from host memory (no GPU sync needed!)
        tail = int(self.h_input_tail[0])
        slot = tail % self.ring_size

        # Convert h and J to device arrays if provided
        h_ptr = 0
        h_size = 0
        if h is not None:
            # Store h array on device (keep reference to prevent garbage collection)
            self.d_h_array = cp.asarray(h, dtype=cp.float32)
            h_ptr = self.d_h_array.data.ptr
            h_size = len(h)
            print(f"[DEBUG] Enqueuing h array: ptr={h_ptr}, size={h_size}")

        J_ptr = 0
        J_size = 0
        if J is not None:
            # J is a dict, convert to CSR format
            # For now, just store the pointer (CSR data is already on device)
            J_ptr = self.d_csr_J_vals.data.ptr
            J_size = len(self.d_csr_J_vals)
            print(f"[DEBUG] Enqueuing J array: ptr={J_ptr}, size={J_size}")

        # Create JobDesc struct
        # struct JobDesc {
        #     int job_id, num_reads, num_betas, num_sweeps_per_beta;
        #     int csr_row_ptr_offset, csr_col_ind_offset, N;
        #     float* h; int h_size; float* J; int J_size; float* beta_schedule;
        #     int8_t* output_samples; float* output_energies;
        #     unsigned int seed;
        # };
        # Format: 7 ints (i) + pointer(Q) + int(i) + pointer(Q) + int(i) + 3 pointers(QQQ) + int(I)
        job_bytes = struct.pack(
            'iiiiiiiQiQiQQQI',  # Total: 7i + Q + i + Q + i + 3Q + I = 17 fields
            job_id,                    # job_id
            num_reads,                 # num_reads
            num_betas,                 # num_betas
            num_sweeps_per_beta,       # num_sweeps_per_beta
            0,                         # csr_row_ptr_offset (unused)
            0,                         # csr_col_ind_offset (unused)
            N,                         # N
            h_ptr,                     # h pointer
            h_size,                    # h_size
            J_ptr,                     # J pointer (CSR J values)
            J_size,                    # J_size
            self.d_beta_schedule.data.ptr,  # beta_schedule pointer
            0,                         # output_samples pointer (null)
            0,                         # output_energies pointer (null)
            seed                       # seed (unsigned int)
        )

        # Write directly to device ring buffer
        offset = slot * self._sizeof_jobdesc
        job_bytes_array = np.frombuffer(job_bytes, dtype=np.uint8)

        # Copy to device (this happens on the default stream)
        self.d_input_ring_bytes[offset:offset + len(job_bytes_array)] = cp.asarray(job_bytes_array)

        # Increment tail (this signals the kernel that a job is available)
        new_tail = int(tail + 1)

        # Update host memory directly (kernel will see it via pinned memory)
        self.h_input_tail[0] = new_tail

        # NOTE: We cannot use synchronize() here because the persistent kernel
        # runs indefinitely. The kernel uses volatile reads which should see
        # the update eventually due to memory coherency.

    def try_dequeue_result(self) -> Optional[Dict]:
        """
        Poll all output slots for a ready result (non-blocking).

        Returns:
            Result dict with job_id, min_energy, avg_energy, num_reads, N,
            samples_offset, energies_offset, or None if no results ready
        """
        # NOTE: Do NOT call stream.synchronize() here!
        # The kernel is waiting for the host to set ready=2, so if we synchronize,
        # we'll deadlock waiting for the kernel to finish while the kernel waits for us.
        # Instead, we poll the mapped host memory directly (zero-copy, no sync needed).

        # Poll all output slots
        for slot_idx in range(self.num_blocks):
            result = self._read_output_slot(slot_idx)
            if result and result['ready'] == 1:
                # Event-driven debug: print when result is found
                print(f"[HOST] Found result in slot {slot_idx}: job_id={result['job_id']}, min_e={result['min_energy']}", flush=True)
                # Mark as collected
                self._mark_slot_collected(slot_idx)
                return result

        return None

    def _read_output_slot(self, slot_idx: int) -> Optional[Dict]:
        """Read OutputSlot from device."""
        offset = slot_idx * self._sizeof_outputslot
        slot_bytes = self.h_output_slots[offset:offset + self._sizeof_outputslot]
        ready, job_id, min_e, avg_e, num_reads, N, samples_off, energies_off = struct.unpack('iiffiiII', slot_bytes)
        if ready != 0 and slot_idx < 5:
            print(f"[HOST] Slot {slot_idx}: ready={ready}, job_id={job_id}, min_e={min_e}")
        return {
            'ready': ready,
            'job_id': job_id,
            'min_energy': min_e,
            'avg_energy': avg_e,
            'num_reads': num_reads,
            'N': N,
            'samples_offset': samples_off,
            'energies_offset': energies_off,
        }

    def _mark_slot_collected(self, slot_idx: int) -> None:
        """Mark slot as collected (ready=2)."""
        offset = slot_idx * self._sizeof_outputslot
        struct.pack_into('i', self.h_output_slots, offset, 2)

    def get_samples(self, result: Dict) -> np.ndarray:
        """Get samples from result."""
        offset = result['samples_offset']
        count = result['num_reads'] * result['N']
        return self.h_samples_pool[offset:offset + count].reshape(result['num_reads'], result['N'])

    def get_energies(self, result: Dict) -> np.ndarray:
        """Get energies from result."""
        offset = result['energies_offset']
        count = result['num_reads']
        return self.h_energies_pool[offset:offset + count]

    def get_kernel_state(self) -> int:
        """Get current kernel state (0=STATE_RUNNING, 1=STATE_IDLE)."""
        return int(self.h_kernel_state[0])

    def stop_immediate(self) -> None:
        """
        Stop the persistent kernel immediately (CONTROL_STOP).

        Does not wait for queued jobs to complete.
        """
        self.h_control_flag[0] = 1  # CONTROL_STOP
        self.stream.synchronize()

    def stop_drain(self) -> None:
        """
        Stop the persistent kernel after draining queue (CONTROL_DRAIN).

        Finishes all queued jobs before exiting.
        """
        self.h_control_flag[0] = 2  # CONTROL_DRAIN
        self.stream.synchronize()

    def stop(self, drain: bool = True) -> None:
        """
        Stop the persistent kernel (deprecated - use stop_immediate or stop_drain).

        Args:
            drain: If True, finish current jobs. If False, immediate shutdown.
        """
        if drain:
            self.stop_drain()
        else:
            self.stop_immediate()


class CudaKernelRealSA:
    """Persistent CUDA kernel with real simulated annealing algorithm."""

    def __init__(self, ring_size: int = 16, max_threads_per_job: int = 256,
                 max_h_size: int = 10000, max_J_size: int = 100000, max_N: int = 4600,
                 debug_verbose: int = 0, debug_kernel: int = 0, debug_workers: int = 0,
                 verbose: bool = True):
        """
        Initialize persistent CUDA kernel with real SA.

        Args:
            ring_size: Size of input ring buffer
            max_threads_per_job: Maximum threads per job
            max_h_size: Maximum size of h array
            max_J_size: Maximum size of J array
            max_N: Maximum number of variables
            debug_verbose: Enable DEBUG_VERBOSE (0 or 1)
            debug_kernel: Enable DEBUG_KERNEL (0 or 1)
            debug_workers: Enable DEBUG_WORKERS (0 or 1)
            verbose: Enable Python print statements (default True)
        """
        self.ring_size = ring_size
        self.max_threads_per_job = max_threads_per_job
        self.verbose = verbose
        self.max_h_size = max_h_size
        self.max_J_size = max_J_size
        self.max_N = max_N

        # Query SM count
        self.num_blocks = self._get_sm_count()

        # Thread safety for enqueue operations
        self._enqueue_lock = threading.Lock()

        # Build compile options with debug flags
        options = ['--use_fast_math', '--maxrregcount=64']
        if debug_verbose:
            options.append('-DDEBUG_VERBOSE=1')
        if debug_kernel:
            options.append('-DDEBUG_KERNEL=1')
        if debug_workers:
            options.append('-DDEBUG_WORKERS=1')

        # Compile kernel with fast math optimizations
        # Note: NVRTC doesn't support -O3, uses --use_fast_math instead
        self.module = cp.RawModule(
            code=self._load_kernel_code(),
            options=tuple(options)
        )
        self.kernel = self.module.get_function('cuda_sa_persistent_real')

        # Struct sizes
        # JobDesc: job_id, num_reads, num_betas, num_sweeps_per_beta (4 ints)
        #          csr_row_ptr*, csr_col_ind* (2 pointers)
        #          N (int), h* (pointer), h_size (int), csr_J_vals* (pointer), J_size (int)
        #          beta_schedule*, output_samples*, output_energies* (3 pointers)
        #          seed (unsigned int)
        self._jobdesc_fmt = 'iiiiQQiQiQiQQQI'
        self._sizeof_jobdesc = struct.calcsize(self._jobdesc_fmt)
        self._sizeof_outputslot = 32

        # Control/head/tail/state pointers
        cudart = _load_cudart()

        cudaHostAllocMapped = 2
        cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        cudart.cudaHostAlloc.restype = ctypes.c_int
        cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

        def _alloc_mapped_int32():
            h_ptr = ctypes.c_void_p()
            err = cudart.cudaHostAlloc(ctypes.byref(h_ptr), 4, cudaHostAllocMapped)
            if err != 0:
                raise RuntimeError(f"cudaHostAlloc failed: error {err}")
            d_ptr = ctypes.c_void_p()
            err = cudart.cudaHostGetDevicePointer(ctypes.byref(d_ptr), h_ptr, 0)
            if err != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer failed: error {err}")
            h_view = np.ctypeslib.as_array(ctypes.cast(h_ptr.value, ctypes.POINTER(ctypes.c_int32)), shape=(1,))
            d_arr = cp.ndarray(1, dtype=cp.int32,
                               memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(d_ptr.value, 4, self), 0))
            return h_ptr, d_ptr, h_view, d_arr

        def _alloc_mapped_bytes(nbytes: int):
            h_ptr = ctypes.c_void_p()
            err = cudart.cudaHostAlloc(ctypes.byref(h_ptr), nbytes, cudaHostAllocMapped)
            if err != 0:
                raise RuntimeError(f"cudaHostAlloc(bytes) failed: error {err}")
            d_ptr = ctypes.c_void_p()
            err = cudart.cudaHostGetDevicePointer(ctypes.byref(d_ptr), h_ptr, 0)
            if err != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer(bytes) failed: error {err}")
            h_view = np.ctypeslib.as_array(ctypes.cast(h_ptr.value, ctypes.POINTER(ctypes.c_ubyte)), shape=(nbytes,))
            d_arr = cp.ndarray(nbytes, dtype=cp.uint8,
                               memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(d_ptr.value, nbytes, self), 0))
            return h_ptr, d_ptr, h_view, d_arr

        # Allocate ring buffer for job pointers (mapped memory)
        ring_bytes = ring_size * 8  # 8 bytes per pointer
        self.h_input_ring_ptr, self.d_input_ring_ptr, self.h_input_ring_bytes, _ = _alloc_mapped_bytes(ring_bytes)
        # Cast to uint64 array for pointer storage
        self.h_input_ring_ptrs = np.ctypeslib.as_array(
            ctypes.cast(self.h_input_ring_ptr.value, ctypes.POINTER(ctypes.c_uint64)),
            shape=(ring_size,)
        )
        self.d_input_ring_ptrs = cp.ndarray(ring_size, dtype=cp.uint64,
                                             memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_input_ring_ptr.value, ring_bytes, self), 0))

        # Allocate host_writing_mutex (1 when Python is writing, 0 otherwise)
        self.h_host_writing_mutex_ptr, self.d_host_writing_mutex_ptr, self.h_host_writing_mutex, self.d_host_writing_mutex_arr = _alloc_mapped_int32()

        self.h_input_tail_ptr, self.d_input_tail_ptr, self.h_input_tail, self.d_input_tail_arr = _alloc_mapped_int32()
        self.h_control_flag_ptr, self.d_control_flag_ptr, self.h_control_flag, self.d_control_flag_arr = _alloc_mapped_int32()
        self.h_kernel_state_ptr, self.d_kernel_state_ptr, self.h_kernel_state, self.d_kernel_state_arr = _alloc_mapped_int32()

        # Initialize values
        self.h_input_tail[0] = 0
        self.h_control_flag[0] = 0
        self.h_kernel_state[0] = 1
        self.h_host_writing_mutex[0] = 0  # Not writing initially

        # Head is device-only (kernel updates it with atomicCAS)
        self.d_input_head_arr = cp.zeros(1, dtype=cp.int32)
        # Initialize ring buffer to zeros
        for i in range(ring_size):
            self.h_input_ring_ptrs[i] = 0

        # Allocate output slots
        max_jobs = self.num_blocks
        self.h_output_slots_ptr, self.d_output_slots_ptr, self.h_output_slots, self.d_output_slots_bytes = _alloc_mapped_bytes(self.num_blocks * self._sizeof_outputslot)

        max_samples_per_job = max_threads_per_job * max_N
        max_energies_per_job = max_threads_per_job
        self.max_samples_per_job = max_samples_per_job
        self.max_energies_per_job = max_energies_per_job

        # Allocate samples/energies pools
        bytes_samples = max_jobs * max_samples_per_job * 4
        bytes_energies = max_jobs * max_energies_per_job * 4
        self.h_samples_pool_ptr, self.d_samples_pool_ptr, self.h_samples_pool_bytes, _ = _alloc_mapped_bytes(bytes_samples)
        self.h_energies_pool_ptr, self.d_energies_pool_ptr, self.h_energies_pool_bytes, _ = _alloc_mapped_bytes(bytes_energies)
        self.h_samples_pool = self.h_samples_pool_bytes.view(np.float32)
        self.h_energies_pool = self.h_energies_pool_bytes.view(np.float32)
        self.d_samples_pool = cp.ndarray(max_jobs * max_samples_per_job, dtype=cp.float32,
                                         memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_samples_pool_ptr.value, bytes_samples, self), 0))
        self.d_energies_pool = cp.ndarray(max_jobs * max_energies_per_job, dtype=cp.float32,
                                          memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(self.d_energies_pool_ptr.value, bytes_energies, self), 0))

        # Keep references to per-job GPU arrays to prevent garbage collection
        # Each enqueue_job() will allocate new GPU memory and add it here
        self._job_data_refs = []

        # Track current batch state
        self._batch_start_tail = 0  # Tail value at start of current batch
        self._batch_job_count = 0   # Number of jobs in current batch

        # Delta energy workspace (one slice per global thread across all blocks)
        workspace_capacity = max_N
        self.d_delta_energy_workspace = cp.zeros(self.num_blocks * max_threads_per_job * workspace_capacity, dtype=cp.int8)

        # Launch kernel (no global CSR arrays - each job has its own)
        self.stream = cp.cuda.Stream(non_blocking=True)
        if self.verbose:
            print(f"[PYTHON] Launching kernel with {self.num_blocks} blocks, {max_threads_per_job} threads/block", flush=True)
        self.kernel(
            (self.num_blocks,), (max_threads_per_job,),
            (
                self.d_input_ring_ptrs,  # Array of pointers to JobDesc
                self.ring_size,
                self.d_input_head_arr,  # Device-only atomic counter
                self.d_input_tail_arr,  # Mapped memory (host writes, kernel reads)
                self.d_host_writing_mutex_arr,  # Mutex: 1=host writing, 0=idle
                self.d_output_slots_bytes,
                self.d_control_flag_arr,
                self.d_kernel_state_arr,
                self.d_samples_pool,
                self.d_energies_pool,
                max_samples_per_job,
                max_energies_per_job,
                self.d_delta_energy_workspace,
                max_N,
            ),
            stream=self.stream
        )
        if self.verbose:
            print(f"[PYTHON] Kernel launched successfully", flush=True)

    @staticmethod
    def _get_sm_count() -> int:
        """Get number of SMs on current GPU."""
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        return props['multiProcessorCount']

    @staticmethod
    def _load_kernel_code() -> str:
        """Load CUDA kernel code from file."""
        kernel_file = os.path.join(os.path.dirname(__file__), 'cuda_sa.cu')
        with open(kernel_file, 'r') as f:
            return f.read()

    def enqueue_job(self, job_id: int, h: Optional[Dict] = None, J: Optional[Dict] = None,
                    num_reads: int = 100, num_betas: int = 10, num_sweeps_per_beta: int = 100,
                    N: Optional[int] = None, seed: Optional[int] = None,
                    beta_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Enqueue a job to the persistent kernel.

        Args:
            job_id: Unique job identifier
            h: Linear terms (dict or numpy array, or None for zeros)
            J: Quadratic terms (dict or None for zeros)
            num_reads: Number of samples to generate
            num_betas: Number of temperature steps
            num_sweeps_per_beta: Sweeps per temperature
            N: Number of variables (auto-detected if not provided)
            seed: RNG seed for reproducibility (if None, uses random seed)
            beta_range: (hot_beta, cold_beta) tuple or None for auto-compute from problem
        """
        # Validate num_reads doesn't exceed thread capacity
        if num_reads > self.max_threads_per_job:
            raise ValueError(
                f"num_reads={num_reads} exceeds max_threads_per_job={self.max_threads_per_job}. "
                f"Each job can have at most {self.max_threads_per_job} reads."
            )
        if h is None:
            h = {}
        if J is None:
            J = {}

        # Determine N if not provided
        if N is None:
            N = max(max(h.keys()) if isinstance(h, dict) and h else 0,
                    max(max(i, j) for i, j in J.keys()) if J else 0) + 1

        # Convert J to symmetric CSR
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        rows = [[] for _ in range(N)]
        vals = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i >= N or j >= N:
                continue
            rows[i].append(j)
            vals[i].append(int(Jij))
            rows[j].append(i)
            vals[j].append(int(Jij))
        nnz = 0
        for i in range(N):
            csr_row_ptr[i] = nnz
            if rows[i]:
                order = np.argsort(rows[i])
                rows[i] = [rows[i][k] for k in order]
                vals[i] = [vals[i][k] for k in order]
            nnz += len(rows[i])
        csr_row_ptr[N] = nnz
        csr_col_ind = np.array([c for row in rows for c in row], dtype=np.int32)
        csr_J_vals = np.array([v for row in vals for v in row], dtype=np.int8)

        # Allocate fresh GPU memory for this job's CSR data
        # Use cp.array() to ensure a copy, not just a view
        d_csr_row_ptr = cp.array(csr_row_ptr, dtype=cp.int32, copy=True)
        d_csr_col_ind = cp.array(csr_col_ind, dtype=cp.int32, copy=True)
        d_csr_J_vals = cp.array(csr_J_vals, dtype=cp.int8, copy=True)
        cp.cuda.Stream.null.synchronize()  # Ensure allocation completes

        # Prepare h on device (allocate fresh memory for this job)
        h_ptr = 0
        h_size = 0
        if isinstance(h, dict):
            # Convert dict to dense array
            h_arr = np.zeros(N, dtype=np.float32)
            for idx, val in h.items():
                if idx < N:
                    h_arr[idx] = float(val)
            d_h_array = cp.asarray(h_arr, dtype=cp.float32)
            h_ptr = d_h_array.data.ptr
            h_size = N
        elif isinstance(h, np.ndarray):
            if h.dtype != np.float32:
                h = h.astype(np.float32, copy=False)
            if h.shape[0] != N:
                # Pad or trim to N
                h_fixed = np.zeros(N, dtype=np.float32)
                h_fixed[:min(N, h.shape[0])] = h[:min(N, h.shape[0])]
                h = h_fixed
            d_h_array = cp.asarray(h, dtype=cp.float32)
            h_ptr = d_h_array.data.ptr
            h_size = N
        else:
            # No h provided
            d_h_array = None
            h_ptr = 0
            h_size = 0

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Compute beta schedule (geometric, matching Metal/D-Wave)
        if beta_range is None:
            # Auto-compute beta range from problem
            beta_range = _default_ising_beta_range(h if isinstance(h, dict) else {}, J if isinstance(J, dict) else {})

        hot_beta, cold_beta = beta_range

        # Create geometric beta schedule
        if num_betas == 1:
            beta_schedule = np.array([cold_beta], dtype=np.float32)
        else:
            beta_schedule = np.geomspace(hot_beta, cold_beta, num=num_betas, dtype=np.float32)

        # Upload beta schedule to fresh GPU memory
        d_beta_schedule = cp.asarray(beta_schedule, dtype=cp.float32)
        beta_ptr = d_beta_schedule.data.ptr

        # Get CSR pointers (from fresh allocations above)
        csr_row_ptr_ptr = d_csr_row_ptr.data.ptr
        csr_col_ind_ptr = d_csr_col_ind.data.ptr
        csr_J_vals_ptr = d_csr_J_vals.data.ptr

        # Pack JobDesc with pointers
        job_bytes = struct.pack(
            self._jobdesc_fmt,
            job_id,                     # job_id
            num_reads,                  # num_reads
            num_betas,                  # num_betas
            num_sweeps_per_beta,        # num_sweeps_per_beta
            csr_row_ptr_ptr,            # csr_row_ptr pointer
            csr_col_ind_ptr,            # csr_col_ind pointer
            N,                          # N
            h_ptr,                      # h pointer
            h_size,                     # h_size
            csr_J_vals_ptr,             # csr_J_vals pointer
            int(len(csr_J_vals)),       # J_size
            beta_ptr,                   # beta_schedule pointer
            0,                          # output_samples pointer (kernel uses global pool)
            0,                          # output_energies pointer (kernel uses global pool)
            seed                        # seed (unsigned int)
        )
        job_bytes_array = np.frombuffer(job_bytes, dtype=np.uint8)

        # Allocate JobDesc in GPU memory (this happens on default stream)
        d_jobdesc = cp.asarray(job_bytes_array, dtype=cp.uint8)
        jobdesc_ptr = d_jobdesc.data.ptr

        # Keep references to prevent garbage collection
        job_data = {
            'csr_row_ptr': d_csr_row_ptr,
            'csr_col_ind': d_csr_col_ind,
            'csr_J_vals': d_csr_J_vals,
            'h_array': d_h_array,
            'beta_schedule': d_beta_schedule,
            'jobdesc': d_jobdesc  # Keep JobDesc alive
        }
        self._job_data_refs.append(job_data)

        # CRITICAL SECTION: Reserve slot and write pointer
        # Lock prevents multiple Python threads from getting same slot
        with self._enqueue_lock:
            # Check if adding this job would overflow the ring buffer
            if self._batch_job_count >= self.ring_size:
                raise RuntimeError(
                    f"Batch overflow: cannot enqueue more than {self.ring_size} jobs without calling signal_batch_ready(). "
                    f"Current batch has {self._batch_job_count} jobs."
                )

            # Wait for GPU to finish previous batch (signal == 0)
            wait_count = 0
            while self.h_host_writing_mutex[0] != 0:
                time.sleep(0.001)  # 1ms
                wait_count += 1
                if wait_count > 10000:  # 10 second timeout
                    raise RuntimeError("Timeout waiting for GPU to finish previous batch")

            # Read current tail
            tail = int(self.h_input_tail[0])
            slot = tail % self.ring_size

            # Write pointer to HOST side of mapped ring buffer
            self.h_input_ring_ptrs[slot] = jobdesc_ptr

            # Increment tail (but don't signal yet - wait for batch complete)
            self.h_input_tail[0] = tail + 1

            # Track batch state
            self._batch_job_count += 1

            # Store for signal_batch_ready() call
            self._last_enqueued_slot = slot
            self._last_enqueued_tail = tail + 1

        if self.verbose:
            print(f"[PYTHON] Enqueued job_id={job_id}, num_reads={num_reads}, slot={slot}, tail={tail}->{tail+1}, batch={self._batch_job_count}/{self.ring_size}, ptr=0x{jobdesc_ptr:x}", flush=True)

    def signal_batch_ready(self):
        """Signal to GPU that a batch of jobs is ready to be dequeued."""
        with self._enqueue_lock:
            if self._batch_job_count == 0:
                if self.verbose:
                    print(f"[PYTHON] Warning: signal_batch_ready() called with no jobs in batch", flush=True)
                return

            # Set signal to 1 to tell GPU batch is ready
            self.h_host_writing_mutex[0] = 1

            # Force the write to commit by reading it back
            # This ensures the value is visible to the GPU
            readback = int(self.h_host_writing_mutex[0])
            assert readback == 1, f"Signal write failed: expected 1, got {readback}"

            if self.verbose:
                print(f"[PYTHON] Signaled batch ready ({self._batch_job_count} jobs, tail={self.h_input_tail[0]})", flush=True)

            # Reset batch counter for next batch
            self._batch_start_tail = int(self.h_input_tail[0])
            self._batch_job_count = 0

    def signal_batch_notready(self):
        """Signal to GPU to pause processing while host writes more jobs.

        This allows the host to safely reset the signal and write additional jobs
        to the ring buffer when space becomes available.
        """
        with self._enqueue_lock:
            # Only reset if currently signaled as ready
            if self.h_host_writing_mutex[0] == 1:
                self.h_host_writing_mutex[0] = 0

                # Force the write to commit
                readback = int(self.h_host_writing_mutex[0])
                assert readback == 0, f"Signal write failed: expected 0, got {readback}"

                if self.verbose:
                    print(f"[PYTHON] Signaled batch not ready, waiting 200ms for GPU to pause", flush=True)

                # Wait for GPU to see the signal and stop dequeuing
                time.sleep(0.2)  # 200ms as requested

    def enqueue_batch_streaming(self, jobs: list):
        """Enqueue multiple jobs with automatic batching and streaming.

        This implements the logic:
        - If ringbuf not full, enqueue jobs up to ringbuf capacity
        - If signal_ready, call signal_batch_notready() and wait 200ms
        - Signal batch ready when batch is full or no more jobs
        - If more jobs remain, poll until ringbuffer empties, then repeat

        Args:
            jobs: List of job specifications, each is a dict with keys:
                  'job_id', 'h', 'J', 'num_reads', 'num_betas', 'num_sweeps_per_beta',
                  and optionally 'N', 'seed', 'beta_range'
        """
        jobs_remaining = list(jobs)  # Make a copy

        while jobs_remaining:
            # If signal is ready (GPU is processing), reset it to safely add more jobs
            if self.h_host_writing_mutex[0] == 1:
                self.signal_batch_notready()

            # Enqueue jobs until ring buffer is full or no more jobs
            batch_count = 0
            while jobs_remaining and self._batch_job_count < self.ring_size:
                job = jobs_remaining.pop(0)
                self.enqueue_job(
                    job_id=job['job_id'],
                    h=job.get('h'),
                    J=job.get('J'),
                    num_reads=job['num_reads'],
                    num_betas=job.get('num_betas', 256),
                    num_sweeps_per_beta=job.get('num_sweeps_per_beta', 1),
                    N=job.get('N'),
                    seed=job.get('seed'),
                    beta_range=job.get('beta_range')
                )
                batch_count += 1

            # Signal batch ready to enable GPU processing
            if batch_count > 0:
                self.signal_batch_ready()

            # If more jobs remain, poll until ring buffer has space
            if jobs_remaining:
                if self.verbose:
                    print(f"[PYTHON] Waiting for ring buffer to empty ({len(jobs_remaining)} jobs remaining)...", flush=True)
                while self.h_host_writing_mutex[0] != 0:
                    time.sleep(0.01)  # Poll every 10ms
                if self.verbose:
                    print(f"[PYTHON] Ring buffer ready for next batch", flush=True)

    def try_dequeue_result(self) -> Optional[Dict]:
        """Poll all output slots for a ready result (non-blocking)."""
        for slot_idx in range(self.num_blocks):
            result = self._read_output_slot(slot_idx)
            if result and result['ready'] == 1:
                # Mark as collected (ready=2) so we don't re-read this result
                self._mark_slot_collected(slot_idx)
                return result
            # Skip slots that are ready=0 (empty) or ready=2 (already collected)
        return None

    def debug_dump_current_problem(self, head: int = 5) -> Dict:
        """Return and print a summary of the current problem loaded on device.
        Includes N, h stats, CSR nnz/degree and first few entries.
        """
        summary: Dict[str, object] = {}
        # N and h
        N = int(self.d_h_array.size) if hasattr(self, 'd_h_array') and self.d_h_array is not None else 0
        summary['N'] = N
        if N > 0:
            h_head = cp.asnumpy(self.d_h_array[: min(head, N)])
            h_nz = int(cp.count_nonzero(self.d_h_array).get())
            sum_abs_h = float(cp.sum(cp.abs(self.d_h_array)).get())
        else:
            h_head = np.array([], dtype=np.float32)
            h_nz = 0
            sum_abs_h = 0.0
        summary['h_head'] = h_head
        summary['h_nz'] = h_nz
        summary['sum_abs_h'] = sum_abs_h
        # CSR
        if N > 0:
            row_ptr_host = cp.asnumpy(self.d_csr_row_ptr[: N + 1])
            row0 = int(row_ptr_host[0])
            nnz = int(row_ptr_host[N] - row0)
            summary['csr_row_ptr_0'] = row0
            summary['csr_row_ptr_N'] = int(row_ptr_host[N])
            summary['nnz'] = nnz
            first_deg = int(row_ptr_host[1] - row_ptr_host[0]) if N >= 1 else 0
            last_deg = int(row_ptr_host[N] - row_ptr_host[N - 1]) if N >= 2 else 0
            summary['first_deg'] = first_deg
            summary['last_deg'] = last_deg
            take = min(head, nnz)
            if nnz > 0:
                cols_head = cp.asnumpy(self.d_csr_col_ind[row0 : row0 + take])
                J_head = cp.asnumpy(self.d_csr_J_vals[row0 : row0 + take])
            else:
                cols_head = np.array([], dtype=np.int32)
                J_head = np.array([], dtype=np.int8)
            summary['col_ind_head'] = cols_head
            summary['J_vals_head'] = J_head
        # Print concise summary
        print(f"[HOST] Device problem summary: N={summary.get('N')} h_nz={h_nz} sum|h|={sum_abs_h:.1f} nnz={summary.get('nnz')}\n"
              f"       first_deg={summary.get('first_deg')} last_deg={summary.get('last_deg')}\n"
              f"       h_head={h_head[:min(5, len(h_head))]} J_head={summary.get('J_vals_head')} cols_head={summary.get('col_ind_head')}")
        # Expected nnz based on last enqueue, if known
        if hasattr(self, '_last_expected_nnz'):
            expected = int(self._last_expected_nnz)
            print(f"[HOST] Expected nnz from enqueue: {expected}")
            summary['expected_nnz'] = expected
        return summary

    def _read_output_slot(self, slot_idx: int) -> Optional[Dict]:
        """Read output slot data."""
        offset = slot_idx * self._sizeof_outputslot
        slot_bytes = self.h_output_slots[offset:offset + self._sizeof_outputslot]
        ready, job_id, min_e, avg_e, num_reads, N, samples_off, energies_off = struct.unpack('iiffiiII', slot_bytes)
        return {
            'ready': ready,
            'job_id': job_id,
            'min_energy': min_e,
            'avg_energy': avg_e,
            'num_reads': num_reads,
            'N': N,
            'samples_offset': samples_off,
            'energies_offset': energies_off,
        }

    def _mark_slot_collected(self, slot_idx: int) -> None:
        """Mark slot as collected (ready=2)."""
        offset = slot_idx * self._sizeof_outputslot
        struct.pack_into('i', self.h_output_slots, offset, 2)

    def get_kernel_state(self) -> int:
        """Get current kernel state (0=STATE_RUNNING, 1=STATE_IDLE)."""
        return int(self.h_kernel_state[0])

    def get_ring_buffer_state(self) -> dict:
        """Get current ring buffer state for diagnostics."""
        # Note: head is in device memory, need to copy to host
        head_val = int(self.d_input_head_arr.get()[0])
        return {
            'head': head_val,
            'tail': int(self.h_input_tail[0]),
            'mutex': int(self.h_host_writing_mutex[0]),
            'batch_job_count': self._batch_job_count,
            'ring_size': self.ring_size,
            'kernel_state': int(self.h_kernel_state[0])
        }

    def get_samples(self, result: Dict) -> np.ndarray:
        """Get samples from result."""
        offset = result['samples_offset']
        count = result['num_reads'] * result['N']
        return self.h_samples_pool[offset:offset + count].reshape(result['num_reads'], result['N'])

    def get_energies(self, result: Dict) -> np.ndarray:
        """Get energies from result."""
        offset = result['energies_offset']
        count = result['num_reads']
        return self.h_energies_pool[offset:offset + count]

    def stop_immediate(self) -> None:
        """
        Stop the persistent kernel immediately (CONTROL_STOP).

        Does not wait for queued jobs to complete.
        """
        self.h_control_flag[0] = 1  # CONTROL_STOP
        self.stream.synchronize()

    def stop_drain(self) -> None:
        """
        Stop the persistent kernel after draining queue (CONTROL_DRAIN).

        Finishes all queued jobs before exiting.
        """
        self.h_control_flag[0] = 2  # CONTROL_DRAIN
        self.stream.synchronize()

    def stop(self, drain: bool = True) -> None:
        """
        Stop the persistent kernel (deprecated - use stop_immediate or stop_drain).

        Args:
            drain: If True, finish current jobs. If False, immediate shutdown.
        """
        if drain:
            self.stop_drain()
        else:
            self.stop_immediate()

    def _read_output_slot(self, slot_idx: int) -> Optional[Dict]:
        """Read OutputSlot from device."""
        offset = slot_idx * self._sizeof_outputslot

        # Read directly from mapped host view (written by device)
        slot_bytes = self.h_output_slots[offset:offset + self._sizeof_outputslot].tobytes()

        # Unpack OutputSlot: ready(i), job_id(i), min_energy(f), avg_energy(f),
        #                     num_reads(i), N(i), samples_offset(i), energies_offset(i)
        # Format: 2 ints + 2 floats + 4 ints = 'iiffiiii'
        fields = struct.unpack('iiffiiii', slot_bytes)

        return {
            'ready': fields[0],
            'job_id': fields[1],
            'min_energy': fields[2],
            'avg_energy': fields[3],
            'num_reads': fields[4],
            'N': fields[5],
            'samples_offset': fields[6],
            'energies_offset': fields[7]
        }

    def _mark_slot_collected(self, slot_idx: int) -> None:
        """Mark an output slot as collected (ready=2) to prevent re-reading."""
        offset = slot_idx * self._sizeof_outputslot

        # Write ready=2 to the first 4 bytes (mapped host view)
        # This marks it as collected and signals to kernel that we read the job
        ready_bytes = struct.pack('i', 2)
        self.h_output_slots[offset:offset+4] = np.frombuffer(ready_bytes, dtype=np.uint8)

    def _reset_slot_for_reuse(self, slot_idx: int) -> None:
        """Reset an output slot from ready=2 back to ready=0 for next job."""
        offset = slot_idx * self._sizeof_outputslot

        # Write ready=0 to the first 4 bytes (mapped host view)
        # This signals to kernel that the slot is available for the next result
        ready_bytes = struct.pack('i', 0)
        self.h_output_slots[offset:offset+4] = np.frombuffer(ready_bytes, dtype=np.uint8)

    def get_samples(self, result: Dict) -> np.ndarray:
        """
        Extract samples from result.

        Args:
            result: Result dict from try_dequeue_result()

        Returns:
            Samples array of shape (num_reads, N)
        """
        offset = result['samples_offset']
        num_reads = result['num_reads']
        N = result['N']
        num_floats = num_reads * N

        # Read directly from mapped host memory
        samples_flat = self.h_samples_pool[offset:offset + num_floats].copy()
        return samples_flat.reshape(num_reads, N)

    def get_kernel_state(self) -> int:
        """Get current kernel state (0=IDLE, 1=RUNNING)."""
        return int(self.h_kernel_state[0])

    def get_energies(self, result: Dict) -> np.ndarray:
        """
        Extract energies from result.

        Args:
            result: Result dict from try_dequeue_result()

        Returns:
            Energies array of shape (num_reads,)
        """
        offset = result['energies_offset']
        num_reads = result['num_reads']

        # Read directly from mapped host memory
        arr = self.h_energies_pool[offset:offset + num_reads].copy()
        return arr
