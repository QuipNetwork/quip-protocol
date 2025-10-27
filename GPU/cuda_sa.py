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

            # CUDA kernel outputs SPIN format {-1, +1} directly (unpacked state)
            # No conversion needed
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


