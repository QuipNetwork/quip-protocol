"""CUDA SA support types: IsingJob dataclass and mock kernel for testing."""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class IsingJob:
    """Represents a single Ising problem to be solved on GPU."""

    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    num_reads: int
    num_sweeps: int
    num_sweeps_per_beta: int
    beta_schedule: Optional[np.ndarray] = None
    seed: Optional[int] = None
    job_id: Optional[int] = None


class CudaKernelMock:
    """Mock CUDA kernel for testing without GPU.

    Simulates async job processing with configurable delays.
    """

    def __init__(self, processing_delay: float = 0.01):
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
            name="CudaKernelMock-Worker",
        )
        self.worker_thread.start()

    def stop(self, drain: bool = True):
        """Stop background worker."""
        if drain:
            deadline = time.time() + 30.0
            while (
                len(self.job_queue) > 0
                and time.time() < deadline
            ):
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
        **kwargs,
    ) -> None:
        """Enqueue a mock job."""
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
        """No-op for mock kernel."""

    def try_dequeue_result(self) -> Optional[Dict]:
        """Try to dequeue a mock result (non-blocking)."""
        with self.lock:
            if len(self.result_queue) == 0:
                return None
            return self.result_queue.pop(0)

    def get_kernel_state(self) -> int:
        """Get kernel state (0=RUNNING, 1=IDLE)."""
        with self.lock:
            if (
                len(self.job_queue) > 0
                or len(self.result_queue) > 0
            ):
                return 0
            return 1

    def get_samples(self, result: Dict) -> np.ndarray:
        """Extract samples from result."""
        return result['samples']

    def get_energies(self, result: Dict) -> np.ndarray:
        """Extract energies from result."""
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

            time.sleep(self.processing_delay)

            N = job['N']
            num_reads = job['num_reads']

            samples = np.random.randint(
                0, 2, size=(num_reads, N), dtype=np.int8,
            )
            samples = samples * 2 - 1

            energies = (
                np.random.randn(num_reads).astype(np.float32)
                * 100 - 14000
            )

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
