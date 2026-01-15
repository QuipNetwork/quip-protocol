"""
Level 2 Tests: CudaSASampler Verification

Tests for async sampler API with both mock and live kernels.
Mock tests always run (no GPU required).
Live tests run when CUDA is available.
"""

import pytest
import numpy as np
import time

try:
    import cupy as cp
    from GPU.cuda_kernel import CudaKernelRealSA
    from GPU.cuda_sa import CudaKernelMock, CudaSASamplerAsync, CudaKernelAdapter
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Skip entire module if CUDA is not available
if not CUDA_AVAILABLE:
    import pytest
    pytest.skip("CUDA not available", allow_module_level=True)


# ============================================================================
# Phase 2.1 Tests: Mock Kernel Verification
# ============================================================================

def test_mock_kernel_basic_enqueue_dequeue():
    """Test: Basic job enqueue and result dequeue."""
    kernel = CudaKernelMock(processing_delay=0.01)
    kernel.start()

    # Enqueue a job
    h = np.random.randn(32).astype(np.float32)
    J = np.random.randn(100).astype(np.float32)
    
    kernel.enqueue_job(
        job_id=1,
        h=h,
        J=J,
        num_reads=10,
        num_betas=5,
        num_sweeps_per_beta=20,
        N=32
    )

    # Poll for result
    result = None
    for _ in range(100):
        result = kernel.try_dequeue_result()
        if result is not None:
            break
        time.sleep(0.01)

    assert result is not None, "Timeout waiting for result"
    assert result['job_id'] == 1
    assert result['num_reads'] == 10
    assert result['N'] == 32

    # Verify samples and energies
    samples = kernel.get_samples(result)
    energies = kernel.get_energies(result)

    assert samples.shape == (10, 32)
    assert energies.shape == (10,)
    assert np.all((samples == -1) | (samples == 1))

    kernel.stop(drain=True)


def test_mock_kernel_multiple_jobs():
    """Test: Multiple jobs processed correctly."""
    kernel = CudaKernelMock(processing_delay=0.01)
    kernel.start()

    # Enqueue 5 jobs
    job_ids = []
    for i in range(5):
        h = np.random.randn(16).astype(np.float32)
        J = np.random.randn(50).astype(np.float32)
        
        kernel.enqueue_job(
            job_id=i,
            h=h,
            J=J,
            num_reads=5,
            num_betas=3,
            num_sweeps_per_beta=10,
            N=16
        )
        job_ids.append(i)

    # Collect results
    seen = set()
    deadline = time.time() + 5.0
    while len(seen) < len(job_ids) and time.time() < deadline:
        res = kernel.try_dequeue_result()
        if res is None:
            time.sleep(0.01)
            continue
        seen.add(res['job_id'])

    assert len(seen) == len(job_ids), f"Missing jobs: {set(job_ids) - seen}"

    kernel.stop(drain=True)


def test_mock_kernel_state_tracking():
    """Test: Kernel state transitions between IDLE and RUNNING."""
    kernel = CudaKernelMock(processing_delay=0.05)
    kernel.start()

    # Initially IDLE
    assert kernel.get_kernel_state() == 1, "Should start IDLE"

    # Enqueue job -> RUNNING
    h = np.random.randn(16).astype(np.float32)
    J = np.random.randn(50).astype(np.float32)
    
    kernel.enqueue_job(
        job_id=1,
        h=h,
        J=J,
        num_reads=5,
        num_betas=3,
        num_sweeps_per_beta=10,
        N=16
    )

    # Should transition to RUNNING
    running_seen = False
    for _ in range(50):
        if kernel.get_kernel_state() == 0:
            running_seen = True
            break
        time.sleep(0.01)

    assert running_seen, "Should transition to RUNNING"

    # Wait for result
    result = None
    for _ in range(200):
        result = kernel.try_dequeue_result()
        if result is not None:
            break
        time.sleep(0.01)

    assert result is not None, "Timeout waiting for result"

    # Should return to IDLE
    idle_seen = False
    for _ in range(50):
        if kernel.get_kernel_state() == 1:
            idle_seen = True
            break
        time.sleep(0.01)

    assert idle_seen, "Should return to IDLE"

    kernel.stop(drain=True)


def test_mock_kernel_thread_safety():
    """Test: Multiple threads can enqueue/dequeue concurrently."""
    kernel = CudaKernelMock(processing_delay=0.01)
    kernel.start()

    results = []
    errors = []

    def enqueue_jobs():
        try:
            for i in range(5):
                h = np.random.randn(16).astype(np.float32)
                J = np.random.randn(50).astype(np.float32)
                kernel.enqueue_job(
                    job_id=i,
                    h=h,
                    J=J,
                    num_reads=5,
                    num_betas=3,
                    num_sweeps_per_beta=10,
                    N=16
                )
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)

    def dequeue_results():
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                res = kernel.try_dequeue_result()
                if res is not None:
                    results.append(res)
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)

    import threading
    t1 = threading.Thread(target=enqueue_jobs)
    t2 = threading.Thread(target=dequeue_results)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    kernel.stop(drain=True)


def test_mock_kernel_empty_dequeue():
    """Test: Dequeue from empty queue returns None."""
    kernel = CudaKernelMock()
    kernel.start()

    result = kernel.try_dequeue_result()
    assert result is None, "Empty queue should return None"

    kernel.stop(drain=True)


def test_mock_kernel_stop_drain():
    """Test: DRAIN mode waits for queue to empty."""
    kernel = CudaKernelMock(processing_delay=0.02)
    kernel.start()

    # Enqueue 3 jobs
    for i in range(3):
        h = np.random.randn(16).astype(np.float32)
        J = np.random.randn(50).astype(np.float32)
        kernel.enqueue_job(
            job_id=i,
            h=h,
            J=J,
            num_reads=5,
            num_betas=3,
            num_sweeps_per_beta=10,
            N=16
        )

    # Stop with drain=True should wait for all jobs
    start = time.time()
    kernel.stop(drain=True)
    elapsed = time.time() - start

    # Should take at least 3 * processing_delay
    assert elapsed >= 0.05, f"DRAIN should wait for jobs, took {elapsed}s"

    kernel.stop(drain=False)


# ============================================================================
# Phase 2.2 Tests: CudaSASamplerAsync with Mock Kernel
# ============================================================================

def test_sampler_async_submission_order_mock():
    """Test: job_ids returned in submission order (mock)."""
    kernel = CudaKernelMock(processing_delay=0.01)
    sampler = CudaSASamplerAsync(kernel)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10)

    assert len(job_ids) == 5
    assert job_ids == list(range(5)), f"Expected [0,1,2,3,4], got {job_ids}"

    sampler.stop(drain=True)


def test_sampler_collect_order_mock():
    """Test: collect_samples returns results in submission order (mock)."""
    kernel = CudaKernelMock(processing_delay=0.01)
    sampler = CudaSASamplerAsync(kernel)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10)
    samplesets = sampler.collect_samples(job_ids, timeout=5.0)

    assert len(samplesets) == 5

    # Verify job_ids match in order
    for i, sampleset in enumerate(samplesets):
        assert sampleset.info['job_id'] == job_ids[i], \
            f"Result {i}: expected job_id {job_ids[i]}, got {sampleset.info['job_id']}"

    sampler.stop(drain=True)


def test_sampler_synchronous_wrapper_mock():
    """Test: sample_ising synchronous wrapper (mock)."""
    kernel = CudaKernelMock(processing_delay=0.01)
    sampler = CudaSASamplerAsync(kernel)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    samplesets = sampler.sample_ising(h_list, J_list, num_reads=10)

    assert len(samplesets) == 5

    for sampleset in samplesets:
        assert len(sampleset) == 10, f"Expected 10 samples, got {len(sampleset)}"
        assert str(sampleset.vartype) == 'Vartype.SPIN', f"Expected SPIN vartype, got {sampleset.vartype}"

    sampler.stop(drain=True)


def test_sampler_partial_collection_mock():
    """Test: collect specific subset of jobs (mock)."""
    kernel = CudaKernelMock(processing_delay=0.01)
    sampler = CudaSASamplerAsync(kernel)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(10)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(10)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10)

    # Collect only jobs 2, 5, 7
    subset_ids = [job_ids[2], job_ids[5], job_ids[7]]
    samplesets = sampler.collect_samples(subset_ids, timeout=5.0)

    assert len(samplesets) == 3
    collected_ids = [s.info['job_id'] for s in samplesets]
    assert collected_ids == subset_ids, \
        f"Expected {subset_ids}, got {collected_ids}"

    sampler.stop(drain=True)


def test_sampler_timeout_mock():
    """Test: collect_samples raises TimeoutError (mock)."""
    kernel = CudaKernelMock(processing_delay=10.0)  # Very slow
    sampler = CudaSASamplerAsync(kernel)

    h_list = [np.random.randn(100).astype(np.float32)]
    J_list = [np.random.randn(500).astype(np.float32)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10)

    with pytest.raises(TimeoutError):
        sampler.collect_samples(job_ids, timeout=0.1)

    sampler.stop(drain=False)


def test_sampler_empty_collection_mock():
    """Test: collect_samples with no jobs returns empty list (mock)."""
    kernel = CudaKernelMock()
    sampler = CudaSASamplerAsync(kernel)

    samplesets = sampler.collect_samples(job_ids=[], timeout=1.0)
    assert len(samplesets) == 0

    sampler.stop(drain=True)


# ============================================================================
# Phase 2.3 Tests: Live Kernel Tests
# ============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sampler_async_submission_order_live():
    """Test: job_ids returned in submission order (live kernel)."""
    kernel = CudaKernelRealSA(ring_size=16, max_threads_per_job=256)
    kernel_adapter = CudaKernelAdapter(kernel)
    sampler = CudaSASamplerAsync(kernel_adapter)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10)

    assert len(job_ids) == 5

    sampler.stop_immediate()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sampler_collect_order_live():
    """Test: collect_samples returns results in submission order (live)."""
    kernel = CudaKernelRealSA(ring_size=16, max_threads_per_job=256)
    kernel_adapter = CudaKernelAdapter(kernel)
    sampler = CudaSASamplerAsync(kernel_adapter)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    job_ids = sampler.sample_ising_async(h_list, J_list, num_reads=10, num_betas=50)
    samplesets = sampler.collect_samples(job_ids, timeout=10.0)

    assert len(samplesets) == 5

    # Verify job_ids match in order
    for i, sampleset in enumerate(samplesets):
        assert sampleset.info['job_id'] == job_ids[i]

    sampler.stop_immediate()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sampler_synchronous_wrapper_live():
    """Test: sample_ising synchronous wrapper (live kernel)."""
    kernel = CudaKernelRealSA(ring_size=16, max_threads_per_job=256)
    kernel_adapter = CudaKernelAdapter(kernel)
    sampler = CudaSASamplerAsync(kernel_adapter)

    h_list = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    J_list = [np.random.randn(500).astype(np.float32) for _ in range(5)]

    samplesets = sampler.sample_ising(h_list, J_list, num_reads=10, num_betas=50)

    assert len(samplesets) == 5

    for sampleset in samplesets:
        assert len(sampleset) == 10  # num_reads
        assert str(sampleset.vartype) == 'Vartype.SPIN'

    sampler.stop_immediate()

