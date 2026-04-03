"""Regression tests: GPU-reported energy must match Python recomputation.

The mining pipeline trusts GPU-reported energies during mining, then
recomputes during build_block(). If they disagree, the block is rejected.

Tests isolate the discrepancy layer by layer:
  A) Single model, synchronous — basic energy correctness
  B) Multiple models, streaming — slot recycling
  C) Known-answer on small graph — validates kernel math
  D) Full pipeline with IsingFeeder — subprocess boundary + evaluate/validate

Run with:
    pytest tests/test_energy_mismatch.py -v -s
"""
import time

import numpy as np
import pytest

try:
    import cupy as cp
    from GPU.cuda_sa import CudaSASampler
    from shared.quantum_proof_of_work import (
        energy_of_solution,
        generate_ising_model_from_nonce,
    )
    from shared.ising_model import IsingModel
    from dwave_topologies import DEFAULT_TOPOLOGY
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False


@pytest.fixture(autouse=True, scope="function")
def cleanup_cuda():
    """Reset CUDA device between tests."""
    if CUDA_AVAILABLE:
        try:
            cp.cuda.Device().synchronize()
            time.sleep(0.1)
        except Exception:
            pass
    yield
    if CUDA_AVAILABLE:
        try:
            time.sleep(0.2)
            cp.cuda.Device().synchronize()
        except Exception:
            pass


def _compare_energies(sampleset, h, J, nodes):
    """Compare GPU-reported energies with Python recomputation.

    Returns list of (idx, gpu_energy, python_energy, abs_diff) for mismatches.
    """
    gpu_energies = sampleset.record.energy
    mismatches = []
    for idx in range(len(gpu_energies)):
        solution = list(sampleset.record.sample[idx])
        gpu_e = float(gpu_energies[idx])
        py_e = energy_of_solution(solution, h, J, nodes)
        if gpu_e != py_e:
            mismatches.append((idx, gpu_e, py_e, abs(gpu_e - py_e)))
    return mismatches


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_single_model_energy_match():
    """Single Ising model on full topology — basic energy correctness.

    Isolates CSR loading, kernel energy tracking, and solution unpacking.
    """
    topology = DEFAULT_TOPOLOGY
    nodes = list(topology.graph.nodes())
    edges = list(topology.graph.edges())

    h, J = generate_ising_model_from_nonce(12345, nodes, edges)

    sampler = CudaSASampler(topology=topology, max_sms=1)
    try:
        ss = sampler.sample_ising(
            [h], [J], num_reads=32, num_sweeps=1000,
        )[0]
        mismatches = _compare_energies(ss, h, J, nodes)
        assert len(mismatches) == 0, (
            f"{len(mismatches)} samples have mismatched energies. "
            f"Worst diff: {max(m[3] for m in mismatches):.0f}"
        )
    finally:
        sampler.signal_exit(wait=True)
        sampler.close()


@pytest.mark.timeout(300)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_streaming_slot_recycling():
    """Multiple models via streaming — tests 3-slot rotation correctness."""
    topology = DEFAULT_TOPOLOGY
    nodes = list(topology.graph.nodes())
    edges = list(topology.graph.edges())

    num_models = 12
    models = []
    for i in range(num_models):
        nonce = 10000 + i
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        models.append(IsingModel(h=h, J=J, nonce=nonce, salt=b'\x00' * 32))

    sampler = CudaSASampler(topology=topology, max_sms=2)
    try:
        total_mismatches = 0
        total_samples = 0
        stream = sampler.sample_ising_streaming(
            iter(models),
            num_reads=32, num_sweeps=1000,
            num_kernels=2, poll_timeout=120.0,
        )
        for model, ss in stream:
            h, J = generate_ising_model_from_nonce(model.nonce, nodes, edges)
            mismatches = _compare_energies(ss, h, J, nodes)
            total_samples += len(ss.record.energy)
            total_mismatches += len(mismatches)
    finally:
        sampler.signal_exit(wait=True)
        sampler.close()

    assert total_mismatches == 0, (
        f"{total_mismatches}/{total_samples} samples mismatched "
        f"across {num_models} models"
    )


@pytest.mark.timeout(300)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_full_pipeline_with_feeder():
    """Full production path: IsingFeeder -> GPU -> evaluate -> validate.

    Exercises the exact code path that produces debug/ mismatch files:
    IsingFeeder (subprocess) -> GPU streaming -> evaluate_sampleset
    -> compute_derived_fields recomputation.
    """
    from shared.block import BlockRequirements, QuantumProof
    from shared.ising_feeder import IsingFeeder
    from shared.quantum_proof_of_work import evaluate_sampleset

    topology = DEFAULT_TOPOLOGY
    nodes = list(topology.graph.nodes())
    edges = list(topology.graph.edges())

    requirements = BlockRequirements(
        difficulty_energy=-200.0,
        min_diversity=0.1,
        min_solutions=5,
        timeout_to_difficulty_adjustment_decay=300,
    )

    feeder = IsingFeeder(
        prev_hash=b'\xab' * 32,
        miner_id="test-gpu-0",
        cur_index=38,
        nodes=nodes,
        edges=edges,
        buffer_size=6,
        max_workers=2,
        seed=42,
    )

    sampler = CudaSASampler(topology=topology, max_sms=2)
    try:
        models = [feeder.pop_blocking() for _ in range(10)]
        stream = sampler.sample_ising_streaming(
            iter(models),
            num_reads=64, num_sweeps=2048,
            num_kernels=2, poll_timeout=120.0,
        )

        mismatches_found = 0
        start = time.time()
        for model, ss in stream:
            result = evaluate_sampleset(
                ss, requirements, nodes, edges,
                model.nonce, model.salt,
                prev_timestamp=int(time.time()) - 100,
                start_time=start,
                miner_id="test-gpu-0",
                miner_type="GPU-CUDA",
            )
            if result is None:
                continue

            proof = QuantumProof(
                nonce=result.nonce,
                salt=result.salt,
                nodes=result.variable_order or result.node_list,
                edges=result.edge_list,
                solutions=result.solutions,
                mining_time=result.mining_time,
                energy=result.energy,
                diversity=result.diversity,
                num_valid_solutions=result.num_valid,
            )
            proof.compute_derived_fields()

            if proof.energy != result.energy:
                mismatches_found += 1

        assert mismatches_found == 0, (
            f"{mismatches_found} models had miner vs recomputed energy mismatch"
        )
    finally:
        feeder.stop()
        sampler.signal_exit(wait=True)
        sampler.close()
