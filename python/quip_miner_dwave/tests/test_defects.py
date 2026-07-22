"""Defect clamping unit tests."""
from quip_miner_dwave.defects import (
    DefectInfo,
    clamp_fixed_variables,
    prepare_problem,
    reconstruct_sample,
)


def test_no_defects_passthrough():
    h = {0: 1.0, 1: -1.0}
    j = {(0, 1): 0.5}
    h2, j2, info = prepare_problem(h, j)
    assert h2 == h
    assert j2 == j
    assert info is None


def test_clamp_and_reconstruct_energy():
    h = {0: 1.0, 1: -0.5, 2: 0.25}
    j = {(0, 1): 0.5, (1, 2): -0.3}
    h_r, j_r, fixed, offset, removed = clamp_fixed_variables(
        h, j, nonce_seed=42, defective_qubits=[1], defective_edges=set()
    )
    assert 1 not in h_r
    assert (0, 1) not in j_r
    assert (1, 2) not in j_r
    assert 1 in fixed
    # Reduced problem on remaining qubits
    reduced_sample = {0: 1, 2: -1}
    # Dummy reduced energy (QPU would return this)
    e_red = h_r.get(0, 0) * 1 + h_r.get(2, 0) * (-1)
    full, e_full = reconstruct_sample(
        reduced_sample,
        e_red,
        DefectInfo(fixed, offset, removed),
    )
    assert 1 in full
    # Full energy should match direct evaluation
    e_direct = sum(h[i] * full[i] for i in full)
    for (u, v), val in j.items():
        e_direct += val * full[u] * full[v]
    assert abs(e_full - e_direct) < 1e-9
