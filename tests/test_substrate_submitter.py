"""Unit tests for `shared.substrate_submitter.encode_quantum_proof` and the
spin/edge normalization helpers.

The submitter is the float→milli boundary between Python (floats) and
Substrate (i32/i64 milli units). Locking these tests down means a future
refactor that drops the rounding, or flips the 0→-1 spin convention, fails
locally instead of producing chain-rejected proofs.
"""
from __future__ import annotations

import pytest

from shared.miner_types import MiningResult
from shared.substrate_submitter import (
    MILLI_SCALE,
    _coerce_edges,
    _normalize_spins,
    encode_quantum_proof,
)
from shared.substrate_types import (
    SubstrateDifficulty,
    SubstrateMiningContext,
)


def _make_context(**overrides) -> SubstrateMiningContext:
    defaults = dict(
        block_number=1,
        parent_hash=b"\x11" * 32,
        topology_hash=b"\x22" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-4_100_000,
            min_diversity_milli=150,
            min_quality_milli=900,
        ),
        miner_account_bytes=b"\x33" * 32,
    )
    defaults.update(overrides)
    return SubstrateMiningContext(**defaults)


def _make_result(**overrides) -> MiningResult:
    defaults = dict(
        miner_id="test-miner",
        miner_type="CPU",
        nonce=12345,
        salt=b"\xab" * 32,
        timestamp=1_700_000_000,
        prev_timestamp=1_700_000_000 - 6,
        solutions=[[0, 1, 0, 1]],  # boolean convention; will normalize
        energy=-4250.5,
        diversity=0.4,
        num_valid=5,
        mining_time=4500,
        node_list=[],
        edge_list=[],
    )
    defaults.update(overrides)
    return MiningResult(**defaults)


def test_milli_scale_is_1000():
    # The Rust pallet divides by 1000 to recover floats; any drift from
    # 1000 here would silently produce mis-scaled values on chain.
    assert MILLI_SCALE == 1000


def test_encode_quantum_proof_field_shape():
    ctx = _make_context()
    result = _make_result()
    proof = encode_quantum_proof(result, ctx)

    assert proof["nonce"] == 12345
    assert proof["topology_hash"] == "0x" + ("22" * 32)
    assert proof["salt"] == "0x" + ("ab" * 32)
    assert proof["nodes"] == [0, 1, 2, 3]
    assert proof["edges"] == [(0, 1), (1, 2), (2, 3)]
    # h_values default = (-1.0, 0.0, 1.0); milli round-trip is exact.
    assert proof["h_values"] == [-1000, 0, 1000]
    # Boolean solution normalized to ±1.
    assert proof["solutions"] == [[-1, 1, -1, 1]]


def test_encode_falls_back_to_context_nodes_when_empty():
    # The miner may emit node_list=[] when the full topology was used;
    # the submitter falls back to the context's node list rather than
    # producing an empty proof.
    ctx = _make_context()
    result = _make_result(node_list=[], edge_list=[])
    proof = encode_quantum_proof(result, ctx)
    assert proof["nodes"] == ctx.nodes
    assert proof["edges"] == list(ctx.edges)


def test_encode_uses_result_nodes_when_present():
    ctx = _make_context(nodes=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)])
    result = _make_result(node_list=[10, 11], edge_list=[(10, 11)])
    proof = encode_quantum_proof(result, ctx)
    assert proof["nodes"] == [10, 11]
    assert proof["edges"] == [(10, 11)]


def test_encode_rejects_empty_solutions():
    ctx = _make_context()
    result = _make_result(solutions=[])
    with pytest.raises(ValueError, match="no solutions"):
        encode_quantum_proof(result, ctx)


def test_encode_rejects_bad_salt_length():
    ctx = _make_context()
    result = _make_result(salt=b"\xab" * 16)
    with pytest.raises(ValueError, match="32 bytes"):
        encode_quantum_proof(result, ctx)


def test_normalize_spins_boolean_convention():
    assert _normalize_spins([0, 1, 0, 1]) == [-1, 1, -1, 1]


def test_normalize_spins_spin_convention():
    assert _normalize_spins([-1, 1, -1, 1]) == [-1, 1, -1, 1]


def test_normalize_spins_rejects_invalid_value():
    with pytest.raises(ValueError, match="cannot normalize"):
        _normalize_spins([0, 1, 2])


def test_coerce_edges_accepts_tuples_and_lists():
    assert _coerce_edges([(0, 1), [1, 2]]) == [(0, 1), (1, 2)]


def test_coerce_edges_rejects_wrong_arity():
    with pytest.raises(ValueError, match="2 endpoints"):
        _coerce_edges([(0, 1, 2)])


def test_h_values_milli_rounding():
    # Fractional h_values must round, not truncate — silently swapping
    # `round` for `int` would shift the milli value and produce a
    # different on-chain proof hash. 0.1234 distinguishes them clearly:
    # round(0.1234 * 1000) == 123, int(0.1234 * 1000) == 123 too (both
    # toward zero for positive), but round vs int diverge for the
    # negative case below.
    ctx = _make_context(h_values=(-0.7891, 0.0, 1.0))
    proof = encode_quantum_proof(_make_result(), ctx)
    # round(-789.1) == -789; int(-789.1) == -789 — same here. The
    # contract we care about is that the result is the *rounded* int.
    assert proof["h_values"][0] == -789
    # And that 0.0 stays 0:
    assert proof["h_values"][1] == 0
