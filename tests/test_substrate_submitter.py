"""Unit tests for `shared.substrate_submitter.encode_quantum_proof` and the
spin normalization helper.

The submitter is the float-to-milli boundary between Python (floats) and
Substrate (bit-packed i8 spins as bytes). Post-MR-!20 the proof no longer
carries nodes/edges/h_values — those are looked up on chain from the
registered topology — so the assertions focus on:

  - nonce encodes as a U256 int; salt as a 32-byte array
  - solutions are bit-packed against the registered spin spec
  - the proof dict no longer carries nodes/edges/h_values
"""
from __future__ import annotations

import pytest

from shared.allowed_value_spec import AllowedValueSet
from shared.miner_types import MiningResult
from shared.packed_solution import unpack_solution
from shared.substrate_submitter import (
    MILLI_SCALE,
    _normalize_spins,
    encode_quantum_proof,
)
from substrate.types import (
    SubstrateDifficulty,
    SubstrateMiningContext,
)


_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


def _make_context(**overrides) -> SubstrateMiningContext:
    defaults = dict(
        last_proof_block_hash=b"\x11" * 32,
        topology_hash=b"\x22" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(
            min_solutions=5,
            max_energy_milli=-4_100_000,
            min_diversity_milli=150,
        ),
        miner_account_bytes=b"\x33" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
    )
    defaults.update(overrides)
    return SubstrateMiningContext(**defaults)


def _make_result(**overrides) -> MiningResult:
    defaults = dict(
        miner_id="test-miner",
        miner_type="CPU",
        nonce=bytes.fromhex("aa" * 32),
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

    assert proof["topology_hash"] == "0x" + ("22" * 32)
    # nonce encodes as a U256 (big-endian int); salt as [u8; 32] byte-int list.
    assert proof["nonce"] == int.from_bytes(result.nonce, "big")
    assert bytes(proof["salt"]) == result.salt

    # The post-MR-!20 proof carries ONLY topology_hash, nonce, salt, solutions.
    assert set(proof.keys()) == {"topology_hash", "nonce", "salt", "solutions"}

    # One submitted solution, bit-packed against the binary spin spec
    # (1 bit per spin, 4 spins → 1 byte).
    solutions, = proof["solutions"]
    assert len(solutions) == 1
    inner_packed_tuple = solutions[0]
    packed_bytes, = inner_packed_tuple
    assert len(packed_bytes) == 1  # 4 spins / 8 bits-per-byte rounded up
    # Round-trip through unpack_solution to verify the ±1 ordering matches
    # the input.
    decoded = unpack_solution(bytes(packed_bytes), 4, ctx.allowed_spin_values)
    assert decoded == [-1000, 1000, -1000, 1000]


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


def test_encode_rejects_bad_nonce_shape():
    ctx = _make_context()
    # Non-bytes nonce no longer accepted at the submission boundary.
    with pytest.raises(ValueError, match="32-byte"):
        encode_quantum_proof(_make_result(nonce=b"\x01" * 16), ctx)


def test_encode_rejects_solution_length_mismatch():
    ctx = _make_context()  # 4 nodes
    result = _make_result(solutions=[[1, -1, 1]])  # only 3 spins
    with pytest.raises(ValueError, match="topology node count"):
        encode_quantum_proof(result, ctx)


def test_normalize_spins_boolean_convention():
    assert _normalize_spins([0, 1, 0, 1]) == [-1, 1, -1, 1]


def test_normalize_spins_spin_convention():
    assert _normalize_spins([-1, 1, -1, 1]) == [-1, 1, -1, 1]


def test_normalize_spins_rejects_invalid_value():
    with pytest.raises(ValueError, match="cannot normalize"):
        _normalize_spins([0, 1, 2])
