"""Unit tests for shared.packed_solution.

Mirrors ``crates/quantum-validation/src/packed.rs::tests`` so the bit-level
layout stays in lock-step between Python and Rust.
"""
from __future__ import annotations

import pytest

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
    InvalidEncodedValue,
)
from shared.packed_solution import (
    PackedSolutionLengthMismatch,
    pack_solution,
    packed_solution_byte_len,
    unpack_solution,
)


_BINARY = AllowedValueSet((-1000, 1000))
_TERNARY = AllowedValueSet((-6000, 0, 6000))


# ---------------------------------------------------------------------------
# Round-trip behaviour
# ---------------------------------------------------------------------------


def test_binary_spins_round_trip():
    spins = [-1000, 1000, -1000, 1000, 1000, -1000, -1000, 1000, 1000]
    packed = pack_solution(spins, _BINARY)
    # 9 spins × 1 bit = 9 bits → 2 bytes
    assert len(packed) == 2
    assert unpack_solution(packed, len(spins), _BINARY) == spins


def test_ternary_h_round_trip():
    values = [-6000, 0, 6000, -6000, 0, 6000, -6000]
    packed = pack_solution(values, _TERNARY)
    # 7 × 2 bits = 14 bits → 2 bytes
    assert len(packed) == 2
    assert unpack_solution(packed, len(values), _TERNARY) == values


def test_integer_range_round_trip():
    spec = AllowedValueIntegerRange(min=-2, max=2)  # 5 values → 3 bits per spin
    values = [-2000, 0, 2000, 1000, -1000]
    packed = pack_solution(values, spec)
    assert unpack_solution(packed, len(values), spec) == values


def test_continuous_range_round_trip_uses_thirty_two_bits():
    spec = AllowedValueContinuousRange(min=-6000, max=6000)
    values = [-6000, -2500, 0, 2500, 6000]
    packed = pack_solution(values, spec)
    assert len(packed) == len(values) * 4
    assert unpack_solution(packed, len(values), spec) == values


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_index_in_ternary_decoded_as_error():
    # 2 bits per value, but pattern 0b11 == 3 is out of range for a 3-entry Set.
    bad_packed = bytes([0b11_11_11_11])
    with pytest.raises(InvalidEncodedValue):
        unpack_solution(bad_packed, 4, _TERNARY)


def test_length_mismatch_reported():
    # 9 spins need 2 bytes; supply 1.
    with pytest.raises(PackedSolutionLengthMismatch):
        unpack_solution(b"\x00", 9, _BINARY)


def test_packed_solution_byte_len_matches_pack_len():
    for num_spins in (1, 7, 8, 9, 16, 100):
        spec = _BINARY
        spins = [(-1000) if i % 2 == 0 else 1000 for i in range(num_spins)]
        packed = pack_solution(spins, spec)
        assert len(packed) == packed_solution_byte_len(num_spins, spec)
