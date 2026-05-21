"""Unit tests for shared.allowed_value_spec.

Mirrors the cargo-side tests in
``crates/quantum-validation/src/puzzle_spec.rs::tests`` so the two
implementations stay in lock-step on bit-widths, decode behavior, and
canonical-bytes invariants.
"""
from __future__ import annotations

import pytest

from shared.allowed_value_spec import (
    MILLI_SCALE,
    AllowedValueContinuousRange,
    AllowedValueIntegerRange,
    AllowedValueSet,
    EmptyAllowedValues,
    EncodingTooWide,
    InvalidEncodedValue,
    bits_per_value,
    canonical_bytes,
    decode_value,
    encode_value,
)


# ---------------------------------------------------------------------------
# bits_per_value
# ---------------------------------------------------------------------------


def test_binary_set_uses_one_bit():
    assert bits_per_value(AllowedValueSet((-1000, 1000))) == 1


def test_ternary_set_uses_two_bits():
    assert bits_per_value(AllowedValueSet((-6000, 0, 6000))) == 2


def test_integer_range_uses_minimum_bits():
    # 13 values -> 4 bits
    assert bits_per_value(AllowedValueIntegerRange(min=-6, max=6)) == 4


def test_continuous_range_uses_thirty_two_bits():
    assert (
        bits_per_value(AllowedValueContinuousRange(min=-6000, max=6000)) == 32
    )


def test_empty_set_rejected():
    with pytest.raises(EmptyAllowedValues):
        bits_per_value(AllowedValueSet(()))


def test_inverted_integer_range_rejected():
    with pytest.raises(EmptyAllowedValues):
        bits_per_value(AllowedValueIntegerRange(min=5, max=2))


def test_inverted_continuous_range_rejected():
    with pytest.raises(EmptyAllowedValues):
        bits_per_value(AllowedValueContinuousRange(min=5, max=2))


def test_wide_set_rejected():
    # 2^9 = 512 distinct values -> 9 bits, exceeds MAX_INDEXED_BITS=8
    with pytest.raises(EncodingTooWide):
        bits_per_value(AllowedValueSet(tuple(range(512))))


# ---------------------------------------------------------------------------
# decode_value
# ---------------------------------------------------------------------------


def test_decode_set_recovers_values():
    spec = AllowedValueSet((-6000, 0, 6000))
    assert decode_value(spec, 0) == -6000
    assert decode_value(spec, 1) == 0
    assert decode_value(spec, 2) == 6000
    with pytest.raises(InvalidEncodedValue):
        decode_value(spec, 3)


def test_decode_integer_range_scales_by_milli():
    spec = AllowedValueIntegerRange(min=-6, max=6)
    assert decode_value(spec, 0) == -6000
    assert decode_value(spec, 6) == 0
    assert decode_value(spec, 12) == 6000
    with pytest.raises(InvalidEncodedValue):
        decode_value(spec, 13)


def test_decode_continuous_range_round_trip():
    spec = AllowedValueContinuousRange(min=-MILLI_SCALE, max=MILLI_SCALE)
    # Positive within range — raw = value.
    assert decode_value(spec, 500) == 500
    # Negative within range — raw is the u32 two's-complement of the i32.
    assert decode_value(spec, (-500) & 0xFFFFFFFF) == -500
    with pytest.raises(InvalidEncodedValue):
        # Out of [-1000, 1000].
        decode_value(spec, 2000)


# ---------------------------------------------------------------------------
# encode_value (used by pack_solution)
# ---------------------------------------------------------------------------


def test_encode_value_round_trips_set():
    spec = AllowedValueSet((-1000, 1000))
    assert encode_value(spec, -1000) == 0
    assert encode_value(spec, 1000) == 1
    with pytest.raises(InvalidEncodedValue):
        encode_value(spec, 0)


def test_encode_value_integer_range_requires_multiple_of_milli():
    spec = AllowedValueIntegerRange(min=-3, max=3)
    assert encode_value(spec, -3 * MILLI_SCALE) == 0
    assert encode_value(spec, 0) == 3
    with pytest.raises(InvalidEncodedValue):
        encode_value(spec, 500)  # not a multiple of MILLI_SCALE


def test_encode_value_continuous_range_is_two_complement():
    spec = AllowedValueContinuousRange(min=-MILLI_SCALE, max=MILLI_SCALE)
    assert encode_value(spec, 500) == 500
    assert encode_value(spec, -500) == (-500) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# canonical_bytes
# ---------------------------------------------------------------------------


def test_canonical_bytes_are_order_independent_for_sets():
    a = AllowedValueSet((6000, -6000, 0))
    b = AllowedValueSet((0, 6000, -6000))
    assert canonical_bytes(a) == canonical_bytes(b)


def test_canonical_bytes_distinguish_variants():
    s = AllowedValueSet((0,))
    ir = AllowedValueIntegerRange(min=0, max=0)
    cr = AllowedValueContinuousRange(min=0, max=0)
    assert canonical_bytes(s) != canonical_bytes(ir)
    assert canonical_bytes(s) != canonical_bytes(cr)
    assert canonical_bytes(ir) != canonical_bytes(cr)


def test_canonical_bytes_contain_discriminant():
    assert canonical_bytes(AllowedValueSet((-1000, 1000)))[0] == 0x00
    assert canonical_bytes(AllowedValueIntegerRange(min=-1, max=1))[0] == 0x01
    assert canonical_bytes(AllowedValueContinuousRange(min=-1, max=1))[0] == 0x02
