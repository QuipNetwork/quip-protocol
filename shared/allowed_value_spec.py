"""Allowed-value sampling specs for nonce-seeded puzzle generation.

Python mirror of ``crates/quantum-validation/src/puzzle_spec.rs`` in
``quip-protocol-rs``. ``AllowedValueSpec`` describes how a deterministic RNG
picks per-node h fields, per-edge j couplings, and per-spin solution values.
The variant determines both the sampling distribution and the on-chain
bit-width used when a value is encoded in a SCALE payload.

All milli-precision values are i32 (divide by ``MILLI_SCALE`` to read as
float). ``MILLI_SCALE`` is defined here and re-exported from
``substrate.submitter`` so existing submitter callers don't need a second
import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Tuple, Union


# Milli-precision scale factor. Mirrors quantum_validation::fixed::MILLI_SCALE.
MILLI_SCALE: int = 1000

# Maximum supported bit-width for an indexed encoding (Set or IntegerRange).
# Values wider than this fall under ContinuousRange (raw 32-bit).
MAX_INDEXED_BITS: int = 8

# i32 range used to clamp encoded values and detect overflow.
_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_U32_MASK = (1 << 32) - 1


class AllowedValueSpecError(ValueError):
    """Base class for AllowedValueSpec encode/decode errors.

    Subclasses mirror the Rust ``ValidationError`` variants so callers can
    distinguish empty specs, encoding too wide, and invalid encoded values
    when mapping back to pallet error codes.
    """


class EmptyAllowedValues(AllowedValueSpecError):
    """Spec is empty (zero-length Set) or has inverted bounds (max < min)."""


class EncodingTooWide(AllowedValueSpecError):
    """Indexed encoding would require more than MAX_INDEXED_BITS bits."""

    def __init__(self, bits: int) -> None:
        super().__init__(
            f"encoding requires {bits} bits per value, exceeds protocol maximum"
        )
        self.bits = bits


class InvalidEncodedValue(AllowedValueSpecError):
    """Encoded value does not map to any allowed entry under this spec."""

    def __init__(self, raw: int) -> None:
        super().__init__(
            f"encoded value {raw} is not a valid representation under this spec"
        )
        self.raw = raw


@dataclass(frozen=True)
class AllowedValueSet:
    """Uniform draw from an explicit set of milli-precision values.

    Bit-width is ``ceil(log2(len(values)))`` (minimum 1 bit). The Set must be
    non-empty. Bit-pattern ``i`` decodes to ``values[i]``.
    """

    values: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(int(v) for v in self.values))
        for v in self.values:
            if not (_I32_MIN <= v <= _I32_MAX):
                raise ValueError(
                    f"AllowedValueSet value {v} does not fit in i32"
                )


@dataclass(frozen=True)
class AllowedValueIntegerRange:
    """Uniform integer draw from ``[min, max]`` whole integers.

    Sampled values are scaled by ``MILLI_SCALE``. Bit-width is
    ``ceil(log2(max - min + 1))`` (minimum 1 bit).
    """

    min: int
    max: int

    def __post_init__(self) -> None:
        for name, v in (("min", self.min), ("max", self.max)):
            if not (_I32_MIN <= v <= _I32_MAX):
                raise ValueError(f"{name}={v} does not fit in i32")


@dataclass(frozen=True)
class AllowedValueContinuousRange:
    """Uniform milli-precision draw from ``[min, max]`` in i32 units.

    Bit-width is always 32; each value is encoded as a raw big-endian i32.
    """

    min: int
    max: int

    def __post_init__(self) -> None:
        for name, v in (("min", self.min), ("max", self.max)):
            if not (_I32_MIN <= v <= _I32_MAX):
                raise ValueError(f"{name}={v} does not fit in i32")


AllowedValueSpec = Union[
    AllowedValueSet, AllowedValueIntegerRange, AllowedValueContinuousRange
]


# ---------------------------------------------------------------------------
# Bit-width / sampling / encode / decode primitives
# ---------------------------------------------------------------------------


def _bits_for_count(count: int) -> int:
    """Minimum bits required to encode ``count`` distinct values (>= 1)."""
    if count <= 1:
        return 1
    return (count - 1).bit_length()


def bits_per_value(spec: AllowedValueSpec) -> int:
    """Bit-width per encoded value for the given spec.

    Raises :class:`EmptyAllowedValues` for empty Sets or inverted ranges, and
    :class:`EncodingTooWide` if an indexed variant would need more than
    ``MAX_INDEXED_BITS`` bits.
    """
    if isinstance(spec, AllowedValueSet):
        if not spec.values:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        bits = _bits_for_count(len(spec.values))
        _check_indexed_bits(bits)
        return bits
    if isinstance(spec, AllowedValueIntegerRange):
        if spec.max < spec.min:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        span = spec.max - spec.min + 1
        bits = _bits_for_count(span)
        _check_indexed_bits(bits)
        return bits
    if isinstance(spec, AllowedValueContinuousRange):
        if spec.max < spec.min:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        return 32
    _unknown_variant(spec)


def _check_indexed_bits(bits: int) -> None:
    if bits > MAX_INDEXED_BITS:
        raise EncodingTooWide(bits)


def _unknown_variant(spec: AllowedValueSpec) -> NoReturn:
    raise TypeError(f"unknown AllowedValueSpec variant: {type(spec).__name__}")


def sample(spec: AllowedValueSpec, rng) -> int:
    """Draw one milli-value from the spec using ``rng.next_u32()``.

    Mirrors ``AllowedValueSpec::sample`` in Rust. ``rng`` must expose a
    ``next_u32()`` method (e.g. :class:`shared.chacha8.ChaCha8Rng`).
    """
    if isinstance(spec, AllowedValueSet):
        if not spec.values:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        index = rng.next_u32() % len(spec.values)
        return int(spec.values[index])
    if isinstance(spec, AllowedValueIntegerRange):
        if spec.max < spec.min:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        span = spec.max - spec.min + 1
        offset = rng.next_u32() % span
        return decode_value(spec, offset)
    if isinstance(spec, AllowedValueContinuousRange):
        if spec.max < spec.min:
            raise EmptyAllowedValues("allowed value spec is empty or inverted")
        span = spec.max - spec.min + 1
        offset = rng.next_u32() % span
        return int(spec.min + offset)
    _unknown_variant(spec)


def decode_value(spec: AllowedValueSpec, raw: int) -> int:
    """Decode a packed integer to its milli-value interpretation.

    ``raw`` is the integer read from a bit-packed payload. For Set and
    IntegerRange it is an index/offset; for ContinuousRange it is the raw
    32-bit value reinterpreted as i32.
    """
    if isinstance(spec, AllowedValueSet):
        if raw < 0 or raw >= len(spec.values):
            raise InvalidEncodedValue(raw)
        return int(spec.values[raw])
    if isinstance(spec, AllowedValueIntegerRange):
        span = spec.max - spec.min + 1
        if raw < 0 or raw >= span:
            raise InvalidEncodedValue(raw)
        value = (spec.min + raw) * MILLI_SCALE
        if not (_I32_MIN <= value <= _I32_MAX):
            raise OverflowError(
                f"decoded value {value} does not fit in i32 for IntegerRange"
            )
        return int(value)
    if isinstance(spec, AllowedValueContinuousRange):
        # raw is a u32 reinterpreted as i32 (matches Rust `value as i32`).
        value = raw if raw < (1 << 31) else raw - (1 << 32)
        if value < spec.min or value > spec.max:
            raise InvalidEncodedValue(raw)
        return int(value)
    _unknown_variant(spec)


def encode_value(spec: AllowedValueSpec, value: int) -> int:
    """Inverse of :func:`decode_value` — produces the raw packed integer.

    Used by :mod:`shared.packed_solution` when miners pack outgoing solutions.
    """
    if isinstance(spec, AllowedValueSet):
        try:
            return spec.values.index(int(value))
        except ValueError as exc:
            raise InvalidEncodedValue(value) from exc
    if isinstance(spec, AllowedValueIntegerRange):
        if value % MILLI_SCALE != 0:
            raise InvalidEncodedValue(value)
        whole = value // MILLI_SCALE
        if whole < spec.min or whole > spec.max:
            raise InvalidEncodedValue(value)
        return int(whole - spec.min)
    if isinstance(spec, AllowedValueContinuousRange):
        if value < spec.min or value > spec.max:
            raise InvalidEncodedValue(value)
        # Encode i32 as u32 two's-complement (matches `value as u32` in Rust).
        return int(value) & _U32_MASK
    _unknown_variant(spec)


# ---------------------------------------------------------------------------
# Canonical hash bytes (used by topology_hash)
# ---------------------------------------------------------------------------


def canonical_bytes(spec: AllowedValueSpec) -> bytes:
    """Order-stable byte representation used by :func:`topology_hash`.

    Mirrors ``AllowedValueSpec::canonical_bytes`` in Rust. Two Sets that
    differ only by element ordering produce identical canonical bytes; two
    variants with different discriminants always produce distinct bytes.
    """
    if isinstance(spec, AllowedValueSet):
        out = bytearray([0x00])
        for v in sorted(spec.values):
            out += int(v).to_bytes(4, "big", signed=True)
        return bytes(out)
    if isinstance(spec, AllowedValueIntegerRange):
        return (
            b"\x01"
            + int(spec.min).to_bytes(4, "big", signed=True)
            + int(spec.max).to_bytes(4, "big", signed=True)
        )
    if isinstance(spec, AllowedValueContinuousRange):
        return (
            b"\x02"
            + int(spec.min).to_bytes(4, "big", signed=True)
            + int(spec.max).to_bytes(4, "big", signed=True)
        )
    _unknown_variant(spec)


# ---------------------------------------------------------------------------
# substrate-interface compose-friendly serialization
# ---------------------------------------------------------------------------


def scale_dict(spec: AllowedValueSpec) -> dict:
    """Build the substrate-interface call-param dict for an AllowedValueSpec.

    The Rust enum is registered with metadata as a SCALE enum; substrate-
    interface accepts ``{"VariantName": payload}`` for SCALE enums. The
    ``Set`` payload is a ``BoundedVec<i32>`` which substrate-interface wants
    wrapped in a 1-tuple — same rule as ``BoundedVec`` elsewhere in the
    codebase (see ``substrate/submitter.py``).
    """
    if isinstance(spec, AllowedValueSet):
        return {"Set": (list(int(v) for v in spec.values),)}
    if isinstance(spec, AllowedValueIntegerRange):
        return {"IntegerRange": {"min": int(spec.min), "max": int(spec.max)}}
    if isinstance(spec, AllowedValueContinuousRange):
        return {"ContinuousRange": {"min": int(spec.min), "max": int(spec.max)}}
    _unknown_variant(spec)


__all__ = [
    "MILLI_SCALE",
    "MAX_INDEXED_BITS",
    "AllowedValueSet",
    "AllowedValueIntegerRange",
    "AllowedValueContinuousRange",
    "AllowedValueSpec",
    "AllowedValueSpecError",
    "EmptyAllowedValues",
    "EncodingTooWide",
    "InvalidEncodedValue",
    "bits_per_value",
    "sample",
    "decode_value",
    "encode_value",
    "canonical_bytes",
    "scale_dict",
]
