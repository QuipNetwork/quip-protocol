"""Bit-packed solution payloads for QuantumProof submissions.

Python mirror of ``crates/quantum-validation/src/packed.rs`` in
``quip-protocol-rs``. The submitted Ising solution is a vector of spins; with
the default binary spin spec the natural wire format is one bit per spin,
shrinking each payload 8x versus the prior ``Vec<i8>`` format.

Indexed (Set / IntegerRange) layouts pack LSB-first within each byte and
walk upward across byte boundaries. The continuous variant writes each spin
as a 4-byte big-endian i32. Decode is the inverse and validates byte length
exactly so a truncated transport fails fast instead of silently zero-padding.
"""
from __future__ import annotations

from typing import List, Sequence

from shared.allowed_value_spec import (
    AllowedValueContinuousRange,
    AllowedValueSpec,
    InvalidEncodedValue,
    bits_per_value,
    decode_value,
    encode_value,
)


class PackedSolutionLengthMismatch(ValueError):
    """The packed payload byte length disagrees with the spec + spin count."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"packed solution length mismatch: expected {expected} bytes, "
            f"got {actual}"
        )
        self.expected = expected
        self.actual = actual


def packed_solution_byte_len(num_spins: int, spec: AllowedValueSpec) -> int:
    """Bytes required to pack ``num_spins`` values under ``spec``."""
    bits = bits_per_value(spec)
    return (num_spins * bits + 7) // 8


def pack_solution(spins: Sequence[int], spec: AllowedValueSpec) -> bytes:
    """Pack a sequence of milli-precision spins into the wire format.

    Mirrors ``packed::pack_solution`` in Rust. For 32-bit (continuous)
    layouts, each spin occupies 4 bytes in big-endian order. For indexed
    layouts, the raw bit pattern of each spin (from
    :func:`shared.allowed_value_spec.encode_value`) is written
    little-endian within the byte sequence, starting at the LSB of byte 0.
    """
    bits = bits_per_value(spec)
    byte_len = packed_solution_byte_len(len(spins), spec)
    out = bytearray(byte_len)

    if isinstance(spec, AllowedValueContinuousRange) or bits == 32:
        for i, value in enumerate(spins):
            raw = encode_value(spec, int(value))
            out[i * 4 : i * 4 + 4] = raw.to_bytes(4, "big", signed=False)
        return bytes(out)

    for spin_index, value in enumerate(spins):
        raw = encode_value(spec, int(value))
        bit_offset = spin_index * bits
        byte_index = bit_offset // 8
        intra_byte = bit_offset % 8

        shifted = raw << intra_byte
        out[byte_index] |= shifted & 0xFF
        if byte_index + 1 < byte_len:
            out[byte_index + 1] |= (shifted >> 8) & 0xFF
        if byte_index + 2 < byte_len and intra_byte + bits > 16:
            out[byte_index + 2] |= (shifted >> 16) & 0xFF
    return bytes(out)


def unpack_solution(
    packed: bytes, num_spins: int, spec: AllowedValueSpec
) -> List[int]:
    """Decode a bit-packed payload into ``num_spins`` milli-values.

    Raises :class:`PackedSolutionLengthMismatch` if the buffer size does
    not match what the spec implies for ``num_spins``, and
    :class:`shared.allowed_value_spec.InvalidEncodedValue` if any decoded bit
    pattern is out of range.
    """
    bits = bits_per_value(spec)
    expected = packed_solution_byte_len(num_spins, spec)
    if len(packed) != expected:
        raise PackedSolutionLengthMismatch(expected, len(packed))

    if isinstance(spec, AllowedValueContinuousRange) or bits == 32:
        return _decode_continuous(packed, num_spins, spec)

    return _decode_indexed(packed, num_spins, bits, spec)


def _decode_indexed(
    packed: bytes, num_spins: int, bits: int, spec: AllowedValueSpec
) -> List[int]:
    mask = (1 << bits) - 1 if bits < 32 else 0xFFFFFFFF
    out: List[int] = []
    for spin_index in range(num_spins):
        bit_offset = spin_index * bits
        byte_index = bit_offset // 8
        intra_byte = bit_offset % 8

        # Two-byte window suffices for bits <= 8 (intra_byte <= 7, span <= 15).
        b0 = packed[byte_index]
        b1 = packed[byte_index + 1] if byte_index + 1 < len(packed) else 0
        raw = ((b0 | (b1 << 8)) >> intra_byte) & mask
        out.append(decode_value(spec, raw))
    return out


def _decode_continuous(
    packed: bytes, num_spins: int, spec: AllowedValueSpec
) -> List[int]:
    out: List[int] = []
    for i in range(num_spins):
        chunk = packed[i * 4 : i * 4 + 4]
        if len(chunk) != 4:
            raise PackedSolutionLengthMismatch(num_spins * 4, len(packed))
        raw = int.from_bytes(chunk, "big", signed=False)
        try:
            out.append(decode_value(spec, raw))
        except InvalidEncodedValue:
            raise
    return out


__all__ = [
    "PackedSolutionLengthMismatch",
    "packed_solution_byte_len",
    "pack_solution",
    "unpack_solution",
]
