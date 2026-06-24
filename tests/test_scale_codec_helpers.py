# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Unit tests for stateless SCALE helpers in ``substrate.scale_codec``.

These tests are purely offline — no chain connection required. They build
hand-crafted SCALE byte buffers and verify the decoders produce the correct
output. Compact-encoding helpers are copied from ``test_mining_snapshot_decode``
rather than imported so this file has no cross-test dependencies.
"""

from __future__ import annotations

import pytest
from scalecodec.base import ScaleBytes

from substrate.scale_codec import _decode_h256_vec


def _encode_compact_u32(n: int) -> bytes:
    """Reference SCALE compact encoder for u32. Mirrors the decoder under test."""
    if n < 0:
        raise ValueError(f"compact u32 must be non-negative, got {n}")
    if n < 0x40:
        return bytes([n << 2])
    if n < 0x4000:
        return ((n << 2) | 0b01).to_bytes(2, "little")
    if n < 0x4000_0000:
        return ((n << 2) | 0b10).to_bytes(4, "little")
    raise NotImplementedError("big-integer compact mode not exercised by these tests")


def _build_h256_vec_bytes(hashes: list[bytes]) -> bytes:
    """Encode a Vec<H256> as compact-u32 length + concatenated 32-byte hashes."""
    out = _encode_compact_u32(len(hashes))
    for h in hashes:
        assert len(h) == 32, f"each hash must be 32 bytes, got {len(h)}"
        out += h
    return out


# ---------------------------------------------------------------------------
# _decode_h256_vec
# ---------------------------------------------------------------------------


def test_decode_h256_vec_empty():
    """A zero-length Vec<H256> decodes to an empty list."""
    buf = ScaleBytes(_build_h256_vec_bytes([]))
    result = _decode_h256_vec(buf)
    assert result == []
    assert buf.get_remaining_length() == 0


def test_decode_h256_vec_single():
    """A single-element Vec<H256> decodes to a one-element list."""
    h = bytes(range(32))
    buf = ScaleBytes(_build_h256_vec_bytes([h]))
    result = _decode_h256_vec(buf)
    assert result == [h]
    assert buf.get_remaining_length() == 0


def test_decode_h256_vec_three_hashes():
    """Three distinct hashes decode in order."""
    hashes = [bytes([i] * 32) for i in range(3)]
    buf = ScaleBytes(_build_h256_vec_bytes(hashes))
    result = _decode_h256_vec(buf)
    assert result == hashes
    assert buf.get_remaining_length() == 0


def test_decode_h256_vec_preserves_order():
    """The output list preserves the wire order of the hashes."""
    hashes = [bytes([0xAA] * 32), bytes([0xBB] * 32), bytes([0xCC] * 32)]
    buf = ScaleBytes(_build_h256_vec_bytes(hashes))
    result = _decode_h256_vec(buf)
    assert result[0] == bytes([0xAA] * 32)
    assert result[1] == bytes([0xBB] * 32)
    assert result[2] == bytes([0xCC] * 32)


def test_decode_h256_vec_trailing_bytes_left_for_caller():
    """Bytes after the vec are left in the buffer for the caller to check."""
    h = bytes([0x01] * 32)
    raw = _build_h256_vec_bytes([h]) + b"\xde\xad"
    buf = ScaleBytes(raw)
    result = _decode_h256_vec(buf)
    assert result == [h]
    # Caller is responsible for asserting get_remaining_length() == 0.
    assert buf.get_remaining_length() == 2


def test_decode_h256_vec_short_read_raises():
    """A truncated buffer surfaces a short-read error."""
    # Claim 2 hashes but only provide 1.
    raw = _encode_compact_u32(2) + bytes([0x55] * 32)
    buf = ScaleBytes(raw)
    with pytest.raises(ValueError, match="short read"):
        _decode_h256_vec(buf)


def test_decode_h256_vec_compact_two_byte_length():
    """Compact two-byte length prefix (n ≥ 64) decodes correctly."""
    hashes = [bytes([i % 256] * 32) for i in range(64)]
    buf = ScaleBytes(_build_h256_vec_bytes(hashes))
    result = _decode_h256_vec(buf)
    assert len(result) == 64
    assert result[63] == bytes([63 % 256] * 32)
    assert buf.get_remaining_length() == 0


# ---------------------------------------------------------------------------
# query_mineable_topologies / query_difficulty_for — offline decode tests
# These test the SCALE decode logic by monkeypatching _state_call so no
# chain connection is needed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_mineable_topologies_empty(monkeypatch):
    """An empty Vec<H256> from the runtime API returns []."""
    from substrate.client import SubstrateClient

    raw_hex = "0x" + _build_h256_vec_bytes([]).hex()
    client = SubstrateClient.__new__(SubstrateClient)

    async def _fake_state_call(method, param, block_hash):
        assert method == "QuantumPowApi_mineable_topologies"
        assert param == "0x"
        return raw_hex

    monkeypatch.setattr(client, "_state_call", _fake_state_call)
    result = await client.query_mineable_topologies()
    assert result == []


@pytest.mark.asyncio
async def test_query_mineable_topologies_two_hashes(monkeypatch):
    """Two hashes from the runtime API decode correctly."""
    from substrate.client import SubstrateClient

    hashes = [bytes([0xAA] * 32), bytes([0xBB] * 32)]
    raw_hex = "0x" + _build_h256_vec_bytes(hashes).hex()
    client = SubstrateClient.__new__(SubstrateClient)

    async def _fake_state_call(method, param, block_hash):
        return raw_hex

    monkeypatch.setattr(client, "_state_call", _fake_state_call)
    result = await client.query_mineable_topologies()
    assert result == hashes


@pytest.mark.asyncio
async def test_query_difficulty_for_none(monkeypatch):
    """Option::None (0x00) from the runtime API returns Python None."""
    from substrate.client import SubstrateClient

    client = SubstrateClient.__new__(SubstrateClient)

    async def _fake_state_call(method, param, block_hash):
        assert method == "QuantumPowApi_difficulty_for"
        return "0x00"

    monkeypatch.setattr(client, "_state_call", _fake_state_call)
    result = await client.query_difficulty_for(b"\xab" * 32)
    assert result is None


@pytest.mark.asyncio
async def test_query_difficulty_for_some(monkeypatch):
    """Option::Some(DifficultyConfig) decodes to a SubstrateDifficulty."""
    from substrate.client import SubstrateClient
    from substrate.types import SubstrateDifficulty

    # Build Option::Some(DifficultyConfig { min_solutions: 3, max_energy_milli: -500,
    #                                       min_diversity_milli: 200 })
    inner = (
        b"\x01"  # Option::Some tag
        + (3).to_bytes(4, "little")
        + (-500).to_bytes(8, "little", signed=True)
        + (200).to_bytes(4, "little")
    )
    raw_hex = "0x" + inner.hex()
    client = SubstrateClient.__new__(SubstrateClient)

    async def _fake_state_call(method, param, block_hash):
        return raw_hex

    monkeypatch.setattr(client, "_state_call", _fake_state_call)
    result = await client.query_difficulty_for(b"\xcd" * 32)
    assert isinstance(result, SubstrateDifficulty)
    assert result.min_solutions == 3
    assert result.max_energy_milli == -500
    assert result.min_diversity_milli == 200


@pytest.mark.asyncio
async def test_query_difficulty_for_bad_topology_hash_length():
    """Passing a hash of wrong length raises ValueError before any RPC."""
    from substrate.client import SubstrateClient

    client = SubstrateClient.__new__(SubstrateClient)
    with pytest.raises(ValueError, match="32 bytes"):
        await client.query_difficulty_for(b"\xab" * 16)
