"""Unit tests for the chain-manifest and block-by-hash wire codecs."""

import struct

import pytest

from shared.sync_messages import (
    BLOCK_HASH_SIZE,
    MANIFEST_ENTRY_SIZE,
    MAX_MANIFEST_ENTRIES,
    decode_block_by_hash_request,
    decode_block_by_hash_response,
    decode_manifest_request,
    decode_manifest_response,
    encode_block_by_hash_request,
    encode_block_by_hash_response,
    encode_manifest_request,
    encode_manifest_response,
)


def _hash(i: int) -> bytes:
    return i.to_bytes(BLOCK_HASH_SIZE, "big")


# ---------------------------------------------------------------------------
# CHAIN_MANIFEST_REQUEST
# ---------------------------------------------------------------------------


def test_manifest_request_roundtrip():
    locator = [_hash(10), _hash(9), _hash(7), _hash(3), _hash(0)]
    payload = encode_manifest_request(locator, limit=128)
    got_locator, got_limit = decode_manifest_request(payload)
    assert got_locator == locator
    assert got_limit == 128


def test_manifest_request_empty_locator_and_zero_limit():
    payload = encode_manifest_request([], limit=0)
    got_locator, got_limit = decode_manifest_request(payload)
    assert got_locator == []
    assert got_limit == 0


def test_manifest_request_rejects_wrong_hash_length():
    with pytest.raises(ValueError, match="31 bytes"):
        encode_manifest_request([b"\x00" * 31], limit=1)


def test_manifest_request_rejects_limit_over_cap():
    with pytest.raises(ValueError, match="out of range"):
        encode_manifest_request([_hash(0)], limit=MAX_MANIFEST_ENTRIES + 1)


def test_manifest_request_rejects_negative_limit():
    with pytest.raises(ValueError, match="out of range"):
        encode_manifest_request([_hash(0)], limit=-1)


def test_manifest_request_decode_rejects_short_header():
    with pytest.raises(ValueError, match="too short"):
        decode_manifest_request(b"\x00" * 4)


def test_manifest_request_decode_rejects_truncated_payload():
    payload = encode_manifest_request([_hash(0), _hash(1)], limit=4)
    with pytest.raises(ValueError, match="truncated"):
        decode_manifest_request(payload[:-5])


def test_manifest_request_decode_rejects_locator_over_cap():
    # Handcraft a payload that advertises a locator length just above the cap.
    malicious = struct.pack('!II', 1, MAX_MANIFEST_ENTRIES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        decode_manifest_request(malicious)


# ---------------------------------------------------------------------------
# CHAIN_MANIFEST_RESPONSE
# ---------------------------------------------------------------------------


def test_manifest_response_roundtrip():
    entries = [(i, _hash(i)) for i in range(5, 15)]
    payload = encode_manifest_response(entries)
    got = decode_manifest_response(payload)
    assert got == entries


def test_manifest_response_empty():
    payload = encode_manifest_response([])
    assert decode_manifest_response(payload) == []


def test_manifest_response_encode_rejects_non_ascending_indices():
    with pytest.raises(ValueError, match="not greater"):
        encode_manifest_response([(5, _hash(5)), (4, _hash(4))])


def test_manifest_response_encode_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="not greater"):
        encode_manifest_response([(5, _hash(5)), (5, _hash(5))])


def test_manifest_response_encode_rejects_wrong_hash_size():
    with pytest.raises(ValueError, match="31 bytes"):
        encode_manifest_response([(1, b"\x00" * 31)])


def test_manifest_response_encode_rejects_over_cap():
    too_many = [(i, _hash(i)) for i in range(MAX_MANIFEST_ENTRIES + 1)]
    with pytest.raises(ValueError, match="too many"):
        encode_manifest_response(too_many)


def test_manifest_response_decode_rejects_short_header():
    with pytest.raises(ValueError, match="too short"):
        decode_manifest_response(b"\x00")


def test_manifest_response_decode_rejects_truncated():
    payload = encode_manifest_response([(1, _hash(1)), (2, _hash(2))])
    with pytest.raises(ValueError, match="truncated"):
        decode_manifest_response(payload[:-5])


def test_manifest_response_decode_rejects_non_ascending():
    # Construct a malicious payload: count=2, then (5, h5), (4, h4).
    malicious = (
        struct.pack('!I', 2)
        + struct.pack('!I', 5) + _hash(5)
        + struct.pack('!I', 4) + _hash(4)
    )
    with pytest.raises(ValueError, match="not greater"):
        decode_manifest_response(malicious)


def test_manifest_response_decode_rejects_count_over_cap():
    malicious = struct.pack('!I', MAX_MANIFEST_ENTRIES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        decode_manifest_response(malicious)


def test_manifest_response_wire_size_matches_formula():
    """Wire size should be exactly 4 + N * 36 bytes for N entries."""
    entries = [(i, _hash(i)) for i in range(100)]
    payload = encode_manifest_response(entries)
    assert len(payload) == 4 + 100 * MANIFEST_ENTRY_SIZE


# ---------------------------------------------------------------------------
# BLOCK_BY_HASH_REQUEST
# ---------------------------------------------------------------------------


def test_block_by_hash_request_roundtrip():
    h = _hash(42)
    payload = encode_block_by_hash_request(h)
    assert decode_block_by_hash_request(payload) == h


def test_block_by_hash_request_encode_rejects_wrong_length():
    with pytest.raises(ValueError, match="31 bytes"):
        encode_block_by_hash_request(b"\x00" * 31)


def test_block_by_hash_request_decode_rejects_wrong_length():
    with pytest.raises(ValueError, match="31 bytes"):
        decode_block_by_hash_request(b"\x00" * 31)
    with pytest.raises(ValueError, match="33 bytes"):
        decode_block_by_hash_request(b"\x00" * 33)


# ---------------------------------------------------------------------------
# BLOCK_BY_HASH_RESPONSE
# ---------------------------------------------------------------------------


def test_block_by_hash_response_none_roundtrip():
    assert encode_block_by_hash_response(None) == b""
    assert decode_block_by_hash_response(b"") is None
