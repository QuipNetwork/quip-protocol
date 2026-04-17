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


def test_manifest_request_roundtrip_including_empty():
    for locator, limit in [([], 0), ([_hash(10), _hash(5), _hash(0)], 128)]:
        got_locator, got_limit = decode_manifest_request(
            encode_manifest_request(locator, limit)
        )
        assert got_locator == locator and got_limit == limit


def test_manifest_request_rejects_malformed():
    with pytest.raises(ValueError, match="out of range"):
        encode_manifest_request([], MAX_MANIFEST_ENTRIES + 1)
    with pytest.raises(ValueError, match="bytes"):
        encode_manifest_request([b"\x00" * 31], 1)
    with pytest.raises(ValueError, match="too short"):
        decode_manifest_request(b"\x00" * 4)
    with pytest.raises(ValueError, match="truncated"):
        decode_manifest_request(encode_manifest_request([_hash(0), _hash(1)], 4)[:-5])
    with pytest.raises(ValueError, match="exceeds"):
        decode_manifest_request(struct.pack('!II', 1, MAX_MANIFEST_ENTRIES + 1))


def test_manifest_response_roundtrip_including_empty():
    for entries in ([], [(i, _hash(i)) for i in range(5, 15)]):
        assert decode_manifest_response(encode_manifest_response(entries)) == entries


def test_manifest_response_rejects_malformed():
    with pytest.raises(ValueError, match="not greater"):
        encode_manifest_response([(5, _hash(5)), (4, _hash(4))])
    with pytest.raises(ValueError, match="bytes"):
        encode_manifest_response([(1, b"\x00" * 31)])
    with pytest.raises(ValueError, match="too many"):
        encode_manifest_response([(i, _hash(i)) for i in range(MAX_MANIFEST_ENTRIES + 1)])
    payload = encode_manifest_response([(1, _hash(1)), (2, _hash(2))])
    with pytest.raises(ValueError, match="truncated"):
        decode_manifest_response(payload[:-5])
    with pytest.raises(ValueError, match="not greater"):
        decode_manifest_response(
            struct.pack('!I', 2)
            + struct.pack('!I', 5) + _hash(5)
            + struct.pack('!I', 4) + _hash(4)
        )


def test_manifest_response_wire_size():
    entries = [(i, _hash(i)) for i in range(100)]
    assert len(encode_manifest_response(entries)) == 4 + 100 * MANIFEST_ENTRY_SIZE


def test_block_by_hash_request_roundtrip_and_rejects_wrong_length():
    assert decode_block_by_hash_request(encode_block_by_hash_request(_hash(42))) == _hash(42)
    with pytest.raises(ValueError, match="bytes"):
        encode_block_by_hash_request(b"\x00" * 31)
    with pytest.raises(ValueError, match="bytes"):
        decode_block_by_hash_request(b"\x00" * 31)


def test_block_by_hash_response_none_roundtrip():
    assert encode_block_by_hash_response(None) == b""
    assert decode_block_by_hash_response(b"") is None
