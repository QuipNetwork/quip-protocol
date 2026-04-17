"""Tests for the server-side CHAIN_MANIFEST and BLOCK_BY_HASH handlers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _utils import hash_for as _hash
from shared.node_client import QuicMessage, QuicMessageType
from shared.sync_messages import (
    BLOCK_HASH_SIZE,
    decode_manifest_response,
    encode_block_by_hash_request,
    encode_manifest_request,
)


class _FakeBlock:
    def __init__(self, index: int, block_hash: bytes, network_bytes: bytes | None = None):
        self.header = SimpleNamespace(index=index)
        self.hash = block_hash
        self._network_bytes = network_bytes

    def to_network(self) -> bytes:
        assert self._network_bytes is not None
        return self._network_bytes


def _shell(length: int):
    from shared.network_node import NetworkNode
    node = object.__new__(NetworkNode)
    node.chain = []
    node.chain_by_hash = {}
    node.logger = MagicMock()
    for i in range(length):
        blk = _FakeBlock(i, _hash(i))
        node.chain.append(blk)
        node.chain_by_hash[blk.hash] = blk
    return node


def _request(msg_type: QuicMessageType, payload: bytes) -> QuicMessage:
    return QuicMessage(msg_type=msg_type, request_id=1, payload=payload)


@pytest.mark.asyncio
async def test_manifest_handler_returns_slice_after_locator_match_respecting_limit():
    node = _shell(length=20)
    # Locator hits at index 10; limit caps the response.
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request([_hash(10), _hash(0)], limit=5),
    )
    resp = await node._quic_handle_chain_manifest_request(req)
    assert resp.msg_type == QuicMessageType.CHAIN_MANIFEST_RESPONSE
    assert decode_manifest_response(resp.payload) == [(i, _hash(i)) for i in range(11, 16)]


@pytest.mark.asyncio
async def test_manifest_handler_picks_tipmost_locator_hit_and_empty_when_none():
    node = _shell(length=15)
    # Out-of-order locator: tip-most hit (12) wins, not the older (5).
    resp = await node._quic_handle_chain_manifest_request(
        _request(
            QuicMessageType.CHAIN_MANIFEST_REQUEST,
            encode_manifest_request([_hash(12), _hash(5)], limit=100),
        )
    )
    assert decode_manifest_response(resp.payload) == [(13, _hash(13)), (14, _hash(14))]

    # Nothing in the locator matches — empty result.
    resp = await node._quic_handle_chain_manifest_request(
        _request(
            QuicMessageType.CHAIN_MANIFEST_REQUEST,
            encode_manifest_request([b"\xff" * BLOCK_HASH_SIZE], limit=100),
        )
    )
    assert decode_manifest_response(resp.payload) == []


@pytest.mark.asyncio
async def test_handlers_return_error_response_on_malformed_payload():
    node = _shell(length=5)
    for msg_type, bad_payload in [
        (QuicMessageType.CHAIN_MANIFEST_REQUEST, b"\x00\x00"),
        (QuicMessageType.BLOCK_BY_HASH_REQUEST, b"\x00" * 31),
    ]:
        handler = (
            node._quic_handle_chain_manifest_request
            if msg_type == QuicMessageType.CHAIN_MANIFEST_REQUEST
            else node._quic_handle_block_by_hash_request
        )
        resp = await handler(_request(msg_type, bad_payload))
        assert resp.msg_type == QuicMessageType.ERROR_RESPONSE
        assert b"malformed" in resp.payload


@pytest.mark.asyncio
async def test_block_by_hash_handler_found_and_not_found():
    node = _shell(length=5)
    # Replace block 3 with one carrying real serialized bytes.
    block = _FakeBlock(3, _hash(3), network_bytes=b"WIRE-BLOCK-3")
    node.chain[3] = block
    node.chain_by_hash[_hash(3)] = block

    resp = await node._quic_handle_block_by_hash_request(
        _request(QuicMessageType.BLOCK_BY_HASH_REQUEST, encode_block_by_hash_request(_hash(3)))
    )
    assert resp.msg_type == QuicMessageType.BLOCK_BY_HASH_RESPONSE
    assert resp.payload == b"WIRE-BLOCK-3"

    # Unknown hash → empty payload (NOT_FOUND).
    resp = await node._quic_handle_block_by_hash_request(
        _request(
            QuicMessageType.BLOCK_BY_HASH_REQUEST,
            encode_block_by_hash_request(b"\xff" * BLOCK_HASH_SIZE),
        )
    )
    assert resp.msg_type == QuicMessageType.BLOCK_BY_HASH_RESPONSE
    assert resp.payload == b""
