"""Tests for the server-side CHAIN_MANIFEST and BLOCK_BY_HASH handlers.

Exercises ``_quic_handle_chain_manifest_request`` and
``_quic_handle_block_by_hash_request`` in isolation from the QUIC
transport by constructing a NetworkNode shell via ``object.__new__``
and populating only the chain/index attributes the handlers read.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.node_client import QuicMessage, QuicMessageType
from shared.sync_messages import (
    BLOCK_HASH_SIZE,
    decode_block_by_hash_response,
    decode_manifest_response,
    encode_block_by_hash_request,
    encode_manifest_request,
)


def _hash(i: int) -> bytes:
    return i.to_bytes(BLOCK_HASH_SIZE, "big")


class _FakeBlock:
    """Minimal Block stand-in.

    The handlers need ``header.index`` and ``hash``; for
    ``BLOCK_BY_HASH_RESPONSE`` they also call ``to_network()``.
    """

    def __init__(self, index: int, block_hash: bytes, network_bytes: bytes | None = None):
        self.header = SimpleNamespace(index=index)
        self.hash = block_hash
        self._network_bytes = network_bytes

    def to_network(self) -> bytes:
        if self._network_bytes is None:
            raise RuntimeError("_FakeBlock missing network_bytes; test needs it")
        return self._network_bytes


def _make_network_node_with_chain(length: int):
    """Build a NetworkNode shell with a linear chain of ``length`` fake blocks."""
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


# ---------------------------------------------------------------------------
# CHAIN_MANIFEST_REQUEST handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_handler_returns_entries_after_locator_match():
    """Server returns canonical entries starting at latest_common_ancestor + 1."""
    node = _make_network_node_with_chain(length=20)
    # Client has genesis..10 and is asking for 11..19. Send a tip-first
    # locator that includes hash 10 as the most-recent known block.
    locator = [_hash(10), _hash(8), _hash(4), _hash(0)]
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request(locator, limit=100),
    )

    response = await node._quic_handle_chain_manifest_request(req)

    assert response.msg_type == QuicMessageType.CHAIN_MANIFEST_RESPONSE
    entries = decode_manifest_response(response.payload)
    expected = [(i, _hash(i)) for i in range(11, 20)]
    assert entries == expected


@pytest.mark.asyncio
async def test_manifest_handler_respects_limit():
    node = _make_network_node_with_chain(length=100)
    locator = [_hash(5), _hash(0)]
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request(locator, limit=8),
    )

    response = await node._quic_handle_chain_manifest_request(req)

    entries = decode_manifest_response(response.payload)
    assert entries == [(i, _hash(i)) for i in range(6, 14)]


@pytest.mark.asyncio
async def test_manifest_handler_picks_newest_matching_locator_hash():
    """When the locator contains several hashes on our chain, the latest wins."""
    node = _make_network_node_with_chain(length=15)
    # Out-of-order locator: older hash comes first, but our handler must
    # scan in order and pick the first (tip-most) hit.
    locator = [_hash(12), _hash(5)]  # tip-first: 12 is newer
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request(locator, limit=100),
    )

    response = await node._quic_handle_chain_manifest_request(req)
    entries = decode_manifest_response(response.payload)
    assert entries == [(13, _hash(13)), (14, _hash(14))]


@pytest.mark.asyncio
async def test_manifest_handler_returns_empty_when_no_locator_matches():
    node = _make_network_node_with_chain(length=10)
    # Locator hashes that are not on our chain.
    locator = [b"\xff" * BLOCK_HASH_SIZE, b"\xee" * BLOCK_HASH_SIZE]
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request(locator, limit=100),
    )

    response = await node._quic_handle_chain_manifest_request(req)
    entries = decode_manifest_response(response.payload)
    assert entries == []


@pytest.mark.asyncio
async def test_manifest_handler_returns_empty_when_caller_at_tip():
    """Client's tip is our tip — nothing to send."""
    node = _make_network_node_with_chain(length=20)
    locator = [_hash(19), _hash(18)]
    req = _request(
        QuicMessageType.CHAIN_MANIFEST_REQUEST,
        encode_manifest_request(locator, limit=100),
    )

    response = await node._quic_handle_chain_manifest_request(req)
    entries = decode_manifest_response(response.payload)
    assert entries == []


@pytest.mark.asyncio
async def test_manifest_handler_errors_on_malformed_payload():
    node = _make_network_node_with_chain(length=5)
    req = _request(QuicMessageType.CHAIN_MANIFEST_REQUEST, b"\x00\x00")  # too short

    response = await node._quic_handle_chain_manifest_request(req)

    assert response.msg_type == QuicMessageType.ERROR_RESPONSE
    assert b"malformed" in response.payload


# ---------------------------------------------------------------------------
# BLOCK_BY_HASH_REQUEST handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_by_hash_handler_returns_block_when_present():
    node = _make_network_node_with_chain(length=5)
    # Replace block 3 with one that has real network bytes.
    block = _FakeBlock(3, _hash(3), network_bytes=b"WIRE-BLOCK-3")
    node.chain[3] = block
    node.chain_by_hash[_hash(3)] = block

    req = _request(
        QuicMessageType.BLOCK_BY_HASH_REQUEST,
        encode_block_by_hash_request(_hash(3)),
    )

    response = await node._quic_handle_block_by_hash_request(req)

    assert response.msg_type == QuicMessageType.BLOCK_BY_HASH_RESPONSE
    assert response.payload == b"WIRE-BLOCK-3"


@pytest.mark.asyncio
async def test_block_by_hash_handler_returns_empty_payload_for_unknown_hash():
    node = _make_network_node_with_chain(length=5)
    req = _request(
        QuicMessageType.BLOCK_BY_HASH_REQUEST,
        encode_block_by_hash_request(b"\xff" * BLOCK_HASH_SIZE),
    )

    response = await node._quic_handle_block_by_hash_request(req)

    assert response.msg_type == QuicMessageType.BLOCK_BY_HASH_RESPONSE
    assert response.payload == b""
    assert decode_block_by_hash_response(response.payload) is None


@pytest.mark.asyncio
async def test_block_by_hash_handler_errors_on_malformed_payload():
    node = _make_network_node_with_chain(length=5)
    req = _request(QuicMessageType.BLOCK_BY_HASH_REQUEST, b"\x00" * 31)  # wrong length

    response = await node._quic_handle_block_by_hash_request(req)

    assert response.msg_type == QuicMessageType.ERROR_RESPONSE
    assert b"malformed" in response.payload
