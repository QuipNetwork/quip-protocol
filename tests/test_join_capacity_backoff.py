"""Tests for capacity-aware JOIN backoff and gossip target filtering.

Exercises the coordinator-side changes that break the JOIN retry storm:
  * ``connect_to_peer`` treats ``status == "at_capacity"`` as a
    non-success and records a cooldown.
  * ``gossip_broadcast`` skips peers that SWIM has declared DEAD.
  * ``_setup_child_logging`` (in ``connection_process`` and
    ``listener_process``) silences aioquic internals.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.block import MinerInfo
from shared.network_node import NetworkNode, Message
from shared.peer_ban_list import PeerBanList, CAPACITY_COOLDOWN_MIN
from shared.swim_detector import PeerState, SwimDetector


def _make_info(miner_id: str = "m1") -> MinerInfo:
    return MinerInfo(
        miner_id=miner_id,
        miner_type="CPU",
        reward_address=b"\x01" * 32,
        ecdsa_public_key=b"\x02" * 32,
        wots_public_key=b"\x03" * 32,
        next_wots_public_key=b"\x04" * 32,
    )


def _make_node() -> NetworkNode:
    """Build a NetworkNode shell with just the fields the paths we
    exercise touch."""
    node = object.__new__(NetworkNode)
    node.peers = {}
    node.public_host = "127.0.0.1:9999"
    node.heartbeats = {}
    node.peer_versions = {}
    node.initial_peers = []
    node.logger = MagicMock()
    node.telemetry = MagicMock()
    node._swim_detector = SwimDetector()
    node._peer_scorer = MagicMock()
    node._ban_list = PeerBanList()
    node._load_monitor = MagicMock()
    node._load_monitor.should_accept_join = MagicMock(return_value=True)
    node._load_monitor.is_overloaded = MagicMock(return_value=False)
    node.node_client = MagicMock()
    node.node_client.join_network_via_peer = AsyncMock()
    node._apply_transitive_peer_versions = MagicMock()
    node._format_ban_remaining = NetworkNode._format_ban_remaining
    node.fanout = 3
    node.recent_messages = {}
    node.gossip_lock = asyncio.Lock()
    node._record_recent_message = MagicMock()
    node.gossip_to = AsyncMock(return_value=True)
    node.info = MagicMock(return_value=_make_info("self"))
    node.descriptor = MagicMock(return_value={})
    return node


# --- connect_to_peer honors at_capacity ---

@pytest.mark.asyncio
async def test_connect_to_peer_records_cooldown_on_at_capacity():
    node = _make_node()
    node.node_client.join_network_via_peer.return_value = {
        "status": "at_capacity",
        "peers": {"alt1:20049": _make_info("a").to_json()},
        "peer_versions": {},
    }

    ok = await NetworkNode.connect_to_peer(node, "busy:20049")

    assert ok is False
    assert node._ban_list.is_banned("busy:20049")
    remaining = node._ban_list.time_remaining("busy:20049")
    assert 0 < remaining <= CAPACITY_COOLDOWN_MIN + 1


@pytest.mark.asyncio
async def test_connect_to_peer_success_does_not_cooldown():
    node = _make_node()
    node.add_peer = AsyncMock(return_value=False)
    node.node_client.join_network_via_peer.return_value = {
        "status": "ok",
        "peers": {},
        "peer_versions": {},
        "descriptor": {},
    }

    await NetworkNode.connect_to_peer(node, "ok:20049")

    assert not node._ban_list.is_banned("ok:20049")


@pytest.mark.asyncio
async def test_connect_to_peer_skipped_when_overloaded():
    node = _make_node()
    node._load_monitor.should_accept_join.return_value = False
    node.node_client.join_network_via_peer.reset_mock()

    ok = await NetworkNode.connect_to_peer(node, "healthy:20049")

    assert ok is False
    node.node_client.join_network_via_peer.assert_not_called()


@pytest.mark.asyncio
async def test_connect_to_peer_overload_does_not_block_initial_peers():
    node = _make_node()
    node._load_monitor.should_accept_join.return_value = False
    node.initial_peers = ["seed:20049"]
    node.add_peer = AsyncMock(return_value=False)
    node.node_client.join_network_via_peer.return_value = {
        "status": "ok",
        "peers": {},
        "peer_versions": {},
        "descriptor": {},
    }

    ok = await NetworkNode.connect_to_peer(node, "seed:20049")

    assert ok is True
    node.node_client.join_network_via_peer.assert_awaited_once()


# --- gossip_broadcast skips DEAD peers ---

@pytest.mark.asyncio
async def test_gossip_skips_dead_peers():
    node = _make_node()
    node.peers = {
        "alive:20049": _make_info("a"),
        "dead:20049": _make_info("d"),
    }
    node._swim_detector.add_peer("alive:20049")
    node._swim_detector.add_peer("dead:20049")
    # Force dead peer to DEAD state
    node._swim_detector._peers["dead:20049"].state = PeerState.DEAD

    sent_to: list[str] = []
    async def capture(peer, msg):
        sent_to.append(peer)
        return True
    node.gossip_to = capture

    msg = Message(
        type="noop",
        sender=node.public_host,
        timestamp=1.0,
        data=b"",
        id="gossip-1",
    )
    await NetworkNode.gossip_broadcast(node, msg, fanout=5)

    assert "alive:20049" in sent_to
    assert "dead:20049" not in sent_to


# --- child-process logging silences aioquic ---

def test_connection_child_silences_aioquic():
    # Save current level to restore
    quic_logger = logging.getLogger("quic")
    prev = quic_logger.level
    try:
        # Raise to INFO first to prove the function lowered it
        quic_logger.setLevel(logging.INFO)

        from shared.connection_process import _setup_child_logging
        _setup_child_logging("127.0.0.1:9999")

        assert quic_logger.level == logging.WARNING
        assert logging.getLogger("aioquic").level == logging.WARNING
    finally:
        quic_logger.setLevel(prev)


def test_listener_child_silences_aioquic():
    quic_logger = logging.getLogger("quic")
    prev = quic_logger.level
    try:
        quic_logger.setLevel(logging.INFO)

        from shared.listener_process import _setup_child_logging
        _setup_child_logging()

        assert quic_logger.level == logging.WARNING
        assert logging.getLogger("aioquic").level == logging.WARNING
    finally:
        quic_logger.setLevel(prev)
