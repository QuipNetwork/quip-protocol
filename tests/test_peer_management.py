"""Tests for two-tier peer management (active + candidate pools)."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.block import MinerInfo
from shared.network_node import CandidatePeer, NetworkNode
from shared.swim_detector import SwimDetector
from shared.time_utils import utc_timestamp_float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_info(miner_id: str = "m1") -> MinerInfo:
    """Build a minimal MinerInfo for testing."""
    return MinerInfo(
        miner_id=miner_id,
        miner_type="CPU",
        reward_address=b"\x01" * 32,
        ecdsa_public_key=b"\x02" * 32,
        wots_public_key=b"\x03" * 32,
        next_wots_public_key=b"\x04" * 32,
    )


def _make_node(
    max_active: int = 3,
    max_candidate: int = 5,
    heartbeat_timeout: float = 300.0,
) -> NetworkNode:
    """Build a NetworkNode stub for unit testing.

    Bypasses the full Node/NetworkNode __init__ and sets only the
    fields that add_peer / remove_peer / promotion touch.
    """
    node = object.__new__(NetworkNode)

    # Fields from Node that add_peer / remove_peer use.
    node.peers = {}
    node.on_new_node = None

    # Fields from NetworkNode.__init__
    node.public_host = "127.0.0.1:9999"
    node.heartbeat_interval = 15
    node.heartbeat_timeout = heartbeat_timeout
    node.node_timeout = 60
    node._max_active_peers = max_active
    node._max_candidate_peers = max_candidate
    node._candidate_peers = {}
    node.heartbeats = {}
    node.peer_versions = {}
    node.net_lock = asyncio.Lock()
    node._has_ever_had_peers = False
    node.node_client = None
    node.logger = MagicMock()
    node.telemetry = MagicMock()
    node._swim_detector = SwimDetector()
    node._peer_scorer = MagicMock()
    node._peer_scorer.remove_peer = MagicMock()
    node._peer_loads = {}
    node._block_inventory = MagicMock()
    node._block_inventory.remove_peer = MagicMock()
    node._ban_list = MagicMock()
    node._ban_list.is_banned = MagicMock(return_value=False)
    node.fanout = 3
    node.recent_messages = {}
    node.gossip_lock = asyncio.Lock()
    node._announced_nodes = {}
    node._announced_nodes_ttl = 300.0
    # Stub gossip_new_node and info() to avoid QUIC / crypto calls.
    node.gossip_new_node = AsyncMock()
    node.info = MagicMock(return_value=_make_info("test-node"))
    return node


# ---------------------------------------------------------------------------
# CandidatePeer dataclass
# ---------------------------------------------------------------------------

class TestCandidatePeer:
    def test_defaults(self):
        c = CandidatePeer(
            info=_make_info(), discovered_at=1.0, source="gossip",
        )
        assert c.probe_attempts == 0
        assert c.last_probe_at == 0.0
        assert c.descriptor is None

    def test_mutable_fields(self):
        c = CandidatePeer(
            info=_make_info(), discovered_at=1.0, source="gossip",
        )
        c.probe_attempts = 3
        c.last_probe_at = 99.0
        assert c.probe_attempts == 3


# ---------------------------------------------------------------------------
# add_peer: two-tier gating
# ---------------------------------------------------------------------------

class TestAddPeerTwoTier:
    @pytest.fixture
    def node(self):
        return _make_node(max_active=3, max_candidate=5)

    @pytest.mark.asyncio
    async def test_adds_to_active_when_room(self, node):
        added = await node.add_peer("10.0.0.1:20001", _make_info("a"))
        assert added is True
        assert "10.0.0.1:20001" in node.peers
        assert "10.0.0.1:20001" not in node._candidate_peers

    @pytest.mark.asyncio
    async def test_routes_to_candidate_when_active_full(self, node):
        for i in range(3):
            await node.add_peer(
                f"10.0.0.{i + 1}:20001", _make_info(f"m{i}"),
            )
        assert len(node.peers) == 3

        added = await node.add_peer(
            "10.0.1.1:20001", _make_info("overflow"),
        )
        assert added is False
        assert "10.0.1.1:20001" not in node.peers
        assert "10.0.1.1:20001" in node._candidate_peers

    @pytest.mark.asyncio
    async def test_connected_bypasses_cap(self, node):
        for i in range(3):
            await node.add_peer(
                f"10.0.0.{i + 1}:20001", _make_info(f"m{i}"),
            )

        added = await node.add_peer(
            "10.0.2.1:20001", _make_info("direct"), connected=True,
        )
        assert added is True
        assert "10.0.2.1:20001" in node.peers
        assert len(node.peers) == 4  # exceeds max_active_peers

    @pytest.mark.asyncio
    async def test_update_existing_active_peer(self, node):
        await node.add_peer("10.0.0.1:20001", _make_info("orig"))
        result = await node.add_peer("10.0.0.1:20001", _make_info("upd"))
        assert result is False  # not new
        assert "10.0.0.1:20001" in node.peers

    @pytest.mark.asyncio
    async def test_candidate_eviction_when_pool_full(self, node):
        # Fill active set
        for i in range(3):
            await node.add_peer(
                f"10.0.0.{i + 1}:20001", _make_info(f"m{i}"),
            )

        # Fill candidate pool (max=5)
        for i in range(5):
            await node.add_peer(
                f"10.0.1.{i + 1}:20001", _make_info(f"c{i}"),
            )
        assert len(node._candidate_peers) == 5

        # Adding one more evicts the oldest candidate
        await node.add_peer("10.0.1.99:20001", _make_info("extra"))
        assert len(node._candidate_peers) == 5
        assert "10.0.1.99:20001" in node._candidate_peers

    @pytest.mark.asyncio
    async def test_skip_self_address(self, node):
        added = await node.add_peer("127.0.0.1:9999", _make_info())
        assert added is False
        assert "127.0.0.1:9999" not in node.peers

    @pytest.mark.asyncio
    async def test_skip_banned_peer(self, node):
        node._ban_list.is_banned = MagicMock(return_value=True)
        added = await node.add_peer("10.0.3.1:20001", _make_info())
        assert added is False

    @pytest.mark.asyncio
    async def test_promotion_removes_from_candidates(self, node):
        """When a candidate is promoted via connected=True, it leaves
        the candidate pool."""
        for i in range(3):
            await node.add_peer(
                f"10.0.0.{i + 1}:20001", _make_info(f"m{i}"),
            )
        await node.add_peer("10.0.4.1:20001", _make_info("cand"))
        assert "10.0.4.1:20001" in node._candidate_peers

        await node.add_peer(
            "10.0.4.1:20001", _make_info("cand"), connected=True,
        )
        assert "10.0.4.1:20001" in node.peers
        assert "10.0.4.1:20001" not in node._candidate_peers


# ---------------------------------------------------------------------------
# remove_peer: cleans up candidates
# ---------------------------------------------------------------------------

class TestRemovePeerCandidateCleanup:
    @pytest.mark.asyncio
    async def test_remove_peer_clears_candidate(self):
        node = _make_node()
        for i in range(3):
            await node.add_peer(
                f"10.0.0.{i + 1}:20001", _make_info(f"m{i}"),
            )
        await node.add_peer("10.0.5.1:20001", _make_info("c"))
        assert "10.0.5.1:20001" in node._candidate_peers

        node.remove_peer("10.0.5.1:20001")
        assert "10.0.5.1:20001" not in node._candidate_peers


# ---------------------------------------------------------------------------
# _healthy_peers_snapshot
# ---------------------------------------------------------------------------

class TestHealthyPeersSnapshot:
    @pytest.mark.asyncio
    async def test_filters_stale_peers(self):
        node = _make_node(max_active=10, heartbeat_timeout=300.0)
        now = utc_timestamp_float()

        await node.add_peer("10.0.6.1:20001", _make_info("f"))
        node.heartbeats["10.0.6.1:20001"] = now - 10  # fresh

        await node.add_peer("10.0.6.2:20001", _make_info("s"))
        node.heartbeats["10.0.6.2:20001"] = now - 200  # stale (> 150)

        await node.add_peer("10.0.6.3:20001", _make_info("n"))
        # no heartbeat entry → excluded

        async with node.net_lock:
            healthy = node._healthy_peers_snapshot()

        assert "10.0.6.1:20001" in healthy
        assert "10.0.6.2:20001" not in healthy
        assert "10.0.6.3:20001" not in healthy


# ---------------------------------------------------------------------------
# _try_promote_candidates
# ---------------------------------------------------------------------------

class TestCandidatePromotion:
    @pytest.mark.asyncio
    async def test_promotes_reachable_candidate(self):
        node = _make_node(max_active=3, max_candidate=5)
        await node.add_peer("10.0.7.1:20001", _make_info("a"))
        await node.add_peer("10.0.7.2:20001", _make_info("b"))

        node._candidate_peers["10.0.7.3:20001"] = CandidatePeer(
            info=_make_info("c"),
            discovered_at=time.monotonic() - 60,
            source="gossip",
        )

        node.node_client = MagicMock()
        node.node_client.send_heartbeat = AsyncMock(return_value=True)

        await node._try_promote_candidates()

        assert "10.0.7.3:20001" in node.peers
        assert "10.0.7.3:20001" not in node._candidate_peers

    @pytest.mark.asyncio
    async def test_evicts_after_3_failed_probes(self):
        node = _make_node(max_active=3)
        await node.add_peer("10.0.8.1:20001", _make_info("a"))
        await node.add_peer("10.0.8.2:20001", _make_info("b"))

        node._candidate_peers["10.0.8.3:20001"] = CandidatePeer(
            info=_make_info("bad"),
            discovered_at=time.monotonic() - 60,
            source="gossip",
            probe_attempts=2,
        )

        node.node_client = MagicMock()
        node.node_client.send_heartbeat = AsyncMock(return_value=False)

        await node._try_promote_candidates()

        assert "10.0.8.3:20001" not in node._candidate_peers
        assert "10.0.8.3:20001" not in node.peers

    @pytest.mark.asyncio
    async def test_skips_when_active_full(self):
        node = _make_node(max_active=2)
        await node.add_peer("10.0.9.1:20001", _make_info("a"))
        await node.add_peer("10.0.9.2:20001", _make_info("b"))

        node._candidate_peers["10.0.9.3:20001"] = CandidatePeer(
            info=_make_info("c"),
            discovered_at=time.monotonic() - 60,
            source="gossip",
        )

        node.node_client = MagicMock()
        node.node_client.send_heartbeat = AsyncMock(return_value=True)

        await node._try_promote_candidates()

        assert "10.0.9.3:20001" in node._candidate_peers


# ---------------------------------------------------------------------------
# node_cleanup_loop: fast eviction of unproven peers
# ---------------------------------------------------------------------------

class TestFastEviction:
    @pytest.mark.asyncio
    async def test_unproven_peer_evicted_fast(self):
        """Peers with no successful heartbeat are removed after
        node_timeout (60s), not the full heartbeat_timeout (300s)."""
        node = _make_node(max_active=5, heartbeat_timeout=300.0)
        node.node_timeout = 60.0

        await node.add_peer("10.0.10.1:20001", _make_info("u"))
        health = node._swim_detector._peers.get("10.0.10.1:20001")
        assert health is not None
        health.joined_at = time.monotonic() - 120  # well past 60s

        await node.add_peer("10.0.10.2:20001", _make_info("p"))
        node.heartbeats["10.0.10.2:20001"] = utc_timestamp_float() - 10

        # Inline the dead-node detection logic from node_cleanup_loop
        current_time = utc_timestamp_float()
        now_mono = time.monotonic()
        dead_nodes = []
        for host in list(node.peers.keys()):
            hb_time = node.heartbeats.get(host)
            if hb_time is not None:
                if current_time - hb_time > node.heartbeat_timeout:
                    dead_nodes.append(host)
            else:
                h = node._swim_detector._peers.get(host)
                age = (
                    now_mono - h.joined_at if h
                    else node.node_timeout + 1
                )
                if age > node.node_timeout:
                    dead_nodes.append(host)

        assert "10.0.10.1:20001" in dead_nodes
        assert "10.0.10.2:20001" not in dead_nodes


# ---------------------------------------------------------------------------
# gossip_new_node: per-host announcement dedup
# ---------------------------------------------------------------------------

class TestGossipNewNodeDedup:
    """Verifies that gossip_new_node skips origination when the host
    has been announced within _announced_nodes_ttl, preventing the
    N² new_node amplification observed in production."""

    @pytest.fixture
    def node(self):
        node = _make_node()
        # Use the real gossip_new_node (with its dedup gate), and stub
        # only the downstream broadcast.
        node.gossip_new_node = NetworkNode.gossip_new_node.__get__(node)
        node.gossip = AsyncMock()
        return node

    @pytest.mark.asyncio
    async def test_first_call_originates(self, node):
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        assert node.gossip.await_count == 1
        assert "10.0.11.1:20001" in node._announced_nodes

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_skips(self, node):
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        assert node.gossip.await_count == 1

    @pytest.mark.asyncio
    async def test_call_after_ttl_expiry_re_announces(self, node):
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        # Backdate the entry past the TTL window.
        node._announced_nodes["10.0.11.1:20001"] = (
            time.time() - node._announced_nodes_ttl - 1
        )
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        assert node.gossip.await_count == 2

    @pytest.mark.asyncio
    async def test_distinct_hosts_both_announce(self, node):
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        await node.gossip_new_node("10.0.11.2:20001", _make_info("b"))
        assert node.gossip.await_count == 2

    @pytest.mark.asyncio
    async def test_remove_peer_clears_announcement(self, node):
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        node.remove_peer("10.0.11.1:20001")
        assert "10.0.11.1:20001" not in node._announced_nodes
        # Now a fresh announcement should proceed.
        await node.gossip_new_node("10.0.11.1:20001", _make_info("a"))
        assert node.gossip.await_count == 2


# ---------------------------------------------------------------------------
# _prune_announced_nodes: TTL expiry
# ---------------------------------------------------------------------------

class TestPruneAnnouncedNodes:
    def test_prunes_only_entries_older_than_ttl(self):
        node = _make_node()
        now = time.time()
        node._announced_nodes["fresh:1"] = now - 10
        node._announced_nodes["stale:1"] = now - node._announced_nodes_ttl - 5
        node._announced_nodes["stale:2"] = now - node._announced_nodes_ttl - 60

        removed = node._prune_announced_nodes(now=now)

        assert removed == 2
        assert "fresh:1" in node._announced_nodes
        assert "stale:1" not in node._announced_nodes
        assert "stale:2" not in node._announced_nodes
