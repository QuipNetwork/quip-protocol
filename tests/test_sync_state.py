"""Tests for NetworkNode synchronization state management.

Verifies the three-state sync model:
  BOOTSTRAP  — never had peers, solo mining allowed (auto_mine)
  SYNCED     — connected + verified chain, mining allowed
  DESYNCED   — had peers then lost them, mining blocked until re-sync
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal NetworkNode fixture that avoids the heavy Node.__init__ path
# ---------------------------------------------------------------------------

def _make_network_node(auto_mine: bool = True):
    """Build a lightweight NetworkNode-like object for sync state tests.

    We patch the expensive parent __init__ and only set up the attributes
    that the sync state code paths actually use.
    """
    from shared.network_node import NetworkNode

    config = {
        "listen": "127.0.0.1",
        "port": 0,
        "node_name": "test-node",
        "secret": "test-secret",
        "auto_mine": auto_mine,
        "peer": [],
        "tofu": False,
        "telemetry_enabled": False,
    }

    # Patch the heavy parent init (Node.__init__) to avoid miner setup
    with patch("shared.network_node.Node.__init__", return_value=None):
        # Bypass the full __init__; we'll set needed attrs manually
        node = object.__new__(NetworkNode)

    # --- Attrs that __init__ sets and our tests need ---
    node.config = config
    node.auto_mine = auto_mine
    node.peers = {}
    node.heartbeats = {}
    node.logger = MagicMock()
    node.node_client = MagicMock()
    node.net_lock = asyncio.Lock()
    node.public_host = "127.0.0.1:0"
    node._is_mining = False

    # Sync state
    node._synchronized = threading.Event()
    node._has_ever_had_peers = False
    node.sync_block_cache = {}
    node._sync_failure_count = 0
    node._last_sync_target = 0
    node._max_sync_failures = 3

    # Telemetry stub
    node.telemetry = MagicMock()

    # Ban list stub (from our earlier work)
    from shared.peer_ban_list import PeerBanList
    node._ban_list = PeerBanList()
    node._own_ban_list = node._ban_list

    # Callbacks
    node.on_new_node = None
    node.on_node_lost = None

    # Stubs for methods called by add_peer / remove_node
    node.add_or_update_peer = MagicMock(return_value=True)
    node.get_latest_block = MagicMock()
    node.peer_versions = {}

    # node_client methods that are awaited need AsyncMock
    node.node_client.remove_peer = AsyncMock()
    node.node_client.add_peer = MagicMock()

    # info() is called by send_heartbeat — stub it
    node.info = MagicMock(return_value=MagicMock())

    return node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bootstrap_sets_synchronized():
    """With auto_mine and no peers ever, check_synchronized sets synchronized."""
    node = _make_network_node(auto_mine=True)
    node.get_latest_block.return_value = MagicMock(header=MagicMock(index=0))

    result = await node.check_synchronized()
    assert result == 0
    assert node.synchronized is True


@pytest.mark.asyncio
async def test_desync_clears_synchronized_on_last_peer_removal():
    """Removing the last peer should clear synchronized state."""
    node = _make_network_node(auto_mine=True)

    # Simulate having connected to a peer
    node._has_ever_had_peers = True
    node._synchronized.set()
    node.peers = {"peer1:20049": MagicMock()}

    assert node.synchronized is True

    # Remove the last peer
    await node.remove_node("peer1:20049")

    assert node.synchronized is False
    assert not node.peers


@pytest.mark.asyncio
async def test_desync_blocks_mining_via_check_synchronized():
    """After desync (lost all peers), check_synchronized returns 0 but
    synchronized remains False, so the server_loop mining guard blocks mining.
    """
    node = _make_network_node(auto_mine=True)
    node._has_ever_had_peers = True
    # _synchronized is NOT set (cleared by remove_node)
    node.get_latest_block.return_value = MagicMock(header=MagicMock(index=0))

    result = await node.check_synchronized()
    assert result == 0  # nothing to sync to (no peers)
    assert node.synchronized is False  # but mining is still blocked


@pytest.mark.asyncio
async def test_partial_peer_loss_keeps_synchronized():
    """Losing one peer when others remain should NOT clear sync state."""
    node = _make_network_node(auto_mine=True)
    node._has_ever_had_peers = True
    node._synchronized.set()
    node.peers = {
        "peer1:20049": MagicMock(),
        "peer2:20049": MagicMock(),
    }

    await node.remove_node("peer1:20049")

    assert node.synchronized is True
    assert "peer2:20049" in node.peers


@pytest.mark.asyncio
async def test_add_peer_sets_has_ever_had_peers():
    """Adding a new peer should set _has_ever_had_peers."""
    node = _make_network_node(auto_mine=True)
    assert node._has_ever_had_peers is False

    from shared.block import MinerInfo
    info = MinerInfo(
        miner_id="test",
        miner_type="cpu",
        reward_address=b"\x00" * 33,
        ecdsa_public_key=b"\x00" * 33,
        wots_public_key=b"\x00" * 32,
        next_wots_public_key=b"\x00" * 32,
    )

    await node.add_peer("new-peer:20049", info)
    assert node._has_ever_had_peers is True


@pytest.mark.asyncio
async def test_heartbeat_works_when_desynced():
    """Heartbeats should be sent even when not synchronized."""
    node = _make_network_node(auto_mine=True)
    node._has_ever_had_peers = True
    # _synchronized is NOT set
    assert node.synchronized is False

    node.node_client.send_heartbeat = AsyncMock(return_value=True)
    result = await node.send_heartbeat("peer1:20049")
    assert result is True
    node.node_client.send_heartbeat.assert_called_once()


@pytest.mark.asyncio
async def test_chain_reset_clears_has_ever_had_peers():
    """Chain reset should reset _has_ever_had_peers for fresh bootstrap."""
    node = _make_network_node(auto_mine=True)
    node._has_ever_had_peers = True
    node._synchronized.set()
    node.chain = [MagicMock()]  # single genesis block
    node.chain_lock = asyncio.Lock()
    node.genesis_block = node.chain[0]
    node.previous_epoch = None
    node.enable_epoch_storage = False
    node.epoch_block_store = None
    node.reset_scheduled = True
    node.reset_start_time = 1.0
    node.reset_timer_task = None

    await node._execute_chain_reset()

    assert node._has_ever_had_peers is False
    assert node.synchronized is False


@pytest.mark.asyncio
async def test_no_auto_mine_raises_with_no_peers():
    """Without auto_mine, check_synchronized should raise when no peers."""
    node = _make_network_node(auto_mine=False)
    node.get_latest_block.return_value = MagicMock(header=MagicMock(index=0))

    with pytest.raises(RuntimeError, match="No peers to synchronize with"):
        await node.check_synchronized()
