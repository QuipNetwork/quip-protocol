"""Tests that Node.receive_block cancels in-progress mining on a peer win.

Regression coverage for the case where a peer broadcast a winning block at
our target height while we were still mining. Before the fix, we kept
mining until our own miner returned — producing a valid-but-stale solution
that consensus already rejected. The expected behaviour: as soon as we
accept a peer's block into the chain, we must tear down the active mining
attempt so we can pivot to the new tip.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.node import Node


def _bare_node() -> Node:
    """Build a Node stripped down to just what receive_block touches.

    Uses ``object.__new__`` to skip __init__ so we don't spin up real
    miner workers, logging queues, or BlockSigner — we only need the
    chain index, the lock, the mining flag, and the logger.
    """
    node = object.__new__(Node)
    node.chain = []
    node.chain_by_hash = {}
    node.chain_lock = asyncio.Lock()
    node.logger = MagicMock()
    node._is_mining = False
    node._mining_stop_event = None
    node.on_block_mined = None
    # Seed a genesis so get_latest_block() works and _index_append has
    # something to truncate against.
    genesis = SimpleNamespace(header=SimpleNamespace(index=0), hash=b"\x00" * 32)
    node._index_append(genesis)
    return node


def _peer_block(index: int = 1, block_hash: bytes = b"\x11" * 32):
    return SimpleNamespace(
        header=SimpleNamespace(index=index, timestamp=1),
        hash=block_hash,
        miner_info=SimpleNamespace(miner_id="peer-node"),
    )


@pytest.mark.asyncio
async def test_receive_block_stops_mining_when_peer_wins():
    """Accepting a peer's block at our mining target must trigger stop_mining."""
    node = _bare_node()
    node._is_mining = True

    # Patch check_block to accept (keeps the test focused on the
    # post-accept branch where the fix lives).
    node.check_block = AsyncMock(return_value=(True, None))
    # Record stop_mining calls rather than running the real cancel flow.
    node.stop_mining = AsyncMock()

    accepted, reason = await node.receive_block(_peer_block(index=1))

    # Let the fire-and-forget task scheduled in receive_block run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert accepted
    assert reason is None
    assert node.stop_mining.await_count == 1, (
        "receive_block must cancel active mining when a peer's block is "
        "accepted; otherwise we waste effort mining on a now-stale tip"
    )


@pytest.mark.asyncio
async def test_receive_block_noop_when_not_mining():
    """Accepting a peer's block while idle must not invoke stop_mining."""
    node = _bare_node()
    node._is_mining = False

    node.check_block = AsyncMock(return_value=(True, None))
    node.stop_mining = AsyncMock()

    await node.receive_block(_peer_block(index=1))
    await asyncio.sleep(0)

    assert node.stop_mining.await_count == 0


@pytest.mark.asyncio
async def test_receive_block_stops_mining_on_reorg():
    """A reorg (peer block at the same height with a different hash) must
    also tear down mining.

    The fix's docstring is explicit: "Any accepted block makes our
    current mining target stale." That includes reorgs, where the height
    is unchanged but the canonical tip flips. Reorgs go through
    _index_truncate (a different branch from the simple append) — if
    stop_mining were ever moved inside an ``index == head + 1`` guard
    during a future refactor, reorgs would silently leave the miner
    burning cycles on a now-stale target.
    """
    node = _bare_node()
    # Seed an additional block 1 we already accepted; the peer's block 1
    # below has a different hash, so receive_block will go down the
    # reorg/truncate path.
    existing_block_1 = SimpleNamespace(
        header=SimpleNamespace(index=1), hash=b"\xaa" * 32,
    )
    node._index_append(existing_block_1)
    node._is_mining = True

    node.check_block = AsyncMock(return_value=(True, None))
    node.stop_mining = AsyncMock()

    await node.receive_block(_peer_block(index=1, block_hash=b"\x22" * 32))

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert node.stop_mining.await_count == 1, (
        "reorg at the current mining height must trigger stop_mining "
        "just like a forward-progress peer win"
    )
