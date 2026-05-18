"""Unit tests for the fork-aware ``BlockSynchronizer``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from _utils import hash_for as _hash
from shared.block_synchronizer import BlockSynchronizer, TipGroup


def _fake_block(index: int, block_hash: bytes, previous_hash: bytes = b""):
    return SimpleNamespace(
        header=SimpleNamespace(index=index, previous_hash=previous_hash),
        hash=block_hash,
    )


def _make_sync(
    *,
    peers: Optional[Dict[str, object]] = None,
    local_tip_hash: bytes = _hash(0),
    local_blocks_by_hash: Optional[Dict[bytes, object]] = None,
    receive_block_queue: Optional[asyncio.Queue] = None,
) -> BlockSynchronizer:
    client = MagicMock()
    client.peers = peers if peers is not None else {}
    client.get_peer_block = AsyncMock()
    client.get_chain_manifest = AsyncMock()
    client.get_peer_block_by_hash = AsyncMock()

    tip = _fake_block(0, local_tip_hash)
    by_hash = local_blocks_by_hash if local_blocks_by_hash is not None else {
        local_tip_hash: tip
    }

    return BlockSynchronizer(
        node_client=client,
        receive_block_queue=receive_block_queue or asyncio.Queue(),
        local_tip=lambda: tip,
        local_locator=lambda: [local_tip_hash],
        local_get_block_by_hash=by_hash.get,
        max_in_flight=4,
    )


async def _fake_processor(queue: asyncio.Queue, *, reject_index: Optional[int] = None):
    while True:
        block, future, _force_reorg, _source = await queue.get()
        future.set_result(reject_index is None or block.header.index != reject_index)
        if reject_index is not None and block.header.index == reject_index:
            return


# ---------------------------------------------------------------------------
# Phase 1 — tip discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_tips_groups_ranks_and_filters():
    """Group by head hash, rank by (height, peers, hash), drop tips <= local."""
    sync = _make_sync(peers={"A": {}, "B": {}, "C": {}, "D": {}})
    h_tall = _hash(100)
    h_mid_a = b"\x01" + b"\x00" * 31
    h_mid_b = b"\x02" + b"\x00" * 31
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(20, h_tall),   # A — tallest, unique
        _fake_block(15, h_mid_b),  # B — lex-larger
        _fake_block(15, h_mid_a),  # C — tied with D on hash
        _fake_block(15, h_mid_a),  # D
    ]
    groups, surveyed, failed = await sync._discover_tips(
        local_height=0, max_height_hint=None
    )
    assert [(g.height, sorted(g.peers)) for g in groups] == [
        (20, ["A"]),
        (15, ["C", "D"]),  # more peers wins the tiebreak
        (15, ["B"]),
    ]
    assert (surveyed, failed) == (4, 0)


@pytest.mark.asyncio
async def test_discover_tips_honors_hints_and_handles_failures():
    sync = _make_sync(peers={"A": {}, "B": {}, "C": {}})
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(100, _hash(1)),  # A — in range
        _fake_block(500, _hash(2)),  # B — exceeds hint, skip
        None,                         # C — fetch failure, skip
    ]
    groups, surveyed, failed = await sync._discover_tips(
        local_height=0, max_height_hint=200
    )
    assert [(g.height, g.peers) for g in groups] == [(100, ["A"])]
    assert (surveyed, failed) == (3, 1)


# ---------------------------------------------------------------------------
# Phase 2 — manifest fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_fetch_single_peer_and_cross_check_agreement():
    head = _hash(5)
    good = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    sync = _make_sync()
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[good, good[:2]])

    # Two peers in the group: both agree on overlap → accept primary's manifest.
    group = TipGroup(height=5, head_hash=head, peers=["A", "B"])
    assert await sync._fetch_manifest(group, set()) == good


@pytest.mark.asyncio
async def test_manifest_fetch_disagreement_demotes_both_peers():
    head = _hash(5)
    a = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    b = [(1, _hash(1)), (2, _hash(99)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    sync = _make_sync()
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[a, b])

    backoff: set = set()
    assert await sync._fetch_manifest(
        TipGroup(height=5, head_hash=head, peers=["A", "B"]), backoff
    ) is None
    assert backoff == {"A", "B"}


@pytest.mark.asyncio
async def test_manifest_fetch_fails_when_head_not_reached():
    """Peer returns entries but never covers the advertised head."""
    sync = _make_sync()
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[
        [(1, _hash(1)), (2, _hash(2))],  # partial
        [],                               # nothing more
    ])
    assert await sync._fetch_manifest(
        TipGroup(height=10, head_hash=_hash(10), peers=["A"]), set()
    ) is None


# ---------------------------------------------------------------------------
# Phase 3 — download with linkage audit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_happy_path():
    h0, h1, h2, h3 = _hash(0), _hash(1), _hash(2), _hash(3)
    sync = _make_sync(
        local_tip_hash=h0, local_blocks_by_hash={h0: _fake_block(0, h0)}
    )
    mapping = {
        h1: _fake_block(1, h1, previous_hash=h0),
        h2: _fake_block(2, h2, previous_hash=h1),
        h3: _fake_block(3, h3, previous_hash=h2),
    }
    sync.node_client.get_peer_block_by_hash = AsyncMock(
        side_effect=lambda _peer, h: mapping[h]
    )
    downloaded = await sync._download_blocks(
        TipGroup(height=3, head_hash=h3, peers=["A"]),
        [(1, h1), (2, h2), (3, h3)],
        set(),
    )
    assert downloaded is not None and set(downloaded) == {1, 2, 3}


@pytest.mark.asyncio
async def test_download_demotes_peer_on_linkage_mismatch():
    h0, h1, h2 = _hash(0), _hash(1), _hash(2)
    sync = _make_sync(
        local_tip_hash=h0, local_blocks_by_hash={h0: _fake_block(0, h0)}
    )
    # Block 2's previous_hash points at garbage, not manifest[0].hash.
    sync.node_client.get_peer_block_by_hash = AsyncMock(side_effect=lambda _p, h: {
        h1: _fake_block(1, h1, previous_hash=h0),
        h2: _fake_block(2, h2, previous_hash=_hash(999)),
    }[h])
    backoff: set = set()
    assert await sync._download_blocks(
        TipGroup(height=2, head_hash=h2, peers=["A"]), [(1, h1), (2, h2)], backoff
    ) is None
    assert "A" in backoff


@pytest.mark.asyncio
async def test_download_demotes_peer_when_first_block_parent_unknown_locally():
    h0, h1 = _hash(0), _hash(1)
    sync = _make_sync(
        local_tip_hash=h0, local_blocks_by_hash={h0: _fake_block(0, h0)}
    )
    sync.node_client.get_peer_block_by_hash = AsyncMock(
        return_value=_fake_block(1, h1, previous_hash=_hash(777))
    )
    backoff: set = set()
    assert await sync._download_blocks(
        TipGroup(height=1, head_hash=h1, peers=["A"]), [(1, h1)], backoff
    ) is None
    assert "A" in backoff


# ---------------------------------------------------------------------------
# Phase 4 — commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_happy_and_abort_paths():
    h0, h1, h2 = _hash(0), _hash(1), _hash(2)
    manifest = [(1, h1), (2, h2)]
    downloaded = {
        1: _fake_block(1, h1, previous_hash=h0),
        2: _fake_block(2, h2, previous_hash=h1),
    }

    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(receive_block_queue=queue)
    processor = asyncio.create_task(_fake_processor(queue))
    try:
        assert await sync._commit(manifest, downloaded) == (2, None)
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    # Now simulate rejection at index 2.
    queue = asyncio.Queue()
    sync.receive_block_queue = queue
    processor = asyncio.create_task(_fake_processor(queue, reject_index=2))
    try:
        assert await sync._commit(manifest, downloaded) == (1, 2)
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# End-to-end sync_blocks orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_blocks_returns_success_when_no_peers_advertise_longer_chain():
    sync = _make_sync(peers={})
    result = await sync.sync_blocks()
    assert result.success and result.committed == 0 and result.groups_tried == 0


@pytest.mark.asyncio
async def test_sync_blocks_falls_through_to_next_group_on_disagreement():
    """Two peers in the top group disagree; synchronizer demotes both and picks the next group."""
    h0 = _hash(0)
    bad_head, good_head = _hash(100), _hash(200)
    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(
        peers={"A": {}, "B": {}, "C": {}},
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
        receive_block_queue=queue,
    )
    sync.node_client.get_peer_block = AsyncMock(side_effect=[
        _fake_block(2, bad_head),
        _fake_block(2, bad_head),
        _fake_block(2, good_head),
    ])
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[
        [(1, _hash(1)), (2, bad_head)],   # A — good
        [(1, _hash(9)), (2, bad_head)],   # B — disagrees at idx 1
        [(1, _hash(11)), (2, good_head)], # C — honest, picked next
    ])
    good = {
        _hash(11): _fake_block(1, _hash(11), previous_hash=h0),
        good_head: _fake_block(2, good_head, previous_hash=_hash(11)),
    }
    sync.node_client.get_peer_block_by_hash = AsyncMock(
        side_effect=lambda _p, h: good[h]
    )

    processor = asyncio.create_task(_fake_processor(queue))
    try:
        result = await sync.sync_blocks()
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert result.success
    assert result.target_hash == good_head
    assert result.groups_tried == 2
    assert result.peers_backed_off == 2  # A and B
