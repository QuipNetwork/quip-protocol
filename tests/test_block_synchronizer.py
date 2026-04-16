"""Unit tests for the fork-aware ``BlockSynchronizer``.

Covers each phase in isolation with a mocked ``NodeClient``:

  * Phase 1 — tip discovery, grouping, ranking
  * Phase 2 — manifest fetch + cross-check + pagination
  * Phase 3 — block-by-hash download + linkage audit
  * Phase 4 — commit via the node's processing queue
  * ``sync_blocks`` end-to-end flow on mocked transport
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.block_synchronizer import (
    BlockSynchronizer,
    SyncResult,
    TipGroup,
)


def _hash(i: int) -> bytes:
    return i.to_bytes(32, "big")


def _fake_block(
    index: int,
    block_hash: bytes,
    previous_hash: bytes | None = None,
):
    """Minimal ``Block`` stand-in used by the synchronizer."""
    return SimpleNamespace(
        header=SimpleNamespace(
            index=index,
            previous_hash=previous_hash if previous_hash is not None else b"",
        ),
        hash=block_hash,
    )


def _make_sync(
    *,
    peers: Optional[Dict[str, object]] = None,
    local_tip_index: int = 0,
    local_tip_hash: bytes = _hash(0),
    local_locator: Optional[List[bytes]] = None,
    local_blocks_by_hash: Optional[Dict[bytes, object]] = None,
    receive_block_queue: Optional[asyncio.Queue] = None,
    max_in_flight: int = 4,
) -> BlockSynchronizer:
    """Assemble a synchronizer with mocked dependencies."""
    client = MagicMock()
    client.peers = peers if peers is not None else {}
    client.get_peer_block = AsyncMock()
    client.get_chain_manifest = AsyncMock()
    client.get_peer_block_by_hash = AsyncMock()

    tip = _fake_block(local_tip_index, local_tip_hash)
    locator = local_locator if local_locator is not None else [local_tip_hash]
    by_hash = local_blocks_by_hash if local_blocks_by_hash is not None else {
        local_tip_hash: tip
    }

    return BlockSynchronizer(
        node_client=client,
        receive_block_queue=receive_block_queue or asyncio.Queue(),
        local_tip=lambda: tip,
        local_locator=lambda: list(locator),
        local_get_block_by_hash=by_hash.get,
        max_in_flight=max_in_flight,
    )


# ---------------------------------------------------------------------------
# Phase 1 — _discover_tips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_tips_groups_peers_by_head_hash():
    sync = _make_sync(
        peers={"A:1": {}, "B:1": {}, "C:1": {}},
        local_tip_index=0,
    )
    # A and B share head hash h_tall, C is on h_short (same height).
    h_tall = _hash(100)
    h_short = _hash(101)
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(10, h_tall),   # A
        _fake_block(10, h_tall),   # B
        _fake_block(10, h_short),  # C
    ]
    groups = await sync._discover_tips(local_height=0, max_height_hint=None)

    assert len(groups) == 2
    # Ranked: equal height, more peers wins.
    assert groups[0].head_hash == h_tall
    assert set(groups[0].peers) == {"A:1", "B:1"}
    assert groups[1].head_hash == h_short
    assert groups[1].peers == ["C:1"]


@pytest.mark.asyncio
async def test_discover_tips_ranks_by_height_then_peers_then_hash():
    sync = _make_sync(peers={"A:1": {}, "B:1": {}, "C:1": {}, "D:1": {}})
    h_high = _hash(50)
    h_mid_a = b"\x01" + b"\x00" * 31  # lex-smaller of the two mid tips
    h_mid_b = b"\x02" + b"\x00" * 31
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(20, h_high),   # A — tallest
        _fake_block(15, h_mid_b),  # B
        _fake_block(15, h_mid_a),  # C
        _fake_block(15, h_mid_a),  # D
    ]

    groups = await sync._discover_tips(local_height=0, max_height_hint=None)

    heights = [g.height for g in groups]
    assert heights == [20, 15, 15]
    # Among the two height-15 groups, the one with more peers comes first;
    # then by hash lexicographic order.
    assert groups[1].head_hash == h_mid_a  # 2 peers
    assert groups[2].head_hash == h_mid_b  # 1 peer


@pytest.mark.asyncio
async def test_discover_tips_filters_tips_not_strictly_longer():
    sync = _make_sync(peers={"A:1": {}, "B:1": {}}, local_tip_index=10)
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(10, _hash(1)),  # same height — skip
        _fake_block(8, _hash(2)),   # shorter — skip
    ]
    assert await sync._discover_tips(local_height=10, max_height_hint=None) == []


@pytest.mark.asyncio
async def test_discover_tips_honors_height_hint_upper_bound():
    sync = _make_sync(peers={"A:1": {}, "B:1": {}})
    sync.node_client.get_peer_block.side_effect = [
        _fake_block(100, _hash(1)),  # in range
        _fake_block(500, _hash(2)),  # exceeds hint — skip
    ]
    groups = await sync._discover_tips(local_height=0, max_height_hint=200)
    assert len(groups) == 1
    assert groups[0].height == 100


@pytest.mark.asyncio
async def test_discover_tips_skips_peers_that_fail():
    sync = _make_sync(peers={"A:1": {}, "B:1": {}})
    sync.node_client.get_peer_block.side_effect = [
        None,                         # A — failed
        _fake_block(5, _hash(1)),     # B
    ]
    groups = await sync._discover_tips(local_height=0, max_height_hint=None)
    assert len(groups) == 1
    assert groups[0].peers == ["B:1"]


@pytest.mark.asyncio
async def test_discover_tips_no_peers_returns_empty():
    sync = _make_sync(peers={})
    assert await sync._discover_tips(local_height=0, max_height_hint=None) == []


# ---------------------------------------------------------------------------
# Phase 2 — _fetch_manifest
# ---------------------------------------------------------------------------


def _manifest_matches(expected, actual):
    return list(expected) == list(actual)


@pytest.mark.asyncio
async def test_manifest_fetch_single_peer_happy_path():
    head = _hash(5)
    entries = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    sync = _make_sync()
    sync.node_client.get_chain_manifest = AsyncMock(return_value=entries)

    group = TipGroup(height=5, head_hash=head, peers=["A:1"])
    backoff: set = set()

    manifest = await sync._fetch_manifest(group, backoff)
    assert manifest == entries
    assert backoff == set()


@pytest.mark.asyncio
async def test_manifest_fetch_cross_check_disagreement_demotes_both():
    head = _hash(5)
    good = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    bad = [(1, _hash(1)), (2, _hash(99)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    sync = _make_sync()
    # Order-sensitive: whichever peer is picked primary returns "good", the
    # other returns "bad". Both responses are tried so we can return the
    # same payloads regardless of order.
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[good, bad])

    group = TipGroup(height=5, head_hash=head, peers=["A:1", "B:1"])
    backoff: set = set()

    manifest = await sync._fetch_manifest(group, backoff)
    assert manifest is None
    assert backoff == {"A:1", "B:1"}


@pytest.mark.asyncio
async def test_manifest_fetch_agrees_on_overlap_when_secondary_shorter():
    """A secondary peer may return fewer entries but must agree on overlaps."""
    head = _hash(5)
    primary = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3)), (4, _hash(4)), (5, head)]
    secondary = [(1, _hash(1)), (2, _hash(2))]  # consistent prefix
    sync = _make_sync()
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=[primary, secondary])

    group = TipGroup(height=5, head_hash=head, peers=["A:1", "B:1"])
    backoff: set = set()
    manifest = await sync._fetch_manifest(group, backoff)
    assert manifest == primary
    assert backoff == set()


@pytest.mark.asyncio
async def test_manifest_fetch_returns_none_when_head_not_reached():
    """Peer responds but doesn't cover the pinned head; reject."""
    sync = _make_sync()
    partial = [(1, _hash(1)), (2, _hash(2))]
    sync.node_client.get_chain_manifest = AsyncMock(
        side_effect=[partial, []]  # page 1 partial; page 2 empty => stop
    )

    group = TipGroup(height=10, head_hash=_hash(10), peers=["A:1"])
    backoff: set = set()
    manifest = await sync._fetch_manifest(group, backoff)
    assert manifest is None


@pytest.mark.asyncio
async def test_manifest_fetch_all_peers_backed_off_returns_none():
    sync = _make_sync()
    # Peer raises → demoted; no peers remain.
    sync.node_client.get_chain_manifest = AsyncMock(side_effect=RuntimeError("boom"))
    group = TipGroup(height=5, head_hash=_hash(5), peers=["A:1"])
    backoff: set = set()
    manifest = await sync._fetch_manifest(group, backoff)
    assert manifest is None
    assert backoff == {"A:1"}


def test_manifests_agree_accepts_overlap_with_no_disagreement():
    a = [(1, _hash(1)), (2, _hash(2)), (3, _hash(3))]
    b = [(2, _hash(2)), (3, _hash(3)), (4, _hash(4))]
    assert BlockSynchronizer._manifests_agree(a, b) is True


def test_manifests_agree_rejects_any_disagreement():
    a = [(1, _hash(1)), (2, _hash(2))]
    b = [(2, _hash(99))]
    assert BlockSynchronizer._manifests_agree(a, b) is False


# ---------------------------------------------------------------------------
# Phase 3 — _download_blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_blocks_happy_path():
    # Manifest: (1, h1), (2, h2), (3, h3). Local tip is genesis with hash h0.
    h0, h1, h2, h3 = _hash(0), _hash(1), _hash(2), _hash(3)
    manifest = [(1, h1), (2, h2), (3, h3)]

    sync = _make_sync(
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
    )

    def fetch(peer, block_hash):
        mapping = {
            h1: _fake_block(1, h1, previous_hash=h0),
            h2: _fake_block(2, h2, previous_hash=h1),
            h3: _fake_block(3, h3, previous_hash=h2),
        }
        return mapping[block_hash]

    sync.node_client.get_peer_block_by_hash = AsyncMock(side_effect=fetch)

    group = TipGroup(height=3, head_hash=h3, peers=["A:1"])
    backoff: set = set()

    downloaded = await sync._download_blocks(group, manifest, backoff)
    assert downloaded is not None
    assert set(downloaded.keys()) == {1, 2, 3}
    assert backoff == set()


@pytest.mark.asyncio
async def test_download_rejects_wrong_index_and_demotes_peer():
    h0, h1 = _hash(0), _hash(1)
    manifest = [(1, h1)]
    sync = _make_sync(
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
    )
    # Peer returns a block with wrong index.
    sync.node_client.get_peer_block_by_hash = AsyncMock(
        return_value=_fake_block(99, h1, previous_hash=h0)
    )

    group = TipGroup(height=1, head_hash=h1, peers=["A:1"])
    backoff: set = set()

    downloaded = await sync._download_blocks(group, manifest, backoff)
    assert downloaded is None
    assert "A:1" in backoff


@pytest.mark.asyncio
async def test_download_rejects_wrong_parent_hash_and_demotes_peer():
    h0, h1, h2 = _hash(0), _hash(1), _hash(2)
    manifest = [(1, h1), (2, h2)]
    sync = _make_sync(
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
    )

    def fetch(peer, block_hash):
        if block_hash == h1:
            return _fake_block(1, h1, previous_hash=h0)
        # Block 2 advertises the wrong parent (should be h1).
        return _fake_block(2, h2, previous_hash=_hash(999))

    sync.node_client.get_peer_block_by_hash = AsyncMock(side_effect=fetch)
    group = TipGroup(height=2, head_hash=h2, peers=["A:1"])
    backoff: set = set()

    downloaded = await sync._download_blocks(group, manifest, backoff)
    assert downloaded is None
    assert "A:1" in backoff


@pytest.mark.asyncio
async def test_download_rejects_not_found_and_demotes_peer():
    h0, h1 = _hash(0), _hash(1)
    manifest = [(1, h1)]
    sync = _make_sync(
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
    )
    sync.node_client.get_peer_block_by_hash = AsyncMock(return_value=None)
    group = TipGroup(height=1, head_hash=h1, peers=["A:1"])
    backoff: set = set()
    downloaded = await sync._download_blocks(group, manifest, backoff)
    assert downloaded is None
    assert "A:1" in backoff


@pytest.mark.asyncio
async def test_download_first_block_parent_must_be_on_local_chain():
    h0, h1 = _hash(0), _hash(1)
    manifest = [(1, h1)]
    # Local chain has h0 as genesis/tip. Peer returns block whose parent
    # hash is _hash(777), which is not in our local by-hash index.
    sync = _make_sync(
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
    )
    sync.node_client.get_peer_block_by_hash = AsyncMock(
        return_value=_fake_block(1, h1, previous_hash=_hash(777))
    )
    group = TipGroup(height=1, head_hash=h1, peers=["A:1"])
    backoff: set = set()

    downloaded = await sync._download_blocks(group, manifest, backoff)
    assert downloaded is None
    assert "A:1" in backoff


# ---------------------------------------------------------------------------
# Phase 4 — _commit
# ---------------------------------------------------------------------------


async def _fake_block_processor(
    queue: asyncio.Queue,
    *,
    accept: bool = True,
    reject_index: Optional[int] = None,
):
    """Drain ``queue`` and complete each future with ``accept``.

    If ``reject_index`` is set, that specific block is rejected and the
    processor exits to mimic the real node's behavior on failure.
    """
    while True:
        block, future, force_reorg, source = await queue.get()
        if reject_index is not None and block.header.index == reject_index:
            future.set_result(False)
            return
        future.set_result(accept)


@pytest.mark.asyncio
async def test_commit_awaits_each_future_in_index_order():
    h0, h1, h2 = _hash(0), _hash(1), _hash(2)
    manifest = [(1, h1), (2, h2)]
    downloaded = {
        1: _fake_block(1, h1, previous_hash=h0),
        2: _fake_block(2, h2, previous_hash=h1),
    }
    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(receive_block_queue=queue)

    processor = asyncio.create_task(_fake_block_processor(queue, accept=True))
    try:
        committed, failed = await sync._commit(manifest, downloaded)
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert committed == 2
    assert failed is None


@pytest.mark.asyncio
async def test_commit_aborts_on_rejected_block():
    h0, h1, h2 = _hash(0), _hash(1), _hash(2)
    manifest = [(1, h1), (2, h2)]
    downloaded = {
        1: _fake_block(1, h1, previous_hash=h0),
        2: _fake_block(2, h2, previous_hash=h1),
    }
    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(receive_block_queue=queue)

    processor = asyncio.create_task(
        _fake_block_processor(queue, reject_index=2)
    )
    try:
        committed, failed = await sync._commit(manifest, downloaded)
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert committed == 1
    assert failed == 2


# ---------------------------------------------------------------------------
# End-to-end sync_blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_blocks_no_peers_returns_success():
    sync = _make_sync(peers={})
    result = await sync.sync_blocks()
    assert result.success is True
    assert result.committed == 0
    assert result.groups_tried == 0


@pytest.mark.asyncio
async def test_sync_blocks_full_flow_single_honest_peer():
    """End-to-end sync against one cooperative peer."""
    h0 = _hash(0)
    head_hash = _hash(3)
    manifest_entries = [(1, _hash(1)), (2, _hash(2)), (3, head_hash)]

    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(
        peers={"A:1": {}},
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
        receive_block_queue=queue,
    )

    sync.node_client.get_peer_block = AsyncMock(
        return_value=_fake_block(3, head_hash)
    )
    sync.node_client.get_chain_manifest = AsyncMock(return_value=manifest_entries)

    def fetch(peer, block_hash):
        mapping = {
            _hash(1): _fake_block(1, _hash(1), previous_hash=h0),
            _hash(2): _fake_block(2, _hash(2), previous_hash=_hash(1)),
            head_hash: _fake_block(3, head_hash, previous_hash=_hash(2)),
        }
        return mapping[block_hash]

    sync.node_client.get_peer_block_by_hash = AsyncMock(side_effect=fetch)

    processor = asyncio.create_task(_fake_block_processor(queue))
    try:
        result = await sync.sync_blocks()
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert result.success is True
    assert result.committed == 3
    assert result.target_height == 3
    assert result.target_hash == head_hash
    assert result.groups_tried == 1


@pytest.mark.asyncio
async def test_sync_blocks_falls_through_to_next_group_on_manifest_disagreement():
    """When two peers in one group disagree, the next candidate is tried."""
    h0 = _hash(0)
    bad_head = _hash(100)
    good_head = _hash(200)
    bad_manifest_a = [(1, _hash(1)), (2, bad_head)]
    bad_manifest_b = [(1, _hash(9)), (2, bad_head)]  # disagrees at idx 1
    good_manifest = [(1, _hash(11)), (2, good_head)]

    queue: asyncio.Queue = asyncio.Queue()
    sync = _make_sync(
        peers={"A:1": {}, "B:1": {}, "C:1": {}},
        local_tip_hash=h0,
        local_blocks_by_hash={h0: _fake_block(0, h0)},
        receive_block_queue=queue,
    )

    # Tip survey: A and B on bad_head (height 2, wins); C on good_head (same).
    # Ranked by peer count: bad_head (2 peers) first, good_head (1 peer) second.
    sync.node_client.get_peer_block = AsyncMock(
        side_effect=[
            _fake_block(2, bad_head),   # A
            _fake_block(2, bad_head),   # B
            _fake_block(2, good_head),  # C
        ]
    )
    # Manifest responses: A returns bad_a, B returns bad_b (disagree),
    # after A and B get demoted we fall to good_head's C who returns good.
    sync.node_client.get_chain_manifest = AsyncMock(
        side_effect=[bad_manifest_a, bad_manifest_b, good_manifest]
    )

    def fetch(peer, block_hash):
        mapping = {
            _hash(11): _fake_block(1, _hash(11), previous_hash=h0),
            good_head: _fake_block(2, good_head, previous_hash=_hash(11)),
        }
        return mapping[block_hash]

    sync.node_client.get_peer_block_by_hash = AsyncMock(side_effect=fetch)

    processor = asyncio.create_task(_fake_block_processor(queue))
    try:
        result = await sync.sync_blocks()
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert result.success is True
    assert result.target_hash == good_head
    assert result.groups_tried == 2
    assert result.peers_backed_off == 2  # A and B
