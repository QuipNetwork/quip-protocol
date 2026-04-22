"""Regression tests for TelemetryManager epoch-hash detection.

The bug (original): ``TelemetryManager`` only set its epoch key when
``record_block`` saw a block with index 1. ``sync_epoch_from_chain`` can
fix it up from an existing chain, but it was only called once during
``NetworkNode.start()``, at which point the chain only had genesis. If
block 1 then arrived via sync or gossip, nothing triggered the detector
again, so every subsequent ``record_block(N)`` wrote to
``telemetry/genesis/{N}.json``. ``TelemetryCache`` ignores that directory
because ``"genesis".isdigit()`` is False, leaving ``latest_block_index``
permanently stuck at 0 (or the pre-restart value from an older
digit-named epoch dir).

Fix: ``_on_block_received`` re-runs ``sync_epoch_from_chain`` on the
current chain before recording. Re-keying on block-1 hash change is also
supported so the telemetry dir follows the canonical chain on a
reorg-at-height-1.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.network_node import NetworkNode
from shared.telemetry import TelemetryManager


def _block(index: int, timestamp: int, block_hash: bytes | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            index=index,
            timestamp=timestamp,
            previous_hash=b"\x00" * 32,
        ),
        hash=block_hash if block_hash is not None else bytes([index % 256]) * 32,
        miner_info=SimpleNamespace(
            miner_id="m",
            miner_type="CPU",
            ecdsa_public_key=b"\x01" * 32,
        ),
        quantum_proof=SimpleNamespace(
            energy=-1.0,
            diversity=0.0,
            num_valid_solutions=0,
            mining_time=0.0,
            nonce=0,
            nodes=[],
            edges=[],
        ),
        next_block_requirements=None,
    )


def _bare_network_node(tmp_path) -> NetworkNode:
    """Skip NetworkNode.__init__; wire up only the telemetry path."""
    node = object.__new__(NetworkNode)
    node.logger = MagicMock()
    node.max_sync_block_index = 1024
    node.reset_scheduled = False
    node.on_block_received = None
    node._stats_cache = None
    node.telemetry = TelemetryManager(
        telemetry_dir=str(tmp_path),
        enabled=True,
    )
    return node


def test_on_block_received_detects_epoch_from_chain(tmp_path):
    """A mid-epoch block must land in a hash-named epoch directory.

    Simulates a node that has synced [genesis, block_1, ..., block_5] and
    is now receiving gossip for block 6. Telemetry writes to
    {block_1.hash[:16]}/6.json — not to genesis/6.json — so the cache can
    surface it.
    """
    node = _bare_network_node(tmp_path)
    block_1_hash = bytes([0xAB]) * 32
    node.chain = [
        _block(index=0, timestamp=0),
        _block(index=1, timestamp=42_000, block_hash=block_1_hash),
        _block(index=2, timestamp=42_100),
        _block(index=3, timestamp=42_200),
        _block(index=4, timestamp=42_300),
        _block(index=5, timestamp=42_400),
    ]

    assert node.telemetry._block_1_hash is None

    incoming = _block(index=6, timestamp=42_500)
    node._on_block_received(incoming)

    expected_dir = block_1_hash.hex()[:16]
    assert (tmp_path / expected_dir).is_dir()
    assert not (tmp_path / "genesis").exists(), (
        "block 6 should not land under genesis/ — the cache filters it out"
    )
    assert (tmp_path / expected_dir / "6.json").is_file()


def test_sync_epoch_updates_on_block_1_hash_change(tmp_path):
    """Reorg at height 1: the epoch key must follow the canonical chain.

    Hash-keyed telemetry is content-addressed, so a changed block 1 means
    a changed chain. We re-key to follow it. The prior dir stays on disk
    and is labelled stale_fork by the cache using current_epoch.json.
    """
    tm = TelemetryManager(telemetry_dir=str(tmp_path), enabled=True)

    hash_a = bytes([0x11]) * 32
    hash_b = bytes([0x22]) * 32

    tm.sync_epoch_from_chain([_block(0, 0), _block(1, 999, hash_a)])
    assert tm._block_1_hash == hash_a

    tm.sync_epoch_from_chain([_block(1, 111, hash_b)])
    assert tm._block_1_hash == hash_b


def test_subsequent_blocks_all_land_in_detected_epoch(tmp_path):
    """Once the epoch is detected, every later block must land in it too.

    Guards against a regression where we might re-check the chain on each
    incoming block and accidentally reset or shift _block_1_hash when the
    chain is unchanged. The cache scan that feeds
    /api/v1/telemetry/status.latest_block_index only surfaces hex-named
    dirs, so every record_block() after detection must keep writing to
    the same epoch dir for the dashboard to see incrementing values.
    """
    node = _bare_network_node(tmp_path)
    block_1_hash = bytes([0xAB]) * 32
    node.chain = [
        _block(index=0, timestamp=0),
        _block(index=1, timestamp=42_000, block_hash=block_1_hash),
    ]

    for idx in range(2, 12):
        new_block = _block(index=idx, timestamp=42_000 + idx * 100)
        node.chain.append(new_block)
        node._on_block_received(new_block)

    epoch_dir = tmp_path / block_1_hash.hex()[:16]
    written = sorted(int(p.stem) for p in epoch_dir.glob("*.json"))
    assert written == list(range(2, 12)), (
        f"expected blocks 2..11 written under {epoch_dir.name}/, got {written}; "
        f"tmp_path contents: {[p.name for p in tmp_path.iterdir()]}"
    )
    assert node.telemetry._block_1_hash == block_1_hash


def test_block_1_arriving_as_incoming_sets_epoch_via_record_block(tmp_path):
    """Block 1 arriving via gossip on a chain holding only genesis must
    still land in its own hash-named epoch dir, not under genesis/.

    sync_epoch_from_chain runs first but finds no block 1 (only genesis
    is in the chain at this point — the incoming block has not been
    appended yet). record_block then sets the hash via its existing
    ``idx == 1 and _block_1_hash is None`` branch.
    """
    node = _bare_network_node(tmp_path)
    node.chain = [_block(index=0, timestamp=0)]

    block_1_hash = bytes([0xCD]) * 32
    incoming = _block(index=1, timestamp=42_000, block_hash=block_1_hash)
    node._on_block_received(incoming)

    expected_dir = block_1_hash.hex()[:16]
    assert (tmp_path / expected_dir / "1.json").is_file()
    assert not (tmp_path / "genesis").exists()
    assert node.telemetry._block_1_hash == block_1_hash


def test_on_block_received_handles_empty_chain(tmp_path):
    """An empty chain must not raise (defensive against early-startup races)."""
    node = _bare_network_node(tmp_path)
    node.chain = []

    node._on_block_received(_block(index=2, timestamp=42_200))
    assert node.telemetry._block_1_hash is None
