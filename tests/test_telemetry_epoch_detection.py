"""Regression tests for TelemetryManager epoch-timestamp detection.

The bug: ``TelemetryManager._epoch_timestamp`` is only set when
``record_block`` sees a block with index 1. ``sync_epoch_from_chain`` can
fix it up from an existing chain, but it was only called once during
``NetworkNode.start()``, at which point the chain only had genesis. If
block 1 then arrived via sync or gossip, nothing triggered the detector
again, so every subsequent ``record_block(N)`` wrote to ``telemetry/
genesis/{N}.json``. ``TelemetryCache`` ignores that directory because
``"genesis".isdigit()`` is False, leaving ``latest_block_index``
permanently stuck at 0 (or the pre-restart value from an older
digit-named epoch dir).

Fix: have ``_on_block_received`` re-run ``sync_epoch_from_chain`` on the
current chain before recording. It early-returns after the first
successful detection, so the cost is a single dict lookup per block.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.network_node import NetworkNode
from shared.telemetry import TelemetryManager


def _block(index: int, timestamp: int) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            index=index,
            timestamp=timestamp,
            previous_hash=b"\x00" * 32,
        ),
        hash=bytes([index % 256]) * 32,
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
    """A mid-epoch block must land in an epoch directory, never genesis/.

    Simulates the scenario where the node has already synced [genesis,
    block_1, ..., block_5] and is now receiving gossip for block 6. The
    telemetry should write to {block_1.timestamp}/6.json — not to
    genesis/6.json — so that TelemetryCache can surface it.
    """
    node = _bare_network_node(tmp_path)
    node.chain = [
        _block(index=0, timestamp=0),      # genesis
        _block(index=1, timestamp=42_000), # block 1 sets the epoch key
        _block(index=2, timestamp=42_100),
        _block(index=3, timestamp=42_200),
        _block(index=4, timestamp=42_300),
        _block(index=5, timestamp=42_400),
    ]

    # Fresh telemetry manager — mimics a restarted node that hasn't seen
    # block 1 go through record_block() yet.
    assert node.telemetry._epoch_timestamp is None

    incoming = _block(index=6, timestamp=42_500)
    node._on_block_received(incoming)

    epoch_dir = tmp_path / "42000"
    genesis_dir = tmp_path / "genesis"
    assert epoch_dir.is_dir(), (
        f"expected {epoch_dir} to be created from block_1.timestamp; "
        f"instead got contents: {list(tmp_path.iterdir())}"
    )
    assert not genesis_dir.exists(), (
        "block 6 should not be written under genesis/ — "
        "TelemetryCache filters that directory out"
    )
    assert (epoch_dir / "6.json").is_file()


def test_sync_epoch_does_not_overwrite_existing_timestamp(tmp_path):
    """Once detected, the epoch timestamp must not shift on later calls.

    The early-return guard makes the per-block call cheap and prevents
    the epoch dir from jumping around mid-run if the chain is later
    rewritten (e.g. via reorg of block 1, however unlikely).
    """
    tm = TelemetryManager(telemetry_dir=str(tmp_path), enabled=True)

    chain = [_block(index=0, timestamp=0), _block(index=1, timestamp=999)]
    tm.sync_epoch_from_chain(chain)
    assert tm._epoch_timestamp == 999

    # Replacing block 1 with a different timestamp should NOT re-mutate —
    # we don't want the epoch dir jumping around mid-run.
    tm.sync_epoch_from_chain([_block(index=1, timestamp=111)])
    assert tm._epoch_timestamp == 999


def test_subsequent_blocks_all_land_in_detected_epoch(tmp_path):
    """Once the epoch is detected, every later block must land in it too.

    Guards against a regression where we might re-check the chain on each
    incoming block and accidentally reset or shift _epoch_timestamp. The
    TelemetryCache scan that feeds /api/v1/telemetry/status.latest_block_index
    only looks at digit-named directories, so every record_block() after
    the detection must keep writing to the same epoch dir for the
    dashboard to see incrementing values.
    """
    node = _bare_network_node(tmp_path)
    node.chain = [
        _block(index=0, timestamp=0),
        _block(index=1, timestamp=42_000),
    ]

    # Walk the chain forward, appending each new block before handing it
    # to _on_block_received — the order mimics sync/gossip arrival.
    for idx in range(2, 12):
        new_block = _block(index=idx, timestamp=42_000 + idx * 100)
        node.chain.append(new_block)
        node._on_block_received(new_block)

    epoch_dir = tmp_path / "42000"
    written = sorted(int(p.stem) for p in epoch_dir.glob("*.json"))
    assert written == list(range(2, 12)), (
        f"expected blocks 2..11 written under 42000/, got {written}; "
        f"tmp_path contents: {[p.name for p in tmp_path.iterdir()]}"
    )
    # _epoch_timestamp must not drift after initial detection.
    assert node.telemetry._epoch_timestamp == 42_000


def test_block_1_arriving_as_incoming_sets_epoch_via_record_block(tmp_path):
    """Block 1 arriving via gossip on a chain holding only genesis must
    still land in its own epoch dir, not under genesis/.

    sync_epoch_from_chain runs first but finds no block 1 (only genesis
    is in the chain at this point — the incoming block has not been
    appended yet). record_block then sets the epoch via its existing
    ``idx == 1 and _epoch_timestamp is None`` branch. Both paths must
    leave the file under {timestamp}/1.json.
    """
    node = _bare_network_node(tmp_path)
    node.chain = [_block(index=0, timestamp=0)]  # genesis only

    incoming = _block(index=1, timestamp=42_000)
    node._on_block_received(incoming)

    assert (tmp_path / "42000" / "1.json").is_file()
    assert not (tmp_path / "genesis").exists(), (
        "block 1 must never land under genesis/ — TelemetryCache "
        "filters that directory out"
    )
    assert node.telemetry._epoch_timestamp == 42_000


def test_on_block_received_handles_empty_chain(tmp_path):
    """An empty chain must not raise (defensive against early-startup races)."""
    node = _bare_network_node(tmp_path)
    node.chain = []

    # Should be a no-op for sync_epoch_from_chain (no block 1 to find)
    # and then write the incoming block to genesis/ since no epoch is
    # known yet. The contract here is "do not raise" — the genesis
    # routing is acceptable until block 1 arrives and triggers a real
    # epoch detection on the next call.
    node._on_block_received(_block(index=2, timestamp=42_200))
    assert node.telemetry._epoch_timestamp is None
