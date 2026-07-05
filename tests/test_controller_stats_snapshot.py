"""Verify the controller writes its stats snapshot to the expected path."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_controller_writes_stats_snapshot(tmp_path: Path):
    """The controller's StatsSnapshotWriter populates the snapshot file periodically."""
    from substrate.miner_controller import build_stats_snapshot_for_telemetry

    # Set up controller-shaped stats container (matches today's `self.stats`).
    fake_stats = {
        "heads_observed": 100,
        "contexts_dispatched": 50,
        "proofs_submitted": 17,
        "active_url": "http://validator-a",
    }

    snapshot_path = tmp_path / "telemetry-stats.json"

    from shared.stats_snapshot import StatsSnapshotWriter, read_snapshot
    writer = StatsSnapshotWriter(
        path=snapshot_path,
        get_snapshot=lambda: build_stats_snapshot_for_telemetry(_fake_controller_for(fake_stats)),
        interval_s=0.01,
    )
    shutdown = asyncio.Event()
    seen: dict = {}

    async def stop_after_one_write():
        await asyncio.sleep(0.05)
        # Read before shutdown: graceful shutdown removes the file.
        seen["snap"] = read_snapshot(snapshot_path)
        shutdown.set()

    await asyncio.gather(writer.run(shutdown), stop_after_one_write())

    snap = seen["snap"]
    assert snap is not None
    assert snap["controller"]["heads_observed"] == 100
    assert snap["controller"]["proofs_submitted"] == 17
    assert snap["controller"]["active_url"] == "http://validator-a"


def test_snapshot_includes_pool_sync_state():
    """The pool's last_sync_state rides the snapshot so dashboards see it."""
    from substrate.miner_controller import build_stats_snapshot_for_telemetry

    c = _fake_controller_for({"heads_observed": 1})
    c.pool = type(
        "P",
        (),
        {"last_sync_state": {"is_syncing": True, "current_block": 5, "url": "ws://a"}},
    )()
    snap = build_stats_snapshot_for_telemetry(c)
    assert snap["sync_state"] == {
        "is_syncing": True,
        "current_block": 5,
        "url": "ws://a",
    }


def test_snapshot_sync_state_none_without_pool():
    """Controllers without a pool attribute (tests, legacy) degrade to None."""
    from substrate.miner_controller import build_stats_snapshot_for_telemetry

    snap = build_stats_snapshot_for_telemetry(_fake_controller_for({}))
    assert snap["sync_state"] is None


def _fake_controller_for(stats: dict):
    """Minimal controller stand-in exposing the fields build_stats_snapshot_for_telemetry reads."""
    class _C:
        pass
    c = _C()
    c.stats = type("S", (), stats)()
    for k, v in stats.items():
        setattr(c.stats, k, v)
    c.pool_active_url = stats.get("active_url")
    return c
