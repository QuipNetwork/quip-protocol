"""Verify the controller writes its stats snapshot to the expected path."""
from __future__ import annotations

import asyncio
import json
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

    async def stop_after_one_write():
        await asyncio.sleep(0.05)
        shutdown.set()

    await asyncio.gather(writer.run(shutdown), stop_after_one_write())

    snap = read_snapshot(snapshot_path)
    assert snap is not None
    assert snap["controller"]["heads_observed"] == 100
    assert snap["controller"]["proofs_submitted"] == 17
    assert snap["controller"]["active_url"] == "http://validator-a"


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
