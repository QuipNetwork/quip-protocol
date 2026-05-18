"""Tests for ``TelemetryManager.enable_async_writes``.

The coordinator used to call ``_atomic_write_json`` inline from the
asyncio thread on every heartbeat, which starved the event loop once
disk latency crept up. After ``enable_async_writes`` the write moves
through a bounded queue and a dedicated task that runs
``asyncio.to_thread``, so callers return immediately.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from shared.telemetry import TelemetryManager


@pytest.mark.asyncio
async def test_async_writes_flush_nodes_json():
    """Node updates after ``enable_async_writes`` eventually hit disk."""
    with tempfile.TemporaryDirectory() as td:
        tm = TelemetryManager(telemetry_dir=td)
        tm.enable_async_writes()

        tm.update_node("peer-A:1", "active", last_heartbeat=1.0)
        tm.update_node("peer-B:2", "active", last_heartbeat=2.0)

        target = Path(td) / "nodes.json"
        for _ in range(100):
            await asyncio.sleep(0.02)
            if target.exists():
                data = json.loads(target.read_text())
                if "peer-B:2" in data.get("nodes", {}):
                    break

        assert target.exists()
        data = json.loads(target.read_text())
        assert "peer-A:1" in data["nodes"]
        assert "peer-B:2" in data["nodes"]

        tm.stop()


@pytest.mark.asyncio
async def test_sync_path_still_works_without_enable():
    """Without ``enable_async_writes``, writes run inline (tests path)."""
    with tempfile.TemporaryDirectory() as td:
        tm = TelemetryManager(telemetry_dir=td)
        tm.update_node("peer-A:1", "active", last_heartbeat=1.0)

        target = Path(td) / "nodes.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert "peer-A:1" in data["nodes"]


@pytest.mark.asyncio
async def test_stop_cancels_writer_task():
    """``stop`` cancels the writer task cleanly."""
    with tempfile.TemporaryDirectory() as td:
        tm = TelemetryManager(telemetry_dir=td)
        tm.enable_async_writes()
        assert tm._writer_task is not None
        tm.stop()
        # Give the cancellation a tick to propagate.
        await asyncio.sleep(0.05)
        assert tm._writer_task is None


@pytest.mark.asyncio
async def test_full_queue_drops_without_blocking():
    """A saturated queue drops writes instead of stalling callers."""
    with tempfile.TemporaryDirectory() as td:
        tm = TelemetryManager(telemetry_dir=td)
        tm.enable_async_writes(maxsize=2)

        # Flood synchronously; put_nowait into a size-2 queue saturates
        # before the writer task gets scheduled.
        for i in range(200):
            tm.update_node(f"p{i}:1", "active", last_heartbeat=float(i))

        # None of those calls should have blocked; we arrive here
        # promptly regardless of drops.
        await asyncio.sleep(0.2)
        tm.stop()
