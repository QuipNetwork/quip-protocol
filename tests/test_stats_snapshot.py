"""Tests for shared.stats_snapshot.

`StatsSnapshotWriter` runs in the controller process and atomically
writes a stats snapshot to disk every interval. The telemetry sibling
process reads it on each /api/v1/stats request. Communication is
file-based so the two processes don't share live IPC.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from shared.stats_snapshot import StatsSnapshotWriter, read_snapshot


@pytest.mark.asyncio
async def test_writer_writes_initial_snapshot(tmp_path: Path):
    """One pass through the writer's loop produces the JSON file."""
    snapshot_path = tmp_path / "stats.json"
    snapshot_data = {"heads_observed": 7, "active_url": "http://a"}

    writer = StatsSnapshotWriter(
        path=snapshot_path,
        get_snapshot=lambda: snapshot_data,
        interval_s=0.01,
    )
    shutdown_event = asyncio.Event()

    async def stop_after_one_write():
        await asyncio.sleep(0.05)  # let writer run ≥ once
        shutdown_event.set()

    await asyncio.gather(writer.run(shutdown_event), stop_after_one_write())

    assert snapshot_path.exists()
    loaded = json.loads(snapshot_path.read_text())
    assert loaded == snapshot_data


@pytest.mark.asyncio
async def test_writer_updates_snapshot_each_interval(tmp_path: Path):
    """If get_snapshot() returns changing data, the file reflects the latest."""
    snapshot_path = tmp_path / "stats.json"
    state = {"value": 0}

    def get_snapshot():
        state["value"] += 1
        return dict(state)  # copy so subsequent writes are independent

    writer = StatsSnapshotWriter(
        path=snapshot_path,
        get_snapshot=get_snapshot,
        interval_s=0.01,
    )
    shutdown_event = asyncio.Event()

    async def stop_after_several_writes():
        await asyncio.sleep(0.1)  # at least ~10 writes at 0.01s interval
        shutdown_event.set()

    await asyncio.gather(writer.run(shutdown_event), stop_after_several_writes())

    loaded = json.loads(snapshot_path.read_text())
    assert loaded["value"] >= 3  # latest write should be at least the 3rd


def test_read_snapshot_returns_dict_when_file_exists(tmp_path: Path):
    """read_snapshot returns the parsed JSON when the file is present."""
    snapshot_path = tmp_path / "stats.json"
    snapshot_path.write_text('{"a": 1, "b": "hello"}')
    assert read_snapshot(snapshot_path) == {"a": 1, "b": "hello"}


def test_read_snapshot_returns_none_when_file_missing(tmp_path: Path):
    """A missing snapshot is a normal startup state, not an error."""
    assert read_snapshot(tmp_path / "missing.json") is None


def test_read_snapshot_returns_none_on_corrupt_file(tmp_path: Path):
    """A partially-written or corrupt file must not crash the reader.

    Atomic writes via os.replace should prevent torn reads in practice,
    but defense-in-depth: bad JSON → None, not an exception.
    """
    snapshot_path = tmp_path / "stats.json"
    snapshot_path.write_text("not valid json at all")
    assert read_snapshot(snapshot_path) is None


@pytest.mark.asyncio
async def test_writer_atomic_write_no_partial_files(tmp_path: Path):
    """The writer uses a tmp file + os.replace so the target is never partial."""
    snapshot_path = tmp_path / "stats.json"

    def get_snapshot():
        # large enough that a non-atomic write would visibly be partial
        return {"big_list": list(range(1000))}

    writer = StatsSnapshotWriter(
        path=snapshot_path,
        get_snapshot=get_snapshot,
        interval_s=0.005,
    )
    shutdown_event = asyncio.Event()

    async def reader_loop():
        """Read the snapshot many times concurrently with writer; never see partial JSON."""
        for _ in range(50):
            data = read_snapshot(snapshot_path)
            if data is not None:
                # If we got data, it must be the full payload.
                assert len(data["big_list"]) == 1000
            await asyncio.sleep(0.001)
        shutdown_event.set()

    await asyncio.gather(writer.run(shutdown_event), reader_loop())


@pytest.mark.asyncio
async def test_writer_handles_get_snapshot_exception(tmp_path: Path):
    """If get_snapshot() raises, the writer logs and continues — does not crash the loop."""
    snapshot_path = tmp_path / "stats.json"
    call_count = {"n": 0}

    def get_snapshot():
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated stats-collection bug")
        return {"call": call_count["n"]}

    writer = StatsSnapshotWriter(
        path=snapshot_path,
        get_snapshot=get_snapshot,
        interval_s=0.01,
    )
    shutdown_event = asyncio.Event()

    async def stop_after_several_writes():
        await asyncio.sleep(0.1)
        shutdown_event.set()

    await asyncio.gather(writer.run(shutdown_event), stop_after_several_writes())

    # Loop must have continued past the exception
    assert call_count["n"] > 2
