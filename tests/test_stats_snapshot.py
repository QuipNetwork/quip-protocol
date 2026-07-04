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
        reads_succeeded = 0
        for _ in range(50):
            data = read_snapshot(snapshot_path)
            if data is not None:
                # If we got data, it must be the full payload.
                assert len(data["big_list"]) == 1000
                reads_succeeded += 1
            await asyncio.sleep(0.001)
        assert reads_succeeded > 0, "writer never produced a readable snapshot"
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


# ----------------------------------------------------------------------
# Multi-snapshot aggregator (Docker entrypoint use case)
# ----------------------------------------------------------------------

from shared.stats_snapshot import (  # noqa: E402
    merge_snapshots,
    read_all_snapshots,
    snapshot_filename_for,
)


def test_snapshot_filename_for_canonical_kinds():
    assert snapshot_filename_for("cpu") == "telemetry-stats-cpu.json"
    assert snapshot_filename_for("gpu") == "telemetry-stats-gpu.json"
    assert snapshot_filename_for("qpu") == "telemetry-stats-qpu.json"


def test_snapshot_filename_for_empty_falls_back_to_default():
    """Single-process legacy callers pass "" — must still produce a
    well-formed filename so the legacy snapshot read path keeps working."""
    assert snapshot_filename_for("") == "telemetry-stats-default.json"


def test_snapshot_filename_for_sanitizes_path_separators():
    """Untrusted kind strings can't escape the snapshot dir — non-alnum
    characters are stripped before being concatenated into the filename."""
    assert snapshot_filename_for("../etc/passwd") == "telemetry-stats-etcpasswd.json"


def test_read_all_snapshots_globs_per_kind_files(tmp_path):
    """Aggregator reads every telemetry-stats-*.json in the dir, in
    deterministic order (alphabetical by kind)."""
    (tmp_path / "telemetry-stats-cpu.json").write_text(
        json.dumps({"mode": "cpu", "controller": {"heads_observed": 1}})
    )
    (tmp_path / "telemetry-stats-qpu.json").write_text(
        json.dumps({"mode": "qpu", "controller": {"heads_observed": 2}})
    )
    # Non-snapshot files in the same dir are ignored — operators may
    # mount /data/runtime for misc state.
    (tmp_path / "unrelated.json").write_text(json.dumps({"x": 1}))

    snaps = read_all_snapshots(tmp_path)
    assert len(snaps) == 2
    assert [s["mode"] for s in snaps] == ["cpu", "qpu"]


def test_read_all_snapshots_skips_corrupt_files(tmp_path):
    """A mid-replace file shouldn't sink the aggregator's whole read —
    skip the bad one and serve what's available."""
    (tmp_path / "telemetry-stats-cpu.json").write_text("{not json")
    (tmp_path / "telemetry-stats-qpu.json").write_text(
        json.dumps({"mode": "qpu", "controller": {}})
    )
    snaps = read_all_snapshots(tmp_path)
    assert len(snaps) == 1
    assert snaps[0]["mode"] == "qpu"


def test_read_all_snapshots_missing_dir_returns_empty(tmp_path):
    assert read_all_snapshots(tmp_path / "does-not-exist") == []


def test_merge_snapshots_empty_returns_none():
    assert merge_snapshots([]) is None
    assert merge_snapshots([{}, None]) is None  # type: ignore[list-item]


def test_merge_snapshots_sums_controller_counters():
    """heads_observed / proofs_submitted / etc. must SUM across child
    snapshots — the operator wants total work performed by the
    container, not per-mode breakdown for the headline counter."""
    cpu = {"mode": "cpu", "controller": {
        "heads_observed": 100, "proofs_submitted": 3,
        "contexts_dispatched": 100,
    }}
    qpu = {"mode": "qpu", "controller": {
        "heads_observed": 100, "proofs_submitted": 2,
        "contexts_dispatched": 100,
    }}
    merged = merge_snapshots([cpu, qpu])
    assert merged is not None
    assert merged["controller"]["heads_observed"] == 200
    assert merged["controller"]["proofs_submitted"] == 5
    assert merged["controller"]["contexts_dispatched"] == 200


def test_merge_snapshots_first_active_url_wins():
    """active_url is per-pool — each child has its own. First-found
    keeps the merged view stable rather than flipping per request."""
    a = {"mode": "cpu", "controller": {"active_url": "ws://a"}}
    b = {"mode": "qpu", "controller": {"active_url": "ws://b"}}
    merged = merge_snapshots([a, b])
    assert merged is not None
    assert merged["controller"]["active_url"] == "ws://a"


def test_merge_snapshots_unions_miners_dedup_by_id():
    """Each child reports only its own handles; merged view is the full
    inventory. Duplicate ids across children (shouldn't happen in
    practice — each handle is owned by exactly one MinerCore) get
    deduplicated so dashboards don't double-count."""
    cpu = {"mode": "cpu", "miners": [
        {"id": "rig-CPU-1", "type": "CPU"},
        {"id": "rig-CPU-2", "type": "CPU"},
    ]}
    qpu = {"mode": "qpu", "miners": [
        {"id": "rig-QPU-DWAVE-1", "type": "QPU"},
        # Hypothetical duplicate — drop the second occurrence.
        {"id": "rig-CPU-1", "type": "CPU"},
    ]}
    merged = merge_snapshots([cpu, qpu])
    assert merged is not None
    assert [m["id"] for m in merged["miners"]] == [
        "rig-CPU-1", "rig-CPU-2", "rig-QPU-DWAVE-1",
    ]


def test_merge_snapshots_identity_first_nonempty():
    """node_id / ss58 / descriptor / survey describe the container as a
    whole. First child to publish a non-empty value wins; later
    children's identical (or stale) copies don't override."""
    a = {"mode": "cpu",
         "node_id": None, "ss58_address": None,
         "descriptor": {}, "miner_survey": {}}
    b = {"mode": "qpu",
         "node_id": "rig-01", "ss58_address": "5GPP",
         "descriptor": {"cpus": 8},
         "miner_survey": {"v": "quip.miner_survey.v1"}}
    merged = merge_snapshots([a, b])
    assert merged is not None
    assert merged["node_id"] == "rig-01"
    assert merged["ss58_address"] == "5GPP"
    assert merged["descriptor"] == {"cpus": 8}
    assert merged["miner_survey"]["v"] == "quip.miner_survey.v1"


def test_merge_snapshots_per_mode_breakdown_under_modes_key():
    """Aggregated counters are useful for the headline number, but
    operators investigating "is the qpu doing anything?" need per-mode
    figures. Stash each child's raw controller dict under modes[<mode>]."""
    cpu = {"mode": "cpu", "controller": {"proofs_submitted": 3},
           "miners": [{"id": "rig-CPU-1", "type": "CPU"}]}
    qpu = {"mode": "qpu", "controller": {"proofs_submitted": 2},
           "miners": [{"id": "rig-QPU-1", "type": "QPU"}]}
    merged = merge_snapshots([cpu, qpu])
    assert merged is not None
    assert set(merged["modes"].keys()) == {"cpu", "qpu"}
    assert merged["modes"]["cpu"]["controller"]["proofs_submitted"] == 3
    assert merged["modes"]["qpu"]["controller"]["proofs_submitted"] == 2


def test_merge_snapshots_unknown_mode_gets_synthetic_slot():
    """Snapshots from miners that don't set `mode` (older images, dev
    paths) still appear in the breakdown under unknown.<i>."""
    a = {"controller": {"proofs_submitted": 1}, "miners": []}
    b = {"controller": {"proofs_submitted": 2}, "miners": []}
    merged = merge_snapshots([a, b])
    assert merged is not None
    assert set(merged["modes"].keys()) == {"unknown.0", "unknown.1"}


def test_merge_snapshots_single_snapshot_passes_through():
    """N=1 case: aggregator running with one active mode degrades to
    the same shape as the single-snapshot legacy reader. Operators
    upgrading from single-mode to multi-mode see the same JSON
    structure for their existing scrapers."""
    cpu = {
        "mode": "cpu",
        "controller": {"heads_observed": 50, "proofs_submitted": 1, "active_url": "ws://a"},
        "node_id": "rig", "ss58_address": "5GPP",
        "miners": [{"id": "rig-CPU-1", "type": "CPU"}],
        "descriptor": {"cpus": 4},
        "miner_survey": {},
        "attempts_dir": "/data/attempts",
    }
    merged = merge_snapshots([cpu])
    assert merged is not None
    assert merged["controller"]["heads_observed"] == 50
    assert merged["miners"] == [{"id": "rig-CPU-1", "type": "CPU"}]
    assert merged["node_id"] == "rig"
    assert set(merged["modes"].keys()) == {"cpu"}


def test_merge_snapshots_takes_first_nonnull_sync_state():
    """sync_state merges first-wins — all children share one validator."""
    merged = merge_snapshots([
        {"mode": "cpu", "controller": {}, "sync_state": None},
        {"mode": "qpu", "controller": {}, "sync_state": {"is_syncing": True}},
    ])
    assert merged["sync_state"] == {"is_syncing": True}
