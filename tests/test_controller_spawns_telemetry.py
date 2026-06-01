"""Verify the controller always spawns the telemetry sibling.

The sibling process is the sole telemetry surface — there is no
in-process server. Tests cover the default port and a custom override.
"""
from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import pytest

from substrate.miner_controller import SubstrateMinerController


def _make_ctrl(monkeypatch, tmp_path, telemetry_port):
    spawned = []
    original_process = mp.Process

    class _TrackingProcess(original_process):
        def start(self):
            spawned.append(self)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    monkeypatch.setattr(mp, "Process", _TrackingProcess)

    ctrl = SubstrateMinerController.__new__(SubstrateMinerController)
    ctrl._runtime_dir = tmp_path
    ctrl._telemetry_port = telemetry_port
    ctrl._telemetry_proc = None
    ctrl._telemetry_shutdown_event = None
    ctrl.pool = SimpleNamespace(urls=("http://test:9944",))
    # New (Wave 3) attributes — kept defaulted so the legacy single-
    # process spawn path matches the v0.2-early behavior.
    ctrl.snapshot_kind = ""
    ctrl._spawn_telemetry_sibling_enabled = True  # noqa: SLF001
    return ctrl, spawned


@pytest.mark.parametrize("telemetry_port", [8086, 9999])
def test_controller_spawns_telemetry_port_reaches_sibling(
    monkeypatch, tmp_path, telemetry_port
):
    """The constructor port (default 8086 or a custom override) flows through
    to the sibling kwargs alongside the port-independent validator URLs and
    snapshot path."""
    ctrl, spawned = _make_ctrl(monkeypatch, tmp_path, telemetry_port=telemetry_port)
    ctrl._spawn_telemetry_sibling()
    assert len(spawned) == 1
    assert spawned[0]._kwargs["listen_port"] == telemetry_port
    assert spawned[0]._kwargs["validator_urls"] == ["http://test:9944"]
    # Default snapshot filename: empty snapshot_kind resolves to
    # `telemetry-stats-default.json` (per `snapshot_filename_for("")`).
    assert spawned[0]._kwargs["stats_snapshot_path"] == str(
        tmp_path / "telemetry-stats-default.json"
    )
