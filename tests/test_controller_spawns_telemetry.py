"""Verify the controller always spawns the telemetry sibling.

The sibling process is the sole telemetry surface — there is no
in-process server. Tests cover the default port and a custom override.
"""
from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

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
    return ctrl, spawned


def test_controller_spawns_telemetry_with_default_port(monkeypatch, tmp_path):
    """Default constructor port (8086) reaches the sibling kwargs."""
    ctrl, spawned = _make_ctrl(monkeypatch, tmp_path, telemetry_port=8086)
    ctrl._spawn_telemetry_sibling()
    assert len(spawned) == 1
    assert spawned[0]._kwargs["listen_port"] == 8086
    assert spawned[0]._kwargs["validator_urls"] == ["http://test:9944"]
    assert spawned[0]._kwargs["stats_snapshot_path"] == str(
        tmp_path / "telemetry-stats.json"
    )


def test_controller_spawns_telemetry_with_custom_port(monkeypatch, tmp_path):
    """A custom telemetry_port flows through to the sibling kwargs."""
    ctrl, spawned = _make_ctrl(monkeypatch, tmp_path, telemetry_port=9999)
    ctrl._spawn_telemetry_sibling()
    assert len(spawned) == 1
    assert spawned[0]._kwargs["listen_port"] == 9999
