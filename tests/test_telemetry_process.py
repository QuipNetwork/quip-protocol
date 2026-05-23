"""Tests for shared.telemetry_process.telemetry_main.

The telemetry sibling process owns its own aiohttp app, a SubstrateClient
with URL failover, and serves /api/v1/stats from the file the controller
writes via StatsSnapshotWriter.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import socket
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_telemetry_process_starts_and_serves_stats(tmp_path: Path):
    """Spawn telemetry process; verify /api/v1/stats returns the file contents."""
    from shared.telemetry_process import telemetry_main

    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text(json.dumps(
        {"controller": {"heads_observed": 42, "active_url": "http://a"}}
    ))

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],  # not actually used in this test
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        # Wait for server to come up
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                import urllib.request
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/stats", timeout=0.5,
                ).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("telemetry process did not start in 5s")

        data = json.loads(resp)
        assert data["success"] is True
        assert data["data"]["controller"]["heads_observed"] == 42
        assert data["data"]["controller"]["active_url"] == "http://a"
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_process_returns_503_when_snapshot_missing(tmp_path: Path):
    """If the snapshot file doesn't exist yet, /api/v1/stats returns 503."""
    from shared.telemetry_process import telemetry_main

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(tmp_path / "missing.json"),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        deadline = time.time() + 5.0
        last_status = None
        while time.time() < deadline:
            try:
                import urllib.request
                import urllib.error
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/v1/stats", timeout=0.5,
                    )
                    last_status = 200
                except urllib.error.HTTPError as e:
                    last_status = e.code
                break
            except Exception:
                time.sleep(0.1)
        assert last_status == 503
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_process_responds_to_shutdown_event(tmp_path: Path):
    """Setting shutdown_event causes the process to exit cleanly."""
    from shared.telemetry_process import telemetry_main

    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text("{}")
    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        time.sleep(0.5)  # let it come up
        shutdown_event.set()
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
