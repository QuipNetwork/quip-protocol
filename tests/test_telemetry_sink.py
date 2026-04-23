"""Tests for ``TelemetrySink`` and ``telemetry_aggregator``.

Exercises the AF_UNIX DGRAM observability pipeline that moved
``gossip_stats`` off the critical control-plane pipe.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from shared.telemetry_aggregator import spawn_telemetry_aggregator
from shared.telemetry_sink import (
    MAX_DGRAM_BYTES,
    TelemetrySink,
    configure_sink,
    get_sink,
)


def test_sink_no_listener_drops_silently():
    """Emitting before aggregator binds does not raise."""
    sink = TelemetrySink("/tmp/quip-nonexistent-telemetry.sock")
    ok = sink.emit({"peer": "a:1", "sent": 1})
    assert ok is False
    assert sink.stats()["dropped_no_listener"] >= 1


def test_sink_oversize_payload_is_dropped():
    """Payloads over MAX_DGRAM_BYTES are refused before any send."""
    sink = TelemetrySink("/tmp/quip-oversize-telemetry.sock")
    huge = "x" * (MAX_DGRAM_BYTES + 1)
    ok = sink.emit({"peer": "a:1", "blob": huge})
    assert ok is False
    assert sink.stats()["dropped_oversize"] == 1


def test_sink_serialize_failure_is_dropped():
    """Unserializable payloads (circular refs) drop rather than raise."""
    sink = TelemetrySink("/tmp/quip-badser-telemetry.sock")
    circular: dict = {}
    circular["self"] = circular
    ok = sink.emit(circular)
    assert ok is False
    # Counter increments on serialize failure; no listener wouldn't be
    # reached because the payload never gets to the socket step.
    assert sink.stats()["dropped_error"] == 1


def test_aggregator_writes_per_peer_json_end_to_end():
    """Loopback: sink.emit → aggregator → file on disk."""
    with tempfile.TemporaryDirectory() as td:
        sock_path = os.path.join(td, "tel.sock")
        handle = spawn_telemetry_aggregator(
            socket_path=sock_path, telemetry_dir=td,
        )
        try:
            # Wait for aggregator to bind.
            for _ in range(50):
                if os.path.exists(sock_path):
                    break
                time.sleep(0.1)
            assert os.path.exists(sock_path)

            configure_sink(sock_path)
            sink = get_sink()
            assert sink is not None
            for i in range(5):
                sink.emit({
                    "peer": "peer-A:20049",
                    "sent": 10 + i,
                    "responded": 7 + i,
                    "failed": 3,
                    "ts": time.time(),
                })

            # Aggregator flushes on a 5 s cadence; wait a bit longer.
            out = Path(td) / "gossip" / "peer-A_20049.json"
            for _ in range(80):
                if out.exists():
                    break
                time.sleep(0.1)
            assert out.exists()

            data = json.loads(out.read_text())
            assert data["peer"] == "peer-A:20049"
            # Last-wins after aggregation.
            assert data["sent"] == 14
            assert data["responded"] == 11
            assert data["failed"] == 3
            # success_rate mirrors responded/sent.
            assert data["success_rate"] == pytest.approx(11 / 14, abs=1e-3)
        finally:
            handle.shutdown(timeout=5.0)
            assert not handle.process.is_alive()
