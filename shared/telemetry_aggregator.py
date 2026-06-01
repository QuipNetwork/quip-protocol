"""
Telemetry aggregator process.

Drains AF_UNIX DGRAM observability events from workers into per-peer
JSON files under ``<telemetry_dir>/gossip/``. Spawned by the coordinator
alongside the listener and connection-worker processes.

Runs without an asyncio event loop — a single blocking ``recvfrom`` with
a short timeout is enough since this process has no other responsibility.
A stall here only drops datagrams (kernel-side queue overflow) and can
never cascade back into coordinator/worker deadlock, which is the whole
point of moving observability off the control plane.

SPDX-License-Identifier: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import multiprocessing.synchronize
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from shared.logging_config import QuipFormatter
from shared.telemetry import atomic_write_json


_RECV_BUF = 16384
"""Single-read buffer. Must be >= TelemetrySink.MAX_DGRAM_BYTES."""

_WRITE_INTERVAL = 5.0
"""Per-peer snapshot rewrite cadence (seconds); coalesces bursty updates."""


@dataclass
class _PeerStats:
    """Latest aggregated stats for one peer."""
    sent: int = 0
    responded: int = 0
    failed: int = 0
    last_event_ts: float = 0.0
    pending_write: bool = False


def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(QuipFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("telemetry-agg")


def _sanitize_peer(addr: str) -> str:
    """Convert ``host:port`` into a filesystem-safe basename."""
    return addr.replace(":", "_").replace("/", "_")


def _bind_socket(path: str, log: logging.Logger) -> socket.socket:
    """Bind AF_UNIX DGRAM at *path*, removing any stale socket file."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    # 1 MB recv buffer oversizes observed burst traffic and gives the
    # kernel headroom during a brief aggregator pause. Failure to set
    # it is non-fatal — the system default still works.
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    except OSError:
        pass
    sock.bind(path)
    try:
        os.chmod(path, 0o660)
    except OSError as exc:
        log.warning("chmod %s failed: %s", path, exc)
    log.info("Telemetry aggregator bound to %s", path)
    return sock


def telemetry_aggregator_main(
    socket_path: str,
    telemetry_dir: str,
    stop_event: mp.synchronize.Event,
) -> None:
    """Aggregator entry point."""

    def _signal_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log = _setup_logging()
    out_dir = Path(telemetry_dir) / "gossip"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sock = _bind_socket(socket_path, log)
    except OSError as exc:
        log.error("Failed to bind telemetry socket at %s: %s", socket_path, exc)
        return

    stats: Dict[str, _PeerStats] = {}
    last_flush = time.monotonic()
    sock.settimeout(1.0)

    try:
        while not stop_event.is_set():
            data: Optional[bytes]
            try:
                data, _addr = sock.recvfrom(_RECV_BUF)
            except socket.timeout:
                data = None

            if data:
                _ingest(data, stats, log)

            now = time.monotonic()
            if now - last_flush >= _WRITE_INTERVAL:
                _flush(stats, out_dir, log)
                last_flush = now
    finally:
        _flush(stats, out_dir, log)
        try:
            sock.close()
        except OSError:
            pass
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        log.info("Telemetry aggregator stopped")


def _ingest(
    data: bytes, stats: Dict[str, _PeerStats], log: logging.Logger,
) -> None:
    try:
        event = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.debug("telemetry_aggregator: invalid datagram: %s", exc)
        return
    peer = event.get("peer")
    if not isinstance(peer, str) or not peer:
        return
    rec = stats.setdefault(peer, _PeerStats())
    rec.sent = int(event.get("sent", rec.sent))
    rec.responded = int(event.get("responded", rec.responded))
    rec.failed = int(event.get("failed", rec.failed))
    rec.last_event_ts = float(event.get("ts", time.time()))
    rec.pending_write = True


def _flush(
    stats: Dict[str, _PeerStats], out_dir: Path, log: logging.Logger,
) -> None:
    for peer, rec in stats.items():
        if not rec.pending_write:
            continue
        rate = (rec.responded / rec.sent) if rec.sent else 0.0
        payload = {
            "peer": peer,
            "sent": rec.sent,
            "responded": rec.responded,
            "failed": rec.failed,
            "success_rate": round(rate, 4),
            "last_update": rec.last_event_ts,
        }
        path = out_dir / f"{_sanitize_peer(peer)}.json"
        try:
            atomic_write_json(path, payload)
            rec.pending_write = False
        except OSError as exc:
            log.warning("telemetry_aggregator: write %s failed: %s", path, exc)


@dataclass
class TelemetryAggregatorHandle:
    """Coordinator-side handle to the aggregator process."""
    process: mp.Process
    stop_event: mp.synchronize.Event
    socket_path: str

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)


def spawn_telemetry_aggregator(
    socket_path: str, telemetry_dir: str,
) -> TelemetryAggregatorHandle:
    """Spawn the aggregator and return a handle."""
    stop_event = mp.Event()
    process = mp.Process(
        target=telemetry_aggregator_main,
        args=(socket_path, telemetry_dir, stop_event),
        daemon=True,
    )
    process.start()
    return TelemetryAggregatorHandle(
        process=process, stop_event=stop_event, socket_path=socket_path,
    )
