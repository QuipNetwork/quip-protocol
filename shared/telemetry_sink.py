"""
Non-blocking AF_UNIX DGRAM telemetry sink for per-worker observability.

Per-peer child processes emit pure observability events (gossip_stats
today; rolling RTT and probe stats tomorrow) through a process-global
``TelemetrySink``. The sink opens a non-blocking ``AF_UNIX`` ``SOCK_DGRAM``
socket connected to the ``telemetry_aggregator`` at the path configured
by the coordinator (typically ``<telemetry_dir>/telemetry.sock``).

Datagrams are lossy by kernel overflow policy — a writer never blocks.
Observability is kept entirely off the control-plane pipe, which cannot
tolerate pauses.

SPDX-License-Identifier: AGPL-3.0-or-later
"""

from __future__ import annotations

import errno
import json
import logging
import socket
from typing import Optional


MAX_DGRAM_BYTES = 8192
"""Per-datagram size cap; oversized payloads are dropped with a debug log
so datagrams remain atomic (AF_UNIX DGRAM does not re-assemble)."""


class TelemetrySink:
    """Fire-and-forget non-blocking AF_UNIX DGRAM emitter.

    ``emit(event)`` serializes a dict as JSON and hands it to the kernel.
    Emission never raises: kernel buffer overflow, missing aggregator,
    and oversized payloads are all counted as debug-level drops.

    The socket is opened lazily on first emit so a worker that never
    produces telemetry pays nothing.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._socket_path = socket_path
        self._logger = logger or logging.getLogger(__name__)
        self._sock: Optional[socket.socket] = None
        self._dropped_buffer_full = 0
        self._dropped_no_listener = 0
        self._dropped_oversize = 0
        self._dropped_error = 0

    def _ensure_socket(self) -> Optional[socket.socket]:
        if self._sock is not None:
            return self._sock
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.connect(self._socket_path)
        except OSError:
            # Aggregator not listening yet; retry on next emit.
            return None
        self._sock = sock
        return sock

    def emit(self, event: dict) -> bool:
        """Serialize *event* and send one datagram. Never raises.

        Returns True when the datagram was handed to the kernel, False
        when dropped for any reason (serialize failure, oversize, no
        listener, kernel buffer full, socket error).
        """
        try:
            payload = json.dumps(event, default=str).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self._dropped_error += 1
            self._logger.debug("telemetry_sink: serialize failed: %s", exc)
            return False
        if len(payload) > MAX_DGRAM_BYTES:
            self._dropped_oversize += 1
            self._logger.debug(
                "telemetry_sink: oversize payload %d > %d bytes",
                len(payload), MAX_DGRAM_BYTES,
            )
            return False

        sock = self._ensure_socket()
        if sock is None:
            self._dropped_no_listener += 1
            return False

        try:
            sock.send(payload)
            return True
        except BlockingIOError:
            self._dropped_buffer_full += 1
            return False
        except (ConnectionRefusedError, FileNotFoundError):
            self._close()
            self._dropped_no_listener += 1
            return False
        except OSError as exc:
            if exc.errno in (errno.ECONNREFUSED, errno.ENOENT):
                self._close()
                self._dropped_no_listener += 1
                return False
            self._logger.debug("telemetry_sink: socket error: %s", exc)
            self._close()
            self._dropped_error += 1
            return False

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        self._close()

    def stats(self) -> dict:
        return {
            "dropped_buffer_full": self._dropped_buffer_full,
            "dropped_no_listener": self._dropped_no_listener,
            "dropped_oversize": self._dropped_oversize,
            "dropped_error": self._dropped_error,
        }


_SINK: Optional[TelemetrySink] = None
_SINK_PATH: Optional[str] = None


def configure_sink(socket_path: str) -> None:
    """Set the socket path used by future ``get_sink()`` calls.

    Each worker process calls this once early in startup; the sink
    itself is constructed lazily on first ``emit``.
    """
    global _SINK_PATH, _SINK
    _SINK_PATH = socket_path
    _SINK = None


def get_sink() -> Optional[TelemetrySink]:
    """Return the process-global sink, lazily constructed.

    Returns ``None`` if ``configure_sink()`` has not been called in this
    process, which is the expected state in tests that do not provide an
    aggregator.
    """
    global _SINK
    if _SINK is not None:
        return _SINK
    if _SINK_PATH is None:
        return None
    _SINK = TelemetrySink(_SINK_PATH)
    return _SINK
