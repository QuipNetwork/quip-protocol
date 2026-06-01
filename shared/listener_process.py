"""Dedicated listener process for inbound QUIC traffic.

A single child process owns the UDP listen socket and runs the
``aioquic`` server in its own event loop. The coordinator (parent
``NetworkNode``) no longer runs QUIC transport — TLS handshakes,
packet decryption, and frame parsing all happen in the listener,
isolated from coordinator state and locks.

Message handling is split:

* **HEARTBEAT** is handled locally from a cached peer/ban snapshot.
  The listener sends the response immediately on the QUIC connection
  and fires a ``peer_heartbeat`` event to the coordinator so it can
  update SWIM / scorer / telemetry asynchronously. No IPC round-trip
  on the hot path, and no ``net_lock`` contention.

* **JOIN_REQUEST** is answered locally with ``at_capacity`` when the
  claimed peer is in the banned set (from the coordinator's
  ``record_capacity_rejection`` ledger). Any other JOIN is forwarded
  to the coordinator. This shields the coord event loop from JOIN
  retry storms by peers that don't honor the backoff hint.

* **All other message types** are forwarded to the coordinator via
  IPC (``inbound_message`` event with a correlation ``msg_id``). The
  listener awaits ``inbound_response`` and sends it on QUIC.

IPC protocol
------------
Parent -> Child::

    {"cmd": "peer_snapshot", "peers": {host: {"version": str}},
                              "banned": [host, ...]}
    {"cmd": "inbound_response", "msg_id": int,
                                 "payload_b64": <base64> | None}
    {"cmd": "shutdown"}

Child -> Parent::

    {"event": "listener_ready", "port": int}
    {"event": "peer_heartbeat", "peer": str, "version": str,
                                 "timestamp": float}
    {"event": "inbound_message", "msg_id": int, "peer": str,
                                  "raw_b64": <base64>}
    {"event": "listener_error", "message": str}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import multiprocessing as mp
import multiprocessing.synchronize
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from shared.logging_config import QuipFormatter
from shared.pipe_sender import AsyncPipeSender


_DEFAULT_FORWARD_TIMEOUT = 4.5  # below peer-side HB timeout (5s)

_LISTENER_SENDER: Optional[AsyncPipeSender] = None
"""Process-global outbound sender (listener → coord). Initialized in
``_listener_loop`` and closed in its ``finally``. Module-global is safe
because the listener is its own OS process."""


async def _send_event(
    msg: dict, *, critical: bool = False, timeout: float = 5.0,
) -> bool:
    """Async send helper for listener → coord events.

    Mirrors ``connection_process._send_event``: ``critical=True`` awaits
    enqueue (for ``inbound_message`` forwards that a correlated future
    is waiting on), otherwise drop-newest (``peer_heartbeat`` and
    ``listener_ready``).
    """
    sender = _LISTENER_SENDER
    if sender is None:
        return False
    if critical:
        return await sender.send_blocking(msg, timeout=timeout)
    return sender.send_nowait(msg)


def _send_event_nowait(msg: dict) -> bool:
    """Sync enqueue for async-adjacent sync callers (e.g. the aioquic
    protocol's ``_handle_heartbeat_local`` is sync but called from an
    async task on the event-loop thread, so ``put_nowait`` is safe).
    """
    sender = _LISTENER_SENDER
    if sender is None:
        return False
    return sender.send_nowait(msg)


def _setup_child_logging() -> logging.Logger:
    """Configure console logging with QuipFormatter for the child."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(QuipFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Silence aioquic internals — see connection_process for rationale.
    logging.getLogger("quic").setLevel(logging.WARNING)
    logging.getLogger("aioquic").setLevel(logging.WARNING)
    return logging.getLogger("listener")


@dataclass
class ListenerProcessHandle:
    """Parent-side handle to the listener child process.

    Outbound commands go through ``sender`` (an ``AsyncPipeSender``),
    so ``send_cmd`` never blocks the coordinator's event loop on a
    full pipe. Inbound events are drained from the coordinator's
    event-driven reader.
    """
    process: mp.Process
    pipe: mp.connection.Connection
    stop_event: mp.synchronize.Event
    sender: AsyncPipeSender = field(init=False)
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # Sender lazy-starts its drainer on first send.
        self.sender = AsyncPipeSender(
            self.pipe, maxsize=256, name="listener",
        )

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send_cmd(self, cmd: dict) -> bool:
        """Enqueue a command for async send. Drops on full queue."""
        return self.sender.send_nowait(cmd)

    async def send_cmd_blocking(
        self, cmd: dict, timeout: float = 5.0,
    ) -> bool:
        """Await enqueue with timeout. For lifecycle / critical commands."""
        return await self.sender.send_blocking(cmd, timeout=timeout)

    def poll(self, timeout: float = 0) -> bool:
        try:
            return self.pipe.poll(timeout)
        except (BrokenPipeError, OSError, EOFError):
            return False

    def recv(self) -> Optional[dict]:
        """Non-blocking receive. Returns None if nothing available, or
        a synthetic ``listener_died`` event if the pipe is broken."""
        try:
            if self.pipe.poll(0):
                return self.pipe.recv()
        except (BrokenPipeError, OSError, EOFError):
            return {"event": "listener_died", "reason": "pipe broken"}
        return None

    def shutdown(self) -> None:
        """Request graceful shutdown.

        ``stop_event`` is the primary path; the shutdown command is
        best-effort (may be dropped on full queue, in which case the
        child exits via the ``stop_event`` poll loop).
        """
        self.stop_event.set()
        self.send_cmd({"cmd": "shutdown"})

    def force_stop(self, timeout: float = 3.0) -> None:
        """Terminate the child, escalating to kill if needed."""
        self.stop_event.set()
        self.sender._closing = True
        if not self.process.is_alive():
            return
        self.process.terminate()
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)

    async def aclose(self, timeout: float = 2.0) -> None:
        """Async-close the outbound sender (drainer task)."""
        await self.sender.close(timeout=timeout)


@dataclass
class _PeerSnapshotCache:
    """Snapshot of coordinator state pushed by the parent.

    Read by local fast-path handlers (HEARTBEAT, STATUS, STATS,
    PEERS, TELEMETRY_*) so the listener can answer without an IPC
    round-trip. ``peer_snapshot`` updates versions/banned;
    ``read_snapshot`` updates the pre-serialized response payloads.

    Each ``*_response`` field holds the *payload* bytes that the
    listener wraps in a ``QuicMessage.create_response`` when
    serving. ``None`` or an empty value means "fall back to
    forwarding the message to the coordinator."
    """
    versions: Dict[str, str] = field(default_factory=dict)
    banned: Set[str] = field(default_factory=set)
    status_response: Optional[bytes] = None
    stats_response: Optional[bytes] = None
    peers_response: Optional[bytes] = None
    telemetry_status_response: Optional[bytes] = None
    telemetry_nodes_response: Optional[bytes] = None
    telemetry_epochs_response: Optional[bytes] = None
    telemetry_latest_response: Optional[bytes] = None
    telemetry_block_responses: Dict[Tuple[str, int], bytes] = field(
        default_factory=dict,
    )
    read_snapshot_ts: float = 0.0
    staleness_limit: float = 30.0
    # Cache observability counters. ``hits`` increments on a successful
    # local serve; ``miss_stale`` when the snapshot has aged out;
    # ``miss_field`` when the snapshot is fresh but the requested field
    # is None (serializer failed coordinator-side or value not yet
    # populated). A predominantly ``miss_field`` pattern points at a
    # broken serializer, ``miss_stale`` at a snapshot-loop issue.
    hits: int = 0
    miss_stale: int = 0
    miss_field: int = 0

    def read_snapshot_fresh(self) -> bool:
        """True when the read snapshot is within its staleness limit."""
        if self.read_snapshot_ts <= 0:
            return False
        return (time.time() - self.read_snapshot_ts) < self.staleness_limit


def listener_process_main(
    pipe: mp.connection.Connection,
    stop_event: mp.synchronize.Event,
    config_bundle: dict,
) -> None:
    """Child process entry point.

    ``config_bundle`` must be picklable and contain:
        bind_address: str
        port: int
        tls_cert_file: str
        tls_key_file: str
        alpn_protocol: str
        max_datagram_frame_size: int
        idle_timeout: float
        max_datagram_message_size: int
    """
    def _signal_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log = _setup_child_logging()
    log.info("Listener process starting")

    from shared.event_loop import create_event_loop
    loop = create_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _listener_loop(pipe, stop_event, config_bundle, log)
        )
    except Exception as exc:
        log.exception(f"Listener process crashed: {exc}")
        _send_safe(pipe, {
            "event": "listener_error",
            "message": str(exc),
        })
    finally:
        loop.close()
        log.info("Listener process stopped")


async def _listener_loop(
    pipe: mp.connection.Connection,
    stop_event: mp.synchronize.Event,
    config_bundle: dict,
    log: logging.Logger,
) -> None:
    """Bind aioquic server, pump IPC, and run until shutdown."""
    from aioquic.asyncio import serve
    from aioquic.quic.configuration import QuicConfiguration

    global _LISTENER_SENDER
    _LISTENER_SENDER = AsyncPipeSender(
        pipe, maxsize=256, name="listener", logger=log,
    )
    _LISTENER_SENDER.start()

    cache = _PeerSnapshotCache()
    pending_forwards: Dict[int, asyncio.Future] = {}
    next_msg_id = 1

    # Build aioquic configuration
    configuration = QuicConfiguration(
        is_client=False,
        max_datagram_frame_size=config_bundle["max_datagram_frame_size"],
        alpn_protocols=[config_bundle["alpn_protocol"]],
        idle_timeout=float(config_bundle.get("idle_timeout", 300.0)),
    )
    configuration.load_cert_chain(
        config_bundle["tls_cert_file"],
        config_bundle["tls_key_file"],
    )

    max_datagram_message_size = int(
        config_bundle["max_datagram_message_size"]
    )

    def _alloc_msg_id() -> int:
        nonlocal next_msg_id
        mid = next_msg_id
        next_msg_id += 1
        return mid

    loop = asyncio.get_event_loop()

    def _create_protocol(quic, **kwargs):
        return _build_listener_protocol(
            quic=quic,
            pipe=pipe,
            cache=cache,
            pending_forwards=pending_forwards,
            alloc_msg_id=_alloc_msg_id,
            log=log,
            max_datagram_message_size=max_datagram_message_size,
            forward_timeout=_DEFAULT_FORWARD_TIMEOUT,
            stream_handler=kwargs.get("stream_handler"),
        )

    server = await serve(
        host=config_bundle["bind_address"],
        port=int(config_bundle["port"]),
        configuration=configuration,
        create_protocol=_create_protocol,
    )
    await _send_event({
        "event": "listener_ready",
        "port": int(config_bundle["port"]),
    }, critical=True)
    log.info(
        f"Listener bound to {config_bundle['bind_address']}:"
        f"{config_bundle['port']}"
    )

    cache_log_task = asyncio.create_task(
        _periodic_cache_log(cache, log, stop_event),
        name="listener-cache-log",
    )

    try:
        while not stop_event.is_set():
            try:
                has_data = await loop.run_in_executor(
                    None, pipe.poll, 0.5,
                )
            except (BrokenPipeError, OSError, EOFError):
                log.warning("Parent pipe broken, shutting down")
                break

            if not has_data:
                continue

            try:
                cmd = pipe.recv()
            except (BrokenPipeError, OSError, EOFError):
                log.warning("Parent pipe broken during recv")
                break

            if not isinstance(cmd, dict):
                continue

            op = cmd.get("cmd")
            if op == "shutdown":
                break
            elif op == "peer_snapshot":
                cache.versions = dict(cmd.get("peers") or {})
                cache.banned = set(cmd.get("banned") or [])
            elif op == "read_snapshot":
                _apply_read_snapshot(cache, cmd)
            elif op == "inbound_response":
                mid = cmd.get("msg_id")
                fut = pending_forwards.pop(mid, None) if mid is not None else None
                if fut is not None and not fut.done():
                    payload_b64 = cmd.get("payload_b64")
                    payload = (
                        base64.b64decode(payload_b64)
                        if payload_b64 else None
                    )
                    fut.set_result(payload)
            else:
                log.warning(f"Unknown listener cmd: {op}")
    finally:
        server.close()
        cache_log_task.cancel()
        try:
            await cache_log_task
        except (asyncio.CancelledError, Exception):
            pass
        # Drop outstanding forwards so their protocol tasks don't hang.
        for fut in pending_forwards.values():
            if not fut.done():
                fut.set_result(None)
        # Flush queued events (e.g. last peer_heartbeat) before exit.
        sender = _LISTENER_SENDER
        if sender is not None:
            try:
                await sender.close(timeout=2.0)
            except Exception:
                pass


def _build_listener_protocol(
    quic,
    pipe: mp.connection.Connection,
    cache: _PeerSnapshotCache,
    pending_forwards: Dict[int, asyncio.Future],
    alloc_msg_id,
    log: logging.Logger,
    max_datagram_message_size: int,
    forward_timeout: float,
    stream_handler=None,
):
    """Construct a ``QuicConnectionProtocol`` subclass instance.

    Defined as a factory (not a module-level class) so it can close
    over the per-listener state (cache, pending_forwards, pipe)
    without introducing a shared-mutable-module-state pattern.
    """
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.events import (
        QuicEvent, DatagramFrameReceived, StreamDataReceived,
        ConnectionTerminated, HandshakeCompleted,
    )
    from shared.node_client import QuicMessage, QuicMessageType

    class _ListenerProtocol(QuicConnectionProtocol):
        def __init__(self, quic_conn, stream_handler=None):
            super().__init__(quic_conn, stream_handler)
            self._peer_address: Optional[str] = None
            self._stream_buffers: Dict[int, bytearray] = {}

        def quic_event_received(self, event: QuicEvent) -> None:
            if isinstance(event, HandshakeCompleted):
                peername = (
                    self._quic._network_paths[0].addr
                    if self._quic._network_paths else None
                )
                if peername:
                    self._peer_address = f"{peername[0]}:{peername[1]}"
            elif isinstance(event, DatagramFrameReceived):
                asyncio.create_task(self._handle_bytes(event.data, None))
            elif isinstance(event, StreamDataReceived):
                buf = self._stream_buffers.setdefault(
                    event.stream_id, bytearray(),
                )
                buf.extend(event.data)
                if event.end_stream:
                    data = bytes(self._stream_buffers.pop(event.stream_id))
                    asyncio.create_task(
                        self._handle_bytes(data, event.stream_id)
                    )
            elif isinstance(event, ConnectionTerminated):
                self._stream_buffers.clear()

        async def _handle_bytes(
            self, data: bytes, response_stream_id: Optional[int],
        ) -> None:
            try:
                msg = QuicMessage.from_bytes(data)
            except Exception as exc:
                log.debug(
                    f"Invalid message from {self._peer_address}: {exc}"
                )
                return

            response: Optional[QuicMessage] = None
            try:
                if msg.msg_type == QuicMessageType.HEARTBEAT:
                    response = _handle_heartbeat_local(
                        msg, self._peer_address, cache, pipe, log,
                    )
                elif msg.msg_type == QuicMessageType.JOIN_REQUEST:
                    local = _handle_join_local(msg, cache)
                    if local is not None:
                        response = local
                    else:
                        response = await _forward_to_coordinator(
                            msg, self._peer_address, pipe,
                            pending_forwards, alloc_msg_id,
                            forward_timeout,
                        )
                else:
                    cached = _try_cached_read(msg, cache)
                    if cached is not None:
                        response = cached
                    else:
                        response = await _forward_to_coordinator(
                            msg, self._peer_address, pipe,
                            pending_forwards, alloc_msg_id,
                            forward_timeout,
                        )
            except Exception as exc:
                log.exception(
                    f"Handler error for {msg.msg_type.name}: {exc}"
                )
                response = msg.create_error_response(str(exc))

            if response is None:
                # Every well-formed request must elicit a response —
                # silent drop leaves clients unable to distinguish a
                # dead node from a degraded one, and bypasses
                # already-existing client error handling. Reaching
                # here means the cache was stale AND the coordinator
                # forward returned None (a contract violation logged
                # at WARNING by ``_forward_to_coordinator``); convert
                # the silent drop to an explicit ERROR_RESPONSE so
                # the client sees what's wrong on the next round-trip.
                response = msg.create_error_response(
                    f"unable to serve {msg.msg_type.name}: "
                    "cache stale and coordinator unreachable",
                )

            try:
                response_bytes = response.to_bytes()
                if len(response_bytes) > max_datagram_message_size:
                    sid = self._quic.get_next_available_stream_id()
                    self._quic.send_stream_data(
                        sid, response_bytes, end_stream=True,
                    )
                else:
                    self._quic.send_datagram_frame(response_bytes)
                self.transmit()
            except Exception as exc:
                log.error(f"Failed to send response: {exc}")

    return _ListenerProtocol(quic, stream_handler)


async def _periodic_cache_log(
    cache: _PeerSnapshotCache,
    log: logging.Logger,
    stop_event: mp.synchronize.Event,
    interval: float = 60.0,
) -> None:
    """Emit a cache-health summary every ``interval`` seconds.

    A predominantly-stale cache (``miss_stale`` rising while ``hits``
    stays flat) points at the snapshot loop being slow, dead, or its
    IPC drops being silently dropped. A predominantly-field-missing
    cache (``miss_field`` rising) points at a coordinator-side
    serializer failing — the warnings raised by ``_warn_rate_limited``
    on the parent should pinpoint which one.

    Snapshot age is reported in seconds since the last successful
    apply; -1 if no snapshot has ever been received.
    """
    while not stop_event.is_set():
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        age = (
            time.time() - cache.read_snapshot_ts
            if cache.read_snapshot_ts > 0 else -1.0
        )
        log.info(
            "listener-cache stats: hits=%d miss_stale=%d miss_field=%d "
            "snapshot_age=%.1fs status=%s stats=%s peers=%s",
            cache.hits, cache.miss_stale, cache.miss_field, age,
            "present" if cache.status_response else "ABSENT",
            "present" if cache.stats_response else "ABSENT",
            "present" if cache.peers_response else "ABSENT",
        )


def _apply_read_snapshot(cache: _PeerSnapshotCache, cmd: dict) -> None:
    """Replace the cache's read-snapshot fields from an IPC command.

    The coordinator sends already-serialized JSON response bytes
    for each type. ``None`` means "coordinator couldn't build this
    right now; keep serving from the coordinator via forward."
    """
    cache.status_response = _as_bytes(cmd.get("status"))
    cache.stats_response = _as_bytes(cmd.get("stats"))
    cache.peers_response = _as_bytes(cmd.get("peers"))
    telemetry = cmd.get("telemetry") or {}
    cache.telemetry_status_response = _as_bytes(telemetry.get("status"))
    cache.telemetry_nodes_response = _as_bytes(telemetry.get("nodes"))
    cache.telemetry_epochs_response = _as_bytes(telemetry.get("epochs"))
    cache.telemetry_latest_response = _as_bytes(telemetry.get("latest"))
    cache.telemetry_block_responses = {
        (epoch, int(idx)): _as_bytes(payload)
        for (epoch, idx), payload in (telemetry.get("blocks") or {}).items()
        if payload is not None
    }
    cache.read_snapshot_ts = time.time()


def _as_bytes(value) -> Optional[bytes]:
    """Normalize bytes/bytearray/None to Optional[bytes]."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def _try_cached_read(msg, cache: _PeerSnapshotCache):
    """Return a response built from the cache, or None to forward.

    Only handles the read-only message types. Returns None if the
    cache is stale, missing the needed payload, or the message type
    isn't in the local fast path. Updates the cache's hit/miss
    counters so the listener loop can log periodic health summaries.
    """
    from shared.node_client import QuicMessageType
    if not cache.read_snapshot_fresh():
        cache.miss_stale += 1
        return None

    t = msg.msg_type
    if t == QuicMessageType.STATUS_REQUEST:
        payload = cache.status_response
    elif t == QuicMessageType.STATS_REQUEST:
        payload = cache.stats_response
    elif t == QuicMessageType.PEERS_REQUEST:
        payload = cache.peers_response
    elif t == QuicMessageType.TELEMETRY_STATUS_REQUEST:
        payload = cache.telemetry_status_response
    elif t == QuicMessageType.TELEMETRY_NODES_REQUEST:
        payload = cache.telemetry_nodes_response
    elif t == QuicMessageType.TELEMETRY_EPOCHS_REQUEST:
        payload = cache.telemetry_epochs_response
    elif t == QuicMessageType.TELEMETRY_LATEST_REQUEST:
        payload = cache.telemetry_latest_response
    elif t == QuicMessageType.TELEMETRY_BLOCK_REQUEST:
        payload = _lookup_telemetry_block(msg, cache)
    else:
        # Not a read-only message; not a real cache miss — don't count.
        return None

    if payload is None:
        cache.miss_field += 1
        return None
    cache.hits += 1
    return msg.create_response(payload)


def _lookup_telemetry_block(msg, cache: _PeerSnapshotCache) -> Optional[bytes]:
    """Resolve a TELEMETRY_BLOCK_REQUEST against the local cache."""
    try:
        params = json.loads(msg.payload.decode("utf-8"))
        epoch = params["epoch"]
        block_index = int(params["block_index"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    return cache.telemetry_block_responses.get((epoch, block_index))


def _handle_heartbeat_local(
    msg,
    peer_address: Optional[str],
    cache: _PeerSnapshotCache,
    pipe: mp.connection.Connection,
    log: logging.Logger,
):
    """Build HEARTBEAT response from the local cache.

    Fires a ``peer_heartbeat`` event to the coordinator so it can
    update SWIM / scorer / telemetry at its own pace; the response
    is generated synchronously and sent immediately.
    """
    from shared.node_client import QuicMessage
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return msg.create_error_response("Invalid heartbeat payload")

    sender = data.get("sender")
    net_version = data.get("version")
    timestamp = data.get("timestamp", time.time())

    # Banned peers are refused without telling the coordinator.
    if sender and sender in cache.banned:
        return msg.create_response(
            json.dumps({"status": "backed_off"}).encode("utf-8")
        )

    # Coordinator will do full SWIM / telemetry update on its own loop.
    # Drop-newest: missed heartbeats re-emit on the next inbound, and
    # blocking here would stall the aioquic handler for every peer.
    if sender:
        _send_event_nowait({
            "event": "peer_heartbeat",
            "peer": sender,
            "version": net_version,
            "timestamp": timestamp,
        })

    return msg.create_response(
        json.dumps({"status": "ok"}).encode("utf-8")
    )


def _handle_join_local(msg, cache: _PeerSnapshotCache):
    """Answer JOIN locally when the claimed peer is in the banned set.

    The coordinator's capacity-rejection path records the claimed
    address in the ban list; the banned set is pushed to the
    listener every 5s via ``peer_snapshot``. On JOIN retries inside
    the cooldown window, answer from cache so the coordinator sees
    zero IPC pressure from the storm. Returns ``None`` to fall
    through to ``_forward_to_coordinator`` for the first rejection
    (which does the full snapshot/alt computation).

    Response shape matches the coordinator's fast-reject: same
    ``status: at_capacity`` envelope with empty alternatives, so
    clients honoring the protocol see no behavioral difference.
    """
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return None

    claimed = data.get("host")
    if not claimed or claimed not in cache.banned:
        return None

    return msg.create_response(json.dumps({
        "status": "at_capacity",
        "peers": {},
        "peer_versions": {},
    }).encode("utf-8"))


async def _forward_to_coordinator(
    msg,
    peer_address: Optional[str],
    pipe: mp.connection.Connection,
    pending_forwards: Dict[int, asyncio.Future],
    alloc_msg_id,
    forward_timeout: float,
):
    """Ship a message to the coordinator and await the matching response."""
    from shared.node_client import QuicMessage

    msg_id = alloc_msg_id()
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    pending_forwards[msg_id] = future
    raw_b64 = base64.b64encode(msg.to_bytes()).decode("ascii")
    # Critical: coordinator is about to respond via ``inbound_response``
    # and the QUIC peer is waiting on the reply. Timeout mirrors the
    # forward_timeout so we fail fast when the coord is overloaded.
    ok = await _send_event({
        "event": "inbound_message",
        "msg_id": msg_id,
        "peer": peer_address or "",
        "raw_b64": raw_b64,
    }, critical=True, timeout=forward_timeout)
    if not ok:
        pending_forwards.pop(msg_id, None)
        return msg.create_error_response("Coordinator pipe broken")

    try:
        payload = await asyncio.wait_for(future, timeout=forward_timeout)
    except asyncio.TimeoutError:
        pending_forwards.pop(msg_id, None)
        return msg.create_error_response("Coordinator timeout")

    if payload is None:
        # Coordinator replied but with payload_b64 == None. This means
        # ``_handle_quic_message`` returned None (which it shouldn't
        # for any read message type) or the coordinator failed to
        # decode the forwarded raw bytes. Either is a contract bug —
        # surface it loudly so the next failure isn't another silent
        # drop. The caller's invariant repair returns ERROR_RESPONSE.
        logging.getLogger("listener").warning(
            "Coordinator returned payload_b64=None for %s "
            "(request_id=%d); silent-drop path was triggered",
            msg.msg_type.name, msg.request_id,
        )
        return None
    try:
        return QuicMessage.from_bytes(payload)
    except Exception:
        return msg.create_error_response("Invalid coordinator response")


def _send_safe(pipe: mp.connection.Connection, msg: dict) -> bool:
    try:
        pipe.send(msg)
        return True
    except (BrokenPipeError, OSError):
        return False


def spawn_listener_process(
    config_bundle: dict,
) -> ListenerProcessHandle:
    """Spawn the listener process and return a parent-side handle.

    ``config_bundle`` must be picklable (no live sockets or loggers);
    see ``listener_process_main`` for required keys.
    """
    parent_pipe, child_pipe = mp.Pipe(duplex=True)
    stop_event = mp.Event()
    process = mp.Process(
        target=listener_process_main,
        args=(child_pipe, stop_event, config_bundle),
        daemon=True,
    )
    process.start()
    child_pipe.close()
    return ListenerProcessHandle(
        process=process, pipe=parent_pipe, stop_event=stop_event,
    )
