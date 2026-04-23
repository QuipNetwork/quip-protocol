"""Per-connection child process for peer isolation.

Each active peer connection gets its own OS process with its own asyncio
event loop and QUIC client. The process communicates with the parent
NetworkNode via a bidirectional ``multiprocessing.Pipe``.

IPC protocol (parent <-> child)
-------------------------------
Parent -> Child::

    {"cmd": "heartbeat", "public_host": str, "version": str, "miner_info": dict}
    {"cmd": "gossip", "payload": <base64-encoded bytes>}
    {"cmd": "request_block", "request_id": int, "block_num": int}
    {"cmd": "request_block_header", "request_id": int, "block_num": int}
    {"cmd": "request_status", "request_id": int}
    {"cmd": "request_peers", "request_id": int}
    {"cmd": "probe_request", "request_id": int, "target": str, "probe_id": str}
    {"cmd": "shutdown"}

Child -> Parent::

    {"event": "heartbeat_ok"}
    {"event": "heartbeat_fail", "reason": str}
    {"event": "gossip_result", "success": bool}
    {"event": "block_data", "request_id": int, "data": <base64>, "block_num": int}
    {"event": "block_header_data", "request_id": int, "data": <base64>, "block_num": int}
    {"event": "status_data", "request_id": int, "data": dict}
    {"event": "peers_data", "request_id": int, "data": dict}
    {"event": "probe_result", "request_id": int, "result": Optional[bool]}
    {"event": "disconnected", "reason": str}
    {"event": "error", "request_id": Optional[int], "message": str}

``request_id`` is opaque to the child; the parent uses it to resolve
``asyncio.Future`` objects waiting on the response. Fire-and-forget
commands (``heartbeat``, ``gossip``, ``shutdown``) omit it.
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
from typing import Optional

from shared.logging_config import QuipFormatter
from shared.pipe_sender import AsyncPipeSender
from shared.telemetry_sink import configure_sink, get_sink


def _setup_child_logging(peer: str) -> logging.Logger:
    """Configure console logging for child process."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(QuipFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger(f"conn[{peer}]")


@dataclass
class ConnectionProcessHandle:
    """Parent-side handle to a connection child process.

    Outbound commands go through ``sender`` (an ``AsyncPipeSender``)
    so that ``send_cmd`` never blocks the coordinator's event loop on
    a full pipe. Inbound events are still polled via ``recv`` from the
    coordinator's event-driven reader.
    """
    peer_address: str
    process: mp.Process
    pipe: mp.connection.Connection
    stop_event: mp.synchronize.Event
    sender: AsyncPipeSender = field(init=False)
    started_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # The sender lazy-starts its drainer on first send so this
        # object is safe to construct in sync contexts (tests). In
        # production it is constructed from ``ProcessPool.spawn`` on
        # the coordinator's event-loop thread.
        self.sender = AsyncPipeSender(
            self.pipe, maxsize=256, name=self.peer_address,
        )

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send_cmd(self, cmd: dict) -> bool:
        """Enqueue a command for async send. Drops on full queue.

        Non-blocking; the actual ``pipe.send`` runs off-thread via the
        sender's drainer task. Returns False if the queue is full or
        the pipe is already known broken.
        """
        return self.sender.send_nowait(cmd)

    async def send_cmd_blocking(
        self, cmd: dict, timeout: float = 5.0,
    ) -> bool:
        """Await enqueue with timeout. For lifecycle / critical commands."""
        return await self.sender.send_blocking(cmd, timeout=timeout)

    def poll(self, timeout: float = 0) -> bool:
        """Check if data is available on the pipe."""
        try:
            return self.pipe.poll(timeout)
        except (BrokenPipeError, OSError, EOFError):
            return False

    def recv(self) -> Optional[dict]:
        """Non-blocking receive. Returns None if nothing available.

        Returns a synthetic ``{"event": "disconnected", ...}`` dict
        if the pipe is broken, so callers can distinguish "no data"
        from "child process is gone."
        """
        try:
            if self.pipe.poll(0):
                msg = self.pipe.recv()
                self.last_activity = time.monotonic()
                return msg
        except (BrokenPipeError, OSError, EOFError):
            return {"event": "disconnected", "reason": "pipe broken"}
        return None

    def shutdown(self) -> None:
        """Request graceful shutdown of the child process.

        The ``stop_event`` is the primary path; the shutdown command
        is best-effort (may be dropped on full queue, in which case
        the child exits via the ``stop_event`` poll loop instead).
        """
        self.stop_event.set()
        self.send_cmd({"cmd": "shutdown"})

    def force_stop(self, timeout: float = 3.0) -> None:
        """Terminate the child process, escalating to kill if needed."""
        self.stop_event.set()
        # The sender owns an asyncio task; best-effort signal it to
        # stop accepting new sends. The async ``close`` is handled by
        # the ProcessPool teardown path which has access to the loop.
        self.sender._closing = True
        if not self.process.is_alive():
            return
        self.process.terminate()
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)

    async def aclose(self, timeout: float = 2.0) -> None:
        """Async-close the outbound sender (drainer task).

        Pool teardown calls this before ``force_stop`` to flush any
        queued shutdown commands rather than dropping them.
        """
        await self.sender.close(timeout=timeout)


def connection_process_main(
    pipe: mp.connection.Connection,
    peer_address: str,
    node_timeout: float,
    verify_tls: bool,
    stop_event: mp.synchronize.Event,
    connect_timeout: Optional[float] = None,
    telemetry_socket: Optional[str] = None,
) -> None:
    """Child process entry point: own event loop, single QUIC connection.

    Args:
        pipe: Bidirectional pipe to parent process.
        peer_address: ``host:port`` of the remote peer to connect to.
        node_timeout: Default timeout for QUIC requests.
        verify_tls: Whether to verify TLS certificates.
        stop_event: Multiprocessing event set by parent to request shutdown.
        connect_timeout: Upper bound on a single connect attempt
            (including the close handshake on unreachable peers).
            ``None`` preserves the default ~5 s handshake + up to
            ~10 s cleanup behavior.
        telemetry_socket: Path to the AF_UNIX DGRAM sink exposed by
            ``telemetry_aggregator``. When provided, observability
            events (``gossip_stats`` today) are emitted there instead
            of the control pipe. ``None`` disables observability
            emission — useful in tests.
    """
    def _signal_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log = _setup_child_logging(peer_address)
    log.info(f"Connection process started for {peer_address}")

    if telemetry_socket:
        configure_sink(telemetry_socket)

    from shared.event_loop import create_event_loop
    loop = create_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _connection_loop(pipe, peer_address, node_timeout,
                             verify_tls, stop_event, log, connect_timeout)
        )
    except Exception as exc:
        log.error(f"Connection process crashed: {exc}")
        _send_safe(pipe, {"event": "disconnected", "reason": str(exc)})
    finally:
        loop.close()
        log.info(f"Connection process stopped for {peer_address}")


@dataclass
class GossipStats:
    """Rolling counters for outbound GOSSIP success/failure per peer.

    Populated by ``_handle_gossip`` and reported every
    ``_GOSSIP_STATS_INTERVAL`` seconds so operators can see which peers
    reply reliably vs. which drop our datagrams. A failure here means
    either a 5 s client-side timeout (``node_client.py:362``) or an
    error response — both count equally for health-tracking purposes.
    """
    sent: int = 0
    responded: int = 0
    failed: int = 0


_GOSSIP_STATS_INTERVAL = 60.0

_WORKER_SENDER: Optional[AsyncPipeSender] = None
"""Process-global outbound sender (worker → coord). Initialized by
``_connection_loop`` at startup and closed in its ``finally``. Module-
global is safe because one connection_process == one OS process."""


async def _send_event(
    msg: dict, *, critical: bool = False, timeout: float = 5.0,
) -> bool:
    """Async send helper for worker → coord events.

    ``critical=True`` awaits enqueue with *timeout*; used for RPC
    replies and terminal events where a silent drop would strand a
    coord-side future. ``critical=False`` is drop-newest, used for
    idempotent/re-emittable events (heartbeat_ok/fail, gossip_result)
    whose loss the coordinator tolerates.
    """
    sender = _WORKER_SENDER
    if sender is None:
        return False
    if critical:
        return await sender.send_blocking(msg, timeout=timeout)
    return sender.send_nowait(msg)


def _report_gossip_stats(
    stats: 'GossipStats',
    pipe: mp.connection.Connection,
    peer_address: str,
    log: logging.Logger,
) -> None:
    """Log a one-line summary and emit stats via the telemetry sink.

    Pure observability: never crosses the control-plane pipe. Emission
    is lossy (AF_UNIX DGRAM, non-blocking); the ``pipe`` argument stays
    in the signature for backward-compatible callers but is unused.
    """
    del pipe  # control-plane pipe intentionally unused for observability
    if stats.sent == 0:
        return
    rate = (stats.responded / stats.sent) * 100.0
    log.info(
        f"gossip_stats {peer_address}: sent={stats.sent} "
        f"responded={stats.responded} failed={stats.failed} "
        f"({rate:.0f}% success)"
    )
    sink = get_sink()
    if sink is not None:
        sink.emit({
            "peer": peer_address,
            "sent": stats.sent,
            "responded": stats.responded,
            "failed": stats.failed,
            "ts": time.time(),
        })


async def _connection_loop(
    pipe: mp.connection.Connection,
    peer_address: str,
    node_timeout: float,
    verify_tls: bool,
    stop_event: mp.synchronize.Event,
    log: logging.Logger,
    connect_timeout: Optional[float] = None,
) -> None:
    """Async main loop: establish connection, then process commands."""
    from shared.node_client import NodeClient

    global _WORKER_SENDER
    _WORKER_SENDER = AsyncPipeSender(
        pipe, maxsize=128, name=f"worker:{peer_address}", logger=log,
    )
    _WORKER_SENDER.start()

    client = NodeClient(
        node_timeout=node_timeout,
        verify_tls=verify_tls,
        connect_timeout=connect_timeout,
    )
    await client.start()

    loop = asyncio.get_event_loop()
    stats = GossipStats()
    last_stats_report = time.monotonic()

    try:
        # Race connection establishment against shutdown
        connect_task = asyncio.create_task(
            _connect_with_cancel(client, peer_address, stop_event, loop)
        )
        connected = await connect_task
        if not connected:
            if not stop_event.is_set():
                await _send_event({
                    "event": "disconnected",
                    "reason": "Failed to establish QUIC connection",
                }, critical=True)
            return

        # Main command loop
        while not stop_event.is_set():
            now = time.monotonic()
            if now - last_stats_report >= _GOSSIP_STATS_INTERVAL:
                _report_gossip_stats(stats, pipe, peer_address, log)
                last_stats_report = now

            # Poll pipe for commands (non-blocking via executor)
            try:
                has_data = await loop.run_in_executor(
                    None, pipe.poll, 0.5
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
            elif op == "heartbeat":
                await _handle_heartbeat(client, pipe, peer_address, cmd, log)
            elif op == "gossip":
                await _handle_gossip(
                    client, pipe, peer_address, cmd, log, stats
                )
            elif op == "request_block":
                await _handle_request_block(
                    client, pipe, peer_address, cmd, log
                )
            elif op == "request_block_header":
                await _handle_request_block_header(
                    client, pipe, peer_address, cmd, log
                )
            elif op == "request_status":
                await _handle_request_status(
                    client, pipe, peer_address, cmd, log
                )
            elif op == "request_peers":
                await _handle_request_peers(
                    client, pipe, peer_address, cmd, log
                )
            elif op == "probe_request":
                await _handle_probe_request(
                    client, pipe, peer_address, cmd, log
                )
            else:
                log.warning(f"Unknown command: {op}")

    finally:
        _report_gossip_stats(stats, pipe, peer_address, log)
        try:
            await asyncio.wait_for(client.stop(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass
        # Drain the outbound sender so queued RPC replies reach the
        # coordinator before the process exits.
        sender = _WORKER_SENDER
        if sender is not None:
            try:
                await sender.close(timeout=2.0)
            except Exception:
                pass


async def _connect_with_cancel(
    client: 'NodeClient',
    peer_address: str,
    stop_event: mp.synchronize.Event,
    loop: asyncio.AbstractEventLoop,
) -> bool:
    """Attempt connection, cancelling early if stop_event is set."""
    connect_task = asyncio.ensure_future(
        client.connect_to_peer(peer_address)
    )
    # Poll stop_event every 0.5s while connection is in progress
    while not connect_task.done():
        if stop_event.is_set():
            connect_task.cancel()
            return False
        await asyncio.sleep(0.5)
    try:
        return connect_task.result()
    except asyncio.CancelledError:
        return False
    except Exception as exc:
        logging.getLogger(__name__).debug(
            f"Connection to {peer_address} failed: {exc}"
        )
        return False


async def _handle_heartbeat(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Send heartbeat to peer via the dedicated connection."""
    from shared.block import MinerInfo
    try:
        public_host = cmd.get("public_host", "")
        miner_info_data = cmd.get("miner_info", {})
        miner_info = MinerInfo.from_json(json.dumps(miner_info_data))
        ok = await client.send_heartbeat(peer_address, public_host, miner_info)
        if ok:
            await _send_event({"event": "heartbeat_ok"})
        else:
            await _send_event({
                "event": "heartbeat_fail",
                "reason": "no response",
            })
    except Exception as exc:
        log.debug(f"Heartbeat error: {exc}")
        await _send_event({"event": "heartbeat_fail", "reason": str(exc)})


async def _handle_gossip(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
    stats: 'GossipStats',
) -> None:
    """Forward a gossip message to the peer."""
    stats.sent += 1
    try:
        from shared.network_node import Message
        payload_b64 = cmd.get("payload", "")
        payload_bytes = base64.b64decode(payload_b64)
        message = Message.from_network(payload_bytes)
        ok = await client.gossip_to(peer_address, message)
        if ok:
            stats.responded += 1
        else:
            stats.failed += 1
        await _send_event({"event": "gossip_result", "success": ok})
    except Exception as exc:
        stats.failed += 1
        log.debug(f"Gossip error: {exc}")
        await _send_event({"event": "gossip_result", "success": False})


async def _handle_request_block(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Download a specific block from the peer."""
    request_id = cmd.get("request_id")
    block_num = cmd.get("block_num", 0)
    try:
        block = await client.get_peer_block(peer_address, block_num)
        if block is not None:
            block_bytes = block.to_network()
            await _send_event({
                "event": "block_data",
                "request_id": request_id,
                "block_num": block_num,
                "data": base64.b64encode(block_bytes).decode('ascii'),
            }, critical=True)
        else:
            await _send_event({
                "event": "error",
                "request_id": request_id,
                "message": f"Block {block_num} not available from {peer_address}",
            }, critical=True)
    except Exception as exc:
        log.debug(f"Block request error: {exc}")
        await _send_event({
            "event": "error", "request_id": request_id, "message": str(exc),
        }, critical=True)


async def _handle_request_block_header(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Download only the block header from the peer (smaller than full block)."""
    request_id = cmd.get("request_id")
    block_num = cmd.get("block_num", 0)
    try:
        header = await client.get_peer_block_header(peer_address, block_num)
        if header is not None:
            header_bytes = header.to_network()
            await _send_event({
                "event": "block_header_data",
                "request_id": request_id,
                "block_num": block_num,
                "data": base64.b64encode(header_bytes).decode('ascii'),
            }, critical=True)
        else:
            await _send_event({
                "event": "error",
                "request_id": request_id,
                "message": f"Header {block_num} not available from {peer_address}",
            }, critical=True)
    except Exception as exc:
        log.debug(f"Block header request error: {exc}")
        await _send_event({
            "event": "error", "request_id": request_id, "message": str(exc),
        }, critical=True)


async def _handle_request_status(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Get status from the peer."""
    request_id = cmd.get("request_id")
    try:
        status = await client.get_peer_status(peer_address)
        if status is not None:
            await _send_event({
                "event": "status_data",
                "request_id": request_id,
                "data": status,
            }, critical=True)
        else:
            await _send_event({
                "event": "error",
                "request_id": request_id,
                "message": f"No status from {peer_address}",
            }, critical=True)
    except Exception as exc:
        log.debug(f"Status request error: {exc}")
        await _send_event({
            "event": "error", "request_id": request_id, "message": str(exc),
        }, critical=True)


async def _handle_request_peers(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Get peer list from the peer."""
    from shared.node_client import QuicMessageType
    request_id = cmd.get("request_id")
    try:
        protocol = await client._get_connection(peer_address)
        if protocol is None:
            await _send_event({
                "event": "error",
                "request_id": request_id,
                "message": f"No connection to {peer_address}",
            }, critical=True)
            return
        response = await protocol.send_request(
            QuicMessageType.PEERS_REQUEST, b'',
            timeout=client.node_timeout,
        )
        if response and response.msg_type == QuicMessageType.PEERS_RESPONSE:
            data = json.loads(response.payload.decode('utf-8'))
            await _send_event({
                "event": "peers_data",
                "request_id": request_id,
                "data": data,
            }, critical=True)
        else:
            await _send_event({
                "event": "error",
                "request_id": request_id,
                "message": f"No peers response from {peer_address}",
            }, critical=True)
    except Exception as exc:
        log.debug(f"Peers request error: {exc}")
        await _send_event({
            "event": "error", "request_id": request_id, "message": str(exc),
        }, critical=True)


async def _handle_probe_request(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Ask this peer (the prober) to probe a target on our behalf.

    ``peer_address`` is the prober; the target and probe_id come from
    the command payload. Result mirrors ``NodeClient.send_probe_request``:
    True=alive, False=unreachable, None=error/timeout.
    """
    request_id = cmd.get("request_id")
    target = cmd.get("target", "")
    probe_id = cmd.get("probe_id", "")
    try:
        result = await client.send_probe_request(
            peer_address, target, probe_id,
        )
        await _send_event({
            "event": "probe_result",
            "request_id": request_id,
            "result": result,
        }, critical=True)
    except Exception as exc:
        log.debug(f"Probe request error: {exc}")
        await _send_event({
            "event": "probe_result",
            "request_id": request_id,
            "result": None,
        }, critical=True)


def _send_safe(pipe: mp.connection.Connection, msg: dict) -> bool:
    """Send a message over the pipe, returning False on broken pipe."""
    try:
        pipe.send(msg)
        return True
    except (BrokenPipeError, OSError):
        return False


def spawn_connection_process(
    peer_address: str,
    node_timeout: float = 10.0,
    verify_tls: bool = False,
    connect_timeout: Optional[float] = None,
    telemetry_socket: Optional[str] = None,
) -> ConnectionProcessHandle:
    """Spawn a new connection process for the given peer.

    Returns a ``ConnectionProcessHandle`` that the parent uses to
    send commands and receive events.
    """
    parent_pipe, child_pipe = mp.Pipe(duplex=True)
    stop_event = mp.Event()
    process = mp.Process(
        target=connection_process_main,
        args=(child_pipe, peer_address, node_timeout, verify_tls,
              stop_event, connect_timeout, telemetry_socket),
        daemon=True,
    )
    process.start()
    # Close child end in parent to avoid FD leak
    child_pipe.close()
    return ConnectionProcessHandle(
        peer_address=peer_address,
        process=process,
        pipe=parent_pipe,
        stop_event=stop_event,
    )
