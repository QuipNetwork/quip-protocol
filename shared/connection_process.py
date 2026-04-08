"""Per-connection child process for peer isolation.

Each active peer connection gets its own OS process with its own asyncio
event loop and QUIC client. The process communicates with the parent
NetworkNode via a bidirectional ``multiprocessing.Pipe``.

IPC protocol (parent <-> child)
-------------------------------
Parent -> Child::

    {"cmd": "heartbeat", "public_host": str, "version": str, "miner_info": dict}
    {"cmd": "gossip", "payload": <base64-encoded bytes>}
    {"cmd": "request_block", "block_num": int}
    {"cmd": "request_status"}
    {"cmd": "request_peers"}
    {"cmd": "shutdown"}

Child -> Parent::

    {"event": "heartbeat_ok"}
    {"event": "heartbeat_fail", "reason": str}
    {"event": "gossip_result", "success": bool}
    {"event": "block_data", "data": <base64-encoded bytes>}
    {"event": "status_data", "data": dict}
    {"event": "peers_data", "data": dict}
    {"event": "disconnected", "reason": str}
    {"event": "error", "message": str}
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
    """Parent-side handle to a connection child process."""
    peer_address: str
    process: mp.Process
    pipe: mp.connection.Connection
    stop_event: mp.synchronize.Event
    started_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send_cmd(self, cmd: dict) -> bool:
        """Send a command to the child process. Returns False if pipe is broken."""
        try:
            self.pipe.send(cmd)
            return True
        except (BrokenPipeError, OSError):
            return False

    def poll(self, timeout: float = 0) -> bool:
        """Check if data is available on the pipe."""
        try:
            return self.pipe.poll(timeout)
        except (BrokenPipeError, OSError, EOFError):
            return False

    def recv(self) -> Optional[dict]:
        """Non-blocking receive. Returns None if nothing available or pipe broken."""
        try:
            if self.pipe.poll(0):
                msg = self.pipe.recv()
                self.last_activity = time.monotonic()
                return msg
        except (BrokenPipeError, OSError, EOFError):
            pass
        return None

    def shutdown(self) -> None:
        """Request graceful shutdown of the child process."""
        self.stop_event.set()
        self.send_cmd({"cmd": "shutdown"})

    def force_stop(self, timeout: float = 3.0) -> None:
        """Terminate the child process, escalating to kill if needed."""
        self.stop_event.set()
        if not self.process.is_alive():
            return
        self.process.terminate()
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)


def connection_process_main(
    pipe: mp.connection.Connection,
    peer_address: str,
    node_timeout: float,
    verify_tls: bool,
    stop_event: mp.synchronize.Event,
) -> None:
    """Child process entry point: own event loop, single QUIC connection.

    Args:
        pipe: Bidirectional pipe to parent process.
        peer_address: ``host:port`` of the remote peer to connect to.
        node_timeout: Default timeout for QUIC requests.
        verify_tls: Whether to verify TLS certificates.
        stop_event: Multiprocessing event set by parent to request shutdown.
    """
    def _signal_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log = _setup_child_logging(peer_address)
    log.info(f"Connection process started for {peer_address}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _connection_loop(pipe, peer_address, node_timeout,
                             verify_tls, stop_event, log)
        )
    except Exception as exc:
        log.error(f"Connection process crashed: {exc}")
        _send_safe(pipe, {"event": "disconnected", "reason": str(exc)})
    finally:
        loop.close()
        log.info(f"Connection process stopped for {peer_address}")


async def _connection_loop(
    pipe: mp.connection.Connection,
    peer_address: str,
    node_timeout: float,
    verify_tls: bool,
    stop_event: mp.synchronize.Event,
    log: logging.Logger,
) -> None:
    """Async main loop: establish connection, then process commands."""
    from shared.node_client import NodeClient

    client = NodeClient(node_timeout=node_timeout, verify_tls=verify_tls)
    await client.start()

    loop = asyncio.get_event_loop()

    try:
        # Race connection establishment against shutdown
        connect_task = asyncio.create_task(
            _connect_with_cancel(client, peer_address, stop_event, loop)
        )
        connected = await connect_task
        if not connected:
            if not stop_event.is_set():
                _send_safe(pipe, {
                    "event": "disconnected",
                    "reason": "Failed to establish QUIC connection",
                })
            return

        # Main command loop
        while not stop_event.is_set():
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
                await _handle_gossip(client, pipe, peer_address, cmd, log)
            elif op == "request_block":
                await _handle_request_block(
                    client, pipe, peer_address, cmd, log
                )
            elif op == "request_status":
                await _handle_request_status(client, pipe, peer_address, log)
            elif op == "request_peers":
                await _handle_request_peers(client, pipe, peer_address, log)
            else:
                log.warning(f"Unknown command: {op}")

    finally:
        try:
            await asyncio.wait_for(client.stop(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
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
    except (asyncio.CancelledError, Exception):
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
            _send_safe(pipe, {"event": "heartbeat_ok"})
        else:
            _send_safe(pipe, {
                "event": "heartbeat_fail",
                "reason": "no response",
            })
    except Exception as exc:
        log.debug(f"Heartbeat error: {exc}")
        _send_safe(pipe, {"event": "heartbeat_fail", "reason": str(exc)})


async def _handle_gossip(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Forward a gossip message to the peer."""
    try:
        from shared.network_node import Message
        payload_b64 = cmd.get("payload", "")
        payload_bytes = base64.b64decode(payload_b64)
        message = Message.from_network(payload_bytes)
        ok = await client.gossip_to(peer_address, message)
        _send_safe(pipe, {"event": "gossip_result", "success": ok})
    except Exception as exc:
        log.debug(f"Gossip error: {exc}")
        _send_safe(pipe, {"event": "gossip_result", "success": False})


async def _handle_request_block(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    cmd: dict,
    log: logging.Logger,
) -> None:
    """Download a specific block from the peer."""
    block_num = cmd.get("block_num", 0)
    try:
        block = await client.get_peer_block(peer_address, block_num)
        if block is not None:
            block_bytes = block.to_network()
            _send_safe(pipe, {
                "event": "block_data",
                "block_num": block_num,
                "data": base64.b64encode(block_bytes).decode('ascii'),
            })
        else:
            _send_safe(pipe, {
                "event": "error",
                "message": f"Block {block_num} not available from {peer_address}",
            })
    except Exception as exc:
        log.debug(f"Block request error: {exc}")
        _send_safe(pipe, {"event": "error", "message": str(exc)})


async def _handle_request_status(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    log: logging.Logger,
) -> None:
    """Get status from the peer."""
    try:
        status = await client.get_peer_status(peer_address)
        if status is not None:
            _send_safe(pipe, {"event": "status_data", "data": status})
        else:
            _send_safe(pipe, {
                "event": "error",
                "message": f"No status from {peer_address}",
            })
    except Exception as exc:
        log.debug(f"Status request error: {exc}")
        _send_safe(pipe, {"event": "error", "message": str(exc)})


async def _handle_request_peers(
    client: 'NodeClient',
    pipe: mp.connection.Connection,
    peer_address: str,
    log: logging.Logger,
) -> None:
    """Get peer list from the peer."""
    from shared.node_client import QuicMessageType
    try:
        protocol = await client._get_connection(peer_address)
        if protocol is None:
            _send_safe(pipe, {
                "event": "error",
                "message": f"No connection to {peer_address}",
            })
            return
        response = await protocol.send_request(
            QuicMessageType.PEERS_REQUEST, b'',
            timeout=client.node_timeout,
        )
        if response and response.msg_type == QuicMessageType.PEERS_RESPONSE:
            data = json.loads(response.payload.decode('utf-8'))
            _send_safe(pipe, {"event": "peers_data", "data": data})
        else:
            _send_safe(pipe, {
                "event": "error",
                "message": f"No peers response from {peer_address}",
            })
    except Exception as exc:
        log.debug(f"Peers request error: {exc}")
        _send_safe(pipe, {"event": "error", "message": str(exc)})


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
) -> ConnectionProcessHandle:
    """Spawn a new connection process for the given peer.

    Returns a ``ConnectionProcessHandle`` that the parent uses to
    send commands and receive events.
    """
    parent_pipe, child_pipe = mp.Pipe(duplex=True)
    stop_event = mp.Event()
    process = mp.Process(
        target=connection_process_main,
        args=(child_pipe, peer_address, node_timeout, verify_tls, stop_event),
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
