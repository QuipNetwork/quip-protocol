"""Process pool for per-connection peer management.

Manages a pool of ``ConnectionProcessHandle`` instances, one per active
peer. Enforces a configurable ``max_connections`` limit, routes commands
to individual peers, and collects events from all child processes.

The parent ``NetworkNode`` interacts exclusively through this pool rather
than holding QUIC connections directly.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from shared.connection_process import (
    ConnectionProcessHandle,
    spawn_connection_process,
)


EventCallback = Callable[[str, dict], None]
"""(peer_address, event_dict) -> None. Dispatched on the event loop
thread by the add_reader callback; must not block."""


@dataclass
class ProcessPoolConfig:
    """Configuration for the peer connection process pool."""
    max_connections: int = 50
    node_timeout: float = 10.0
    verify_tls: bool = False
    # How long to wait before considering a process dead (no activity)
    process_idle_timeout: float = 600.0
    # Upper bound on a single connect attempt (handshake + cleanup).
    # ``None`` preserves the default ~5 s handshake + up to ~10 s
    # cleanup behavior. Tests probing unreachable addresses should set
    # a small value (e.g. 1.0) to avoid burning time on QUIC's PTO-
    # backoff close retries.
    connect_timeout: Optional[float] = None
    # Refuse to respawn a peer's process within this many seconds of
    # its last kill. Guards against tight spawn/die loops when a peer
    # is unreachable or crashes its child repeatedly.
    spawn_cooldown: float = 30.0
    # Path to the AF_UNIX DGRAM sink exposed by ``telemetry_aggregator``.
    # Workers emit observability events (``gossip_stats``) here instead
    # of through the control pipe. ``None`` disables emission.
    telemetry_socket: Optional[str] = None


class ProcessPool:
    """Manages per-connection child processes for all active peers.

    Each peer gets its own OS process with a dedicated QUIC connection.
    The pool enforces connection limits and provides a unified interface
    for the parent NetworkNode to interact with peers.

    Args:
        config: Pool configuration.
        logger: Logger instance.
    """

    def __init__(
        self,
        config: Optional[ProcessPoolConfig] = None,
        logger: Optional[logging.Logger] = None,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_event: Optional[EventCallback] = None,
    ):
        self.config = config or ProcessPoolConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._handles: Dict[str, ConnectionProcessHandle] = {}
        # Timestamp of the most recent kill per peer, used to enforce
        # ``config.spawn_cooldown`` against tight respawn loops.
        self._last_kill_time: Dict[str, float] = {}
        # When *loop* and *on_event* are provided, the pool drains each
        # handle's pipe via ``loop.add_reader`` (event-driven). When
        # absent, callers must drive ``poll_events`` themselves.
        # Provided in production by NetworkNode.start(); left None in
        # tests that predate the callback path.
        self._loop = loop
        self._on_event = on_event

    @property
    def connection_count(self) -> int:
        """Number of active connection processes."""
        return len(self._handles)

    @property
    def is_at_capacity(self) -> bool:
        """Whether the pool has reached max_connections."""
        return self.connection_count >= self.config.max_connections

    @property
    def connected_peers(self) -> List[str]:
        """List of peer addresses with active connections."""
        return list(self._handles.keys())

    def has_peer(self, peer: str) -> bool:
        """Check if a connection process exists for this peer."""
        return peer in self._handles

    def spawn(self, peer_address: str) -> bool:
        """Spawn a new connection process for a peer.

        Returns:
            True if spawned successfully, False if at capacity,
            already connected, or the peer is still in spawn cooldown.
        """
        if peer_address in self._handles:
            self.logger.debug(f"Already connected to {peer_address}")
            return False

        if self.is_at_capacity:
            self.logger.warning(
                f"At capacity ({self.config.max_connections}), "
                f"cannot spawn connection to {peer_address}"
            )
            return False

        last_kill = self._last_kill_time.get(peer_address)
        if last_kill is not None:
            elapsed = time.monotonic() - last_kill
            if elapsed < self.config.spawn_cooldown:
                self.logger.debug(
                    f"Spawn for {peer_address} throttled "
                    f"({elapsed:.1f}s < {self.config.spawn_cooldown}s)"
                )
                return False

        handle = spawn_connection_process(
            peer_address=peer_address,
            node_timeout=self.config.node_timeout,
            verify_tls=self.config.verify_tls,
            connect_timeout=self.config.connect_timeout,
            telemetry_socket=self.config.telemetry_socket,
        )
        self._handles[peer_address] = handle
        self._register_reader(peer_address, handle)
        self.logger.info(
            f"Spawned connection process for {peer_address} "
            f"(pid={handle.process.pid}, {self.connection_count}/"
            f"{self.config.max_connections})"
        )
        return True

    def _register_reader(
        self, peer_address: str, handle: ConnectionProcessHandle,
    ) -> None:
        """Wire the handle's pipe into the event loop's selector.

        No-op when ``loop``/``on_event`` were not provided (test path).
        """
        if self._loop is None or self._on_event is None:
            return
        try:
            fd = handle.pipe.fileno()
        except (OSError, ValueError):
            return
        self._loop.add_reader(
            fd, self._on_handle_readable, peer_address, handle,
        )

    def _unregister_reader(self, handle: ConnectionProcessHandle) -> None:
        """Detach the handle's pipe from the loop. Idempotent and
        tolerant of closed fds, dead loops, and double-unregister."""
        if self._loop is None:
            return
        try:
            fd = handle.pipe.fileno()
        except (OSError, ValueError):
            return
        try:
            self._loop.remove_reader(fd)
        except (ValueError, OSError, RuntimeError):
            # ValueError: fd already closed. OSError: loop shutting
            # down. RuntimeError: reader already removed.
            pass

    def _on_handle_readable(
        self, peer_address: str, handle: ConnectionProcessHandle,
    ) -> None:
        """Drain all pending events from *handle* and dispatch each.

        Runs on the event loop thread; called by ``loop.add_reader``
        whenever the kernel reports the pipe readable. Must not block
        — slow handlers should ``asyncio.create_task(...)`` themselves.
        """
        if self._on_event is None:
            return
        while True:
            try:
                if not handle.pipe.poll(0):
                    return
                msg = handle.pipe.recv()
            except (BrokenPipeError, OSError, EOFError) as exc:
                # Pipe gone: emit a synthetic disconnected and stop
                # listening on the fd. ``reap_dead`` will clean up the
                # entry on its next pass.
                self._unregister_reader(handle)
                self._on_event(peer_address, {
                    "event": "disconnected",
                    "reason": f"pipe broken: {exc}",
                })
                return
            try:
                self._on_event(peer_address, msg)
            except Exception:
                self.logger.exception(
                    "ProcessPool dispatch raised for %s", peer_address,
                )

    def kill(self, peer_address: str, timeout: float = 3.0) -> bool:
        """Stop and remove a connection process for a peer.

        Returns True if the peer had a connection that was removed.
        Also records a spawn cooldown to suppress immediate respawn.
        """
        handle = self._handles.pop(peer_address, None)
        if handle is None:
            return False

        self._unregister_reader(handle)
        handle.shutdown()
        handle.force_stop(timeout=timeout)
        self._last_kill_time[peer_address] = time.monotonic()
        self.logger.info(f"Killed connection process for {peer_address}")
        return True

    def send_heartbeat(
        self,
        peer_address: str,
        public_host: str,
        miner_info_dict: dict,
    ) -> bool:
        """Send a heartbeat command to a specific peer's process."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "heartbeat",
            "public_host": public_host,
            "miner_info": miner_info_dict,
        })

    def broadcast_heartbeat(
        self, public_host: str, miner_info_dict: dict
    ) -> int:
        """Send heartbeat to all connected peers.

        Returns the number of peers the command was sent to.
        """
        cmd = {
            "cmd": "heartbeat",
            "public_host": public_host,
            "miner_info": miner_info_dict,
        }
        sent = 0
        for handle in self._handles.values():
            if handle.send_cmd(cmd):
                sent += 1
        return sent

    def send_gossip(
        self, peer_address: str, message_bytes: bytes
    ) -> bool:
        """Send a gossip message to a specific peer's process."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "gossip",
            "payload": base64.b64encode(message_bytes).decode('ascii'),
        })

    def broadcast_gossip(self, message_bytes: bytes) -> int:
        """Send a gossip message to all connected peers.

        Returns the number of peers the command was sent to.
        """
        payload_b64 = base64.b64encode(message_bytes).decode('ascii')
        cmd = {"cmd": "gossip", "payload": payload_b64}
        sent = 0
        for handle in self._handles.values():
            if handle.send_cmd(cmd):
                sent += 1
        return sent

    def request_block(
        self, peer_address: str, block_num: int, request_id: int,
    ) -> bool:
        """Request a specific block from a peer.

        The ``request_id`` lets the caller resolve the matching
        ``block_data``/``error`` event emitted by the child.
        """
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "request_block",
            "request_id": request_id,
            "block_num": block_num,
        })

    def request_block_header(
        self, peer_address: str, block_num: int, request_id: int,
    ) -> bool:
        """Request a specific block's header from a peer.

        Mirrors ``request_block`` but resolves on the
        ``block_header_data`` event with a smaller payload.
        """
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "request_block_header",
            "request_id": request_id,
            "block_num": block_num,
        })

    def request_status(self, peer_address: str, request_id: int) -> bool:
        """Request status from a peer."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "request_status", "request_id": request_id,
        })

    def request_peers(self, peer_address: str, request_id: int) -> bool:
        """Request peer list from a peer."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "request_peers", "request_id": request_id,
        })

    def send_probe_request(
        self,
        prober: str,
        target: str,
        probe_id: str,
        request_id: int,
    ) -> bool:
        """Ask ``prober`` (an already-spawned peer) to probe ``target``.

        SWIM indirect probe. Result comes back as a ``probe_result``
        event tagged with ``request_id``.
        """
        handle = self._handles.get(prober)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "probe_request",
            "request_id": request_id,
            "target": target,
            "probe_id": probe_id,
        })

    def poll_events(self) -> List[Tuple[str, dict]]:
        """Collect all pending events from all child processes.

        Returns a list of ``(peer_address, event_dict)`` tuples.
        Non-blocking: returns immediately with whatever is available.
        """
        events: List[Tuple[str, dict]] = []
        dead_peers: List[str] = []

        for peer, handle in self._handles.items():
            # Check if process is still alive
            if not handle.is_alive():
                dead_peers.append(peer)
                events.append((peer, {
                    "event": "disconnected",
                    "reason": "process died",
                }))
                continue

            # Drain all available messages from this peer
            while True:
                msg = handle.recv()
                if msg is None:
                    break
                events.append((peer, msg))

        # Clean up dead processes
        for peer in dead_peers:
            handle = self._handles.pop(peer, None)
            if handle is not None:
                handle.force_stop(timeout=1.0)

        return events

    def reap_dead(self) -> List[str]:
        """Remove processes that have died. Returns list of removed peers."""
        dead = [
            peer for peer, handle in self._handles.items()
            if not handle.is_alive()
        ]
        for peer in dead:
            handle = self._handles.pop(peer, None)
            if handle is not None:
                self._unregister_reader(handle)
                handle.force_stop(timeout=1.0)
                self.logger.warning(
                    f"Reaped dead connection process for {peer}"
                )
        return dead

    def get_load_metrics(self) -> dict:
        """Return current pool load metrics."""
        return {
            "connection_count": self.connection_count,
            "max_connections": self.config.max_connections,
            "utilization": (
                self.connection_count / self.config.max_connections
                if self.config.max_connections > 0
                else 0.0
            ),
            "peers": self.connected_peers,
        }

    def get_least_active_peers(self, count: int) -> List[str]:
        """Return the *count* peers with the oldest last_activity timestamp.

        Useful for selecting peers to shed during rebalancing.
        """
        ranked = sorted(
            self._handles.items(),
            key=lambda item: item[1].last_activity,
        )
        return [peer for peer, _ in ranked[:count]]

    def shutdown_all(self, timeout: float = 5.0) -> None:
        """Gracefully shut down all connection processes.

        Sync path: used from ``atexit`` and tests. The asyncio sender
        drainer tasks are cancelled when the owning event loop exits;
        callers already on an event loop should prefer
        ``aclose_senders`` first to flush queued shutdown commands.
        """
        # Detach from the event loop before sending shutdown so the
        # loop doesn't fire late-readable callbacks into a closed
        # dispatcher.
        for handle in self._handles.values():
            self._unregister_reader(handle)

        # Send shutdown to all
        for handle in self._handles.values():
            handle.shutdown()

        # Wait for graceful exit
        deadline = time.monotonic() + timeout
        for handle in self._handles.values():
            remaining = max(0.1, deadline - time.monotonic())
            handle.process.join(timeout=remaining)

        # Force-kill any that didn't exit
        for peer, handle in self._handles.items():
            if handle.is_alive():
                self.logger.warning(
                    f"Force-killing connection process for {peer}"
                )
                handle.force_stop(timeout=1.0)

        self._handles.clear()
        self.logger.info("All connection processes shut down")

    async def aclose_senders(self, timeout: float = 2.0) -> None:
        """Flush each handle's outbound sender before process teardown.

        Call this from the coordinator's event loop before the sync
        ``shutdown_all`` so queued ``shutdown`` commands and any final
        RPC replies get a chance to reach their workers.
        """
        if not self._handles:
            return
        await asyncio.gather(
            *(h.aclose(timeout=timeout) for h in self._handles.values()),
            return_exceptions=True,
        )
