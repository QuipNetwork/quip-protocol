"""Process pool for per-connection peer management.

Manages a pool of ``ConnectionProcessHandle`` instances, one per active
peer. Enforces a configurable ``max_connections`` limit, routes commands
to individual peers, and collects events from all child processes.

The parent ``NetworkNode`` interacts exclusively through this pool rather
than holding QUIC connections directly.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shared.connection_process import (
    ConnectionProcessHandle,
    spawn_connection_process,
)


@dataclass
class ProcessPoolConfig:
    """Configuration for the peer connection process pool."""
    max_connections: int = 50
    node_timeout: float = 10.0
    verify_tls: bool = False
    # How long to wait before considering a process dead (no activity)
    process_idle_timeout: float = 600.0


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
    ):
        self.config = config or ProcessPoolConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._handles: Dict[str, ConnectionProcessHandle] = {}

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
            True if spawned successfully, False if at capacity or
            already connected.
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

        handle = spawn_connection_process(
            peer_address=peer_address,
            node_timeout=self.config.node_timeout,
            verify_tls=self.config.verify_tls,
        )
        self._handles[peer_address] = handle
        self.logger.info(
            f"Spawned connection process for {peer_address} "
            f"(pid={handle.process.pid}, {self.connection_count}/"
            f"{self.config.max_connections})"
        )
        return True

    def kill(self, peer_address: str, timeout: float = 3.0) -> bool:
        """Stop and remove a connection process for a peer.

        Returns True if the peer had a connection that was removed.
        """
        handle = self._handles.pop(peer_address, None)
        if handle is None:
            return False

        handle.shutdown()
        handle.force_stop(timeout=timeout)
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

    def request_block(self, peer_address: str, block_num: int) -> bool:
        """Request a specific block from a peer."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({
            "cmd": "request_block",
            "block_num": block_num,
        })

    def request_status(self, peer_address: str) -> bool:
        """Request status from a peer."""
        handle = self._handles.get(peer_address)
        if handle is None:
            return False
        return handle.send_cmd({"cmd": "request_status"})

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
        """Gracefully shut down all connection processes."""
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
