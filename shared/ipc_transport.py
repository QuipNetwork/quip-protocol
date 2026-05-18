"""ZMQ-based IPC transport for inter-process communication.

Provides ROUTER/DEALER sockets for parent-child communication with
automatic identity routing. Falls back to multiprocessing pipes when
pyzmq is not installed.

Socket topology::

    Parent (ROUTER)
    ├── DEALER (connection process 1, identity="peer1:20049")
    ├── DEALER (connection process 2, identity="peer2:20049")
    └── DEALER (miner service 1, identity="miner-cpu-1")

Usage::

    # Parent side
    router = IPCRouter("ipc:///tmp/quip-router.sock")
    await router.start()
    await router.send_to(b"peer1:20049", msg_bytes)
    identity, data = await router.recv()

    # Child side
    dealer = IPCDealer("ipc:///tmp/quip-router.sock", b"peer1:20049")
    await dealer.start()
    await dealer.send(msg_bytes)
    data = await dealer.recv()
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check if ZMQ is available
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


def get_default_ipc_address(name: str = "quip-router") -> str:
    """Generate a unique IPC address for this process."""
    return f"ipc://{tempfile.gettempdir()}/quip-{name}-{os.getpid()}.sock"


class IPCRouter:
    """Parent-side ROUTER socket for dispatching to child processes.

    Each child connects with a unique identity. The ROUTER socket
    routes messages to specific children by identity.

    Args:
        bind_address: ZMQ IPC address to bind to.
        logger: Logger instance.
    """

    def __init__(
        self,
        bind_address: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not ZMQ_AVAILABLE:
            raise RuntimeError(
                "pyzmq is required for ZMQ IPC transport. "
                "Install with: pip install pyzmq"
            )
        self.bind_address = bind_address or get_default_ipc_address()
        self.logger = logger or logging.getLogger(__name__)
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None

    async def start(self) -> None:
        """Bind the ROUTER socket."""
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.bind(self.bind_address)
        # Restrict the IPC socket file to the current user. Without
        # this, any local user on the host can connect to the ROUTER
        # and impersonate a miner identity on multi-tenant systems.
        if self.bind_address.startswith("ipc://"):
            sock_path = self.bind_address[6:]
            try:
                os.chmod(sock_path, 0o600)
            except OSError as exc:
                self.logger.warning(
                    "Failed to restrict permissions on IPC socket %s: %s",
                    sock_path, exc,
                )
        self.logger.info(f"IPC ROUTER bound to {self.bind_address}")

    async def send_to(self, identity: bytes, data: bytes) -> bool:
        """Send data to a specific child identified by identity.

        Returns False if the identity is not connected.
        """
        if self._socket is None:
            return False
        try:
            await self._socket.send_multipart([identity, b'', data])
            return True
        except zmq.ZMQError as exc:
            if exc.errno == zmq.EHOSTUNREACH:
                self.logger.debug(f"IPC: identity {identity!r} not connected")
                return False
            raise

    async def recv(self) -> Tuple[bytes, bytes]:
        """Receive a message from any child.

        Returns (identity, data) tuple.
        """
        if self._socket is None:
            raise RuntimeError("Router not started")
        parts = await self._socket.recv_multipart()
        # ROUTER frames: [identity, empty, data]
        identity = parts[0]
        data = parts[-1]
        return identity, data

    async def recv_nowait(self) -> Optional[Tuple[bytes, bytes]]:
        """Non-blocking receive. Returns None if no message available."""
        if self._socket is None:
            return None
        try:
            parts = await self._socket.recv_multipart(flags=zmq.NOBLOCK)
            identity = parts[0]
            data = parts[-1]
            return identity, data
        except zmq.Again:
            return None

    async def broadcast(self, data: bytes, identities: list[bytes]) -> int:
        """Send data to multiple children. Returns count of successful sends."""
        sent = 0
        for identity in identities:
            if await self.send_to(identity, data):
                sent += 1
        return sent

    def has_poll_fd(self) -> bool:
        """Whether the socket has a pollable file descriptor."""
        return self._socket is not None

    async def stop(self) -> None:
        """Close the ROUTER socket and context."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        # Clean up IPC socket file
        if self.bind_address.startswith("ipc://"):
            sock_path = self.bind_address[6:]
            try:
                os.unlink(sock_path)
            except OSError:
                pass


class IPCDealer:
    """Child-side DEALER socket for communicating with parent ROUTER.

    Args:
        connect_address: ZMQ IPC address of the parent ROUTER.
        identity: Unique identity for this child (e.g., peer address).
        logger: Logger instance.
    """

    def __init__(
        self,
        connect_address: str,
        identity: bytes,
        logger: Optional[logging.Logger] = None,
    ):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("pyzmq required for ZMQ IPC transport")
        self.connect_address = connect_address
        self.identity = identity
        self.logger = logger or logging.getLogger(__name__)
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None

    async def start(self) -> None:
        """Connect the DEALER socket to the parent ROUTER."""
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, self.identity)
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.connect(self.connect_address)
        self.logger.debug(
            f"IPC DEALER connected to {self.connect_address} "
            f"as {self.identity!r}"
        )

    async def send(self, data: bytes) -> None:
        """Send data to the parent ROUTER."""
        if self._socket is None:
            raise RuntimeError("Dealer not started")
        await self._socket.send_multipart([b'', data])

    async def recv(self) -> bytes:
        """Receive data from the parent ROUTER."""
        if self._socket is None:
            raise RuntimeError("Dealer not started")
        parts = await self._socket.recv_multipart()
        return parts[-1]

    async def recv_timeout(self, timeout_ms: int = 500) -> Optional[bytes]:
        """Receive with timeout. Returns None if no message within timeout."""
        if self._socket is None:
            return None
        poller = zmq.asyncio.Poller()
        poller.register(self._socket, zmq.POLLIN)
        events = await poller.poll(timeout=timeout_ms)
        if events:
            parts = await self._socket.recv_multipart()
            return parts[-1]
        return None

    async def stop(self) -> None:
        """Close the DEALER socket and context."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
