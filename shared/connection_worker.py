"""Persistent connection worker process for non-blocking peer discovery.

Isolates QUIC connection establishment and JOIN handshakes into a child
process so the main server_loop is never blocked waiting for slow peers.

IPC protocol
------------
Request queue (parent -> worker):
    {"op": "connect", "peers": [...], "join_data": {...}, "initial_peers": [...]}
    {"op": "shutdown"}

Result queue (worker -> parent):
    {"peer": "host:port", "success": True,  "peers_map": {...},
     "responder_descriptor": {...} | None,
     "peer_versions": {host: version, ...} | None}
    {"peer": "host:port", "success": False, "peers_map": None,
     "responder_descriptor": None, "peer_versions": None}
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import signal
import time
from typing import Optional

from shared.logging_config import QuipFormatter

logger = logging.getLogger(__name__)


def _setup_child_logging() -> logging.Logger:
    """Configure console logging with QuipFormatter for child process."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(QuipFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("connection_worker")


async def _try_join_peer(
    client,
    peer: str,
    join_data: dict,
    bypass_ban: bool,
    result_queue: mp.Queue,
    log: logging.Logger,
) -> None:
    """Attempt JOIN handshake with one peer, put result on queue."""
    try:
        result = await client.join_network_via_peer(
            peer, join_data, bypass_ban=bypass_ban,
        )
        peers_map = result.get("peers", {}) if result else None
        responder_descriptor = result.get("descriptor") if result else None
        peer_versions_map = result.get("peer_versions") if result else None
        result_queue.put({
            "peer": peer,
            "success": result is not None,
            "peers_map": peers_map,
            "responder_descriptor": responder_descriptor,
            "peer_versions": peer_versions_map,
        })
        if result is not None:
            log.info(f"JOIN succeeded with {peer}")
        else:
            log.debug(f"JOIN failed with {peer}")
    except Exception as exc:
        log.debug(f"JOIN error with {peer}: {exc}")
        result_queue.put({
            "peer": peer,
            "success": False,
            "peers_map": None,
            "responder_descriptor": None,
            "peer_versions": None,
        })


def connection_worker_main(
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.synchronize.Event,
    node_timeout: float,
    connect_timeout: Optional[float] = None,
) -> None:
    """Child process entry point: own event loop, own NodeClient."""
    shutdown_flag = False

    def _signal_handler(_signum, _frame):
        nonlocal shutdown_flag
        shutdown_flag = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log = _setup_child_logging()
    log.info("Connection worker started")

    from shared.event_loop import create_event_loop
    loop = create_event_loop()
    asyncio.set_event_loop(loop)

    from shared.node_client import NodeClient
    client = NodeClient(node_timeout=node_timeout, connect_timeout=connect_timeout)

    async def _run() -> None:
        await client.start()
        try:
            while not shutdown_flag and not stop_event.is_set():
                try:
                    msg = request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                op = msg.get("op")
                if op == "shutdown":
                    break

                if op == "connect":
                    peers = msg["peers"]
                    join_data = msg["join_data"]
                    initial_set = set(msg.get("initial_peers", []))

                    tasks = [
                        _try_join_peer(
                            client, peer, join_data,
                            bypass_ban=(peer in initial_set),
                            result_queue=result_queue,
                            log=log,
                        )
                        for peer in peers
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await asyncio.wait_for(client.stop(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    log.info("Connection worker stopped")


class ConnectionWorkerHandle:
    """Parent-side wrapper around a persistent connection worker process."""

    def __init__(self, node_timeout: float = 10.0, connect_timeout: Optional[float] = None):
        self._node_timeout = node_timeout
        self._connect_timeout = connect_timeout
        self._request_queue: mp.Queue = mp.Queue()
        self._result_queue: mp.Queue = mp.Queue()
        self._stop_event: mp.synchronize.Event = mp.Event()
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        """Spawn the worker process."""
        self._process = mp.Process(
            target=connection_worker_main,
            args=(
                self._request_queue,
                self._result_queue,
                self._stop_event,
                self._node_timeout,
                self._connect_timeout,
            ),
            daemon=True,
        )
        self._process.start()
        logger.info("Connection worker process started (pid=%s)", self._process.pid)

    def request_connections(
        self,
        peers: list[str],
        join_data: dict,
        initial_peers: set[str] | None = None,
    ) -> None:
        """Submit a batch of peers to connect to. Non-blocking."""
        self._request_queue.put({
            "op": "connect",
            "peers": peers,
            "join_data": join_data,
            "initial_peers": list(initial_peers or []),
        })

    def poll_results(self) -> list[dict]:
        """Drain all available results. Non-blocking, returns empty list if none."""
        results: list[dict] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self._process is not None and self._process.is_alive()

    def close(self) -> None:
        """Gracefully shut down the worker process."""
        if self._process is None:
            return

        self._stop_event.set()

        # Send explicit shutdown command in case the process is blocked on get()
        try:
            self._request_queue.put_nowait({"op": "shutdown"})
        except Exception:
            pass

        self._process.join(timeout=5.0)
        if self._process.is_alive():
            logger.warning("Connection worker did not exit, terminating")
            self._process.terminate()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()

        # Drain queues to avoid macOS/Linux deadlock on join
        for q in (self._request_queue, self._result_queue):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

        self._process = None
        logger.info("Connection worker closed")
