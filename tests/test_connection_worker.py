"""Tests for ConnectionWorkerHandle and connection_worker_main."""

import time

import pytest

from shared.connection_worker import ConnectionWorkerHandle

# Worker passes a short connect_timeout so the underlying NodeClient
# caps both the QUIC handshake wait AND aioquic's close handshake
# (by clamping idle_timeout). Unreachable peers fail in ~1 s instead
# of ~15 s, keeping this file well under its 30 s per-test timeout.
_FAST_CONNECT_TIMEOUT = 1.0
_UNREACHABLE_PEER_TIMEOUT = 5.0


@pytest.fixture
def worker():
    """Create a connection worker, tear it down after the test."""
    handle = ConnectionWorkerHandle(
        node_timeout=5.0, connect_timeout=_FAST_CONNECT_TIMEOUT,
    )
    handle.start()
    yield handle
    if handle.is_alive():
        handle.close()


def _poll_until(worker, count: int, timeout: float) -> list[dict]:
    """Poll worker results until *count* results or *timeout* expires."""
    deadline = time.monotonic() + timeout
    collected: list[dict] = []
    while len(collected) < count and time.monotonic() < deadline:
        collected.extend(worker.poll_results())
        time.sleep(0.2)
    return collected


def test_worker_start_stop(worker):
    """Worker process starts and can be cleanly shut down."""
    assert worker.is_alive()
    worker.close()
    assert not worker.is_alive()


def test_poll_results_empty(worker):
    """Polling before any requests returns an empty list."""
    assert worker.poll_results() == []


@pytest.mark.timeout(30)
def test_request_to_unreachable_peer(worker):
    """Connection to an unreachable peer returns a failure result."""
    worker.request_connections(
        peers=["127.0.0.1:1"],
        join_data={
            "host": "127.0.0.1:9999",
            "version": "0.0.0",
            "capabilities": ["mining"],
            "info": "{}",
        },
    )
    results = _poll_until(worker, count=1, timeout=_UNREACHABLE_PEER_TIMEOUT)

    assert len(results) == 1
    assert results[0]["peer"] == "127.0.0.1:1"
    assert results[0]["success"] is False
    assert results[0]["peers_map"] is None


def test_worker_crash_recovery():
    """After the worker process dies, is_alive returns False."""
    handle = ConnectionWorkerHandle(node_timeout=5.0)
    handle.start()
    assert handle.is_alive()

    handle._process.terminate()
    handle._process.join(timeout=3.0)

    assert not handle.is_alive()
    handle.close()


@pytest.mark.timeout(15)
def test_concurrent_peers_faster_than_sequential(worker):
    """Multiple unreachable peers are tried concurrently, not sequentially.

    With a 1 s connect cap per peer, sequential would be ~3 s for three
    peers. Concurrent dispatch should complete in roughly one peer's
    timeout plus the polling margin.
    """
    peers = ["127.0.0.1:1", "127.0.0.1:2", "127.0.0.1:3"]
    worker.request_connections(
        peers=peers,
        join_data={
            "host": "127.0.0.1:9999",
            "version": "0.0.0",
            "capabilities": ["mining"],
            "info": "{}",
        },
    )

    t0 = time.perf_counter()
    collected = _poll_until(worker, count=3, timeout=_UNREACHABLE_PEER_TIMEOUT)
    elapsed = time.perf_counter() - t0

    assert len(collected) == 3
    # Sequential would be ~3 * connect_timeout. Concurrent should land
    # near one connect_timeout plus polling margin.
    assert elapsed < 3.0, f"Took {elapsed:.1f}s, expected < 3s for concurrent peers"


def test_close_is_idempotent():
    """Calling close() twice does not raise."""
    handle = ConnectionWorkerHandle(node_timeout=5.0)
    handle.start()
    handle.close()
    handle.close()


@pytest.mark.timeout(30)
def test_initial_peers_bypass_ban(worker):
    """Initial peers are passed to the worker with bypass_ban semantics."""
    worker.request_connections(
        peers=["127.0.0.1:1"],
        join_data={
            "host": "127.0.0.1:9999",
            "version": "0.0.0",
            "capabilities": ["mining"],
            "info": "{}",
        },
        initial_peers={"127.0.0.1:1"},
    )
    results = _poll_until(worker, count=1, timeout=_UNREACHABLE_PEER_TIMEOUT)
    assert len(results) == 1
