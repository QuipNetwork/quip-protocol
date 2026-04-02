"""Tests for ConnectionWorkerHandle and connection_worker_main."""

import time

import pytest

from shared.connection_worker import ConnectionWorkerHandle

# QUIC connection cleanup to unreachable peers takes ~15s (aioquic retries
# UDP Initial packets, then wait_connected times out, then __aexit__ runs
# the close handshake). Test deadlines account for this.
_UNREACHABLE_PEER_TIMEOUT = 20.0


@pytest.fixture
def worker():
    """Create a connection worker, tear it down after the test."""
    handle = ConnectionWorkerHandle(node_timeout=5.0)
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


@pytest.mark.timeout(45)
def test_concurrent_peers_faster_than_sequential(worker):
    """Multiple unreachable peers are tried concurrently, not sequentially.

    Each unreachable peer takes ~15s (QUIC handshake timeout + connection
    cleanup). Sequentially that would be ~45s for 3 peers. Concurrently
    it should complete in roughly one peer's timeout.
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
    collected = _poll_until(worker, count=3, timeout=30.0)
    elapsed = time.perf_counter() - t0

    assert len(collected) == 3
    # Sequential would take ~45s. Concurrent should be ~15s + overhead.
    assert elapsed < 25.0, f"Took {elapsed:.1f}s, expected < 25s for concurrent peers"


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
