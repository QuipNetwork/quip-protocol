"""Tests for ProcessPool and ConnectionProcessHandle."""

import time

import pytest

from shared.connection_process import ConnectionProcessHandle, spawn_connection_process
from shared.process_pool import ProcessPool, ProcessPoolConfig


# Unreachable peer for testing. Both the pool and direct spawns pass a
# 1 s connect_timeout into NodeClient, which caps the handshake wait
# AND clamps QUIC's idle_timeout so aioquic's close-handshake retries
# give up promptly — otherwise each attempt burns ~15 s.
_UNREACHABLE = "127.0.0.1:1"
_FAST_CONNECT_TIMEOUT = 1.0
_CONN_TIMEOUT = 5.0


@pytest.fixture
def pool():
    """Create a process pool with small limits for testing.

    ``spawn_cooldown=0`` disables the respawn throttle so unrelated
    tests (e.g. ``test_after_kill_can_respawn``) stay deterministic.
    Throttle behavior has its own fixture below.
    """
    cfg = ProcessPoolConfig(
        max_connections=3, node_timeout=3.0,
        connect_timeout=_FAST_CONNECT_TIMEOUT,
        spawn_cooldown=0.0,
    )
    p = ProcessPool(config=cfg)
    yield p
    p.shutdown_all(timeout=5.0)


@pytest.fixture
def throttled_pool():
    """Process pool with a real spawn cooldown for throttle tests."""
    cfg = ProcessPoolConfig(
        max_connections=3, node_timeout=3.0,
        connect_timeout=_FAST_CONNECT_TIMEOUT,
        spawn_cooldown=5.0,
    )
    p = ProcessPool(config=cfg)
    yield p
    p.shutdown_all(timeout=5.0)


class TestConnectionProcessHandle:
    """Tests for the per-process handle."""

    def test_spawn_creates_alive_process(self):
        """Spawning creates a live child process."""
        handle = spawn_connection_process(
            _UNREACHABLE, node_timeout=2.0,
            connect_timeout=_FAST_CONNECT_TIMEOUT,
        )
        try:
            assert handle.is_alive()
            assert handle.peer_address == _UNREACHABLE
        finally:
            handle.force_stop()

    def test_shutdown_stops_process(self):
        """Shutdown command causes the child to exit."""
        handle = spawn_connection_process(
            _UNREACHABLE, node_timeout=2.0,
            connect_timeout=_FAST_CONNECT_TIMEOUT,
        )
        handle.shutdown()
        handle.process.join(timeout=10.0)
        assert not handle.is_alive()

    @pytest.mark.timeout(25)
    def test_disconnected_event_on_unreachable(self):
        """Child sends disconnected event when peer is unreachable."""
        handle = spawn_connection_process(
            _UNREACHABLE, node_timeout=2.0,
            connect_timeout=_FAST_CONNECT_TIMEOUT,
        )
        try:
            # Wait for the disconnected event
            deadline = time.monotonic() + _CONN_TIMEOUT
            events = []
            while time.monotonic() < deadline:
                msg = handle.recv()
                if msg is not None:
                    events.append(msg)
                    if msg.get("event") == "disconnected":
                        break
                time.sleep(0.2)

            assert any(
                e.get("event") == "disconnected" for e in events
            ), f"Expected disconnected event, got: {events}"
        finally:
            handle.force_stop()

    def test_force_stop_kills_process(self):
        """force_stop terminates even a stuck process."""
        handle = spawn_connection_process(
            _UNREACHABLE, node_timeout=2.0,
            connect_timeout=_FAST_CONNECT_TIMEOUT,
        )
        assert handle.is_alive()
        handle.force_stop(timeout=3.0)
        assert not handle.is_alive()


class TestProcessPool:
    """Tests for the pool manager."""

    def test_spawn_adds_peer(self, pool):
        """Spawning a peer adds it to the pool."""
        assert pool.spawn(_UNREACHABLE)
        assert pool.has_peer(_UNREACHABLE)
        assert pool.connection_count == 1

    def test_spawn_duplicate_returns_false(self, pool):
        """Spawning the same peer twice returns False."""
        pool.spawn(_UNREACHABLE)
        assert pool.spawn(_UNREACHABLE) is False

    def test_max_connections_enforced(self, pool):
        """Pool rejects connections beyond max_connections."""
        for port in range(1, 4):
            assert pool.spawn(f"127.0.0.1:{port}")
        assert pool.connection_count == 3
        assert pool.is_at_capacity
        assert pool.spawn("127.0.0.1:4") is False

    def test_kill_removes_peer(self, pool):
        """Killing a peer removes it from the pool."""
        pool.spawn(_UNREACHABLE)
        assert pool.kill(_UNREACHABLE)
        assert not pool.has_peer(_UNREACHABLE)
        assert pool.connection_count == 0

    def test_kill_nonexistent_returns_false(self, pool):
        """Killing a peer that doesn't exist returns False."""
        assert pool.kill("nonexistent:1234") is False

    def test_connected_peers(self, pool):
        """connected_peers lists all active peer addresses."""
        pool.spawn("127.0.0.1:1")
        pool.spawn("127.0.0.1:2")
        peers = pool.connected_peers
        assert set(peers) == {"127.0.0.1:1", "127.0.0.1:2"}

    @pytest.mark.timeout(25)
    def test_poll_events_from_unreachable(self, pool):
        """poll_events collects disconnected events from children."""
        pool.spawn(_UNREACHABLE)

        deadline = time.monotonic() + _CONN_TIMEOUT
        all_events = []
        while time.monotonic() < deadline:
            events = pool.poll_events()
            all_events.extend(events)
            if any(e.get("event") == "disconnected" for _, e in all_events):
                break
            time.sleep(0.3)

        disconnected = [
            (p, e) for p, e in all_events
            if e.get("event") == "disconnected"
        ]
        assert len(disconnected) >= 1
        assert disconnected[0][0] == _UNREACHABLE

    def test_reap_dead_removes_terminated(self, pool):
        """reap_dead cleans up processes that have exited."""
        pool.spawn(_UNREACHABLE)
        handle = pool._handles[_UNREACHABLE]
        # Force-kill the child
        handle.process.terminate()
        handle.process.join(timeout=5.0)

        dead = pool.reap_dead()
        assert _UNREACHABLE in dead
        assert not pool.has_peer(_UNREACHABLE)

    def test_get_load_metrics(self, pool):
        """Load metrics reflect current pool state."""
        pool.spawn("127.0.0.1:1")
        metrics = pool.get_load_metrics()
        assert metrics["connection_count"] == 1
        assert metrics["max_connections"] == 3
        assert 0.3 < metrics["utilization"] < 0.4

    def test_shutdown_all(self, pool):
        """shutdown_all cleanly stops all processes."""
        pool.spawn("127.0.0.1:1")
        pool.spawn("127.0.0.1:2")
        assert pool.connection_count == 2
        pool.shutdown_all(timeout=10.0)
        assert pool.connection_count == 0

    def test_get_least_active_peers(self, pool):
        """get_least_active_peers returns oldest-active peers."""
        pool.spawn("127.0.0.1:1")
        time.sleep(0.01)
        pool.spawn("127.0.0.1:2")
        time.sleep(0.01)
        pool.spawn("127.0.0.1:3")

        least = pool.get_least_active_peers(2)
        assert least[0] == "127.0.0.1:1"
        assert least[1] == "127.0.0.1:2"

    def test_after_kill_can_respawn(self, pool):
        """After killing a peer, we can spawn a new connection to it."""
        pool.spawn(_UNREACHABLE)
        pool.kill(_UNREACHABLE)
        assert pool.spawn(_UNREACHABLE) is True


class TestSpawnThrottle:
    """Spawn cooldown guards against tight respawn loops."""

    def test_throttle_blocks_immediate_respawn(self, throttled_pool):
        """spawn() within cooldown after kill() returns False."""
        assert throttled_pool.spawn(_UNREACHABLE) is True
        assert throttled_pool.kill(_UNREACHABLE) is True
        # Immediate respawn should be refused.
        assert throttled_pool.spawn(_UNREACHABLE) is False

    def test_throttle_allows_spawn_for_different_peer(self, throttled_pool):
        """Throttle is per-peer; other peers can still spawn."""
        throttled_pool.spawn(_UNREACHABLE)
        throttled_pool.kill(_UNREACHABLE)
        # Different address bypasses the cooldown.
        assert throttled_pool.spawn("127.0.0.1:2") is True


class TestRequestIdCorrelation:
    """Request-ID threading on RPC-style child commands."""

    def test_request_block_sends_request_id(self, pool):
        """request_block hands the given id to the child's cmd."""
        pool.spawn(_UNREACHABLE)
        # Hand off the command; child won't reply (peer unreachable),
        # but the send_cmd round-trip succeeds.
        assert pool.request_block(_UNREACHABLE, 42, request_id=1001) is True

    def test_request_status_sends_request_id(self, pool):
        pool.spawn(_UNREACHABLE)
        assert pool.request_status(_UNREACHABLE, request_id=1002) is True

    def test_request_peers_sends_request_id(self, pool):
        pool.spawn(_UNREACHABLE)
        assert pool.request_peers(_UNREACHABLE, request_id=1003) is True

    def test_send_probe_request_sends_request_id(self, pool):
        pool.spawn(_UNREACHABLE)
        ok = pool.send_probe_request(
            _UNREACHABLE, "127.0.0.1:65535", "probe-xyz",
            request_id=1004,
        )
        assert ok is True

    def test_rpc_methods_fail_without_peer(self, pool):
        """Unknown peer returns False for all RPC methods."""
        assert pool.request_block("9.9.9.9:1", 0, 1) is False
        assert pool.request_status("9.9.9.9:1", 1) is False
        assert pool.request_peers("9.9.9.9:1", 1) is False
        assert pool.send_probe_request("9.9.9.9:1", "x:1", "p", 1) is False
