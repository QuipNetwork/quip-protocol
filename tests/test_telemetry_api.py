"""Tests for telemetry REST API endpoints in shared.rest_api."""

import asyncio
import json
import os
import time

import aiohttp
import pytest
from aiohttp.test_utils import AioHTTPTestCase
from unittest.mock import MagicMock

from shared.rest_api import RestApiServer
from shared.telemetry_cache import TelemetryCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from _utils import write_json as _write_json  # noqa: E402


def _make_block_json(block_index, epoch_ts):
    return {
        "block_index": block_index,
        "block_hash": f"deadbeef{block_index:04d}",
        "timestamp": epoch_ts + block_index * 10,
        "previous_hash": "0" * 64,
        "miner": {"miner_id": "test-node", "miner_type": "CPU"},
        "quantum_proof": {
            "energy": -3950.0, "diversity": 0.22,
            "num_valid_solutions": 5, "mining_time": 5.0,
            "nonce": 12345, "num_nodes": 100, "num_edges": 200,
        },
        "requirements": {
            "difficulty_energy": -4100.0, "min_diversity": 0.15,
            "min_solutions": 5,
        },
    }


def _make_nodes_json():
    return {
        "updated_at": "2026-04-02T23:31:53.610902+00:00",
        "node_count": 1, "active_count": 1,
        "nodes": {
            "127.0.0.1:8085": {
                "address": "127.0.0.1:8085",
                "miner_id": "node-0",
                "status": "active",
            },
        },
    }


def _populate_telemetry(base_dir):
    """Create a telemetry directory with test data."""
    tdir = os.path.join(base_dir, "telemetry")
    os.makedirs(tdir, exist_ok=True)
    _write_json(os.path.join(tdir, "nodes.json"), _make_nodes_json())
    epoch = "abababababababab"
    full_block_1_hash = "ab" * 32
    for i in range(1, 4):
        _write_json(
            os.path.join(tdir, epoch, f"{i}.json"),
            _make_block_json(i, 1775167182),
        )
    _write_json(os.path.join(tdir, "current_epoch.json"), {
        "epoch": epoch,
        "block_1_hash": full_block_1_hash,
        "updated_at": "2026-04-02T23:31:53.610902+00:00",
    })
    return tdir


class MockNetworkNode:
    """Minimal mock for RestApiServer."""

    def __init__(self):
        self.running = True
        self.public_host = "127.0.0.1:20049"
        self.peers = {}
        self.heartbeats = {}
        self.net_lock = asyncio.Lock()
        self._stats_cache = None
        self._stats_cache_lock = asyncio.Lock()
        self.miner_handles = []
        self.block_processing_queue = asyncio.Queue()
        self.gossip_processing_queue = asyncio.Queue()
        self.pending_transactions = []
        self.transactions_lock = asyncio.Lock()

    def info(self):
        mock_info = MagicMock()
        mock_info.to_json.return_value = '{"miner_id":"test"}'
        mock_info.version = "0.0.1"
        return mock_info

    def get_latest_block(self):
        return None

    def get_block(self, index):
        return None


# ---------------------------------------------------------------------------
# Open-access telemetry tests (no auth token)
# ---------------------------------------------------------------------------

class TestTelemetryEndpoints(AioHTTPTestCase):
    """Test telemetry REST endpoints without auth."""

    async def get_application(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        tdir = _populate_telemetry(self._tmpdir)
        self._cache = TelemetryCache(telemetry_dir=tdir, refresh_interval=60)
        await self._cache._refresh()
        self._server = RestApiServer(
            network_node=MockNetworkNode(),
            host="127.0.0.1",
            port=0,
            tls_port=-1,
            telemetry_cache=self._cache,
            telemetry_access_token="",
            telemetry_rate_limit_rpm=600,
        )
        return self._server._create_app()

    async def test_status(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/status")
        assert resp.status == 200
        body = await resp.json()
        assert body["success"] is True
        assert body["data"]["total_blocks"] == 3
        assert body["data"]["latest_epoch"] == "abababababababab"

    async def test_status_etag_304(self):
        resp1 = await self.client.request("GET", "/api/v1/telemetry/status")
        etag = resp1.headers.get("ETag")
        assert etag

        resp2 = await self.client.request(
            "GET", "/api/v1/telemetry/status",
            headers={"If-None-Match": etag},
        )
        assert resp2.status == 304

    async def test_nodes(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/nodes")
        assert resp.status == 200
        body = await resp.json()
        assert body["data"]["node_count"] == 1

    async def test_nodes_etag_304(self):
        resp1 = await self.client.request("GET", "/api/v1/telemetry/nodes")
        etag = resp1.headers.get("ETag")
        resp2 = await self.client.request(
            "GET", "/api/v1/telemetry/nodes",
            headers={"If-None-Match": etag},
        )
        assert resp2.status == 304

    async def test_epochs(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/epochs")
        assert resp.status == 200
        body = await resp.json()
        epochs = body["data"]["epochs"]
        assert len(epochs) == 1
        assert epochs[0]["block_count"] == 3

    async def test_block(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/abababababababab/blocks/1",
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["data"]["block_index"] == 1

    async def test_block_not_found(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/abababababababab/blocks/99",
        )
        assert resp.status == 404

    async def test_block_invalid_epoch(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/9999999/blocks/1",
        )
        assert resp.status == 404

    async def test_block_etag_304(self):
        resp1 = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/abababababababab/blocks/1",
        )
        etag = resp1.headers.get("ETag")
        resp2 = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/abababababababab/blocks/1",
            headers={"If-None-Match": etag},
        )
        assert resp2.status == 304

    async def test_latest(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/latest")
        assert resp.status == 200
        body = await resp.json()
        assert body["data"]["block_index"] == 3
        assert body["data"]["epoch"] == "abababababababab"

    async def test_non_telemetry_unaffected(self):
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200

    async def test_blocks_range_default(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/abababababababab/blocks",
        )
        assert resp.status == 200
        body = await resp.json()
        data = body["data"]
        assert data["epoch"] == "abababababababab"
        assert data["start"] == 1
        assert data["count"] == 3
        assert data["next_start"] is None
        assert data["limit_cap"] == 1000
        indices = [b["block_index"] for b in data["blocks"]]
        assert indices == [1, 2, 3]

    async def test_blocks_range_with_limit(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=1&limit=2",
        )
        assert resp.status == 200
        data = (await resp.json())["data"]
        assert data["count"] == 2
        assert [b["block_index"] for b in data["blocks"]] == [1, 2]
        # limit=2 stops at block 2 but epoch's last is 3 → must paginate
        assert data["next_start"] == 3

    async def test_blocks_range_pagination_end(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=3&limit=1000",
        )
        data = (await resp.json())["data"]
        assert data["count"] == 1
        assert data["next_start"] is None

    async def test_blocks_range_start_clamped_up(self):
        # Epoch's first_block is 1; asking for start=-0 or below should
        # either 400 or clamp up. We chose: negative = 400, zero = 400.
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=0",
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["code"] == "INVALID_RANGE"

    async def test_blocks_range_negative_start(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=-5",
        )
        assert resp.status == 400
        assert (await resp.json())["code"] == "INVALID_RANGE"

    async def test_blocks_range_non_numeric(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=abc",
        )
        assert resp.status == 400
        assert (await resp.json())["code"] == "INVALID_RANGE"

    async def test_blocks_range_limit_capped(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=1&limit=5000",
        )
        assert resp.status == 200
        data = (await resp.json())["data"]
        assert data["limit_cap"] == 1000
        # The test fixture only has 3 blocks, so count is 3 regardless.
        assert data["count"] == 3

    async def test_blocks_range_unknown_epoch(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/9999999/blocks",
        )
        assert resp.status == 404
        assert (await resp.json())["code"] == "EPOCH_NOT_FOUND"

    async def test_blocks_range_past_end(self):
        resp = await self.client.request(
            "GET",
            "/api/v1/telemetry/epochs/abababababababab/blocks?start=999&limit=10",
        )
        assert resp.status == 200
        data = (await resp.json())["data"]
        assert data["count"] == 0
        assert data["blocks"] == []
        assert data["next_start"] is None


# ---------------------------------------------------------------------------
# Auth-protected telemetry tests
# ---------------------------------------------------------------------------

class TestTelemetryAuth(AioHTTPTestCase):
    """Test telemetry auth middleware."""

    async def get_application(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        tdir = _populate_telemetry(self._tmpdir)
        self._cache = TelemetryCache(telemetry_dir=tdir, refresh_interval=60)
        await self._cache._refresh()
        self._server = RestApiServer(
            network_node=MockNetworkNode(),
            host="127.0.0.1",
            port=0,
            tls_port=-1,
            telemetry_cache=self._cache,
            telemetry_access_token="secret-token-123",
            telemetry_rate_limit_rpm=600,
        )
        return self._server._create_app()

    async def test_rejects_no_token(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/status")
        assert resp.status == 401
        body = await resp.json()
        assert body["code"] == "UNAUTHORIZED"

    async def test_rejects_wrong_token(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/status",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status == 401

    async def test_accepts_correct_token(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/status",
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status == 200

    async def test_health_not_affected(self):
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

class TestTelemetryRateLimit(AioHTTPTestCase):
    """Test telemetry rate-limit middleware."""

    async def get_application(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        tdir = _populate_telemetry(self._tmpdir)
        self._cache = TelemetryCache(telemetry_dir=tdir, refresh_interval=60)
        await self._cache._refresh()
        self._server = RestApiServer(
            network_node=MockNetworkNode(),
            host="127.0.0.1",
            port=0,
            tls_port=-1,
            telemetry_cache=self._cache,
            telemetry_access_token="",
            telemetry_rate_limit_rpm=1,  # 1 request/min
        )
        return self._server._create_app()

    async def test_rate_limit_enforced(self):
        # Exhaust the burst allowance
        for _ in range(10):
            await self.client.request("GET", "/api/v1/telemetry/status")

        resp = await self.client.request("GET", "/api/v1/telemetry/status")
        assert resp.status == 429
        body = await resp.json()
        assert body["code"] == "RATE_LIMITED"

    async def test_health_not_rate_limited(self):
        # Exhaust telemetry rate limit
        for _ in range(10):
            await self.client.request("GET", "/api/v1/telemetry/status")

        # Health should still work
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200


# ---------------------------------------------------------------------------
# SSE shutdown tests
# ---------------------------------------------------------------------------

async def test_stop_cancels_sse_streams(tmp_path):
    """stop() must unblock in-flight SSE handlers without waiting 15s.

    The SSE keepalive loop sleeps 15s between pings. Without explicit
    task cancellation, stop() would return but handler tasks would
    linger in the sleep, delaying full shutdown and leaking tasks.
    """
    tdir = _populate_telemetry(str(tmp_path))
    cache = TelemetryCache(telemetry_dir=tdir, refresh_interval=60)
    await cache._refresh()

    server = RestApiServer(
        network_node=MockNetworkNode(),
        host="127.0.0.1",
        port=0,  # random free port
        tls_port=-1,
        telemetry_cache=cache,
        telemetry_access_token="",
        telemetry_rate_limit_rpm=600,
    )
    await server.start()
    try:
        # Resolve the ephemeral port the runner bound to.
        sockets = server._http_runner.addresses
        assert sockets, "HTTP runner has no bound sockets"
        host, port = sockets[0][0], sockets[0][1]
        url = f"http://{host}:{port}/api/v1/telemetry/stream"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200
                # Give the handler a moment to register and enter sleep.
                await asyncio.sleep(0.1)
                assert len(server._sse_clients) == 1

                t0 = time.monotonic()
                await server.stop()
                elapsed = time.monotonic() - t0
    finally:
        # stop() is idempotent; ensure cleanup even on test failure.
        if server._http_runner is not None:
            await server.stop()

    assert elapsed < 2.0, (
        f"stop() took {elapsed:.2f}s; SSE handler was not cancelled"
    )
    assert server._sse_clients == []
