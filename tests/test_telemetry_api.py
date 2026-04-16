"""Tests for telemetry REST API endpoints in shared.rest_api."""

import asyncio
import json
import os

import pytest
from aiohttp.test_utils import AioHTTPTestCase
from unittest.mock import MagicMock

from shared.rest_api import RestApiServer
from shared.telemetry_cache import TelemetryCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


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
    epoch = "1775167182"
    for i in range(1, 4):
        _write_json(
            os.path.join(tdir, epoch, f"{i}.json"),
            _make_block_json(i, int(epoch)),
        )
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
        assert body["data"]["latest_epoch"] == "1775167182"

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
            "GET", "/api/v1/telemetry/epochs/1775167182/blocks/1",
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["data"]["block_index"] == 1

    async def test_block_not_found(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/1775167182/blocks/99",
        )
        assert resp.status == 404

    async def test_block_invalid_epoch(self):
        resp = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/9999999/blocks/1",
        )
        assert resp.status == 404

    async def test_block_etag_304(self):
        resp1 = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/1775167182/blocks/1",
        )
        etag = resp1.headers.get("ETag")
        resp2 = await self.client.request(
            "GET", "/api/v1/telemetry/epochs/1775167182/blocks/1",
            headers={"If-None-Match": etag},
        )
        assert resp2.status == 304

    async def test_latest(self):
        resp = await self.client.request("GET", "/api/v1/telemetry/latest")
        assert resp.status == 200
        body = await resp.json()
        assert body["data"]["block_index"] == 3
        assert body["data"]["epoch"] == "1775167182"

    async def test_non_telemetry_unaffected(self):
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200


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
