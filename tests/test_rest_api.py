"""Tests for REST API server."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from shared.rest_api import RestApiServer


class MockNetworkNode:
    """Mock NetworkNode for testing REST API."""

    def __init__(self):
        self.running = True
        self.public_host = "127.0.0.1:20049"
        self.peers = {}
        self.heartbeats = {}
        self.net_lock = asyncio.Lock()
        self._stats_cache = {"test": "stats"}
        self._stats_cache_lock = asyncio.Lock()
        self.miner_handles = []
        self.block_processing_queue = asyncio.Queue()
        self.gossip_processing_queue = asyncio.Queue()
        self.pending_transactions = []
        self.transactions_lock = asyncio.Lock()

    def info(self):
        """Return mock MinerInfo."""
        mock_info = MagicMock()
        mock_info.to_json.return_value = json.dumps({
            "miner_id": "test_miner",
            "version": "0.0.1",
            "public_key_hex": "abcd1234"
        })
        mock_info.version = "0.0.1"
        return mock_info

    def get_latest_block(self):
        """Return mock latest block."""
        return self._create_mock_block(10)

    def get_block(self, index):
        """Return mock block by index."""
        if 1 <= index <= 10:
            return self._create_mock_block(index)
        return None

    def _create_mock_block(self, index):
        """Create a mock block matching real Block/BlockHeader schema."""
        mock_block = MagicMock()
        mock_block.header = MagicMock()
        mock_block.header.index = index
        mock_block.header.timestamp = 1700000000 + index
        mock_block.header.previous_hash = b'\x00' * 32
        mock_block.header.data_hash = b'\x02' * 32
        mock_block.header.version = 2
        mock_block.miner_info = MagicMock()
        mock_block.miner_info.to_json.return_value = json.dumps({
            "miner_id": "test_miner",
            "miner_type": "CPU",
            "reward_address": "aa" * 32,
            "ecdsa_public_key": "bb" * 32,
            "wots_public_key": "cc" * 32,
            "next_wots_public_key": "dd" * 32,
        })
        mock_block.quantum_proof = MagicMock()
        mock_block.quantum_proof.energy = -4200.0
        mock_block.quantum_proof.diversity = 0.15
        mock_block.quantum_proof.num_valid_solutions = 5
        mock_block.quantum_proof.mining_time = 1.5
        mock_block.quantum_proof.nonce = 12345
        mock_block.next_block_requirements = MagicMock()
        mock_block.next_block_requirements.difficulty_energy = -4100.0
        mock_block.next_block_requirements.min_diversity = 0.2
        mock_block.next_block_requirements.min_solutions = 5
        mock_block.transactions = []
        mock_block.hash = b'\x04' * 32
        mock_block.signature = b'\x03' * 64
        return mock_block

    def _track_peer_timestamp(self, timestamp):
        """Track peer timestamp (no-op for mock)."""
        pass

    async def refresh_peer_info(self, sender):
        """Refresh peer info (no-op for mock)."""
        pass

    async def add_peer(self, address, info):
        """Add a peer."""
        self.peers[address] = info


class TestRestApiEndpoints(AioHTTPTestCase):
    """Test REST API endpoints using aiohttp test client."""

    async def get_application(self):
        """Create test application."""
        self.mock_node = MockNetworkNode()
        self.rest_server = RestApiServer(
            network_node=self.mock_node,
            host="127.0.0.1",
            port=8080
        )
        return self.rest_server._create_app()

    async def test_health_endpoint(self):
        """Test /health endpoint."""
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["node_running"] is True

    async def test_status_endpoint(self):
        """Test /api/v1/status endpoint."""
        resp = await self.client.request("GET", "/api/v1/status")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["host"] == "127.0.0.1:20049"
        assert data["data"]["running"] is True

    async def test_stats_endpoint(self):
        """Test /api/v1/stats endpoint."""
        resp = await self.client.request("GET", "/api/v1/stats")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["test"] == "stats"

    async def test_peers_endpoint(self):
        """Test /api/v1/peers endpoint."""
        resp = await self.client.request("GET", "/api/v1/peers")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert "peers" in data["data"]
        assert data["data"]["count"] == 0

    async def test_get_latest_block(self):
        """Test /api/v1/block/latest endpoint."""
        resp = await self.client.request("GET", "/api/v1/block/latest")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["header"]["index"] == 10

    async def test_get_block_by_number(self):
        """Test /api/v1/block/{n} endpoint."""
        resp = await self.client.request("GET", "/api/v1/block/5")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["header"]["index"] == 5

    async def test_get_block_not_found(self):
        """Test /api/v1/block/{n} with non-existent block."""
        resp = await self.client.request("GET", "/api/v1/block/999")
        assert resp.status == 404

        data = await resp.json()
        assert data["success"] is False
        assert data["code"] == "BLOCK_NOT_FOUND"

    async def test_get_block_invalid_number(self):
        """Test /api/v1/block/{n} with invalid block number."""
        resp = await self.client.request("GET", "/api/v1/block/invalid")
        assert resp.status == 400

        data = await resp.json()
        assert data["success"] is False
        assert data["code"] == "INVALID_BLOCK_NUMBER"

    async def test_get_block_header(self):
        """Test /api/v1/block/{n}/header endpoint."""
        resp = await self.client.request("GET", "/api/v1/block/3/header")
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["index"] == 3

    async def test_heartbeat_endpoint(self):
        """Test /api/v1/heartbeat endpoint."""
        resp = await self.client.request(
            "POST",
            "/api/v1/heartbeat",
            json={"sender": "test-peer:8085", "timestamp": 1700000000.0}
        )
        assert resp.status == 200

        data = await resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "ok"

    async def test_heartbeat_invalid_json(self):
        """Test /api/v1/heartbeat with invalid JSON."""
        resp = await self.client.request(
            "POST",
            "/api/v1/heartbeat",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status == 400

        data = await resp.json()
        assert data["success"] is False
        assert data["code"] == "INVALID_JSON"

    async def test_cors_headers(self):
        """Test that CORS headers are present."""
        resp = await self.client.request("GET", "/health")
        assert "Access-Control-Allow-Origin" in resp.headers
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    async def test_options_request(self):
        """Test OPTIONS request for CORS preflight."""
        resp = await self.client.request("OPTIONS", "/api/v1/status")
        assert resp.status == 200
        assert "Access-Control-Allow-Methods" in resp.headers


class TestRestApiResponses:
    """Tests for REST API response formatting."""

    def test_success_response_format(self):
        """Test success response has correct structure."""
        mock_node = MockNetworkNode()
        server = RestApiServer(network_node=mock_node)

        response = server._success_response({"key": "value"})

        assert response.content_type == "application/json"
        body = json.loads(response.body)
        assert body["success"] is True
        assert body["data"]["key"] == "value"
        assert "timestamp" in body

    def test_error_response_format(self):
        """Test error response has correct structure."""
        mock_node = MockNetworkNode()
        server = RestApiServer(network_node=mock_node)

        response = server._error_response("Something went wrong", "TEST_ERROR", 400)

        assert response.status == 400
        body = json.loads(response.body)
        assert body["success"] is False
        assert body["error"] == "Something went wrong"
        assert body["code"] == "TEST_ERROR"
        assert "timestamp" in body


class TestRestApiBlockConversion:
    """Tests for block/header to dict conversion."""

    def test_header_to_dict(self):
        """Test block header conversion to dict."""
        mock_node = MockNetworkNode()
        server = RestApiServer(network_node=mock_node)

        mock_header = MagicMock()
        mock_header.index = 5
        mock_header.timestamp = 1700000005
        mock_header.previous_hash = b'\x00' * 32
        mock_header.data_hash = b'\x02' * 32
        mock_header.version = 2

        result = server._header_to_dict(mock_header)

        assert result["index"] == 5
        assert result["timestamp"] == 1700000005
        assert result["previous_hash"] == "00" * 32
        assert result["data_hash"] == "02" * 32
        assert result["version"] == 2

    def test_block_to_dict(self):
        """Test block conversion to dict."""
        mock_node = MockNetworkNode()
        server = RestApiServer(network_node=mock_node)

        mock_block = mock_node._create_mock_block(7)

        result = server._block_to_dict(mock_block)

        assert result["header"]["index"] == 7
        assert "transactions" in result
        assert "signature" in result
        assert "hash" in result
        assert result["quantum_proof"]["energy"] == -4200.0
        assert result["quantum_proof"]["num_valid_solutions"] == 5
        assert result["miner_info"]["miner_id"] == "test_miner"
        assert result["next_block_requirements"]["difficulty_energy"] == -4100.0
