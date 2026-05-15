"""Unit tests for `shared.telemetry_api.TelemetryApiServer`.

Tests the response envelope shape, route registration, and the simple
handlers that don't need a live chain. Substrate-backed handlers
(`/api/v1/block/*`, `/api/v1/status`) are covered by the live-chain
integration test in `test_substrate_miner_controller.py` end-to-end —
duplicating them here would just mirror that coverage with mocks.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from shared.miner_core import MinerCore
from shared.telemetry_api import TelemetryApiServer


@pytest.fixture
async def server():
    """Build a TelemetryApiServer over a real MinerCore + mocked client/signer.

    `MinerCore` is cheap to construct with an empty miner config; the chain-
    backed dependencies are mocked because we're only exercising routes that
    don't await them.
    """
    core = MinerCore(node_id="telemetry-test", miners_config={})
    client = MagicMock()
    client.get_head = AsyncMock(return_value=b"\xab" * 32)
    client.get_block_number = AsyncMock(return_value=42)
    client.query_miner = AsyncMock(return_value=None)
    signer = MagicMock()
    signer.ss58_address.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    signer.account_id_bytes.return_value = b"\x12" * 32

    s = TelemetryApiServer(
        core=core,
        client=client,
        signer=signer,
        controller=None,
        host="127.0.0.1",
        port=_pick_free_port(),
    )
    await s.start()
    try:
        yield s
    finally:
        await s.stop()
        core.close()


def _pick_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _url(server: TelemetryApiServer, path: str) -> str:
    return f"http://{server.host}:{server.port}{path}"


async def test_health_returns_envelope(server):
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/health")) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["success"] is True
            assert body["data"]["status"] == "ok"
            assert "timestamp" in body


async def test_status_includes_chain_head_and_signer(server):
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/api/v1/status")) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["success"] is True
            data = body["data"]
            # Chain head pulled from the mocked client.
            assert data["chain"]["head_hash"] == "0x" + ("ab" * 32)
            assert data["chain"]["head_number"] == 42
            # Signer identity surfaced under the legacy field name.
            assert data["ss58_address"] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            assert data["miner_registered"] is False
            # is_mining is False because controller is None — we're just
            # standing up the telemetry server in isolation.
            assert data["is_mining"] is False


async def test_system_returns_descriptor(server):
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/api/v1/system")) as resp:
            assert resp.status == 200
            data = (await resp.json())["data"]
            assert data["node_id"] == "telemetry-test"


async def test_stats_returns_minercore_aggregate(server):
    """The `/api/v1/stats` shape must include the legacy fields so dashboards
    that read `total_blocks_attempted` / `win_rate` keep working."""
    server.core.record_dispatch()
    server.core.record_dispatch()
    server.core.record_result(winning_miner_id="t-cpu-1", mining_time=1.5)
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/api/v1/stats")) as resp:
            assert resp.status == 200
            data = (await resp.json())["data"]
            assert data["total_blocks_attempted"] == 2
            assert data["total_blocks_won"] == 1
            assert data["win_rate"] == 0.5
            assert data["wins_per_miner"] == {"t-cpu-1": 1}


async def test_solve_returns_503(server):
    """POST /api/v1/solve is parked in v0.2; the legacy DWave sampler isn't
    wired into the telemetry surface. Verify the error envelope is honest."""
    async with aiohttp.ClientSession() as http:
        async with http.post(
            _url(server, "/api/v1/solve"),
            json={"h": {}, "J": {}, "num_reads": 10},
        ) as resp:
            assert resp.status == 503
            body = await resp.json()
            assert body["success"] is False
            assert body["code"] == "SOLVE_DISABLED"


async def test_unknown_path_returns_404_envelope(server):
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/api/v1/nope")) as resp:
            assert resp.status == 404
            body = await resp.json()
            assert body["success"] is False
            assert body["code"] == "NOT_FOUND"


async def test_legacy_p2p_paths_404(server):
    """The v0.1 P2P paths must not exist in v0.2. Dashboards that still hit
    them get a clean 404 envelope, not silent success."""
    legacy_paths = [
        "/api/v1/peers",
        "/api/v1/join",
        "/api/v1/gossip",
        "/api/v1/heartbeat",
    ]
    async with aiohttp.ClientSession() as http:
        for path in legacy_paths:
            async with http.get(_url(server, path)) as resp:
                assert resp.status == 404, (
                    f"legacy P2P path {path} unexpectedly served"
                )


async def test_block_number_validation(server):
    """Non-integer or negative block numbers come back as INVALID_BLOCK_NUMBER."""
    async with aiohttp.ClientSession() as http:
        async with http.get(_url(server, "/api/v1/block/not-a-number")) as resp:
            assert resp.status == 400
            assert (await resp.json())["code"] == "INVALID_BLOCK_NUMBER"
