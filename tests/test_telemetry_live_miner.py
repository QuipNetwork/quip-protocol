"""Live-miner integration tests for `shared.telemetry_api.TelemetryApiServer`.

Mocked unit tests for the server live in `test_telemetry_api.py`. This
file exercises the full stack:

    MinerCore + SubstrateMinerController + TelemetryApiServer

against the docker chain, then hits each REST route while the controller
is actively mining and asserts the response reflects live state. Catches
integration regressions the mock-driven tests miss — most notably the
controller → MinerCore stats wiring added in Phase 6.

Auto-skipped when the docker chain at `ws://localhost:9944` isn't
reachable.
"""
from __future__ import annotations

import asyncio
import socket

import aiohttp
import pytest

from dwave_topologies.topologies.zephyr import zephyr
from shared.miner_core import MinerCore
from shared.telemetry_api import TelemetryApiServer
# Reuse the live-chain bootstrap context manager.
from tests.test_substrate_miner_controller import (
    DEFAULT_URL,
    _chain_reachable,
    _live_controller,
)


def _pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(240)
async def test_telemetry_reflects_live_miner_state(tmp_path):
    """End-to-end: a running miner's state shows through the REST API.

    Flow:
      1. Build a MinerCore configured with a CPU handle bound to Z(9,2).
      2. Bring up a SubstrateMinerController against the chain via
         `_live_controller(..., core=core)` — the controller records into
         the same MinerCore the telemetry server reads from.
      3. Start a TelemetryApiServer over the live core/client/signer/controller.
      4. While the controller mines, poll the REST endpoints and assert:
         - `/api/v1/status` returns the chain head + `miner_registered=True`
         - `/api/v1/system` returns the descriptor
         - `/api/v1/stats` reflects `controller.proofs_submitted`,
           `total_blocks_attempted`, and `total_blocks_won` updating
         - `/api/v1/block/latest` returns chain data shaped correctly
    """
    seed_topology_mt = (9, 2)
    miners_config = {
        "cpu": {
            "num_cpus": 1,
            "args": {"topology": zephyr(*seed_topology_mt)},
        }
    }
    core = MinerCore(node_id="telemetry-live-test", miners_config=miners_config)

    try:
        async with _live_controller(
            tmp_path,
            seed_topology_mt=seed_topology_mt,
            core=core,
        ) as (controller, _run_task, _handle, keystore, client):

            server = TelemetryApiServer(
                core=core,
                client=client,
                signer=keystore.signer,
                controller=controller,
                host="127.0.0.1",
                port=_pick_free_port(),
            )
            await server.start()

            # Set the proof callback BEFORE the smoke HTTP probes — on a
            # fast box the first proof can land while we're hitting the
            # status endpoints. If we waited to set the callback, the
            # event would never fire and the test would time out.
            proof_seen = asyncio.Event()

            async def on_proof(receipt, ctx):
                proof_seen.set()

            controller.on_proof_submitted = on_proof
            # If a proof has already landed before we wired the callback
            # (the controller has been running since `_live_controller`
            # yielded), set the event manually.
            if controller.stats.proofs_submitted >= 1:
                proof_seen.set()

            try:
                base = f"http://{server.host}:{server.port}"
                async with aiohttp.ClientSession() as http:
                    # /api/v1/system — descriptor reflects the test MinerCore.
                    async with http.get(f"{base}/api/v1/system") as resp:
                        if resp.status != 200:
                            pytest.fail(f"/api/v1/system: {resp.status} {await resp.text()}")
                        data = (await resp.json())["data"]
                        assert data["node_id"] == "telemetry-live-test"

                    # /api/v1/status — chain head + signer identity.
                    async with http.get(f"{base}/api/v1/status") as resp:
                        if resp.status != 200:
                            pytest.fail(f"/api/v1/status: {resp.status} {await resp.text()}")
                        data = (await resp.json())["data"]
                        assert data["miner_registered"] is True
                        assert data["is_mining"] is True
                        assert data["chain"]["head_number"] >= 1
                        assert data["chain"]["head_hash"].startswith("0x")
                        assert data["ss58_address"] == keystore.signer.ss58_address()

                    # /api/v1/block/latest — chain-backed response.
                    async with http.get(f"{base}/api/v1/block/latest") as resp:
                        if resp.status != 200:
                            pytest.fail(
                                f"/api/v1/block/latest: status={resp.status} "
                                f"body={await resp.text()}"
                            )
                        block_data = (await resp.json())["data"]
                        assert "hash" in block_data
                        # The handler returns the substrate header under
                        # "header"; legacy "number" lives in there.
                        assert "header" in block_data
                        assert "number" in block_data["header"]

                    # Wait for at least one proof acceptance so the stats
                    # endpoint has live numbers to surface. The unit test
                    # in test_telemetry_api.py already exercises the
                    # zero-state response; here we want non-zero. Callback
                    # was wired before the smoke probes so this `wait` is
                    # tight: it returns immediately if a proof already
                    # landed during the smoke loop.
                    try:
                        await asyncio.wait_for(proof_seen.wait(), timeout=150)
                    except asyncio.TimeoutError:
                        pytest.fail(
                            f"no proof in 150s; stats={controller.stats}"
                        )

                    # /api/v1/stats — both MinerCore aggregate fields and
                    # the controller's substrate-specific counters should
                    # be non-zero now that a proof has landed.
                    async with http.get(f"{base}/api/v1/stats") as resp:
                        assert resp.status == 200
                        stats = (await resp.json())["data"]
                        # MinerCore-driven legacy shape (Phase 6 wiring).
                        assert stats["total_blocks_attempted"] >= 1
                        assert stats["total_blocks_won"] >= 1
                        assert stats["win_rate"] > 0.0
                        winning_miner = next(iter(stats["wins_per_miner"]))
                        assert stats["wins_per_miner"][winning_miner] >= 1
                        # Controller-driven new shape.
                        ctl = stats["controller"]
                        assert ctl["proofs_submitted"] >= 1
                        assert ctl["submission_errors"] == 0
                        assert ctl["heads_observed"] >= 1
            finally:
                await server.stop()
    finally:
        core.close()
