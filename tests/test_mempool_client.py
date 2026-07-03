"""Live-chain integration tests for the QuantumComputeMempool surfaces
added in Phase 8a.

Exercises:
  - `register_solver` → `query_solver` round-trip
  - `deregister_solver` → re-`query_solver` returns None
  - `propose_job` (via //Alice as the proposer) → `query_job_order`
    decodes back to the original IsingParams
  - `get_events_at` returns a `JobProposed` entry for the propose call's
    block

Auto-skips when the docker chain at `ws://localhost:9944` isn't reachable.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from shared.keystore_hybrid import generate
from substrate.mempool_types import (
    IsingParams,
    JobMode,
    MinerType,
    OrderStatus,
    ResultDelivery,
    RewardResolution,
)
from substrate.miner_bootstrap import _resolve_dev_signer, _sudo_call
from substrate.client import SubstrateClient


DEFAULT_URL = os.environ.get("QUIP_SUBSTRATE_URL", "ws://localhost:9944")


def _chain_reachable(url: str) -> bool:
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


pytestmark = pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)


async def _funded_test_signer(tmp_path: Path, client: SubstrateClient):
    """Mint a fresh hybrid keystore and seed it with funds from //Alice.

    Phase 7's HybridSigner is the only signing path the chain accepts;
    //Alice resolves to its pinned master seed via DEV_HYBRID_SEEDS.
    """
    keystore = generate(tmp_path / "hybrid_signing.json")
    alice = _resolve_dev_signer("//Alice")
    balance = await client.query_balance(keystore.signer.account_id_bytes())
    if balance < 2_000_000_000_000:
        receipt = await client.submit_extrinsic(
            "Balances",
            "transfer_keep_alive",
            {
                "dest": {"Id": "0x" + keystore.signer.account_id_bytes().hex()},
                "value": 10_000_000_000_000,
            },
            alice,
            wait_for="inblock",
        )
        if receipt.error:
            pytest.fail(f"funding transfer failed: {receipt.error}")
    return keystore


@pytest.mark.timeout(90)
async def test_register_and_deregister_solver(tmp_path):
    """Round-trip solver registration: register → query → deregister → query None."""
    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    try:
        keystore = await _funded_test_signer(tmp_path, client)

        # If we somehow inherit a previous run's registration (the fresh
        # keystore makes this unlikely but not impossible if tmp_path
        # collides), deregister first so the test starts clean.
        existing = await client.query_solver(keystore.signer.account_id_bytes())
        if existing is not None:
            cleanup = await client.submit_extrinsic(
                "QuantumComputeMempool",
                "deregister_solver",
                {},
                keystore.signer,
                wait_for="inblock",
            )
            assert cleanup.error is None, f"cleanup deregister failed: {cleanup.error}"

        receipt = await client.submit_extrinsic(
            "QuantumComputeMempool",
            "register_solver",
            {"solver_type": MinerType.CPU.to_scale_variant()},
            keystore.signer,
            wait_for="inblock",
        )
        assert receipt.error is None, f"register_solver: {receipt.error}"

        info = await client.query_solver(keystore.signer.account_id_bytes())
        assert info is not None
        assert info.solver_type == MinerType.CPU
        assert info.solutions_submitted == 0
        assert info.rewards_earned == 0
        assert info.account == keystore.signer.account_id_bytes()

        deregister = await client.submit_extrinsic(
            "QuantumComputeMempool",
            "deregister_solver",
            {},
            keystore.signer,
            wait_for="inblock",
        )
        assert deregister.error is None, f"deregister_solver: {deregister.error}"

        post = await client.query_solver(keystore.signer.account_id_bytes())
        assert post is None
    finally:
        await client.close()


@pytest.mark.timeout(90)
async def test_propose_job_and_query_back(tmp_path):
    """End-to-end propose: register a job spec via sudo (root-only), have
    //Alice propose an Ising job against it, then query JobOrders[order_id]
    and assert the IsingParams survives the SCALE round-trip byte-equally.
    """
    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    try:
        alice = _resolve_dev_signer("//Alice")

        # Register a fresh JobSpec. `register_job_spec` is ROOT-ONLY (blessed
        # protocol/team templates) with an explicit `builder` first param —
        # sudo dispatch arrives as Root, so the builder account is passed
        # explicitly. A signed call fails BadOrigin (and the receipt's
        # `.error` can be None even then, so it can't be relied on there).
        # Using a unique random name keeps reruns idempotent without us
        # needing to query the spec_id back.
        import os as _os
        name = b"phase8a-test-" + _os.urandom(8).hex().encode()
        register_receipt = await _sudo_call(
            client,
            alice,
            "QuantumComputeMempool",
            "register_job_spec",
            {
                "builder": "0x" + alice.account_id_bytes().hex(),
                "name": (list(name),),  # BoundedVec composite
                "formulation": "Ising",
                "validation_program": None,
                "transform_program": None,
            },
        )

        # Pull the spec_id out of the JobSpecRegistered event in the
        # inclusion block. Phase 8a's hybrid extrinsic path doesn't
        # populate receipt.events (filtering by extrinsic index requires
        # more plumbing — deferred); we use get_events_at instead, which
        # is the path the Phase 8c controller will use anyway.
        register_block = (
            bytes.fromhex(register_receipt.block_hash[2:])
            if register_receipt.block_hash
            and register_receipt.block_hash.startswith("0x")
            else None
        )
        if register_block is None:
            pytest.fail("register_job_spec returned no block_hash")
        events = await client.get_events_at(register_block)
        spec_id = None
        for ev in events:
            if (
                ev["module_id"] == "QuantumComputeMempool"
                and ev["event_id"] == "JobSpecRegistered"
            ):
                attrs = ev["attributes"]
                spec_id_raw = (
                    attrs.get("spec_id") if isinstance(attrs, dict) else None
                )
                if isinstance(spec_id_raw, str):
                    spec_id = bytes.fromhex(
                        spec_id_raw[2:]
                        if spec_id_raw.startswith("0x")
                        else spec_id_raw
                    )
                elif spec_id_raw is not None:
                    spec_id = bytes(spec_id_raw)
                break
        if spec_id is None:
            pytest.fail(f"JobSpecRegistered event not found in {events}")

        ising = IsingParams(
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            h_values=(100, -100, 50, -50),
            j_values=(-200, 300, -100),
            min_energy_milli=None,
            min_diversity_milli=None,
            min_solutions=1,
        )

        # Reward must meet the chain's MinReward; 1 UNIT (1e12 plancks) is
        # the canonical floor on the dev runtime.
        propose_receipt = await client.submit_extrinsic(
            "QuantumComputeMempool",
            "propose_job",
            {
                "spec_id": "0x" + spec_id.hex(),
                "ising_params": ising.to_scale_dict(),
                "reward": 2_000_000_000_000,  # 2 UNIT
                "mode": JobMode.open().to_scale_dict(),
                "resolution": RewardResolution.single_best().to_scale_dict(),
                "deadline_blocks": 50,
                "block_wait": 5,
                "delivery": ResultDelivery.on_chain_only().to_scale_dict(),
            },
            alice,
            wait_for="inblock",
        )
        if propose_receipt.error:
            pytest.fail(f"propose_job: {propose_receipt.error}")

        # Extract order_id from the JobProposed event in the propose
        # call's inclusion block (same get_events_at pattern as above).
        propose_block_hash = (
            bytes.fromhex(propose_receipt.block_hash[2:])
            if propose_receipt.block_hash
            and propose_receipt.block_hash.startswith("0x")
            else None
        )
        if propose_block_hash is None:
            pytest.fail("propose_job returned no block_hash")
        propose_events = await client.get_events_at(propose_block_hash)
        order_id = None
        for ev in propose_events:
            if (
                ev["module_id"] == "QuantumComputeMempool"
                and ev["event_id"] == "JobProposed"
            ):
                attrs = ev["attributes"]
                if isinstance(attrs, dict) and "order_id" in attrs:
                    order_id = int(attrs["order_id"])
                    break
        if order_id is None:
            pytest.fail(f"JobProposed event not found in {propose_events}")

        # Query the order back; the ising params should round-trip exactly.
        order = await client.query_job_order(order_id)
        assert order is not None
        assert order.proposer == alice.account_id_bytes()
        assert order.spec_id == spec_id
        assert order.reward == 2_000_000_000_000
        assert order.status == OrderStatus.OPENED
        assert order.solution_count == 0
        assert order.first_solution_at is None
        assert order.ising_params == ising
        assert order.mode.tag == "Open"
        assert order.resolution.tag == "SingleBest"
        assert order.delivery.tag == "OnChainOnly"
        assert order.timing.deadline_blocks == 50
        assert order.timing.block_wait == 5

        # Sanity-check the get_events_at path matched propose_events
        # above; it's the same call the Phase 8c controller will use.
        assert any(
            e["module_id"] == "QuantumComputeMempool"
            and e["event_id"] == "JobProposed"
            for e in propose_events
        )
    finally:
        await client.close()
