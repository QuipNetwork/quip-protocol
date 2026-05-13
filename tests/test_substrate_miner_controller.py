"""Unit + integration tests for `shared.substrate_miner_controller`.

Unit tests cover the submission-error classification logic and the stale
result drop path in isolation. The integration test drives the controller
end-to-end against the docker chain: bootstrap a fresh account, spin up the
controller with a single CPU miner, and assert at least one
`QuantumPow.ProofAccepted` event lands.

The integration test is auto-skipped when the docker chain isn't reachable.
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.substrate_miner_controller import (
    FATAL_SUBMISSION_ERRORS,
    STALE_SUBMISSION_ERRORS,
    SubstrateMinerController,
    classify_submission,
)
from shared.substrate_types import (
    CANONICAL_H_VALUES,
    ExtrinsicReceipt,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


# ----------------------------------------------------------------------
# Pure classifier tests
# ----------------------------------------------------------------------


def test_classify_success():
    assert classify_submission(ExtrinsicReceipt(extrinsic_hash="0xabc")) == "ok"


@pytest.mark.parametrize("error_name", STALE_SUBMISSION_ERRORS)
def test_classify_stale_error_names(error_name):
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error=f"QuantumPow.{error_name}",
    )
    assert classify_submission(receipt) == "stale"


@pytest.mark.parametrize("error_name", FATAL_SUBMISSION_ERRORS)
def test_classify_fatal_error_names(error_name):
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error=f"System.{error_name}",
    )
    assert classify_submission(receipt) == "fatal"


def test_classify_unknown_error_is_fatal():
    """Unknown errors fail-loud rather than mine-forever-against-mystery-state."""
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error="Module(error=SomeNeverSeenError, index=99)",
    )
    assert classify_submission(receipt) == "fatal"


def test_classify_substring_match():
    """The receipt error includes substrate-interface's Module(...) wrapper —
    classifier must match by substring so the bare error name is enough."""
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error="Module(error=InvalidNonce, pallet='QuantumPow', index=42)",
    )
    assert classify_submission(receipt) == "stale"


# ----------------------------------------------------------------------
# Stale result drop
# ----------------------------------------------------------------------


def _context(block_number: int, parent_hash: bytes) -> SubstrateMiningContext:
    return SubstrateMiningContext(
        block_number=block_number,
        parent_hash=parent_hash,
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(1, 0, 0, 0),
        miner_account_bytes=b"\x42" * 32,
        h_values=CANONICAL_H_VALUES,
    )


async def test_handle_result_drops_stale_envelope():
    """A result whose context.block_number != current_context.block_number
    should be dropped without calling submit_proof."""
    from shared.miner_types import MiningResult
    from shared.substrate_miner_controller import _ResultEnvelope

    # Bypass __init__; we only exercise _handle_result.
    controller = SubstrateMinerController.__new__(SubstrateMinerController)
    controller.client = MagicMock()
    controller.signer = MagicMock()
    controller.on_proof_submitted = None
    controller._current_context = _context(100, b"\xaa" * 32)
    from shared.substrate_miner_controller import ControllerStats
    controller.stats = ControllerStats()

    envelope = _ResultEnvelope(
        result=MiningResult(
            miner_id="test", miner_type="CPU",
            nonce=1, salt=b"\x00" * 32, timestamp=0, prev_timestamp=0,
            solutions=[[1, -1, 1, -1]], energy=-1.0, diversity=0.5,
            num_valid=1, mining_time=0,
            node_list=[0, 1, 2, 3], edge_list=[(0, 1), (1, 2), (2, 3)],
            variable_order=[0, 1, 2, 3],
        ),
        context=_context(99, b"\xbb" * 32),  # stale: different block_number
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1
    assert controller.stats.proofs_submitted == 0


# ----------------------------------------------------------------------
# Integration test against live docker chain
# ----------------------------------------------------------------------


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


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(180)
async def test_controller_submits_proof_end_to_end(tmp_path):
    """Spin up a controller against the live chain, mine one proof.

    Inlines bootstrap (sudo-seeds Difficulty + Z(2,2) topology if missing,
    funds the signer via direct //Alice transfer, registers as miner) so
    the test doesn't depend on a running faucet bot. Builds a CPU miner
    with the matching topology. Self-contained — works against a fresh
    `docker compose down -v && up -d` chain.
    """
    from dwave_topologies.topologies.zephyr import zephyr
    from shared.keystore import generate
    from shared.miner_bootstrap import (
        BootstrapConfig,
        _maybe_seed_chain,
    )
    from shared.miner_worker import MinerHandle
    from shared.signer import Sr25519Signer
    from shared.substrate_client import SubstrateClient

    keystore_path = tmp_path / "signing.json"
    keystore = generate(keystore_path)

    # Use Z(9,2) — the legacy chain's default. The genesis-style difficulty
    # threshold (-2500 milli) is calibrated for that GSE range (≈ -4100).
    # Smaller graphs like Z(2,2) need a relaxed difficulty to find solutions
    # at all; we keep this test on the well-calibrated path.
    seed_topology_mt = (9, 2)

    setup_client = SubstrateClient(url=DEFAULT_URL)
    await setup_client.connect()
    try:
        # Sudo-seed difficulty + topology if missing. The helper is idempotent.
        await _maybe_seed_chain(
            setup_client,
            BootstrapConfig(
                node_url=DEFAULT_URL,
                signer_key_path=keystore_path,
                seed_chain=True,
                seed_topology_mt=seed_topology_mt,
            ),
        )

        # Fund the signer from //Alice directly (no faucet bot needed).
        alice = Sr25519Signer.from_uri("//Alice")
        balance = await setup_client.query_balance(keystore.signer.account_id_bytes())
        if balance < 2_000_000_000_000:
            await setup_client.submit_extrinsic(
                "Balances",
                "transfer_keep_alive",
                {
                    "dest": "0x" + keystore.signer.account_id_bytes().hex(),
                    "value": 10_000_000_000_000,
                },
                alice,
                wait_for="inblock",
            )

        # Register the miner.
        if await setup_client.query_miner(keystore.signer.account_id_bytes()) is None:
            receipt = await setup_client.submit_extrinsic(
                "QuantumPow", "register_miner", {}, keystore.signer,
                wait_for="inblock",
            )
            if not receipt.is_success:
                pytest.fail(f"register_miner failed: {receipt.error}")

        head = await setup_client.get_head()
        snap = await setup_client.get_mining_snapshot(
            at=head, miner_account_bytes=keystore.signer.account_id_bytes()
        )
        if snap is None:
            pytest.fail("chain not seeded after sudo-seed step")
        if snap.difficulty.max_energy_milli == 0:
            pytest.fail("chain difficulty is all-zeros after seed")
        chain_topology_hash = snap.topology_hash
    finally:
        await setup_client.close()

    # Build a CPU miner handle whose sampler topology matches the chain's
    # registered topology. The bootstrap-seeded labels come from the same
    # `dwave_networkx.zephyr_graph(m, t)` source the sampler uses, so the
    # labels match byte-for-byte once the SA sampler is constructed with
    # `topology=zephyr(m, t)`.
    spec = {
        "id": "test-controller-cpu",
        "kind": "cpu",
        "args": {"topology": zephyr(*seed_topology_mt)},
    }
    handle = MinerHandle(spec=spec)

    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    controller = SubstrateMinerController(
        client=client,
        signer=keystore.signer,
        miner_handles=[handle],
        topology_hash=chain_topology_hash,
    )

    proof_submitted = asyncio.Event()

    async def on_proof(receipt, ctx):
        proof_submitted.set()

    controller.on_proof_submitted = on_proof

    run_task = asyncio.create_task(controller.run())
    try:
        # Give the controller up to 120s to land a proof. CPU SA on Z(2,2)
        # at the seeded difficulty finishes in 10-30s typically.
        try:
            await asyncio.wait_for(proof_submitted.wait(), timeout=120)
        except asyncio.TimeoutError:
            pytest.fail(
                f"controller did not submit a proof in 120s. stats={controller.stats}"
            )
        # At least one proof submitted and zero fatal errors.
        assert controller.stats.proofs_submitted >= 1
        assert controller.stats.submission_errors == 0
    finally:
        controller.shutdown()
        try:
            await asyncio.wait_for(run_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        await client.close()
        handle.req.put({"op": "shutdown"})
        handle.proc.join(timeout=5)
        if handle.proc.is_alive():
            handle.proc.terminate()
            handle.proc.join(timeout=2)
