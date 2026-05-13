"""Unit + integration tests for `shared.substrate_miner_controller`.

Unit tests cover the submission-error classification logic, stale-result
drop paths, dispatch-tracking races, the fatal-receipt raise path, head
coalescing, subscription-client deadlock guard, topology-pin fail-fast,
unregistered-miner fail-fast, and the two-client teardown ownership.

The integration test at the bottom drives the controller end-to-end against
the docker chain (auto-skipped when the chain isn't reachable).
"""
from __future__ import annotations

import asyncio
import os
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate
from shared.miner_bootstrap import BootstrapConfig, _maybe_seed_chain, _resolve_dev_signer
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.substrate_client import SubstrateClient
from shared.substrate_miner_controller import (
    FATAL_SUBMISSION_ERRORS,
    STALE_SUBMISSION_ERRORS,
    ControllerStats,
    SubmissionOutcome,
    SubstrateMinerController,
    _ResultEnvelope,
    classify_submission,
)
from shared.substrate_types import (
    CANONICAL_H_VALUES,
    ExtrinsicReceipt,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _context(
    block_number: int,
    parent_hash: bytes,
    *,
    topology_hash: bytes = b"\xcd" * 32,
) -> SubstrateMiningContext:
    return SubstrateMiningContext(
        block_number=block_number,
        parent_hash=parent_hash,
        topology_hash=topology_hash,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(1, 0, 0, 0),
        miner_account_bytes=b"\x42" * 32,
        h_values=CANONICAL_H_VALUES,
    )


def _mining_result() -> MiningResult:
    return MiningResult(
        miner_id="test",
        miner_type="CPU",
        nonce=1,
        salt=b"\x00" * 32,
        timestamp=0,
        prev_timestamp=0,
        solutions=[[1, -1, 1, -1]],
        energy=-1.0,
        diversity=0.5,
        num_valid=1,
        mining_time=0,
        node_list=[0, 1, 2, 3],
        edge_list=[(0, 1), (1, 2), (2, 3)],
        variable_order=[0, 1, 2, 3],
    )


def _bare_controller() -> SubstrateMinerController:
    """Controller without calling __init__ — for unit tests that only
    exercise a single method. Sets up the attributes that method needs."""
    c = SubstrateMinerController.__new__(SubstrateMinerController)
    c.client = MagicMock()
    c.signer = MagicMock()
    c.signer.account_id_bytes.return_value = b"\x42" * 32
    c.signer.ss58_address.return_value = "5Test"
    c.on_proof_submitted = None
    c._current_context = None
    c.stats = ControllerStats()
    c.miner_handles = []
    c._dispatched = {}
    c._consecutive_none_snapshots = 0
    c.topology_hash = None
    return c


# ----------------------------------------------------------------------
# Pure classifier tests
# ----------------------------------------------------------------------


def test_classify_success():
    assert classify_submission(ExtrinsicReceipt(extrinsic_hash="0xabc")) is SubmissionOutcome.OK


@pytest.mark.parametrize("error_name", STALE_SUBMISSION_ERRORS)
def test_classify_stale_error_names_substrate_format(error_name):
    """Use the exact format substrate-interface emits — `Module(error=X, ...)`
    — not the bare `pallet.Error` shorthand, to catch real-wire regressions."""
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error=f"Module(error={error_name}, pallet='QuantumPow', index=0)",
    )
    assert classify_submission(receipt) is SubmissionOutcome.STALE


@pytest.mark.parametrize("error_name", FATAL_SUBMISSION_ERRORS)
def test_classify_fatal_error_names_substrate_format(error_name):
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error=f"Module(error={error_name}, pallet='QuantumPow', index=0)",
    )
    assert classify_submission(receipt) is SubmissionOutcome.FATAL


def test_classify_unknown_error_is_fatal():
    """Unknown errors fail-loud rather than mine-forever-against-mystery-state."""
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error="Module(error=SomeNeverSeenError, index=99)",
    )
    assert classify_submission(receipt) is SubmissionOutcome.FATAL


def test_classify_outcome_compares_equal_to_string_literal():
    """SubmissionOutcome inherits from str so callers comparing to bare
    string literals keep working. Pins that contract."""
    outcome = classify_submission(ExtrinsicReceipt(extrinsic_hash="0xabc"))
    assert outcome == "ok"
    assert outcome is SubmissionOutcome.OK


# ----------------------------------------------------------------------
# Stale result drop
# ----------------------------------------------------------------------


async def test_handle_result_drops_stale_envelope_block_number():
    """A result whose context.block_number != current_context.block_number
    should be dropped without calling submit_proof."""
    controller = _bare_controller()
    controller._current_context = _context(100, b"\xaa" * 32)

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(99, b"\xbb" * 32),  # stale: different block_number
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1
    assert controller.stats.proofs_submitted == 0


async def test_handle_result_drops_stale_envelope_parent_hash_only():
    """A result whose context.parent_hash differs but block_number matches
    (forked chain scenario) should still be dropped."""
    controller = _bare_controller()
    controller._current_context = _context(100, b"\xaa" * 32)

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(100, b"\xff" * 32),  # same block, different parent
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1


async def test_handle_result_drops_stale_envelope_topology_hash():
    """A result whose topology_hash differs should be dropped — a
    governance rotation within the block window must not pass through."""
    controller = _bare_controller()
    controller._current_context = _context(100, b"\xaa" * 32, topology_hash=b"\x11" * 32)

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(100, b"\xaa" * 32, topology_hash=b"\x22" * 32),
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1


async def test_handle_result_drops_when_current_context_none():
    """A result arriving before the first head dispatch lands has nothing
    to compare against — must still be dropped, not submitted."""
    controller = _bare_controller()
    controller._current_context = None

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(100, b"\xaa" * 32),
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1
    assert controller.stats.proofs_submitted == 0


# ----------------------------------------------------------------------
# Fatal _handle_result raise path
# ----------------------------------------------------------------------


async def test_handle_result_raises_on_fatal_receipt(monkeypatch):
    """When `submit_proof` returns a receipt the classifier deems fatal
    (BadSignature / MinerNotRegistered / BadProof), `_handle_result` must
    raise RuntimeError so the controller exits — silently swallowing a
    fatal leaves the miner mining forever against a chain it can't
    submit to."""
    controller = _bare_controller()
    ctx = _context(100, b"\xaa" * 32)
    controller._current_context = ctx

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=BadSignature, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "shared.substrate_miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="test-0")
    with pytest.raises(RuntimeError, match="submit_proof failed fatally"):
        await controller._handle_result(envelope)
    assert controller.stats.submission_errors == 1
    assert controller.stats.last_submission_error is not None
    assert "BadSignature" in controller.stats.last_submission_error


async def test_handle_result_records_rpc_error_text(monkeypatch):
    """When `submit_proof` raises an RPC exception, the controller should
    log+count it AND stash the error text on stats for telemetry."""
    controller = _bare_controller()
    ctx = _context(100, b"\xaa" * 32)
    controller._current_context = ctx

    async def fake_submit_proof(*args, **kwargs):
        raise ConnectionError("websocket disconnected")

    monkeypatch.setattr(
        "shared.substrate_miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="test-0")
    await controller._handle_result(envelope)  # no raise — RPC errors are dropped
    assert controller.stats.submission_errors == 1
    assert controller.stats.proofs_submitted == 0
    assert "websocket disconnected" in (controller.stats.last_submission_error or "")


# ----------------------------------------------------------------------
# Per-handle dispatch tracking (P1 review finding)
# ----------------------------------------------------------------------


async def test_dispatch_tracking_pairs_result_with_handle_context(monkeypatch):
    """A late MiningResult from a handle that was dispatched against an
    OLD context must be paired with that old context — not the controller's
    newer `_current_context`. Otherwise the staleness check classifies a
    legitimately-old result as fresh and submits it against the wrong nonce."""
    controller = _bare_controller()
    old_ctx = _context(100, b"\xaa" * 32)
    new_ctx = _context(101, b"\xbb" * 32)
    controller._current_context = new_ctx  # controller moved on
    controller._dispatched["slow-handle"] = old_ctx  # handle still on old

    # Simulate the drainer's behavior directly: build the envelope using
    # the dispatched context (what the controller code does at line ~430).
    dispatched_ctx = controller._dispatched["slow-handle"]
    envelope = _ResultEnvelope(
        result=_mining_result(), context=dispatched_ctx, handle_id="slow-handle"
    )

    submit_calls: list = []

    async def fake_submit_proof(*args, **kwargs):
        submit_calls.append(args)
        return ExtrinsicReceipt(extrinsic_hash="0xabc")

    monkeypatch.setattr(
        "shared.substrate_miner_controller.submit_proof", fake_submit_proof
    )

    await controller._handle_result(envelope)
    # The envelope's context is old_ctx (block 100), but current is new_ctx
    # (block 101) — _handle_result must drop as stale, not submit.
    assert submit_calls == []
    assert controller.stats.stale_drops == 1
    assert controller.stats.proofs_submitted == 0


# ----------------------------------------------------------------------
# Encoder errors are not RPC errors
# ----------------------------------------------------------------------


async def test_handle_result_raises_on_encoder_value_error(monkeypatch):
    """An encoder ValueError (no solutions, bad salt length) is a code bug,
    not a transient RPC blip — must NOT be silently swallowed by the
    submit_proof except-Exception catch."""
    controller = _bare_controller()
    ctx = _context(100, b"\xaa" * 32)
    controller._current_context = ctx

    def fake_encode(result, context):
        raise ValueError("MiningResult has no solutions to submit")

    monkeypatch.setattr(
        "shared.substrate_miner_controller.encode_quantum_proof", fake_encode
    )

    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="test-0")
    with pytest.raises(RuntimeError, match="proof encoding failed"):
        await controller._handle_result(envelope)


# ----------------------------------------------------------------------
# Constructor invariants
# ----------------------------------------------------------------------


def test_init_rejects_same_client_for_subscription():
    """Passing the same SubstrateClient for both submission and subscription
    deadlocks substrate-interface (websocket held in receive mode by the
    subscribe loop). The constructor must reject this foot-gun."""
    client = MagicMock(spec=SubstrateClient)
    signer = MagicMock()
    handle = MagicMock()
    handle.miner_id = "test-0"
    with pytest.raises(ValueError, match="separate SubstrateClient"):
        SubstrateMinerController(
            client=client,
            signer=signer,
            miner_handles=[handle],
            subscription_client=client,  # the foot-gun
        )


def test_init_rejects_empty_miner_handles():
    client = MagicMock(spec=SubstrateClient)
    signer = MagicMock()
    with pytest.raises(ValueError, match="at least one MinerHandle"):
        SubstrateMinerController(
            client=client,
            signer=signer,
            miner_handles=[],
        )


# ----------------------------------------------------------------------
# Startup checks
# ----------------------------------------------------------------------


async def test_verify_registered_fails_when_miner_missing():
    """`_verify_registered` must raise RuntimeError when the signer account
    isn't in `QuantumPow.Miners`. Without this fail-fast, the controller
    would silently dispatch work whose proofs the chain will reject."""
    controller = _bare_controller()
    controller.client.query_miner = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="not in QuantumPow.Miners"):
        await controller._verify_registered(b"\x42" * 32)


# ----------------------------------------------------------------------
# Head coalescing + None-snapshot handling
# ----------------------------------------------------------------------


async def test_handle_head_topology_hash_mismatch_raises():
    """When the operator pins --topology-hash and the snapshot reports a
    different one, the controller must fail-fast rather than mine against
    the wrong topology."""
    controller = _bare_controller()
    controller.topology_hash = b"\xaa" * 32  # pinned
    controller.client.get_mining_snapshot = AsyncMock(
        return_value=_context(100, b"\xff" * 32, topology_hash=b"\xbb" * 32)
    )
    controller.signer.account_id_bytes = MagicMock(return_value=b"\x42" * 32)

    with pytest.raises(RuntimeError, match="topology-hash does not match"):
        await controller._handle_head(b"\xff" * 32, 100)


async def test_handle_head_none_snapshot_tracks_consecutive_count():
    """`None` snapshots increment a counter and escalate to RuntimeError
    after _NONE_SNAPSHOT_FAIL_THRESHOLD — otherwise an RPC-broken chain
    looks identical to "no work right now"."""
    from shared.substrate_miner_controller import _NONE_SNAPSHOT_FAIL_THRESHOLD

    controller = _bare_controller()
    controller.client.get_mining_snapshot = AsyncMock(return_value=None)
    controller.signer.account_id_bytes = MagicMock(return_value=b"\x42" * 32)

    # First N-1 Nones: increment stat, log warning, return without raising.
    for i in range(_NONE_SNAPSHOT_FAIL_THRESHOLD - 1):
        await controller._handle_head(b"\xff" * 32, i)
    assert controller.stats.none_snapshots_seen == _NONE_SNAPSHOT_FAIL_THRESHOLD - 1
    assert controller._consecutive_none_snapshots == _NONE_SNAPSHOT_FAIL_THRESHOLD - 1

    # Nth: raises RuntimeError.
    with pytest.raises(RuntimeError, match="chain may be stuck or RPC is broken"):
        await controller._handle_head(b"\xff" * 32, _NONE_SNAPSHOT_FAIL_THRESHOLD)


async def test_handle_head_resets_consecutive_none_on_success():
    """A successful snapshot resets the consecutive-None counter so a
    transient blip doesn't accumulate forever."""
    controller = _bare_controller()
    controller._consecutive_none_snapshots = 3  # simulate prior blips
    controller.client.get_mining_snapshot = AsyncMock(
        return_value=_context(100, b"\xff" * 32)
    )
    controller.signer.account_id_bytes = MagicMock(return_value=b"\x42" * 32)
    controller.miner_handles = []  # no dispatch needed for this path

    await controller._handle_head(b"\xff" * 32, 100)
    assert controller._consecutive_none_snapshots == 0


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


def _chain_requires_hybrid_signer(url: str) -> bool:
    """Detect whether the chain's extrinsic signature type is hybrid.

    The hybrid scheme (sr25519 + ML-DSA-44) merged into `quip-protocol-rs`
    `main` in mid-2026; chains built from that point reject `MultiSignature`
    extrinsics. Python-side hybrid signing lands in Phase 7 of the v0.1 -> v0.2
    refactor — until then, the end-to-end mining flow is structurally
    blocked. Returns True if the chain's metadata exposes
    `quip_transaction_crypto::HybridTxSignature` as the signature type so
    callers can skip the test cleanly with a clear reason.
    """
    if not _chain_reachable(url):
        return False
    try:
        from substrateinterface import SubstrateInterface
        si = SubstrateInterface(url=url)
        md = si.get_metadata()
        types_list = md.value[1]['V14']['types']['types']
        for t in types_list:
            path = t['type'].get('path') or []
            if 'HybridTxSignature' in path:
                return True
        return False
    except Exception:
        return False


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(180)
async def test_controller_submits_proof_end_to_end(tmp_path):
    """Spin up a controller against the live chain, mine one proof.

    Inlines bootstrap (sudo-seeds Z(9,2) topology + difficulty if missing,
    funds the signer via direct //Alice transfer, registers as miner) so
    the test doesn't depend on a running faucet bot. Builds a CPU miner
    with the matching topology. Self-contained — works against a fresh
    `docker compose down -v && up -d` chain.

    Phase 7 (hybrid sr25519+ML-DSA-44) update: uses `HybridSigner` for
    both the test miner keystore and Alice funding. `//Alice` is resolved
    via `DEV_HYBRID_SEEDS` to its precomputed 32-byte master seed.
    """
    keystore_path = tmp_path / "hybrid_signing.json"
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
        # On the hybrid chain, //Alice resolves to the HybridSigner derived
        # from the precomputed DEV_HYBRID_SEEDS entry, not the sr25519 URI
        # derivation — see shared.miner_bootstrap._resolve_dev_signer.
        alice = _resolve_dev_signer("//Alice")
        balance = await setup_client.query_balance(keystore.signer.account_id_bytes())
        if balance < 2_000_000_000_000:
            await setup_client.submit_extrinsic(
                "Balances",
                "transfer_keep_alive",
                {
                    "dest": {"Id": "0x" + keystore.signer.account_id_bytes().hex()},
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
