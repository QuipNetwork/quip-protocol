"""Unit + integration tests for `substrate.miner_controller`.

Unit tests cover the submission-error classification logic, stale-result
drop paths, dispatch-tracking races, the fatal-receipt raise path, head
coalescing, subscription-client deadlock guard, topology-pin fail-fast,
unregistered-miner fail-fast, and the two-client teardown ownership.

Integration tests drive the controller end-to-end against the docker chain:

  - `test_controller_submits_proof_end_to_end` — smoke test, one proof
  - `test_controller_long_haul_multi_block` — Phase 6 verification: at least
    three proofs over a longer window, zero fatal submission errors, and
    matching chain-side `MinerInfo.proofs_submitted` counter

Live-chain tests share the `_live_controller` async context manager so the
bootstrap (seed / fund / register) wiring isn't duplicated; the helper is
also re-used from `test_telemetry_live_miner.py` for the live-miner
telemetry assertions.

All integration tests auto-skip when the docker chain isn't reachable.
"""
from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate
from shared.miner_bootstrap import BootstrapConfig, _maybe_seed_chain, _resolve_dev_signer
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from substrate.client import SubstrateClient
from substrate.miner_controller import (
    FATAL_SUBMISSION_ERRORS,
    STALE_SUBMISSION_ERRORS,
    ControllerStats,
    SubmissionOutcome,
    SubstrateMinerController,
    _ResultEnvelope,
    classify_submission,
)
from shared.allowed_value_spec import AllowedValueSet
from substrate.types import (
    ExtrinsicReceipt,
    SubstrateDifficulty,
    SubstrateMiningContext,
)


_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _context(
    last_proof_block_hash: bytes,
    *,
    topology_hash: bytes = b"\xcd" * 32,
) -> SubstrateMiningContext:
    return SubstrateMiningContext(
        last_proof_block_hash=last_proof_block_hash,
        topology_hash=topology_hash,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(1, 0, 0),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x99" * 32,
        block_number=1,
    )


def _mining_result() -> MiningResult:
    return MiningResult(
        miner_id="test",
        miner_type="CPU",
        nonce=(1).to_bytes(32, "big"),
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


def _set_current(controller, ctx) -> None:
    """Helper: set both `_current_context` and `_current_work_key` so the
    staleness check in `_handle_result` finds a baseline to compare against.
    Phase 4 (storm-prevention) split the work-key check out of the
    context-equality check, so tests must now seed both."""
    from substrate.miner_controller import _work_key
    controller._current_context = ctx
    controller._current_work_key = _work_key(ctx)


def _bare_controller() -> SubstrateMinerController:
    """Controller without calling __init__ — for unit tests that only
    exercise a single method. Sets up the attributes that method needs."""
    c = SubstrateMinerController.__new__(SubstrateMinerController)
    # After the pool.get("rpc") removal: parent owns build_client (compose+
    # sign only) and pool_client (swap-aware reads + submit).
    c.build_client = MagicMock()
    c.pool_client = MagicMock()
    # Default get_block_number returns 0; tests override as needed.
    c.pool_client.get_head = AsyncMock(return_value=b"\xff" * 32)
    c.pool_client.get_block_number = AsyncMock(return_value=0)
    c._shutdown_event = asyncio.Event()
    c.signer = MagicMock()
    c.signer.account_id_bytes.return_value = b"\x42" * 32
    c.signer.ss58_address.return_value = "5Test"
    c.on_proof_submitted = None
    c._current_context = None
    c._current_work_key = None
    c.stats = ControllerStats()
    c.miner_handles = []
    c._dispatch_contexts = {}
    from collections import OrderedDict
    c._closed_work_keys = OrderedDict()
    c._highest_handled_block = 0
    c._last_pushed_threshold_milli = 0
    c.topology_hash = None
    c.core = None  # Phase 6: optional MinerCore for telemetry
    # Submission tuning (tip + retry bounds); defaults reproduce pre-tip
    # behavior. _handle_result reads submission_config.tip_plancks.
    from shared.miner_config import SubmissionConfig
    c.submission_config = SubmissionConfig()
    # Submission log is created by __init__; bare controllers used in
    # unit tests either don't exercise the submit path or patch the
    # log explicitly. Tests that touch _handle_result will set this.
    from shared.mining_attempt_log import SubmissionLogger
    import tempfile
    c._submission_log = SubmissionLogger(
        log_dir=Path(tempfile.mkdtemp(prefix="quip-test-")),
    )
    # Anticipatory-submission state (Task 6b).
    c._latest_preview = {}
    c._pow_constants = None
    c._base_difficulty_by_key = {}
    c._anticipatory_fired = set()
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


async def test_handle_result_drops_stale_envelope_last_proof_block_hash():
    """A result whose context.last_proof_block_hash differs from
    current_context.last_proof_block_hash should be dropped without calling
    submit_proof — the round has rolled over and the proof is built
    against the wrong nonce."""
    controller = _bare_controller()
    _set_current(controller, _context(b"\xaa" * 32))

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(b"\xbb" * 32),  # stale: different last proof block hash
        handle_id="test-0",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1
    assert controller.stats.proofs_submitted == 0


async def test_handle_result_drops_stale_envelope_topology_hash():
    """A result whose topology_hash differs should be dropped — a
    governance rotation within the block window must not pass through."""
    controller = _bare_controller()
    _set_current(controller, _context(b"\xaa" * 32, topology_hash=b"\x11" * 32))

    envelope = _ResultEnvelope(
        result=_mining_result(),
        context=_context(b"\xaa" * 32, topology_hash=b"\x22" * 32),
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
        context=_context(b"\xaa" * 32),
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
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=BadSignature, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
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
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    async def fake_submit_proof(*args, **kwargs):
        raise ConnectionError("websocket disconnected")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
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
    old_ctx = _context(b"\xaa" * 32)
    new_ctx = _context(b"\xbb" * 32)
    _set_current(controller, new_ctx)  # controller moved on
    # Old dispatch's immutable context (what the drainer would look up
    # by (handle_id, dispatch_id) and attach to the envelope).
    controller._dispatch_contexts[("slow-handle", 1)] = old_ctx

    # Simulate the drainer's behavior directly: build the envelope using
    # the dispatched context paired by (handle_id, dispatch_id).
    dispatched_ctx = controller._dispatch_contexts[("slow-handle", 1)]
    envelope = _ResultEnvelope(
        result=_mining_result(), context=dispatched_ctx, handle_id="slow-handle"
    )

    submit_calls: list = []

    async def fake_submit_proof(*args, **kwargs):
        submit_calls.append(args)
        return ExtrinsicReceipt(extrinsic_hash="0xabc")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
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
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    def fake_encode(result, context):
        raise ValueError("MiningResult has no solutions to submit")

    monkeypatch.setattr(
        "substrate.miner_controller.encode_quantum_proof", fake_encode
    )

    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="test-0")
    with pytest.raises(RuntimeError, match="proof encoding failed"):
        await controller._handle_result(envelope)


# ----------------------------------------------------------------------
# Constructor invariants
# ----------------------------------------------------------------------


def test_init_rejects_empty_miner_handles():
    """Pool-based __init__ still validates that at least one handle is
    attached. (The old foot-gun check rejecting `subscription_client is
    client` is gone — the pool guarantees distinct slots by role name,
    so the bug it prevented is impossible by construction.)"""
    from substrate.pool import ValidatorPool

    pool = ValidatorPool(urls=["ws://test:9944"])
    signer = MagicMock()
    with pytest.raises(ValueError, match="at least one MinerHandle"):
        SubstrateMinerController(
            pool=pool,
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
    controller.pool_client.query_miner = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="not in QuantumPow.Miners"):
        await controller._verify_registered(b"\x42" * 32)


async def test_mark_work_key_closed_records_block_number(monkeypatch):
    """After a successful submit_proof, the closed-work-key entry must
    carry the receipt's block hash resolved into a block number — so
    `_handle_result` can later classify stale-vs-current heads."""
    from substrate.miner_controller import (
        ClosedWorkRecord,
        _work_key,
    )

    controller = _bare_controller()
    ctx = _context(b"\x10" * 32)
    _set_current(controller, ctx)
    controller.pool_client.get_block_number = AsyncMock(return_value=42)
    # Return the won block number (42 here matches the get_block_number stub).
    controller._verify_proof_recorded = AsyncMock(return_value=42)  # type: ignore[assignment]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            block_hash="0x" + "bb" * 32,
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="h1")
    await controller._handle_result(envelope)
    # Let the post-win refresh task start (and immediately complete on stubs)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    record = controller._closed_work_keys[_work_key(ctx)]
    assert isinstance(record, ClosedWorkRecord)
    assert record.accepted_block_number == 42
    assert record.accepted_block_hash == b"\xbb" * 32


# ----------------------------------------------------------------------
# Post-OK proof verification (Phase 3b)
# ----------------------------------------------------------------------


async def test_verify_proof_recorded_match_returns_won_block_number():
    """On a verified win _verify_proof_recorded returns the won PoW block number."""
    from substrate.types import (
        WinningSolution,
        WinningSolutionWithNonce,
    )
    controller = _bare_controller()
    pubkey = b"\x42" * 32
    nonce = b"\x77" * 32
    controller.signer.account_id_bytes = MagicMock(return_value=pubkey)
    controller.pool_client.query_last_proof_block_number = AsyncMock(return_value=12)
    controller.pool_client.query_winning_solution = AsyncMock(
        return_value=WinningSolutionWithNonce(
            solution=WinningSolution(
                miner=pubkey,
                salt=b"\x00" * 32,
                energy_milli=-4_200_000,
                reward=0,
                submitted_at=12,
                difficulty=SubstrateDifficulty(1, 0, 0),
                last_proof_block_hash=b"\x10" * 32,
            ),
            nonce=nonce,
        )
    )

    ctx = _context(b"\x10" * 32)
    result = _mining_result()
    object.__setattr__(result, "nonce", nonce)  # MiningResult is frozen=False; safe
    envelope = _ResultEnvelope(result=result, context=ctx, handle_id="h1")

    won_block = await controller._verify_proof_recorded(envelope)
    assert won_block == 12, "expected won PoW block number 12"
    assert won_block is not None and won_block >= 0


async def test_verify_proof_recorded_mismatch_returns_negative():
    """On a mismatch (someone else won) _verify_proof_recorded returns -1."""
    from substrate.types import (
        WinningSolution,
        WinningSolutionWithNonce,
    )
    controller = _bare_controller()
    pubkey = b"\x42" * 32
    controller.signer.account_id_bytes = MagicMock(return_value=pubkey)
    controller.pool_client.query_last_proof_block_number = AsyncMock(return_value=12)
    controller.pool_client.query_winning_solution = AsyncMock(
        return_value=WinningSolutionWithNonce(
            solution=WinningSolution(
                miner=b"\x99" * 32,  # someone else won
                salt=b"\x00" * 32,
                energy_milli=-4_200_000,
                reward=0,
                submitted_at=12,
                difficulty=SubstrateDifficulty(1, 0, 0),
                last_proof_block_hash=b"\x10" * 32,
            ),
            nonce=b"\xaa" * 32,
        )
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=_context(b"\x10" * 32), handle_id="h1"
    )
    result = await controller._verify_proof_recorded(envelope)
    assert result is not None and result < 0, "expected negative sentinel for mismatch"
    assert controller.stats.proofs_unverified == 0  # only bumped by caller



async def test_verify_proof_recorded_no_winning_solution_returns_negative():
    """When LastProofBlock is set but winning_solution(N) returns None, the
    chain reported a proof block with no recorded solution — treat as not-won
    (the -1 sentinel), distinct from an inconclusive RPC failure (None)."""
    controller = _bare_controller()
    controller.signer.account_id_bytes = MagicMock(return_value=b"\x42" * 32)
    controller.pool_client.query_last_proof_block_number = AsyncMock(return_value=12)
    controller.pool_client.query_winning_solution = AsyncMock(return_value=None)

    envelope = _ResultEnvelope(
        result=_mining_result(), context=_context(b"\x10" * 32), handle_id="h1"
    )
    result = await controller._verify_proof_recorded(envelope)
    assert result is not None and result < 0, (
        "expected negative sentinel when winning_solution(N) returns None"
    )


async def test_verify_proof_recorded_rpc_failure_returns_none():
    """RPC failure during verification returns None so the caller can
    proceed with close-on-receipt fallback rather than retry-storming."""
    controller = _bare_controller()
    controller.signer.account_id_bytes = MagicMock(return_value=b"\x42" * 32)
    controller.pool_client.query_last_proof_block_number = AsyncMock(
        side_effect=RuntimeError("rpc dead")
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=_context(b"\x10" * 32), handle_id="h1"
    )
    assert await controller._verify_proof_recorded(envelope) is None



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


@asynccontextmanager
async def _live_controller(
    tmp_path: Path,
    *,
    seed_topology_mt: Tuple[int, int] = (9, 2),
    core=None,
) -> AsyncIterator[tuple]:
    """Bring up a SubstrateMinerController against the docker chain.

    Yields `(controller, run_task, handle, keystore, client)`. On exit,
    signals the controller to shut down, joins the run-task, and tears down
    the MinerHandle subprocess.

    The bootstrap path mirrors the production `quip-miner bootstrap`
    workflow: sudo-seed Z(m,t) topology + relaxed difficulty if the chain
    hasn't been seeded yet, fund the test signer from //Alice (resolved via
    `DEV_HYBRID_SEEDS` on hybrid chains), and register the signer as a
    miner. Idempotent — repeated runs against the same chain are fine.

    Pass `core=MinerCore(...)` to wire the controller into a MinerCore so
    legacy `/api/v1/stats` fields (`total_blocks_attempted` /
    `total_blocks_won`) update — required by the live-miner telemetry test.
    """
    keystore_path = tmp_path / "hybrid_signing.json"
    keystore = generate(keystore_path)

    setup_client = SubstrateClient(url=DEFAULT_URL)
    await setup_client.connect()
    try:
        # `force_reseed_difficulty=True` matters across tests: the chain's
        # runtime adjustment tightens min_diversity / max_energy between
        # proofs, so without a reset the long-haul / telemetry tests run
        # into infeasible difficulty after the smoke test mines its block.
        await _maybe_seed_chain(
            setup_client,
            BootstrapConfig(
                validators=(DEFAULT_URL,),
                signer_key_path=keystore_path,
                seed_chain=True,
                seed_topology_mt=seed_topology_mt,
                force_reseed_difficulty=True,
            ),
        )

        # //Alice resolves to the HybridSigner derived from
        # DEV_HYBRID_SEEDS — not substrate-interface URI derivation — on
        # hybrid chains. See shared.miner_bootstrap._resolve_dev_signer.
        alice = _resolve_dev_signer("//Alice")
        balance = await setup_client.query_balance(
            keystore.signer.account_id_bytes()
        )
        if balance < 2_000_000_000_000:
            await setup_client.submit_extrinsic(
                "Balances",
                "transfer_keep_alive",
                {
                    "dest": {
                        "Id": "0x" + keystore.signer.account_id_bytes().hex()
                    },
                    "value": 10_000_000_000_000,
                },
                alice,
                wait_for="inblock",
            )

        if await setup_client.query_miner(
            keystore.signer.account_id_bytes()
        ) is None:
            receipt = await setup_client.submit_extrinsic(
                "QuantumPow", "register_miner", {}, keystore.signer,
                wait_for="inblock",
            )
            if not receipt.is_success:
                pytest.fail(f"register_miner failed: {receipt.error}")

        head = await setup_client.get_head()
        snap = await setup_client.get_mining_snapshot(
            at=head,
            miner_account_bytes=keystore.signer.account_id_bytes(),
        )
        if snap is None:
            pytest.fail("chain not seeded after sudo-seed step")
        if snap.difficulty.max_energy_milli == 0:
            pytest.fail("chain difficulty is all-zeros after seed")
        chain_topology_hash = snap.topology_hash
    finally:
        await setup_client.close()

    # If a MinerCore was provided, reuse its handle so the controller-driven
    # `record_dispatch` / `record_result` calls land on the same instance
    # the telemetry server reads from. Otherwise build a standalone handle.
    if core is not None and core.miner_handles:
        handle = core.miner_handles[0]
    else:
        spec = {
            "id": "test-controller-cpu",
            "kind": "cpu",
            "args": {"topology": zephyr(*seed_topology_mt)},
        }
        handle = MinerHandle(spec=spec)

    from substrate.pool import ValidatorPool
    pool = ValidatorPool(urls=[DEFAULT_URL])
    controller = SubstrateMinerController(
        pool=pool,
        signer=keystore.signer,
        miner_handles=[handle],
        topology_hash=chain_topology_hash,
        core=core,
    )

    # Tests that use the yielded ``client`` (e.g., to query the chain
    # after a submission) need a direct SubstrateClient — the controller
    # now keeps its build_client private and reads/submits go through
    # the swap-aware pool.
    client = SubstrateClient(urls=[DEFAULT_URL])
    await client.connect()
    run_task = asyncio.create_task(controller.run())
    try:
        # Yield after one scheduler tick so controller.run() has reached
        # at least its initial setup.
        await asyncio.sleep(0)
        yield controller, run_task, handle, keystore, client
    finally:
        controller.shutdown()
        try:
            await asyncio.wait_for(run_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            # Surface a real controller-shutdown bug in test logs rather
            # than swallowing it. The test has already yielded its
            # assertions, so we log here and move on rather than masking
            # the original failure with this one.
            import logging
            logging.getLogger(__name__).exception(
                "controller.run() raised during _live_controller teardown"
            )
        await client.close()
        await pool.close()
        # When `core` owns the handle, let `core.close()` tear it down; the
        # caller is responsible for that. Otherwise we built the handle
        # ourselves and own its lifecycle.
        if core is None or handle not in core.miner_handles:
            handle.req.put({"op": "shutdown"})
            handle.proc.join(timeout=5)
            if handle.proc.is_alive():
                handle.proc.terminate()
                handle.proc.join(timeout=2)


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(180)
async def test_controller_submits_proof_end_to_end(tmp_path):
    """Smoke test: spin up a controller against the live chain, mine one proof.

    Self-contained — works against a fresh `docker compose down -v && up -d`
    chain. The full bootstrap (sudo-seed Z(9,2) + funding + registration)
    runs inside `_live_controller`.
    """
    async with _live_controller(tmp_path) as (
        controller, _run_task, _handle, _keystore, _client,
    ):
        proof_submitted = asyncio.Event()

        async def on_proof(receipt, ctx):
            proof_submitted.set()

        controller.on_proof_submitted = on_proof
        try:
            await asyncio.wait_for(proof_submitted.wait(), timeout=120)
        except asyncio.TimeoutError:
            pytest.fail(
                f"controller did not submit a proof in 120s. "
                f"stats={controller.stats}"
            )
        assert controller.stats.proofs_submitted >= 1
        assert controller.stats.submission_errors == 0


@pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)
@pytest.mark.timeout(360)
async def test_controller_long_haul_multi_block(tmp_path):
    """Phase 6 verification: sustain mining across multiple head changes.

    Mines until at least `TARGET_PROOFS` proofs land, then asserts:
      - `proofs_submitted` reached the target
      - zero fatal submission errors accumulated
      - chain-side `MinerInfo.proofs_submitted` reflects the run

    The CPU SA path on Z(9,2) at the relaxed seed difficulty typically lands
    a proof every 10-30s; a 5-minute budget for ≥3 proofs gives comfortable
    headroom even on a slow shared dev machine.
    """
    target_proofs = 3

    async with _live_controller(tmp_path) as (
        controller, _run_task, _handle, keystore, client,
    ):
        proof_event = asyncio.Event()

        async def on_proof(receipt, ctx):
            if controller.stats.proofs_submitted >= target_proofs:
                proof_event.set()

        controller.on_proof_submitted = on_proof
        try:
            await asyncio.wait_for(proof_event.wait(), timeout=300)
        except asyncio.TimeoutError:
            pytest.fail(
                f"only {controller.stats.proofs_submitted}/{target_proofs} "
                f"proofs in 300s. stats={controller.stats}"
            )

        assert controller.stats.proofs_submitted >= target_proofs
        assert controller.stats.submission_errors == 0
        # Phase 5b's stale-result-drop fix: drops should stay near-zero
        # across multiple proofs. Allow a small tolerance for genuine
        # head-change races (a proof completes just as a new head arrives),
        # but a regression that drops every other result would breach this.
        assert controller.stats.stale_drops <= target_proofs
        # Chain-side counter should reflect at least one acceptance.
        # `MinerInfo.proofs_submitted` lags briefly behind the controller's
        # local counter because the chain only writes after extrinsic
        # finalization; we don't strictly require equality, just non-zero.
        info = await client.query_miner(keystore.signer.account_id_bytes())
        assert info is not None
        assert info.proofs_submitted >= 1


# ----------------------------------------------------------------------
# Submission storm / cancel race regressions (see SUBMISSIONSTORM.md)
# ----------------------------------------------------------------------


class _FakeStopEvent:
    """Drop-in for mp.Event in tests — same `is_set/set/clear` surface."""

    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False


class _FakeHandle:
    """Test double for MinerHandle exposing only what the controller
    touches in _handle_result / _cancel_siblings_for_won_work.

    The real MinerHandle owns a subprocess; using a fake keeps these
    regressions fast and deterministic. Mirrors the live API:
    `miner_id`, `stop_event`, `cancel()`, `mine_work_item()` returning
    a dispatch_id, plus the `_active_dispatch_id` field the controller
    reads when deciding which dispatches to drain on a new head.
    """

    def __init__(self, miner_id: str):
        self.miner_id = miner_id
        self.stop_event = _FakeStopEvent()
        self._next_dispatch_id = 0
        self._active_dispatch_id = 0
        self.cancel_calls = 0
        self.mine_calls: list = []

    def cancel(self):
        self.cancel_calls += 1
        self.stop_event.set()

    def mine_work_item(self, context):
        self._next_dispatch_id += 1
        self._active_dispatch_id = self._next_dispatch_id
        self.stop_event.clear()
        self.mine_calls.append((self._active_dispatch_id, context))
        return self._active_dispatch_id


async def test_handle_result_cancels_siblings_on_ok(monkeypatch):
    """Submission storm fix: when one handle's proof is accepted, every
    other handle must be cancelled immediately. Without this, siblings
    keep mining the same context and submitting redundant proofs that
    the chain rejects but our RPC still has to process."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    winner = _FakeHandle("winner")
    sibling_a = _FakeHandle("sibling-a")
    sibling_b = _FakeHandle("sibling-b")
    winner._active_dispatch_id = 1
    sibling_a._active_dispatch_id = 1
    sibling_b._active_dispatch_id = 1
    controller.miner_handles = [winner, sibling_a, sibling_b]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(extrinsic_hash="0xabc")  # OK

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="winner"
    )
    await controller._handle_result(envelope)

    assert controller.stats.proofs_submitted == 1
    assert winner.cancel_calls == 0  # don't cancel the winner
    assert sibling_a.cancel_calls == 1
    assert sibling_b.cancel_calls == 1


async def test_handle_result_drops_duplicate_after_ok(monkeypatch):
    """Second result for the same work key — landing after the first was
    accepted — must be dropped without another submit_proof call. This
    is the belt-and-suspenders behind sibling-cancel for results that
    were already in flight when the first OK landed."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = [_FakeHandle("winner"), _FakeHandle("late")]

    submit_call_count = 0

    async def fake_submit_proof(*args, **kwargs):
        nonlocal submit_call_count
        submit_call_count += 1
        return ExtrinsicReceipt(extrinsic_hash=f"0x{submit_call_count:04x}")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    # First result lands and gets accepted.
    await controller._handle_result(
        _ResultEnvelope(
            result=_mining_result(), context=ctx, handle_id="winner",
        )
    )
    assert submit_call_count == 1
    assert controller.stats.proofs_submitted == 1

    # Second result for the same work key — should be dropped without
    # calling submit_proof a second time.
    await controller._handle_result(
        _ResultEnvelope(
            result=_mining_result(), context=ctx, handle_id="late",
        )
    )
    assert submit_call_count == 1  # unchanged
    assert controller.stats.proofs_submitted == 1
    assert controller.stats.duplicate_result_drops == 1


async def test_handle_result_pairs_late_result_with_dispatch_context(monkeypatch):
    """Cancel race fix: a late result from a cancelled dispatch must be
    associated with the exact context that dispatch was given, not the
    handle's currently-active context. Verified through the immutable
    `_dispatch_contexts[(handle_id, dispatch_id)]` map."""
    controller = _bare_controller()
    old_ctx = _context(b"\xaa" * 32)
    new_ctx = _context(b"\xbb" * 32)
    _set_current(controller, new_ctx)
    controller._dispatch_contexts[("slow-handle", 1)] = old_ctx
    controller._dispatch_contexts[("slow-handle", 2)] = new_ctx

    # Look up by dispatch_id=1 (the cancelled one) — must yield old_ctx.
    looked_up = controller._dispatch_contexts[("slow-handle", 1)]
    assert looked_up is old_ctx
    assert looked_up.last_proof_block_hash == b"\xaa" * 32

    # And the envelope built from that lookup is a stale result the
    # controller drops via the work-key check.
    envelope = _ResultEnvelope(
        result=_mining_result(), context=looked_up, handle_id="slow-handle",
    )
    await controller._handle_result(envelope)
    assert controller.stats.stale_drops == 1


def test_minerhandle_cancel_does_not_enqueue_stop_mining(monkeypatch):
    """The legacy `MinerHandle.cancel()` queued an untagged `stop_mining`
    op that could be consumed by a *later* dispatch's req.get and
    cancel the new work with a stale cancel. The fix is to not queue
    anything — stop_event is the single signal."""
    import multiprocessing as mp
    from shared.miner_worker import MinerHandle

    # Bypass __init__ to avoid spawning a real subprocess.
    handle = MinerHandle.__new__(MinerHandle)
    handle.spec = {"id": "test", "kind": "cpu"}
    handle.req = mp.Queue()
    handle.resp = mp.Queue()
    handle.stop_event = mp.Event()
    handle._next_dispatch_id = 0
    handle._active_dispatch_id = 0

    handle.cancel()
    assert handle.stop_event.is_set()
    # Queue must be empty — no stop_mining op enqueued.
    import queue
    try:
        msg = handle.req.get(timeout=0.05)
        pytest.fail(f"cancel() enqueued unexpected op: {msg}")
    except queue.Empty:
        pass  # expected


def test_minerhandle_mine_work_item_returns_dispatch_id():
    """`mine_work_item` must increment + return the dispatch_id so the
    controller can key its immutable (handle_id, dispatch_id) →
    context map by it."""
    import multiprocessing as mp
    from shared.miner_worker import MinerHandle

    handle = MinerHandle.__new__(MinerHandle)
    handle.spec = {"id": "test", "kind": "cpu"}
    handle.req = mp.Queue()
    handle.resp = mp.Queue()
    handle.stop_event = mp.Event()
    handle._next_dispatch_id = 0
    handle._active_dispatch_id = 0

    fake_ctx = object()
    d1 = handle.mine_work_item(fake_ctx)
    d2 = handle.mine_work_item(fake_ctx)
    d3 = handle.mine_work_item(fake_ctx)
    assert (d1, d2, d3) == (1, 2, 3)
    assert handle._active_dispatch_id == 3

    # And the worker request includes the dispatch_id.
    msg = handle.req.get(timeout=0.5)
    assert msg["op"] == "mine_work_item"
    assert msg["dispatch_id"] == 1


# ----------------------------------------------------------------------
# Chain-derived Sol# (Task 4): chain_block_number / pow_sequence wiring
# ----------------------------------------------------------------------


async def test_handle_result_won_records_chain_block_number(monkeypatch):
    """On a verified win, submission.json must record ``chain_block_number``
    as the won PoW block number returned by ``_verify_proof_recorded`` (i.e.
    ``QuantumPow.LastProofBlock``), not just the receipt-derived block number.
    This is the chain-derived Sol# for won proofs."""
    import json

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = [_FakeHandle("winner")]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            block_hash="0x" + "bb" * 32,
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    # Stub verify to return the won block number (77).
    controller._verify_proof_recorded = AsyncMock(return_value=77)  # type: ignore[assignment]
    # get_block_number used for accepted_block_number fallback.
    controller.pool_client.get_block_number = AsyncMock(return_value=99)

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="winner",
    )
    await controller._handle_result(envelope)

    # Won path: chain_block_number must be the verify-path's block (77), not 99.
    assert controller.stats.proofs_submitted == 1
    sub_path = controller._submission_log.log_dir / str(envelope.dispatch_id) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "submitted_inblock"
    assert record["chain_block_number"] == 77, (
        "chain_block_number on a won proof must be LastProofBlock from the "
        "verify path, not the receipt-derived block number"
    )


async def test_handle_result_not_won_records_pow_sequence(monkeypatch):
    """On a stale (not-won) outcome, submission.json must record ``pow_sequence``
    as the miner's ``QuantumPow.Miners.proofs_submitted`` counter so the
    dashboard can display a stable, chain-derived Sol# even without a won block."""
    import json

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = [_FakeHandle("loser")]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=InvalidNonce, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    # query_proofs_submitted should return the miner's current cumulative count.
    controller.pool_client.query_proofs_submitted = AsyncMock(return_value=55)

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="loser",
    )
    await controller._handle_result(envelope)

    assert controller.stats.stale_drops == 1
    sub_path = controller._submission_log.log_dir / str(envelope.dispatch_id) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "rejected_stale"
    assert record["pow_sequence"] == 55, (
        "pow_sequence must be written for not-won outcomes so the dashboard "
        "can display the miner's cumulative proofs_submitted as Sol#"
    )


async def test_handle_result_not_won_pow_sequence_none_on_rpc_failure(monkeypatch):
    """If ``query_proofs_submitted`` fails (e.g. chain unreachable), the
    submission must still be logged with ``pow_sequence=None`` — never raised."""
    import json

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = [_FakeHandle("loser")]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=InvalidNonce, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    controller.pool_client.query_proofs_submitted = AsyncMock(
        side_effect=ConnectionError("rpc dead")
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="loser",
    )
    # Must not raise.
    await controller._handle_result(envelope)

    sub_path = controller._submission_log.log_dir / str(envelope.dispatch_id) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "rejected_stale"
    assert record["pow_sequence"] is None, (
        "pow_sequence must be None (not absent, not an exception) when the "
        "chain RPC is unavailable at submission time"
    )


# ----------------------------------------------------------------------
# Anticipatory-submission preview store (Task 6a)
# ----------------------------------------------------------------------


def _preview_msg(dispatch_id: int, *, floor: float, miner_id: str = "p0") -> dict:
    """Build a worker `{"op": "preview"}` resp_q message."""
    return {
        "op": "preview",
        "id": miner_id,
        "dispatch_id": dispatch_id,
        "data": {
            "dispatch_id": dispatch_id,
            "nonce": (7).to_bytes(32, "big"),
            "salt": b"\x11" * 32,
            "solutions": [[1, -1, 1, -1]],
            "submit_floor_energy": floor,
            "energy": floor,
            "num_valid": 3,
            "diversity": 0.5,
        },
    }


def test_store_preview_lands_in_store_keyed_by_work_key():
    """A `{"op":"preview"}` message must land in `_latest_preview` keyed by
    the dispatched context's work key — and must NOT produce a result or
    submission."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    controller._latest_preview = {}
    controller._result_queue = asyncio.Queue()
    handle = _FakeHandle("p0")
    ctx = _context(b"\xaa" * 32)
    controller._dispatch_contexts[("p0", 1)] = ctx

    controller._store_preview(handle, _preview_msg(1, floor=-2.0))

    key = _work_key(ctx)
    assert key in controller._latest_preview
    entry = controller._latest_preview[key]
    assert entry["submit_floor_energy"] == -2.0
    assert entry["handle_id"] == "p0"
    assert entry["context"] is ctx
    # No result/submission side effects.
    assert controller._result_queue.empty()
    assert controller.stats.proofs_submitted == 0


def test_store_preview_keeps_lowest_floor_for_work_key():
    """When multiple previews arrive for the same work key, the store keeps
    the lowest (best) submit_floor_energy; a worse floor is ignored."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    controller._latest_preview = {}
    handle = _FakeHandle("p0")
    ctx = _context(b"\xaa" * 32)
    controller._dispatch_contexts[("p0", 1)] = ctx
    key = _work_key(ctx)

    controller._store_preview(handle, _preview_msg(1, floor=-2.0))
    # Worse floor (less negative) → ignored.
    controller._store_preview(handle, _preview_msg(1, floor=-1.0))
    assert controller._latest_preview[key]["submit_floor_energy"] == -2.0
    # Better floor (more negative) → replaces.
    controller._store_preview(handle, _preview_msg(1, floor=-3.5))
    assert controller._latest_preview[key]["submit_floor_energy"] == -3.5


def test_store_preview_dropped_when_no_dispatch_context():
    """A preview for an unknown dispatch_id is dropped (no context to key
    against) — must not raise or populate the store."""
    controller = _bare_controller()
    controller._latest_preview = {}
    handle = _FakeHandle("p0")
    # No _dispatch_contexts entry for dispatch_id=99.
    controller._store_preview(handle, _preview_msg(99, floor=-2.0))
    assert controller._latest_preview == {}


# ----------------------------------------------------------------------
# Anticipatory submission (Task 6b)
# ----------------------------------------------------------------------


def _stub_predictor_inputs(controller, *, last_proof_block: int = 10) -> None:
    """Wire fake chain reads so `_anticipatory_inputs` resolves without RPC.

    The decay math itself is covered by `test_difficulty_decay.py`; these
    tests pin the controller's orchestration, so we feed concrete inputs
    and (separately) stub the predictor's `B*` output.
    """
    from substrate.types import PowConstants

    controller.pool_client.query_pow_constants = AsyncMock(
        return_value=PowConstants(
            epoch_length=5,
            curve_c_easy_milli=800,
            curve_c_knee_milli=750,
            curve_c_hard_milli=700,
        )
    )
    controller.pool_client.query_difficulty = AsyncMock(
        return_value=SubstrateDifficulty(1, -1000, 0)
    )
    controller.pool_client.query_last_proof_block_number = AsyncMock(
        return_value=last_proof_block
    )


def _store_preview_entry(controller, ctx, *, floor: float = -3.0) -> None:
    """Populate `_latest_preview[work_key(ctx)]` directly."""
    from substrate.miner_controller import _work_key

    controller._latest_preview[_work_key(ctx)] = {
        "handle_id": "p0",
        "context": ctx,
        "dispatch_id": 1,
        "nonce": (7).to_bytes(32, "big"),
        "salt": b"\x11" * 32,
        "solutions": [[1, -1, 1, -1]],
        "submit_floor_energy": floor,
        "energy": floor,
        "num_valid": 3,
        "diversity": 0.5,
    }


def _ctx_at_block(last_proof_block_hash: bytes, block_number: int):
    """A context at a specific head block number."""
    return SubstrateMiningContext(
        last_proof_block_hash=last_proof_block_hash,
        topology_hash=b"\xcd" * 32,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
        difficulty=SubstrateDifficulty(1, 0, 0),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x99" * 32,
        block_number=block_number,
    )


async def test_anticipatory_no_fire_before_b_star_minus_one(monkeypatch):
    """With B*=20, a head at block 18 (< B*-1=19) must NOT fire."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 18)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    fired = False

    async def fake_submit_with_retry(*args, **kwargs):
        nonlocal fired
        fired = True
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.SUCCESS)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller._maybe_anticipatory_fire(ctx, key)
    assert fired is False
    assert controller.stats.proofs_submitted == 0


async def test_anticipatory_fires_at_b_star_minus_one_success(monkeypatch):
    """At block 19 (= B*-1 for B*=20), the controller fires and on SUCCESS
    verifies, records, and marks the work key closed."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    controller._verify_proof_recorded = AsyncMock(return_value=42)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    captured = {}

    async def fake_submit_with_retry(build, pool, signer, result, context, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        captured["result"] = result
        captured["context"] = context
        return SubmitResult(
            action=SubmitRetryAction.SUCCESS,
            receipt=ExtrinsicReceipt(extrinsic_hash="0xabc"),
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller._maybe_anticipatory_fire(ctx, key)

    assert controller.stats.proofs_submitted == 1
    assert key in controller._closed_work_keys
    # The fired proof was reconstructed from the preview.
    assert captured["result"].nonce == (7).to_bytes(32, "big")
    assert captured["result"].num_valid == 3
    assert captured["context"] is ctx


async def test_anticipatory_verify_fail_records_chain_error_and_refires(monkeypatch):
    """An anticipatory fire with receipt OK but ``_verify_proof_recorded``
    returning -1 (chain recorded a different proof) must: NOT close the work
    key, clear ``_anticipatory_fired`` (so the next head re-fires), and write
    a ``chain_error`` submission-log row so the failure is visible."""
    import json

    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    controller._verify_proof_recorded = AsyncMock(return_value=-1)
    controller.pool_client.query_proofs_submitted = AsyncMock(return_value=77)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.SUCCESS,
            receipt=ExtrinsicReceipt(extrinsic_hash="0xabc", block_hash="0xdef"),
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller._maybe_anticipatory_fire(ctx, key)

    # Not won: key stays open, mid-fire mark cleared so a later head re-fires,
    # preview retained.
    assert controller.stats.proofs_submitted == 0
    assert controller.stats.proofs_unverified == 1
    assert key not in controller._closed_work_keys
    assert key not in controller._anticipatory_fired
    assert key in controller._latest_preview
    # A chain_error submission-log row was written (dispatch_id=1 from preview).
    sub_path = controller._submission_log.log_dir / "1" / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "chain_error"
    assert record["num_valid"] == 3
    assert record["pow_sequence"] == 77
    assert record["error"] == "receipt OK but proof not recorded by chain"


async def test_anticipatory_retry_keeps_preview_for_next_head(monkeypatch):
    """A RETRY-exhausted fire must keep the preview and clear the mid-fire
    mark so a later head can fire again."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.RETRY, attempts=4)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller._maybe_anticipatory_fire(ctx, key)

    assert controller.stats.proofs_submitted == 0
    assert key not in controller._closed_work_keys
    assert key in controller._latest_preview  # preview retained
    assert key not in controller._anticipatory_fired  # mid-fire mark cleared


async def test_anticipatory_round_stale_discards_preview(monkeypatch):
    """STOP_ROUND_STALE means the nonce is dead — discard the preview and
    any pending anticipatory state."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.STOP_ROUND_STALE)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller._maybe_anticipatory_fire(ctx, key)

    assert controller.stats.proofs_submitted == 0
    assert key not in controller._latest_preview  # discarded
    assert key not in controller._anticipatory_fired
    assert controller.stats.stale_drops == 1


async def test_anticipatory_no_fire_when_b_star_none(monkeypatch):
    """If the candidate never clears within the horizon (B*=None), do
    nothing this head."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)

    fired = False

    async def fake_submit_with_retry(*args, **kwargs):
        nonlocal fired
        fired = True
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.SUCCESS)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: None,
    )

    await controller._maybe_anticipatory_fire(ctx, key)
    assert fired is False


def test_evict_anticipatory_state_prunes_all():
    """Eviction drops the preview, the cached base difficulty, and the
    fired mark for a work key."""
    controller = _bare_controller()
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    _store_preview_entry(controller, ctx)
    controller._base_difficulty_by_key[key] = SubstrateDifficulty(1, -1000, 0)
    controller._anticipatory_fired.add(key)

    controller._evict_anticipatory_state(key)

    assert key not in controller._latest_preview
    assert key not in controller._base_difficulty_by_key
    assert key not in controller._anticipatory_fired


async def test_dedup_worker_result_after_anticipatory_fire(monkeypatch):
    """After an anticipatory SUCCESS for a work key, a worker-returned
    result for the same key must not double-submit."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    # Simulate "anticipatory fire in progress / done" for this key.
    controller._anticipatory_fired.add(key)

    submit_calls = 0

    async def fake_submit_proof(*args, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        return ExtrinsicReceipt(extrinsic_hash="0xabc")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    await controller._handle_result(
        _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="p0")
    )
    assert submit_calls == 0  # worker result was a no-op
    assert controller.stats.duplicate_result_drops == 1


async def test_rollover_evicts_stale_preview_and_skips_fire(monkeypatch):
    """A new head with a different last_proof_block_hash must evict the old
    preview and NOT fire the stale candidate."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    controller.events = None
    controller.miner_handles = [_FakeHandle("p0")]
    controller.core = None
    controller._done_queues = {}
    controller.signer.account_id_bytes.return_value = b"\x42" * 32

    old_ctx = _ctx_at_block(b"\xaa" * 32, 18)
    old_key = _work_key(old_ctx)
    _store_preview_entry(controller, old_ctx)
    controller._base_difficulty_by_key[old_key] = SubstrateDifficulty(1, -1000, 0)
    # Controller is currently on the old key.
    controller._current_context = old_ctx
    controller._current_work_key = old_key

    fired = False

    async def fake_submit_with_retry(*args, **kwargs):
        nonlocal fired
        fired = True
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.SUCCESS)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    # Predictor would say "fire now" if it ran — proves the eviction, not
    # an early-return, is what prevents the stale fire.
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 0,
    )
    _stub_predictor_inputs(controller)

    # New round rolls in.
    new_ctx = _ctx_at_block(b"\xbb" * 32, 19)
    await controller.on_new_head(new_ctx)

    assert old_key not in controller._latest_preview  # evicted on rollover
    assert old_key not in controller._base_difficulty_by_key
    assert fired is False  # the stale candidate never fired


async def test_anticipatory_fire_paced_worker_idle(monkeypatch):
    """Even when the worker is paced (idle, no dispatch result), a captured
    preview drives a fire at B*-1 through `on_new_head`."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    controller.events = None
    handle = _FakeHandle("p0")  # idle: _active_dispatch_id == 0
    controller.miner_handles = [handle]
    controller.core = None
    controller._done_queues = {}
    controller._verify_proof_recorded = AsyncMock(return_value=42)
    _stub_predictor_inputs(controller)

    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    key = _work_key(ctx)
    _store_preview_entry(controller, ctx)
    # Controller already mining this key (so it's the active key), worker idle.
    controller._current_context = ctx
    controller._current_work_key = key

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.SUCCESS,
            receipt=ExtrinsicReceipt(extrinsic_hash="0xabc"),
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )
    monkeypatch.setattr(
        "substrate.miner_controller.block_when_energy_clears",
        lambda *a, **k: 20,
    )

    await controller.on_new_head(ctx)

    assert controller.stats.proofs_submitted == 1
    assert key in controller._closed_work_keys
    # Worker was never dispatched (the SUCCESS fire closed the round first).
    assert handle.mine_calls == []
