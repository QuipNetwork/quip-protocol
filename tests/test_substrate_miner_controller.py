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
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from dwave_topologies.topologies.zephyr import zephyr
from shared.keystore_hybrid import generate
from substrate.miner_bootstrap import BootstrapConfig, _maybe_seed_chain, _resolve_dev_signer
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from substrate.client import SubstrateClient
from chain_probe import chain_reachable as _chain_reachable
from chain_probe import dev_chain_reachable as _dev_chain_reachable
from substrate.miner_controller import (
    EARLY_SUBMISSION_ERRORS,
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


# Solution number the test harness seeds for the active round. Submissions
# for that round land under {SOLN}/submission.json (the on-disk key is the
# chain-global solution number, not dispatch_id).
_TEST_SOLUTION_NUMBER = 196


def _set_current(controller, ctx) -> None:
    """Helper: set `_current_context`, `_current_work_key`, and the round's
    solution number so the staleness check in `_handle_result` finds a
    baseline and submissions resolve the on-disk archive key.
    Phase 4 (storm-prevention) split the work-key check out of the
    context-equality check, so tests must now seed both."""
    from substrate.miner_controller import _work_key
    controller._current_context = ctx
    controller._current_work_key = _work_key(ctx)
    controller._solution_number_by_work_key[_work_key(ctx)] = (
        _TEST_SOLUTION_NUMBER
    )


class _StubScheduler:
    """Minimal WorkScheduler stand-in for bare-controller unit tests.

    Reads the controller's live ``miner_handles`` / ``_dispatch_contexts``
    at call time (tests reassign both after construction) and mirrors the
    real scheduler's dispatch surface: ``dispatch_pow`` broadcasts,
    ``fill_idle`` dispatches to idle handles only, ``dispatch_context``
    resolves the immutable per-dispatch context map.
    """

    def __init__(self, controller) -> None:
        self._controller = controller
        # Handles the "active mempool job" owns; cancel_pow_siblings
        # spares them exactly like the real scheduler.
        self.job_owned: set[str] = set()

    def dispatch_context(self, handle_id, dispatch_id):
        return self._controller._dispatch_contexts.get((handle_id, dispatch_id))

    def cancel_pow_siblings(self, winning_handle_id):
        for h in self._controller.miner_handles:
            if h.miner_id == winning_handle_id or h.miner_id in self.job_owned:
                continue
            h.cancel()

    async def dispatch_pow(self, context, *, solution_number=None):
        return {
            h.miner_id: h.mine_work_item(context, solution_number=solution_number)
            for h in self._controller.miner_handles
        }

    async def fill_idle(self, context, *, solution_number=None):
        return {
            h.miner_id: h.mine_work_item(context, solution_number=solution_number)
            for h in self._controller.miner_handles
            if h._active_dispatch_id == 0
        }


def _bare_controller() -> SubstrateMinerController:
    """Controller without calling __init__ — for unit tests that only
    exercise a single method. Sets up the attributes that method needs."""
    c = SubstrateMinerController.__new__(SubstrateMinerController)
    # T7: handle ops delegate to the scheduler; the stub reads the
    # controller's mutable handle list / context map at call time.
    # (`_dispatch_contexts` below is now test-only seed data the stub
    # scheduler's dispatch_context resolves — the real map lives on the
    # WorkScheduler.)
    c._scheduler = _StubScheduler(c)
    # After the pool.get("rpc") removal: parent owns build_client (compose+
    # sign only) and pool_client (swap-aware reads + submit).
    c.build_client = MagicMock()
    c.pool_client = MagicMock()
    # Default get_block_number returns 0; tests override as needed.
    c.pool_client.get_head = AsyncMock(return_value=b"\xff" * 32)
    c.pool_client.get_block_number = AsyncMock(return_value=0)
    c.pool_client.query_latest_qblock_id = AsyncMock(return_value=None)
    # Default: no on-chain timestamp anchor (best-effort; tests override).
    c.pool_client.query_block_timestamp_ms = AsyncMock(return_value=None)
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
    # Work-item announce de-dup: on_new_head logs at INFO only when the
    # work key changes, DEBUG otherwise. __init__ seeds this to None.
    c._last_logged_work_key = None
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
    c._latest_budget = {}
    c._participated = OrderedDict()
    c._pow_constants = None
    c._base_difficulty_by_key = {}
    c._decay_schedule_by_key = {}  # WorkKey -> (schedule, last_proof_block, epoch_length)
    c._anticipatory_fired = set()
    # Cadence fire timer (Task 7). Real TimingTracker with no observed head:
    # fire_deadline_monotonic returns None until a test seeds an anchor.
    from substrate.decay_timing import TimingTracker
    c._timing = TimingTracker()
    c._fire_timer_task = None
    c._last_fire_status_key = None
    # Resilience state (dark-validator fallbacks): no event manager, no
    # pending submission, schedule retry unthrottled, horizon log unarmed.
    c.events = None
    c._pending_submission = None
    c._replaying = None
    c._schedule_retry_next_monotonic = 0.0
    c._decay_horizon_logged_key = None
    # Per-round solution-number cache (on-disk archive key). Empty by
    # default; _set_current seeds it for the active round so submissions
    # land under the matching {solution_number}/ dir.
    c._solution_number_by_work_key = {}
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


def _chain_has_per_topology_difficulty(url: str) -> bool:
    """True if the chain's runtime exposes per-topology difficulty.

    `quip-protocol-rs` MR !42 replaced the global `QuantumPow.Difficulty`
    value with a per-topology `QuantumPow.Difficulties` storage map. The live
    controller tests drive `_maybe_seed_chain`, which queries/sets difficulty
    through that map; against an older runtime the seed path raises
    `StorageFunctionNotFound` deep in bootstrap. Probe the metadata so those
    tests skip cleanly (rather than hard-fail) until the runtime is deployed.
    """
    if not _chain_reachable(url):
        return False
    try:
        from substrateinterface import SubstrateInterface
        si = SubstrateInterface(url=url)
        pallet = si.get_metadata().get_metadata_pallet("QuantumPow")
        return (
            pallet is not None
            and pallet.get_storage_function("Difficulties") is not None
        )
    except Exception:
        return False


# Evaluated once at collection time; the live controller tests below gate on
# it so they skip cleanly against a pre-MR!42 runtime.
_CHAIN_HAS_PER_TOPOLOGY_DIFFICULTY = _chain_has_per_topology_difficulty(DEFAULT_URL)


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
        # hybrid chains. See substrate.miner_bootstrap._resolve_dev_signer.
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
    from substrate.work_scheduler import WorkScheduler
    pool = ValidatorPool(urls=[DEFAULT_URL])
    controller = SubstrateMinerController(
        pool=pool,
        signer=keystore.signer,
        miner_handles=[handle],
        topology_hash=chain_topology_hash,
        core=core,
    )
    # T7 wiring: the WorkScheduler owns the handle's drainer and all
    # dispatch; the controller delegates (mirrors _build_scheduler_stack).
    scheduler = WorkScheduler(
        [handle],
        on_pow_result=controller.enqueue_pow_result,
        on_worker_message=controller.handle_worker_message,
        provide_pow_context=controller.provide_pow_context,
        on_fatal=lambda _hid, _reason: controller.shutdown(),
    )
    controller.attach_scheduler(scheduler)

    # Tests that use the yielded ``client`` (e.g., to query the chain
    # after a submission) need a direct SubstrateClient — the controller
    # now keeps its build_client private and reads/submits go through
    # the swap-aware pool.
    client = SubstrateClient(urls=[DEFAULT_URL])
    await client.connect()
    scheduler.start()
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
        await scheduler.stop()
        await client.close()
        await pool.shutdown()
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
    not _dev_chain_reachable(DEFAULT_URL),
    reason=f"no dev substrate chain at {DEFAULT_URL}",
)
@pytest.mark.skipif(
    not _CHAIN_HAS_PER_TOPOLOGY_DIFFICULTY,
    reason="chain runtime lacks per-topology QuantumPow.Difficulties (pre-MR!42)",
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
    not _dev_chain_reachable(DEFAULT_URL),
    reason=f"no dev substrate chain at {DEFAULT_URL}",
)
@pytest.mark.skipif(
    not _CHAIN_HAS_PER_TOPOLOGY_DIFFICULTY,
    reason="chain runtime lacks per-topology QuantumPow.Difficulties (pre-MR!42)",
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

    def mine_work_item(self, context, *, solution_number=None):
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


async def test_won_work_sibling_cancel_spares_job_owned_handles(monkeypatch):
    """T7 regression: post-cutover `miner_handles` is ALL handles, so the
    won-work sibling cancel must route through the scheduler — a direct
    handle.cancel() would steal an active mempool job's dispatches (burning
    its single requeue; paid samples on opted-in QPU)."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    winner = _FakeHandle("winner")
    pow_sibling = _FakeHandle("pow-sibling")
    job_handle = _FakeHandle("job-handle")
    winner._active_dispatch_id = 1
    pow_sibling._active_dispatch_id = 1
    job_handle._active_dispatch_id = 1
    controller.miner_handles = [winner, pow_sibling, job_handle]
    controller._scheduler.job_owned = {"job-handle"}

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(extrinsic_hash="0xabc")  # OK

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="winner"
    )
    await controller._handle_result(envelope)

    assert pow_sibling.cancel_calls == 1
    assert job_handle.cancel_calls == 0  # the job's handle is spared
    assert winner.cancel_calls == 0


async def test_handle_result_redispatches_idle_handle_on_verify_mismatch(monkeypatch):
    """OK receipt but the runtime recorded someone else's proof
    (_verify_proof_recorded returns -1): the controller must NOT close the
    work key, and must re-dispatch the SAME context to IDLE handles only —
    skipping a busy handle — while bumping contexts_dispatched. This is the
    anti-deadlock guarantee that keeps an idle worker from sitting forever
    after a silent rejection. It exercises _finalize_accepted_proof's
    mismatch branch and _redispatch_after_verify_fail (gate_idle=True), which
    are reachable only through _handle_result's OK path."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)

    idle = _FakeHandle("idle")            # _active_dispatch_id == 0 → eligible
    busy = _FakeHandle("busy")
    busy._active_dispatch_id = 7          # active → must be skipped
    controller.miner_handles = [idle, busy]

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(extrinsic_hash="0xabc")  # classifies OK

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    controller._verify_proof_recorded = AsyncMock(return_value=-1)  # type: ignore[assignment]

    await controller._handle_result(
        _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="idle")
    )

    # Anti-deadlock invariant: the work key stays open (not won by us).
    assert _work_key(ctx) not in controller._closed_work_keys
    assert controller.stats.proofs_submitted == 0
    # Only the idle handle was re-dispatched; the busy one was skipped.
    assert len(idle.mine_calls) == 1
    assert idle.mine_calls[0][1] is ctx
    assert len(busy.mine_calls) == 0
    assert controller.stats.contexts_dispatched == 1


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
    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
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
    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
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

    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "rejected_stale"
    assert record["pow_sequence"] is None, (
        "pow_sequence must be None (not absent, not an exception) when the "
        "chain RPC is unavailable at submission time"
    )


async def test_handle_result_fills_device_access_time_from_qpu_sum(monkeypatch):
    """QPU path: the submitted MiningResult carries the summed per-attempt
    QPU access time, not the wall clock."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    monkeypatch.setattr(controller, "_sum_qpu_access_us", lambda n: 555_000)

    captured = {}

    async def fake_submit_proof(bc, pc, signer, result, context, **kw):
        captured["device"] = result.device_access_time_us
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=InvalidNonce, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    controller.pool_client.query_proofs_submitted = AsyncMock(return_value=0)
    envelope = _ResultEnvelope(result=_mining_result(), context=ctx, handle_id="t-0")
    await controller._handle_result(envelope)
    assert captured["device"] == 555_000


async def test_handle_result_falls_back_to_wall_clock(monkeypatch):
    """No QPU time recorded -> wall clock seconds * 1e6."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    monkeypatch.setattr(controller, "_sum_qpu_access_us", lambda n: None)

    captured = {}

    async def fake_submit_proof(bc, pc, signer, result, context, **kw):
        captured["device"] = result.device_access_time_us
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=InvalidNonce, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )
    controller.pool_client.query_proofs_submitted = AsyncMock(return_value=0)
    result = _mining_result()
    # mining_time is typed int but dataclasses don't enforce it; use a float
    # to pin that sub-second precision is preserved (the old floor-then-scale
    # bug would report 7_000_000 instead of 7_500_000).
    result.mining_time = 7.5
    envelope = _ResultEnvelope(result=result, context=ctx, handle_id="t-0")
    await controller._handle_result(envelope)
    assert captured["device"] == 7_500_000


_RECORD_SUBMISSION_LOG_COMMON = {
    "solution_number": _TEST_SOLUTION_NUMBER,
    "miner_id": "miner-7",
    "miner_type": "cpu",
    "energy_milli": -1234,
    "diversity_milli": 250,
    "threshold_milli": -1000,
    "last_proof_block_hash_hex": "0x" + "ab" * 32,
    "num_valid": 5,
}


@pytest.mark.parametrize(
    "outcome,extra",
    [
        # chain_error / RPC-error
        ("chain_error", {"pow_sequence": 11, "error": "RuntimeError: boom"}),
        # submitted_inblock / OK
        (
            "submitted_inblock",
            {
                "extrinsic_hash": "0xext",
                "chain_block_hash": "0xblk",
                "chain_block_number": 999,
                "qpu_access_us_total": 61000,
            },
        ),
        # rejected_stale / STALE
        (
            "rejected_stale",
            {"extrinsic_hash": "0xext", "pow_sequence": 12, "error": "stale"},
        ),
        # chain_error / FATAL
        (
            "chain_error",
            {"extrinsic_hash": "0xext", "pow_sequence": 13, "error": "fatal"},
        ),
    ],
)
def test_record_submission_forwards_kwargs(outcome, extra):
    """``_record_submission`` writes the same row as a flat ``record(...)`` call.

    Guards the helper that collapsed the four formerly inline
    ``self._submission_log.record(...)`` blocks in ``_handle_result``: the row
    it writes must be byte-identical (modulo the always-changing ``ts_ns``) to
    a direct ``record(**log_common, outcome=outcome, **extra)`` call.
    """
    import tempfile

    from shared.mining_attempt_log import SubmissionLogger

    controller = _bare_controller()
    controller._record_submission(
        dict(_RECORD_SUBMISSION_LOG_COMMON), outcome, **extra
    )
    helper_path = (
        controller._submission_log.log_dir
        / str(_TEST_SOLUTION_NUMBER)
        / "submission.json"
    )
    helper_record = json.loads(helper_path.read_text())

    direct_log = SubmissionLogger(
        log_dir=Path(tempfile.mkdtemp(prefix="quip-test-direct-")),
    )
    direct_log.record(
        **_RECORD_SUBMISSION_LOG_COMMON, outcome=outcome, **extra
    )
    direct_path = (
        direct_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
    )
    direct_record = json.loads(direct_path.read_text())

    # ts_ns is wall-clock and necessarily differs between the two writes.
    helper_record.pop("ts_ns", None)
    direct_record.pop("ts_ns", None)
    assert helper_record == direct_record


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
            "miner_type": "cpu",
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

    # Seed the round's solution number so an anticipatory fire's submission
    # record lands under {_TEST_SOLUTION_NUMBER}/ (the on-disk archive key).
    controller._solution_number_by_work_key[_work_key(ctx)] = (
        _TEST_SOLUTION_NUMBER
    )
    controller._latest_preview[_work_key(ctx)] = {
        "handle_id": "p0",
        "context": ctx,
        "dispatch_id": 1,
        "miner_type": "cpu",
        "nonce": (7).to_bytes(32, "big"),
        "salt": b"\x11" * 32,
        "solutions": [[1, -1, 1, -1]],
        "submit_floor_energy": floor,
        "energy": floor,
        "num_valid": 3,
        "diversity": 0.5,
        "decay_num": 2,
        "valid_at_block": 20,
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


async def test_fire_preview_success_records_and_closes(monkeypatch):
    """A SUCCESS fire verifies, records, and marks the work key closed.

    Exercises ``_fire_preview`` directly — the cadence timer is what decides
    *when* to call it, but the SUCCESS branch behavior is the same.
    """
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    controller._verify_proof_recorded = AsyncMock(return_value=42)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    preview = controller._latest_preview[key]

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

    await controller._fire_preview(ctx, key, preview, 20)

    assert controller.stats.proofs_submitted == 1
    assert key in controller._closed_work_keys
    # The fired proof was reconstructed from the preview, carrying the real
    # source backend (cpu), not the "anticipatory" path marker.
    assert captured["result"].nonce == (7).to_bytes(32, "big")
    assert captured["result"].num_valid == 3
    assert captured["result"].miner_type == "cpu"
    assert captured["context"] is ctx
    # The submission-log row records the real backend for per-backend
    # dashboard attribution (not "anticipatory").
    import json
    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "submitted_inblock"
    assert record["miner_type"] == "cpu"


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
    preview = controller._latest_preview[key]

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.SUCCESS,
            receipt=ExtrinsicReceipt(extrinsic_hash="0xabc", block_hash="0xdef"),
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)

    # Not won: key stays open, mid-fire mark cleared so a later head re-fires,
    # preview retained.
    assert controller.stats.proofs_submitted == 0
    assert controller.stats.proofs_unverified == 1
    assert key not in controller._closed_work_keys
    assert key not in controller._anticipatory_fired
    assert key in controller._latest_preview
    # A chain_error submission-log row was written (dispatch_id=1 from preview).
    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
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
    preview = controller._latest_preview[key]

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(action=SubmitRetryAction.RETRY, attempts=4)

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)

    assert controller.stats.proofs_submitted == 0
    assert key not in controller._closed_work_keys
    assert key in controller._latest_preview  # preview retained
    assert key not in controller._anticipatory_fired  # mid-fire mark cleared


async def test_anticipatory_round_stale_discards_preview(monkeypatch):
    """STOP_ROUND_STALE means the nonce is dead — discard the preview and
    any pending anticipatory state."""
    import json

    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    controller.pool_client.query_proofs_submitted = AsyncMock(return_value=88)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    preview = controller._latest_preview[key]

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.STOP_ROUND_STALE,
            error="Module(error=InvalidNonce, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)

    assert controller.stats.proofs_submitted == 0
    assert key not in controller._latest_preview  # discarded
    assert key not in controller._anticipatory_fired
    assert controller.stats.stale_drops == 1
    # Audit parity: a rejected_stale row is written (real backend, chain Sol#).
    sub_path = controller._submission_log.log_dir / str(_TEST_SOLUTION_NUMBER) / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["outcome"] == "rejected_stale"
    assert record["miner_type"] == "cpu"
    assert record["num_valid"] == 3
    assert record["pow_sequence"] == 88
    assert "InvalidNonce" in record["error"]


# ----------------------------------------------------------------------
# Task 7: free-running cadence fire timer reads worker-computed valid_at_block
# ----------------------------------------------------------------------


def _seed_cadence_state(controller, *, valid_at_block, decay_num=2):
    """Bind a controller to an active preview carrying a worker win-time."""
    ctx = _ctx_at_block(b"\xaa" * 32, 5)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    _store_preview_entry(controller, ctx)
    controller._latest_preview[key]["valid_at_block"] = valid_at_block
    controller._latest_preview[key]["decay_num"] = decay_num
    controller._current_context = ctx
    controller._current_work_key = key
    return ctx, key


async def test_cadence_timer_fires_at_deadline(monkeypatch):
    """With a timing anchor putting the deadline in the past, the cadence
    tick fires the active preview once at ``b_star == valid_at_block``."""
    controller = _bare_controller()
    ctx, key = _seed_cadence_state(controller, valid_at_block=20)

    # Seed the tracker anchor so fire_deadline_monotonic returns a past time:
    # anchor far in the past with a tiny interval => deadline << now.
    now = asyncio.get_running_loop().time()
    controller._timing.observe_head(
        block_number=20,
        chain_ts_s=1000.0,
        monotonic_now=now - 100.0,
        wallclock_now=1000.0,
    )

    fired = []

    async def fake_fire(c, k, p, b):
        fired.append((k, b))

    monkeypatch.setattr(controller, "_fire_preview", fake_fire)

    await controller._maybe_fire_on_cadence()

    assert len(fired) == 1
    assert fired[0][0] == key
    assert fired[0][1] == 20


async def test_cadence_no_fire_when_valid_at_block_zero(monkeypatch):
    """A non-decay preview carries ``valid_at_block=0`` (the legacy/sentinel
    path, or a schedule-less round after a transient RPC failure). Firing on
    it would make ``fire_deadline_monotonic(b_star=0)`` resolve to the distant
    past — with a real positive anchor block — and submit an un-gated
    candidate every tick (submission storm). The guard must suppress it."""
    controller = _bare_controller()
    ctx, key = _seed_cadence_state(controller, valid_at_block=0, decay_num=0)

    # Realistic positive anchor: two heads near block 1000. Without the guard,
    # fire_deadline_monotonic(b_star=0) = anchor_mono + (0 - 1000)*interval -
    # lag => far in the past => deadline <= now => it WOULD fire.
    now = asyncio.get_running_loop().time()
    controller._timing.observe_head(
        block_number=999, chain_ts_s=5994.0,
        monotonic_now=now - 6.0, wallclock_now=5994.0,
    )
    controller._timing.observe_head(
        block_number=1000, chain_ts_s=6000.0,
        monotonic_now=now, wallclock_now=6000.0,
    )

    fired = []
    monkeypatch.setattr(
        controller, "_fire_preview",
        lambda *a, **k: fired.append(a),
    )

    await controller._maybe_fire_on_cadence()
    assert fired == []


async def test_cadence_no_fire_before_deadline(monkeypatch):
    """A deadline in the future (anchor block far below valid_at) does not
    fire on this tick."""
    controller = _bare_controller()
    _seed_cadence_state(controller, valid_at_block=200)

    now = asyncio.get_running_loop().time()
    # Anchor at block 10 with a 6 s interval: deadline for block 200 is far out.
    controller._timing.observe_head(
        block_number=9, chain_ts_s=900.0, monotonic_now=now, wallclock_now=900.0,
    )
    controller._timing.observe_head(
        block_number=10, chain_ts_s=906.0, monotonic_now=now, wallclock_now=906.0,
    )

    fired = []
    monkeypatch.setattr(
        controller, "_fire_preview",
        lambda *a, **k: fired.append(a),
    )

    await controller._maybe_fire_on_cadence()
    assert fired == []


async def test_cadence_timer_dedups_when_already_fired(monkeypatch):
    """A key already in ``_anticipatory_fired`` does not re-fire."""
    controller = _bare_controller()
    _ctx, key = _seed_cadence_state(controller, valid_at_block=20)
    controller._anticipatory_fired.add(key)

    now = asyncio.get_running_loop().time()
    controller._timing.observe_head(
        block_number=20, chain_ts_s=1000.0,
        monotonic_now=now - 100.0, wallclock_now=1000.0,
    )

    fired = []
    monkeypatch.setattr(
        controller, "_fire_preview", lambda *a, **k: fired.append(a),
    )

    await controller._maybe_fire_on_cadence()
    assert fired == []


async def test_cadence_no_fire_without_preview(monkeypatch):
    """An active key with no stored preview does not fire."""
    controller = _bare_controller()
    ctx = _ctx_at_block(b"\xaa" * 32, 5)
    from substrate.miner_controller import _work_key
    controller._current_context = ctx
    controller._current_work_key = _work_key(ctx)  # no _latest_preview entry

    now = asyncio.get_running_loop().time()
    controller._timing.observe_head(
        block_number=20, chain_ts_s=1000.0,
        monotonic_now=now - 100.0, wallclock_now=1000.0,
    )

    fired = []
    monkeypatch.setattr(
        controller, "_fire_preview", lambda *a, **k: fired.append(a),
    )

    await controller._maybe_fire_on_cadence()
    assert fired == []


async def test_fire_timer_cancelled_on_teardown():
    """``_teardown`` cancels the fire-timer task and clears the handle."""
    controller = _bare_controller()
    controller.miner_handles = []
    controller._drainer_tasks = []
    controller.events = None
    controller._stats_writer_task = None
    controller._telemetry_shutdown_event = None
    controller._telemetry_proc = None
    controller._event_manager_task = None

    task = asyncio.create_task(controller._fire_timer_loop())
    controller._fire_timer_task = task
    await asyncio.sleep(0)  # let the loop start

    await controller._teardown()

    assert task.cancelled() or task.done()
    assert controller._fire_timer_task is None


def test_evict_anticipatory_state_prunes_all():
    """Eviction drops the preview, the cached base difficulty, the decay
    schedule, and the fired mark for a work key."""
    controller = _bare_controller()
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    _store_preview_entry(controller, ctx)
    controller._base_difficulty_by_key[key] = SubstrateDifficulty(1, -1000, 0)
    controller._decay_schedule_by_key[key] = ([-1000, -990], 10, 5)
    controller._anticipatory_fired.add(key)

    controller._evict_anticipatory_state(key)

    assert key not in controller._latest_preview
    assert key not in controller._base_difficulty_by_key
    assert key not in controller._decay_schedule_by_key
    assert key not in controller._anticipatory_fired


async def test_on_new_head_attaches_decay_schedule(monkeypatch):
    """After on_new_head dispatches, the context handed to mine_work_item
    must carry a monotonic decay_schedule, a positive epoch_length, and
    a non-negative last_proof_block."""
    controller = _bare_controller()
    controller.events = None
    handle = _FakeHandle("p0")
    controller.miner_handles = [handle]
    controller.core = None
    controller._done_queues = {}
    _stub_predictor_inputs(controller, last_proof_block=10)

    ctx = _ctx_at_block(b"\xaa" * 32, 15)

    await controller.on_new_head(ctx)

    # The handle must have been dispatched.
    assert handle.mine_calls, "expected at least one mine_work_item dispatch"
    _dispatch_id, dispatched_ctx = handle.mine_calls[-1]

    assert dispatched_ctx.decay_schedule is not None, "decay_schedule must be attached"
    assert len(dispatched_ctx.decay_schedule) > 1, (
        "schedule must have more than one step (base + at least one decay step)"
    )
    # Monotonic non-decreasing (decay only eases max_energy_milli upward).
    sched = dispatched_ctx.decay_schedule
    assert all(
        sched[i] <= sched[i + 1] for i in range(len(sched) - 1)
    ), "decay_schedule must be monotonic non-decreasing"
    assert dispatched_ctx.epoch_length > 0, "epoch_length must be positive"
    assert dispatched_ctx.last_proof_block >= 0, "last_proof_block must be non-negative"


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
    _stub_predictor_inputs(controller)

    # New round rolls in.
    new_ctx = _ctx_at_block(b"\xbb" * 32, 19)
    await controller.on_new_head(new_ctx)

    assert old_key not in controller._latest_preview  # evicted on rollover
    assert old_key not in controller._base_difficulty_by_key
    assert fired is False  # the stale candidate never fired


# ----------------------------------------------------------------------
# Live QPU budget: store + telemetry snapshot surfacing
# ----------------------------------------------------------------------


def test_store_budget_lands_in_latest_budget_by_miner_id():
    """A `{"op":"budget"}` push lands in `_latest_budget` keyed by miner id."""
    from types import SimpleNamespace

    controller = _bare_controller()
    handle = SimpleNamespace(miner_id="qpu-1")
    stats = {"cumulative_used_seconds": 12.5, "blocks_skipped": 2}
    controller._store_budget(handle, {"op": "budget", "data": stats})
    assert controller._latest_budget["qpu-1"] == stats


def test_store_budget_ignores_malformed_payload():
    """A non-dict payload is dropped, not stored (can't poison telemetry)."""
    from types import SimpleNamespace

    controller = _bare_controller()
    handle = SimpleNamespace(miner_id="qpu-1")
    controller._store_budget(handle, {"op": "budget", "data": None})
    assert controller._latest_budget == {}


def test_snapshot_surfaces_per_miner_qpu_budget():
    """build_stats_snapshot_for_telemetry attaches the live budget per miner."""
    from types import SimpleNamespace
    from substrate.miner_controller import build_stats_snapshot_for_telemetry

    controller = _bare_controller()
    controller.signer = None  # skip the survey path
    budget = {"cumulative_used_seconds": 12.5, "blocks_skipped": 2}
    controller._latest_budget = {"qpu-1": budget}
    controller.core = SimpleNamespace(
        node_id="node-x",
        miner_handles=[SimpleNamespace(miner_id="qpu-1", miner_type="QPU")],
        descriptor=lambda: {},
    )
    snap = build_stats_snapshot_for_telemetry(controller)
    miner = snap["miners"][0]
    assert miner["id"] == "qpu-1"
    assert miner["qpu_budget"] == budget


def test_snapshot_qpu_budget_is_none_when_unreported():
    """A miner that never pushed budget shows qpu_budget=None (not missing)."""
    from types import SimpleNamespace
    from substrate.miner_controller import build_stats_snapshot_for_telemetry

    controller = _bare_controller()
    controller.signer = None
    controller.core = SimpleNamespace(
        node_id="node-x",
        miner_handles=[SimpleNamespace(miner_id="cpu-1", miner_type="CPU")],
        descriptor=lambda: {},
    )
    snap = build_stats_snapshot_for_telemetry(controller)
    assert snap["miners"][0]["qpu_budget"] is None


# ----------------------------------------------------------------------
# Task 7: _evict_anticipatory_state re-arms the throttled status line
# ----------------------------------------------------------------------


def test_evict_resets_fire_status_key():
    """Evicting a key clears ``_last_fire_status_key`` so the next round's
    candidate logs even if its (valid_at_block, decay_num) collides."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    key = _work_key(_context(b"\xaa" * 32))
    controller._last_fire_status_key = (20, 2)

    controller._evict_anticipatory_state(key)

    assert controller._last_fire_status_key is None


# ----------------------------------------------------------------------
# Participation marker (write-once MinerRegistry.participate per solution#, node-level)
# ----------------------------------------------------------------------


async def test_participation_marker_submits_with_call_params():
    controller = _bare_controller()
    controller.build_client.submit_extrinsic = AsyncMock(
        return_value=MagicMock(error=None)
    )
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=4)
    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 7, "miner": "qpu-0",
         "kind": "qpu", "budget_seconds": 90.0}
    )
    controller.build_client.submit_extrinsic.assert_awaited_once()
    args, kwargs = controller.build_client.submit_extrinsic.call_args
    assert args[0] == "MinerRegistry"
    assert args[1] == "participate"
    assert args[2] == {
        "qblock_id": 5,
        "kind": "QpuDwave",
        "budget_seconds": 90,
    }
    assert kwargs["wait_for"] == "inblock"


async def test_participation_marker_uses_first_qblock_when_no_latest_id():
    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=None)
    controller.build_client.submit_extrinsic = AsyncMock(
        return_value=MagicMock(error=None)
    )
    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 1, "miner": "cpu-0",
         "kind": "cpu"}
    )
    args, _kwargs = controller.build_client.submit_extrinsic.call_args
    assert args[2]["qblock_id"] == 1
    assert args[2]["kind"] == "Cpu"


async def test_participation_remark_retries_transient_then_succeeds():
    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=2)
    # First attempt fails with a stale-nonce "outdated" error (the reported
    # 1010 Invalid Transaction); the retry re-composes a fresh nonce and lands.
    controller.build_client.submit_extrinsic = AsyncMock(
        side_effect=[
            RuntimeError("1010 Invalid Transaction: Transaction is outdated"),
            MagicMock(error=None),
        ]
    )
    slept: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        slept.append(seconds)

    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 3, "miner": "5Test",
         "kind": "cpu"},
        sleeper=_record_sleep,
    )
    assert controller.build_client.submit_extrinsic.await_count == 2
    assert len(slept) == 1  # one backoff between the two attempts


async def test_participation_remark_swallows_persistent_failure():
    from substrate.miner_controller import _PARTICIPATION_REMARK_RETRIES

    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=2)
    controller.build_client.submit_extrinsic = AsyncMock(
        side_effect=RuntimeError("rpc down")
    )

    async def _no_sleep(_seconds: float) -> None:
        return None

    # Must not raise — retries the bounded number of times, then gives up.
    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 2, "miner": "5Test",
         "kind": "qpu"},
        sleeper=_no_sleep,
    )
    assert (
        controller.build_client.submit_extrinsic.await_count
        == _PARTICIPATION_REMARK_RETRIES + 1
    )


async def test_participation_remark_times_out_a_hung_submit():
    """A submit that never returns is bounded, retried, then swallowed.

    Regression: a half-dead validator that accepts the extrinsic but never
    reports inclusion once froze the fire-and-forget marker task forever
    (no timeout on the ``wait_for="inblock"`` submit), silently pinning the
    on-chain participation marker while the chain advanced. The submit must
    be watch-timed like the win path so a hang becomes a transient failure.
    """
    from substrate.miner_controller import _PARTICIPATION_REMARK_RETRIES

    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=2)

    async def _never_returns(*_args, **_kwargs):
        await asyncio.Event().wait()  # blocks forever

    controller.build_client.submit_extrinsic = AsyncMock(side_effect=_never_returns)

    async def _no_sleep(_seconds: float) -> None:
        return None

    # Test-level guard: without the fix the coroutine hangs and this fires.
    await asyncio.wait_for(
        controller._submit_participation_remark(
            {"schema": "quip-participation", "solution": 2, "miner": "5Test",
             "kind": "qpu"},
            sleeper=_no_sleep,
            submit_timeout=0.01,
        ),
        timeout=5.0,
    )
    # Each hung attempt times out and is retried the bounded number of times.
    assert (
        controller.build_client.submit_extrinsic.await_count
        == _PARTICIPATION_REMARK_RETRIES + 1
    )


async def test_mark_participating_dedups_per_solution_across_instances():
    controller = _bare_controller()
    controller._submit_participation_remark = AsyncMock(return_value=None)
    # Two different miner instances report the SAME solution # → exactly one
    # node-level marker (deduped on solution#, not on the per-instance worker).
    controller._mark_participating(
        {"solution_number": 5, "kind": "qpu", "budget_seconds": 90.0}
    )
    controller._mark_participating({"solution_number": 5, "kind": "cpu"})
    await asyncio.sleep(0)  # let the spawned task run
    controller._submit_participation_remark.assert_awaited_once()
    payload = controller._submit_participation_remark.call_args.args[0]
    assert payload == {
        "schema": "quip-participation", "solution": 5, "miner": "5Test",
        "kind": "qpu", "budget_seconds": 90.0,
    }
    # A different solution # fires again.
    controller._mark_participating({"solution_number": 6, "kind": "qpu"})
    await asyncio.sleep(0)
    assert controller._submit_participation_remark.await_count == 2


async def test_mark_participating_uses_node_id_and_omits_budget():
    controller = _bare_controller()
    controller._submit_participation_remark = AsyncMock(return_value=None)
    controller._mark_participating({"solution_number": 9, "kind": "cpu"})
    await asyncio.sleep(0)
    payload = controller._submit_participation_remark.call_args.args[0]
    assert "budget_seconds" not in payload
    assert payload["kind"] == "cpu"
    # Node identity (signer ss58), not the per-instance worker id.
    assert payload["miner"] == "5Test"


async def test_mark_participating_skips_when_signer_address_unavailable():
    controller = _bare_controller()
    controller._submit_participation_remark = AsyncMock(return_value=None)
    # _mark_participating runs unguarded on the drain loop; a signer failure
    # must NOT propagate (the drain loop's broad except would shut the
    # controller down — an observability failure crashing mining).
    controller.signer.ss58_address.side_effect = RuntimeError("keystore locked")
    controller._mark_participating({"solution_number": 11, "kind": "cpu"})
    await asyncio.sleep(0)
    controller._submit_participation_remark.assert_not_awaited()
    # The solution must NOT be pre-marked done, so a recovered later report
    # still fires (no permanent suppression from a transient signer failure).
    controller.signer.ss58_address.side_effect = None
    controller.signer.ss58_address.return_value = "5Test"
    controller._mark_participating({"solution_number": 11, "kind": "cpu"})
    await asyncio.sleep(0)
    controller._submit_participation_remark.assert_awaited_once()


async def test_mark_participating_evicts_oldest_deterministically(monkeypatch):
    import substrate.miner_controller as mc
    monkeypatch.setattr(mc, "_PARTICIPATION_RETENTION", 3)
    controller = _bare_controller()
    controller._submit_participation_remark = AsyncMock(return_value=None)
    # Add 4 with retention 3 → the OLDEST (1) is evicted, never an arbitrary
    # (possibly still-active) entry that would re-fire its marker.
    for sol in (1, 2, 3, 4):
        controller._mark_participating({"solution_number": sol, "kind": "cpu"})
    await asyncio.sleep(0)
    assert list(controller._participated.keys()) == [2, 3, 4]


async def test_participation_remark_receipt_error_is_terminal():
    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(return_value=2)
    controller.build_client.submit_extrinsic = AsyncMock(
        return_value=MagicMock(error="System.ExtrinsicFailed")
    )

    async def _no_sleep(_seconds: float) -> None:
        return None

    # An included-but-rejected marker won't be fixed by resubmitting the same
    # call — it's terminal, not retried (no pointless resubmission storm).
    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 4, "miner": "5Test",
         "kind": "cpu"},
        sleeper=_no_sleep,
    )
    assert controller.build_client.submit_extrinsic.await_count == 1


async def test_participation_remark_retries_when_fallback_path_also_fails():
    controller = _bare_controller()
    controller.pool_client.query_latest_qblock_id = AsyncMock(side_effect=[2, 3])
    # Attempt 1 loses a stale-nonce race. Attempt 2 re-reads LatestQBlockId and
    # recomposes for the new candidate qblock before landing.
    controller.build_client.submit_extrinsic = AsyncMock(side_effect=[
        RuntimeError("1010 Transaction is outdated"),
        MagicMock(error=None),
    ])

    async def _no_sleep(_seconds: float) -> None:
        return None

    await controller._submit_participation_remark(
        {"schema": "quip-participation", "solution": 8, "miner": "5Test",
         "kind": "qpu"},
        sleeper=_no_sleep,
    )
    assert controller.build_client.submit_extrinsic.await_count == 2
    assert controller.build_client.submit_extrinsic.call_args_list[0].args[2]["qblock_id"] == 3
    assert controller.build_client.submit_extrinsic.call_args_list[1].args[2]["qblock_id"] == 4



# ----------------------------------------------------------------------
# Precise per-solution QPU spend at win (summed from the attempt log)
# ----------------------------------------------------------------------


def test_sum_qpu_access_us_sums_across_handles(monkeypatch):
    controller = _bare_controller()
    controller.miner_handles = [
        MagicMock(miner_id="qpu-0"), MagicMock(miner_id="cpu-0"),
    ]
    attempts = {
        "qpu-0": [{"qpu_access_time_us": 40_000}, {"qpu_access_time_us": 61_000}],
        "cpu-0": [{"qpu_access_time_us": None}, {}],  # CPU carries no QPU time
    }
    monkeypatch.setattr(
        "substrate.miner_controller.query_by_solution_number",
        lambda miner_id, n, *, log_dir=None, limit=None: attempts[miner_id],
    )
    assert controller._sum_qpu_access_us(7) == 101_000


def test_sum_qpu_access_us_none_for_missing_solution(monkeypatch):
    controller = _bare_controller()
    controller.miner_handles = [MagicMock(miner_id="qpu-0")]
    monkeypatch.setattr(
        "substrate.miner_controller.query_by_solution_number",
        lambda *a, **k: [],
    )
    # No solution number -> None (cannot key the attempt log).
    assert controller._sum_qpu_access_us(None) is None
    # Empty attempt log sums to 0 (not None).
    assert controller._sum_qpu_access_us(7) == 0


def test_sum_qpu_access_us_swallows_read_errors(monkeypatch):
    controller = _bare_controller()
    controller.miner_handles = [MagicMock(miner_id="qpu-0")]

    def _boom(*a, **k):
        raise OSError("attempt log unreadable")

    monkeypatch.setattr(
        "substrate.miner_controller.query_by_solution_number", _boom
    )
    assert controller._sum_qpu_access_us(7) is None


# ----------------------------------------------------------------------
# Pending submissions (optimistic submit-on-reconnect) + EARLY outcome
# ----------------------------------------------------------------------


class _IdleHandle:
    """Idle fake handle: records fill_idle re-dispatches."""

    def __init__(self, miner_id: str = "h0") -> None:
        self.miner_id = miner_id
        self._active_dispatch_id = 0
        self.dispatched: list = []

    def mine_work_item(self, context, *, solution_number=None) -> int:
        self.dispatched.append(context)
        self._active_dispatch_id += 1
        return self._active_dispatch_id


@pytest.mark.parametrize("error_name", EARLY_SUBMISSION_ERRORS)
def test_classify_early_error_names_substrate_format(error_name):
    """EARLY beats the unknown→FATAL default: a local-decay overshoot
    submission must queue-and-retry, never crash the controller."""
    receipt = ExtrinsicReceipt(
        extrinsic_hash="0xabc",
        error=f"Module(error={error_name}, pallet='QuantumPow', index=0)",
    )
    assert classify_submission(receipt) is SubmissionOutcome.EARLY


async def test_submit_failure_queues_pending_and_fills_idle(monkeypatch):
    """An RPC-failed submit holds the result for replay (not a drop) and
    resumes mining the round on the now-idle emitter."""
    from substrate.miner_controller import _work_key

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    handle = _IdleHandle()
    controller.miner_handles = [handle]

    async def fake_submit_proof(*args, **kwargs):
        raise ConnectionError("websocket disconnected")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="test-0",
    )
    await controller._handle_result(envelope)

    pending = controller._pending_submission
    assert pending is not None
    assert pending.work_key == _work_key(ctx)
    assert pending.attempts == 1
    # effective_floor falls back to energy (-1.0) => -1000 milli.
    assert pending.floor_milli == -1000
    assert handle.dispatched == [ctx], "fill_idle must resume the round"
    assert controller.stats.submission_errors == 1


async def test_early_receipt_queues_pending(monkeypatch):
    """An EARLY receipt (threshold not yet decayed to the floor) queues the
    result instead of raising like the former unknown→FATAL path."""
    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = []

    async def fake_submit_proof(*args, **kwargs):
        return ExtrinsicReceipt(
            extrinsic_hash="0xabc",
            error="Module(error=InsufficientEnergy, pallet='QuantumPow', index=0)",
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="test-0",
    )
    await controller._handle_result(envelope)  # must not raise
    assert controller._pending_submission is not None


async def test_pending_keeps_lower_floor():
    """A better (lower-floor) sibling replaces the held envelope; a worse
    one only bumps attempt/backoff state."""
    import dataclasses as _dc

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = []
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    log_common = {"solution_number": 1, "miner_id": "t", "miner_type": "CPU",
                  "energy_milli": 0, "diversity_milli": 0, "threshold_milli": 0,
                  "last_proof_block_hash_hex": "0x", "num_valid": 1}

    def _envelope(floor):
        result = _dc.replace(_mining_result(), submit_floor_energy=floor)
        return _ResultEnvelope(result=result, context=ctx, handle_id="t")

    await controller._queue_pending_submission(
        _envelope(-5.0), key, error="e1", log_common=log_common,
    )
    assert controller._pending_submission.floor_milli == -5000

    # Worse floor: state bumps, envelope kept.
    await controller._queue_pending_submission(
        _envelope(-4.0), key, error="e2", log_common=log_common,
    )
    assert controller._pending_submission.floor_milli == -5000
    assert controller._pending_submission.attempts == 2

    # Better floor: replaces.
    await controller._queue_pending_submission(
        _envelope(-6.0), key, error="e3", log_common=log_common,
    )
    assert controller._pending_submission.floor_milli == -6000
    assert controller._pending_submission.attempts == 3


def _seed_pending(controller, ctx, *, attempts=1, last_attempt_age_s=100.0):
    """Install a pending submission directly (bypasses the queue path)."""
    import time as _time
    from substrate.miner_controller import _PendingSubmission, _work_key

    envelope = _ResultEnvelope(
        result=_mining_result(), context=ctx, handle_id="t",
    )
    now = _time.monotonic()
    pending = _PendingSubmission(
        envelope=envelope,
        work_key=_work_key(ctx),
        floor_milli=-1000,
        queued_at_monotonic=now - last_attempt_age_s,
        last_attempt_monotonic=now - last_attempt_age_s,
        attempts=attempts,
    )
    controller._pending_submission = pending
    return pending


async def test_replay_on_recovery_hands_pending_to_result_path():
    """With the pool fresh and backoff elapsed, the replay takes the
    pending OUT and re-drives the full result path."""
    import time as _time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    pending = _seed_pending(controller, ctx)
    controller.events = SimpleNamespace(
        last_successful_poll_monotonic=_time.monotonic(),
    )
    controller._handle_result = AsyncMock()

    await controller._maybe_replay_pending()

    controller._handle_result.assert_awaited_once_with(pending.envelope)
    assert controller._pending_submission is None


async def test_replay_skipped_while_pool_stale():
    """No replay while the pool is still dark — wait for reconnect."""
    import time as _time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    _seed_pending(controller, ctx)
    controller.events = SimpleNamespace(
        last_successful_poll_monotonic=_time.monotonic() - 100.0,
    )
    controller._handle_result = AsyncMock()

    await controller._maybe_replay_pending()

    controller._handle_result.assert_not_awaited()
    assert controller._pending_submission is not None


async def test_replay_respects_backoff():
    import time as _time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    _seed_pending(controller, ctx, last_attempt_age_s=0.0)  # just attempted
    controller.events = SimpleNamespace(
        last_successful_poll_monotonic=_time.monotonic(),
    )
    controller._handle_result = AsyncMock()

    await controller._maybe_replay_pending()

    controller._handle_result.assert_not_awaited()
    assert controller._pending_submission is not None


async def test_pending_dropped_on_work_key_roll():
    """A pending bound to a rolled round is dropped, both by the replay
    guard and by _evict_anticipatory_state."""
    from unittest.mock import AsyncMock

    controller = _bare_controller()
    ctx_old = _context(b"\xaa" * 32)
    pending = _seed_pending(controller, ctx_old)
    # Round rolled: current key differs.
    _set_current(controller, _context(b"\xbb" * 32))
    controller._handle_result = AsyncMock()

    await controller._maybe_replay_pending()
    controller._handle_result.assert_not_awaited()
    assert controller._pending_submission is None

    # Eviction path drops it too.
    controller._pending_submission = pending
    controller._evict_anticipatory_state(pending.work_key)
    assert controller._pending_submission is None


async def test_replay_failure_requeues(monkeypatch):
    """A replay that fails again on RPC lands back in the pending slot
    with bumped attempts (bounded by the round roll, per policy)."""
    import time as _time
    from types import SimpleNamespace

    controller = _bare_controller()
    ctx = _context(b"\xaa" * 32)
    _set_current(controller, ctx)
    controller.miner_handles = []
    _seed_pending(controller, ctx, attempts=1)
    controller.events = SimpleNamespace(
        last_successful_poll_monotonic=_time.monotonic(),
    )

    async def fake_submit_proof(*args, **kwargs):
        raise ConnectionError("still down")

    monkeypatch.setattr(
        "substrate.miner_controller.submit_proof", fake_submit_proof
    )

    await controller._maybe_replay_pending()

    assert controller._pending_submission is not None
    assert controller._pending_submission.attempts == 2


async def test_fire_suppressed_while_pending(monkeypatch):
    """The anticipatory fire yields to a queued live-submittable result."""
    controller = _bare_controller()
    ctx, key = _seed_cadence_state(controller, valid_at_block=20)

    now = asyncio.get_running_loop().time()
    controller._timing.observe_head(
        block_number=20,
        chain_ts_s=1000.0,
        monotonic_now=now - 100.0,
        wallclock_now=1000.0,
    )

    fired = []

    async def fake_fire(c, k, p, b):
        fired.append((k, b))

    monkeypatch.setattr(controller, "_fire_preview", fake_fire)
    _seed_pending(controller, ctx)

    await controller._maybe_fire_on_cadence()
    assert fired == [], "fire must be suppressed while a pending exists"

    # Sanity: with the pending cleared, the same state fires.
    controller._pending_submission = None
    await controller._maybe_fire_on_cadence()
    assert len(fired) == 1


# ----------------------------------------------------------------------
# consecutive_submit_failures on the anticipatory-fire path (QUI-899)
#
# The gh-20 counter is the signal an operator uses to tell "mining but
# landing nothing" from a healthy miner. It was only ever wired into the
# worker-result path, so a miner stuck in the QUI-899 loop — anticipatory
# fire, RETRY-exhausted, repeat every tick — left it frozen at whatever the
# last worker-path submission set. The counter read healthy while the bug
# was live, which is what made the failure so hard to see from outside.
# ----------------------------------------------------------------------


async def test_retry_exhausted_counts_a_submit_failure(monkeypatch):
    """RETRY-exhausted is a failed submit and must move the gh-20 counter.

    Without this the QUI-899 loop is invisible on /api/v1/status.
    """
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    preview = controller._latest_preview[key]

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.RETRY, attempts=4, error="TimeoutError"
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)
    assert controller.stats.consecutive_submit_failures == 1

    # A miner stuck in this loop keeps climbing, which is what makes the
    # stuck state distinguishable from a quiet-but-healthy one.
    await controller._fire_preview(ctx, key, preview, 21)
    assert controller.stats.consecutive_submit_failures == 2
    assert controller.stats.last_submission_error == "TimeoutError"


async def test_anticipatory_success_clears_the_failure_streak(monkeypatch):
    """A landed anticipatory proof must reset the streak to 0.

    Otherwise a miner whose wins all come through the anticipatory path
    carries a stale non-zero count forever and reads as broken.
    """
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    controller._verify_proof_recorded = AsyncMock(return_value=42)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    preview = controller._latest_preview[key]
    controller.stats.consecutive_submit_failures = 7

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.SUCCESS,
            receipt=ExtrinsicReceipt(extrinsic_hash="0xabc"),
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)

    assert controller.stats.proofs_submitted == 1
    assert controller.stats.consecutive_submit_failures == 0


async def test_stop_fatal_counts_a_submit_failure(monkeypatch):
    """STOP_FATAL must move the counter, matching _handle_result's FATAL
    branch which increments both submission_errors and the streak."""
    controller = _bare_controller()
    _stub_predictor_inputs(controller)
    ctx = _ctx_at_block(b"\xaa" * 32, 19)
    _store_preview_entry(controller, ctx)
    from substrate.miner_controller import _work_key
    key = _work_key(ctx)
    preview = controller._latest_preview[key]

    async def fake_submit_with_retry(*args, **kwargs):
        from substrate.submitter import SubmitResult, SubmitRetryAction
        return SubmitResult(
            action=SubmitRetryAction.STOP_FATAL, error="BadProof"
        )

    monkeypatch.setattr(
        "substrate.miner_controller.submit_with_retry", fake_submit_with_retry
    )

    await controller._fire_preview(ctx, key, preview, 20)

    assert controller.stats.submission_errors == 1
    assert controller.stats.consecutive_submit_failures == 1
