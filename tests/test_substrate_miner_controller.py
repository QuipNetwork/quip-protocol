"""Unit + integration tests for `shared.substrate_miner_controller`.

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
    c.core = None  # Phase 6: optional MinerCore for telemetry
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
# Subscription failover loop (unit)
# ----------------------------------------------------------------------


def _subscribe_controller(subscription_client) -> SubstrateMinerController:
    """Bare controller wired up enough to exercise `_subscribe_heads`."""
    c = _bare_controller()
    c._subscription_client = subscription_client
    c._shutdown_event = asyncio.Event()
    c._latest_head = None
    c._head_signal = asyncio.Event()
    return c


class _FlakySubscriptionClient:
    """Stub `SubstrateClient`-like object whose `subscribe_new_heads` and
    `reconnect` behaviors are scripted per test."""

    def __init__(self, subscribe_outcomes, reconnect_outcomes=None):
        # Each entry is either `None` (return cleanly) or an `Exception`
        # instance to raise. Consumed in order.
        self._subscribe_outcomes = list(subscribe_outcomes)
        self._reconnect_outcomes = list(reconnect_outcomes or [])
        self.subscribe_calls = 0
        self.reconnect_calls = 0

    async def subscribe_new_heads(self, callback):
        self.subscribe_calls += 1
        if not self._subscribe_outcomes:
            raise AssertionError("subscribe called more times than scripted")
        outcome = self._subscribe_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

    async def reconnect(self):
        self.reconnect_calls += 1
        if not self._reconnect_outcomes:
            return
        outcome = self._reconnect_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome


async def test_subscribe_heads_clean_exit_no_failover():
    """A clean subscribe_new_heads return must NOT trigger any failover."""
    sub = _FlakySubscriptionClient(subscribe_outcomes=[None])
    controller = _subscribe_controller(sub)
    await controller._subscribe_heads()
    assert sub.subscribe_calls == 1
    assert sub.reconnect_calls == 0


async def test_subscribe_heads_failover_then_resubscribes():
    """A WebSocketException triggers reconnect, then loop re-subscribes."""
    from websocket import WebSocketConnectionClosedException

    sub = _FlakySubscriptionClient(
        subscribe_outcomes=[
            WebSocketConnectionClosedException("socket closed"),
            None,  # second subscribe call returns cleanly
        ],
    )
    controller = _subscribe_controller(sub)
    await controller._subscribe_heads()
    assert sub.subscribe_calls == 2
    assert sub.reconnect_calls == 1


async def test_subscribe_heads_no_validator_reachable_shuts_down():
    """If reconnect itself can't find a validator, the controller shuts down
    rather than spinning forever."""
    from shared.substrate_client import NoValidatorReachable, ValidatorAttempt
    from websocket import WebSocketConnectionClosedException

    fatal = NoValidatorReachable(
        attempts=[
            ValidatorAttempt(url="ws://a:9944", exc_type="OSError", message="down")
        ]
    )
    sub = _FlakySubscriptionClient(
        subscribe_outcomes=[WebSocketConnectionClosedException("dropped")],
        reconnect_outcomes=[fatal],
    )
    controller = _subscribe_controller(sub)
    shutdown_calls = []
    controller.shutdown = lambda: shutdown_calls.append(True) or controller._shutdown_event.set()
    await controller._subscribe_heads()
    assert shutdown_calls == [True]
    assert sub.subscribe_calls == 1
    assert sub.reconnect_calls == 1
    # The structured failure is recorded for the operator-facing stats blob.
    assert "no validators reachable" in (controller.stats.last_submission_error or "").lower()


async def test_subscribe_heads_non_connection_exception_still_shuts_down():
    """Generic exceptions in subscribe must NOT trigger failover — they're
    bugs, not transient connection loss. Preserves pre-failover behavior."""
    sub = _FlakySubscriptionClient(
        subscribe_outcomes=[RuntimeError("decoder explosion")],
    )
    controller = _subscribe_controller(sub)
    shutdown_calls = []
    controller.shutdown = lambda: shutdown_calls.append(True) or controller._shutdown_event.set()
    await controller._subscribe_heads()
    assert shutdown_calls == [True]
    assert sub.subscribe_calls == 1
    assert sub.reconnect_calls == 0


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

    client = SubstrateClient(url=DEFAULT_URL)
    await client.connect()
    controller = SubstrateMinerController(
        client=client,
        signer=keystore.signer,
        miner_handles=[handle],
        topology_hash=chain_topology_hash,
        core=core,
    )

    run_task = asyncio.create_task(controller.run())
    try:
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
