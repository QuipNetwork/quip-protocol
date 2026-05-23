"""Substrate miner controller — Phase 4 main loop.

Subscribes to new best heads on the substrate node, fetches a fresh
`MiningSnapshot` at each head, dispatches a `SubstrateMiningContext` to every
attached `MinerHandle`, and submits the first valid `MiningResult` back to
the chain as a `QuantumPow.submit_proof` extrinsic.

Lifecycle:

    pool = ValidatorPool(urls=["ws://primary:9944", "ws://standby:9944"])
    controller = SubstrateMinerController(pool, signer, miner_handles)
    try:
        await controller.run()  # blocks until shutdown() or fatal error
    finally:
        controller.shutdown()
        await pool.close()

The controller does not own the lifecycle of its inputs. Construction of the
`ValidatorPool`, `Signer`, and `MinerHandle[]` is the caller's job, as is
their cleanup. Phase 5 will wrap this together with `MinerCore` and the
telemetry API so a single CLI entry point can spin everything up.

Key behaviors:
  - Fail-fast at startup if the signer's account isn't in `QuantumPow.Miners`.
  - Fail-fast if `--topology-hash` is set and doesn't match the snapshot.
  - On every new best head: cancel current work, fetch snapshot at the new
    head hash, dispatch a fresh context to every handle.
  - On every `MiningResult`: encode and submit. Classify the receipt error
    into stale (continue) or fatal (exit) buckets.
  - On shutdown: cancel all handles, drain pending results, return.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue as _queue
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, Tuple

from websocket import WebSocketException

from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.signer import Signer
from substrate.client import NoValidatorReachable, SubstrateClient
from shared.mining_attempt_log import SubmissionLogger
from substrate.submitter import encode_quantum_proof, submit_proof
from substrate.pool import ValidatorPool
from substrate.types import (
    ExtrinsicReceipt,
    SubstrateMiningContext,
)


logger = get_logger("substrate_miner_controller")


class _MinerCoreStats(Protocol):
    """Subset of `shared.miner_core.MinerCore` the controller depends on.

    Structural type rather than direct `MinerCore` import keeps the
    controller usable without a full MinerCore (e.g., a thin test
    double), while still catching signature drift at type-check time.
    """

    def record_dispatch(self) -> None: ...

    def record_result(self, *, winning_miner_id: str, mining_time: float) -> None: ...


# Threshold for consecutive `None` snapshots before the controller raises.
# A few in a row at startup is expected (chain may not be seeded yet); ten
# in a row strongly indicates RPC corruption or a genuinely stuck chain.
_NONE_SNAPSHOT_FAIL_THRESHOLD = 10


class _OperatorFailLoud(RuntimeError):
    """Marker exception for `_handle_head` paths the operator must see.

    Raised explicitly in `_handle_head` for misconfiguration / stuck-chain
    conditions that the controller cannot recover from without operator
    intervention (consecutive-None-snapshot escalation, topology-hash
    mismatch). `_main_loop` re-raises this class but not bare
    `RuntimeError` — the substrate client's `state_call` helpers raise
    plain `RuntimeError` for transient RPC errors, which must remain in
    the drop-and-retry-next-head branch.

    Inherits from `RuntimeError` for compatibility with existing tests
    that use `pytest.raises(RuntimeError)` — the controller's
    `except _OperatorFailLoud` clause still only matches this subclass.
    """


# Cap on how many recently-won work keys we remember. Sized for the
# expected window of late results: a miner with N handles can in the
# worst case have N-1 pending results land after an OK on the same key,
# and the head changes every ~6s so older keys are no longer reachable
# anyway. 16 is generous for the practical N≤8 handle counts.
_CLOSED_WORK_KEYS_CAP = 16


# Per-handle ring buffer of recent dispatch contexts. Indexed by
# dispatch_id; pruned when older than the most-recent N attempts. Sized
# to absorb late results from a few cancelled dispatches without ever
# losing the immutable mapping for an in-flight one.
_DISPATCH_CONTEXT_RETENTION = 4


# A "work key" uniquely identifies the puzzle a context is mining
# against: same (last_proof_block_hash, topology_hash) means the pallet will
# derive the same nonce and therefore the same Ising problem. Used both
# for stale-result detection AND for marking a work item closed/won
# after an accepted submission. Unlike the previous (block_number,
# parent_hash, ...) key, this one only changes when a *round* closes
# (a proof wins), not on every new block — matching the post-fix nonce
# contract.
WorkKey = Tuple[bytes, bytes]


def _work_key(ctx: "SubstrateMiningContext") -> WorkKey:
    return (ctx.last_proof_block_hash, ctx.topology_hash)


# Minimum interval between active head refreshes triggered by the
# "current already-won" branch (same work key, head number strictly
# greater than the accepted block). Without a debounce, every head that
# carries the still-old last_proof_block_hash would kick a refresh —
# fine, but noisy. One per second per closed key is plenty: the round
# rolls over on the next block after the runtime catches up.
_ALREADY_WON_REFRESH_DEBOUNCE_S = 1.0


# Delay between detecting a zero-seed snapshot and the follow-up
# refresh. The zero seed appears transiently at the exact accepted
# block; the runtime repopulates LastProofBlock at block N+1 (~one
# block time). A short delay both (a) avoids tight refresh→handle
# loops while the chain is still on the bad block and (b) gives the
# subscription path room to deliver a fresh wake on its own.
#
# Note: with the post-win fast-forward path (`_post_win_fast_forward`)
# now driving the post-accept transition directly, the zero-seed
# branch no longer needs to schedule its own delayed refresh — the
# fast-forward task polls until the seed has rolled. This constant
# is retained for the rare case where a non-post-win path encounters
# zero-seed; the branch in `_handle_head` returns immediately rather
# than scheduling, and the next subscription/poll wake recovers.
_ZERO_SEED_REFRESH_DELAY_S = 2.0


# Post-win fast-forward: cadence at which we poll the rpc client for
# best head + snapshot to detect the round rolling on chain. 300 ms
# is roughly one block / 20 — fast enough to catch the transition
# within a single block boundary, slow enough that 20 polls per block
# is bounded RPC load.
_POST_WIN_FAST_FORWARD_INTERVAL_S = 0.3

# Post-win fast-forward: bounded deadline. ~10 s covers a full round
# at moderate difficulty without spinning forever if the chain stalls
# or someone else's proof reorganizes the round before ours visibly
# rolls. On timeout we silently fall back to the normal subscription
# path; the stats counter `post_win_fast_forward_timeouts` makes the
# fall-back observable.
_POST_WIN_FAST_FORWARD_TIMEOUT_S = 10.0


# Submission errors that mean "this proof raced a chain state change; drop
# and continue mining the next snapshot". Classification is by `module.Error`
# pallet error name, matched as substring inside the receipt error message
# (substrate-interface formats receipts as `Module(error=Foo, ...)`).
STALE_SUBMISSION_ERRORS = (
    "InvalidNonce",
    "ProofLimitReached",
    "TopologyNotRegistered",
    "InvalidTopology",
)

# Submission errors that indicate something the controller can't recover
# from — bad signing key, deregistered account, bad signed extension.
FATAL_SUBMISSION_ERRORS = (
    "MinerNotRegistered",
    "BadSignature",
    "BadProof",
)


class SubmissionOutcome(str, Enum):
    """Three-way receipt classification.

    Inherits from `str` so existing `==` comparisons against `"ok" / "stale"
    / "fatal"` literals keep working — but typo'd comparisons (e.g.
    `outcome == "oka"`) now fail mypy/ty checks.
    """

    OK = "ok"
    STALE = "stale"
    FATAL = "fatal"


@dataclass
class ControllerStats:
    """Lightweight counters surfaced to telemetry."""

    heads_observed: int = 0
    contexts_dispatched: int = 0
    results_received: int = 0
    proofs_submitted: int = 0
    stale_drops: int = 0
    submission_errors: int = 0
    # Set when a head's mining_snapshot RPC returned None (chain unseeded
    # or RPC corruption). Tracked separately so operators can distinguish
    # "no work right now" from "actively failing submissions".
    none_snapshots_seen: int = 0
    # Last error text from a failed submit_proof RPC, for telemetry.
    last_submission_error: Optional[str] = None
    # Per-handle worker-error counts. Surfaced via {"op": "error"} messages
    # from miner_worker.py so callers can spot handles that consistently
    # crash on dispatch.
    miner_errors: dict[str, int] = field(default_factory=dict)
    # Results dropped because the work key was already won by a prior
    # sibling submission this head. Distinct from stale_drops (head
    # changed) — this counts the storm-prevention path.
    duplicate_result_drops: int = 0
    # Heads observed where the controller skipped dispatch because the
    # last_proof_block_hash had already been won this round. Distinct from
    # duplicate_result_drops (counted per *result*, after mining wasted
    # CPU/GPU time) — this counts the cheap pre-dispatch guard.
    heads_skipped_already_won: int = 0
    # Heads dropped because their block_number was lower than the
    # highest one already handled. Non-zero here means the subscription
    # layer fed us stale historical heads — the symptom that surfaced
    # the per-header-lookup bug in subscribe_new_heads.
    heads_dropped_stale_number: int = 0
    # Heads dropped because `_handle_head` raised a non-connection
    # exception (e.g., snapshot SCALE decode shape drift after a
    # runtime upgrade, RuntimeError from `state_call`). Non-zero here
    # means a head was lost but the controller stayed alive. A
    # persistently growing counter indicates a runtime API shape
    # mismatch that needs an operator update.
    heads_dropped_handler_error: int = 0
    # Submissions that produced an OK receipt but whose post-submit
    # winning_solution check did not match our miner+nonce. Non-zero
    # here means we accepted a receipt that the runtime did not
    # actually record (or recorded against a different submitter).
    proofs_unverified: int = 0
    # Active best-head refreshes the controller initiated outside the
    # subscription cadence (post-win kick, stale-post-win detect, zero-
    # seed snapshot). Pairs with stale_post_win_heads_dropped — every
    # detection should track to at least one refresh.
    heads_refreshed_active: int = 0
    # Subscription delivered a head whose number is <= the block that
    # already accepted our proof for the still-open work key. Means the
    # subscription path is lagging the rpc path; we refresh actively
    # instead of idling on it.
    stale_post_win_heads_dropped: int = 0
    # Snapshot at a non-genesis head carried last_proof_block_hash =
    # 0x00..00. Observed transiently at the exact accepted block; the
    # next block returns the receipt hash. Refusing dispatch here
    # prevents mining a degenerate seed.
    zero_seed_snapshots_dropped: int = 0
    # Gauge — best rpc head number minus latest subscription head number
    # at the moment a stale-post-win was detected. Non-zero means the
    # subscription is genuinely behind; zero with stale_post_win > 0
    # means the lag is at a finer grain than block number.
    subscription_lag_blocks: int = 0
    # Heads where the rpc client reported a strictly newer best than the
    # subscription wake delivered, so `_handle_head` re-anchored to the
    # rpc view before evaluating the snapshot. Counts the lag we
    # routinely paper over rather than just the lag we trip on
    # post-win.
    heads_promoted_to_rpc: int = 0
    # Active refresh responses that came back at or below the caller's
    # `min_block_number` floor (e.g., post-win kick with the rpc client
    # returning a head <= the block we just won at — possible when the
    # rpc client just reconnected to a lagging peer). Counts how often
    # the floor guard saves us.
    heads_refresh_below_floor: int = 0
    # Post-win fast-forwards that successfully observed the next round
    # rolling on chain and woke the main loop with a fresh head. Pairs
    # with `proofs_submitted` — each successful submit should produce
    # one fast-forward in normal operation.
    post_win_fast_forwards: int = 0
    # Post-win fast-forwards that hit the bounded deadline without
    # observing the round roll (e.g., the chain stalled, or someone
    # else's proof reorganized the round). Non-zero is a warning sign
    # but not a failure — the subscription path remains the fallback.
    post_win_fast_forward_timeouts: int = 0
    # Heads observed whose work key matches the controller's current
    # work key. With the round-driven cancel reorder, these no longer
    # trigger cancel+re-dispatch; the in-flight mining attempt is
    # allowed to continue. Counts the cancel churn we no longer waste.
    heads_same_key_skipped: int = 0


@dataclass
class ClosedWorkRecord:
    """Metadata for an already-won work key.

    Stored as the value in `_closed_work_keys` so the controller can
    distinguish a *stale* head (number <= accepted_block_number — the
    subscription is lagging the submission path) from a *current*
    already-won head (number > accepted_block_number — the runtime
    just hasn't rolled the round yet). The first wants an active
    refresh; the second wants a debounced one.

    Why a record (not just the block number): the debounce timer for
    the "current already-won" branch and the `stale_head_skips`
    counter both want per-key locality, and bundling them keeps the
    LRU eviction simple — one cache, one entry per key.
    """

    accepted_block_hash: bytes
    accepted_block_number: int
    closed_at_monotonic: float
    last_already_won_refresh_monotonic: float = 0.0
    stale_head_skips: int = 0


@dataclass
class _ResultEnvelope:
    """Tag a `MiningResult` with the context it was produced against.

    The worker only returns the bare result; the controller knows which
    context was last dispatched and pairs them here so the submitter has
    everything it needs (and so a result against a stale context can be
    discarded if a head change happened mid-mine).

    ``dispatch_id`` is carried through the envelope so submission log
    records can correlate against the worker-side attempt log via
    ``(handle_id, dispatch_id)`` — see ``shared/mining_attempt_log.py``.
    """

    result: MiningResult
    context: SubstrateMiningContext
    handle_id: str
    dispatch_id: int = 0


class SubstrateMinerController:
    """Coordinate mining across one or more `MinerHandle`s against a substrate chain.

    Subscribes to new best heads (via a dedicated `subscription_client`),
    dispatches a fresh `SubstrateMiningContext` to every attached handle on
    each head change, drains `MiningResult`s back, and submits the first
    valid one as a `QuantumPow.submit_proof` extrinsic. See the module
    docstring for the full lifecycle.

    Args:
        pool: ValidatorPool providing slot clients indexed by role. The
            controller asks the pool for `rpc` (for state queries and
            extrinsic submission) and `subscribe.pow` (for the head
            subscription). The pool guarantees distinct clients per
            role, which sidesteps the substrate-interface deadlock
            where a single websocket can't both subscribe and submit.
        signer: Signer for the miner account; must be registered in
            `QuantumPow.Miners` before `run()` is invoked (verified at
            startup).
        miner_handles: One-or-more `MinerHandle`s the controller will
            dispatch work to. Must be non-empty.
        topology_hash: Optional pin. If set, the controller raises
            RuntimeError on the first head whose snapshot reports a
            different topology — prevents an operator from accidentally
            mining against a rotated topology.
        on_proof_submitted: Optional async callback invoked after a
            successful `submit_proof`. Use for telemetry / metrics.
    """

    def __init__(
        self,
        pool: ValidatorPool,
        signer: Signer,
        miner_handles: list[MinerHandle],
        *,
        topology_hash: Optional[bytes] = None,
        on_proof_submitted: Optional[
            Callable[[ExtrinsicReceipt, SubstrateMiningContext], Awaitable[None]]
        ] = None,
        core: Optional[_MinerCoreStats] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "SubstrateMinerController requires at least one MinerHandle"
            )
        self._pool = pool
        # `client` and `_subscription_client` are populated lazily in
        # `run()` — `__init__` must not touch the network. Existing
        # method bodies still reference `self.client` directly; the
        # type annotation is Optional only during the (synchronous)
        # constructor window.
        self.client: Optional[SubstrateClient] = None
        self._subscription_client: Optional[SubstrateClient] = None
        # Optional MinerCore hook. When provided, the controller calls
        # `core.record_dispatch()` once per head (not per handle — that would
        # double-count when more than one miner is attached) and
        # `core.record_result(winning_miner_id, mining_time)` on chain-accepted
        # proofs. Keeps `/api/v1/stats`'s legacy `total_blocks_attempted` /
        # `total_blocks_won` / `wins_per_miner` fields live without coupling
        # the controller's type to MinerCore.
        self.core = core
        # Client slots are pulled from the pool in `run()` — see comment
        # there. We deliberately do NOT touch the network in __init__.
        # The pool guarantees the rpc and subscribe slots are distinct
        # SubstrateClient instances, which sidesteps the substrate-
        # interface deadlock where a single websocket can't both
        # subscribe and submit (receive-mode hold).
        self.signer = signer
        self.miner_handles = miner_handles
        self.topology_hash = topology_hash
        self.on_proof_submitted = on_proof_submitted
        self.stats = ControllerStats()

        self._current_context: Optional[SubstrateMiningContext] = None
        # Per-(handle_id, dispatch_id) immutable context map. Each
        # mine_work_item dispatch installs a fresh entry; the drainer
        # pairs results with the exact context they were dispatched
        # against by looking up (handle_id, dispatch_id) from the
        # response. Without this immutability, the previous
        # `_dispatched[handle_id] = ctx` pattern could surface a result
        # from dispatch N paired with dispatch N+1's context.
        self._dispatch_contexts: dict[
            Tuple[str, int], SubstrateMiningContext
        ] = {}
        # Current "work key" (block_number, parent_hash, topology_hash).
        # Set after a successful snapshot fetch; consulted by the
        # storm-prevention check in _handle_result.
        self._current_work_key: Optional[WorkKey] = None
        # LRU map of work keys for which the chain already accepted one
        # of our proofs this head. Subsequent same-key results from
        # sibling handles are dropped without resubmission — that's the
        # actual fix for submission storming. The value carries the
        # accepted block hash/number so `_handle_head` can distinguish
        # a stale subscription head (number <= accepted_block_number)
        # from a current-round head (number > accepted_block_number);
        # see ClosedWorkRecord. OrderedDict keeps eviction simple
        # (popitem(last=False)) and order-aware.
        self._closed_work_keys: "OrderedDict[WorkKey, ClosedWorkRecord]" = (
            OrderedDict()
        )
        # In-flight guard for _refresh_latest_head. The subscription
        # path can fire multiple stale-post-win events in quick
        # succession (one per stale head delivered); we only need one
        # active refresh outstanding at a time.
        self._refresh_in_flight: bool = False
        # Latest-only head channel. A slow submit_proof can hold the main
        # loop in `_handle_result` for seconds; during that window the
        # chain advances by multiple blocks. We don't want to dispatch
        # against the oldest pending head when we come back — only the
        # newest matters. maxsize=1 + replace-on-write does that.
        self._latest_head: Optional[tuple[bytes, int]] = None
        self._head_signal = asyncio.Event()
        # Highest block_number successfully handled. Belt-and-suspenders
        # against the stale-head backlog bug: even if a regression in
        # SubstrateClient.subscribe_new_heads starts feeding us historical
        # block hashes again, this guard short-circuits them before they
        # can call mining_snapshot at a stale `at` and reopen the
        # already-won round_seed loop.
        self._highest_handled_block: int = 0
        # Last energy threshold (milli) pushed to worker handles via
        # MinerHandle.set_live_threshold_milli(). Used to suppress
        # redundant writes — the snapshot's max_energy_milli only
        # changes when the chain crosses a decay-step boundary (every
        # ``epoch_length`` blocks elapsed since LastProofBlock), so most
        # heads don't need a refresh. Initialised to 0 (never pushed).
        self._last_pushed_threshold_milli: int = 0
        # Per-controller submission log — paired with per-worker attempt
        # logs via (handle_id, dispatch_id). The query API in
        # shared.telemetry_api hits both. assign_id() reserves the
        # next monotonic solution_id; record() writes the outcome.
        self._submission_log = SubmissionLogger()
        self._result_queue: asyncio.Queue[_ResultEnvelope] = asyncio.Queue()
        # Per-handle done-sentinel queue. The worker emits
        # `{"op": "work_item_done"}` when its mining loop exits with no
        # result; the drainer pushes that into the handle's queue and
        # _await_handle_done pops from it. Using a queue (not an event)
        # so multiple done sentinels don't get coalesced into a single
        # set/clear cycle.
        # Per-handle done-sentinel queue. The drainer pushes the
        # response's dispatch_id (not None) so _await_handle_done can
        # block until the *specific* dispatch being cancelled acks.
        self._done_queues: dict[str, asyncio.Queue[int]] = {
            h.miner_id: asyncio.Queue() for h in miner_handles
        }
        self._shutdown_event = asyncio.Event()
        self._drainer_tasks: list[asyncio.Task] = []
        self._subscription_task: Optional[asyncio.Task] = None
        # Running count of consecutive None snapshots; reset on a successful
        # snapshot. Escalates to RuntimeError after _NONE_SNAPSHOT_FAIL_THRESHOLD.
        self._consecutive_none_snapshots = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any thread/task."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main loop. Returns on shutdown or fatal error."""
        # Resolve slot clients from the pool. The pool lazy-connects on
        # first get(), so this is where we actually touch the network —
        # __init__ stays synchronous and side-effect free.
        self.client = await self._pool.get("rpc")
        self._subscription_client = await self._pool.get("subscribe.pow")

        account = self.signer.account_id_bytes()
        await self._verify_registered(account)

        # Drainer tasks consume each handle's resp queue and post into our
        # asyncio result queue. Start them before the first dispatch so we
        # never miss an early result.
        for handle in self.miner_handles:
            task = asyncio.create_task(
                self._drain_handle(handle),
                name=f"drain-{handle.miner_id}",
            )
            self._drainer_tasks.append(task)

        # Start the subscription task. The blocking subscribe loop
        # inside substrate-interface runs on the subscription client's
        # executor — keeping it off the main rpc slot means
        # submit_extrinsic / state_call traffic doesn't deadlock
        # against an active subscription. Both slots are already
        # connected (the pool's get() did that).
        self._subscription_task = asyncio.create_task(
            self._subscribe_heads(),
            name="head-subscription",
        )

        # Prime the loop with the current head so we don't sit idle waiting
        # for the next slot. block_number is informational only here
        # (the head handler re-fetches the snapshot at this hash and uses
        # the snapshot's authoritative block_number).
        head = await self.client.get_head()
        block_number = await self.client.get_block_number(at=head)
        self._latest_head = (head, block_number)
        self._head_signal.set()

        try:
            await self._main_loop()
        finally:
            await self._teardown()

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    async def _verify_registered(self, account: bytes) -> None:
        miner_info = await self.client.query_miner(account)
        if miner_info is None:
            raise RuntimeError(
                f"signer account 0x{account.hex()} is not in "
                "QuantumPow.Miners — run `quip-miner bootstrap` first"
            )
        logger.info(
            "miner verified registered: ss58=%s deposit=%d submitted=%d won=%d",
            self.signer.ss58_address(),
            miner_info.deposit,
            miner_info.proofs_submitted,
            miner_info.proofs_won,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        while not self._shutdown_event.is_set():
            head_task = asyncio.create_task(self._head_signal.wait())
            result_task = asyncio.create_task(self._result_queue.get())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            done, pending = await asyncio.wait(
                [head_task, result_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            # Drain cancellations so they don't surface as warnings. Narrow
            # catch: CancelledError is expected, anything else is a bug
            # (these tasks await an Event/Queue.get/Event.wait — no other
            # exception path is legal).
            for t in pending:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception(
                        "unexpected exception draining pending task %s",
                        t.get_name(),
                    )

            if shutdown_task in done:
                return

            # Process result first when both are ready: `result_task.result()`
            # has already dequeued an envelope from `_result_queue` — if we
            # skipped to the head branch and `continue`d, that envelope
            # would be silently lost. Heads coalesce via `_latest_head`
            # and are cheap to re-handle; results don't.
            if result_task in done:
                envelope = result_task.result()
                try:
                    await self._handle_result(envelope)
                except (WebSocketException, ConnectionError) as exc:
                    # Failover already rotated; the chain will surface
                    # the result again as a stale-drop or accept on the
                    # new validator. Don't crash the controller.
                    self.stats.submission_errors += 1
                    self.stats.last_submission_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.warning(
                        "submit/result handling hit a connection drop "
                        "(%s: %s); failover swung the client, "
                        "controller continues",
                        type(exc).__name__,
                        exc,
                    )

            if head_task in done:
                # Snapshot the latest head and clear the signal atomically
                # so a head arriving mid-handle gets coalesced into the
                # next iteration rather than dropped.
                self._head_signal.clear()
                latest = self._latest_head
                if latest is not None:
                    head_hash, block_number = latest
                    try:
                        await self._handle_head(head_hash, block_number)
                    except (WebSocketException, ConnectionError) as exc:
                        # `SubstrateClient._run` reconnects on these but
                        # re-raises so the caller decides retry vs fail.
                        # The next head (~6s) will land on the new
                        # validator; drop this one rather than tearing
                        # down the controller mid-rotation.
                        logger.warning(
                            "head handling hit a connection drop "
                            "(%s: %s); failover already swung the client, "
                            "next head will retry",
                            type(exc).__name__,
                            exc,
                        )
                    except _OperatorFailLoud:
                        # `_handle_head` raises `_OperatorFailLoud` for
                        # operator-must-see conditions: consecutive-None-
                        # snapshot escalation (chain stuck / RPC broken)
                        # or topology-hash mismatch (misconfiguration).
                        # These must not be swallowed by the generic-drop
                        # branch below — they need to tear the controller
                        # down so the operator sees the configured error
                        # message instead of an infinite log spam.
                        #
                        # Bare `RuntimeError` is NOT caught here on
                        # purpose: substrate-client `state_call` raises
                        # plain RuntimeError on transient RPC errors, and
                        # those must remain in the drop-and-retry-next-
                        # head branch.
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # Non-connection, non-fail-loud errors from
                        # `_handle_head` (e.g. snapshot SCALE decode shape
                        # drift) would otherwise propagate out of
                        # `_main_loop` and tear down the controller —
                        # then `_teardown` runs, `run()` returns, and the
                        # CLI's `asyncio.wait` shuts everything down on a
                        # single runtime-API shape mismatch. Drop the
                        # head, increment a stat, and let the next head
                        # retry. Operators see the counter + log; a
                        # persistently broken runtime shape will show up
                        # as a flatlined `heads_observed` vs. growing
                        # `heads_dropped_handler_error`.
                        self.stats.heads_dropped_handler_error += 1
                        logger.warning(
                            "head handling raised %s: %s; dropping head "
                            "0x%s... and waiting for next",
                            type(exc).__name__,
                            exc,
                            head_hash.hex()[:16],
                        )

    async def _handle_head(self, head_hash: bytes, block_number: int) -> None:
        self.stats.heads_observed += 1

        # Subscription wake → "the chain moved" signal. The subscription
        # client's websocket is in receive mode, and any RPC issued on
        # it (e.g., the pump's own `get_chain_head`) competes with the
        # backlog of pending notification frames — head-of-line
        # blocking. The result, observed in production: subscription
        # delivers a head whose number is 8+ blocks behind what the rpc
        # client sees a few ms later (gauge: `subscription_lag_blocks`).
        #
        # Promote to the rpc client's view before evaluating the
        # snapshot. The rpc socket has no subscription holding it; its
        # `get_head()` returns truly-current state. If the rpc query
        # fails, fall back to the subscription's head_hash — that's
        # what we'd have used anyway and the stale-post-win classifier
        # will catch over-old work keys downstream.
        try:
            rpc_head = await self.client.get_head()
            rpc_number = await self.client.get_block_number(at=rpc_head)
            if rpc_number > block_number:
                # Update the lag gauge here so it reflects every
                # observed gap, not just the ones that trip into the
                # stale-post-win branch below.
                self.stats.subscription_lag_blocks = (
                    rpc_number - block_number
                )
                logger.debug(
                    "promoting head %d → %d (rpc ahead of subscription) "
                    "0x%s... → 0x%s...",
                    block_number,
                    rpc_number,
                    head_hash.hex()[:16],
                    rpc_head.hex()[:16],
                )
                head_hash = rpc_head
                block_number = rpc_number
                self.stats.heads_promoted_to_rpc += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "rpc-head promotion failed (using subscription head): %s: %s",
                type(exc).__name__,
                exc,
            )

        # Stale-block-number guard. A backlog in the subscription layer
        # (or a fork that diverged from canonical before our latest
        # handled block) could feed us an older head — mining_snapshot
        # at that hash would correctly return a stale round_seed and the
        # storm-prevention guard further down would idle us on a
        # closed work_key. Catch it here instead and log loudly.
        #
        # Note: after rpc-head promotion above, `block_number` may be
        # the rpc view, which is strictly >= the subscription's number.
        # The guard still catches genuinely stale subscription heads
        # that the rpc promotion couldn't lift past _highest_handled_block
        # (e.g., a fork rollback).
        if block_number < self._highest_handled_block:
            self.stats.heads_dropped_stale_number += 1
            logger.warning(
                "dropping stale head number=%d (highest_handled=%d) "
                "head=0x%s...; subscription backlog or fork",
                block_number,
                self._highest_handled_block,
                head_hash.hex()[:16],
            )
            return

        # Post-MR-!20: the on-chain `derive_nonce` hashes the SCALE-encoded
        # account ID (`blake2_256(account.encode())`) to produce a width-stable
        # 32-byte miner identity. For sr25519/AccountId32 this is just
        # `blake2_256` of the 32-byte pubkey, but routing through the same
        # helper keeps the contract clear if a wider AccountId is introduced.
        canonical_miner = hashlib.blake2b(
            self.signer.account_id_bytes(), digest_size=32
        ).digest()
        # Evaluate the runtime API at the node's current best block
        # (at=None), not at the head hash we received. The node's view
        # is strictly fresher than any (hash, number) we hold — there's
        # no client→node round-trip between "what's best" and "snapshot
        # at best." This also collapses the window in which the
        # zero-seed transient appears: by the time the node processes
        # state_call, the runtime's LastProofBlock is more likely to
        # have been repopulated.
        #
        # The promoted head/number above is still used for logging,
        # stale-guard, and `_highest_handled_block` bookkeeping —
        # snapshot freshness is a separate axis from head-tracking
        # freshness.
        context = await self.client.get_mining_snapshot(
            at=None,
            topology_hash=self.topology_hash,
            miner_account_bytes=canonical_miner,
        )
        if context is None:
            self.stats.none_snapshots_seen += 1
            self._consecutive_none_snapshots += 1
            logger.warning(
                "mining_snapshot returned None at block %d (head=0x%s); "
                "chain may not be seeded yet (consecutive_none=%d)",
                block_number,
                head_hash.hex()[:16],
                self._consecutive_none_snapshots,
            )
            if self._consecutive_none_snapshots >= _NONE_SNAPSHOT_FAIL_THRESHOLD:
                raise _OperatorFailLoud(
                    f"mining_snapshot returned None for "
                    f"{self._consecutive_none_snapshots} consecutive heads "
                    "— chain may be stuck or RPC is broken. Run "
                    "`quip-miner bootstrap` or check the node."
                )
            return
        self._consecutive_none_snapshots = 0

        if (
            self.topology_hash is not None
            and context.topology_hash != self.topology_hash
        ):
            raise _OperatorFailLoud(
                "configured --topology-hash does not match snapshot: "
                f"expected 0x{self.topology_hash.hex()}, got 0x{context.topology_hash.hex()}"
            )

        # Zero-seed guard. The mining_snapshot at the *exact* accepted
        # block has been observed returning last_proof_block_hash = 0x00..00
        # — a transient runtime state between "proof submitted" and
        # "round rolled" where the storage item is reset but not yet
        # repopulated. Block N+1's snapshot returns the real receipt
        # hash. Dispatching against the zero seed would produce a
        # degenerate work key the chain will reject; refuse and
        # actively refresh instead.
        #
        # The `_highest_handled_block > 0` clause leaves the legitimate
        # genesis/bootstrap path alone (no proof has ever been submitted
        # so a zero last_proof_block_hash is the expected starting state).
        # Push the chain's live decayed energy threshold to every
        # worker handle whenever it has actually changed (i.e. this
        # head crossed an epoch-step decay boundary). The snapshot's
        # `difficulty.max_energy_milli` is computed by
        # `current_difficulty_for(block_number)` on the chain side
        # (lib.rs:679), so what we push here is exactly what the
        # chain would validate against right now — no client-side
        # decay replay needed. The worker reads this in the ratchet
        # path to decide when a stored-best candidate has become
        # eligible for submission. See shared/difficulty_decay.py
        # for the equivalent Python computation (tests / debugging).
        live_threshold = int(context.difficulty.max_energy_milli)
        if live_threshold != self._last_pushed_threshold_milli:
            logger.info(
                "live energy threshold changed: %d → %d milli "
                "(head=%d, last_proof=0x%s...)",
                self._last_pushed_threshold_milli,
                live_threshold,
                block_number,
                context.last_proof_block_hash.hex()[:16],
            )
            for handle in self.miner_handles:
                handle.set_live_threshold_milli(live_threshold)
            self._last_pushed_threshold_milli = live_threshold

        if (
            context.last_proof_block_hash == b"\x00" * 32
            and self._highest_handled_block > 0
        ):
            self.stats.zero_seed_snapshots_dropped += 1
            logger.warning(
                "snapshot at head %d (0x%s...) carries zero "
                "last_proof_block_hash; refusing dispatch "
                "(highest_handled=%d)",
                block_number,
                head_hash.hex()[:16],
                self._highest_handled_block,
            )
            # The post-win fast-forward task (started in _handle_result
            # after every accepted submission) is already polling for
            # the round to roll; it'll wake the main loop with a fresh
            # head as soon as LastProofBlock is repopulated. No need
            # for a per-zero-seed refresh schedule here — multiple
            # mechanisms targeting the same condition just add noise.
            return

        new_work_key = _work_key(context)

        # Closed-work-key branch (runs BEFORE the same-key short-circuit
        # so we preserve detailed logging + refresh triggers for the
        # post-win idle window). If we already won this round, don't
        # redispatch. Split into two sub-cases driven by the accepted
        # block number we recorded at close time:
        #
        #   (1) stale post-win — the subscription delivered a head whose
        #       number is <= the block that accepted our proof. This is
        #       the lag symptom: the subscription path trails the rpc
        #       path. Don't bump _highest_handled_block (a refreshed
        #       head must be allowed past the stale-number guard at the
        #       top of this method) and kick an active refresh.
        #
        #   (2) current already-won — head number is strictly past the
        #       accepted block but the runtime hasn't rolled the round
        #       yet (snapshot still returns the same last_proof_block_hash).
        #       Sit idle as before, but also kick a debounced refresh —
        #       if the runtime rolled between subscription wake and
        #       snapshot fetch, we'll see the new round on the next
        #       refresh-driven dispatch instead of waiting for the
        #       natural subscription cadence.
        record = self._closed_work_keys.get(new_work_key)
        if record is not None:
            self.stats.heads_skipped_already_won += 1
            if block_number <= record.accepted_block_number:
                record.stale_head_skips += 1
                self.stats.stale_post_win_heads_dropped += 1
                logger.info(
                    "stale post-win head number=%d 0x%s... "
                    "(accepted at block %d, last_proof_block_hash=0x%s...); "
                    "refreshing",
                    block_number,
                    head_hash.hex()[:16],
                    record.accepted_block_number,
                    context.last_proof_block_hash.hex()[:16],
                )
                # Sample subscription lag once per stale detection.
                await self._record_subscription_lag(block_number)
                await self._refresh_latest_head(
                    "stale post-win head",
                    min_block_number=record.accepted_block_number,
                )
                return
            now = time.monotonic()
            since_last = now - record.last_already_won_refresh_monotonic
            logger.info(
                "head number=%d 0x%s... carries already-won "
                "last_proof_block_hash=0x%s... (accepted at block %d); "
                "waiting for next round (skipping dispatch)",
                block_number,
                head_hash.hex()[:16],
                context.last_proof_block_hash.hex()[:16],
                record.accepted_block_number,
            )
            # Still a successful head observation — bump so a backlog of
            # older heads doesn't slip past the stale-number guard.
            self._highest_handled_block = block_number
            if since_last >= _ALREADY_WON_REFRESH_DEBOUNCE_S:
                record.last_already_won_refresh_monotonic = now
                await self._refresh_latest_head(
                    "current already-won head",
                    min_block_number=record.accepted_block_number,
                )
            return

        # Same-key short-circuit (round-driven controller). If the
        # snapshot's work key matches what we're already mining, the
        # round hasn't rolled — don't cancel the in-flight mining
        # attempt just because a new chain block arrived. PoW work
        # only changes when last_proof_block_hash changes; same key
        # means same Ising problem, and a fresh dispatch would be
        # redundant. The in-flight result will arrive via _result_queue
        # when mining completes; head wakes that don't change the
        # round are a no-op for the mining pipeline.
        #
        # This guards both (a) the steady-state case where a miner
        # finds a valid solution after several block-wakes, and
        # (b) the high-difficulty case where a long mining attempt
        # spans many same-round head wakes — cancel-on-every-head
        # would truncate the attempt every ~6 s and never let SA
        # converge.
        if new_work_key == self._current_work_key:
            # Defense-in-depth: the short-circuit assumes a same-key head
            # means mining is already in flight. That holds in the happy
            # path, but any `_handle_result` early-return that leaves
            # `_current_work_key` set without re-dispatching (verify
            # mismatch, RPC submit error, future code paths) will trip
            # this branch with every handle idle and silently freeze the
            # miner until something else rolls the round. Detect that
            # exact state and re-dispatch instead of returning. The
            # primary verify-mismatch path re-dispatches inline (see
            # `_handle_result`); this is the backstop for the rest.
            all_idle = all(
                h._active_dispatch_id == 0 for h in self.miner_handles
            )
            if not all_idle:
                self.stats.heads_same_key_skipped += 1
                self._highest_handled_block = block_number
                return
            logger.warning(
                "same-key head with all handles idle (work_key=0x%s..., "
                "head=%d) — re-dispatching same context to resume mining",
                new_work_key[0].hex()[:16],
                block_number,
            )
            # Fall through to the dispatch block below so the same code
            # path handles redispatch. `_current_context` is still the
            # context for this key, so the loop below will use it.

        # Work key changed — cancel any in-flight mining on the *prior*
        # key, then wait in parallel for each handle to ack the cancel.
        # Without this synchronization, cancel() → clear() → dispatch
        # can wipe a cancel before the worker observes it, leaving the
        # worker mining the OLD context against a stop_event tied to
        # the NEW dispatch. The worker emits a `work_item_done`
        # sentinel (tagged with the dispatch_id) when its loop exits
        # with no result; the drainer surfaces that via
        # _await_handle_done so we confirm cancellation of the
        # *specific* dispatch we issued before re-clearing the event.
        #
        # Parallel wait (asyncio.gather) keeps total stall to ~0.5s
        # regardless of handle count — sequential waits would block
        # for N×0.5s, significant against a 6s block time.
        #
        # Reordered (was: cancel on every head): the round-driven
        # short-circuit above means we only reach here when the work
        # key actually changed, so cancelling stale-round mining is
        # both necessary and correct.
        cancelled_dispatches: list[tuple[MinerHandle, int]] = []
        for handle in self.miner_handles:
            if handle._active_dispatch_id != 0:
                cancelled_dispatches.append(
                    (handle, handle._active_dispatch_id)
                )
                handle.cancel()
        if cancelled_dispatches:
            await asyncio.gather(
                *[
                    self._await_handle_done(
                        h, dispatch_id=did, timeout=0.5,
                    )
                    for h, did in cancelled_dispatches
                ]
            )

        self._current_context = context
        self._current_work_key = new_work_key

        logger.info(
            "new head: number=%d head=0x%s... last_proof_block_hash=0x%s... "
            "topology=0x%s... nodes=%d edges=%d",
            block_number,
            head_hash.hex()[:16],
            context.last_proof_block_hash.hex()[:16],
            context.topology_hash.hex()[:16],
            len(context.nodes),
            len(context.edges),
        )

        # Dispatch the new context to every handle. Each one will generate
        # its own salts and explore the search space independently. The
        # dispatch_id returned by mine_work_item keys an immutable
        # (handle_id, dispatch_id) → context map so late results from a
        # cancelled dispatch can still find their original context.
        for handle in self.miner_handles:
            dispatch_id = handle.mine_work_item(context)
            self._dispatch_contexts[(handle.miner_id, dispatch_id)] = context
            self._prune_dispatch_contexts(handle.miner_id, dispatch_id)
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            # One attempt per head, regardless of how many handles fanned out.
            self.core.record_dispatch()
        # Bump after the full body completes so an early-return path
        # (None snapshot, topology mismatch, etc.) doesn't lock out
        # retries at the same block number.
        self._highest_handled_block = block_number

    async def _handle_result(self, envelope: _ResultEnvelope) -> None:
        self.stats.results_received += 1
        envelope_key = _work_key(envelope.context)

        # Storm prevention: if we already submitted an accepted proof
        # for this work key this head, drop sibling/duplicate results
        # without re-submitting. Distinct from the stale check below —
        # there the head has moved on; here the head is still current
        # but we've already won it. Without this, every additional
        # handle for the same context fires another submit_proof and
        # spams the chain (and our RPC) with redundant work.
        if envelope_key in self._closed_work_keys:
            self.stats.duplicate_result_drops += 1
            logger.info(
                "dropping duplicate result from %s: work_key already won "
                "(last_proof_block_hash=0x%s...)",
                envelope.handle_id,
                envelope.context.last_proof_block_hash.hex()[:16],
            )
            return

        # Drop results produced against a stale context. The controller
        # already moved on to a new round; the chain would reject this
        # proof. topology_hash is included so a governance call that
        # rotates the topology within a single round doesn't slip through
        # as a round-seed match.
        if (
            self._current_work_key is None
            or envelope_key != self._current_work_key
        ):
            self.stats.stale_drops += 1
            current_seed_hex = (
                "0x" + self._current_work_key[0].hex()[:16] + "..."
                if self._current_work_key is not None
                else "<none>"
            )
            logger.info(
                "dropping stale result from %s: last_proof_block_hash=0x%s... "
                "(current=%s)",
                envelope.handle_id,
                envelope.context.last_proof_block_hash.hex()[:16],
                current_seed_hex,
            )
            return

        # Encoder errors (ValueError on no-solutions / wrong-salt-length)
        # are code defects — retrying won't help. Keep them out of the
        # RPC-error catch below so they raise loudly instead of being
        # logged-and-dropped like a transient network blip.
        try:
            encode_quantum_proof(envelope.result, envelope.context)
        except ValueError as exc:
            raise RuntimeError(f"proof encoding failed (bug): {exc}") from exc

        # Reserve a solution_id before the network round-trip so that
        # even an RPC failure produces a log entry with a stable id.
        solution_id = self._submission_log.assign_id()
        result_energy_milli = int(envelope.result.energy * 1000)
        result_diversity_milli = int(envelope.result.diversity * 1000)
        snapshot_threshold_milli = int(envelope.context.difficulty.max_energy_milli)
        last_proof_hex = "0x" + envelope.context.last_proof_block_hash.hex()

        try:
            receipt = await submit_proof(
                self.client, self.signer, envelope.result, envelope.context
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.submission_errors += 1
            self.stats.last_submission_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "submit_proof RPC failed for result from %s: %s",
                envelope.handle_id,
                exc,
            )
            self._submission_log.record(
                solution_id=solution_id,
                miner_id=envelope.handle_id,
                dispatch_id=envelope.dispatch_id,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="chain_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        outcome = classify_submission(receipt)
        if outcome is SubmissionOutcome.OK:
            # Verify the runtime actually recorded *our* proof before
            # closing the work key. Without this, a receipt that looked
            # OK (made it into a block, no obvious ExtrinsicFailed) but
            # whose proof the runtime silently rejected leaves us idling
            # on `_closed_work_keys` forever while the chain keeps
            # advancing the round on someone else's proof. The check
            # reads `LastProofBlock` and `winning_solution(N)` and
            # matches miner+nonce against the receipt.
            #
            # Best-effort: RPC errors during verification only WARN —
            # failing closed here would cause the same submission storm
            # we just stopped, since the next head would re-dispatch
            # this exact context. Persistent failures stay visible via
            # stats.proofs_unverified.
            verified = await self._verify_proof_recorded(envelope)
            if verified is False:
                self.stats.proofs_unverified += 1
                self._submission_log.record(
                    solution_id=solution_id,
                    miner_id=envelope.handle_id,
                    dispatch_id=envelope.dispatch_id,
                    energy_milli=result_energy_milli,
                    diversity_milli=result_diversity_milli,
                    threshold_milli=snapshot_threshold_milli,
                    last_proof_block_hash_hex=last_proof_hex,
                    outcome="chain_error",
                    extrinsic_hash=receipt.extrinsic_hash,
                    chain_block_hash=receipt.block_hash,
                    error="receipt OK but proof not recorded by chain",
                )
                # Re-dispatch immediately on the same context. The
                # worker is idle (mine_result drainer already cleared
                # `_active_dispatch_id`), and waiting for the next head
                # would deadlock: the verify-mismatch path leaves
                # `_current_work_key` set, so every subsequent head
                # carrying the same `last_proof_block_hash` falls into
                # the same-key short-circuit at the top of
                # `_handle_head` and skips dispatch. Without an active
                # re-dispatch here, the worker sits idle until someone
                # else wins the round and rolls `LastProofBlock`.
                if (
                    envelope_key == self._current_work_key
                    and self._current_context is not None
                ):
                    redispatch_context = self._current_context
                    redispatched: list[str] = []
                    for handle in self.miner_handles:
                        if handle._active_dispatch_id != 0:
                            continue
                        new_dispatch_id = handle.mine_work_item(redispatch_context)
                        self._dispatch_contexts[
                            (handle.miner_id, new_dispatch_id)
                        ] = redispatch_context
                        self._prune_dispatch_contexts(
                            handle.miner_id, new_dispatch_id,
                        )
                        redispatched.append(handle.miner_id)
                    if redispatched:
                        self.stats.contexts_dispatched += len(redispatched)
                    logger.warning(
                        "submit_proof receipt OK but verification failed for "
                        "%s (extrinsic=%s block=%s); NOT closing work key — "
                        "re-dispatched %d handle(s) on same context: %s",
                        envelope.handle_id,
                        receipt.extrinsic_hash,
                        receipt.block_hash,
                        len(redispatched),
                        redispatched,
                    )
                else:
                    # Work key already rolled while we were verifying —
                    # the natural head path will dispatch the new round.
                    logger.warning(
                        "submit_proof receipt OK but verification failed for "
                        "%s (extrinsic=%s block=%s); work key already "
                        "advanced (envelope_key matches=%s, current_context=%s) — "
                        "deferring to next head",
                        envelope.handle_id,
                        receipt.extrinsic_hash,
                        receipt.block_hash,
                        envelope_key == self._current_work_key,
                        self._current_context is not None,
                    )
                return
            self.stats.proofs_submitted += 1
            # Resolve the receipt's block hash → block number so the
            # work-key record can distinguish stale subscription heads
            # (number <= accepted) from current already-won heads. The
            # receipt only carries `block_hash` (hex string); we fetch
            # the number via the shallow header read.
            #
            # Best-effort: a failure here would leave us without a
            # ground-truth number, so we fall back to `_highest_handled_block + 1`.
            # The fallback is correct in the common case (we won the
            # round at or just past the most recent head we processed)
            # and the subscription will re-converge within one or two
            # heads even if it's slightly off.
            accepted_block_hash = b""
            accepted_block_number = self._highest_handled_block + 1
            if receipt.block_hash:
                try:
                    accepted_block_hash = bytes.fromhex(
                        receipt.block_hash[2:]
                        if receipt.block_hash.startswith("0x")
                        else receipt.block_hash
                    )
                    accepted_block_number = await self.client.get_block_number(
                        at=accepted_block_hash
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "could not resolve accepted block number for "
                        "receipt block=%s (%s: %s); using fallback=%d",
                        receipt.block_hash,
                        type(exc).__name__,
                        exc,
                        accepted_block_number,
                    )
            record = ClosedWorkRecord(
                accepted_block_hash=accepted_block_hash,
                accepted_block_number=accepted_block_number,
                closed_at_monotonic=time.monotonic(),
            )
            # Mark this work key as won and cancel sibling handles
            # immediately so they don't keep mining (and then submitting
            # redundant proofs the chain will reject). This is the
            # primary submission-storm fix; the duplicate-drop check at
            # the top of _handle_result is the belt-and-suspenders
            # backstop for sibling results that were already in flight
            # when we got here.
            self._mark_work_key_closed(envelope_key, record)
            self._cancel_siblings_for_won_work(envelope.handle_id)
            # Start a fast-forward task: poll the rpc client every
            # ~300 ms until the snapshot's last_proof_block_hash equals
            # our accepted hash (the round has visibly rolled), then
            # wake the main loop with a fresh head. Bypasses the
            # subscription path's lag and also handles the zero-seed
            # transient at the accepted block (same condition: round
            # hasn't rolled yet). Bounded by
            # _POST_WIN_FAST_FORWARD_TIMEOUT_S so a stalled chain
            # doesn't spin forever.
            asyncio.create_task(
                self._post_win_fast_forward(
                    record.accepted_block_hash,
                    record.accepted_block_number,
                ),
                name="post-win-fast-forward",
            )
            logger.info(
                "submit_proof accepted: extrinsic=%s block=%s number=%d miner=%s",
                receipt.extrinsic_hash,
                receipt.block_hash,
                accepted_block_number,
                self.signer.ss58_address(),
            )
            self._submission_log.record(
                solution_id=solution_id,
                miner_id=envelope.handle_id,
                dispatch_id=envelope.dispatch_id,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="submitted_inblock",
                extrinsic_hash=receipt.extrinsic_hash,
                chain_block_hash=receipt.block_hash,
                chain_block_number=accepted_block_number,
            )
            if self.core is not None:
                self.core.record_result(
                    winning_miner_id=envelope.result.miner_id,
                    mining_time=float(envelope.result.mining_time),
                )
            if self.on_proof_submitted is not None:
                try:
                    await self.on_proof_submitted(receipt, envelope.context)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "on_proof_submitted callback raised (proof was submitted): "
                        "%s: %s",
                        type(exc).__name__,
                        exc,
                    )
        elif outcome is SubmissionOutcome.STALE:
            self.stats.stale_drops += 1
            logger.info(
                "submit_proof dropped as stale: error=%s extrinsic=%s",
                receipt.error,
                receipt.extrinsic_hash,
            )
            self._submission_log.record(
                solution_id=solution_id,
                miner_id=envelope.handle_id,
                dispatch_id=envelope.dispatch_id,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="rejected_stale",
                extrinsic_hash=receipt.extrinsic_hash,
                error=str(receipt.error or ""),
            )
        else:  # FATAL
            self.stats.submission_errors += 1
            self.stats.last_submission_error = str(receipt.error or "")
            self._submission_log.record(
                solution_id=solution_id,
                miner_id=envelope.handle_id,
                dispatch_id=envelope.dispatch_id,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="chain_error",
                extrinsic_hash=receipt.extrinsic_hash,
                error=str(receipt.error or ""),
            )
            raise RuntimeError(
                f"submit_proof failed fatally: error={receipt.error} "
                f"extrinsic={receipt.extrinsic_hash}"
            )

    async def _await_handle_done(
        self,
        handle: MinerHandle,
        *,
        dispatch_id: int,
        timeout: float,
    ) -> None:
        """Wait briefly for the worker to confirm cancellation.

        Pops sentinels from the handle's done queue until one carrying
        the *expected* dispatch_id arrives, or the timeout fires. A
        sentinel for an older dispatch (one we already moved past)
        is discarded silently — it can show up when the previous
        cancel raced its own sentinel and the new dispatch is now
        cleaning up.

        On timeout we proceed anyway — but log a WARNING because the
        worker did NOT ack the cancel of this specific dispatch in
        time, so the next dispatch's stop_event.clear() may race
        the prior cancel.
        """
        sentinel_queue = self._done_queues.get(handle.miner_id)
        if sentinel_queue is None:
            return
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    "handle %s did not ack cancel of dispatch_id=%d within "
                    "%.1fs; dispatching anyway — next mine_work_item may "
                    "race the prior cancel",
                    handle.miner_id,
                    dispatch_id,
                    timeout,
                )
                return
            try:
                got = await asyncio.wait_for(
                    sentinel_queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "handle %s did not ack cancel of dispatch_id=%d within "
                    "%.1fs; dispatching anyway — next mine_work_item may "
                    "race the prior cancel",
                    handle.miner_id,
                    dispatch_id,
                    timeout,
                )
                return
            # Match exactly — older dispatch sentinels arriving here are
            # already-resolved and can be discarded. A sentinel for a
            # dispatch newer than the one we're waiting for is logically
            # impossible (we haven't issued one yet on this handle).
            if got is None or got == dispatch_id:
                return

    async def _verify_proof_recorded(self, envelope: _ResultEnvelope) -> Optional[bool]:
        """Confirm the runtime recorded this miner's proof on chain.

        Returns:
          - ``True`` if ``QuantumPow.LastProofBlock`` resolves to a
            ``winning_solution`` whose ``miner`` and ``nonce`` match the
            envelope's submission.
          - ``False`` if a winning_solution was returned but does NOT
            match us (someone else won this round, or the chain rolled
            back our submission silently).
          - ``None`` if the verification RPC itself failed — caller
            should treat as inconclusive and proceed with closing the
            work key (the alternative is a submission-storm loop, which
            is worse than a single false-positive close).
        """
        try:
            last_block = await self.client.query_last_proof_block_number()
            if last_block is None:
                logger.warning(
                    "post-OK verify: LastProofBlock is unset — chain may "
                    "not have committed our proof yet (will trust receipt)"
                )
                return None
            winning = await self.client.query_winning_solution(last_block)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post-OK verify RPC failed (%s: %s); proceeding with "
                "close-on-receipt fallback",
                type(exc).__name__,
                exc,
            )
            return None
        if winning is None:
            logger.warning(
                "post-OK verify: winning_solution(%d) returned None; "
                "chain reported LastProofBlock but no winning solution",
                last_block,
            )
            return False
        # `winning.solution.miner` is the chain's AccountId32 of the
        # submitter — that's the raw 32-byte pubkey, NOT the
        # blake2_256-hashed canonical_miner we put into the snapshot
        # context for derive_nonce. Compare against the signer's
        # account_id_bytes directly.
        expected_miner = self.signer.account_id_bytes()
        expected_nonce = envelope.result.nonce
        actual_miner = bytes(winning.solution.miner)
        actual_nonce = bytes(winning.nonce)
        if actual_miner == expected_miner and actual_nonce == expected_nonce:
            return True
        logger.warning(
            "post-OK verify mismatch at block %d: "
            "expected miner=0x%s... nonce=0x%s..., "
            "actual miner=0x%s... nonce=0x%s...",
            last_block,
            expected_miner.hex()[:16],
            expected_nonce.hex()[:16],
            actual_miner.hex()[:16],
            actual_nonce.hex()[:16],
        )
        return False

    async def _refresh_latest_head(
        self,
        reason: str,
        *,
        min_block_number: Optional[int] = None,
    ) -> None:
        """Actively re-poll the rpc client for best head.

        The controller is otherwise reactive to subscription
        deliveries, which suffer head-of-line blocking on a congested
        subscription socket — observed in production: notifications
        arriving 30–90 s late. This helper feeds `_latest_head` +
        `_head_signal` out-of-band so we don't sit idle waiting for
        the next natural wake.

        Uses the **rpc client** (`self.client`), not the subscription
        client. An earlier version polled the subscription client to
        avoid contending with `submit_extrinsic` traffic — but the
        subscription socket is the very source of lag we're escaping,
        so polling it returned heads as stale as the subscription
        itself (e.g., `number=18` when the chain was at 29). The rpc
        socket is uncontended modulo brief `submit_extrinsic` windows
        (~3 s, every ~5 s); refresh polls are sub-second and
        serialize harmlessly behind any in-flight submit via
        `_call_lock`.

        `min_block_number` lets the caller reject rpc responses that
        couldn't possibly be correct. The known case: post-win kick,
        where the rpc client may have just reconnected to a lagging
        validator peer (no per-call freshness validation in
        `SubstrateClient.get_head()`); we should never accept a head
        number `<=` the block our submission was already accepted in.

        Each rpc call is wrapped in `wait_for(1.0)` — an unreachable
        rpc must fail fast so the post-win flow doesn't hang.

        A monotonic guard prevents a slow polled response from
        overwriting a newer head delivered by the subscription pump
        in the meantime. `_refresh_in_flight` debounces overlapping
        calls — the subscription path can fire many stale-post-win
        events in rapid succession; one refresh is enough.
        """
        if self.client is None:
            return
        if self._refresh_in_flight:
            return
        self._refresh_in_flight = True
        try:
            try:
                head = await asyncio.wait_for(self.client.get_head(), 1.0)
                number = await asyncio.wait_for(
                    self.client.get_block_number(at=head), 1.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "active head refresh (%s) timed out after 1.0s; "
                    "rpc unreachable — will retry on next trigger",
                    reason,
                )
                return
            except Exception as exc:  # noqa: BLE001
                # Best-effort — failing here would not change behavior,
                # the next subscription wake will eventually arrive.
                logger.warning(
                    "active head refresh (%s) failed: %s: %s",
                    reason,
                    type(exc).__name__,
                    exc,
                )
                return
            if min_block_number is not None and number <= min_block_number:
                # Floor guard: the caller asserted "any head number <=
                # this is impossible." Reject as a lagging-peer response.
                self.stats.heads_refresh_below_floor += 1
                logger.info(
                    "active head refresh (%s): polled number=%d <= floor=%d; "
                    "rpc peer is lagging — ignoring response",
                    reason,
                    number,
                    min_block_number,
                )
                return
            prev = self._latest_head
            if prev is not None and number < prev[1]:
                logger.debug(
                    "active head refresh (%s): polled number=%d < latest=%d; "
                    "ignoring (monotonic guard)",
                    reason,
                    number,
                    prev[1],
                )
                return
            # Same-head no-op: avoid re-signaling the main loop with a
            # head it just processed. Without this, a zero-seed
            # detection at block N triggers refresh → polls rpc → still
            # N → re-signals → main loop processes N again → zero-seed
            # again → tight loop until the chain advances.
            # Different hash at the same number is a fork swap — treat
            # as new and re-process.
            if prev is not None and prev == (head, number):
                logger.debug(
                    "active head refresh (%s): polled head identical to "
                    "current latest (%d 0x%s...); skipping re-signal",
                    reason,
                    number,
                    head.hex()[:16],
                )
                return
            self._latest_head = (head, number)
            self._head_signal.set()
            self.stats.heads_refreshed_active += 1
            logger.info(
                "active head refresh (%s): number=%d head=0x%s...",
                reason,
                number,
                head.hex()[:16],
            )
        finally:
            self._refresh_in_flight = False

    async def _post_win_fast_forward(
        self, accepted_hash: bytes, accepted_number: int
    ) -> None:
        """Drive the next dispatch immediately after our proof is accepted.

        After `submit_proof` succeeds at block N, the controller wants
        to start mining the next round as soon as the runtime visibly
        rolls (LastProofBlock = our receipt hash). The naive path waits
        for the subscription to deliver a fresh head, then `_handle_head`
        snapshots, then dispatches. With subscription lag of 30+ seconds
        in production, that wait dominates the inter-proof gap.

        This task bypasses the subscription entirely: poll the rpc
        client every ~300 ms for `get_head` + `get_block_number` +
        `get_mining_snapshot(at=None)` until the snapshot's
        `last_proof_block_hash` equals `accepted_hash`. The moment the
        round rolls, update `_latest_head` + signal main loop. Main
        loop's `_handle_head` then snapshots once more (at its own
        cadence) and dispatches.

        Bounded by `_POST_WIN_FAST_FORWARD_TIMEOUT_S`. On timeout we
        log and fall back to the subscription path — the chain may
        have stalled, our proof may have been reorganized, or someone
        else's proof beat ours into a different ordering. Either way,
        spinning forever helps nothing.

        Also subsumes the prior delayed-refresh-on-zero-seed pattern:
        the zero-seed transient is exactly the "round hasn't rolled
        yet" condition the loop is waiting on, so the same poll handles
        both cases with one mechanism.
        """
        if self.client is None:
            return
        start_monotonic = time.monotonic()
        deadline = start_monotonic + _POST_WIN_FAST_FORWARD_TIMEOUT_S
        poll_count = 0
        canonical_miner = hashlib.blake2b(
            self.signer.account_id_bytes(), digest_size=32
        ).digest()
        while (
            time.monotonic() < deadline
            and not self._shutdown_event.is_set()
        ):
            await asyncio.sleep(_POST_WIN_FAST_FORWARD_INTERVAL_S)
            poll_count += 1
            try:
                head = await asyncio.wait_for(self.client.get_head(), 1.0)
                number = await asyncio.wait_for(
                    self.client.get_block_number(at=head), 1.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception:  # noqa: BLE001 — best-effort poll
                continue
            if number <= accepted_number:
                # rpc peer still lagging the block we won at — keep polling.
                continue
            try:
                ctx = await asyncio.wait_for(
                    self.client.get_mining_snapshot(
                        at=None,
                        topology_hash=self.topology_hash,
                        miner_account_bytes=canonical_miner,
                    ),
                    2.0,
                )
            except asyncio.TimeoutError:
                continue
            except Exception:  # noqa: BLE001 — best-effort poll
                continue
            if ctx is None or ctx.last_proof_block_hash != accepted_hash:
                # Round hasn't visibly rolled to our hash yet. Could
                # be (a) the transient zero-seed window at the exact
                # accepted block, (b) the brief gap before the runtime
                # writes LastProofBlock = our hash, or (c) someone
                # else's proof reorganized the round. We don't try to
                # distinguish here — case (c) is rare, and the bounded
                # deadline below caps how long we'll poll wastefully.
                # On timeout the subscription path takes over.
                continue
            # New round visible with our accepted hash. Wake main loop.
            self._latest_head = (head, number)
            self._head_signal.set()
            self.stats.post_win_fast_forwards += 1
            elapsed = time.monotonic() - start_monotonic
            logger.info(
                "post-win fast-forward: round rolled at block %d after "
                "%d polls (%.1f s); seed=0x%s...",
                number,
                poll_count,
                elapsed,
                accepted_hash.hex()[:16],
            )
            return
        self.stats.post_win_fast_forward_timeouts += 1
        logger.warning(
            "post-win fast-forward timed out after %.1f s (polls=%d, "
            "accepted_block=%d, accepted_hash=0x%s...); falling back to "
            "subscription path",
            _POST_WIN_FAST_FORWARD_TIMEOUT_S,
            poll_count,
            accepted_number,
            accepted_hash.hex()[:16],
        )

    async def _delayed_refresh(self, delay: float, reason: str) -> None:
        """Fire `_refresh_latest_head(reason)` after `delay` seconds.

        Used by the zero-seed branch: the zero seed appears at the
        exact accepted block and clears once the chain advances. An
        immediate refresh would just re-poll the same bad block and
        loop. A short delay aligns the retry with the next block.
        """
        await asyncio.sleep(delay)
        await self._refresh_latest_head(reason)

    async def _record_subscription_lag(self, subscription_number: int) -> None:
        """Update `stats.subscription_lag_blocks` once per detection.

        Called from the stale-post-win branch only — we don't poll on
        a timer, the gauge is best sampled exactly when we already
        know the subscription is behind. Uses the rpc client (not the
        subscription client) so the measurement reflects the
        path-divergence we actually care about.
        """
        if self.client is None:
            return
        try:
            rpc_head = await self.client.get_head()
            rpc_number = await self.client.get_block_number(at=rpc_head)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "subscription lag sampling failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return
        lag = max(0, rpc_number - subscription_number)
        self.stats.subscription_lag_blocks = lag

    def _mark_work_key_closed(
        self, key: WorkKey, record: ClosedWorkRecord
    ) -> None:
        """Record a work key as won and evict the oldest if over cap.

        `record` carries the accepted block hash/number so
        `_handle_head` can distinguish stale subscription heads from
        current-round already-won heads.
        """
        # Move-to-end semantics: if the same key shows up twice (shouldn't,
        # but be defensive) it stays at the back of the LRU. We overwrite
        # the record with the newer one — its accepted_block_number is
        # by definition >= the previous one.
        if key in self._closed_work_keys:
            self._closed_work_keys[key] = record
            self._closed_work_keys.move_to_end(key)
        else:
            self._closed_work_keys[key] = record
            while len(self._closed_work_keys) > _CLOSED_WORK_KEYS_CAP:
                self._closed_work_keys.popitem(last=False)

    def _cancel_siblings_for_won_work(self, winning_handle_id: str) -> None:
        """Cancel all handles other than the winner.

        Mirrors mempool_miner_controller.py's pattern: once one
        submission has been accepted, sibling handles mining the same
        work item should stop immediately. They may already have a
        valid result in flight — that's fine, the duplicate-drop check
        catches it at the top of _handle_result.
        """
        for handle in self.miner_handles:
            if handle.miner_id == winning_handle_id:
                continue
            try:
                handle.cancel()
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "sibling cancel failed for %s (winner=%s): %s: %s",
                    handle.miner_id,
                    winning_handle_id,
                    type(exc).__name__,
                    exc,
                )

    def _prune_dispatch_contexts(
        self, handle_id: str, latest_dispatch_id: int
    ) -> None:
        """Trim per-handle dispatch contexts to the last N attempts.

        Retains a small ring so late results from a cancelled dispatch
        can still find their context, but drops anything older than
        _DISPATCH_CONTEXT_RETENTION attempts back — those are stale
        beyond any plausible late-result window.
        """
        cutoff = latest_dispatch_id - _DISPATCH_CONTEXT_RETENTION
        if cutoff <= 0:
            return
        stale = [
            (h, d) for (h, d) in self._dispatch_contexts
            if h == handle_id and d <= cutoff
        ]
        for key in stale:
            self._dispatch_contexts.pop(key, None)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _drain_handle(self, handle: MinerHandle) -> None:
        """Pull from a handle's response queue and push into the controller.

        substrate-interface uses a blocking `mp.Queue.get`, so we run it in
        the default executor and shuttle results into an asyncio queue the
        main loop can `await` on.
        """
        loop = asyncio.get_running_loop()
        try:
            await self._drain_handle_loop(handle, loop)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            # An unhandled exception in the drainer would otherwise just
            # log via asyncio's unhandled-task hook and the main loop
            # would keep waiting on a queue nothing pushes to. Fail loud:
            # the controller can't make progress without this handle's
            # results, so trigger shutdown.
            logger.exception(
                "drainer for %s crashed; triggering shutdown",
                handle.miner_id,
            )
            self.shutdown()

    async def _drain_handle_loop(
        self, handle: MinerHandle, loop: asyncio.AbstractEventLoop
    ) -> None:
        while not self._shutdown_event.is_set():
            try:
                msg = await loop.run_in_executor(None, handle.resp.get, True, 0.5)
            except _queue.Empty:
                # Timed-out get; the common case. Loop and retry.
                continue
            except (EOFError, BrokenPipeError, OSError) as exc:
                # The worker process is gone or the pipe is broken —
                # without escalating, the controller would spin forever
                # against a dead queue. Trigger shutdown so the operator
                # sees a clean failure rather than "miner stopped
                # producing for unknown reason".
                logger.error(
                    "handle %s response queue broken (%s: %s): "
                    "worker process likely died; shutting down",
                    handle.miner_id,
                    type(exc).__name__,
                    exc,
                )
                self.shutdown()
                return
            if isinstance(msg, dict) and msg.get("op") == "mine_result":
                # Use the response's dispatch_id (not the handle's
                # current one) to look up the immutable context that was
                # dispatched alongside this attempt. A late result from
                # dispatch N must pair with dispatch N's context, even
                # if dispatch N+1 has already been issued.
                dispatch_id = msg.get("dispatch_id")
                key = (handle.miner_id, dispatch_id)
                context = self._dispatch_contexts.get(key)
                result = msg.get("result")
                if context is None or not isinstance(result, MiningResult):
                    logger.warning(
                        "mine_result from %s ignored: dispatch_id=%s "
                        "context-known=%s result-type=%s",
                        handle.miner_id,
                        dispatch_id,
                        context is not None,
                        type(result).__name__,
                    )
                    continue
                await self._result_queue.put(
                    _ResultEnvelope(
                        result=result,
                        context=context,
                        handle_id=handle.miner_id,
                        dispatch_id=int(dispatch_id) if dispatch_id is not None else 0,
                    )
                )
                # A mine_result means the worker's dispatch loop exited
                # with a winning result and the worker is now idle at the
                # next req_q.get(). Mark the handle idle so subsequent
                # heads don't try to cancel a non-existent dispatch and
                # eat a 500ms timeout when no sentinel is forthcoming.
                # Also emit a sentinel for any cancel-await already
                # blocked on this dispatch_id.
                if handle._active_dispatch_id == dispatch_id:
                    handle._active_dispatch_id = 0
                self._done_queues[handle.miner_id].put_nowait(dispatch_id)
            elif isinstance(msg, dict) and msg.get("op") == "work_item_done":
                # Worker finished its mine_work_item loop with no result —
                # almost always because cancel() was observed. Surface it
                # tagged with the dispatch_id so _await_handle_done can
                # synchronize on cancellation of that specific dispatch,
                # not just any sentinel.
                done_dispatch_id = msg.get("dispatch_id")
                if handle._active_dispatch_id == done_dispatch_id:
                    handle._active_dispatch_id = 0
                self._done_queues[handle.miner_id].put_nowait(done_dispatch_id)
            elif isinstance(msg, dict) and msg.get("op") == "error":
                self.stats.miner_errors[handle.miner_id] = (
                    self.stats.miner_errors.get(handle.miner_id, 0) + 1
                )
                logger.error(
                    "miner %s reported error (count=%d): %s",
                    handle.miner_id,
                    self.stats.miner_errors[handle.miner_id],
                    msg.get("message"),
                )
                # Emit the same sentinel as work_item_done so
                # _await_handle_done can synchronize after a mining error.
                # Without this, every mining exception causes a guaranteed
                # timeout on the next cancel.
                err_dispatch_id = msg.get("dispatch_id")
                if handle._active_dispatch_id == err_dispatch_id:
                    handle._active_dispatch_id = 0
                self._done_queues[handle.miner_id].put_nowait(err_dispatch_id)
            elif isinstance(msg, dict) and msg.get("op") == "stats":
                # Stats responses are pulled directly by callers of
                # handle.get_stats(); if one lands here it just means
                # nobody was listening — drop and continue. NOTE: while
                # the controller owns the handle, callers MUST NOT call
                # handle.get_stats() directly — the drainer dequeues the
                # response first and blocks the caller forever.
                pass
            else:
                logger.warning(
                    "handle %s sent unrecognized message type=%s op=%s; dropping",
                    handle.miner_id,
                    type(msg).__name__,
                    msg.get("op") if isinstance(msg, dict) else "n/a",
                )

    async def _subscribe_heads(self) -> None:
        """Subscribe to new best heads, with failover on validator drop.

        Loop semantics:
          - `subscribe_new_heads` returning normally → clean shutdown path.
          - `WebSocketException` / `ConnectionError` → trigger one
            `reconnect()` on the subscription client, then re-subscribe.
          - `NoValidatorReachable` from reconnect → fail-loud: record the
            structured attempt log in `stats.last_submission_error` and
            shut the controller down.
          - Any other exception (decoder bugs, RPC type errors, etc.) is
            treated as fatal and shuts down — those are not transient
            connection loss and should not be retried.
        """

        async def callback(block_hash: bytes, block_number: int) -> None:
            # Latest-only semantics: overwrite whatever's pending. If the
            # main loop is busy in submit_proof while heads pile up, we
            # don't want to dispatch against an old one when it returns.
            self._latest_head = (block_hash, block_number)
            self._head_signal.set()

        while not self._shutdown_event.is_set():
            try:
                await self._subscription_client.subscribe_new_heads(callback)
                return  # clean exit — subscribe returned without error
            except (WebSocketException, ConnectionError) as exc:
                logger.warning(
                    "head subscription dropped on %s (%s: %s); failing over",
                    getattr(self._subscription_client, "current_url", "<unknown>"),
                    type(exc).__name__,
                    exc,
                )
                try:
                    await self._subscription_client.reconnect()
                except NoValidatorReachable as fatal:
                    self.stats.last_submission_error = (
                        f"no validators reachable; head subscription cannot "
                        f"recover: {fatal}"
                    )
                    logger.error(
                        "subscription failover exhausted; triggering shutdown:\n%s",
                        fatal,
                    )
                    self.shutdown()
                    return
                # loop iteration: re-subscribe on the new validator
            except Exception as exc:
                self.stats.last_submission_error = (
                    f"head subscription crashed: {type(exc).__name__}: {exc}"
                )
                logger.exception("head subscription crashed; triggering shutdown")
                # We're already on the asyncio loop; no need for
                # call_soon_threadsafe (the subscription wrapper bridges its
                # background thread into our awaitable callback).
                self.shutdown()
                return

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _teardown(self) -> None:
        logger.info("controller shutting down: cancelling handles, draining tasks")
        for handle in self.miner_handles:
            try:
                handle.cancel()
            except Exception as exc:  # noqa: BLE001 — log cleanup failures
                logger.warning(
                    "teardown: handle %s cancel() raised %s: %s",
                    handle.miner_id,
                    type(exc).__name__,
                    exc,
                )
        for task in self._drainer_tasks:
            task.cancel()
        if self._subscription_task is not None:
            self._subscription_task.cancel()
        # Await cancellations. Narrow catch: CancelledError is expected,
        # other exceptions are real cleanup failures worth surfacing.
        for task in self._drainer_tasks + (
            [self._subscription_task] if self._subscription_task else []
        ):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "teardown: task %s raised during cancellation",
                    task.get_name(),
                )
        self._drainer_tasks.clear()
        self._subscription_task = None
        # Slot clients are owned by the pool; do NOT close them here.
        # The pool's close() (called from the CLI's outer try/finally)
        # tears them down at process shutdown.


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def classify_submission(receipt: ExtrinsicReceipt) -> SubmissionOutcome:
    """Bucket a receipt as `OK` / `STALE` / `FATAL`.

    Stale errors are expected race conditions (head change between mining
    and submission); fatal errors mean the controller can't make progress.
    The error-message match is a substring check because substrate-interface
    formats receipts as `Module(error=Foo, index=N, ...)`.

    Returns SubmissionOutcome. Note: inherits from str so `==` comparisons
    against the literals `"ok"`, `"stale"`, `"fatal"` continue to work.
    """
    if receipt.is_success:
        return SubmissionOutcome.OK
    error = receipt.error or ""
    for name in STALE_SUBMISSION_ERRORS:
        if name in error:
            return SubmissionOutcome.STALE
    for name in FATAL_SUBMISSION_ERRORS:
        if name in error:
            return SubmissionOutcome.FATAL
    # Unknown errors default to fatal — better to crash loudly than to silently
    # mine forever against a chain state we don't understand. Log the raw
    # error here so operators see the unrecognized text in the same line
    # as the classification decision.
    logger.error("unrecognized submission error; classifying as fatal: %r", error)
    return SubmissionOutcome.FATAL


__all__ = [
    "ControllerStats",
    "STALE_SUBMISSION_ERRORS",
    "FATAL_SUBMISSION_ERRORS",
    "SubmissionOutcome",
    "SubstrateMinerController",
    "classify_submission",
]
