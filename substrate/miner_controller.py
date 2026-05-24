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
import multiprocessing as mp
import os
import queue as _queue
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol, Tuple

from websocket import WebSocketException

from shared.asyncio_supervise import supervise
from shared.logging_config import get_logger
from shared.miner_survey import build_miner_survey
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.mining_attempt_log import DEFAULT_LOG_DIR, SubmissionLogger
from shared.signer import Signer
from shared.stats_snapshot import StatsSnapshotWriter
from shared.telemetry_process import telemetry_main
from substrate.client import SubstrateClient
from substrate.pool import ValidatorPool
from substrate.pool_client import PoolClient
from substrate.submitter import encode_quantum_proof, submit_proof
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


def build_stats_snapshot_for_telemetry(controller) -> dict[str, Any]:
    """Serialize the controller's live state into a JSON-safe dict.

    Called periodically by ``StatsSnapshotWriter``. The telemetry
    sibling process reads the resulting file on every endpoint hit
    that needs in-process data (counters, hardware descriptor,
    miner survey, identity) — there is no live IPC between the two
    processes, so this is the channel that keeps dashboard visibility
    intact after the telemetry-process split.

    Top-level keys:
      - ``controller``: operational counters (the original
        silent-subscription-death bug was diagnosed via
        ``controller.stats.heads_observed`` — keep every counter
        operators rely on here).
      - ``node_id``, ``ss58_address``, ``account_id_hex``, ``miners``:
        identity surfaced by ``/api/v1/status``.
      - ``descriptor``: hardware descriptor surfaced by ``/api/v1/system``.
      - ``miner_survey``: ``quip.miner_survey.v1`` payload surfaced by
        ``/api/v1/miner/survey``.
      - ``attempts_dir``: filesystem path the telemetry sibling reads
        the mining-attempts JSONL store from.
    """
    s = controller.stats

    def _g(attr: str, default: int = 0) -> int:
        return getattr(s, attr, default)

    controller_counters = {
        "heads_observed": _g("heads_observed"),
        "contexts_dispatched": _g("contexts_dispatched"),
        "results_received": _g("results_received"),
        "proofs_submitted": _g("proofs_submitted"),
        "stale_drops": _g("stale_drops"),
        "submission_errors": _g("submission_errors"),
        "heads_skipped_already_won": _g("heads_skipped_already_won"),
        "heads_dropped_stale_number": _g("heads_dropped_stale_number"),
        "heads_refreshed_active": _g("heads_refreshed_active"),
        "stale_post_win_heads_dropped": _g("stale_post_win_heads_dropped"),
        "zero_seed_snapshots_dropped": _g("zero_seed_snapshots_dropped"),
        "subscription_lag_blocks": _g("subscription_lag_blocks"),
        "heads_promoted_to_rpc": _g("heads_promoted_to_rpc"),
        "heads_refresh_below_floor": _g("heads_refresh_below_floor"),
        "post_win_fast_forwards": _g("post_win_fast_forwards"),
        "post_win_fast_forward_timeouts": _g("post_win_fast_forward_timeouts"),
        "heads_same_key_skipped": _g("heads_same_key_skipped"),
        "none_snapshots_seen": _g("none_snapshots_seen"),
        "duplicate_result_drops": _g("duplicate_result_drops"),
        "proofs_unverified": _g("proofs_unverified"),
        "active_url": getattr(controller, "pool_active_url", None),
    }

    core = getattr(controller, "core", None)
    signer = getattr(controller, "signer", None)

    node_id: Optional[str] = None
    miners_list: list[dict[str, str]] = []
    descriptor: dict[str, Any] = {}
    survey: dict[str, Any] = {}
    if core is not None:
        node_id = getattr(core, "node_id", None)
        miners_list = [
            {"id": h.miner_id, "type": h.miner_type}
            for h in getattr(core, "miner_handles", [])
        ]
        try:
            descriptor = core.descriptor()
        except Exception as exc:  # noqa: BLE001 — keep snapshot writeable
            descriptor = {"error": f"{type(exc).__name__}: {exc}"}
        if signer is not None:
            try:
                survey = build_miner_survey(core, signer, controller=controller)
            except Exception as exc:  # noqa: BLE001
                survey = {"error": f"{type(exc).__name__}: {exc}"}

    ss58_address: Optional[str] = None
    account_id_hex: Optional[str] = None
    if signer is not None:
        try:
            ss58_address = signer.ss58_address()
            account_id_hex = "0x" + signer.account_id_bytes().hex()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "snapshot: signer identity unavailable: %s: %s",
                type(exc).__name__, exc,
            )

    attempts_dir = str(
        getattr(controller, "_attempts_dir", DEFAULT_LOG_DIR)
    )

    return {
        "controller": controller_counters,
        "node_id": node_id,
        "ss58_address": ss58_address,
        "account_id_hex": account_id_hex,
        "miners": miners_list,
        "descriptor": descriptor,
        "miner_survey": survey,
        "attempts_dir": attempts_dir,
    }


class _OperatorFailLoud(RuntimeError):
    """Marker exception for ``on_new_head`` paths the operator must see.

    Raised explicitly in ``on_new_head`` for misconfiguration / stuck-chain
    conditions that the controller cannot recover from without operator
    intervention (consecutive-None-snapshot escalation, topology-hash
    mismatch).

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
    # Heads dropped because the head handler raised a non-connection
    # exception (e.g., snapshot SCALE decode shape drift after a
    # runtime upgrade). Non-zero here means a head was lost but the
    # controller stayed alive. A persistently growing counter indicates
    # a runtime API shape mismatch that needs an operator update.
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
    # Legacy gauge: previously counted heads where the rpc client
    # reported a newer best than the subscription wake delivered. With
    # the subscription chain removed, this stays at 0 — kept in the
    # struct so telemetry consumers keep working.
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

    A ChainEventManager polls the chain's mining snapshot and fires
    ``on_new_head`` whenever ``(last_proof_block_hash, max_energy_milli)``
    changes. The controller dispatches a fresh `SubstrateMiningContext` to
    every attached handle on each change, drains `MiningResult`s back, and
    submits the first valid one as a `QuantumPow.submit_proof` extrinsic.

    Args:
        pool: ValidatorPool providing the ``rpc`` slot client used for
            state queries and extrinsic submission, plus the active
            validator child for the event manager's poll loop.
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
        runtime_dir: Optional[Path] = None,
        telemetry_port: int = 8086,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "SubstrateMinerController requires at least one MinerHandle"
            )
        self._pool = pool
        # `build_client` is populated lazily in `run()` — `__init__` must
        # not touch the network. The parent process keeps an in-parent
        # SubstrateClient solely for compose + sign work (signer key
        # material never crosses an IPC boundary). Submissions go through
        # ``pool_client.submit_signed_extrinsic`` (swap-aware) instead.
        self.build_client: Optional[SubstrateClient] = None
        # ``pool_client`` is the swap-aware read+submit surface; constructed
        # in ``run()`` once the pool's URL list is known.
        self.pool_client: Optional[PoolClient] = None
        # Optional MinerCore hook. When provided, the controller calls
        # `core.record_dispatch()` once per head (not per handle — that would
        # double-count when more than one miner is attached) and
        # `core.record_result(winning_miner_id, mining_time)` on chain-accepted
        # proofs. Keeps `/api/v1/stats`'s legacy `total_blocks_attempted` /
        # `total_blocks_won` / `wins_per_miner` fields live without coupling
        # the controller's type to MinerCore.
        self.core = core
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
        # actual fix for submission storming. See ClosedWorkRecord.
        # OrderedDict keeps eviction simple (popitem(last=False)) and
        # order-aware.
        self._closed_work_keys: "OrderedDict[WorkKey, ClosedWorkRecord]" = (
            OrderedDict()
        )
        # Highest block_number successfully handled. Belt-and-suspenders
        # against a regression that starts feeding us historical block
        # hashes; the guard short-circuits them before they can call
        # mining_snapshot at a stale `at` and reopen the already-won
        # round_seed loop.
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
        # Public alias for the pool, matching the Plan 3 design. Same object
        # as ``self._pool``; both names point to the back-compat-shimmed
        # ValidatorPool. New event-manager wiring uses ``self.pool``.
        self.pool = pool
        # ChainEventManager task — set in run() after the event manager
        # is constructed. Polls the pool's new send() path at adaptive
        # cadence and dispatches on_new_head when state changes.
        self._event_manager_task: Optional[asyncio.Task] = None
        self.events = None  # set in run()
        self._runtime_dir: Path = (
            runtime_dir
            if runtime_dir is not None
            else Path(os.environ.get("QUIP_RUNTIME_DIR", "/tmp/quip"))
        )
        # Stats snapshot writer — set in run(); None before startup.
        self._stats_snapshot_path: Optional[Path] = None
        self._stats_writer: Optional[StatsSnapshotWriter] = None
        self._stats_writer_task: Optional[asyncio.Task] = None
        # Telemetry sibling process — the sole telemetry surface. The
        # controller process owns no HTTP server; the sibling reads the
        # stats snapshot file and queries the chain via its own
        # SubstrateClient.
        self._telemetry_port: int = int(telemetry_port)
        self._telemetry_proc: Optional[mp.Process] = None
        self._telemetry_shutdown_event = None
        # Mining-attempts log directory — surfaced into the snapshot so
        # the telemetry sibling reads the same JSONL store.
        self._attempts_dir: Path = DEFAULT_LOG_DIR

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def pool_active_url(self) -> Optional[str]:
        """Current active validator URL (or None if pool not started)."""
        if self.pool is None:
            return None
        return self.pool.active_url()

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any thread/task."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main loop. Returns on shutdown or fatal error."""
        # Build a parent-process SubstrateClient for compose+sign only;
        # reads and submissions both go through the swap-aware pool via
        # PoolClient. Signer key material never crosses the mp.Queue IPC
        # boundary — see ``substrate.submitter.submit_proof``.
        self.build_client = SubstrateClient(urls=self._pool.urls)
        await self.build_client.connect()
        self.pool_client = PoolClient(self._pool)
        # Pool must be live before any pool_client.send() — including the
        # startup registration check below. ``_start_event_manager`` skips
        # the redundant spawn via the ``active_url() is None`` guard.
        await self._pool.start()

        account = self.signer.account_id_bytes()
        await self._verify_registered(account)

        # Spawn the telemetry sibling process (no-op if telemetry_port is None).
        self._spawn_telemetry_sibling()

        # Drainer tasks consume each handle's resp queue and post into our
        # asyncio result queue. Start them before the first dispatch so we
        # never miss an early result.
        for handle in self.miner_handles:
            drain_name = f"drain-{handle.miner_id}"
            task = asyncio.create_task(
                supervise(
                    self._drain_handle(handle),
                    name=drain_name,
                    on_failure=self._shutdown_event.set,
                ),
                name=drain_name,
            )
            self._drainer_tasks.append(task)

        # Spawn the pool's active validator child and start the
        # ChainEventManager. The event manager polls the snapshot at
        # adaptive cadence and fires on_new_head on state change.
        await self._start_event_manager(account)

        # Stats snapshot writer — atomic JSON dump to runtime_dir every
        # interval. The telemetry sibling process reads this file via
        # `read_snapshot()` to serve every endpoint that needs in-process
        # data; there is no live IPC.
        self._stats_snapshot_path = self._runtime_dir / "telemetry-stats.json"
        self._stats_writer = StatsSnapshotWriter(
            path=self._stats_snapshot_path,
            get_snapshot=lambda: build_stats_snapshot_for_telemetry(self),
            interval_s=1.0,
        )
        self._stats_writer_task = asyncio.create_task(
            supervise(
                self._stats_writer.run(self._shutdown_event),
                name="stats-snapshot-writer",
                on_failure=self._shutdown_event.set,
            ),
            name="stats-snapshot-writer",
        )

        try:
            await self._main_loop()
        finally:
            await self._teardown()

    # ------------------------------------------------------------------
    # Event manager wiring
    # ------------------------------------------------------------------

    async def _start_event_manager(self, account: bytes) -> None:
        """Spawn the pool's active validator child and start the event manager.

        Polls ``get_mining_snapshot`` at adaptive cadence (default 85% of
        blocktime; 10% when overdue) and fires ``on_new_head`` when the
        snapshot's ``(last_proof_block_hash, max_energy_milli)`` changes.
        The watchdog calls ``pool.force_swap()`` if no change is observed
        for ``dead_blocktime_multiplier × blocktime_s``.
        """
        # Defer imports to keep the legacy path importable even if these
        # modules grow heavier dependencies later.
        from substrate.event_manager import ChainEventManager

        # The on-chain `derive_nonce` hashes the SCALE-encoded account ID
        # to produce a width-stable 32-byte miner identity.
        canonical_miner = hashlib.blake2b(account, digest_size=32).digest()

        # Pool is usually already started by ``run()`` so the startup
        # registration check can read through it. Guard against the
        # double-spawn (which would orphan the first handle) but stay
        # correct if a future caller invokes this in isolation.
        if self.pool.active_url() is None:
            await self.pool.start()

        def state_key(snapshot):
            """Dedup key: fires on every new block.

            Including ``block_hash`` makes the event manager dispatch on
            every block, which is what the mempool path needs (each block
            may carry new mempool events). The PoW path's existing
            same-work-key short-circuits in ``on_new_head`` absorb the
            extra invocations cheaply when ``last_proof_block_hash`` /
            ``max_energy_milli`` haven't changed.

            ``None`` snapshots (chain has no topology registered yet) collapse
            to a stable sentinel so the event manager doesn't crash, and so
            consecutive Nones look like "no state change" to the dedup —
            ``on_new_head`` itself owns the consecutive-None escalation.
            """
            if snapshot is None:
                return ("none-snapshot",)
            return (
                snapshot.last_proof_block_hash,
                int(snapshot.difficulty.max_energy_milli),
                snapshot.block_hash,
            )

        self.events = ChainEventManager(
            pool=self.pool,
            state_key=state_key,
            snapshot_op="get_mining_snapshot",
            snapshot_args={
                "miner_account_bytes": canonical_miner,
                "at": None,
                "topology_hash": self.topology_hash,
            },
            blocktime_s=6.0,
        )
        self.events.subscribe("new_head", self.on_new_head)
        self._event_manager_task = asyncio.create_task(
            supervise(
                self.events.run(),
                name="chain-event-manager",
                on_failure=self._shutdown_event.set,
            ),
            name="chain-event-manager",
        )
        logger.info("ChainEventManager started; polling get_mining_snapshot")

    # ------------------------------------------------------------------
    # Telemetry sibling
    # ------------------------------------------------------------------

    def _spawn_telemetry_sibling(self) -> None:
        """Spawn the telemetry process as a sibling.

        Called unconditionally from ``run()``. The sibling is now the
        only telemetry surface — there is no in-process server.
        Validator URLs are derived from the pool's authoritative list;
        the sibling owns its own SubstrateClient.
        """
        validator_urls = list(self.pool.urls) if self.pool is not None else []

        self._telemetry_shutdown_event = mp.Event()
        self._telemetry_proc = mp.Process(
            target=telemetry_main,
            kwargs={
                "listen_host": "0.0.0.0",
                "listen_port": self._telemetry_port,
                "stats_snapshot_path": str(
                    self._runtime_dir / "telemetry-stats.json"
                ),
                "validator_urls": validator_urls,
                "shutdown_event": self._telemetry_shutdown_event,
            },
            name="quip-telemetry",
        )
        self._telemetry_proc.start()
        # ``pid`` can be ``None`` if a test seam overrides Process.start() to
        # a no-op; use %s so the format succeeds either way.
        logger.info(
            "spawned telemetry sibling: pid=%s port=%d",
            self._telemetry_proc.pid,
            self._telemetry_port,
        )

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    async def _verify_registered(self, account: bytes) -> None:
        miner_info = await self.pool_client.query_miner(account)
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
            result_task = asyncio.create_task(self._result_queue.get())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            done, pending = await asyncio.wait(
                [result_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            # Drain cancellations so they don't surface as warnings. Narrow
            # catch: CancelledError is expected, anything else is a bug
            # (these tasks await an Event/Queue.get — no other
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

    # -------------------------------------------------------------------------
    # on_new_head — event-manager entry point. New heads are surfaced as
    # SubstrateMiningContext deliveries from the ChainEventManager poll loop.
    # -------------------------------------------------------------------------

    async def on_new_head(self, ctx: "Optional[SubstrateMiningContext]") -> None:
        """Event-manager callback: a new chain mining context has arrived.

        Receives the ``SubstrateMiningContext`` that
        ``pool.send("get_mining_snapshot", ...)`` returns (or ``None`` if
        the chain has no registered topology yet).

        Guards (in order):
            1. ``None`` snapshot → bump ``stats.none_snapshots_seen``
               and return. Persistent None is handled by the event
               manager's watchdog (force_swap on no state change).
            2. Topology hash mismatch vs ``self.topology_hash`` → fail
               loud via ``_OperatorFailLoud``.
            3. Push ``live_threshold_milli`` to each handle if changed.
            4. Zero-seed guard (chain transient between win + round-roll).
            5. Closed-work-key short-circuit (we already won this round).
            6. Same-key short-circuit (mining in progress on same key).
            7. Cancel any prior dispatch (key changed) and wait for ack.
            8. Dispatch fresh work to each handle.
        """
        self.stats.heads_observed += 1

        # 1. None snapshot — chain isn't seeded with a topology yet, or
        # the runtime is in a transient state between rounds. Bump the
        # stat and return. The event manager's watchdog (force_swap on
        # no state change for ``dead_blocktime_multiplier × blocktime_s``)
        # handles the validator-side variant; a chain-stuck-globally
        # escalation can be added as a separate subscriber later.
        if ctx is None:
            self.stats.none_snapshots_seen += 1
            logger.warning(
                "event manager: get_mining_snapshot returned None"
            )
            return

        # 2. Topology mismatch — operator configured a different topology
        # than the chain is using. Fail loud rather than silently mining
        # the wrong puzzle.
        if (
            self.topology_hash is not None
            and ctx.topology_hash != self.topology_hash
        ):
            raise _OperatorFailLoud(
                "configured --topology-hash does not match snapshot: "
                f"expected 0x{self.topology_hash.hex()}, got "
                f"0x{ctx.topology_hash.hex()}"
            )

        # 3. Live threshold push (decay-driven; only when changed).
        live_threshold = int(ctx.difficulty.max_energy_milli)
        if live_threshold != self._last_pushed_threshold_milli:
            logger.info(
                "live energy threshold changed: %d → %d milli "
                "(last_proof=0x%s...)",
                self._last_pushed_threshold_milli,
                live_threshold,
                ctx.last_proof_block_hash.hex()[:16],
            )
            for handle in self.miner_handles:
                handle.set_live_threshold_milli(live_threshold)
            self._last_pushed_threshold_milli = live_threshold

        # 4. Zero-seed guard. The chain transiently returns
        # ``last_proof_block_hash = 0x00..00`` between accepting a proof
        # and rolling the round. Dispatching against that would produce a
        # degenerate work key the chain will reject; refuse and wait for
        # the next poll to see a real seed. The genesis/bootstrap window
        # (``_highest_handled_block == 0``) is the legitimate zero state.
        if (
            ctx.last_proof_block_hash == b"\x00" * 32
            and self._highest_handled_block > 0
        ):
            self.stats.zero_seed_snapshots_dropped += 1
            logger.warning(
                "snapshot carries zero last_proof_block_hash; refusing "
                "dispatch (highest_handled=%d)",
                self._highest_handled_block,
            )
            return

        new_work_key = (ctx.last_proof_block_hash, ctx.topology_hash)

        # 5. Closed-work-key: we already won this round.
        if new_work_key in self._closed_work_keys:
            return

        # 6. Same-key short-circuit: mining already in progress.
        if new_work_key == self._current_work_key:
            all_idle = all(
                h._active_dispatch_id == 0 for h in self.miner_handles
            )
            if not all_idle:
                self.stats.heads_same_key_skipped += 1
                return
            # All handles are idle on a known key — fall through to
            # re-dispatch (workers may have been cancelled externally).

        # 7. Work key changed — cancel any prior dispatch, wait for ack.
        # Parallel wait keeps total stall to ~0.5s regardless of handle
        # count. Without this synchronization, cancel() → clear() →
        # dispatch can wipe a cancel before the worker observes it.
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
                    self._await_handle_done(h, dispatch_id=did, timeout=0.5)
                    for h, did in cancelled_dispatches
                ]
            )

        # 8. Dispatch.
        self._current_context = ctx
        self._current_work_key = new_work_key
        logger.info(
            "new head (event manager): last_proof=0x%s... topology=0x%s... "
            "nodes=%d edges=%d",
            ctx.last_proof_block_hash.hex()[:16],
            ctx.topology_hash.hex()[:16],
            len(ctx.nodes),
            len(ctx.edges),
        )
        for handle in self.miner_handles:
            dispatch_id = handle.mine_work_item(ctx)
            self._dispatch_contexts[(handle.miner_id, dispatch_id)] = ctx
            self._prune_dispatch_contexts(handle.miner_id, dispatch_id)
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            self.core.record_dispatch()

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
                self.build_client,
                self.pool_client,
                self.signer,
                envelope.result,
                envelope.context,
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
                # `on_new_head` and skips dispatch. Without an active
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
                    accepted_block_number = await self.pool_client.get_block_number(
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
            # Post-win round-roll detection is handled by the
            # ChainEventManager poll loop: within ~5 s the next snapshot
            # carries the new last_proof_block_hash and on_new_head
            # dispatches fresh work.
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
            last_block = await self.pool_client.query_last_proof_block_number()
            if last_block is None:
                logger.warning(
                    "post-OK verify: LastProofBlock is unset — chain may "
                    "not have committed our proof yet (will trust receipt)"
                )
                return None
            winning = await self.pool_client.query_winning_solution(last_block)
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

    def _mark_work_key_closed(
        self, key: WorkKey, record: ClosedWorkRecord
    ) -> None:
        """Record a work key as won and evict the oldest if over cap.

        `record` carries the accepted block hash/number for telemetry
        and stats consumers.
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
        # Signal the event manager to stop polling. Its internal loops
        # observe the flag and exit cleanly; the task itself is awaited
        # below via the unified cancellation sweep.
        if self.events is not None:
            self.events.request_shutdown()
        if self._event_manager_task is not None:
            self._event_manager_task.cancel()
        # Cancel the stats snapshot writer before awaiting tasks.
        if self._stats_writer_task is not None:
            self._stats_writer_task.cancel()
        # Await cancellations. Narrow catch: CancelledError is expected,
        # other exceptions are real cleanup failures worth surfacing.
        extras: list[asyncio.Task] = []
        if self._event_manager_task is not None:
            extras.append(self._event_manager_task)
        if self._stats_writer_task is not None:
            extras.append(self._stats_writer_task)
        for task in self._drainer_tasks + extras:
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
        self._event_manager_task = None
        # Telemetry sibling shutdown.
        if self._telemetry_shutdown_event is not None:
            self._telemetry_shutdown_event.set()
        if self._telemetry_proc is not None:
            self._telemetry_proc.join(timeout=5)
            if self._telemetry_proc.is_alive():
                logger.warning(
                    "telemetry sibling did not exit; terminating"
                )
                self._telemetry_proc.terminate()
                self._telemetry_proc.join(timeout=2)
            self._telemetry_proc = None
            self._telemetry_shutdown_event = None
        # Close the parent-side build client. The pool's active validator
        # handle is torn down by ``pool.close()`` from the CLI's outer
        # try/finally, not here.
        if self.build_client is not None:
            try:
                await self.build_client.close()
            except Exception:  # noqa: BLE001 — log, don't mask teardown
                logger.exception("teardown: build_client.close() raised")
            self.build_client = None


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
    "build_stats_snapshot_for_telemetry",
    "classify_submission",
]
