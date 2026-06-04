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
        await pool.shutdown()

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
import json
import multiprocessing as mp
import os
import queue as _queue
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace as dataclasses_replace
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol, Tuple

from websocket import WebSocketException

from shared.asyncio_supervise import supervise
from shared.logging_config import get_logger
from shared.miner_config import SubmissionConfig
from shared.miner_survey import build_miner_survey
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.mining_attempt_log import (
    DEFAULT_LOG_DIR,
    SubmissionLogger,
    query_by_solution_number,
)
from shared.signer import Signer
from shared.stats_snapshot import StatsSnapshotWriter, snapshot_filename_for
from shared.telemetry_process import telemetry_main
from substrate.client import SubstrateClient
from substrate.pool import ValidatorPool
from substrate.pool_client import PoolClient
from substrate.remark import submit_remark
from substrate.decay_timing import TimingTracker
from substrate.difficulty_decay import EnergyCurve, build_decay_schedule
from substrate.submitter import (
    SubmitRetryAction,
    encode_quantum_proof,
    submit_proof,
    submit_with_retry,
)
from substrate.types import (
    ExtrinsicReceipt,
    PowConstants,
    SubstrateDifficulty,
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
        "zero_seed_snapshots_dropped": _g("zero_seed_snapshots_dropped"),
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
        latest_budget = getattr(controller, "_latest_budget", {})
        miners_list = [
            {
                "id": h.miner_id,
                "type": h.miner_type,
                "qpu_budget": latest_budget.get(h.miner_id),
            }
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
        # `mode` lets the aggregator (multi-process container) bucket
        # this snapshot by backend without parsing handle ids. Defaults
        # to the controller's `snapshot_kind` field (set by the CLI from
        # the chosen subcommand: cpu/gpu/qpu).
        "mode": getattr(controller, "snapshot_kind", "") or "default",
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

# Upper bound on the write-once participation dedup set. Generous because
# solution numbers are monotonic — eviction can never re-admit an active key.
_PARTICIPATION_RETENTION = 2048


# How far ahead (in blocks) the anticipatory predictor looks for the
# decay block at which a previewed candidate clears. The energy threshold
# eases one step per ``epoch_length`` blocks, so this caps how many decay
# steps out we'll wait for a marginal candidate before treating it as
# "won't clear in any useful window". Sized generously — at the default
# 6 s blocktime this is ~10 min of look-ahead.
_ANTICIPATORY_SEARCH_LIMIT = 100

# Cadence of the free-running fire timer (seconds). Each tick re-evaluates
# the active preview's worker-computed ``valid_at_block`` against the
# cadence clock and fires at ``T* - lag``; small enough to be punctual at a
# 6 s blocktime, large enough to be negligible overhead.
_FIRE_TICK_S = 1.0


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
    # Submissions that produced an OK receipt but whose post-submit
    # winning_solution check did not match our miner+nonce. Non-zero
    # here means we accepted a receipt that the runtime did not
    # actually record (or recorded against a different submitter).
    proofs_unverified: int = 0
    # Snapshot at a non-genesis head carried last_proof_block_hash =
    # 0x00..00. Observed transiently at the exact accepted block; the
    # next block returns the receipt hash. Refusing dispatch here
    # prevents mining a degenerate seed.
    zero_seed_snapshots_dropped: int = 0
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
        snapshot_kind: str = "",
        spawn_telemetry_sibling: bool = True,
        submission_config: Optional[SubmissionConfig] = None,
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
        # Submission tuning (tip + retry bounds). Defaults reproduce the
        # pre-tip, no-extra-retry behavior, so an old config or a caller
        # that omits this keeps working unchanged. Task 6's anticipatory
        # fire loop reads `max_retries` / `retry_backoff_ms` from here.
        self.submission_config = submission_config or SubmissionConfig()
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
        # Last work key announced at INFO by the new-head banner, so a head
        # that re-dispatches the same work item logs at DEBUG instead.
        self._last_logged_work_key: Optional[WorkKey] = None
        # Chain-global solution number (ordinal of the solution being mined,
        # = count(WinningSolutions) + 1) per work key. Resolved once per
        # round and reused for re-dispatches and the submission record so the
        # on-disk archive is keyed consistently. Pruned on round eviction.
        self._solution_number_by_work_key: "dict[WorkKey, int]" = {}
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
        # Per-controller submission log. Keyed by the chain-global solution
        # number (resolved per round in _resolve_solution_number), the same
        # key the worker files its attempts under — so submission.json lands
        # in the matching {solution_number}/ dir with no separate index.
        self._submission_log = SubmissionLogger()
        self._result_queue: asyncio.Queue[_ResultEnvelope] = asyncio.Queue()
        # Anticipatory-submission preview store (Task 6a primitive).
        # Latest best-by-floor candidate previewed by a worker, keyed by
        # WorkKey (last_proof_block_hash, topology_hash). The drainer
        # routes ``{"op": "preview"}`` messages here WITHOUT creating a
        # _ResultEnvelope or submitting; Task 6b reads this to predict the
        # decay-block at which the candidate clears and pre-submit. Keeps
        # the lowest ``submit_floor_energy`` seen for each work key.
        self._latest_preview: dict[WorkKey, dict] = {}
        # Latest live QPU budget snapshot per miner id (worker-initiated
        # ``{"op": "budget"}`` pushes). Surfaced in the telemetry snapshot so
        # operators can see live daily-budget usage; never drives submission.
        self._latest_budget: dict[str, Any] = {}
        # Write-once participation dedup: (miner_id, solution_number) already
        # marked via a System.remark. Solution numbers are monotonic, so the
        # arbitrary-pop bound below can never re-admit a still-active key.
        self._participated: set[tuple[str, int]] = set()
        # Anticipatory-submission state (Task 6b).
        # ``_pow_constants`` caches the four decay constants
        # (epoch_length + curve c-triple) for the session — they only
        # change with a runtime upgrade, so a single RPC at first use
        # suffices. ``_base_difficulty_by_key`` caches the UNDECAYED
        # ``QuantumPow.Difficulty`` baseline per work key (it only changes
        # when a proof wins = a new work key), and ``_anticipatory_fired``
        # records work keys whose preview the controller has already
        # SUCCESS-submitted (or is mid-fire on) so the worker's later
        # returned result for the same key is treated as a no-op.
        self._pow_constants: Optional[PowConstants] = None
        self._base_difficulty_by_key: dict[WorkKey, SubstrateDifficulty] = {}
        # Cached decay schedule per work key (tuple of schedule, last_proof_block,
        # epoch_length). Built once per round from _anticipatory_inputs; evicted
        # alongside the other per-key state in _evict_anticipatory_state.
        self._decay_schedule_by_key: dict = {}  # WorkKey -> (list[int], int, int)
        self._anticipatory_fired: set[WorkKey] = set()
        self._timing = TimingTracker()
        # Free-running cadence fire timer (Task 7). Re-evaluates the active
        # preview's worker-computed ``valid_at_block`` each tick and fires at
        # ``T* - lag``; sole fire authority (no per-head recompute).
        self._fire_timer_task: Optional[asyncio.Task] = None
        # Dedup key for the throttled "best candidate" status line.
        self._last_fire_status_key: Optional[tuple] = None
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
        # `snapshot_kind` tags the per-process snapshot file so a
        # multi-process container (entrypoint supervisor spawning one
        # quip-miner per active backend group) doesn't have N children
        # racing the same file. Default of "" picks the legacy
        # single-snapshot filename for one-shot / single-mode usage.
        self.snapshot_kind: str = snapshot_kind
        # When False, the controller skips the in-process telemetry
        # sibling — used by the Docker entrypoint which spawns a single
        # aggregating sibling outside any child process via
        # `quip-miner telemetry`. Named with the `_spawn_..._enabled`
        # suffix to avoid shadowing the same-named method.
        self._spawn_telemetry_sibling_enabled: bool = spawn_telemetry_sibling
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
        self._stats_snapshot_path = self._runtime_dir / snapshot_filename_for(self.snapshot_kind)
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
        self._fire_timer_task = asyncio.create_task(
            supervise(
                self._fire_timer_loop(),
                name="fire-timer",
                on_failure=self._shutdown_event.set,
            ),
            name="fire-timer",
        )
        logger.info("ChainEventManager started; polling get_mining_snapshot")

    # ------------------------------------------------------------------
    # Telemetry sibling
    # ------------------------------------------------------------------

    def _spawn_telemetry_sibling(self) -> None:
        """Spawn the telemetry process as a sibling.

        Called from ``run()``. Skipped when ``spawn_telemetry_sibling``
        is False — the Docker entrypoint sets this to centralise
        telemetry into a single aggregating sibling that reads every
        child's per-kind snapshot via ``read_all_snapshots`` rather
        than competing for ``telemetry_port``.

        Validator URLs are derived from the pool's authoritative list;
        the sibling owns its own SubstrateClient.
        """
        if not self._spawn_telemetry_sibling_enabled:
            logger.info(
                "telemetry sibling spawn skipped (spawn_telemetry_sibling=False); "
                "snapshot still written to %s",
                self._runtime_dir / snapshot_filename_for(self.snapshot_kind),
            )
            return
        validator_urls = list(self.pool.urls) if self.pool is not None else []

        self._telemetry_shutdown_event = mp.Event()
        self._telemetry_proc = mp.Process(
            target=telemetry_main,
            kwargs={
                "listen_host": "0.0.0.0",
                "listen_port": self._telemetry_port,
                "stats_snapshot_path": str(
                    self._runtime_dir / snapshot_filename_for(self.snapshot_kind)
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
        # Startup self-registers (CLI Guard D) before controllers spawn, but
        # that registration landed via the setup client and may not have
        # propagated to this pool's active validator yet. Retry a few times
        # over ~2 blocks before treating absence as a real failure.
        miner_info = None
        for attempt in range(3):
            miner_info = await self.pool_client.query_miner(account)
            if miner_info is not None:
                break
            if attempt < 2:
                await asyncio.sleep(2.0)
        if miner_info is None:
            raise RuntimeError(
                f"signer account 0x{account.hex()} is not in "
                "QuantumPow.Miners after startup registration; the "
                "register_miner extrinsic did not land or has not yet "
                "propagated to this validator"
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

        new_work_key = _work_key(ctx)

        # Work-key rollover: a new round started. Evict any preview +
        # anticipatory state bound to a *different* key so we never act on
        # a dead round's candidate, and so ``_latest_preview`` doesn't grow
        # unbounded (6a left it unpruned). Done before the closed/same-key
        # short-circuits so the eviction happens even on heads we otherwise
        # skip dispatch for.
        if (
            self._current_work_key is not None
            and new_work_key != self._current_work_key
        ):
            self._evict_anticipatory_state(self._current_work_key)

        # 5. Closed-work-key: we already won this round.
        if new_work_key in self._closed_work_keys:
            return

        # Timing anchor (Task 7): fold this head's chain timestamp into the
        # TimingTracker so the free-running fire timer can convert a worker-
        # computed ``valid_at_block`` into a monotonic fire deadline. The
        # timer — not this callback — is the sole fire authority.
        # SubstrateClient.query_block_timestamp_ms swallows its own errors
        # (returns None), but in pool mode ValidatorPool.send can raise (e.g.
        # ValidatorSwapped) BEFORE reaching the client's try/except. Guard the
        # read so a transient swap can't abort the rest of on_new_head and skip
        # this head's dispatch.
        try:
            ts_ms = await self.pool_client.query_block_timestamp_ms(
                ctx.block_hash,
            )
        except Exception as exc:  # noqa: BLE001 — transient swap must not skip head
            logger.debug("query_block_timestamp_ms failed (ignored): %s", exc)
            ts_ms = None
        if ts_ms is not None:
            self._timing.observe_head(
                block_number=int(ctx.block_number),
                chain_ts_s=ts_ms / 1000.0,
                monotonic_now=asyncio.get_running_loop().time(),
                wallclock_now=time.time(),
            )

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
        # Attach round-constant decay schedule so workers can rank their
        # candidate stash by win-time locally without per-iteration RPCs.
        # Built once per work key; SubstrateMiningContext is frozen so we
        # create an enriched copy via dataclasses.replace before dispatch.
        sched = self._decay_schedule_by_key.get(new_work_key)
        if sched is None:
            inputs = await self._anticipatory_inputs(ctx, new_work_key)
            if inputs is not None:
                base, last_proof_block, constants, curve = inputs
                sched = (
                    build_decay_schedule(
                        int(base.max_energy_milli),
                        curve,
                        _ANTICIPATORY_SEARCH_LIMIT,
                    ),
                    int(last_proof_block),
                    int(constants.epoch_length),
                )
                self._decay_schedule_by_key[new_work_key] = sched
        if sched is not None:
            ctx = dataclasses_replace(
                ctx,
                decay_schedule=sched[0],
                last_proof_block=sched[1],
                epoch_length=sched[2],
            )
        self._current_context = ctx
        self._current_work_key = new_work_key
        # Chain-global solution number for this round — the on-disk archive
        # key the workers write under. Resolved once per round (cached).
        solution_number = await self._resolve_solution_number(new_work_key)
        # Announce at INFO only when the work item changes. A new head that
        # re-dispatches the same (last_proof, topology) — e.g. a QPU worker
        # that went idle between blocks — logs at DEBUG to avoid per-head spam.
        log = (
            logger.info
            if new_work_key != self._last_logged_work_key
            else logger.debug
        )
        self._last_logged_work_key = new_work_key
        log(
            "new head (event manager): solution=%s last_proof=0x%s... "
            "topology=0x%s... nodes=%d edges=%d",
            solution_number,
            ctx.last_proof_block_hash.hex()[:16],
            ctx.topology_hash.hex()[:16],
            len(ctx.nodes),
            len(ctx.edges),
        )
        for handle in self.miner_handles:
            dispatch_id = handle.mine_work_item(
                ctx, solution_number=solution_number,
            )
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

        # De-dup with the anticipatory path: if the controller has already
        # SUCCESS-submitted (or is mid-fire on) this work key via the
        # preview-driven fire loop, the worker's own returned result is a
        # confirmation, not a new submission. Drop it without a second
        # submit_proof. (The SUCCESS case also lands in _closed_work_keys
        # above; this catches the window where a fire is in progress but
        # the key isn't closed yet.)
        if envelope_key in self._anticipatory_fired:
            self.stats.duplicate_result_drops += 1
            logger.info(
                "dropping result from %s: work_key already handled by "
                "anticipatory fire (last_proof_block_hash=0x%s...)",
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

        # The solution number (resolved when this round was dispatched) is
        # the archive key shared with the worker's attempts.
        solution_number = self._solution_number_for_context(envelope.context)
        result_energy_milli = int(envelope.result.energy * 1000)
        result_diversity_milli = int(envelope.result.diversity * 1000)
        snapshot_threshold_milli = int(envelope.context.difficulty.max_energy_milli)
        last_proof_hex = "0x" + envelope.context.last_proof_block_hash.hex()
        log_common: dict[str, Any] = {
            "solution_number": solution_number,
            "miner_id": envelope.handle_id,
            "miner_type": envelope.result.miner_type,
            "energy_milli": result_energy_milli,
            "diversity_milli": result_diversity_milli,
            "threshold_milli": snapshot_threshold_milli,
            "last_proof_block_hash_hex": last_proof_hex,
            "num_valid": envelope.result.num_valid,
        }

        try:
            receipt = await submit_proof(
                self.build_client,
                self.pool_client,
                self.signer,
                envelope.result,
                envelope.context,
                tip=self.submission_config.tip_plancks,
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.submission_errors += 1
            self.stats.last_submission_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "submit_proof RPC failed for result from %s: %s",
                envelope.handle_id,
                exc,
            )
            pow_seq_rpc_err = await self._query_proofs_submitted_safe()
            self._submission_log.record(
                solution_number=solution_number,
                miner_id=envelope.handle_id,
                miner_type=envelope.result.miner_type,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="chain_error",
                num_valid=envelope.result.num_valid,
                pow_sequence=pow_seq_rpc_err,
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
            # verified >= 0  → won, value is the PoW block number
            # verified < 0   → mismatch (-1 sentinel)
            # verified is None → RPC failed, inconclusive
            if verified is not None and verified < 0:
                await self._record_verify_fail(
                    envelope.context, None, log_common,
                    extrinsic_hash=receipt.extrinsic_hash,
                    receipt_block=receipt.block_hash,
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
            accepted_block_hash, accepted_block_number = (
                await self._resolve_accepted_block(receipt.block_hash)
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
            # Precise QPU spent on this solution # (summed from the per-attempt
            # QPU times in the attempt log). Telemetry/log only — not on-chain.
            qpu_access_us_total = self._sum_qpu_access_us(solution_number)
            self._submission_log.record(
                solution_number=solution_number,
                miner_id=envelope.handle_id,
                miner_type=envelope.result.miner_type,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="submitted_inblock",
                num_valid=envelope.result.num_valid,
                extrinsic_hash=receipt.extrinsic_hash,
                chain_block_hash=receipt.block_hash,
                chain_block_number=self._resolve_won_block_number(
                    verified, accepted_block_number
                ),
                qpu_access_us_total=qpu_access_us_total,
            )
            if self.core is not None:
                self.core.record_result(
                    winning_miner_id=envelope.result.miner_id,
                    mining_time=float(envelope.result.mining_time),
                )
            await self._invoke_proof_submitted_callback(receipt, envelope.context)
        elif outcome is SubmissionOutcome.STALE:
            self.stats.stale_drops += 1
            logger.info(
                "submit_proof dropped as stale: error=%s extrinsic=%s",
                receipt.error,
                receipt.extrinsic_hash,
            )
            pow_seq_stale = await self._query_proofs_submitted_safe()
            self._submission_log.record(
                solution_number=solution_number,
                miner_id=envelope.handle_id,
                miner_type=envelope.result.miner_type,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="rejected_stale",
                num_valid=envelope.result.num_valid,
                extrinsic_hash=receipt.extrinsic_hash,
                pow_sequence=pow_seq_stale,
                error=str(receipt.error or ""),
            )
        else:  # FATAL
            self.stats.submission_errors += 1
            self.stats.last_submission_error = str(receipt.error or "")
            pow_seq_fatal = await self._query_proofs_submitted_safe()
            self._submission_log.record(
                solution_number=solution_number,
                miner_id=envelope.handle_id,
                miner_type=envelope.result.miner_type,
                energy_milli=result_energy_milli,
                diversity_milli=result_diversity_milli,
                threshold_milli=snapshot_threshold_milli,
                last_proof_block_hash_hex=last_proof_hex,
                outcome="chain_error",
                num_valid=envelope.result.num_valid,
                extrinsic_hash=receipt.extrinsic_hash,
                pow_sequence=pow_seq_fatal,
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

        On timeout we proceed anyway. This is logged at DEBUG, not
        WARNING: under steady SA load the worker is inside an
        uninterruptible ``sample_ising`` call that takes 30-45s, so
        it physically cannot ack within the 0.5s budget. The downstream
        guards (``dispatch_id`` check in ``_handle_result`` + the
        ``_closed_work_keys`` short-circuit) make the cancel→clear race
        harmless — stale results from the old dispatch land, get
        dropped, and the new dispatch proceeds. The wait remains as a
        fast-path optimization for the rare case where the worker IS
        between SA calls (buffer refill, attempt boundary) and can
        ack within a millisecond.
        """
        sentinel_queue = self._done_queues.get(handle.miner_id)
        if sentinel_queue is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        def _log_timeout() -> None:
            logger.debug(
                "handle %s did not ack cancel of dispatch_id=%d within "
                "%.1fs; dispatching anyway (downstream guards handle "
                "the stale result)",
                handle.miner_id,
                dispatch_id,
                timeout,
            )

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                _log_timeout()
                return
            try:
                got = await asyncio.wait_for(
                    sentinel_queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                _log_timeout()
                return
            # Match exactly — older dispatch sentinels arriving here are
            # already-resolved and can be discarded. A sentinel for a
            # dispatch newer than the one we're waiting for is logically
            # impossible (we haven't issued one yet on this handle).
            if got is None or got == dispatch_id:
                return

    async def _query_proofs_submitted_safe(self) -> Optional[int]:
        """Return ``QuantumPow.Miners[self.signer].proofs_submitted``, or None.

        Best-effort — swallows all exceptions so callers on the hot submit
        path are never blocked by a transient RPC failure. Returns None when
        the miner is unregistered, when the chain is unreachable, or on any
        other error.
        """
        try:
            return await self.pool_client.query_proofs_submitted(
                self.signer.account_id_bytes()
            )
        except Exception:  # noqa: BLE001
            return None

    async def _query_winning_solution_count_safe(self) -> Optional[int]:
        """Return ``count(QuantumPow.WinningSolutions)``, or None on failure.

        Best-effort — swallows all exceptions so a transient RPC failure
        never blocks dispatch. See :meth:`_resolve_solution_number` for how
        the fallback is handled when this returns None.
        """
        try:
            return await self.pool_client.query_winning_solution_count()
        except Exception:  # noqa: BLE001
            return None

    async def _resolve_solution_number(
        self, work_key: WorkKey
    ) -> Optional[int]:
        """Chain-global solution number for the round ``work_key``.

        ``count(WinningSolutions) + 1`` — the ordinal of the solution we're
        mining toward. Cached per work key: resolved once when a round opens
        and reused for re-dispatches and the submission record, so all
        artifacts for the round land under one ``{solution_number}/`` dir.

        On RPC failure, falls back to ``max(known) + 1`` (each round closes
        by adding exactly one winning solution, so one-more-than-the-last is
        the correct estimate) and, failing that, ``None`` — which the
        loggers bucket under solution 0 with a warning.
        """
        cached = self._solution_number_by_work_key.get(work_key)
        if cached is not None:
            return cached
        count = await self._query_winning_solution_count_safe()
        if count is not None:
            solution_number = count + 1
        elif self._solution_number_by_work_key:
            solution_number = max(self._solution_number_by_work_key.values()) + 1
            logger.warning(
                "WinningSolutions count RPC failed; estimating solution "
                "number %d (last known + 1)", solution_number,
            )
        else:
            logger.warning(
                "WinningSolutions count RPC failed and no prior solution "
                "number; this round's attempts will bucket under solution 0"
            )
            return None
        self._solution_number_by_work_key[work_key] = solution_number
        # Bound the map — keep only recent rounds (parity with dispatch
        # context / closed-work-key retention).
        while len(self._solution_number_by_work_key) > _DISPATCH_CONTEXT_RETENTION:
            oldest = next(iter(self._solution_number_by_work_key))
            self._solution_number_by_work_key.pop(oldest, None)
        return solution_number

    def _solution_number_for_context(
        self, ctx: "SubstrateMiningContext"
    ) -> Optional[int]:
        """Solution number for ``ctx``'s round, resolved at dispatch time.

        Submissions reuse the value cached when the round opened so the
        submission record lands in the same ``{solution_number}/`` dir as
        the worker's attempts. Returns None if the round was never
        dispatched (not expected on a real submit path).
        """
        return self._solution_number_by_work_key.get(_work_key(ctx))

    async def _verify_proof_recorded(self, envelope: _ResultEnvelope) -> Optional[int]:
        """Confirm the runtime recorded this miner's proof on chain.

        Returns:
          - A non-negative ``int`` (the won PoW block number, i.e.
            ``QuantumPow.LastProofBlock``) if the winning solution at that
            block matches our miner address and nonce.
          - ``-1`` if a winning solution was returned but does NOT match us
            (someone else won this round, or the chain rolled back our
            submission silently). Use ``< 0`` to test for this case.
          - ``None`` if the verification RPC itself failed — caller should
            treat as inconclusive and proceed with closing the work key
            (the alternative is a submission-storm loop, which is worse than
            a single false-positive close).
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
            return -1
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
            return last_block
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
        return -1

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

    def _store_budget(self, handle: MinerHandle, msg: dict) -> None:
        """Stash a worker's latest live QPU budget snapshot, keyed by miner id.

        Pure storage for the telemetry snapshot (surfaced as
        ``miners[].qpu_budget``). A malformed payload is ignored so a stray
        message can't poison the dashboard. Never submits, never blocks.
        """
        data = msg.get("data")
        if not isinstance(data, dict):
            logger.warning(
                "budget from %s ignored: malformed data (type=%s)",
                handle.miner_id,
                type(data).__name__,
            )
            return
        self._latest_budget[handle.miner_id] = data

    def _sum_qpu_access_us(self, solution_number: Optional[int]) -> Optional[int]:
        """Precise QPU access time the node spent on a solution #, in µs.

        Sums the per-attempt ``qpu_access_time_us`` the miners already record in
        the attempt log under ``{solution#}/`` — the per-solution view of the
        same QPU times ``QPUTimeManager`` aggregates to a lifetime total. Summed
        across every launched miner (CPU/GPU attempts carry no QPU time, so the
        result is the node's true QPU effort on the winning solution). Returns
        ``None`` on read failure so a log hiccup never blocks the win record.
        """
        if solution_number is None:
            return None
        try:
            return sum(
                int(a.get("qpu_access_time_us") or 0)
                for h in self.miner_handles
                for a in query_by_solution_number(
                    h.miner_id, solution_number,
                    log_dir=self._submission_log.log_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001 — observability path
            logger.debug("qpu spend sum failed (ignored): %s", exc)
            return None

    def _mark_participating(self, miner_id: str, msg: dict) -> None:
        """Submit a write-once participation remark for ``(miner_id, sol#)``.

        Dedups on ``(miner_id, solution_number)`` so each miner publishes at
        most one marker per solution #. Spawns a best-effort, supervised task
        to submit the remark (never blocks the drain loop; failures are
        swallowed — participation is observability, not consensus).
        """
        try:
            solution_number = int(msg.get("solution_number", 0))
        except (TypeError, ValueError):
            return
        key = (miner_id, solution_number)
        if key in self._participated:
            return
        self._participated.add(key)
        while len(self._participated) > _PARTICIPATION_RETENTION:
            self._participated.pop()

        payload: dict[str, Any] = {
            "schema": "quip-participation",
            "solution": solution_number,
            "miner": miner_id,
            "kind": msg.get("kind"),
        }
        if "budget_seconds" in msg:
            payload["budget_seconds"] = msg["budget_seconds"]
        asyncio.create_task(
            supervise(
                self._submit_participation_remark(payload),
                name=f"participate-{miner_id}-{solution_number}",
                on_failure=lambda: None,
            ),
            name=f"participate-{miner_id}-{solution_number}",
        )

    async def _submit_participation_remark(self, payload: dict) -> None:
        """Submit one participation remark (best-effort, never raises).

        Prefers ``System.remark_with_event`` (observable in block events),
        falling back to plain ``System.remark`` — the same pattern as the
        auto-identify flow. Logs and swallows any failure so mining continues.
        """
        if self.build_client is None:
            return
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        try:
            receipt, _call_function = await submit_remark(
                self.build_client, self.signer, body,
            )
            if receipt.error:
                logger.warning(
                    "participation remark rejected for %s (%s); mining continues",
                    payload.get("miner"), receipt.error,
                )
                return
            logger.info(
                "participation remark submitted: miner=%s solution=%s budget=%s",
                payload.get("miner"), payload.get("solution"),
                payload.get("budget_seconds"),
            )
        except Exception as exc:  # noqa: BLE001 — observability path
            logger.warning(
                "participation remark failed for %s (%s: %s); mining continues",
                payload.get("miner"), type(exc).__name__, exc,
            )

    def _store_preview(self, handle: MinerHandle, msg: dict) -> None:
        """Stash a worker best-candidate preview keyed by work key.

        Resolves the preview's ``dispatch_id`` to the immutable context it
        was dispatched against (so previews from a stale dispatch land
        under the right work key), then keeps the lowest-floor preview per
        work key. Pure storage — no submission, no _ResultEnvelope. Task 6b
        reads ``self._latest_preview`` to drive prediction/firing.
        """
        dispatch_id = msg.get("dispatch_id")
        data = msg.get("data")
        if not isinstance(data, dict):
            logger.warning(
                "preview from %s ignored: malformed data (type=%s)",
                handle.miner_id,
                type(data).__name__,
            )
            return
        context = self._dispatch_contexts.get((handle.miner_id, dispatch_id))
        if context is None:
            logger.debug(
                "preview from %s dropped: no context for dispatch_id=%s",
                handle.miner_id,
                dispatch_id,
            )
            return
        key = _work_key(context)
        # ``data`` already carries ``dispatch_id`` (the worker echoes it in
        # the payload), so we don't repeat it here — ``**data`` provides it.
        entry = {
            "handle_id": handle.miner_id,
            "context": context,
            **data,
        }
        prev = self._latest_preview.get(key)
        # Keep the lowest (best) submit_floor_energy seen for this work
        # key. Workers already throttle to strict improvements, but two
        # handles can preview the same key; this picks the better one.
        if prev is not None:
            prev_floor = prev.get("submit_floor_energy")
            new_floor = entry.get("submit_floor_energy")
            if (
                prev_floor is not None
                and new_floor is not None
                and new_floor >= prev_floor
            ):
                return
        self._latest_preview[key] = entry

    # ------------------------------------------------------------------
    # Anticipatory submission (Task 6b)
    # ------------------------------------------------------------------

    def _evict_anticipatory_state(self, key: WorkKey) -> None:
        """Drop all preview + fire state bound to a closed work key.

        Called on work-key rollover (a new round) and on STOP_* fire
        outcomes. Keeps ``_latest_preview`` from growing unbounded and
        guarantees we never act on a dead round's candidate.
        """
        self._latest_preview.pop(key, None)
        self._base_difficulty_by_key.pop(key, None)
        self._decay_schedule_by_key.pop(key, None)
        self._anticipatory_fired.discard(key)
        # New round: re-arm the throttled status line so the next candidate
        # logs even if its (valid_at_block, decay_num) collides with the old.
        self._last_fire_status_key = None

    async def _anticipatory_inputs(
        self, ctx: SubstrateMiningContext, key: WorkKey
    ) -> Optional[Tuple[SubstrateDifficulty, int, PowConstants, EnergyCurve]]:
        """Gather + cache the decay-predictor inputs for ``key``.

        Returns ``(base_difficulty, last_proof_block, constants, curve)``
        or ``None`` if any chain read fails (best-effort — a transient RPC
        failure must not crash the head loop; the next head retries).

        ``constants`` (epoch_length + curve c-triple) are session-cached;
        ``base_difficulty`` is cached per work key because it only changes
        when a proof wins (a new key). ``last_proof_block`` is re-read each
        call — cheap, and it's the genesis sentinel that gates decay.
        """
        try:
            if self._pow_constants is None:
                self._pow_constants = await self.pool_client.query_pow_constants()
            constants = self._pow_constants
            if constants is None:
                return None
            base = self._base_difficulty_by_key.get(key)
            if base is None:
                base = await self.pool_client.query_difficulty()
                if base is None:
                    return None
                self._base_difficulty_by_key[key] = base
            last_proof_block = await self.pool_client.query_last_proof_block_number()
        except Exception as exc:  # noqa: BLE001 — best-effort on hot path
            logger.warning(
                "anticipatory: chain read for predictor inputs failed "
                "(%s: %s); skipping this head",
                type(exc).__name__,
                exc,
            )
            return None
        if last_proof_block is None:
            last_proof_block = 0
        curve = EnergyCurve.from_topology(
            num_nodes=len(ctx.nodes),
            num_edges=len(ctx.edges),
            c_easy_milli=constants.curve_c_easy_milli,
            c_knee_milli=constants.curve_c_knee_milli,
            c_hard_milli=constants.curve_c_hard_milli,
        )
        return base, int(last_proof_block), constants, curve

    async def _fire_timer_loop(self) -> None:
        """Cadence-driven fire authority: tick, predict, fire — no head needed.

        Runs until shutdown. Each tick re-evaluates the active preview's
        worker-computed ``valid_at_block`` against the cadence clock and fires
        at ``T* - lag``; resilient to a head feed that pauses between blocks.
        """
        while not self._shutdown_event.is_set():
            try:
                await self._maybe_fire_on_cadence()
            except Exception:  # noqa: BLE001 — one bad tick must not kill the loop
                logger.exception("fire timer tick failed; continuing")
            await asyncio.sleep(_FIRE_TICK_S)

    async def _maybe_fire_on_cadence(self) -> None:
        """Re-evaluate the active preview and fire it if its win-time arrived.

        Reads the worker-computed ``valid_at_block`` from the active preview,
        converts it to a monotonic deadline via the TimingTracker anchor, and
        fires at ``T* - lag``. With no timing anchor yet (no head observed),
        waits for ``observe_head`` to seed one before firing.
        """
        key = self._current_work_key
        if (
            key is None
            or key in self._anticipatory_fired
            or key in self._closed_work_keys
        ):
            return
        preview = self._latest_preview.get(key)
        ctx = self._current_context
        if preview is None or ctx is None:
            return
        valid_at = preview.get("valid_at_block")
        # Reject the non-decay sentinel (0) and any non-positive block. A
        # real decay candidate has valid_at = last_proof_block + decay_num *
        # epoch_length with last_proof_block > 0 in normal operation, so
        # valid_at > 0 always. valid_at == 0 means the worker emitted a
        # non-decay preview (legacy path, or a schedule-less round after a
        # transient RPC failure in _anticipatory_inputs); firing on it would
        # make fire_deadline_monotonic(b_star=0) resolve to the distant past
        # and submit an un-gated candidate every tick. Suppress instead.
        if valid_at is None or int(valid_at) <= 0:
            return
        valid_at = int(valid_at)
        now = asyncio.get_running_loop().time()
        cur_block = self._timing.estimate_block(now_monotonic=now)
        self._log_fire_status(preview, valid_at, cur_block)
        deadline = self._timing.fire_deadline_monotonic(
            b_star=valid_at, now_monotonic=now,
        )
        if deadline is None:
            return  # no timing anchor yet — wait for observe_head to seed one
        if deadline <= now:
            await self._fire_preview(ctx, key, preview, valid_at)

    def _log_fire_status(self, preview, valid_at, cur_block) -> None:
        """Throttled 'best candidate … submittable block X / current ~Y' line."""
        status_key = (valid_at, preview.get("decay_num"))
        if status_key == self._last_fire_status_key:
            return
        self._last_fire_status_key = status_key
        logger.info(
            "best candidate: floor=%.0f submittable at block %d (decay #%s); "
            "current ~block %s",
            float(preview.get("submit_floor_energy", 0.0)),
            valid_at,
            preview.get("decay_num"),
            cur_block if cur_block is not None else "?",
        )

    async def _fire_preview(
        self,
        ctx: SubstrateMiningContext,
        key: WorkKey,
        preview: dict,
        b_star: int,
    ) -> None:
        """Build the candidate proof from a preview and submit-with-retry.

        Branches on the typed :class:`SubmitRetryAction`:
          - ``SUCCESS``  → verify + record + mark key closed (stop firing).
          - ``RETRY``    → keep the preview; a later head fires again.
          - ``STOP_ROUND_STALE`` → evict preview + pending state (dead round).
          - ``STOP_FATAL``       → discard the candidate (await a better preview).
        """
        result = self._result_from_preview(ctx, preview)
        if result is None:
            return
        # Mark mid-fire BEFORE awaiting so a worker result that lands during
        # the submit round-trip is de-duped (treated as confirmation).
        self._anticipatory_fired.add(key)
        handle_id = str(preview.get("handle_id", "anticipatory"))
        logger.info(
            "anticipatory fire: work_key last_proof=0x%s... block=%d B*=%d "
            "floor=%.4f handle=%s",
            ctx.last_proof_block_hash.hex()[:16],
            ctx.block_number,
            b_star,
            float(preview.get("submit_floor_energy", 0.0)),
            handle_id,
        )
        submit_result = await submit_with_retry(
            self.build_client,
            self.pool_client,
            self.signer,
            result,
            ctx,
            tip=self.submission_config.tip_plancks,
            max_retries=self.submission_config.max_retries,
            retry_backoff_ms=self.submission_config.retry_backoff_ms,
        )

        if submit_result.action is SubmitRetryAction.SUCCESS:
            await self._record_anticipatory_success(
                ctx, key, preview, result,
                handle_id=handle_id, receipt=submit_result.receipt,
            )
            return
        if submit_result.action is SubmitRetryAction.RETRY:
            # Retries exhausted this fire only. Keep the preview and clear
            # the mid-fire mark so a later head (decay only eases further)
            # can fire again. Do NOT abandon.
            self._anticipatory_fired.discard(key)
            logger.info(
                "anticipatory fire RETRY-exhausted for work_key 0x%s... "
                "(attempts=%d, error=%s); will retry on a later head",
                ctx.last_proof_block_hash.hex()[:16],
                submit_result.attempts,
                submit_result.error,
            )
            return
        if submit_result.action is SubmitRetryAction.STOP_ROUND_STALE:
            # Nonce bound to a round that advanced — the candidate is dead.
            self.stats.stale_drops += 1
            # Audit parity with _handle_result's STALE path: write a
            # rejected_stale row (with chain-derived Sol#) so anticipatory
            # stale drops are visible in the submission log, then evict.
            await self._record_anticipatory_stale(
                ctx, result, preview, error=submit_result.error,
            )
            logger.info(
                "anticipatory fire STOP_ROUND_STALE for work_key 0x%s... "
                "(error=%s); discarding preview + pending state",
                ctx.last_proof_block_hash.hex()[:16],
                submit_result.error,
            )
            self._evict_anticipatory_state(key)
            return
        # STOP_FATAL — this candidate is genuinely bad; discard it and wait
        # for a better preview to supersede it.
        self.stats.submission_errors += 1
        self.stats.last_submission_error = submit_result.error
        logger.warning(
            "anticipatory fire STOP_FATAL for work_key 0x%s... (error=%s); "
            "discarding candidate, awaiting a better preview",
            ctx.last_proof_block_hash.hex()[:16],
            submit_result.error,
        )
        self._evict_anticipatory_state(key)

    async def _record_anticipatory_stale(
        self,
        ctx: SubstrateMiningContext,
        result: MiningResult,
        preview: dict,
        *,
        error: Optional[str],
    ) -> None:
        """Write a ``rejected_stale`` submission-log row for a STOP_ROUND_STALE fire.

        Mirrors ``_handle_result``'s STALE path so anticipatory stale drops
        are visible in the audit log (with the chain-derived Sol# from a
        best-effort ``proofs_submitted`` read), not just bumped in stats.
        ``miner_type`` is the real source backend carried through the preview.
        """
        pow_seq_stale = await self._query_proofs_submitted_safe()
        self._submission_log.record(
            solution_number=self._solution_number_for_context(ctx),
            miner_id=str(preview.get("handle_id", "anticipatory")),
            miner_type=result.miner_type,
            energy_milli=int(float(preview.get("energy", 0.0)) * 1000),
            diversity_milli=int(float(preview.get("diversity", 0.0)) * 1000),
            threshold_milli=int(ctx.difficulty.max_energy_milli),
            last_proof_block_hash_hex="0x" + ctx.last_proof_block_hash.hex(),
            outcome="rejected_stale",
            num_valid=int(preview.get("num_valid", 0)),
            pow_sequence=pow_seq_stale,
            error=str(error or ""),
        )

    def _result_from_preview(
        self, ctx: SubstrateMiningContext, preview: dict
    ) -> Optional[MiningResult]:
        """Reconstruct a ``MiningResult`` from a stored preview entry.

        The preview carries the submission-load-bearing fields the worker
        emitted (miner_type / nonce / salt / solutions / energy / num_valid /
        diversity); the remaining ``MiningResult`` fields are display-only and
        filled with neutral defaults. ``miner_type`` is the real source
        backend (cpu/gpu/qpu) so per-backend dashboard attribution stays
        accurate on the anticipatory path. Returns ``None`` on a malformed
        preview (logged) so the fire path degrades to a no-op rather than
        crashing.
        """
        try:
            return MiningResult(
                miner_id=str(preview.get("handle_id", "anticipatory")),
                miner_type=str(preview.get("miner_type", "UNKNOWN")),
                nonce=preview["nonce"],
                salt=preview["salt"],
                timestamp=0,
                prev_timestamp=0,
                solutions=preview["solutions"],
                energy=float(preview.get("energy", 0.0)),
                diversity=float(preview.get("diversity", 0.0)),
                num_valid=int(preview.get("num_valid", 0)),
                mining_time=0,
                node_list=list(ctx.nodes),
                edge_list=list(ctx.edges),
                submit_floor_energy=preview.get("submit_floor_energy"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "anticipatory: malformed preview for work_key 0x%s... "
                "(%s: %s); skipping fire",
                ctx.last_proof_block_hash.hex()[:16],
                type(exc).__name__,
                exc,
            )
            return None

    async def _resolve_accepted_block(
        self, receipt_block: Optional[str]
    ) -> Tuple[bytes, int]:
        """Resolve a receipt's block hash → ``(hash_bytes, block_number)``.

        Shared by the OK branch of ``_handle_result`` and the anticipatory
        path: a failure to decode or look up the number falls back to
        ``(b"", self._highest_handled_block + 1)`` — correct in the common
        case (we won at/just past the last processed head) and the
        subscription re-converges within a head or two.
        """
        accepted_block_number = self._highest_handled_block + 1
        accepted_block_hash = b""
        if not receipt_block:
            return accepted_block_hash, accepted_block_number
        try:
            accepted_block_hash = bytes.fromhex(
                receipt_block[2:]
                if receipt_block.startswith("0x")
                else receipt_block
            )
            accepted_block_number = await self.pool_client.get_block_number(
                at=accepted_block_hash
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not resolve accepted block number "
                "for receipt block=%s (%s: %s); using fallback=%d",
                receipt_block,
                type(exc).__name__,
                exc,
                accepted_block_number,
            )
        return accepted_block_hash, accepted_block_number

    async def _record_verify_fail(
        self,
        ctx: SubstrateMiningContext,
        key: Optional[WorkKey],
        log_common: dict,
        *,
        extrinsic_hash: Optional[str],
        receipt_block: Optional[str],
    ) -> None:
        """Record a verify-mismatch failure for both normal and anticipatory paths.

        Bumps ``proofs_unverified``, writes a ``chain_error`` submission-log
        row, and — when *key* is not ``None`` (anticipatory path) — clears the
        mid-fire mark and logs an anticipatory-specific warning.
        """
        self.stats.proofs_unverified += 1
        if key is not None:
            self._anticipatory_fired.discard(key)
        pow_seq_mismatch = await self._query_proofs_submitted_safe()
        self._submission_log.record(
            **log_common,
            outcome="chain_error",
            extrinsic_hash=extrinsic_hash,
            chain_block_hash=receipt_block,
            pow_sequence=pow_seq_mismatch,
            error="receipt OK but proof not recorded by chain",
        )
        if key is not None:
            logger.warning(
                "anticipatory fire receipt OK but verification failed for "
                "work_key 0x%s... (extrinsic=%s); NOT closing key",
                ctx.last_proof_block_hash.hex()[:16],
                extrinsic_hash,
            )

    async def _record_anticipatory_success(
        self,
        ctx: SubstrateMiningContext,
        key: WorkKey,
        preview: dict,
        result: MiningResult,
        *,
        handle_id: str,
        receipt: Optional[ExtrinsicReceipt],
    ) -> None:
        """Verify, record, and close a work key after a SUCCESS fire.

        ``result`` is the candidate proof already reconstructed (and
        validated non-``None``) by :meth:`_fire_preview`. Mirrors the OK
        branch of ``_handle_result``: run the post-OK proof-recorded verify,
        write a submission-log row (with num_valid + chain-derived Sol#),
        bump counters, mark the work key closed so sibling/worker results
        for it become no-ops, and cancel siblings.
        """
        envelope = _ResultEnvelope(
            result=result,
            context=ctx,
            handle_id=handle_id,
            dispatch_id=int(preview.get("dispatch_id", 0) or 0),
        )
        # Shared submission-log fields for both the success and verify-fail
        # rows — keeps the two record() calls in sync.
        log_common: dict[str, Any] = {
            "solution_number": self._solution_number_for_context(ctx),
            "miner_id": handle_id,
            # Real source backend (cpu/gpu/qpu) carried through the preview,
            # so per-backend dashboard attribution stays accurate — NOT the
            # "anticipatory" path marker.
            "miner_type": result.miner_type,
            "energy_milli": int(float(preview.get("energy", 0.0)) * 1000),
            "diversity_milli": int(float(preview.get("diversity", 0.0)) * 1000),
            "threshold_milli": int(ctx.difficulty.max_energy_milli),
            "last_proof_block_hash_hex": "0x" + ctx.last_proof_block_hash.hex(),
            "num_valid": int(preview.get("num_valid", 0)),
        }
        receipt_block = receipt.block_hash if receipt is not None else None
        extrinsic_hash = receipt.extrinsic_hash if receipt is not None else None

        verified = await self._verify_proof_recorded(envelope)
        # verified < 0 → chain recorded a different proof; treat as not-won.
        if verified is not None and verified < 0:
            await self._record_verify_fail(
                ctx, key, log_common,
                extrinsic_hash=extrinsic_hash,
                receipt_block=receipt_block,
            )
            return

        self.stats.proofs_submitted += 1
        accepted_block_hash, accepted_block_number = (
            await self._resolve_accepted_block(receipt_block)
        )
        record = ClosedWorkRecord(
            accepted_block_hash=accepted_block_hash,
            accepted_block_number=accepted_block_number,
            closed_at_monotonic=time.monotonic(),
        )
        self._mark_work_key_closed(key, record)
        self._cancel_siblings_for_won_work(handle_id)
        self._submission_log.record(
            **log_common,
            outcome="submitted_inblock",
            extrinsic_hash=extrinsic_hash,
            chain_block_hash=receipt_block,
            chain_block_number=self._resolve_won_block_number(
                verified, accepted_block_number
            ),
        )
        if self.core is not None:
            self.core.record_result(
                winning_miner_id=handle_id,
                mining_time=0.0,
            )
        logger.info(
            "anticipatory fire accepted: extrinsic=%s block=%s number=%d",
            receipt.extrinsic_hash if receipt is not None else None,
            receipt_block,
            accepted_block_number,
        )
        await self._invoke_proof_submitted_callback(receipt, ctx)

    # ------------------------------------------------------------------
    # Shared won-block helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_won_block_number(
        verified: Optional[int],
        accepted_block_number: int,
    ) -> int:
        """Return the authoritative won block number.

        Uses the verify-path's LastProofBlock when ``verified >= 0``; falls
        back to the receipt-derived ``accepted_block_number`` when verify
        returned ``None`` (inconclusive RPC failure).
        """
        if verified is not None and verified >= 0:
            return verified
        return accepted_block_number

    async def _invoke_proof_submitted_callback(
        self,
        receipt: Optional[ExtrinsicReceipt],
        ctx: SubstrateMiningContext,
    ) -> None:
        """Fire ``on_proof_submitted`` and swallow exceptions with a WARN.

        A ``None`` receipt or unset callback is a no-op.
        """
        if self.on_proof_submitted is None or receipt is None:
            return
        try:
            await self.on_proof_submitted(receipt, ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "on_proof_submitted callback raised (proof was submitted): "
                "%s: %s",
                type(exc).__name__,
                exc,
            )

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

    def _emit_dispatch_sentinel(self, handle: MinerHandle, dispatch_id: object) -> None:
        """Clear the active dispatch and push the done-sentinel onto the done queue.

        Called by the drain loop whenever a message signals that a dispatch is
        terminal (mine_result, work_item_done, error).  Centralises the
        identical three-line pattern that previously appeared in each branch.
        """
        if handle._active_dispatch_id == dispatch_id:
            handle._active_dispatch_id = 0
        self._done_queues[handle.miner_id].put_nowait(dispatch_id)

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
            if not isinstance(msg, dict):
                logger.warning(
                    "handle %s sent unrecognized message type=%s op=n/a; dropping",
                    handle.miner_id,
                    type(msg).__name__,
                )
                continue
            op = msg.get("op")
            if op == "mine_result":
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
                self._emit_dispatch_sentinel(handle, dispatch_id)
            elif op == "work_item_done":
                # Worker finished its mine_work_item loop with no result —
                # almost always because cancel() was observed. Surface it
                # tagged with the dispatch_id so _await_handle_done can
                # synchronize on cancellation of that specific dispatch,
                # not just any sentinel.
                self._emit_dispatch_sentinel(handle, msg.get("dispatch_id"))
            elif op == "error":
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
                self._emit_dispatch_sentinel(handle, msg.get("dispatch_id"))
            elif op == "preview":
                # Anticipatory-submission preview (Task 6a). Stash the
                # worker's latest best-by-floor candidate keyed by the
                # work key it was dispatched against. Do NOT build a
                # _ResultEnvelope and do NOT submit — that's Task 6b's
                # job; this drainer only delivers the primitive.
                self._store_preview(handle, msg)
            elif op == "budget":
                # Live QPU budget snapshot (worker-initiated push). Stash the
                # latest per-miner stats so the telemetry snapshot can surface
                # live usage; never blocks, never submits.
                self._store_budget(handle, msg)
            elif op == "participating":
                # Write-once participation marker for a solution #. Dedup +
                # submit a best-effort System.remark; never blocks the drain.
                self._mark_participating(handle.miner_id, msg)
            elif op == "stats":
                # Stats responses are pulled directly by callers of
                # handle.get_stats(); if one lands here it just means
                # nobody was listening — drop and continue. NOTE: while
                # the controller owns the handle, callers MUST NOT call
                # handle.get_stats() directly — the drainer dequeues the
                # response first and blocks the caller forever.
                pass
            else:
                logger.warning(
                    "handle %s sent unrecognized message type=dict op=%s; dropping",
                    handle.miner_id,
                    op,
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
        # Stop the free-running fire timer (Task 7).
        if self._fire_timer_task is not None:
            self._fire_timer_task.cancel()
        # Cancel the stats snapshot writer before awaiting tasks.
        if self._stats_writer_task is not None:
            self._stats_writer_task.cancel()
        # Await cancellations. Narrow catch: CancelledError is expected,
        # other exceptions are real cleanup failures worth surfacing.
        extras: list[asyncio.Task] = []
        if self._event_manager_task is not None:
            extras.append(self._event_manager_task)
        if self._fire_timer_task is not None:
            extras.append(self._fire_timer_task)
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
        self._fire_timer_task = None
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
        # handle is torn down by ``pool.shutdown()`` from the CLI's outer
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
