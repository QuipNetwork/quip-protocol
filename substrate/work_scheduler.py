"""Single-owner work scheduler: pow as idle filler, mempool jobs as priority.

Owns ALL of the process's `MinerHandle`s. Each handle's resp mp.Queue admits
exactly one drainer, so handles cannot be shared between two controllers —
this scheduler is the one owner, running one drainer task per handle and
routing every result envelope by its context type (`MempoolJobContext` to
the mempool consumer, anything else to the pow consumer).

The core protocol is the atomic preemption in :meth:`preempt_and_dispatch`:

    cancel() -> MANDATORY await of the victim's work_item_done sentinel
    -> dispatch

`MinerHandle.cancel()` only sets the shared per-handle stop_event, and the
NEXT `mine_work_item()` clears it (shared/miner_worker.py:453). Dispatching
before the worker acks therefore *wipes the cancel*: the old item keeps
mining and the new item queues behind it on the worker's serial FIFO req_q
— a priority inversion. Unlike the pow controller's historical 0.5s
best-effort `_await_handle_done`, the wait here has NO
timeout-then-dispatch-anyway: the ack is ~0.1s + one eval pass in the
streaming stack, so the wait is affordable; a handle that never acks is a
dead worker, which the drainer's death detection escalates (unblocking any
pending sentinel wait).

Dispatch policy:

- A mempool job fans out to ALL eligible handles — idle ones immediately,
  busy non-QPU ones via the preemption protocol. First `mine_result` wins;
  siblings are cancelled; terminal accounting compares done-handles against
  the fanned set only.
- QPU handles are never preempted (idle-only dispatch): the split D-Wave
  submitter has no ctl_q, so a preemption strands up to queue_depth
  already-paid jobs.
- A job whose every fanned dispatch terminates result-less (e.g. the QPU
  reservoir gate aborted the dispatch — `work_item_done` with no result)
  is treated as not-executed: requeued once, then dropped with a log.
- Pow is the idle filler: when a handle frees and no job is queued, the
  pow source's `provide_pow_context` callback supplies the current context
  (or None to leave the handle idle until the next head).

This module has no chain dependencies; every collaborator arrives as a
duck-typed handle or a callback, so the whole policy surface is
unit-testable with fakes.
"""
from __future__ import annotations

import asyncio
import logging
import queue as _queue
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from shared.miner_types import MiningResult
from substrate.mempool_types import MempoolJobContext

logger = logging.getLogger(__name__)

# How many dispatch contexts to retain per handle so late results from a
# cancelled dispatch can still find their context (mirrors the pow
# controller's ring).
_DISPATCH_CONTEXT_RETENTION = 4
# Progress-warning cadence while blocked on a victim's done sentinel.
_PREEMPT_WARN_INTERVAL_S = 5.0
# A result-less job is requeued this many times before being dropped.
_JOB_MAX_REQUEUES = 1


@dataclass(frozen=True)
class PowWork:
    """Pow idle-filler payload returned by ``provide_pow_context``."""

    context: Any
    solution_number: Optional[int] = None


@dataclass
class WorkResult:
    """A `MiningResult` paired with the context it was dispatched against.

    ``dispatch_id`` correlates with the worker-side attempt log via
    ``(handle_id, dispatch_id)`` — the same envelope shape both legacy
    controllers used.
    """

    result: MiningResult
    context: Any
    handle_id: str
    dispatch_id: int = 0


@dataclass
class SchedulerStats:
    """Cheap counters for tests, logs, and the telemetry snapshot."""

    contexts_dispatched: int = 0
    results_routed_pow: int = 0
    results_routed_job: int = 0
    stale_job_results_dropped: int = 0
    duplicate_job_results_dropped: int = 0
    jobs_requeued: int = 0
    jobs_dropped: int = 0
    worker_errors: dict[str, int] = field(default_factory=dict)


@dataclass
class _QueuedJob:
    """A mempool job queued for (re-)dispatch.

    ``requeues`` survives across dispatch rounds; everything else is
    per-round state reset by :meth:`reset_for_dispatch`.
    """

    context: Any
    revalidate: Optional[Callable[[], Awaitable[bool]]] = None
    requeues: int = 0
    fanned: dict[str, int] = field(default_factory=dict)
    terminal: set[str] = field(default_factory=set)
    winner: Optional[str] = None
    had_result: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def reset_for_dispatch(self) -> None:
        self.fanned = {}
        self.terminal = set()
        self.winner = None
        self.had_result = False
        self.done = asyncio.Event()


class WorkScheduler:
    """Single owner of all miner handles: drainers, preemption, dispatch.

    Callback contract: ``on_pow_result``/``on_job_result`` are awaited
    INLINE on the delivering handle's drainer task. They must be fast —
    enqueue-and-return. A callback that blocks (e.g. a chain-submit RPC)
    stalls that handle's drainer for the full duration: death detection,
    aux worker ops, and done-sentinel emission (which a preempt may be
    waiting on) all freeze with it. Consumers that submit to chain must
    wire a queue-put here and run the submit elsewhere — never bind
    ``submit_proof`` (or any RPC) directly.

    Args:
        miner_handles: The full handle set this process owns. Duck-typed:
            each needs ``mine_work_item``/``cancel``/``_active_dispatch_id``/
            ``resp``/``miner_id`` (plus optional ``proc`` for death
            detection and ``miner_type`` for the QPU policy).
        on_pow_result: Async callback receiving each pow `WorkResult`.
            Runs on the drainer task — see the callback contract above.
        on_job_result: Async callback receiving the winning mempool
            `WorkResult` (first result per job only). Runs on the
            drainer task — see the callback contract above.
        provide_pow_context: Async idle filler; returns a `PowWork` or
            None (handle stays idle until the next head).
        on_worker_message: Optional sync callback for non-terminal worker
            ops (preview / budget / participating / ...), so the pow
            controller keeps its brain without owning the drainer.
        on_fatal: Optional sync callback ``(handle_id, reason)`` invoked
            when a worker dies; the scheduler also sets its own shutdown
            flag so no caller can keep dispatching into a dead pool.
    """

    def __init__(
        self,
        miner_handles: list,
        *,
        on_pow_result: Optional[Callable[[WorkResult], Awaitable[None]]] = None,
        on_job_result: Optional[Callable[[WorkResult], Awaitable[None]]] = None,
        provide_pow_context: Optional[
            Callable[[], Awaitable[Optional[PowWork]]]
        ] = None,
        on_worker_message: Optional[Callable[[Any, dict], None]] = None,
        on_fatal: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError("WorkScheduler requires at least one miner handle")
        self.miner_handles = list(miner_handles)
        self._handles_by_id = {h.miner_id: h for h in self.miner_handles}
        if len(self._handles_by_id) != len(self.miner_handles):
            raise ValueError("miner handle ids must be unique")
        self._on_pow_result = on_pow_result
        self._on_job_result = on_job_result
        self._provide_pow_context = provide_pow_context
        self._on_worker_message = on_worker_message
        self._on_fatal = on_fatal
        self.stats = SchedulerStats()
        # Immutable (handle_id, dispatch_id) -> context map; late results
        # pair with the exact context they were produced against.
        self._dispatch_contexts: dict[tuple[str, int], Any] = {}
        # Per-handle done-sentinel queues for ALL handles. The drainer
        # pushes the terminal response's dispatch_id (or None on worker
        # death) so a preempt can block until the *specific* dispatch
        # being cancelled acks.
        self._done_queues: dict[str, asyncio.Queue] = {
            h.miner_id: asyncio.Queue() for h in self.miner_handles
        }
        self._dead_handles: set[str] = set()
        self._job_queue: asyncio.Queue[_QueuedJob] = asyncio.Queue()
        self._active_job: Optional[_QueuedJob] = None
        self._mempool_enabled = asyncio.Event()
        self._mempool_enabled.set()
        # Set whenever any handle frees (done sentinel emitted); the job
        # pump waits on it when every handle is QPU-and-busy.
        self._handle_freed = asyncio.Event()
        # Serializes preempt_and_dispatch calls: two concurrent preempts
        # over the same handle would steal each other's done sentinels.
        self._preempt_lock = asyncio.Lock()
        # Non-zero while a preempt is between cancel and dispatch; the
        # idle filler must not backfill into that window (it would trip
        # the dispatch invariant on the handle being preempted).
        self._preempting = 0
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn one drainer task per handle plus the job pump."""
        if self._tasks:
            raise RuntimeError("WorkScheduler already started")
        for handle in self.miner_handles:
            self._tasks.append(
                asyncio.create_task(
                    self._drain_handle(handle), name=f"drain-{handle.miner_id}"
                )
            )
        self._tasks.append(asyncio.create_task(self._job_pump(), name="job-pump"))

    async def stop(self) -> None:
        """Stop all scheduler tasks. Does not touch the worker processes."""
        self._shutdown_event.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def shutdown(self) -> None:
        """Signal shutdown without awaiting task teardown."""
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Public dispatch ops
    # ------------------------------------------------------------------

    def submit_job(
        self,
        context: Any,
        revalidate: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> None:
        """Queue a mempool job for priority dispatch.

        ``revalidate`` is an optional async callback run at dequeue time
        (order re-fetch + OPENED check); returning falsy — or raising —
        drops the job before any handle is touched.
        """
        self._job_queue.put_nowait(
            _QueuedJob(context=context, revalidate=revalidate)
        )

    def disable_mempool(self) -> None:
        """Park the job queue (MEMPOOL_DISABLE). Pow dispatch continues."""
        if not self._mempool_enabled.is_set():
            return
        logger.warning(
            "mempool dispatch disabled; %d queued job(s) parked, pow "
            "mining continues",
            self._job_queue.qsize(),
        )
        self._mempool_enabled.clear()

    async def dispatch_pow(
        self, context: Any, *, solution_number: Optional[int] = None
    ) -> dict[str, int]:
        """Broadcast pow dispatch for a work-key change.

        Preempts (cancel → sentinel → dispatch) every live handle NOT
        owned by the active mempool job — mempool is the priority source,
        so pow must never steal a job's handles; the job keeps running and
        pow takes the rest. The job-owned exclusion is evaluated INSIDE
        the preempt lock (``exclude_job_owned``): a snapshot taken here
        is stale by the time the lock is acquired — the job pump may be
        holding it mid-fan-out, and its ``fanned`` set only exists once
        its own dispatch returns. Busy QPU handles ARE preempted here: on
        a work-key change their in-flight pow work is dead either way.

        Returns ``{miner_id: dispatch_id}`` for every handle dispatched.
        """
        return await self.preempt_and_dispatch(
            self.miner_handles,
            context,
            solution_number=solution_number,
            exclude_job_owned=True,
        )

    async def fill_idle(
        self, context: Any, *, solution_number: Optional[int] = None
    ) -> dict[str, int]:
        """Dispatch *context* to idle, non-job-owned handles; never cancels.

        The pow verify-fail re-dispatch path: busy handles keep whatever
        they are mining. Takes the preempt lock so it cannot interleave
        with a preempt's cancel → sentinel → dispatch window (dispatching
        into that window would trip the busy-handle invariant).
        """
        async with self._preempt_lock:
            dispatched: dict[str, int] = {}
            if self._shutdown_event.is_set():
                return dispatched
            owned = self._job_owned_handle_ids()
            for handle in self.miner_handles:
                if (
                    handle.miner_id in owned
                    or handle.miner_id in self._dead_handles
                    or handle._active_dispatch_id != 0
                ):
                    continue
                dispatched[handle.miner_id] = self._dispatch_to_handle(
                    handle, context, solution_number=solution_number
                )
            return dispatched

    def cancel_pow_siblings(self, winning_handle_id: str) -> None:
        """Cancel every handle except the pow winner and the active job's.

        The pow submission-storm fix: once one proof is accepted, sibling
        handles mining the same work item must stop immediately. Mempool
        is the priority source, so the active job's fanned handles are
        spared — post-cutover the controller's handle list is ALL
        handles, and only the scheduler knows which ones the job owns.
        Cancel-only (no dispatch), so it needs no preempt lock: an extra
        cancel on a preempt victim is idempotent, and an idle handle's
        next ``mine_work_item`` clears the stop event.
        """
        owned = self._job_owned_handle_ids()
        for handle in self.miner_handles:
            if (
                handle.miner_id == winning_handle_id
                or handle.miner_id in owned
                or handle.miner_id in self._dead_handles
            ):
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

    def _job_owned_handle_ids(self) -> set[str]:
        """The active mempool job's fanned handle ids (empty when none).

        ``fanned`` is assigned synchronously with the job's dispatch (no
        awaits in between), so any event-loop caller observes either the
        pre-dispatch empty set or the complete final set — never a
        partial one.
        """
        job = self._active_job
        return set(job.fanned) if job is not None else set()

    def dispatch_context(self, handle_id: str, dispatch_id) -> Optional[Any]:
        """The immutable context dispatched as ``(handle_id, dispatch_id)``.

        Returns ``None`` when the dispatch is older than the retention
        window. Lets the pow controller pair non-terminal worker messages
        (previews) with the exact context they were produced against
        without owning its own dispatch bookkeeping.
        """
        return self._dispatch_contexts.get((handle_id, dispatch_id))

    async def preempt_and_dispatch(
        self,
        handles: list,
        context: Any,
        *,
        solution_number: Optional[int] = None,
        allow_qpu_victims: bool = True,
        exclude_job_owned: bool = False,
    ) -> dict[str, int]:
        """Atomically cancel busy handles, await their acks, then dispatch.

        The three steps must never be split across scheduler decisions:
        the dispatch is what actually stops backend compute (it carries
        the driver ``switch``), and a dispatch issued before the victim's
        done sentinel wipes the cancel (see module docstring).

        Args:
            allow_qpu_victims: When False (the mempool job path), busy
                QPU handles are re-filtered out INSIDE the preempt lock:
                eligibility computed before the lock is stale by the
                time the lock is acquired, and a QPU handle that became
                busy meanwhile must be neither cancelled nor dispatched
                to. The default True keeps broadcast preempts (e.g. the
                pow key-change path) able to cancel busy QPU consumers.
            exclude_job_owned: When True (the pow broadcast path), the
                active job's fanned handles are excluded INSIDE the
                preempt lock — same staleness class as above: the job
                pump may hold the lock mid-fan-out, and its ``fanned``
                set is assigned synchronously with the dispatch, so a
                lock waiter always observes the final set.

        Returns:
            ``{miner_id: dispatch_id}`` for every handle actually
            dispatched (dead and re-filtered handles are skipped).
        """
        async with self._preempt_lock:
            self._preempting += 1
            try:
                live = [
                    h for h in handles if h.miner_id not in self._dead_handles
                ]
                if not allow_qpu_victims:
                    live = [
                        h
                        for h in live
                        if h._active_dispatch_id == 0 or not self._is_qpu(h)
                    ]
                if exclude_job_owned:
                    owned = self._job_owned_handle_ids()
                    live = [h for h in live if h.miner_id not in owned]
                victims = [
                    (h, h._active_dispatch_id)
                    for h in live
                    if h._active_dispatch_id != 0
                ]
                for handle, _ in victims:
                    # Pre-cancel sentinels are stale by definition (the
                    # active dispatch has not emitted its terminal yet);
                    # purge them so only the real ack — or a death None —
                    # can satisfy the mandatory wait below.
                    self._drain_done_queue(handle.miner_id)
                    handle.cancel()
                if victims:
                    await asyncio.gather(
                        *(
                            self._await_done_sentinel(h, did)
                            for h, did in victims
                        )
                    )
                dispatched: dict[str, int] = {}
                for handle in live:
                    if (
                        handle.miner_id in self._dead_handles
                        or self._shutdown_event.is_set()
                    ):
                        continue
                    dispatched[handle.miner_id] = self._dispatch_to_handle(
                        handle, context, solution_number=solution_number
                    )
                return dispatched
            finally:
                self._preempting -= 1

    # ------------------------------------------------------------------
    # Preemption internals
    # ------------------------------------------------------------------

    def _drain_done_queue(self, handle_id: str) -> None:
        """Discard accumulated pre-cancel sentinels for *handle_id*.

        Called synchronously between victim selection and ``cancel()``
        (no await points in between, so no fresh sentinel can be lost):
        anything already queued belongs to an older dispatch and must
        not satisfy the mandatory wait — and unconsumed sentinels would
        otherwise accumulate for the process lifetime.
        """
        sentinel_queue = self._done_queues[handle_id]
        while True:
            try:
                sentinel_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _await_done_sentinel(self, handle, dispatch_id: int) -> None:
        """MANDATORY wait for the victim's work_item_done sentinel.

        No timeout-then-dispatch-anyway: proceeding early lets the next
        ``mine_work_item()``'s ``stop_event.clear()`` un-cancel the victim.
        Logs a progress WARNING every ``_PREEMPT_WARN_INTERVAL_S`` while
        waiting; a handle that never acks is a dead worker, which the
        drainer escalates (pushing a matching-any ``None`` sentinel and
        setting shutdown, either of which releases this wait).
        """
        sentinel_queue = self._done_queues[handle.miner_id]
        waited = 0.0
        while not self._shutdown_event.is_set():
            try:
                got = await asyncio.wait_for(
                    sentinel_queue.get(), timeout=_PREEMPT_WARN_INTERVAL_S
                )
            except asyncio.TimeoutError:
                waited += _PREEMPT_WARN_INTERVAL_S
                logger.warning(
                    "handle %s has not acked cancel of dispatch_id=%d after "
                    "%.1fs; still waiting (a worker that never acks is dead "
                    "— drainer death detection will escalate)",
                    handle.miner_id,
                    dispatch_id,
                    waited,
                )
                continue
            # None matches any wait (worker death); an older dispatch's
            # sentinel is already resolved and is discarded. A sentinel
            # newer than dispatch_id is impossible — no dispatch has been
            # issued on this handle since the cancel.
            if got is None or got == dispatch_id:
                return

    def _dispatch_to_handle(
        self, handle, context: Any, *, solution_number: Optional[int] = None
    ) -> int:
        """Dispatch one context, enforcing the busy-handle invariant."""
        if handle._active_dispatch_id != 0:
            raise RuntimeError(
                f"dispatch invariant violated: handle {handle.miner_id} is "
                f"busy (active_dispatch_id={handle._active_dispatch_id}); "
                "cancel and await its done sentinel first"
            )
        dispatch_id = handle.mine_work_item(
            context, solution_number=solution_number
        )
        self._dispatch_contexts[(handle.miner_id, dispatch_id)] = context
        self._prune_dispatch_contexts(handle.miner_id, dispatch_id)
        self.stats.contexts_dispatched += 1
        return dispatch_id

    def _prune_dispatch_contexts(
        self, handle_id: str, latest_dispatch_id: int
    ) -> None:
        """Trim per-handle dispatch contexts to the last N attempts."""
        cutoff = latest_dispatch_id - _DISPATCH_CONTEXT_RETENTION
        if cutoff <= 0:
            return
        stale = [
            (h, d)
            for (h, d) in self._dispatch_contexts
            if h == handle_id and d <= cutoff
        ]
        for key in stale:
            self._dispatch_contexts.pop(key, None)

    @staticmethod
    def _is_qpu(handle) -> bool:
        return str(getattr(handle, "miner_type", "")).upper().startswith("QPU")

    # ------------------------------------------------------------------
    # Job pump (one active job at a time; fan-out-all per job)
    # ------------------------------------------------------------------

    async def _job_pump(self) -> None:
        try:
            await self._job_pump_loop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a dead pump must fail loud
            logger.exception("job pump crashed; scheduler shutting down")
            self._shutdown_event.set()
            if self._on_fatal is not None:
                try:
                    self._on_fatal("job-pump", "job pump crashed")
                except Exception:  # noqa: BLE001 — escalation must complete
                    logger.exception("on_fatal callback raised")

    async def _job_pump_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await self._mempool_enabled.wait()
            job = await self._job_queue.get()
            if self._shutdown_event.is_set():
                return
            if not self._mempool_enabled.is_set():
                # Disabled while blocked on get(): park the job back.
                self._job_queue.put_nowait(job)
                continue
            await self._run_job(job)

    async def _run_job(self, job: _QueuedJob) -> None:
        job.reset_for_dispatch()
        self._active_job = job
        try:
            eligible = await self._wait_for_eligible_handles()
            if not eligible:  # shutdown while waiting
                return
            # Revalidate AFTER the eligibility wait: with every handle a
            # busy QPU the wait can block for minutes, and an order
            # fulfilled/expired meanwhile must not burn a dispatch (on
            # QPU, paid D-Wave samples). The revalidate callback must
            # observe the post-wait world.
            if not await self._revalidate_job(job):
                return
            fanned = await self.preempt_and_dispatch(
                eligible, job.context, allow_qpu_victims=False
            )
            # No awaits between the preempt returning and this assignment:
            # a drainer cannot observe a fanned dispatch before the job
            # knows its fanned set.
            job.fanned = dict(fanned)
            if not fanned:
                # Every candidate died between selection and dispatch.
                self._complete_active_job()
                return
            logger.info(
                "dispatched mempool job to %d handle(s): %s",
                len(fanned),
                sorted(fanned),
            )
            await self._wait_any(job.done, self._shutdown_event)
        finally:
            if self._active_job is job:
                self._active_job = None
            if not self._shutdown_event.is_set():
                await self._maybe_backfill()

    async def _revalidate_job(self, job: _QueuedJob) -> bool:
        if job.revalidate is None:
            return True
        try:
            ok = bool(await job.revalidate())
        except Exception:  # noqa: BLE001 — a bad order must not kill the pump
            logger.exception("mempool job revalidation raised; dropping job")
            self.stats.jobs_dropped += 1
            return False
        if not ok:
            logger.info(
                "mempool job no longer eligible at dispatch time; dropping"
            )
            self.stats.jobs_dropped += 1
        return ok

    async def _wait_for_eligible_handles(self) -> list:
        """Idle handles plus busy non-QPU handles (the preemption victims).

        QPU handles are never preempted, so when every handle is a busy
        QPU the job waits for one to free instead of dispatching.
        """
        while not self._shutdown_event.is_set():
            # Clear BEFORE computing eligibility so a sentinel landing
            # between the check and the wait re-sets the event.
            self._handle_freed.clear()
            idle: list = []
            preemptable: list = []
            for handle in self.miner_handles:
                if handle.miner_id in self._dead_handles:
                    continue
                if handle._active_dispatch_id == 0:
                    idle.append(handle)
                elif not self._is_qpu(handle):
                    preemptable.append(handle)
            if idle or preemptable:
                return idle + preemptable
            logger.info(
                "no eligible handles for mempool job (busy QPU handles are "
                "never preempted); waiting for a handle to free"
            )
            await self._wait_any(self._handle_freed, self._shutdown_event)
        return []

    def _complete_active_job(self) -> None:
        """Terminal accounting resolved: requeue-once, drop, or succeed."""
        job = self._active_job
        self._active_job = None
        if job is None:
            return
        if job.had_result:
            logger.info(
                "mempool job complete: winner=%s fanned=%d",
                job.winner,
                len(job.fanned),
            )
        elif self._shutdown_event.is_set():
            pass  # shutting down; no requeue churn
        elif job.requeues < _JOB_MAX_REQUEUES:
            job.requeues += 1
            self.stats.jobs_requeued += 1
            logger.warning(
                "all %d fanned dispatch(es) for a mempool job terminated "
                "without a result (budget-gate abort?); requeueing "
                "(attempt %d/%d)",
                len(job.fanned),
                job.requeues,
                _JOB_MAX_REQUEUES,
            )
            self._job_queue.put_nowait(job)
        else:
            self.stats.jobs_dropped += 1
            logger.warning(
                "mempool job still result-less after %d requeue(s); dropping",
                job.requeues,
            )
        job.done.set()

    # ------------------------------------------------------------------
    # Pow idle filler
    # ------------------------------------------------------------------

    async def _maybe_backfill(self) -> None:
        """Refill idle handles with pow work when no job wants them.

        Suppressed while a preempt is in flight (dispatching into that
        window would trip the busy-handle invariant), for handles in the
        active job's fanned set (they belong to the job), while a job
        has not yet fanned out (it is about to claim handles), and while
        a job is queued — with the exception that a *parked* queue
        (mempool disabled) never blocks pow. A handle OUTSIDE the fanned
        set (e.g. a busy QPU handle that finishes its pow item mid-job)
        backfills immediately.
        """
        if self._provide_pow_context is None:
            return
        idle = [h for h in self.miner_handles if self._backfill_eligible(h)]
        if not idle:
            return
        work = await self._provide_pow_context()
        if work is None:
            logger.debug(
                "pow source returned no context; %d handle(s) stay idle",
                len(idle),
            )
            return
        for handle in idle:
            # Re-check: state can change across the provide await.
            if not self._backfill_eligible(handle):
                continue
            self._dispatch_to_handle(
                handle, work.context, solution_number=work.solution_number
            )

    def _backfill_eligible(self, handle) -> bool:
        """Whether the pow idle filler may dispatch to *handle* now."""
        if self._shutdown_event.is_set() or self._preempting > 0:
            return False
        if (
            handle._active_dispatch_id != 0
            or handle.miner_id in self._dead_handles
        ):
            return False
        job = self._active_job
        if job is not None and (not job.fanned or handle.miner_id in job.fanned):
            # Fanned handles are the job's; a not-yet-fanned job is about
            # to claim handles — backfill must not steal either.
            return False
        if self._mempool_enabled.is_set() and not self._job_queue.empty():
            return False
        return True

    # ------------------------------------------------------------------
    # Drainers (ported from the pow + mempool controller drain loops)
    # ------------------------------------------------------------------

    async def _drain_handle(self, handle) -> None:
        loop = asyncio.get_running_loop()
        try:
            await self._drain_handle_loop(handle, loop)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a dead drainer must fail loud
            logger.exception(
                "drainer for %s crashed; escalating", handle.miner_id
            )
            self._escalate_handle_death(handle, "drainer crashed")

    async def _drain_handle_loop(
        self, handle, loop: asyncio.AbstractEventLoop
    ) -> None:
        while not self._shutdown_event.is_set():
            try:
                msg = await loop.run_in_executor(None, handle.resp.get, True, 0.5)
            except _queue.Empty:
                proc = getattr(handle, "proc", None)
                if proc is not None and not proc.is_alive():
                    logger.error(
                        "worker %s died unexpectedly: exitcode=%s",
                        handle.miner_id,
                        getattr(proc, "exitcode", None),
                    )
                    self._escalate_handle_death(handle, "worker process died")
                    return
                continue
            except (EOFError, BrokenPipeError, OSError) as exc:
                logger.error(
                    "handle %s response queue broken (%s: %s): worker "
                    "process likely died",
                    handle.miner_id,
                    type(exc).__name__,
                    exc,
                )
                self._escalate_handle_death(handle, "response queue broken")
                return
            if not isinstance(msg, dict):
                logger.warning(
                    "handle %s sent unrecognized message type=%s; dropping",
                    handle.miner_id,
                    type(msg).__name__,
                )
                continue
            await self._route_worker_message(handle, msg)

    async def _route_worker_message(self, handle, msg: dict) -> None:
        op = msg.get("op")
        if op == "mine_result":
            await self._handle_mine_result(handle, msg)
        elif op == "work_item_done":
            self._emit_dispatch_sentinel(handle, msg.get("dispatch_id"))
            self._note_terminal(handle.miner_id, msg.get("dispatch_id"))
            await self._maybe_backfill()
        elif op == "error":
            self.stats.worker_errors[handle.miner_id] = (
                self.stats.worker_errors.get(handle.miner_id, 0) + 1
            )
            logger.error(
                "miner %s reported error (count=%d): %s",
                handle.miner_id,
                self.stats.worker_errors[handle.miner_id],
                msg.get("message"),
            )
            self._emit_dispatch_sentinel(handle, msg.get("dispatch_id"))
            self._note_terminal(handle.miner_id, msg.get("dispatch_id"))
            await self._maybe_backfill()
        elif self._on_worker_message is not None:
            # Non-terminal ops (preview / budget / participating / stats)
            # belong to the pow controller's brain, not the scheduler.
            try:
                self._on_worker_message(handle, msg)
            except Exception:  # noqa: BLE001 — observer must not kill drain
                logger.exception(
                    "on_worker_message raised for %s op=%s",
                    handle.miner_id,
                    op,
                )
        else:
            logger.debug(
                "handle %s sent op=%s with no aux consumer; dropping",
                handle.miner_id,
                op,
            )

    async def _handle_mine_result(self, handle, msg: dict) -> None:
        dispatch_id = msg.get("dispatch_id")
        context = self._dispatch_contexts.get((handle.miner_id, dispatch_id))
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
            return
        envelope = WorkResult(
            result=result,
            context=context,
            handle_id=handle.miner_id,
            dispatch_id=int(dispatch_id) if dispatch_id is not None else 0,
        )
        is_job = isinstance(context, MempoolJobContext)
        # For a job result the claim (winner election + sibling cancel +
        # had_result) must precede terminal accounting, so a job whose
        # winner is the LAST fanned handle to go terminal still counts
        # as executed.
        deliver_job = self._claim_job_result(envelope) if is_job else False
        self._emit_dispatch_sentinel(handle, dispatch_id)
        self._note_terminal(handle.miner_id, envelope.dispatch_id)
        if is_job:
            if deliver_job:
                self.stats.results_routed_job += 1
                await self._deliver(self._on_job_result, envelope, "mempool")
        else:
            self.stats.results_routed_pow += 1
            await self._deliver(self._on_pow_result, envelope, "pow")
        await self._maybe_backfill()

    def _claim_job_result(self, envelope: WorkResult) -> bool:
        """First result wins: elect the winner and cancel its siblings.

        Returns True when the envelope should be delivered to the mempool
        consumer; stale (no matching active job) and duplicate (winner
        already elected) results are dropped with a log.
        """
        job = self._active_job
        if (
            job is None
            or job.fanned.get(envelope.handle_id) != envelope.dispatch_id
        ):
            self.stats.stale_job_results_dropped += 1
            logger.info(
                "dropping stale mempool result from %s (dispatch_id=%d)",
                envelope.handle_id,
                envelope.dispatch_id,
            )
            return False
        job.had_result = True
        if job.winner is not None:
            self.stats.duplicate_job_results_dropped += 1
            logger.info(
                "dropping duplicate mempool result from %s; first result "
                "already won (%s)",
                envelope.handle_id,
                job.winner,
            )
            return False
        job.winner = envelope.handle_id
        for handle_id, dispatch_id in job.fanned.items():
            if handle_id == envelope.handle_id:
                continue
            sibling = self._handles_by_id.get(handle_id)
            if sibling is None or handle_id in self._dead_handles:
                continue
            if sibling._active_dispatch_id == dispatch_id:
                try:
                    sibling.cancel()
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "sibling cancel failed for %s: %s", handle_id, exc
                    )
        return True

    async def _deliver(
        self,
        callback: Optional[Callable[[WorkResult], Awaitable[None]]],
        envelope: WorkResult,
        kind: str,
    ) -> None:
        if callback is None:
            logger.warning(
                "no %s consumer registered; dropping result from %s",
                kind,
                envelope.handle_id,
            )
            return
        try:
            await callback(envelope)
        except Exception:  # noqa: BLE001 — consumer errors must not kill drain
            logger.exception(
                "%s result consumer raised for %s; result dropped, "
                "scheduler continues",
                kind,
                envelope.handle_id,
            )

    # ------------------------------------------------------------------
    # Sentinels, terminal accounting, death escalation
    # ------------------------------------------------------------------

    def _emit_dispatch_sentinel(self, handle, dispatch_id) -> None:
        """Mark the dispatch terminal and release any blocked preempt."""
        if handle._active_dispatch_id == dispatch_id:
            handle._active_dispatch_id = 0
        self._done_queues[handle.miner_id].put_nowait(dispatch_id)
        self._handle_freed.set()

    def _note_terminal(self, handle_id: str, dispatch_id) -> None:
        """Account a terminal dispatch against the active job's fanned set."""
        job = self._active_job
        if job is None or not job.fanned:
            return
        if job.fanned.get(handle_id) != dispatch_id:
            return
        job.terminal.add(handle_id)
        if len(job.terminal) >= len(job.fanned):
            self._complete_active_job()

    def _escalate_handle_death(self, handle, reason: str) -> None:
        """A worker died: unblock every waiter, then fail loud.

        The pushed ``None`` sentinel matches ANY pending done-sentinel
        wait — a dead worker acks nothing, so the mandatory preempt wait
        must be released here or it would hang forever.
        """
        if handle.miner_id in self._dead_handles:
            return
        self._dead_handles.add(handle.miner_id)
        stale_dispatch = handle._active_dispatch_id
        handle._active_dispatch_id = 0
        # Shutdown first: the completion path must not requeue jobs into
        # a scheduler that is going down.
        self._shutdown_event.set()
        self._done_queues[handle.miner_id].put_nowait(None)
        self._handle_freed.set()
        self._note_terminal(handle.miner_id, stale_dispatch)
        logger.error(
            "handle %s escalated as dead (%s); scheduler shutting down",
            handle.miner_id,
            reason,
        )
        if self._on_fatal is not None:
            try:
                self._on_fatal(handle.miner_id, reason)
            except Exception:  # noqa: BLE001 — escalation must complete
                logger.exception("on_fatal callback raised")

    # ------------------------------------------------------------------
    # Small async utilities
    # ------------------------------------------------------------------

    @staticmethod
    async def _wait_any(*events: asyncio.Event) -> None:
        """Block until any of *events* is set."""
        tasks = [asyncio.ensure_future(event.wait()) for event in events]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
