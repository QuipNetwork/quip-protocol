"""Fake-handle unit suite for substrate.work_scheduler.WorkScheduler.

The scheduler is the single owner of all miner handles: one drainer per
handle, per-handle done-sentinel queues, an atomic
cancel -> await-sentinel -> dispatch preemption protocol, fan-out-all
mempool dispatch with first-result-wins, QPU idle-only policy, and a
requeue-once tolerance for budget-gate aborts.

Everything here runs against duck-typed fake handles (mine_work_item /
cancel / _active_dispatch_id / resp / req / miner_id) — no worker
processes, no chain. The un-cancel regression test is the load-bearing
one: MinerHandle.mine_work_item clears the shared stop_event
(shared/miner_worker.py:453), so a dispatch issued before the victim's
work_item_done sentinel wipes the cancel and wedges the new item behind
the old one on the serial req_q. The fakes model that stop_event
clear-on-dispatch and record a violation if the scheduler ever
dispatches to a still-busy handle.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.miner_types import MiningResult
from substrate.mempool_types import MempoolJobContext
from substrate.work_scheduler import PowWork, WorkScheduler


# ----------------------------------------------------------------------
# Fakes and helpers
# ----------------------------------------------------------------------


class FakeHandle:
    """Duck-typed MinerHandle: models the stop_event un-cancel wipe.

    ``mine_work_item`` clears ``stop_event`` exactly like the real handle
    (shared/miner_worker.py:453) and records a violation when called
    while a previous dispatch is still active — the condition the
    scheduler's dispatch invariant must make impossible.
    """

    def __init__(
        self,
        miner_id: str,
        *,
        miner_type: str = "CPU",
        events: list | None = None,
    ) -> None:
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.req: queue.Queue = queue.Queue()
        self.resp: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._next_dispatch_id = 0
        self._active_dispatch_id = 0
        self.cancel_calls = 0
        self.dispatched: list = []  # (context, solution_number)
        self.violations: list[str] = []
        self.events = events if events is not None else []
        self.alive = True
        self.proc = SimpleNamespace(
            is_alive=lambda: self.alive, exitcode=None
        )

    def mine_work_item(self, context, *, solution_number=None) -> int:
        if self._active_dispatch_id != 0:
            self.violations.append(
                f"dispatch while busy (active={self._active_dispatch_id})"
            )
        self._next_dispatch_id += 1
        self._active_dispatch_id = self._next_dispatch_id
        # Models the real handle's un-cancel wipe: a dispatch issued
        # before the previous cancel was acked erases that cancel.
        self.stop_event.clear()
        self.dispatched.append((context, solution_number))
        self.events.append(
            f"dispatch:{self.miner_id}:{type(context).__name__}"
        )
        return self._active_dispatch_id

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.stop_event.set()
        self.events.append(f"cancel:{self.miner_id}")

    # -- test-side worker simulation -----------------------------------

    def emit_done(self, dispatch_id: int) -> None:
        self.resp.put({"op": "work_item_done", "dispatch_id": dispatch_id})

    def emit_result(self, dispatch_id: int, result: MiningResult) -> None:
        self.resp.put(
            {"op": "mine_result", "dispatch_id": dispatch_id, "result": result}
        )

    def emit_error(self, dispatch_id: int, message: str = "boom") -> None:
        self.resp.put(
            {"op": "error", "dispatch_id": dispatch_id, "message": message}
        )

    def job_dispatches(self) -> list:
        return [
            ctx for ctx, _ in self.dispatched
            if isinstance(ctx, MempoolJobContext)
        ]


def _mining_result(miner_id: str = "h1") -> MiningResult:
    return MiningResult(
        miner_id=miner_id,
        miner_type="CPU",
        nonce=b"\x00" * 32,
        salt=b"",
        timestamp=0,
        prev_timestamp=0,
        solutions=[[1, -1]],
        energy=-1.0,
        diversity=0.5,
        num_valid=1,
        mining_time=1,
        node_list=[0, 1],
        edge_list=[(0, 1)],
    )


def _job_context(order_id: int = 1) -> MempoolJobContext:
    return MempoolJobContext(
        order_id=order_id,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
    )


def _pow_context() -> SimpleNamespace:
    """SubstrateMiningContext stand-in: routing is by NOT-MempoolJobContext."""
    return SimpleNamespace(last_proof_block_hash=b"\x01" * 32)


def _make_scheduler(handles, **overrides) -> WorkScheduler:
    kwargs = dict(on_pow_result=AsyncMock(), on_job_result=AsyncMock())
    kwargs.update(overrides)
    return WorkScheduler(handles, **kwargs)


async def wait_until(cond, *, timeout: float = 3.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{what} not met within {timeout}s")


@asynccontextmanager
async def running(sched: WorkScheduler):
    sched.start()
    try:
        yield sched
    finally:
        await sched.stop()


# ----------------------------------------------------------------------
# Result routing (typed) + drainer basics
# ----------------------------------------------------------------------


async def test_pow_result_routed_to_pow_consumer():
    handle = FakeHandle("h1")
    on_pow = AsyncMock()
    on_job = AsyncMock()
    sched = _make_scheduler([handle], on_pow_result=on_pow, on_job_result=on_job)
    async with running(sched):
        ctx = _pow_context()
        fanned = await sched.preempt_and_dispatch([handle], ctx)
        handle.emit_result(fanned["h1"], _mining_result())
        await wait_until(lambda: on_pow.await_count == 1, what="pow result routed")
    envelope = on_pow.await_args.args[0]
    assert envelope.handle_id == "h1"
    assert envelope.context is ctx
    assert envelope.dispatch_id == fanned["h1"]
    assert on_job.await_count == 0
    # A mine_result is terminal: the handle is idle again.
    assert handle._active_dispatch_id == 0


async def test_mine_result_with_unknown_dispatch_id_dropped():
    handle = FakeHandle("h1")
    on_pow = AsyncMock()
    sched = _make_scheduler([handle], on_pow_result=on_pow)
    async with running(sched):
        handle.emit_result(999, _mining_result())
        await asyncio.sleep(0.3)
    assert on_pow.await_count == 0


async def test_aux_worker_messages_forwarded():
    handle = FakeHandle("h1")
    seen: list = []
    sched = _make_scheduler(
        [handle], on_worker_message=lambda h, m: seen.append((h.miner_id, m))
    )
    async with running(sched):
        handle.resp.put({"op": "preview", "data": {"x": 1}})
        await wait_until(lambda: len(seen) == 1, what="aux message forwarded")
    assert seen == [("h1", {"op": "preview", "data": {"x": 1}})]


# ----------------------------------------------------------------------
# (a) job preempts pow, pow resumes after job terminal
# ----------------------------------------------------------------------


async def test_job_preempts_busy_pow_and_pow_resumes():
    events: list = []
    h1 = FakeHandle("h1", events=events)
    h2 = FakeHandle("h2", events=events)
    pow_ctx2 = _pow_context()
    provide = AsyncMock(return_value=PowWork(context=pow_ctx2, solution_number=7))
    on_job = AsyncMock()
    sched = _make_scheduler(
        [h1, h2], on_job_result=on_job, provide_pow_context=provide
    )
    job_ctx = _job_context()
    async with running(sched):
        pow_dispatch = await sched.preempt_and_dispatch([h1, h2], _pow_context())
        sched.submit_job(job_ctx)
        # Both busy pow handles get cancelled...
        await wait_until(
            lambda: h1.cancel_calls == 1 and h2.cancel_calls == 1,
            what="both pow victims cancelled",
        )
        # ...and nothing is dispatched until the workers ack.
        await asyncio.sleep(0.1)
        assert h1.job_dispatches() == [] and h2.job_dispatches() == []
        h1.emit_done(pow_dispatch["h1"])
        h2.emit_done(pow_dispatch["h2"])
        await wait_until(
            lambda: h1.job_dispatches() == [job_ctx]
            and h2.job_dispatches() == [job_ctx],
            what="job fanned to both handles",
        )
        # First result wins: h1 returns, sibling h2 is cancelled.
        h1.emit_result(h1._active_dispatch_id, _mining_result("h1"))
        await wait_until(lambda: h2.cancel_calls == 2, what="sibling cancelled")
        h2.emit_done(h2._active_dispatch_id)
        # Job terminal on the fanned set -> pow idle filler refills both.
        await wait_until(
            lambda: h1.dispatched[-1][0] is pow_ctx2
            and h2.dispatched[-1][0] is pow_ctx2,
            what="pow refilled after job terminal",
        )
    assert on_job.await_count == 1
    assert on_job.await_args.args[0].handle_id == "h1"
    assert h1.violations == [] and h2.violations == []
    # The refill carried the pow source's solution number through.
    assert h1.dispatched[-1] == (pow_ctx2, 7)


# ----------------------------------------------------------------------
# (b) UN-CANCEL REGRESSION: never dispatch before the done sentinel
# ----------------------------------------------------------------------


async def test_preempt_never_dispatches_before_done_sentinel():
    """A slow cancel ack must delay the dispatch, never be wiped by it.

    The ack takes 0.8s — longer than the pow controller's historical
    0.5s best-effort timeout. A scheduler that copies that
    timeout-then-dispatch-anyway shape dispatches into a busy handle:
    the fake records a violation and the event order breaks.
    """
    events: list = []
    handle = FakeHandle("h1", events=events)
    sched = _make_scheduler([handle])
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([handle], _pow_context())
        pow_did = fanned["h1"]

        async def ack_late() -> None:
            await wait_until(lambda: handle.cancel_calls >= 1)
            await asyncio.sleep(0.8)
            events.append("ack")
            handle.emit_done(pow_did)

        acker = asyncio.create_task(ack_late())
        sched.submit_job(_job_context())
        await wait_until(
            lambda: len(handle.job_dispatches()) == 1,
            what="job dispatched after ack",
        )
        await acker
    assert handle.violations == []
    ack_at = events.index("ack")
    job_at = events.index("dispatch:h1:MempoolJobContext")
    assert ack_at < job_at, f"dispatch preceded the cancel ack: {events}"


async def test_preempt_wait_logs_progress_warning(monkeypatch, caplog):
    import substrate.work_scheduler as ws

    monkeypatch.setattr(ws, "_PREEMPT_WARN_INTERVAL_S", 0.05)
    handle = FakeHandle("h1")
    sched = _make_scheduler([handle])
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([handle], _pow_context())
        sched.submit_job(_job_context())
        with caplog.at_level("WARNING", logger="substrate.work_scheduler"):
            await wait_until(lambda: handle.cancel_calls == 1)
            await asyncio.sleep(0.25)
            handle.emit_done(fanned["h1"])
            await wait_until(lambda: len(handle.job_dispatches()) == 1)
    assert any("has not acked cancel" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# (c) QPU handles: never cancelled, jobs only when idle
# ----------------------------------------------------------------------


async def test_qpu_handle_never_preempted_and_gets_job_when_idle():
    handle = FakeHandle("q1", miner_type="QPU")
    sched = _make_scheduler([handle])
    job_ctx = _job_context()
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([handle], _pow_context())
        sched.submit_job(job_ctx)
        await asyncio.sleep(0.3)
        # Busy QPU handle: no cancel, no dispatch — the job waits.
        assert handle.cancel_calls == 0
        assert handle.job_dispatches() == []
        # The pow item finishes on its own -> idle-only dispatch fires.
        handle.emit_done(fanned["q1"])
        await wait_until(
            lambda: handle.job_dispatches() == [job_ctx],
            what="job dispatched to idle QPU handle",
        )
    assert handle.cancel_calls == 0
    assert handle.violations == []


async def test_job_pump_never_cancels_qpu_that_became_busy_before_lock():
    """QPU-preemption leak regression: victims re-filtered INSIDE the lock.

    The job pump computes eligibility (idle QPU included) outside
    _preempt_lock. If another caller makes that QPU busy while the job's
    preempt waits on the lock, the in-lock victim recomputation must NOT
    cancel the now-busy QPU handle — the job path never preempts QPU.
    """
    cpu = FakeHandle("c1")
    qpu = FakeHandle("q1", miner_type="QPU")
    reached = asyncio.Event()

    async def revalidate() -> bool:
        reached.set()
        return True

    sched = _make_scheduler([cpu, qpu])
    async with running(sched):
        # Hold the preempt lock so the pump blocks AFTER computing
        # eligibility (both handles idle -> both eligible).
        await sched._preempt_lock.acquire()
        try:
            sched.submit_job(_job_context(), revalidate=revalidate)
            await asyncio.wait_for(reached.wait(), timeout=3.0)
            # The pump is now parked on _preempt_lock. A concurrent
            # caller (T7's pow key-change broadcast) dispatches pow to
            # the idle QPU, making it busy.
            qpu.mine_work_item(_pow_context())
        finally:
            sched._preempt_lock.release()
        await wait_until(
            lambda: len(cpu.job_dispatches()) == 1,
            what="job dispatched to cpu only",
        )
        assert qpu.cancel_calls == 0
        assert qpu.job_dispatches() == []
    assert qpu.violations == []


async def test_revalidate_runs_after_eligibility_wait():
    """A job that waited for handles re-checks OPENED after the wait.

    With a single busy QPU handle the eligibility wait blocks
    unboundedly; the revalidate callback must observe the post-wait
    world (order state can change while the job waits), not the state
    at dequeue time.
    """
    qpu = FakeHandle("q1", miner_type="QPU")
    state = {"post_wait": False}
    seen: list[bool] = []

    async def revalidate() -> bool:
        seen.append(state["post_wait"])
        return True

    sched = _make_scheduler([qpu])
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([qpu], _pow_context())
        sched.submit_job(_job_context(), revalidate=revalidate)
        await asyncio.sleep(0.3)
        # Blocked on eligibility (busy QPU): revalidate must not have run.
        assert seen == []
        state["post_wait"] = True
        qpu.emit_done(fanned["q1"])
        await wait_until(
            lambda: qpu.job_dispatches() != [], what="job dispatched"
        )
    assert seen == [True]


async def test_job_fans_to_preempted_cpu_but_never_busy_qpu():
    cpu = FakeHandle("c1")
    qpu = FakeHandle("q1", miner_type="QPU")
    on_job = AsyncMock()
    sched = _make_scheduler([cpu, qpu], on_job_result=on_job)
    job_ctx = _job_context()
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([cpu, qpu], _pow_context())
        sched.submit_job(job_ctx)
        await wait_until(lambda: cpu.cancel_calls == 1, what="cpu preempted")
        cpu.emit_done(fanned["c1"])
        await wait_until(
            lambda: cpu.job_dispatches() == [job_ctx],
            what="job dispatched to cpu only",
        )
        assert qpu.cancel_calls == 0
        assert qpu.job_dispatches() == []
        # Terminal accounting runs against the fanned set (cpu only):
        # the job completes even though the QPU handle is still busy.
        cpu.emit_result(cpu._active_dispatch_id, _mining_result("c1"))
        await wait_until(lambda: on_job.await_count == 1, what="job result routed")
        await wait_until(
            lambda: sched._active_job is None, what="job gate released"
        )
    assert qpu._active_dispatch_id == fanned["q1"]  # untouched


async def test_job_pump_crash_escalates_instead_of_dying_silently():
    """A raising handle.cancel() must not silently kill the job pump.

    The pump wears the same crash guard as the drainers: an unexpected
    exception logs, sets shutdown, and fires on_fatal('job-pump', ...)
    instead of leaving drainers running while jobs queue forever.
    """
    handle = FakeHandle("h1")

    def bad_cancel() -> None:
        raise RuntimeError("cancel exploded")

    handle.cancel = bad_cancel
    fatal: list = []
    sched = _make_scheduler(
        [handle], on_fatal=lambda hid, reason: fatal.append((hid, reason))
    )
    async with running(sched):
        # Make the handle a busy pow victim so the job path calls cancel().
        await sched.preempt_and_dispatch([handle], _pow_context())
        sched.submit_job(_job_context())
        await wait_until(lambda: len(fatal) == 1, what="pump crash escalated")
        assert sched._shutdown_event.is_set()
    assert fatal[0][0] == "job-pump"


async def test_preempt_drains_stale_done_sentinels_before_cancel():
    """Pre-cancel sentinels are purged so they can't satisfy the wait.

    Sentinels are consumed only during a preempt's mandatory wait, so a
    handle's done queue accumulates one per completed dispatch. Worse, a
    stale entry that carries the victim's CURRENT dispatch_id (e.g. a
    duplicate terminal emission) would release the mandatory wait before
    the worker acked the cancel — the exact un-cancel wipe the wait
    exists to prevent. Draining the victim's queue immediately before
    cancel() guarantees only the real ack (or a death None) satisfies
    the wait, and bounds queue growth.
    """
    handle = FakeHandle("h1")
    sched = _make_scheduler([handle])
    async with running(sched):
        first = await sched.preempt_and_dispatch([handle], _pow_context())
        handle.emit_done(first["h1"])
        await wait_until(
            lambda: sched._done_queues["h1"].qsize() == 1,
            what="stale sentinel parked in the done queue",
        )
        second = await sched.preempt_and_dispatch([handle], _pow_context())
        # Worst-case stale entry: a duplicate carrying the current id.
        sched._done_queues["h1"].put_nowait(second["h1"])
        sched.submit_job(_job_context())
        await wait_until(lambda: handle.cancel_calls == 1, what="victim cancelled")
        # Drained before cancel: only the real ack can release the wait,
        # so nothing may be dispatched yet.
        await asyncio.sleep(0.3)
        assert handle.job_dispatches() == []
        assert sched._done_queues["h1"].qsize() == 0
        handle.emit_done(second["h1"])
        await wait_until(
            lambda: len(handle.job_dispatches()) == 1, what="job dispatched"
        )
    assert sched._done_queues["h1"].qsize() == 0
    assert handle.violations == []


# ----------------------------------------------------------------------
# (d) budget-gate abort: requeue once, then drop
# ----------------------------------------------------------------------


async def test_resultless_job_requeued_once_then_dropped():
    handle = FakeHandle("h1")
    on_job = AsyncMock()
    sched = _make_scheduler([handle], on_job_result=on_job)
    async with running(sched):
        sched.submit_job(_job_context())
        await wait_until(
            lambda: len(handle.job_dispatches()) == 1, what="first dispatch"
        )
        handle.emit_done(handle._active_dispatch_id)
        await wait_until(
            lambda: len(handle.job_dispatches()) == 2, what="requeued dispatch"
        )
        handle.emit_done(handle._active_dispatch_id)
        await wait_until(
            lambda: sched.stats.jobs_dropped == 1, what="job dropped"
        )
        await asyncio.sleep(0.2)
    assert len(handle.job_dispatches()) == 2  # exactly one requeue
    assert sched.stats.jobs_requeued == 1
    assert on_job.await_count == 0
    assert sched._active_job is None


# ----------------------------------------------------------------------
# (e) first result wins: siblings cancelled, duplicates dropped
# ----------------------------------------------------------------------


async def test_first_result_wins_cancels_siblings_and_dedups():
    h1 = FakeHandle("h1")
    h2 = FakeHandle("h2")
    on_job = AsyncMock()
    sched = _make_scheduler([h1, h2], on_job_result=on_job)
    async with running(sched):
        sched.submit_job(_job_context())
        await wait_until(
            lambda: len(h1.job_dispatches()) == 1
            and len(h2.job_dispatches()) == 1,
            what="job fanned to both idle handles",
        )
        assert h1.cancel_calls == 0 and h2.cancel_calls == 0
        h1.emit_result(h1._active_dispatch_id, _mining_result("h1"))
        await wait_until(lambda: h2.cancel_calls == 1, what="sibling cancelled")
        # The cancelled sibling raced its own result out anyway.
        h2.emit_result(h2._active_dispatch_id, _mining_result("h2"))
        await wait_until(
            lambda: sched.stats.duplicate_job_results_dropped == 1,
            what="duplicate dropped",
        )
    assert on_job.await_count == 1
    assert on_job.await_args.args[0].handle_id == "h1"
    assert h1.cancel_calls == 0  # the winner is never cancelled


# ----------------------------------------------------------------------
# (f) idle filler: pow source returning None leaves handles idle
# ----------------------------------------------------------------------


async def test_non_fanned_handle_backfills_pow_mid_job():
    """A handle outside the active job's fanned set backfills mid-job.

    The busy QPU handle is never fanned; when its pow item finishes
    while the CPU job is still running, the idle filler must refill it
    with pow instead of leaving it idle until the job completes.
    """
    cpu = FakeHandle("c1")
    qpu = FakeHandle("q1", miner_type="QPU")
    pow_ctx2 = _pow_context()
    provide = AsyncMock(return_value=PowWork(context=pow_ctx2))
    sched = _make_scheduler([cpu, qpu], provide_pow_context=provide)
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([qpu], _pow_context())
        sched.submit_job(_job_context())
        await wait_until(
            lambda: len(cpu.job_dispatches()) == 1, what="job fanned to cpu"
        )
        # QPU finishes its pow item while the job is still running.
        qpu.emit_done(fanned["q1"])
        await wait_until(
            lambda: len(qpu.dispatched) == 2, what="qpu backfilled mid-job"
        )
        assert sched._active_job is not None  # job still in flight
        assert qpu.dispatched[-1][0] is pow_ctx2
        # Fanned handles stay owned by the job: cpu got no pow refill.
        assert len(cpu.dispatched) == 1
        cpu.emit_result(cpu._active_dispatch_id, _mining_result("c1"))
        await wait_until(lambda: sched._active_job is None, what="job complete")
    assert qpu.job_dispatches() == []
    assert qpu.violations == [] and cpu.violations == []


async def test_idle_filler_none_leaves_handles_idle():
    handle = FakeHandle("h1")
    provide = AsyncMock(return_value=None)
    sched = _make_scheduler([handle], provide_pow_context=provide)
    async with running(sched):
        fanned = await sched.preempt_and_dispatch([handle], _pow_context())
        handle.emit_done(fanned["h1"])
        await wait_until(lambda: provide.await_count >= 1, what="pow source asked")
        await asyncio.sleep(0.2)
    assert handle._active_dispatch_id == 0
    assert len(handle.dispatched) == 1  # only the original pow dispatch


# ----------------------------------------------------------------------
# (g) worker death during the mandatory sentinel wait escalates
# ----------------------------------------------------------------------


async def test_worker_death_during_sentinel_wait_escalates():
    handle = FakeHandle("h1")
    fatal: list = []
    sched = _make_scheduler(
        [handle], on_fatal=lambda hid, reason: fatal.append((hid, reason))
    )
    async with running(sched):
        await sched.preempt_and_dispatch([handle], _pow_context())
        sched.submit_job(_job_context())
        await wait_until(lambda: handle.cancel_calls == 1, what="victim cancelled")
        # The worker dies without ever acking the cancel.
        handle.alive = False
        await wait_until(lambda: len(fatal) == 1, what="death escalated")
        await wait_until(
            lambda: sched._shutdown_event.is_set(), what="scheduler shut down"
        )
    assert fatal[0][0] == "h1"
    assert handle.job_dispatches() == []  # never dispatched to a dead handle
    assert handle.violations == []


# ----------------------------------------------------------------------
# Dispatch invariant + context retention
# ----------------------------------------------------------------------


async def test_dispatch_invariant_rejects_busy_handle():
    handle = FakeHandle("h1")
    sched = _make_scheduler([handle])
    handle._active_dispatch_id = 42
    with pytest.raises(RuntimeError, match="busy"):
        sched._dispatch_to_handle(handle, _pow_context())


async def test_dispatch_context_retention_keeps_last_four():
    handle = FakeHandle("h1")
    sched = _make_scheduler([handle])
    for _ in range(6):
        sched._dispatch_to_handle(handle, _pow_context())
        handle._active_dispatch_id = 0  # simulate completion
    kept = sorted(d for (_, d) in sched._dispatch_contexts)
    assert kept == [3, 4, 5, 6]


# ----------------------------------------------------------------------
# Revalidation + disable_mempool
# ----------------------------------------------------------------------


async def test_revalidate_false_drops_job_without_dispatch():
    handle = FakeHandle("h1")
    revalidate = AsyncMock(return_value=False)
    sched = _make_scheduler([handle])
    async with running(sched):
        sched.submit_job(_job_context(), revalidate=revalidate)
        await wait_until(lambda: sched.stats.jobs_dropped == 1, what="job dropped")
        await asyncio.sleep(0.1)
    assert revalidate.await_count == 1
    assert handle.dispatched == []


async def test_disable_mempool_parks_jobs_but_pow_backfills():
    handle = FakeHandle("h1")
    pow_ctx2 = _pow_context()
    provide = AsyncMock(return_value=PowWork(context=pow_ctx2))
    sched = _make_scheduler([handle], provide_pow_context=provide)
    async with running(sched):
        sched.disable_mempool()
        sched.submit_job(_job_context())
        fanned = await sched.preempt_and_dispatch([handle], _pow_context())
        handle.emit_done(fanned["h1"])
        # Parked job never dispatches; the idle filler still runs.
        await wait_until(
            lambda: len(handle.dispatched) == 2, what="pow backfilled"
        )
        await asyncio.sleep(0.2)
    assert handle.job_dispatches() == []
    assert handle.dispatched[-1][0] is pow_ctx2
