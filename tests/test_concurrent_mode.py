"""Scheduler-stack wiring tests for the T7 cutover.

The `--mode pow|mempool|both` handle split is gone: `_prepare_core` builds
one `MinerCore` and `_build_scheduler_stack` puts a single `WorkScheduler`
over ALL of its handles — pow fills idle workers, mempool jobs preempt.
These tests pin the wiring:

  - 1-handle nodes run BOTH work sources (the old `--mode both` exit-5
    case is now the supported default);
  - mempool=False builds a pow-only stack (no producer/submitter glue);
  - the pow controller and the mempool stack share ONE ChainEventManager
    (the producer's per-block poll rides the pow controller's manager);
  - MEMPOOL_DISABLE parks the mempool stack while pow continues.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import quip_cli
from substrate.mempool_stack import MempoolStack
from substrate.mempool_submitter import SubmitOutcome, SubmitReport
from substrate.mempool_types import MempoolJobContext, MinerType, OrderStatus
from substrate.work_scheduler import WorkScheduler


class _StubHandle:
    """Duck-typed MinerHandle: identity + the scheduler's dispatch surface."""

    def __init__(self, name: str) -> None:
        self.miner_id = name
        self.miner_type = "CPU"
        self._active_dispatch_id = 0
        self.resp = MagicMock()

    def mine_work_item(self, context, *, solution_number=None) -> int:
        self._active_dispatch_id += 1
        return self._active_dispatch_id

    def cancel(self) -> None:
        pass


def _fake_keystore():
    ks = MagicMock()
    ks.signer.account_id_bytes.return_value = b"\x42" * 32
    ks.signer.ss58_address.return_value = "5Test"
    return ks


def _fake_binding(*, matches: bool = True):
    return SimpleNamespace(
        matches=matches,
        chain_hash=b"\xaa" * 32,
        expected_hash=b"\xbb" * 32,
        snapshot=SimpleNamespace(
            allowed_h_values=MagicMock(),
            allowed_j_values=MagicMock(),
            allowed_spin_values=MagicMock(),
        ),
    )


def _fake_pool():
    pool = MagicMock()
    pool.urls = ("ws://x:9944",)
    return pool


async def _build_stack(*, mempool_enabled: bool, num_handles: int = 1):
    client = MagicMock()
    client.resolve_topology_binding = AsyncMock(return_value=_fake_binding())
    core = MagicMock()
    core.miner_handles = [_StubHandle(f"h{i}") for i in range(num_handles)]
    return await quip_cli._build_scheduler_stack(
        client=client,
        pool=_fake_pool(),
        core=core,
        keystore=_fake_keystore(),
        topology=MagicMock(),
        miner_kind="cpu",
        telemetry_port=8086,
        submission_config=quip_cli.SubmissionConfig(),
        mempool_enabled=mempool_enabled,
        mempool_min_reward=7,
    )


# ---------------------------------------------------------------------------
# _prepare_core — no split, no exit-5
# ---------------------------------------------------------------------------


def test_prepare_core_builds_unsplit_core(monkeypatch):
    """One core, node_id 'quip-miner', all handles kept together."""
    built = {}

    class _FakeCore:
        def __init__(self, *, node_id, miners_config, topology):
            built.update(node_id=node_id, miners_config=miners_config)
            self.miner_handles = [_StubHandle("h0")]

        def close(self):
            built["closed"] = True

    monkeypatch.setattr(quip_cli, "MinerCore", _FakeCore)
    core = quip_cli._prepare_core("cpu", {"cpu": {"num_cpus": 1}}, MagicMock())
    assert built["node_id"] == "quip-miner"
    assert len(core.miner_handles) == 1
    assert "closed" not in built


def test_prepare_core_single_handle_no_longer_exits_5(monkeypatch):
    """Regression: 1 handle used to exit code 5 under --mode both."""

    class _FakeCore:
        def __init__(self, **_kw):
            self.miner_handles = [_StubHandle("only")]

        def close(self):
            raise AssertionError("must not close a healthy core")

    monkeypatch.setattr(quip_cli, "MinerCore", _FakeCore)
    core = quip_cli._prepare_core("cpu", {}, MagicMock())  # must not raise
    assert [h.miner_id for h in core.miner_handles] == ["only"]


def test_prepare_core_no_handles_exit_code_2(monkeypatch):
    class _FakeCore:
        def __init__(self, **_kw):
            self.miner_handles = []
            self.closed = False

        def close(self):
            self.closed = True

    cores = []
    monkeypatch.setattr(
        quip_cli, "MinerCore",
        lambda **kw: cores.append(_FakeCore(**kw)) or cores[-1],
    )
    with pytest.raises(quip_cli._MiningStartupError) as excinfo:
        quip_cli._prepare_core("cpu", {}, MagicMock())
    assert excinfo.value.code == 2
    assert cores[0].closed


def test_split_handles_for_mode_is_gone():
    """The mode split machinery is deleted, not deprecated."""
    assert not hasattr(quip_cli, "_split_handles_for_mode")
    assert not hasattr(quip_cli, "_MODE_HELP")
    assert "mode" not in quip_cli._MINING_DEFAULTS


# ---------------------------------------------------------------------------
# _build_scheduler_stack — one scheduler owns all handles
# ---------------------------------------------------------------------------


def test_stack_one_handle_serves_both_sources():
    """The previously-impossible 1-handle pow+mempool node builds fine."""
    controller, scheduler, stack = asyncio.run(
        _build_stack(mempool_enabled=True, num_handles=1)
    )
    assert stack is not None
    assert len(scheduler.miner_handles) == 1
    # The scheduler owns ALL handles; controller + stack delegate to it.
    assert controller.miner_handles is scheduler.miner_handles or (
        [h.miner_id for h in controller.miner_handles]
        == [h.miner_id for h in scheduler.miner_handles]
    )
    assert controller._scheduler is scheduler
    assert stack._scheduler is scheduler


def test_stack_mempool_disabled_is_pow_only():
    controller, scheduler, stack = asyncio.run(
        _build_stack(mempool_enabled=False, num_handles=2)
    )
    assert stack is None
    assert controller._scheduler is scheduler
    assert controller._head_subscribers == []
    assert scheduler._on_job_result is None


def test_stack_shares_one_event_manager_with_producer():
    """The producer's per-block poll rides the pow controller's manager."""
    controller, _scheduler, stack = asyncio.run(
        _build_stack(mempool_enabled=True)
    )
    assert controller._head_subscribers == [stack.producer.on_new_block]
    # min_reward from [miner] mempool_min_reward reaches the producer.
    assert stack.producer.min_reward == 7


def test_stack_topology_mismatch_exit_code_4():
    async def _go():
        client = MagicMock()
        client.resolve_topology_binding = AsyncMock(
            return_value=_fake_binding(matches=False)
        )
        core = MagicMock()
        core.miner_handles = [_StubHandle("h0")]
        await quip_cli._build_scheduler_stack(
            client=client,
            pool=_fake_pool(),
            core=core,
            keystore=_fake_keystore(),
            topology=MagicMock(),
            miner_kind="cpu",
            telemetry_port=8086,
            submission_config=quip_cli.SubmissionConfig(),
            mempool_enabled=False,
        )

    with pytest.raises(quip_cli._MiningStartupError) as excinfo:
        asyncio.run(_go())
    assert excinfo.value.code == 4


@pytest.mark.asyncio
async def test_one_event_manager_fans_to_both_consumers(monkeypatch):
    """ONE ChainEventManager: a snapshot fans to on_new_head AND the
    producer's on_new_block — no second poller is constructed."""
    import substrate.event_manager as em

    managers = []

    class _FakeManager:
        def __init__(self, **_kw):
            self.subscribers = []
            managers.append(self)

        def subscribe(self, event_type, callback):
            self.subscribers.append((event_type, callback))

        async def run(self):
            await asyncio.Event().wait()

        def request_shutdown(self):
            pass

    monkeypatch.setattr(em, "ChainEventManager", _FakeManager)
    controller, _scheduler, stack = await _build_stack(mempool_enabled=True)
    controller.pool = MagicMock()
    controller.pool.active_url.return_value = "ws://x:9944"
    await controller._start_event_manager(b"\x42" * 32)
    # Let the supervised tasks reach their first await so the wrapped
    # coroutines start (cancelling earlier leaves them never-awaited and
    # leaks RuntimeWarnings into the next test).
    await asyncio.sleep(0)
    try:
        assert len(managers) == 1
        callbacks = [cb for _, cb in managers[0].subscribers]
        assert controller.on_new_head in callbacks
        assert stack.producer.on_new_block in callbacks
    finally:
        controller._shutdown_event.set()
        for task in (controller._event_manager_task,
                     controller._fire_timer_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(t for t in (controller._event_manager_task,
                          controller._fire_timer_task) if t),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# MempoolStack — feed / submit / park behavior
# ---------------------------------------------------------------------------


def _order(status=OrderStatus.OPENED):
    return SimpleNamespace(
        status=status,
        ising_params=SimpleNamespace(
            nodes=(0, 1),
            edges=((0, 1),),
            h_values=(0, 0),
            j_values=(1000,),
            min_energy_milli=None,
            min_diversity_milli=None,
            min_solutions=None,
        ),
    )


def _stack_with_fakes():
    stack = MempoolStack(
        pool=_fake_pool(),
        signer=_fake_keystore().signer,
        sampler_topology_hash=b"\xbb" * 32,
        allowed_h_values=MagicMock(),
        allowed_j_values=MagicMock(),
        allowed_spin_values=MagicMock(),
        solver_type=MinerType.CPU,
        min_reward=0,
    )
    stack.pool_client = MagicMock()
    scheduler = MagicMock()
    stack.attach_scheduler(scheduler)
    return stack, scheduler


@pytest.mark.asyncio
async def test_feed_jobs_submits_context_with_revalidate():
    stack, scheduler = _stack_with_fakes()
    stack.pool_client.query_job_order = AsyncMock(return_value=_order())
    stack.producer.accepted.put_nowait(11)

    feed = asyncio.create_task(stack._feed_jobs())
    await asyncio.sleep(0.05)
    feed.cancel()
    await asyncio.gather(feed, return_exceptions=True)

    assert scheduler.submit_job.call_count == 1
    context = scheduler.submit_job.call_args.args[0]
    assert isinstance(context, MempoolJobContext)
    assert context.order_id == 11
    revalidate = scheduler.submit_job.call_args.kwargs["revalidate"]
    # Dispatch-time re-check: OPENED passes, CLOSED drops.
    assert await revalidate() is True
    stack.pool_client.query_job_order = AsyncMock(
        return_value=_order(status=OrderStatus.CLOSED)
    )
    assert await revalidate() is False


@pytest.mark.asyncio
async def test_mempool_disable_parks_stack_and_scheduler():
    """A mempool-fatal receipt parks mempool; nothing raises (pow lives)."""
    stack, scheduler = _stack_with_fakes()
    stack.submitter = MagicMock()
    stack.submitter.submit_solution = AsyncMock(
        return_value=SubmitReport(
            SubmitOutcome.MEMPOOL_DISABLE, "SolverNotRegistered"
        )
    )
    work = SimpleNamespace(
        context=SimpleNamespace(order_id=5),
        result=SimpleNamespace(miner_id="h0", mining_time=1.0),
    )
    await stack.on_job_result(work)  # queue-put contract: sync enqueue

    consume = asyncio.create_task(stack._consume_results())
    await asyncio.sleep(0.05)
    consume.cancel()
    await asyncio.gather(consume, return_exceptions=True)

    assert stack.parked is True
    scheduler.disable_mempool.assert_called_once()


@pytest.mark.asyncio
async def test_park_parks_producer_and_wakes_feed_loop():
    """park() must silence the whole discovery side, not just the feed.

    The producer stays subscribed to the shared ChainEventManager for
    the process lifetime, so park() has to make on_new_block a no-op
    (no more per-block RPCs, no unbounded `accepted` growth) — and the
    feed loop must exit promptly instead of fetching one more order
    into the disabled queue.
    """
    stack, scheduler = _stack_with_fakes()
    stack.pool_client.query_job_order = AsyncMock(return_value=_order())

    feed = asyncio.create_task(stack._feed_jobs())
    await asyncio.sleep(0.05)  # feed loop is blocked on accepted.get()
    stack.park("test")
    # An order accepted just before/while parking must NOT be fed.
    stack.producer.accepted.put_nowait(11)
    await asyncio.wait_for(feed, timeout=2)

    assert scheduler.submit_job.call_count == 0
    assert stack.pool_client.query_job_order.await_count == 0
    # The producer itself is parked: new heads poll nothing.
    stack.producer.pool_client = MagicMock()
    await stack.producer.on_new_block(
        SimpleNamespace(block_hash=b"\x10" * 32, block_number=10)
    )
    stack.producer.pool_client.get_events_at.assert_not_called()


@pytest.mark.asyncio
async def test_stack_run_survives_loop_crash_until_shutdown():
    """Failure containment: an internal crash parks mempool but run()
    returns only on shutdown — an early return would tear down pow via
    FIRST_COMPLETED."""
    stack, scheduler = _stack_with_fakes()
    stack.build_client = MagicMock()
    stack.build_client.connect = AsyncMock(
        side_effect=RuntimeError("connect boom")
    )
    stack.build_client.close = AsyncMock()

    run = asyncio.create_task(stack.run())
    await asyncio.sleep(0.05)
    assert not run.done()          # crash did NOT end run()
    assert stack.parked is True    # ...it parked mempool instead
    scheduler.disable_mempool.assert_called_once()
    stack.shutdown()
    await asyncio.wait_for(run, timeout=2)


# ---------------------------------------------------------------------------
# Scheduler still enforces first-wins fan-out with the CLI-built stack
# ---------------------------------------------------------------------------


def test_scheduler_stack_uses_workscheduler_type():
    controller, scheduler, _stack = asyncio.run(
        _build_stack(mempool_enabled=True, num_handles=3)
    )
    assert isinstance(scheduler, WorkScheduler)
    assert len(scheduler.miner_handles) == 3
    assert controller._scheduler is scheduler
