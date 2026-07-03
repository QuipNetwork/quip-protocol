"""MempoolStack — glue between producer, WorkScheduler, and submitter.

The T7 cutover (MEMPOOL_PRIORITY_PLAN) replaced `MempoolMinerController`
with three pieces: `MempoolJobProducer` (per-block discovery), the
`WorkScheduler` (dispatch/preemption over the shared handles), and
`MempoolSubmitter` (submit_solution / claim_reward). This module is the
thin composition that connects them inside the single miner process:

  - the producer is subscribed to the pow controller's ONE
    ChainEventManager (see ``producer`` attribute; `quip_cli` passes
    ``producer.on_new_block`` as a ``head_subscribers`` entry);
  - the feed loop turns accepted order ids into `MempoolJobContext`s and
    calls ``scheduler.submit_job`` with a dispatch-time revalidation
    callback (order re-fetch + OPENED check — moved here from the legacy
    controller's ``_maybe_dispatch_next``);
  - the scheduler delivers winning job results to :meth:`on_job_result`,
    which queue-puts ONLY (the WorkScheduler callback contract: never RPC
    on the drainer task); the consume loop runs the actual submit;
  - a `SubmitOutcome.MEMPOOL_DISABLE` receipt parks the stack
    (``scheduler.disable_mempool()`` + feed loop stops) while pow mining
    continues — mempool failure must never take pow down.

Failure containment: :meth:`run` returns ONLY on :meth:`shutdown`. Any
internal crash logs, parks mempool, and keeps waiting — under the CLI's
FIRST_COMPLETED orchestration an early return would tear down the pow
controller (and, via the supervisor's first-child-exit rule, the node).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from shared.logging_config import get_logger
from substrate.client import SubstrateClient
from substrate.mempool_producer import MempoolJobProducer
from substrate.mempool_submitter import MempoolSubmitter, SubmitOutcome
from substrate.mempool_types import MempoolJobContext, OrderStatus
from substrate.pool_client import PoolClient

if TYPE_CHECKING:
    from shared.allowed_value_spec import AllowedValueSpec
    from shared.signer import Signer
    from substrate.mempool_types import MinerType
    from substrate.pool import ValidatorPool
    from substrate.work_scheduler import WorkResult, WorkScheduler


logger = get_logger("mempool_stack")


class MempoolStack:
    """Run the mempool side of the scheduler stack for one miner process."""

    def __init__(
        self,
        *,
        pool: "ValidatorPool",
        signer: "Signer",
        sampler_topology_hash: bytes,
        allowed_h_values: "AllowedValueSpec",
        allowed_j_values: "AllowedValueSpec",
        allowed_spin_values: "AllowedValueSpec",
        solver_type: "MinerType",
        min_reward: int = 0,
        core: Optional[Any] = None,
    ) -> None:
        """Compose producer + submitter over the swap-aware pool.

        Args:
            pool: The process's ``ValidatorPool``; reads and submits route
                through a ``PoolClient`` over it, extrinsics are composed
                on a parent-process ``build_client`` (connected in
                :meth:`run`; signer key material never crosses IPC).
            signer: Signs mempool extrinsics and supplies Bid eligibility.
            sampler_topology_hash: Canonical 32-byte hash the sampler is
                bound to (``binding.expected_hash``).
            allowed_h_values / allowed_j_values / allowed_spin_values:
                Canonical-form specs the hash was produced under.
            solver_type: Vendor-resolved on-chain solver type.
            min_reward: ``[miner] mempool_min_reward`` (0 = accept all).
            core: Optional MinerCore stats hook (``record_result``).
        """
        self.pool = pool
        self.signer = signer
        self.core = core
        self.pool_client = PoolClient(pool)
        self.build_client = SubstrateClient(urls=pool.urls)
        self.submitter = MempoolSubmitter(
            build_client=self.build_client,
            pool_client=self.pool_client,
            signer=signer,
        )
        self.producer = MempoolJobProducer(
            pool_client=self.pool_client,
            signer=signer,
            sampler_topology_hash=sampler_topology_hash,
            allowed_h_values=allowed_h_values,
            allowed_j_values=allowed_j_values,
            allowed_spin_values=allowed_spin_values,
            solver_type=solver_type,
            min_reward=min_reward,
            on_order_expired=self.submitter.note_order_expired,
        )
        self._scheduler: Optional["WorkScheduler"] = None
        # Winning job results from the scheduler, queue-put by
        # on_job_result and consumed (submitted) by _consume_results.
        self._result_queue: asyncio.Queue["WorkResult"] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        # Set on MEMPOOL_DISABLE: the feed loop stops handing jobs to the
        # scheduler (whose own queue is parked too). Pow continues.
        self._parked = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach_scheduler(self, scheduler: "WorkScheduler") -> None:
        """Bind the WorkScheduler (two-phase init; see quip_cli wiring)."""
        self._scheduler = scheduler

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any task."""
        self._shutdown_event.set()

    @property
    def parked(self) -> bool:
        """True once MEMPOOL_DISABLE parked the stack for this run."""
        return self._parked.is_set()

    async def run(self) -> None:
        """Run the feed / submit / claim loops until :meth:`shutdown`.

        Never returns early: internal crashes are logged and park the
        stack (pow must keep running under the shared FIRST_COMPLETED
        orchestration; an early return would tear it down).
        """
        if self._scheduler is None:
            raise RuntimeError(
                "MempoolStack requires a WorkScheduler; call "
                "attach_scheduler() before run()"
            )
        tasks: list[asyncio.Task] = []
        try:
            await self.build_client.connect()
        except Exception:  # noqa: BLE001 — mempool must not kill pow
            logger.exception(
                "mempool stack: build client connect failed; parking "
                "mempool for this run (pow continues)"
            )
            self.park("build client connect failed")
        else:
            for coro, name in (
                (self._feed_jobs(), "mempool-feed"),
                (self._consume_results(), "mempool-submit"),
                (self.submitter.run_claim_loop(self._shutdown_event),
                 "mempool-claim"),
            ):
                tasks.append(
                    asyncio.create_task(self._contain(coro, name), name=name)
                )
        try:
            await self._shutdown_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.build_client is not None:
                try:
                    await self.build_client.close()
                except Exception:  # noqa: BLE001 — teardown must complete
                    logger.exception("mempool stack: build_client.close raised")

    async def _contain(self, coro, name: str) -> None:
        """Failure containment wrapper: a crashed loop parks, never raises."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — mempool crash must not kill pow
            logger.exception(
                "mempool stack: %s loop crashed; parking mempool for this "
                "run (pow continues)",
                name,
            )
            self.park(f"{name} loop crashed")

    def park(self, reason: str) -> None:
        """Park the whole mempool side: producer, feed loop, job queue.

        The producer is parked too — it stays subscribed to the shared
        ChainEventManager, so without this it would keep issuing a
        per-block event poll and growing ``accepted`` for the rest of
        the run with nothing consuming it.
        """
        if self._parked.is_set():
            return
        self._parked.set()
        logger.error(
            "MEMPOOL PARKED for this run: %s — pow mining continues; "
            "no further mempool jobs will be dispatched",
            reason,
        )
        self.producer.park()
        if self._scheduler is not None:
            self._scheduler.disable_mempool()

    # ------------------------------------------------------------------
    # Scheduler consumer (queue-put contract)
    # ------------------------------------------------------------------

    async def on_job_result(self, work: "WorkResult") -> None:
        """WorkScheduler mempool consumer: enqueue-and-return.

        Runs inline on the delivering handle's drainer task — never
        submit here (a chain RPC would stall the drainer, freezing death
        detection and the done-sentinel emission a preempt may await).
        """
        self._result_queue.put_nowait(work)

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def _feed_jobs(self) -> None:
        """Turn accepted order ids into scheduler jobs with revalidation."""
        while not self._shutdown_event.is_set() and not self._parked.is_set():
            order_id = await self._next_accepted()
            if order_id is None:  # parked / shutting down
                return
            try:
                order = await self.pool_client.query_job_order(order_id)
            except Exception:  # noqa: BLE001 — skip this order, keep feeding
                logger.exception(
                    "query_job_order failed for order=%d; skipping", order_id
                )
                continue
            if order is None or order.status != OrderStatus.OPENED:
                logger.info(
                    "order=%d no longer eligible at feed time: status=%s",
                    order_id,
                    order.status if order else "not_found",
                )
                continue
            context = MempoolJobContext.from_job_order(order_id, order)

            async def _revalidate(oid: int = order_id) -> bool:
                # Dispatch-time OPENED re-check (moved from the legacy
                # controller): an order can expire/fill between feed and
                # dispatch — especially across the scheduler's
                # eligibility wait. A raise here makes the scheduler
                # drop the job (its documented revalidate contract).
                fresh = await self.pool_client.query_job_order(oid)
                return fresh is not None and fresh.status == OrderStatus.OPENED

            self._scheduler.submit_job(context, revalidate=_revalidate)

    async def _next_accepted(self) -> Optional[int]:
        """Next accepted order id, or None once parked / shut down.

        Races the queue get against park/shutdown so the feed loop exits
        promptly instead of idling on a dead queue — and never feeds one
        more order into the scheduler's parked job queue after park().
        """
        get = asyncio.ensure_future(self.producer.accepted.get())
        parked = asyncio.ensure_future(self._parked.wait())
        stopped = asyncio.ensure_future(self._shutdown_event.wait())
        try:
            await asyncio.wait(
                {get, parked, stopped}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._parked.is_set() or self._shutdown_event.is_set():
                return None
            return get.result()
        finally:
            for task in (get, parked, stopped):
                task.cancel()

    async def _consume_results(self) -> None:
        """Submit winning job results; park mempool on a fatal receipt."""
        while not self._shutdown_event.is_set():
            work = await self._result_queue.get()
            order_id = work.context.order_id
            report = await self.submitter.submit_solution(
                order_id, work.result
            )
            if report.outcome is SubmitOutcome.OK and self.core is not None:
                self.core.record_result(
                    winning_miner_id=work.result.miner_id,
                    mining_time=float(work.result.mining_time),
                )
            elif report.outcome is SubmitOutcome.MEMPOOL_DISABLE:
                self.park(
                    f"submit_solution mempool-fatal receipt for "
                    f"order={order_id}: {report.error}"
                )
            # STALE / FAILED: counted + logged by the submitter; move on.


__all__ = ["MempoolStack"]
