"""MempoolMinerController — Phase 8c main loop.

Parallel to `SubstrateMinerController` but driven by
`QuantumComputeMempool.JobProposed` events rather than new chain heads.
Phase 8d will merge the two into a single controller capable of running
PoW and mempool concurrently against the same worker pool; for 8c the
two run independently.

Lifecycle:

    controller = MempoolMinerController(client, signer, miner_handles, ...)
    await controller.run()         # blocks until shutdown() or fatal error
    controller.shutdown()          # idempotent

Behavior summary:

  - Fail-fast at startup if the signer isn't in `QuantumComputeMempool.Solvers`.
  - Subscribe to chain events via the shared `ChainEventManager` — the same
    event source `SubstrateMinerController` uses. The manager polls
    `get_mining_snapshot` at adaptive cadence; the state key includes
    ``block_hash`` so it fires on every block (mempool reacts per-block).
  - On each new block: poll `System.Events` for `QuantumComputeMempool`
    activity, route `JobProposed` to the pending queue (after eligibility
    filtering) and `OrderExpired` to the claimable set.
  - When idle: dequeue the next pending order, fetch the full `JobOrder`,
    build a `MempoolJobContext`, dispatch to all attached `MinerHandle`s.
  - When a `MiningResult` arrives: submit via
    `QuantumComputeMempool.submit_solution`, mark the order as submitted,
    move on to the next pending order. Solutions are not buffered — one
    submission per attempt; multi-solution batching is a Phase 9 concern.
  - Periodically attempt `claim_reward` against expired orders we've
    contributed to.

Eligibility filter (`_should_accept_job`):

  - Topology match: job's `(nodes, edges)` hashed to the same Blake2_256
    digest as the controller's configured `sampler_topology_hash`. The
    sampler is bound to one topology; jobs over a different graph can't
    be mined.
  - Mode filter: `JobMode::Open` always passes;
    `JobMode::Bid{miners?, miner_types?}` passes if our account is in
    `miners` OR our registered solver type is in `miner_types` (matches
    the pallet's OR semantics in `solver_is_eligible`).
"""
from __future__ import annotations

import asyncio
import hashlib
import queue
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from shared.allowed_value_spec import AllowedValueSpec
from shared.asyncio_supervise import supervise
from shared.logging_config import get_logger
from shared.mempool_types import (
    JobOrder,
    MempoolJobContext,
    MempoolSolverInfo,
    MinerType,
    OrderStatus,
    solutions_to_scale,
)
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.signer import Signer
from shared.topology_hash import topology_hash
from substrate.client import SubstrateClient
from substrate.event_manager import ChainEventManager
from substrate.pool import ValidatorPool
from substrate.pool_client import PoolClient
from substrate.types import SubstrateMiningContext


logger = get_logger("mempool_miner_controller")


# Error name fragments classifying a submit_solution / claim_reward
# receipt. Match by substring against the receipt error message.
SOLUTION_STALE_ERRORS = (
    "OrderNotOpen",      # raced an expiry — drop, move on
    "OrderNotFound",     # raced a purge — drop, move on
    "InsufficientEnergy",
    "InsufficientDiversity",
    "InsufficientSolutions",
    "NotEligibleSolver",
)
SOLUTION_FATAL_ERRORS = (
    "SolverNotRegistered",  # operator deregistered out from under us
    "BadSignature",
    "BadProof",
)

CLAIM_STALE_ERRORS = (
    "OrderNotExpired",    # the order didn't expire yet — try again later
    "NotWinner",          # we didn't rank — give up on this order
    "AlreadyClaimed",
    "OrderNotFound",
)


@dataclass
class MempoolControllerStats:
    """Operational counters surfaced via telemetry."""

    heads_observed: int = 0
    events_seen: int = 0
    jobs_filtered: int = 0
    jobs_accepted: int = 0
    contexts_dispatched: int = 0
    results_received: int = 0
    solutions_submitted: int = 0
    solution_stale_drops: int = 0
    solution_errors: int = 0
    rewards_claimed: int = 0
    claim_errors: int = 0


@dataclass
class _MempoolResultEnvelope:
    """Tag a `MiningResult` with the order it was produced against."""

    result: MiningResult
    context: MempoolJobContext
    handle_id: str


class MempoolMinerController:
    """Drive a fleet of `MinerHandle`s against the QuantumComputeMempool pallet."""

    def __init__(
        self,
        pool: ValidatorPool,
        signer: Signer,
        miner_handles: List[MinerHandle],
        *,
        sampler_topology_hash: bytes,
        allowed_h_values: AllowedValueSpec,
        allowed_j_values: AllowedValueSpec,
        allowed_spin_values: AllowedValueSpec,
        solver_type: MinerType,
        on_solution_submitted: Optional[
            Callable[[int, MiningResult], Awaitable[None]]
        ] = None,
        on_reward_claimed: Optional[
            Callable[[int, int], Awaitable[None]]
        ] = None,
        claim_poll_interval: float = 30.0,
        core: Optional[Any] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "MempoolMinerController requires at least one MinerHandle"
            )
        if len(sampler_topology_hash) != 32:
            raise ValueError(
                f"sampler_topology_hash must be 32 bytes, got "
                f"{len(sampler_topology_hash)}"
            )
        self._pool = pool
        self.pool = pool
        # Parent-process SubstrateClient for compose+sign only (signer key
        # material never crosses an IPC boundary). Storage reads and
        # extrinsic submissions both go through ``pool_client`` (swap-
        # aware). Connections open in ``run()`` so ``__init__`` stays
        # synchronous and side-effect free.
        self.build_client: Optional[SubstrateClient] = None
        self.pool_client: Optional[PoolClient] = None
        self.signer = signer
        self.miner_handles = miner_handles
        self.sampler_topology_hash = sampler_topology_hash
        # Stored so `_should_accept_job` can hash incoming jobs with the
        # same canonical form the sampler hash was produced under.
        self.allowed_h_values = allowed_h_values
        self.allowed_j_values = allowed_j_values
        self.allowed_spin_values = allowed_spin_values
        self.solver_type = solver_type
        self.on_solution_submitted = on_solution_submitted
        self.on_reward_claimed = on_reward_claimed
        self.claim_poll_interval = claim_poll_interval
        self.core = core
        self.stats = MempoolControllerStats()

        # Per-(handle_id, dispatch_id) immutable context map. Mirrors
        # the substrate controller's design so a late result from a
        # cancelled dispatch can be paired with the exact context it
        # was produced against, not whatever's currently dispatched.
        self._dispatch_contexts: Dict[
            Tuple[str, int], MempoolJobContext
        ] = {}
        # Pending orders the controller has decided are eligible but hasn't
        # mined yet. FIFO — Phase 9 can swap in a priority queue when the
        # `--mempool-min-reward` / weighted-pick flags land.
        self._pending: deque[int] = deque()
        self._pending_seen: Set[int] = set()
        # Order currently being mined (one at a time in 8c).
        self._active_order: Optional[int] = None
        # Handles that have reported a terminal event (error or
        # work_item_done without a winning result) for `_active_order`.
        # `_active_order` is cleared only when this set covers every
        # handle — clearing on the first error would drop legitimate
        # sibling results that arrive a few ms later.
        self._active_order_done_handles: Set[str] = set()
        # Orders we've submitted a solution for; the chain will eventually
        # emit OrderExpired for these, at which point we try to claim.
        self._submitted_orders: Set[int] = set()
        # Orders for which we saw OrderExpired and have not yet claimed.
        self._claimable: Set[int] = set()
        self._result_queue: asyncio.Queue[_MempoolResultEnvelope] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        self._drainer_tasks: List[asyncio.Task] = []
        # ChainEventManager — set in run(); polls the validator pool and
        # dispatches `new_head` to `on_new_block` on every block.
        self.events: Optional[ChainEventManager] = None
        self._event_manager_task: Optional[asyncio.Task] = None
        self._claim_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any thread."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main loop. Returns on shutdown or fatal error."""
        # Build a parent-process SubstrateClient for compose+sign only;
        # reads and submissions both go through the swap-aware pool via
        # PoolClient. Signer key material never crosses the mp.Queue IPC
        # boundary.
        self.build_client = SubstrateClient(urls=self._pool.urls)
        await self.build_client.connect()
        self.pool_client = PoolClient(self._pool)
        # Pool must be live before any pool_client.send() — including the
        # startup registration check below. ``_start_event_manager`` skips
        # the redundant spawn via the existing ``active_url() is None`` guard.
        if self._pool.active_url() is None:
            await self._pool.start()

        account = self.signer.account_id_bytes()
        await self._verify_solver_registered(account)

        for handle in self.miner_handles:
            task = asyncio.create_task(
                self._drain_handle(handle),
                name=f"mempool-drain-{handle.miner_id}",
            )
            self._drainer_tasks.append(task)

        await self._start_event_manager(account)

        self._claim_task = asyncio.create_task(
            self._periodic_claim_loop(),
            name="mempool-reward-claim",
        )

        try:
            await self._main_loop()
        finally:
            await self._teardown()

    async def _start_event_manager(self, account: bytes) -> None:
        """Wire the shared ChainEventManager into mempool's per-block path.

        Mirrors ``SubstrateMinerController._start_event_manager``. The
        state key includes ``block_hash`` so every new block fires
        ``new_head`` — mempool must react to each block because each may
        carry new ``JobProposed`` / ``OrderExpired`` events. Same key on
        the PoW side is harmless: its ``on_new_head`` short-circuits on
        an unchanged work key.

        Args:
            account: The signer's raw 32-byte AccountId. Blake2-256 hashed
                here to derive the canonical miner identity the runtime
                expects for ``derive_nonce`` — kept consistent with the
                PoW path even though mempool doesn't consume the nonce.
        """
        canonical_miner = hashlib.blake2b(account, digest_size=32).digest()

        # Idempotency: in ``--mode both`` the PoW controller also calls
        # ``pool.start()``. Skip the redundant spawn (leaks a handle and
        # leaves the original orphaned), but stay correct when mempool is
        # the only controller running.
        if self.pool.active_url() is None:
            await self.pool.start()

        def state_key(snapshot):
            """Dedup key: fires on every new block.

            ``None`` snapshots collapse to a stable sentinel; the mempool
            path simply skips them (``on_new_block`` guards ``ctx is None``).
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
                "topology_hash": None,
            },
            blocktime_s=6.0,
        )
        self.events.subscribe("new_head", self.on_new_block)
        self._event_manager_task = asyncio.create_task(
            supervise(
                self.events.run(),
                name="mempool-chain-event-manager",
                on_failure=self._shutdown_event.set,
            ),
            name="mempool-chain-event-manager",
        )
        logger.info(
            "MempoolMinerController: ChainEventManager started"
        )

    async def on_new_block(
        self, ctx: Optional[SubstrateMiningContext]
    ) -> None:
        """Event-manager callback: route per-block mempool events.

        The shared ``ChainEventManager`` fires this with the snapshot
        currently returned by ``get_mining_snapshot`` (or ``None`` when
        no topology is registered). Mempool only needs ``ctx.block_hash``
        and ``ctx.block_number`` from it; everything else is PoW-only.

        Args:
            ctx: Snapshot for the latest best block, or ``None`` if the
                chain has no topology registered yet.
        """
        if ctx is None:
            logger.debug(
                "mempool on_new_block: snapshot is None — no topology "
                "registered yet, skipping event poll"
            )
            return
        await self._process_head(ctx.block_hash, ctx.block_number)

    async def _cancel_and_await(
        self, task: Optional[asyncio.Task], label: str
    ) -> None:
        """Cancel *task* and await it, logging any unexpected exception.

        No-ops when *task* is ``None`` or already done.
        """
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception(
                "teardown: %s task %s raised during cancellation",
                label,
                task.get_name(),
            )

    async def _teardown(self) -> None:
        self._shutdown_event.set()
        for handle in self.miner_handles:
            try:
                handle.cancel()
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "teardown: cancel failed for %s: %s: %s",
                    handle.miner_id,
                    type(exc).__name__,
                    exc,
                )

        # Signal the event manager to stop polling; its supervised task
        # observes the shutdown and exits cleanly. We still cancel below
        # as belt-and-suspenders in case the supervise wrapper is stuck.
        if self.events is not None:
            self.events.request_shutdown()

        for task in [self._event_manager_task, self._claim_task]:
            await self._cancel_and_await(task, "task")
        for task in self._drainer_tasks:
            await self._cancel_and_await(task, "drainer task")
        self._event_manager_task = None
        # Close the parent-side build client. The pool's active validator
        # handle is torn down by ``pool.close()`` from the CLI's outer
        # try/finally.
        if self.build_client is not None:
            try:
                await self.build_client.close()
            except Exception:  # noqa: BLE001 — log, don't mask teardown
                logger.exception("teardown: build_client.close() raised")
            self.build_client = None

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    async def _verify_solver_registered(self, account: bytes) -> None:
        info: Optional[MempoolSolverInfo] = await self.pool_client.query_solver(account)
        if info is None:
            raise RuntimeError(
                f"signer {self.signer.ss58_address()} is not registered as a "
                "QuantumComputeMempool solver — run "
                "`quip-miner register-solver --miner-type <kind>` first"
            )
        if info.solver_type != self.solver_type:
            raise RuntimeError(
                f"signer registered as {info.solver_type.name}, but controller "
                f"configured for {self.solver_type.name}. Deregister and "
                "re-register with the matching --miner-type."
            )
        logger.info(
            "solver verified: ss58=%s type=%s submitted=%d earned=%d",
            self.signer.ss58_address(),
            info.solver_type.name,
            info.solutions_submitted,
            info.rewards_earned,
        )

    # ------------------------------------------------------------------
    # Per-block event polling
    # ------------------------------------------------------------------

    async def _process_head(self, block_hash: bytes, block_number: int) -> None:
        """Poll events at the given block and route mempool ones."""
        self.stats.heads_observed += 1
        try:
            events = await self.pool_client.get_events_at(block_hash)
        except Exception as exc:  # noqa: BLE001 — surface but don't fatal-out
            logger.exception(
                "get_events_at failed for block=%d hash=0x%s: %s",
                block_number,
                block_hash.hex(),
                exc,
            )
            return
        for ev in events:
            if ev["module_id"] != "QuantumComputeMempool":
                continue
            self.stats.events_seen += 1
            await self._handle_mempool_event(ev)

    async def _handle_mempool_event(self, ev: dict) -> None:
        event_id = ev["event_id"]
        attrs = ev["attributes"] or {}
        if not isinstance(attrs, dict):
            return
        if event_id == "JobProposed":
            order_id = int(attrs.get("order_id", -1))
            if order_id < 0 or order_id in self._pending_seen:
                return
            await self._consider_order(order_id)
        elif event_id == "OrderExpired":
            order_id = int(attrs.get("order_id", -1))
            if order_id < 0:
                return
            if order_id in self._submitted_orders:
                self._claimable.add(order_id)

    async def _consider_order(self, order_id: int) -> None:
        """Fetch a proposed order and decide whether to mine it."""
        try:
            order = await self.pool_client.query_job_order(order_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "query_job_order failed for order=%d: %s", order_id, exc
            )
            return
        if order is None:
            return
        if not self._should_accept_job(order):
            self.stats.jobs_filtered += 1
            return
        if order.status != OrderStatus.OPENED:
            return
        self._pending.append(order_id)
        self._pending_seen.add(order_id)
        self.stats.jobs_accepted += 1
        logger.info(
            "accepted JobProposed: order_id=%d reward=%d nodes=%d edges=%d",
            order_id,
            order.reward,
            len(order.ising_params.nodes),
            len(order.ising_params.edges),
        )

    def _should_accept_job(self, order: JobOrder) -> bool:
        """Topology + mode eligibility filter.

        Mirrors the pallet's `solver_is_eligible` for mode; adds an
        upfront topology-hash check so we don't dispatch a job our
        sampler can't actually solve.
        """
        # Topology match — sampler is bound to one specific graph.
        job_topology = topology_hash(
            order.ising_params.nodes,
            order.ising_params.edges,
            self.allowed_h_values,
            self.allowed_j_values,
            self.allowed_spin_values,
        )
        if job_topology != self.sampler_topology_hash:
            logger.debug(
                "skipping order topology mismatch: job=0x%s sampler=0x%s",
                job_topology.hex()[:16],
                self.sampler_topology_hash.hex()[:16],
            )
            return False

        # Mode filter — OR semantics on Bid (account OR solver_type).
        mode = order.mode
        if mode.tag == "Open":
            return True
        if mode.tag == "Bid":
            account = self.signer.account_id_bytes()
            if mode.miners and account in mode.miners:
                return True
            if mode.miner_types and self.solver_type in mode.miner_types:
                return True
            return False
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await self._maybe_dispatch_next()
            await self._wait_for_event()

    async def _wait_for_event(self) -> None:
        """Block until either a result lands or shutdown is requested.

        Per-block wakeups are routed via ``on_new_block`` from the shared
        ``ChainEventManager`` (subscriber callback) — they don't need to
        race the result queue. The main loop only blocks on the two
        intrinsic states: "a result needs handling" and "shutdown".
        """
        result_task = asyncio.create_task(self._result_queue.get())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        try:
            done, pending = await asyncio.wait(
                [result_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if shutdown_task in done:
                return
            if result_task in done:
                envelope = result_task.result()
                await self._handle_result(envelope)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception(
                "unexpected error in main event loop: %s — initiating shutdown",
                exc,
            )
            self._shutdown_event.set()
            raise

    async def _maybe_dispatch_next(self) -> None:
        if self._active_order is not None:
            return
        if not self._pending:
            return
        order_id = self._pending.popleft()
        try:
            order = await self.pool_client.query_job_order(order_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "query_job_order failed for order=%d, requeueing: %s",
                order_id, exc,
            )
            self._pending.appendleft(order_id)
            return
        if order is None or order.status != OrderStatus.OPENED:
            logger.info(
                "order=%d not eligible at dispatch time: status=%s",
                order_id,
                order.status if order else "not_found",
            )
            return
        context = MempoolJobContext.from_job_order(order_id, order)
        self._active_order = order_id
        self._active_order_done_handles = set()
        for handle in self.miner_handles:
            dispatch_id = handle.mine_work_item(context)
            self._dispatch_contexts[(handle.miner_id, dispatch_id)] = context
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            self.core.record_dispatch()
        logger.info(
            "dispatched order_id=%d to %d handles",
            order_id,
            len(self.miner_handles),
        )

    def _clear_active_order(self) -> None:
        self._active_order = None
        self._active_order_done_handles = set()

    def _record_handle_terminal_for_active(
        self, handle_id: str, dispatch_id: int
    ) -> None:
        """Mark `handle_id` as done (error or cancel-completion) for the
        active order. Clears `_active_order` only when every handle has
        reported terminal — clearing earlier would cause `_handle_result`
        to drop legitimate sibling results that arrive moments later.
        """
        if self._active_order is None:
            return
        context = self._dispatch_contexts.get((handle_id, dispatch_id))
        if context is None or context.order_id != self._active_order:
            return
        self._active_order_done_handles.add(handle_id)
        if len(self._active_order_done_handles) >= len(self.miner_handles):
            logger.warning(
                "all %d handles terminated without submission for order=%d; "
                "releasing dispatch gate",
                len(self.miner_handles),
                self._active_order,
            )
            self._clear_active_order()

    async def _handle_result(self, envelope: _MempoolResultEnvelope) -> None:
        self.stats.results_received += 1
        order_id = envelope.context.order_id

        # Drop late results from a previously-canceled order.
        if order_id != self._active_order:
            logger.info(
                "dropping stale mempool result: order=%d (active=%s)",
                order_id,
                self._active_order,
            )
            return

        # Cancel siblings — we already have a solution for this order.
        for handle in self.miner_handles:
            if handle.miner_id != envelope.handle_id:
                try:
                    handle.cancel()
                except Exception:  # noqa: BLE001 — best-effort
                    pass

        solutions = solutions_to_scale(envelope.result.solutions)
        # The pallet bounds solutions to MaxSolutions=20; clip generously.
        if len(solutions) > 20:
            solutions = solutions[:20]

        # `solutions: BoundedVec<BoundedVec<i8, MaxNodes>, MaxSolutions>`
        # — both layers are 1-field composites in substrate metadata,
        # so each inner solution AND the outer list need 1-tuple wrapping.
        # Matches `substrate.submitter.encode_quantum_proof` shape.
        solutions_wrapped = ([(sol,) for sol in solutions],)

        try:
            extrinsic_hex = await self.build_client.build_signed_extrinsic(
                "QuantumComputeMempool",
                "submit_solution",
                {
                    "order_id": order_id,
                    "solutions": solutions_wrapped,
                },
                self.signer,
            )
            receipt = await self.pool_client.submit_signed_extrinsic(
                extrinsic_hex, wait_for="inblock",
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.solution_errors += 1
            logger.exception(
                "submit_solution RPC failed for order=%d: %s", order_id, exc
            )
            self._clear_active_order()
            return

        outcome = _classify_solution(receipt.error)
        if outcome == "ok":
            self.stats.solutions_submitted += 1
            self._submitted_orders.add(order_id)
            logger.info(
                "submit_solution accepted: order=%d extrinsic=%s",
                order_id,
                receipt.extrinsic_hash,
            )
            if self.core is not None:
                self.core.record_result(
                    winning_miner_id=envelope.result.miner_id,
                    mining_time=float(envelope.result.mining_time),
                )
            # Clear active order before callback so dispatch isn't blocked
            # if the callback raises.
            self._clear_active_order()
            if self.on_solution_submitted is not None:
                try:
                    await self.on_solution_submitted(order_id, envelope.result)
                except Exception as exc:
                    logger.exception(
                        "on_solution_submitted callback raised for order=%d: %s",
                        order_id, exc,
                    )
        elif outcome == "stale":
            self.stats.solution_stale_drops += 1
            logger.info(
                "submit_solution dropped as stale: order=%d error=%s",
                order_id,
                receipt.error,
            )
            self._clear_active_order()
        else:  # fatal
            self.stats.solution_errors += 1
            self._clear_active_order()
            raise RuntimeError(
                f"submit_solution failed fatally: order={order_id} "
                f"error={receipt.error}"
            )

    # ------------------------------------------------------------------
    # Reward claiming
    # ------------------------------------------------------------------

    async def _periodic_claim_loop(self) -> None:
        """Periodically attempt to claim any expired-but-unclaimed orders."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.claim_poll_interval,
                )
                return  # shutdown set
            except asyncio.TimeoutError:
                pass
            await self._claim_expired_orders()

    async def _claim_expired_orders(self) -> None:
        if not self._claimable:
            return
        to_remove: List[int] = []
        for order_id in list(self._claimable):
            try:
                extrinsic_hex = await self.build_client.build_signed_extrinsic(
                    "QuantumComputeMempool",
                    "claim_reward",
                    {"order_id": order_id},
                    self.signer,
                )
                receipt = await self.pool_client.submit_signed_extrinsic(
                    extrinsic_hex, wait_for="inblock",
                )
            except Exception as exc:  # noqa: BLE001 — keep trying other orders
                logger.exception(
                    "claim_reward RPC failed for order=%d: %s", order_id, exc
                )
                continue
            outcome = _classify_claim(receipt.error)
            if outcome == "ok":
                self.stats.rewards_claimed += 1
                to_remove.append(order_id)
                logger.info(
                    "claim_reward accepted: order=%d extrinsic=%s",
                    order_id,
                    receipt.extrinsic_hash,
                )
                if self.on_reward_claimed is not None:
                    await self.on_reward_claimed(order_id, 0)
            elif outcome == "stale":
                # OrderNotExpired -> retry on next tick.
                # NotWinner / AlreadyClaimed -> give up.
                err = receipt.error or ""
                if "OrderNotExpired" not in err:
                    to_remove.append(order_id)
            else:
                self.stats.claim_errors += 1
                logger.error(
                    "claim_reward failed fatally for order=%d: error=%s — giving up",
                    order_id,
                    receipt.error,
                )
                to_remove.append(order_id)
        for order_id in to_remove:
            self._claimable.discard(order_id)

    # ------------------------------------------------------------------
    # Worker queue drainer
    # ------------------------------------------------------------------

    async def _drain_handle(self, handle: MinerHandle) -> None:
        """Drain a worker's response queue and post to the controller queue."""
        loop = asyncio.get_running_loop()
        while not self._shutdown_event.is_set():
            try:
                msg = await loop.run_in_executor(
                    None, lambda: handle.resp.get(timeout=5.0)
                )
            except queue.Empty:
                if not handle.proc.is_alive():
                    logger.error(
                        "worker %s died unexpectedly: exitcode=%s",
                        handle.miner_id,
                        handle.proc.exitcode,
                    )
                    if not self._shutdown_event.is_set():
                        self._shutdown_event.set()
                continue
            except (EOFError, BrokenPipeError, OSError) as exc:
                # Pipe broken / worker gone. Mirror the substrate
                # controller's drainer: fail loud and trigger shutdown
                # rather than silently exiting (which would leave the
                # controller running with no signal that one handle's
                # results will never arrive).
                logger.error(
                    "handle %s response queue broken (%s: %s): "
                    "worker process likely died; shutting down",
                    handle.miner_id,
                    type(exc).__name__,
                    exc,
                )
                self._shutdown_event.set()
                return
            if isinstance(msg, dict) and msg.get("op") == "mine_result":
                dispatch_id = msg.get("dispatch_id")
                result = msg.get("result")
                context = self._dispatch_contexts.get(
                    (handle.miner_id, dispatch_id)
                )
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
                    _MempoolResultEnvelope(
                        result=result,
                        context=context,
                        handle_id=handle.miner_id,
                    )
                )
            elif isinstance(msg, dict) and msg.get("op") == "work_item_done":
                done_dispatch_id = msg.get("dispatch_id")
                # Cancel-completion counts as terminal for the active
                # order. Clears `_active_order` only if every handle has
                # reported terminal without a successful submission.
                self._record_handle_terminal_for_active(
                    handle.miner_id, done_dispatch_id
                )
            elif isinstance(msg, dict) and msg.get("op") == "error":
                err_dispatch_id = msg.get("dispatch_id")
                logger.error(
                    "worker %s reported error (dispatch=%s): %s",
                    handle.miner_id,
                    err_dispatch_id,
                    msg.get("message", "<no message>"),
                )
                self.stats.solution_errors += 1
                # Record this handle as terminal. We MUST NOT clear
                # `_active_order` here unconditionally — siblings may
                # still be mining the same order and `_handle_result`
                # gates on `order_id == self._active_order`, so an early
                # clear would drop legitimate sibling submissions as
                # stale. The helper only releases the gate when every
                # handle has reported terminal.
                self._record_handle_terminal_for_active(
                    handle.miner_id, err_dispatch_id
                )
            elif isinstance(msg, dict) and msg.get("op") == "shutdown_ack":
                return


def _classify_solution(error: Optional[str]) -> str:
    if error is None:
        return "ok"
    if any(name in error for name in SOLUTION_STALE_ERRORS):
        return "stale"
    # Unknown error strings are treated as fatal — includes SOLUTION_FATAL_ERRORS
    # and anything unrecognized (version mismatch, config error, etc.).
    return "fatal"


def _classify_claim(error: Optional[str]) -> str:
    if error is None:
        return "ok"
    if any(name in error for name in CLAIM_STALE_ERRORS):
        return "stale"
    return "fatal"


__all__ = [
    "MempoolControllerStats",
    "MempoolMinerController",
    "SOLUTION_FATAL_ERRORS",
    "SOLUTION_STALE_ERRORS",
    "CLAIM_STALE_ERRORS",
]
