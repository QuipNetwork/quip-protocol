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
  - Subscribe to new heads (using a separate `SubstrateClient` instance —
    `substrate-interface` holds the websocket in receive mode during the
    subscription, so a shared connection deadlocks against `submit_extrinsic`).
  - On each new head: poll `System.Events` for `QuantumComputeMempool`
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

from websocket import WebSocketException

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
from shared.substrate_client import NoValidatorReachable, SubstrateClient
from shared.validator_pool import ValidatorPool


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


def topology_hash_from_nodes_edges(
    nodes: Tuple[int, ...], edges: Tuple[Tuple[int, int], ...]
) -> bytes:
    """Compute the chain's blake2_256 topology hash for a `(nodes, edges)` pair.

    Mirrors `pallets/quantum-pow/src/topology.rs::hash_topology` AND
    `quip_cli._topology_hash`; kept here so the mempool controller
    doesn't depend on the CLI module.
    """
    sorted_nodes = sorted(int(n) for n in nodes)
    sorted_edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
    )

    def compact_len(n: int) -> bytes:
        if n < 0x40:
            return bytes([n << 2])
        if n < 0x4000:
            return ((n << 2) | 0b01).to_bytes(2, "little")
        if n < 0x4000_0000:
            return ((n << 2) | 0b10).to_bytes(4, "little")
        raise ValueError(f"compact len {n} out of u32 range")

    buf = compact_len(len(sorted_nodes))
    for n in sorted_nodes:
        buf += n.to_bytes(4, "little")
    buf += compact_len(len(sorted_edges))
    for u, v in sorted_edges:
        buf += u.to_bytes(4, "little") + v.to_bytes(4, "little")
    return hashlib.blake2b(buf, digest_size=32).digest()


class MempoolMinerController:
    """Drive a fleet of `MinerHandle`s against the QuantumComputeMempool pallet."""

    def __init__(
        self,
        pool: ValidatorPool,
        signer: Signer,
        miner_handles: List[MinerHandle],
        sampler_topology_hash: bytes,
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
        # Slot clients lazy-resolved in run() — pool guarantees
        # 'rpc' and 'subscribe.mempool' are distinct SubstrateClient
        # instances, sidestepping the substrate-interface
        # subscribe-during-submit deadlock by construction.
        self.client: Optional[SubstrateClient] = None
        self._subscription_client: Optional[SubstrateClient] = None
        self.signer = signer
        self.miner_handles = miner_handles
        self.sampler_topology_hash = sampler_topology_hash
        self.solver_type = solver_type
        self.on_solution_submitted = on_solution_submitted
        self.on_reward_claimed = on_reward_claimed
        self.claim_poll_interval = claim_poll_interval
        self.core = core
        self.stats = MempoolControllerStats()

        # Per-handle dispatch tracking. When a result arrives the drainer
        # pairs it with the context the handle was given.
        self._dispatched: Dict[str, MempoolJobContext] = {}
        # Pending orders the controller has decided are eligible but hasn't
        # mined yet. FIFO — Phase 9 can swap in a priority queue when the
        # `--mempool-min-reward` / weighted-pick flags land.
        self._pending: deque[int] = deque()
        self._pending_seen: Set[int] = set()
        # Order currently being mined (one at a time in 8c).
        self._active_order: Optional[int] = None
        # Orders we've submitted a solution for; the chain will eventually
        # emit OrderExpired for these, at which point we try to claim.
        self._submitted_orders: Set[int] = set()
        # Orders for which we saw OrderExpired and have not yet claimed.
        self._claimable: Set[int] = set()
        # Latest-only head channel — same shape as the PoW controller's.
        self._latest_head: Optional[Tuple[bytes, int]] = None
        self._head_signal = asyncio.Event()
        self._result_queue: asyncio.Queue[_MempoolResultEnvelope] = asyncio.Queue()
        self._done_queues: Dict[str, asyncio.Queue[None]] = {
            h.miner_id: asyncio.Queue() for h in miner_handles
        }
        self._shutdown_event = asyncio.Event()
        self._drainer_tasks: List[asyncio.Task] = []
        self._subscription_task: Optional[asyncio.Task] = None
        self._claim_task: Optional[asyncio.Task] = None
        self._last_event_block: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any thread."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main loop. Returns on shutdown or fatal error."""
        # Resolve slot clients from the pool. Connections open here
        # (pool.get() lazy-connects), keeping __init__ synchronous
        # and side-effect free.
        self.client = await self._pool.get("rpc")
        self._subscription_client = await self._pool.get("subscribe.mempool")

        account = self.signer.account_id_bytes()
        await self._verify_solver_registered(account)

        for handle in self.miner_handles:
            task = asyncio.create_task(
                self._drain_handle(handle),
                name=f"mempool-drain-{handle.miner_id}",
            )
            self._drainer_tasks.append(task)

        self._subscription_task = asyncio.create_task(
            self._subscribe_heads(),
            name="mempool-head-subscription",
        )
        self._claim_task = asyncio.create_task(
            self._periodic_claim_loop(),
            name="mempool-reward-claim",
        )

        # Prime: also process any open orders that pre-date our subscription.
        await self._refresh_pending_from_storage()

        try:
            await self._main_loop()
        finally:
            await self._teardown()

    async def _teardown(self) -> None:
        self._shutdown_event.set()
        for handle in self.miner_handles:
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001 — best-effort
                pass

        # Close the subscription websocket BEFORE awaiting cancellations.
        # `subscribe_block_headers` runs in a thread that's blocked on the
        # underlying socket recv; without breaking the socket, asyncio's
        # task.cancel() doesn't unstick it and the default executor
        # eventually times out at 300s during loop shutdown. Closing the
        # websocket forces the recv to error out so the thread can exit.
        #
        # The slot is pool-owned, but close() is idempotent — pool.close()
        # at process shutdown will see _iface is already None for this
        # slot and skip cleanly.
        if self._subscription_client is not None:
            try:
                await self._subscription_client.close()
            except Exception:  # noqa: BLE001
                pass

        for task in [self._subscription_task, self._claim_task]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        for task in self._drainer_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    async def _verify_solver_registered(self, account: bytes) -> None:
        info: Optional[MempoolSolverInfo] = await self.client.query_solver(account)
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
    # Head subscription + event polling
    # ------------------------------------------------------------------

    async def _subscribe_heads(self) -> None:
        """Subscribe to new best heads with failover on validator drop.

        Mirrors `SubstrateMinerController._subscribe_heads`:
          - clean return → exit loop
          - `WebSocketException` / `ConnectionError` → one `reconnect()`
            on the subscription client, then re-subscribe
          - `NoValidatorReachable` from reconnect → fatal; record context
            and trigger shutdown
          - any other exception → fatal; record and shut down
        """

        async def _on_head(block_hash: bytes, block_number: int) -> None:
            self._latest_head = (block_hash, block_number)
            self._head_signal.set()

        while not self._shutdown_event.is_set():
            try:
                await self._subscription_client.subscribe_new_heads(_on_head)
                return
            except (WebSocketException, ConnectionError) as exc:
                logger.warning(
                    "mempool head subscription dropped on %s (%s: %s); "
                    "failing over",
                    getattr(
                        self._subscription_client, "current_url", "<unknown>"
                    ),
                    type(exc).__name__,
                    exc,
                )
                try:
                    await self._subscription_client.reconnect()
                except NoValidatorReachable as fatal:
                    logger.error(
                        "mempool subscription failover exhausted; "
                        "triggering shutdown:\n%s",
                        fatal,
                    )
                    self._shutdown_event.set()
                    return
                # loop iteration: re-subscribe on the new validator
            except Exception as exc:  # noqa: BLE001 — surface to logs
                if not self._shutdown_event.is_set():
                    logger.exception(
                        "mempool head subscription crashed: %s", exc
                    )
                    self._shutdown_event.set()
                return

    async def _refresh_pending_from_storage(self) -> None:
        """Pre-fill the pending queue from any currently-Opened orders.

        Without this, a miner that starts up between two head intervals
        would miss every order proposed before its subscription started.
        We bound the lookback to recent orders by walking `NextOrderId`
        downward and stopping at the first closed/expired one we hit, but
        that adds runtime API plumbing — for 8c just iterate forward from
        `0` and let the per-order open-check skip the closed ones.

        substrate-interface doesn't surface bulk storage iteration cleanly
        for us through `SubstrateClient`; this method is left as a no-op
        in 8c. The head-subscription event polling catches all
        post-startup orders, which is the common-case anyway.
        """
        # Phase 9 follow-on: add `iter_storage_map` to SubstrateClient and
        # walk JobOrders. For now, intentionally empty.
        return None

    async def _process_head(self, block_hash: bytes, block_number: int) -> None:
        """Poll events at the given block and route mempool ones."""
        self.stats.heads_observed += 1
        try:
            events = await self.client.get_events_at(block_hash)
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
            order = await self.client.query_job_order(order_id)
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
        job_topology = topology_hash_from_nodes_edges(
            order.ising_params.nodes, order.ising_params.edges
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
        """Block until either a new head arrives, a result lands, or shutdown."""
        head_task = asyncio.create_task(self._head_signal.wait())
        result_task = asyncio.create_task(self._result_queue.get())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        try:
            done, pending = await asyncio.wait(
                [head_task, result_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if shutdown_task in done:
                return
            if head_task in done:
                self._head_signal.clear()
                head = self._latest_head
                if head is not None:
                    block_hash, block_number = head
                    if block_hash != self._last_event_block:
                        self._last_event_block = block_hash
                        await self._process_head(block_hash, block_number)
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
            order = await self.client.query_job_order(order_id)
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
        for handle in self.miner_handles:
            self._dispatched[handle.miner_id] = context
            handle.mine_work_item(context)
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            self.core.record_dispatch()
        logger.info(
            "dispatched order_id=%d to %d handles",
            order_id,
            len(self.miner_handles),
        )

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
        # Matches `shared.substrate_submitter.encode_quantum_proof` shape.
        solutions_wrapped = ([(sol,) for sol in solutions],)

        try:
            receipt = await self.client.submit_extrinsic(
                "QuantumComputeMempool",
                "submit_solution",
                {
                    "order_id": order_id,
                    "solutions": solutions_wrapped,
                },
                self.signer,
                wait_for="inblock",
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.solution_errors += 1
            logger.exception(
                "submit_solution RPC failed for order=%d: %s", order_id, exc
            )
            self._active_order = None
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
            self._active_order = None
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
            self._active_order = None
        else:  # fatal
            self.stats.solution_errors += 1
            self._active_order = None
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
                receipt = await self.client.submit_extrinsic(
                    "QuantumComputeMempool",
                    "claim_reward",
                    {"order_id": order_id},
                    self.signer,
                    wait_for="inblock",
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
            except Exception:  # noqa: BLE001 — queue closed on shutdown
                return
            if isinstance(msg, MiningResult):
                context = self._dispatched.get(handle.miner_id)
                if context is None:
                    logger.warning(
                        "result from %s with no dispatched context; dropping",
                        handle.miner_id,
                    )
                    continue
                await self._result_queue.put(
                    _MempoolResultEnvelope(
                        result=msg,
                        context=context,
                        handle_id=handle.miner_id,
                    )
                )
            elif isinstance(msg, dict) and msg.get("op") == "work_item_done":
                try:
                    self._done_queues[handle.miner_id].put_nowait(None)
                except asyncio.QueueFull:
                    pass
            elif isinstance(msg, dict) and msg.get("op") == "error":
                logger.error(
                    "worker %s reported error: %s",
                    handle.miner_id,
                    msg.get("message", "<no message>"),
                )
                self.stats.solution_errors += 1
            elif isinstance(msg, dict) and msg.get("op") == "shutdown_ack":
                return

    # ------------------------------------------------------------------
    # Subscription cleanup (the substrate-interface subscription thread
    # holds the receive loop; if it dies before shutdown we surface it.)
    # ------------------------------------------------------------------


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
    "topology_hash_from_nodes_edges",
]
