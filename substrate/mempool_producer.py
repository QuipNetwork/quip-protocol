"""MempoolJobProducer — handle-free mempool job discovery.

Extraction of `MempoolMinerController`'s per-block evaluation pipeline
(T5 of MEMPOOL_PRIORITY_PLAN): poll `System.Events` on every new block,
route `QuantumComputeMempool` events, filter `JobProposed` orders through
the eligibility guards, and queue accepted order ids for a consumer (the
T7 WorkScheduler; today the legacy controller keeps its own copy of the
polling shell until the T7 cutover deletes it).

The producer owns no handles and submits nothing. It is registered as a
``new_head`` callback on a caller-supplied ``ChainEventManager``::

    producer = MempoolJobProducer(pool_client=..., signer=..., ...)
    events.subscribe("new_head", producer.on_new_block)

Accepted order ids appear on ``producer.accepted`` (an ``asyncio.Queue``).
The consumer re-validates ``OrderStatus.OPENED`` at dispatch time — an
order can expire or close between acceptance and dequeue.

Guards (in evaluation order):

  - topology + mode (``job_matches_sampler``): ported unchanged from the
    controller — Blake2_256 topology-hash equality and the pallet's
    Bid OR-semantics (account OR solver_type);
  - ``OrderStatus.OPENED`` at discovery;
  - deadline (NEW): drop orders already expired or with fewer than
    ``deadline_margin_blocks`` blocks to effective expiry. The pallet's
    ``lifecycle.rs`` treats ``deadline_blocks`` as relative to
    ``created_at`` and tightens expiry to
    ``first_solution_at + block_wait`` once a first solution lands;
    expiry fires when ``now >= expiry``. Preemption latency (sentinel
    wait + possible driver respawn) makes near-deadline orders unwinnable.
  - min-reward (NEW): drop orders paying below ``min_reward``
    (``[miner] mempool_min_reward``; default 0 = accept all).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Set

from shared.allowed_value_spec import AllowedValueSpec
from shared.logging_config import get_logger
from shared.topology_hash import topology_hash
from substrate.mempool_types import JobOrder, MinerType, OrderStatus

if TYPE_CHECKING:
    from shared.signer import Signer
    from substrate.types import SubstrateMiningContext


logger = get_logger("mempool_producer")


# Minimum blocks-to-expiry an order must have left at discovery time.
# Cancel-ack + a possible driver respawn can eat a block or more, so
# orders closer than this to expiry are unwinnable in practice.
DEADLINE_MARGIN_BLOCKS = 2


def deadline_blocks_remaining(order: JobOrder, current_block: int) -> int:
    """Blocks until the order's effective expiry.

    Mirrors the pallet's ``lifecycle::effective_expiry``: the hard
    deadline is ``created_at + deadline_blocks`` (relative), tightened to
    ``first_solution_at + block_wait`` once a first solution exists. The
    pallet expires an order when ``now >= expiry``, so a return value
    ``<= 0`` means already expired.
    """
    expiry = order.created_at + order.timing.deadline_blocks
    if order.first_solution_at is not None:
        expiry = min(expiry, order.first_solution_at + order.timing.block_wait)
    return expiry - current_block


def job_matches_sampler(
    order: JobOrder,
    *,
    sampler_topology_hash: bytes,
    allowed_h_values: AllowedValueSpec,
    allowed_j_values: AllowedValueSpec,
    allowed_spin_values: AllowedValueSpec,
    account: bytes,
    solver_type: MinerType,
) -> bool:
    """Topology + mode eligibility filter.

    Mirrors the pallet's `solver_is_eligible` for mode; adds an upfront
    topology-hash check so we don't dispatch a job our sampler can't
    actually solve. Ported unchanged from
    ``MempoolMinerController._should_accept_job``.
    """
    # Topology match — sampler is bound to one specific graph.
    job_topology = topology_hash(
        order.ising_params.nodes,
        order.ising_params.edges,
        allowed_h_values,
        allowed_j_values,
        allowed_spin_values,
    )
    if job_topology != sampler_topology_hash:
        logger.debug(
            "skipping order topology mismatch: job=0x%s sampler=0x%s",
            job_topology.hex()[:16],
            sampler_topology_hash.hex()[:16],
        )
        return False

    # Mode filter — OR semantics on Bid (account OR solver_type).
    mode = order.mode
    if mode.tag == "Open":
        return True
    if mode.tag == "Bid":
        if mode.miners and account in mode.miners:
            return True
        if mode.miner_types and solver_type in mode.miner_types:
            return True
        return False
    return False


@dataclass
class MempoolProducerStats:
    """Operational counters surfaced via telemetry."""

    heads_observed: int = 0
    events_seen: int = 0
    jobs_filtered: int = 0
    jobs_deadline_dropped: int = 0
    jobs_reward_dropped: int = 0
    jobs_accepted: int = 0


class MempoolJobProducer:
    """Poll per-block mempool events and queue accepted order ids."""

    def __init__(
        self,
        *,
        pool_client: Any,
        signer: "Signer",
        sampler_topology_hash: bytes,
        allowed_h_values: AllowedValueSpec,
        allowed_j_values: AllowedValueSpec,
        allowed_spin_values: AllowedValueSpec,
        solver_type: MinerType,
        min_reward: int = 0,
        deadline_margin_blocks: int = DEADLINE_MARGIN_BLOCKS,
        on_order_expired: Optional[Callable[[int], None]] = None,
        stats: Optional[MempoolProducerStats] = None,
    ) -> None:
        """Build a producer over a live, swap-aware pool client.

        Args:
            pool_client: Duck-typed ``PoolClient`` — needs
                ``get_events_at(block_hash)`` and ``query_job_order(id)``.
            signer: Supplies the account for Bid-mode eligibility. Key
                material is never used to sign here.
            sampler_topology_hash: 32-byte canonical hash the sampler is
                bound to; jobs over any other graph are filtered.
            allowed_h_values / allowed_j_values / allowed_spin_values:
                Canonical-form specs the sampler hash was produced under.
            solver_type: Our registered on-chain solver type (vendor-
                resolved for QPU).
            min_reward: Drop orders with ``reward`` below this (0 = off).
            deadline_margin_blocks: Minimum blocks-to-expiry at discovery.
            on_order_expired: Invoked with the order id for every
                ``OrderExpired`` event; the consumer (submitter) decides
                whether the order is claimable.
            stats: Counter object; defaults to a fresh
                :class:`MempoolProducerStats`.
        """
        if len(sampler_topology_hash) != 32:
            raise ValueError(
                f"sampler_topology_hash must be 32 bytes, got "
                f"{len(sampler_topology_hash)}"
            )
        if min_reward < 0:
            raise ValueError(f"min_reward must be >= 0, got {min_reward}")
        self.pool_client = pool_client
        self.signer = signer
        self.sampler_topology_hash = sampler_topology_hash
        self.allowed_h_values = allowed_h_values
        self.allowed_j_values = allowed_j_values
        self.allowed_spin_values = allowed_spin_values
        self.solver_type = solver_type
        self.min_reward = min_reward
        self.deadline_margin_blocks = deadline_margin_blocks
        self.on_order_expired = on_order_expired
        self.stats = stats if stats is not None else MempoolProducerStats()
        # Accepted order ids, FIFO. The consumer re-checks OPENED at
        # dispatch time.
        self.accepted: asyncio.Queue[int] = asyncio.Queue()
        # Orders already accepted once — JobProposed dedup. Orders that
        # were *filtered* (not accepted) stay reconsiderable.
        self._pending_seen: Set[int] = set()

    async def on_new_block(self, ctx: Optional["SubstrateMiningContext"]) -> None:
        """ChainEventManager ``new_head`` callback.

        Only ``ctx.block_hash`` and ``ctx.block_number`` are read;
        everything else in the snapshot is PoW-only.
        """
        if ctx is None:
            logger.debug(
                "mempool producer: snapshot is None — no topology "
                "registered yet, skipping event poll"
            )
            return
        await self._process_head(ctx.block_hash, ctx.block_number)

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
            await self._handle_mempool_event(ev, block_number)

    async def _handle_mempool_event(self, ev: dict, block_number: int) -> None:
        event_id = ev["event_id"]
        attrs = ev["attributes"] or {}
        if not isinstance(attrs, dict):
            return
        if event_id == "JobProposed":
            order_id = int(attrs.get("order_id", -1))
            if order_id < 0 or order_id in self._pending_seen:
                return
            await self._consider_order(order_id, block_number)
        elif event_id == "OrderExpired":
            order_id = int(attrs.get("order_id", -1))
            if order_id < 0:
                return
            if self.on_order_expired is not None:
                self.on_order_expired(order_id)

    async def _consider_order(self, order_id: int, block_number: int) -> None:
        """Fetch a proposed order and decide whether to queue it."""
        try:
            order = await self.pool_client.query_job_order(order_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "query_job_order failed for order=%d: %s", order_id, exc
            )
            return
        if order is None:
            return
        if not self._matches_sampler(order):
            self.stats.jobs_filtered += 1
            return
        if order.status != OrderStatus.OPENED:
            return
        remaining = deadline_blocks_remaining(order, block_number)
        if remaining < self.deadline_margin_blocks:
            self.stats.jobs_deadline_dropped += 1
            logger.info(
                "dropping order=%d: %d blocks to expiry at block=%d "
                "(margin=%d)",
                order_id,
                remaining,
                block_number,
                self.deadline_margin_blocks,
            )
            return
        if order.reward < self.min_reward:
            self.stats.jobs_reward_dropped += 1
            logger.info(
                "dropping order=%d: reward=%d below mempool_min_reward=%d",
                order_id,
                order.reward,
                self.min_reward,
            )
            return
        self._pending_seen.add(order_id)
        self.stats.jobs_accepted += 1
        self.accepted.put_nowait(order_id)
        logger.info(
            "accepted JobProposed: order_id=%d reward=%d nodes=%d edges=%d "
            "blocks_to_expiry=%d",
            order_id,
            order.reward,
            len(order.ising_params.nodes),
            len(order.ising_params.edges),
            remaining,
        )

    def _matches_sampler(self, order: JobOrder) -> bool:
        return job_matches_sampler(
            order,
            sampler_topology_hash=self.sampler_topology_hash,
            allowed_h_values=self.allowed_h_values,
            allowed_j_values=self.allowed_j_values,
            allowed_spin_values=self.allowed_spin_values,
            account=self.signer.account_id_bytes(),
            solver_type=self.solver_type,
        )


__all__ = [
    "DEADLINE_MARGIN_BLOCKS",
    "MempoolJobProducer",
    "MempoolProducerStats",
    "deadline_blocks_remaining",
    "job_matches_sampler",
]
