"""MempoolSubmitter — submit_solution / claim_reward side of the mempool stack.

Receipt classification returns an outcome enum: the mempool-fatal error
class (SolverNotRegistered / BadSignature / BadProof / anything
unrecognized) maps to :attr:`SubmitOutcome.MEMPOOL_DISABLE` instead of
raising. Under the single-process scheduler stack a raise would tear down
pow mining too (and via the supervisor's first-child-exit rule, the whole
node); `substrate.mempool_stack` parks the mempool side on this signal
and pow continues.

Extrinsics are composed+signed on a parent-process ``build_client``
(signer key material never crosses an IPC boundary) and submitted through
the swap-aware ``pool_client``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional, Set

from substrateinterface.exceptions import SubstrateRequestException

from shared.logging_config import get_logger
from substrate.client import ExtrinsicRejected
from substrate.mempool_producer import deadline_blocks_remaining
from substrate.mempool_types import OrderStatus, solutions_to_scale

if TYPE_CHECKING:
    from shared.miner_types import MiningResult
    from shared.signer import Signer


logger = get_logger("mempool_submitter")


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

# The pallet bounds solutions to MaxSolutions=20; clip generously.
MAX_SOLUTIONS = 20

# Hard cap on one submit-and-watch attempt (mirrors the pow submitter's
# _SUBMIT_WATCH_TIMEOUT_S): a watch alive past this is a dead subscription.
_SUBMIT_WATCH_TIMEOUT_S = 90.0


class SubmitOutcome(Enum):
    """Terminal state of one submit_solution attempt."""

    OK = "ok"
    STALE = "stale"                      # raced expiry/purge/quality — drop
    FAILED = "failed"                    # transient RPC/build error — drop
    MEMPOOL_DISABLE = "mempool_disable"  # fatal class — park mempool, keep pow


class ClaimOutcome(Enum):
    """Terminal state of one claim_reward attempt."""

    OK = "ok"
    RETRY = "retry"        # OrderNotExpired — try again next tick
    GIVE_UP = "give_up"    # NotWinner/AlreadyClaimed/... — drop quietly
    FAILED = "failed"      # unrecognized error — drop loudly


def classify_submit_receipt(error: Optional[str]) -> SubmitOutcome:
    """Classify a submit_solution receipt error into an outcome.

    Unknown error strings map to MEMPOOL_DISABLE — includes
    SOLUTION_FATAL_ERRORS and anything unrecognized (version mismatch,
    config error, etc.).
    """
    if error is None:
        return SubmitOutcome.OK
    if any(name in error for name in SOLUTION_STALE_ERRORS):
        return SubmitOutcome.STALE
    return SubmitOutcome.MEMPOOL_DISABLE


def classify_claim_receipt(error: Optional[str]) -> ClaimOutcome:
    """Classify a claim_reward receipt error into an outcome."""
    if error is None:
        return ClaimOutcome.OK
    if "OrderNotExpired" in error:
        return ClaimOutcome.RETRY
    if any(name in error for name in CLAIM_STALE_ERRORS):
        return ClaimOutcome.GIVE_UP
    return ClaimOutcome.FAILED


@dataclass(frozen=True)
class SubmitReport:
    """Outcome of one submit_solution attempt plus the receipt error."""

    outcome: SubmitOutcome
    error: Optional[str] = None


@dataclass
class MempoolSubmitterStats:
    """Operational counters surfaced via telemetry.

    Any object with these attributes works (duck-typed) — the legacy
    controller passes its own ``MempoolControllerStats``.
    """

    solutions_submitted: int = 0
    solution_stale_drops: int = 0
    solution_errors: int = 0
    rewards_claimed: int = 0
    claim_errors: int = 0


class MempoolSubmitter:
    """Build, sign, and submit mempool extrinsics; claim expired rewards."""

    # Extra submit attempts after a txpool rejection (nonce race with a
    # sibling pow extrinsic from the same account); each retry recomposes
    # with a freshly read nonce. The backoff must span at least one block:
    # a lower-priority extrinsic cannot REPLACE the pooled sibling at the
    # same nonce ("Priority is too low"), it can only follow once the
    # sibling is included and the account nonce advances. Sub-block
    # backoffs re-read the same nonce and lose the same race every time.
    TXPOOL_RETRIES = 3
    txpool_retry_backoff_s = 3.0
    # ChargeTransactionPayment tip on mempool extrinsics. The shared signer
    # account continuously pools tip-0 pow extrinsics on an actively-winning
    # node, and a tip-0 submit_solution has lower priority — it can neither
    # replace the pooled sibling at its nonce nor win the recompose race
    # (starved indefinitely; observed live). Mempool is the priority work
    # source, so its extrinsics outrank same-account pow traffic at the
    # txpool layer too. 0.002 UNIT — noise next to any order reward.
    tip_plancks = 2_000_000_000

    def __init__(
        self,
        *,
        build_client: Any,
        pool_client: Any,
        signer: "Signer",
        on_solution_submitted: Optional[
            Callable[[int, "MiningResult"], Awaitable[None]]
        ] = None,
        on_reward_claimed: Optional[
            Callable[[int, int], Awaitable[None]]
        ] = None,
        claim_poll_interval: float = 30.0,
        stats: Optional[MempoolSubmitterStats] = None,
        submitted_orders: Optional[Set[int]] = None,
        claimable: Optional[Set[int]] = None,
    ) -> None:
        """Build a submitter over live clients.

        Args:
            build_client: Parent-process ``SubstrateClient`` for
                compose+sign only (duck-typed:
                ``build_signed_extrinsic``).
            pool_client: Swap-aware ``PoolClient`` (duck-typed:
                ``submit_signed_extrinsic``).
            signer: Signs in-process; key material never crosses IPC.
            on_solution_submitted: Awaited after an accepted submission;
                exceptions are logged, not propagated.
            on_reward_claimed: Awaited after an accepted claim.
            claim_poll_interval: Cadence of :meth:`run_claim_loop`.
            stats: Counter object (see :class:`MempoolSubmitterStats`).
            submitted_orders / claimable: Mutable containers for order
                bookkeeping; caller-suppliable for tests that want to
                observe or seed the sets.
        """
        self.build_client = build_client
        self.pool_client = pool_client
        self.signer = signer
        self.on_solution_submitted = on_solution_submitted
        self.on_reward_claimed = on_reward_claimed
        self.claim_poll_interval = claim_poll_interval
        self.stats = stats if stats is not None else MempoolSubmitterStats()
        # Orders we've submitted a solution for; the chain will eventually
        # emit OrderExpired for these, at which point we try to claim.
        self.submitted_orders: Set[int] = (
            submitted_orders if submitted_orders is not None else set()
        )
        # Orders for which we saw OrderExpired and have not yet claimed.
        self.claimable: Set[int] = claimable if claimable is not None else set()

    # ------------------------------------------------------------------
    # Solution submission
    # ------------------------------------------------------------------

    async def submit_solution(
        self, order_id: int, result: "MiningResult"
    ) -> SubmitReport:
        """Submit one mining result for *order_id*.

        Never raises for chain-side failures: transient RPC/build errors
        return FAILED, receipt errors return STALE or MEMPOOL_DISABLE.
        Malformed solutions (non-±1/0 spins) still raise ``ValueError`` —
        that is a miner bug, not a chain condition.
        """
        solutions = solutions_to_scale(result.solutions)
        if len(solutions) > MAX_SOLUTIONS:
            solutions = solutions[:MAX_SOLUTIONS]

        # `solutions: BoundedVec<BoundedVec<i8, MaxNodes>, MaxSolutions>`
        # — both layers are 1-field composites in substrate metadata,
        # so each inner solution AND the outer list need 1-tuple wrapping.
        # Matches `substrate.submitter.encode_quantum_proof` shape.
        solutions_wrapped = ([(sol,) for sol in solutions],)

        try:
            receipt = await self._submit_with_txpool_retry(
                order_id, solutions_wrapped,
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.solution_errors += 1
            logger.exception(
                "submit_solution RPC failed for order=%d: %s", order_id, exc
            )
            return SubmitReport(SubmitOutcome.FAILED, str(exc))

        outcome = classify_submit_receipt(receipt.error)
        if outcome is SubmitOutcome.OK:
            self.stats.solutions_submitted += 1
            self.submitted_orders.add(order_id)
            logger.info(
                "submit_solution accepted: order=%d extrinsic=%s",
                order_id,
                receipt.extrinsic_hash,
            )
            if self.on_solution_submitted is not None:
                try:
                    await self.on_solution_submitted(order_id, result)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "on_solution_submitted callback raised for order=%d: %s",
                        order_id, exc,
                    )
        elif outcome is SubmitOutcome.STALE:
            self.stats.solution_stale_drops += 1
            logger.info(
                "submit_solution dropped as stale: order=%d error=%s",
                order_id,
                receipt.error,
            )
        else:  # MEMPOOL_DISABLE
            self.stats.solution_errors += 1
            logger.error(
                "submit_solution failed with a mempool-fatal error: "
                "order=%d error=%s — signaling MEMPOOL_DISABLE",
                order_id,
                receipt.error,
            )
        return SubmitReport(outcome, receipt.error)

    async def _submit_with_txpool_retry(
        self, order_id: int, solutions_wrapped
    ):
        """Compose + submit, retrying txpool rejections with a fresh nonce.

        The signer account is shared with the pow controller: its proof
        submissions and participation remarks race this extrinsic for the
        next nonce, and the loser is rejected at the txpool (1010
        "Transaction is outdated" / 1014 "Priority is too low"). Each
        retry re-composes via ``build_signed_extrinsic``, which reads the
        account nonce anew — the same policy as the pow path's
        ``submit_with_retry``. Bounded by ``TXPOOL_RETRIES``; the final
        rejection propagates to the caller's FAILED path.
        """
        attempt = 0
        while True:
            extrinsic_hex = await self.build_client.build_signed_extrinsic(
                "QuantumComputeMempool",
                "submit_solution",
                {
                    "order_id": order_id,
                    "solutions": solutions_wrapped,
                },
                self.signer,
                tip=self.tip_plancks,
            )
            try:
                # Watch-timeout cap: a dead watch subscription must retry,
                # not freeze the consume loop (same hang class observed on
                # the pow submit path).
                return await asyncio.wait_for(
                    self.pool_client.submit_signed_extrinsic(
                        extrinsic_hex, wait_for="inblock",
                    ),
                    timeout=_SUBMIT_WATCH_TIMEOUT_S,
                )
            except (
                SubstrateRequestException,
                ExtrinsicRejected,
                asyncio.TimeoutError,
            ) as exc:
                attempt += 1
                if attempt > self.TXPOOL_RETRIES:
                    raise
                logger.warning(
                    "submit_solution txpool rejection for order=%d "
                    "(attempt %d/%d, retrying with a fresh nonce): %s",
                    order_id, attempt, self.TXPOOL_RETRIES, exc,
                )
                await asyncio.sleep(self.txpool_retry_backoff_s * attempt)

    # ------------------------------------------------------------------
    # Reward claiming
    # ------------------------------------------------------------------

    def note_order_expired(self, order_id: int) -> None:
        """Producer's OrderExpired hook: queue a claim if we submitted."""
        if order_id in self.submitted_orders:
            self.claimable.add(order_id)

    async def run_claim_loop(self, shutdown_event: asyncio.Event) -> None:
        """Periodically claim expired-but-unclaimed orders until shutdown."""
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=self.claim_poll_interval,
                )
                return  # shutdown set
            except asyncio.TimeoutError:
                pass
            await self.claim_expired_orders()

    async def claim_expired_orders(self) -> None:
        """Attempt `claim_reward` for every claimable order once.

        Also scans ``submitted_orders`` for orders past their COMPUTED
        expiry: the pallet expires orders lazily (``expire_order_if_needed``
        runs only inside submit_solution / claim_reward / reclaim_order —
        no on_initialize sweep), so on a quiet mempool the ``OrderExpired``
        event never fires on its own and an event-only claim loop would
        wait forever. ``claim_reward`` performs the lazy expiry inline and
        pays, so claiming a logically-expired Opened order is safe.
        """
        await self._scan_submitted_for_lazy_expiry()
        if not self.claimable:
            return
        to_remove: List[int] = []
        for order_id in list(self.claimable):
            try:
                extrinsic_hex = await self.build_client.build_signed_extrinsic(
                    "QuantumComputeMempool",
                    "claim_reward",
                    {"order_id": order_id},
                    self.signer,
                    tip=self.tip_plancks,
                )
                receipt = await self.pool_client.submit_signed_extrinsic(
                    extrinsic_hex, wait_for="inblock",
                )
            except Exception as exc:  # noqa: BLE001 — keep trying other orders
                logger.exception(
                    "claim_reward RPC failed for order=%d: %s", order_id, exc
                )
                continue
            outcome = classify_claim_receipt(receipt.error)
            if outcome is ClaimOutcome.OK:
                self.stats.rewards_claimed += 1
                to_remove.append(order_id)
                logger.info(
                    "claim_reward accepted: order=%d extrinsic=%s",
                    order_id,
                    receipt.extrinsic_hash,
                )
                if self.on_reward_claimed is not None:
                    await self.on_reward_claimed(order_id, 0)
            elif outcome is ClaimOutcome.RETRY:
                pass  # OrderNotExpired — retry on the next tick
            elif outcome is ClaimOutcome.GIVE_UP:
                to_remove.append(order_id)
            else:  # FAILED
                self.stats.claim_errors += 1
                logger.error(
                    "claim_reward failed fatally for order=%d: error=%s — giving up",
                    order_id,
                    receipt.error,
                )
                to_remove.append(order_id)
        for order_id in to_remove:
            self.claimable.discard(order_id)
            self.submitted_orders.discard(order_id)

    async def _scan_submitted_for_lazy_expiry(self) -> None:
        """Move lazily-expired won orders into ``claimable``.

        Best-effort: a query/RPC failure keeps the order in
        ``submitted_orders`` for the next tick.
        """
        outstanding = self.submitted_orders - self.claimable
        if not outstanding:
            return
        try:
            now = await self.pool_client.get_block_number()
        except Exception as exc:  # noqa: BLE001 — retry next tick
            logger.warning("lazy-expiry scan: get_block_number failed: %s", exc)
            return
        for order_id in list(outstanding):
            try:
                order = await self.pool_client.query_job_order(order_id)
            except Exception as exc:  # noqa: BLE001 — retry next tick
                logger.warning(
                    "lazy-expiry scan: query_job_order(%d) failed: %s",
                    order_id, exc,
                )
                continue
            if order is None:
                # Purged — nothing left to claim.
                self.submitted_orders.discard(order_id)
                continue
            if order.status == OrderStatus.EXPIRED or (
                order.status == OrderStatus.OPENED
                and deadline_blocks_remaining(order, now) <= 0
            ):
                logger.info(
                    "lazy-expiry scan: order=%d past expiry (status=%s, "
                    "block=%d); queueing claim",
                    order_id, order.status, now,
                )
                self.claimable.add(order_id)


__all__ = [
    "CLAIM_STALE_ERRORS",
    "ClaimOutcome",
    "MAX_SOLUTIONS",
    "MempoolSubmitter",
    "MempoolSubmitterStats",
    "SOLUTION_FATAL_ERRORS",
    "SOLUTION_STALE_ERRORS",
    "SubmitOutcome",
    "SubmitReport",
    "classify_claim_receipt",
    "classify_submit_receipt",
]
