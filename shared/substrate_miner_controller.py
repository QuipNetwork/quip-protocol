"""Substrate miner controller — Phase 4 main loop.

Subscribes to new best heads on the substrate node, fetches a fresh
`MiningSnapshot` at each head, dispatches a `SubstrateMiningContext` to every
attached `MinerHandle`, and submits the first valid `MiningResult` back to
the chain as a `QuantumPow.submit_proof` extrinsic.

Lifecycle:

    controller = SubstrateMinerController(client, signer, miner_handles)
    await controller.run()      # blocks until shutdown() or fatal error
    # ... external signal ...
    controller.shutdown()       # idempotent

The controller does not own the lifecycle of its inputs. Construction of the
`SubstrateClient`, `Signer`, and `MinerHandle[]` is the caller's job, as is
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
import queue as _queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.signer import Signer
from shared.substrate_client import SubstrateClient
from shared.substrate_submitter import encode_quantum_proof, submit_proof
from shared.substrate_types import (
    ExtrinsicReceipt,
    SubstrateMiningContext,
)


logger = get_logger("substrate_miner_controller")


# Threshold for consecutive `None` snapshots before the controller raises.
# A few in a row at startup is expected (chain may not be seeded yet); ten
# in a row strongly indicates RPC corruption or a genuinely stuck chain.
_NONE_SNAPSHOT_FAIL_THRESHOLD = 10


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


@dataclass
class _ResultEnvelope:
    """Tag a `MiningResult` with the context it was produced against.

    The worker only returns the bare result; the controller knows which
    context was last dispatched and pairs them here so the submitter has
    everything it needs (and so a result against a stale context can be
    discarded if a head change happened mid-mine).
    """

    result: MiningResult
    context: SubstrateMiningContext
    handle_id: str


class SubstrateMinerController:
    """Coordinate mining across one or more `MinerHandle`s against a substrate chain.

    Subscribes to new best heads (via a dedicated `subscription_client`),
    dispatches a fresh `SubstrateMiningContext` to every attached handle on
    each head change, drains `MiningResult`s back, and submits the first
    valid one as a `QuantumPow.submit_proof` extrinsic. See the module
    docstring for the full lifecycle.

    Args:
        client: SubstrateClient used for state queries and extrinsic
            submission. Must NOT also be passed as `subscription_client` —
            substrate-interface holds the websocket in receive mode during
            `subscribe_block_headers`, deadlocking any concurrent call on
            the same connection.
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
        subscription_client: Optional dedicated SubstrateClient for the
            head subscription. If `None`, the controller creates and
            manages its own — see the deadlock note above.
    """

    def __init__(
        self,
        client: SubstrateClient,
        signer: Signer,
        miner_handles: list[MinerHandle],
        *,
        topology_hash: Optional[bytes] = None,
        on_proof_submitted: Optional[
            Callable[[ExtrinsicReceipt, SubstrateMiningContext], Awaitable[None]]
        ] = None,
        subscription_client: Optional[SubstrateClient] = None,
        core: Optional[Any] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "SubstrateMinerController requires at least one MinerHandle"
            )
        self.client = client
        # Optional MinerCore hook. When provided, the controller calls
        # `core.record_dispatch()` once per head (not per handle — that would
        # double-count when more than one miner is attached) and
        # `core.record_result(winning_miner_id, mining_time)` on chain-accepted
        # proofs. Keeps `/api/v1/stats`'s legacy `total_blocks_attempted` /
        # `total_blocks_won` / `wins_per_miner` fields live without coupling
        # the controller's type to MinerCore.
        self.core = core
        # substrate-interface holds the websocket in receive mode for the
        # duration of `subscribe_block_headers`, which makes any concurrent
        # submit_extrinsic / state_call on the same connection hang
        # indefinitely. We use a dedicated subscription client by default —
        # callers can inject their own (e.g. tests) but it must be a
        # *separate* SubstrateClient instance from `client`.
        if subscription_client is None:
            subscription_client = SubstrateClient(url=client.url)
        elif subscription_client is client:
            raise ValueError(
                "subscription_client must be a separate SubstrateClient "
                "instance — substrate-interface websockets are not "
                "concurrent-safe across subscribe + submit"
            )
        self._subscription_client = subscription_client
        self._owns_subscription_client = subscription_client is not client
        self.signer = signer
        self.miner_handles = miner_handles
        self.topology_hash = topology_hash
        self.on_proof_submitted = on_proof_submitted
        self.stats = ControllerStats()

        self._current_context: Optional[SubstrateMiningContext] = None
        # Per-handle dispatch tracking. When a result arrives the drainer
        # pairs it with the context the *handle* was given, not the
        # controller's latest context. Without this, late results from a
        # prior head would be silently misclassified as fresh.
        self._dispatched: dict[str, SubstrateMiningContext] = {}
        # Latest-only head channel. A slow submit_proof can hold the main
        # loop in `_handle_result` for seconds; during that window the
        # chain advances by multiple blocks. We don't want to dispatch
        # against the oldest pending head when we come back — only the
        # newest matters. maxsize=1 + replace-on-write does that.
        self._latest_head: Optional[tuple[bytes, int]] = None
        self._head_signal = asyncio.Event()
        self._result_queue: asyncio.Queue[_ResultEnvelope] = asyncio.Queue()
        # Per-handle done-sentinel queue. The worker emits
        # `{"op": "work_item_done"}` when its mining loop exits with no
        # result; the drainer pushes that into the handle's queue and
        # _await_handle_done pops from it. Using a queue (not an event)
        # so multiple done sentinels don't get coalesced into a single
        # set/clear cycle.
        self._done_queues: dict[str, asyncio.Queue[None]] = {
            h.miner_id: asyncio.Queue() for h in miner_handles
        }
        self._shutdown_event = asyncio.Event()
        self._drainer_tasks: list[asyncio.Task] = []
        self._subscription_task: Optional[asyncio.Task] = None
        # Running count of consecutive None snapshots; reset on a successful
        # snapshot. Escalates to RuntimeError after _NONE_SNAPSHOT_FAIL_THRESHOLD.
        self._consecutive_none_snapshots = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal a graceful shutdown. Safe to call from any thread/task."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main loop. Returns on shutdown or fatal error."""
        account = self.signer.account_id_bytes()
        await self._verify_registered(account)

        # Drainer tasks consume each handle's resp queue and post into our
        # asyncio result queue. Start them before the first dispatch so we
        # never miss an early result.
        for handle in self.miner_handles:
            task = asyncio.create_task(
                self._drain_handle(handle),
                name=f"drain-{handle.miner_id}",
            )
            self._drainer_tasks.append(task)

        # Connect the dedicated subscription client (if we own it) and
        # start the subscription task. The blocking subscribe loop inside
        # substrate-interface runs on the subscription client's executor
        # — keeping it off the main client means submit_extrinsic /
        # state_call traffic doesn't deadlock against an active
        # subscription.
        if self._owns_subscription_client:
            await self._subscription_client.connect()
        self._subscription_task = asyncio.create_task(
            self._subscribe_heads(),
            name="head-subscription",
        )

        # Prime the loop with the current head so we don't sit idle waiting
        # for the next slot. block_number is informational only here
        # (the head handler re-fetches the snapshot at this hash and uses
        # the snapshot's authoritative block_number).
        head = await self.client.get_head()
        block_number = await self.client.get_block_number(at=head)
        self._latest_head = (head, block_number)
        self._head_signal.set()

        try:
            await self._main_loop()
        finally:
            await self._teardown()

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    async def _verify_registered(self, account: bytes) -> None:
        miner_info = await self.client.query_miner(account)
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
            head_task = asyncio.create_task(self._head_signal.wait())
            result_task = asyncio.create_task(self._result_queue.get())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            done, pending = await asyncio.wait(
                [head_task, result_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            # Drain cancellations so they don't surface as warnings. Narrow
            # catch: CancelledError is expected, anything else is a bug
            # (these tasks await an Event/Queue.get/Event.wait — no other
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

            if head_task in done:
                # Snapshot the latest head and clear the signal atomically
                # so a head arriving mid-handle gets coalesced into the
                # next iteration rather than dropped.
                self._head_signal.clear()
                latest = self._latest_head
                if latest is not None:
                    head_hash, block_number = latest
                    await self._handle_head(head_hash, block_number)
                continue

            if result_task in done:
                envelope = result_task.result()
                await self._handle_result(envelope)
                continue

    async def _handle_head(self, head_hash: bytes, block_number: int) -> None:
        self.stats.heads_observed += 1
        # Cancel any in-flight mining on the prior snapshot, then wait in
        # parallel for each handle to ack the cancel. Without this
        # synchronization, cancel() → clear() → dispatch can wipe a cancel
        # before the worker observes it, leaving the worker mining the OLD
        # context against a stop_event tied to the NEW dispatch. The worker
        # emits a `work_item_done` sentinel when its loop exits with no
        # result; the drainer surfaces that via _await_handle_done so we
        # confirm cancellation landed before we re-clear the event.
        #
        # Parallel wait (asyncio.gather) keeps total stall to ~0.5s
        # regardless of handle count — sequential waits would block for
        # N×0.5s, significant against a 6s block time.
        to_drain = [h for h in self.miner_handles if h.miner_id in self._dispatched]
        for handle in to_drain:
            handle.cancel()
        if to_drain:
            await asyncio.gather(
                *[self._await_handle_done(h, timeout=0.5) for h in to_drain]
            )

        context = await self.client.get_mining_snapshot(
            at=head_hash,
            topology_hash=self.topology_hash,
            miner_account_bytes=self.signer.account_id_bytes(),
        )
        if context is None:
            self.stats.none_snapshots_seen += 1
            self._consecutive_none_snapshots += 1
            logger.warning(
                "mining_snapshot returned None at block %d (head=0x%s); "
                "chain may not be seeded yet (consecutive_none=%d)",
                block_number,
                head_hash.hex()[:16],
                self._consecutive_none_snapshots,
            )
            if self._consecutive_none_snapshots >= _NONE_SNAPSHOT_FAIL_THRESHOLD:
                raise RuntimeError(
                    f"mining_snapshot returned None for "
                    f"{self._consecutive_none_snapshots} consecutive heads "
                    "— chain may be stuck or RPC is broken. Run "
                    "`quip-miner bootstrap` or check the node."
                )
            return
        self._consecutive_none_snapshots = 0

        if (
            self.topology_hash is not None
            and context.topology_hash != self.topology_hash
        ):
            raise RuntimeError(
                "configured --topology-hash does not match snapshot: "
                f"expected 0x{self.topology_hash.hex()}, got 0x{context.topology_hash.hex()}"
            )

        self._current_context = context
        logger.info(
            "new head: block=%d hash=0x%s... topology=0x%s... "
            "nodes=%d edges=%d",
            context.block_number,
            head_hash.hex()[:16],
            context.topology_hash.hex()[:16],
            len(context.nodes),
            len(context.edges),
        )

        # Dispatch the new context to every handle. Each one will generate
        # its own salts and explore the search space independently. Track
        # what was dispatched per handle so late results pair with the
        # right context (not whatever's current at receive-time).
        for handle in self.miner_handles:
            self._dispatched[handle.miner_id] = context
            handle.mine_work_item(context)
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            # One attempt per head, regardless of how many handles fanned out.
            self.core.record_dispatch()

    async def _handle_result(self, envelope: _ResultEnvelope) -> None:
        self.stats.results_received += 1
        # Drop results produced against a stale context. The controller
        # already moved on to a new head; the chain would reject this proof.
        # topology_hash is included so a governance call that rotates the
        # topology within a single block doesn't slip through as a
        # block/parent-match.
        if (
            self._current_context is None
            or envelope.context.block_number != self._current_context.block_number
            or envelope.context.parent_hash != self._current_context.parent_hash
            or envelope.context.topology_hash != self._current_context.topology_hash
        ):
            self.stats.stale_drops += 1
            logger.info(
                "dropping stale result from %s: block=%d (current=%s)",
                envelope.handle_id,
                envelope.context.block_number,
                getattr(self._current_context, "block_number", "<none>"),
            )
            return

        # Encoder errors (ValueError on no-solutions / wrong-salt-length)
        # are code defects — retrying won't help. Keep them out of the
        # RPC-error catch below so they raise loudly instead of being
        # logged-and-dropped like a transient network blip.
        try:
            encode_quantum_proof(envelope.result, envelope.context)
        except ValueError as exc:
            raise RuntimeError(
                f"proof encoding failed (bug): {exc}"
            ) from exc

        try:
            receipt = await submit_proof(
                self.client, self.signer, envelope.result, envelope.context
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.submission_errors += 1
            self.stats.last_submission_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "submit_proof RPC failed for result from %s: %s",
                envelope.handle_id,
                exc,
            )
            return

        outcome = classify_submission(receipt)
        if outcome is SubmissionOutcome.OK:
            self.stats.proofs_submitted += 1
            logger.info(
                "submit_proof accepted: extrinsic=%s block=%s miner=%s",
                receipt.extrinsic_hash,
                receipt.block_hash,
                self.signer.ss58_address(),
            )
            if self.core is not None:
                self.core.record_result(
                    winning_miner_id=envelope.result.miner_id,
                    mining_time=float(envelope.result.mining_time),
                )
            if self.on_proof_submitted is not None:
                await self.on_proof_submitted(receipt, envelope.context)
        elif outcome is SubmissionOutcome.STALE:
            self.stats.stale_drops += 1
            logger.info(
                "submit_proof dropped as stale: error=%s extrinsic=%s",
                receipt.error,
                receipt.extrinsic_hash,
            )
        else:  # FATAL
            self.stats.submission_errors += 1
            self.stats.last_submission_error = str(receipt.error or "")
            raise RuntimeError(
                f"submit_proof failed fatally: error={receipt.error} "
                f"extrinsic={receipt.extrinsic_hash}"
            )

    async def _await_handle_done(
        self, handle: MinerHandle, *, timeout: float
    ) -> None:
        """Wait briefly for the worker to confirm cancellation.

        Pops one `work_item_done` sentinel from the handle's done queue
        with a timeout. If the timeout fires we proceed anyway — but log
        a WARNING because the worker did NOT ack the cancel in time, and
        the next dispatch's `stop_event.clear()` may race the prior
        cancel (the very condition the sentinel was added to prevent).
        """
        sentinel_queue = self._done_queues.get(handle.miner_id)
        if sentinel_queue is None:
            return
        try:
            await asyncio.wait_for(sentinel_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "handle %s did not ack cancel within %.1fs; dispatching "
                "anyway — next mine_work_item may race the prior cancel",
                handle.miner_id,
                timeout,
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
        while not self._shutdown_event.is_set():
            try:
                msg = await loop.run_in_executor(
                    None, handle.resp.get, True, 0.5
                )
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
            if isinstance(msg, MiningResult):
                dispatched = self._dispatched.get(handle.miner_id)
                if dispatched is None:
                    logger.warning(
                        "result from %s ignored: no dispatched context recorded",
                        handle.miner_id,
                    )
                    continue
                await self._result_queue.put(
                    _ResultEnvelope(
                        result=msg,
                        context=dispatched,
                        handle_id=handle.miner_id,
                    )
                )
            elif isinstance(msg, dict) and msg.get("op") == "work_item_done":
                # Worker finished its mine_work_item loop with no result —
                # almost always because cancel() was observed. Surface it
                # so _await_handle_done can synchronize on cancellation
                # before the next dispatch clears stop_event.
                self._done_queues[handle.miner_id].put_nowait(None)
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
            elif isinstance(msg, dict) and msg.get("op") == "stats":
                # Stats responses are pulled directly by callers of
                # handle.get_stats(); if one lands here it just means
                # nobody was listening — drop and continue. NOTE: while
                # the controller owns the handle, callers MUST NOT call
                # handle.get_stats() directly — the drainer dequeues the
                # response first and blocks the caller forever.
                pass

    async def _subscribe_heads(self) -> None:
        """Subscribe to new best heads and post into the head queue.

        Wraps `client.subscribe_new_heads` so any exception in the
        subscription thread is logged and triggers controller shutdown
        rather than silently disappearing.
        """

        async def callback(block_hash: bytes, block_number: int) -> None:
            # Latest-only semantics: overwrite whatever's pending. If the
            # main loop is busy in submit_proof while heads pile up, we
            # don't want to dispatch against an old one when it returns.
            self._latest_head = (block_hash, block_number)
            self._head_signal.set()

        try:
            await self._subscription_client.subscribe_new_heads(callback)
        except Exception as exc:
            self.stats.last_submission_error = (
                f"head subscription crashed: {type(exc).__name__}: {exc}"
            )
            logger.exception("head subscription crashed; triggering shutdown")
            # We're already on the asyncio loop; no need for
            # call_soon_threadsafe (the subscription wrapper bridges its
            # background thread into our awaitable callback).
            self.shutdown()

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
        if self._subscription_task is not None:
            self._subscription_task.cancel()
        # Await cancellations. Narrow catch: CancelledError is expected,
        # other exceptions are real cleanup failures worth surfacing.
        for task in self._drainer_tasks + (
            [self._subscription_task] if self._subscription_task else []
        ):
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
        self._subscription_task = None
        if self._owns_subscription_client:
            try:
                await self._subscription_client.close()
            except Exception as exc:  # noqa: BLE001 — log cleanup failures
                logger.warning(
                    "teardown: subscription_client close raised %s: %s",
                    type(exc).__name__,
                    exc,
                )


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
    logger.error(
        "unrecognized submission error; classifying as fatal: %r", error
    )
    return SubmissionOutcome.FATAL


__all__ = [
    "ControllerStats",
    "STALE_SUBMISSION_ERRORS",
    "FATAL_SUBMISSION_ERRORS",
    "SubmissionOutcome",
    "SubstrateMinerController",
    "classify_submission",
]
