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
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.signer import Signer
from shared.substrate_client import SubstrateClient
from shared.substrate_submitter import submit_proof
from shared.substrate_types import (
    ExtrinsicReceipt,
    SubstrateMiningContext,
)


logger = get_logger("substrate_miner_controller")


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


@dataclass
class ControllerStats:
    """Lightweight counters surfaced to telemetry."""

    heads_observed: int = 0
    contexts_dispatched: int = 0
    results_received: int = 0
    proofs_submitted: int = 0
    stale_drops: int = 0
    submission_errors: int = 0


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
    def __init__(
        self,
        client: SubstrateClient,
        signer: Signer,
        miner_handles: List[MinerHandle],
        topology_hash: Optional[bytes] = None,
        on_proof_submitted: Optional[
            Callable[[ExtrinsicReceipt, SubstrateMiningContext], Awaitable[None]]
        ] = None,
        subscription_client: Optional[SubstrateClient] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "SubstrateMinerController requires at least one MinerHandle"
            )
        self.client = client
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
        self._drainer_tasks: List[asyncio.Task] = []
        self._subscription_task: Optional[asyncio.Task] = None

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
        # for the next slot.
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
            # Drain cancellations so they don't surface as warnings.
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

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
        # Cancel any in-flight mining on the prior snapshot. Without
        # synchronization, cancel() → clear() → dispatch can wipe a
        # cancel before the worker observes it, leaving the worker
        # mining the OLD context against a stop_event tied to the NEW
        # dispatch. The worker now emits a `work_item_done` sentinel
        # when its loop exits with no result; the drainer surfaces that
        # via _await_handle_done so we can confirm cancellation landed
        # before we re-clear the event.
        for handle in self.miner_handles:
            if handle.miner_id in self._dispatched:
                handle.cancel()
        # Best-effort drain: wait briefly for each handle to acknowledge
        # the cancel. Workers without an active mining attempt (initial
        # head, or already idle) won't emit a sentinel — the short
        # timeout keeps us responsive in that case.
        for handle in self.miner_handles:
            if handle.miner_id in self._dispatched:
                await self._await_handle_done(handle, timeout=0.5)

        context = await self.client.get_mining_snapshot(
            at=head_hash,
            topology_hash=self.topology_hash,
            miner_account_bytes=self.signer.account_id_bytes(),
        )
        if context is None:
            logger.warning(
                "mining_snapshot returned None at block %d (head=0x%s); "
                "chain may not be seeded yet",
                block_number,
                head_hash.hex()[:16],
            )
            return

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

    async def _handle_result(self, envelope: _ResultEnvelope) -> None:
        self.stats.results_received += 1
        # Drop results produced against a stale context. The controller
        # already moved on to a new head; the chain would reject this proof.
        if (
            self._current_context is None
            or envelope.context.block_number != self._current_context.block_number
            or envelope.context.parent_hash != self._current_context.parent_hash
        ):
            self.stats.stale_drops += 1
            logger.info(
                "dropping stale result from %s: block=%d (current=%s)",
                envelope.handle_id,
                envelope.context.block_number,
                getattr(self._current_context, "block_number", "<none>"),
            )
            return

        try:
            receipt = await submit_proof(
                self.client, self.signer, envelope.result, envelope.context
            )
        except Exception as exc:  # noqa: BLE001 — surface RPC errors to logs
            self.stats.submission_errors += 1
            logger.exception(
                "submit_proof RPC failed for result from %s: %s",
                envelope.handle_id,
                exc,
            )
            return

        outcome = classify_submission(receipt)
        if outcome == "ok":
            self.stats.proofs_submitted += 1
            logger.info(
                "submit_proof accepted: extrinsic=%s block=%s miner=%s",
                receipt.extrinsic_hash,
                receipt.block_hash,
                self.signer.ss58_address(),
            )
            if self.on_proof_submitted is not None:
                await self.on_proof_submitted(receipt, envelope.context)
        elif outcome == "stale":
            self.stats.stale_drops += 1
            logger.info(
                "submit_proof dropped as stale: error=%s extrinsic=%s",
                receipt.error,
                receipt.extrinsic_hash,
            )
        else:  # fatal
            self.stats.submission_errors += 1
            raise RuntimeError(
                f"submit_proof failed fatally: error={receipt.error} "
                f"extrinsic={receipt.extrinsic_hash}"
            )

    async def _await_handle_done(
        self, handle: MinerHandle, *, timeout: float
    ) -> None:
        """Wait briefly for the worker to confirm cancellation.

        Pops one `work_item_done` sentinel from the handle's done queue,
        with a timeout. If the timeout fires we proceed anyway — workers
        without an active mining loop won't emit a sentinel, and we
        don't want to block forever on the initial dispatch.
        """
        queue = self._done_queues.get(handle.miner_id)
        if queue is None:
            return
        try:
            await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

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
            except Exception:
                # Queue.get with timeout raises queue.Empty; loop and retry.
                continue
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
                try:
                    self._done_queues[handle.miner_id].put_nowait(None)
                except asyncio.QueueFull:
                    pass
            elif isinstance(msg, dict) and msg.get("op") == "error":
                logger.error(
                    "miner %s reported error: %s",
                    handle.miner_id,
                    msg.get("message"),
                )
            elif isinstance(msg, dict) and msg.get("op") == "stats":
                # Stats responses are pulled directly by callers of
                # handle.get_stats(); if one lands here it just means
                # nobody was listening — drop and continue.
                pass

    async def _subscribe_heads(self) -> None:
        """Subscribe to new best heads and post into the head queue.

        Wraps `client.subscribe_new_heads` so any exception in the
        subscription thread is logged and triggers controller shutdown
        rather than silently disappearing.
        """
        loop = asyncio.get_running_loop()

        async def callback(block_hash: bytes, block_number: int) -> None:
            # Latest-only semantics: overwrite whatever's pending. If the
            # main loop is busy in submit_proof while heads pile up, we
            # don't want to dispatch against an old one when it returns.
            self._latest_head = (block_hash, block_number)
            self._head_signal.set()

        try:
            await self._subscription_client.subscribe_new_heads(callback)
        except Exception:
            logger.exception("head subscription crashed; triggering shutdown")
            loop.call_soon_threadsafe(self.shutdown)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _teardown(self) -> None:
        logger.info("controller shutting down: cancelling handles, draining tasks")
        for handle in self.miner_handles:
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001
                pass
        for task in self._drainer_tasks:
            task.cancel()
        if self._subscription_task is not None:
            self._subscription_task.cancel()
        # Await cancellations.
        for task in self._drainer_tasks + (
            [self._subscription_task] if self._subscription_task else []
        ):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._drainer_tasks.clear()
        self._subscription_task = None
        if self._owns_subscription_client:
            try:
                await self._subscription_client.close()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def classify_submission(receipt: ExtrinsicReceipt) -> str:
    """Bucket a receipt as `ok` / `stale` / `fatal`.

    Stale errors are expected race conditions (head change between mining
    and submission); fatal errors mean the controller can't make progress.
    The error-message match is a substring check because substrate-interface
    formats receipts as `Module(error=Foo, index=N, ...)`.
    """
    if receipt.is_success:
        return "ok"
    error = receipt.error or ""
    for name in STALE_SUBMISSION_ERRORS:
        if name in error:
            return "stale"
    for name in FATAL_SUBMISSION_ERRORS:
        if name in error:
            return "fatal"
    # Unknown errors default to fatal — better to crash loudly than to silently
    # mine forever against a chain state we don't understand.
    return "fatal"


__all__ = [
    "ControllerStats",
    "STALE_SUBMISSION_ERRORS",
    "FATAL_SUBMISSION_ERRORS",
    "SubstrateMinerController",
    "classify_submission",
]
