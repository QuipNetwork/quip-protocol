"""Substrate miner controller — Phase 4 main loop.

Subscribes to new best heads on the substrate node, fetches a fresh
`MiningSnapshot` at each head, dispatches a `SubstrateMiningContext` to every
attached `MinerHandle`, and submits the first valid `MiningResult` back to
the chain as a `QuantumPow.submit_proof` extrinsic.

Lifecycle:

    pool = ValidatorPool(urls=["ws://primary:9944", "ws://standby:9944"])
    controller = SubstrateMinerController(pool, signer, miner_handles)
    try:
        await controller.run()  # blocks until shutdown() or fatal error
    finally:
        controller.shutdown()
        await pool.close()

The controller does not own the lifecycle of its inputs. Construction of the
`ValidatorPool`, `Signer`, and `MinerHandle[]` is the caller's job, as is
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
import hashlib
import queue as _queue
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, Tuple

from websocket import WebSocketException

from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.miner_worker import MinerHandle
from shared.signer import Signer
from shared.substrate_client import NoValidatorReachable, SubstrateClient
from shared.substrate_submitter import encode_quantum_proof, submit_proof
from shared.validator_pool import ValidatorPool
from shared.substrate_types import (
    ExtrinsicReceipt,
    SubstrateMiningContext,
)


logger = get_logger("substrate_miner_controller")


class _MinerCoreStats(Protocol):
    """Subset of `shared.miner_core.MinerCore` the controller depends on.

    Structural type rather than direct `MinerCore` import keeps the
    controller usable without a full MinerCore (e.g., a thin test
    double), while still catching signature drift at type-check time.
    """

    def record_dispatch(self) -> None: ...

    def record_result(self, *, winning_miner_id: str, mining_time: float) -> None: ...


# Threshold for consecutive `None` snapshots before the controller raises.
# A few in a row at startup is expected (chain may not be seeded yet); ten
# in a row strongly indicates RPC corruption or a genuinely stuck chain.
_NONE_SNAPSHOT_FAIL_THRESHOLD = 10


# Cap on how many recently-won work keys we remember. Sized for the
# expected window of late results: a miner with N handles can in the
# worst case have N-1 pending results land after an OK on the same key,
# and the head changes every ~6s so older keys are no longer reachable
# anyway. 16 is generous for the practical N≤8 handle counts.
_CLOSED_WORK_KEYS_CAP = 16


# Per-handle ring buffer of recent dispatch contexts. Indexed by
# dispatch_id; pruned when older than the most-recent N attempts. Sized
# to absorb late results from a few cancelled dispatches without ever
# losing the immutable mapping for an in-flight one.
_DISPATCH_CONTEXT_RETENTION = 4


# A "work key" uniquely identifies the puzzle a context is mining
# against: same (block_number, parent_hash, topology_hash) means the
# pallet will derive the same nonce and therefore the same Ising
# problem. Used both for stale-result detection AND for marking a
# work item closed/won after an accepted submission.
WorkKey = Tuple[int, bytes, bytes]


def _work_key(ctx: "SubstrateMiningContext") -> WorkKey:
    return (ctx.block_number, ctx.parent_hash, ctx.topology_hash)


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
    # Results dropped because the work key was already won by a prior
    # sibling submission this head. Distinct from stale_drops (head
    # changed) — this counts the storm-prevention path.
    duplicate_result_drops: int = 0


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
        pool: ValidatorPool providing slot clients indexed by role. The
            controller asks the pool for `rpc` (for state queries and
            extrinsic submission) and `subscribe.pow` (for the head
            subscription). The pool guarantees distinct clients per
            role, which sidesteps the substrate-interface deadlock
            where a single websocket can't both subscribe and submit.
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
    """

    def __init__(
        self,
        pool: ValidatorPool,
        signer: Signer,
        miner_handles: list[MinerHandle],
        *,
        topology_hash: Optional[bytes] = None,
        on_proof_submitted: Optional[
            Callable[[ExtrinsicReceipt, SubstrateMiningContext], Awaitable[None]]
        ] = None,
        core: Optional[_MinerCoreStats] = None,
    ) -> None:
        if not miner_handles:
            raise ValueError(
                "SubstrateMinerController requires at least one MinerHandle"
            )
        self._pool = pool
        # `client` and `_subscription_client` are populated lazily in
        # `run()` — `__init__` must not touch the network. Existing
        # method bodies still reference `self.client` directly; the
        # type annotation is Optional only during the (synchronous)
        # constructor window.
        self.client: Optional[SubstrateClient] = None
        self._subscription_client: Optional[SubstrateClient] = None
        # Optional MinerCore hook. When provided, the controller calls
        # `core.record_dispatch()` once per head (not per handle — that would
        # double-count when more than one miner is attached) and
        # `core.record_result(winning_miner_id, mining_time)` on chain-accepted
        # proofs. Keeps `/api/v1/stats`'s legacy `total_blocks_attempted` /
        # `total_blocks_won` / `wins_per_miner` fields live without coupling
        # the controller's type to MinerCore.
        self.core = core
        # Client slots are pulled from the pool in `run()` — see comment
        # there. We deliberately do NOT touch the network in __init__.
        # The pool guarantees the rpc and subscribe slots are distinct
        # SubstrateClient instances, which sidesteps the substrate-
        # interface deadlock where a single websocket can't both
        # subscribe and submit (receive-mode hold).
        self.signer = signer
        self.miner_handles = miner_handles
        self.topology_hash = topology_hash
        self.on_proof_submitted = on_proof_submitted
        self.stats = ControllerStats()

        self._current_context: Optional[SubstrateMiningContext] = None
        # Per-(handle_id, dispatch_id) immutable context map. Each
        # mine_work_item dispatch installs a fresh entry; the drainer
        # pairs results with the exact context they were dispatched
        # against by looking up (handle_id, dispatch_id) from the
        # response. Without this immutability, the previous
        # `_dispatched[handle_id] = ctx` pattern could surface a result
        # from dispatch N paired with dispatch N+1's context.
        self._dispatch_contexts: dict[
            Tuple[str, int], SubstrateMiningContext
        ] = {}
        # Current "work key" (block_number, parent_hash, topology_hash).
        # Set after a successful snapshot fetch; consulted by the
        # storm-prevention check in _handle_result.
        self._current_work_key: Optional[WorkKey] = None
        # LRU set of work keys for which the chain already accepted one
        # of our proofs this head. Subsequent same-key results from
        # sibling handles are dropped without resubmission — that's the
        # actual fix for submission storming. OrderedDict-as-set keeps
        # eviction simple (popitem(last=False)) and order-aware.
        self._closed_work_keys: "OrderedDict[WorkKey, None]" = OrderedDict()
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
        # Per-handle done-sentinel queue. The drainer pushes the
        # response's dispatch_id (not None) so _await_handle_done can
        # block until the *specific* dispatch being cancelled acks.
        self._done_queues: dict[str, asyncio.Queue[int]] = {
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
        # Resolve slot clients from the pool. The pool lazy-connects on
        # first get(), so this is where we actually touch the network —
        # __init__ stays synchronous and side-effect free.
        self.client = await self._pool.get("rpc")
        self._subscription_client = await self._pool.get("subscribe.pow")

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

        # Start the subscription task. The blocking subscribe loop
        # inside substrate-interface runs on the subscription client's
        # executor — keeping it off the main rpc slot means
        # submit_extrinsic / state_call traffic doesn't deadlock
        # against an active subscription. Both slots are already
        # connected (the pool's get() did that).
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
                    try:
                        await self._handle_head(head_hash, block_number)
                    except (WebSocketException, ConnectionError) as exc:
                        # `SubstrateClient._run` reconnects on these but
                        # re-raises so the caller decides retry vs fail.
                        # The next head (~6s) will land on the new
                        # validator; drop this one rather than tearing
                        # down the controller mid-rotation.
                        logger.warning(
                            "head handling hit a connection drop "
                            "(%s: %s); failover already swung the client, "
                            "next head will retry",
                            type(exc).__name__,
                            exc,
                        )
                continue

            if result_task in done:
                envelope = result_task.result()
                try:
                    await self._handle_result(envelope)
                except (WebSocketException, ConnectionError) as exc:
                    # Same shape as the head path: failover already
                    # rotated; the chain will surface the result again as
                    # a stale-drop or accept on the new validator. Don't
                    # crash the controller.
                    self.stats.submission_errors += 1
                    self.stats.last_submission_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.warning(
                        "submit/result handling hit a connection drop "
                        "(%s: %s); failover swung the client, "
                        "controller continues",
                        type(exc).__name__,
                        exc,
                    )
                continue

    async def _handle_head(self, head_hash: bytes, block_number: int) -> None:
        self.stats.heads_observed += 1
        # Cancel any in-flight mining on the prior snapshot, then wait in
        # parallel for each handle to ack the cancel. Without this
        # synchronization, cancel() → clear() → dispatch can wipe a cancel
        # before the worker observes it, leaving the worker mining the OLD
        # context against a stop_event tied to the NEW dispatch. The worker
        # emits a `work_item_done` sentinel (tagged with the dispatch_id)
        # when its loop exits with no result; the drainer surfaces that
        # via _await_handle_done so we confirm cancellation of the
        # *specific* dispatch we issued before re-clearing the event.
        #
        # Parallel wait (asyncio.gather) keeps total stall to ~0.5s
        # regardless of handle count — sequential waits would block for
        # N×0.5s, significant against a 6s block time.
        to_drain = [
            h for h in self.miner_handles
            if h._active_dispatch_id != 0 and not h.stop_event.is_set()
        ]
        cancelled_dispatches: list[tuple[MinerHandle, int]] = []
        for handle in self.miner_handles:
            if handle._active_dispatch_id != 0:
                cancelled_dispatches.append(
                    (handle, handle._active_dispatch_id)
                )
                handle.cancel()
        if cancelled_dispatches:
            await asyncio.gather(
                *[
                    self._await_handle_done(
                        h, dispatch_id=did, timeout=0.5,
                    )
                    for h, did in cancelled_dispatches
                ]
            )

        # Post-MR-!20: the on-chain `derive_nonce` hashes the SCALE-encoded
        # account ID (`blake2_256(account.encode())`) to produce a width-stable
        # 32-byte miner identity. For sr25519/AccountId32 this is just
        # `blake2_256` of the 32-byte pubkey, but routing through the same
        # helper keeps the contract clear if a wider AccountId is introduced.
        canonical_miner = hashlib.blake2b(
            self.signer.account_id_bytes(), digest_size=32
        ).digest()
        context = await self.client.get_mining_snapshot(
            at=head_hash,
            topology_hash=self.topology_hash,
            miner_account_bytes=canonical_miner,
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
        self._current_work_key = _work_key(context)
        logger.info(
            "new head: block=%d hash=0x%s... topology=0x%s... nodes=%d edges=%d",
            context.block_number,
            head_hash.hex()[:16],
            context.topology_hash.hex()[:16],
            len(context.nodes),
            len(context.edges),
        )

        # Dispatch the new context to every handle. Each one will generate
        # its own salts and explore the search space independently. The
        # dispatch_id returned by mine_work_item keys an immutable
        # (handle_id, dispatch_id) → context map so late results from a
        # cancelled dispatch can still find their original context.
        for handle in self.miner_handles:
            dispatch_id = handle.mine_work_item(context)
            self._dispatch_contexts[(handle.miner_id, dispatch_id)] = context
            self._prune_dispatch_contexts(handle.miner_id, dispatch_id)
        self.stats.contexts_dispatched += len(self.miner_handles)
        if self.core is not None:
            # One attempt per head, regardless of how many handles fanned out.
            self.core.record_dispatch()

    async def _handle_result(self, envelope: _ResultEnvelope) -> None:
        self.stats.results_received += 1
        envelope_key = _work_key(envelope.context)

        # Storm prevention: if we already submitted an accepted proof
        # for this work key this head, drop sibling/duplicate results
        # without re-submitting. Distinct from the stale check below —
        # there the head has moved on; here the head is still current
        # but we've already won it. Without this, every additional
        # handle for the same context fires another submit_proof and
        # spams the chain (and our RPC) with redundant work.
        if envelope_key in self._closed_work_keys:
            self.stats.duplicate_result_drops += 1
            logger.info(
                "dropping duplicate result from %s: work_key already won "
                "(block=%d)",
                envelope.handle_id,
                envelope.context.block_number,
            )
            return

        # Drop results produced against a stale context. The controller
        # already moved on to a new head; the chain would reject this proof.
        # topology_hash is included so a governance call that rotates the
        # topology within a single block doesn't slip through as a
        # block/parent-match.
        if (
            self._current_work_key is None
            or envelope_key != self._current_work_key
        ):
            self.stats.stale_drops += 1
            logger.info(
                "dropping stale result from %s: block=%d (current=%s)",
                envelope.handle_id,
                envelope.context.block_number,
                self._current_work_key[0] if self._current_work_key else "<none>",
            )
            return

        # Encoder errors (ValueError on no-solutions / wrong-salt-length)
        # are code defects — retrying won't help. Keep them out of the
        # RPC-error catch below so they raise loudly instead of being
        # logged-and-dropped like a transient network blip.
        try:
            encode_quantum_proof(envelope.result, envelope.context)
        except ValueError as exc:
            raise RuntimeError(f"proof encoding failed (bug): {exc}") from exc

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
            # Mark this work key as won and cancel sibling handles
            # immediately so they don't keep mining (and then submitting
            # redundant proofs the chain will reject). This is the
            # primary submission-storm fix; the duplicate-drop check at
            # the top of _handle_result is the belt-and-suspenders
            # backstop for sibling results that were already in flight
            # when we got here.
            self._mark_work_key_closed(envelope_key)
            self._cancel_siblings_for_won_work(envelope.handle_id)
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
                try:
                    await self.on_proof_submitted(receipt, envelope.context)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "on_proof_submitted callback raised (proof was submitted): "
                        "%s: %s",
                        type(exc).__name__,
                        exc,
                    )
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
        self,
        handle: MinerHandle,
        *,
        dispatch_id: int,
        timeout: float,
    ) -> None:
        """Wait briefly for the worker to confirm cancellation.

        Pops sentinels from the handle's done queue until one carrying
        the *expected* dispatch_id arrives, or the timeout fires. A
        sentinel for an older dispatch (one we already moved past)
        is discarded silently — it can show up when the previous
        cancel raced its own sentinel and the new dispatch is now
        cleaning up.

        On timeout we proceed anyway — but log a WARNING because the
        worker did NOT ack the cancel of this specific dispatch in
        time, so the next dispatch's stop_event.clear() may race
        the prior cancel.
        """
        sentinel_queue = self._done_queues.get(handle.miner_id)
        if sentinel_queue is None:
            return
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    "handle %s did not ack cancel of dispatch_id=%d within "
                    "%.1fs; dispatching anyway — next mine_work_item may "
                    "race the prior cancel",
                    handle.miner_id,
                    dispatch_id,
                    timeout,
                )
                return
            try:
                got = await asyncio.wait_for(
                    sentinel_queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "handle %s did not ack cancel of dispatch_id=%d within "
                    "%.1fs; dispatching anyway — next mine_work_item may "
                    "race the prior cancel",
                    handle.miner_id,
                    dispatch_id,
                    timeout,
                )
                return
            # Match exactly — older dispatch sentinels arriving here are
            # already-resolved and can be discarded. A sentinel for a
            # dispatch newer than the one we're waiting for is logically
            # impossible (we haven't issued one yet on this handle).
            if got is None or got == dispatch_id:
                return

    def _mark_work_key_closed(self, key: WorkKey) -> None:
        """Record a work key as won and evict the oldest if over cap."""
        # Move-to-end semantics: if the same key shows up twice (shouldn't,
        # but be defensive) it stays at the back of the LRU.
        if key in self._closed_work_keys:
            self._closed_work_keys.move_to_end(key)
        else:
            self._closed_work_keys[key] = None
            while len(self._closed_work_keys) > _CLOSED_WORK_KEYS_CAP:
                self._closed_work_keys.popitem(last=False)

    def _cancel_siblings_for_won_work(self, winning_handle_id: str) -> None:
        """Cancel all handles other than the winner.

        Mirrors mempool_miner_controller.py's pattern: once one
        submission has been accepted, sibling handles mining the same
        work item should stop immediately. They may already have a
        valid result in flight — that's fine, the duplicate-drop check
        catches it at the top of _handle_result.
        """
        for handle in self.miner_handles:
            if handle.miner_id == winning_handle_id:
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

    def _prune_dispatch_contexts(
        self, handle_id: str, latest_dispatch_id: int
    ) -> None:
        """Trim per-handle dispatch contexts to the last N attempts.

        Retains a small ring so late results from a cancelled dispatch
        can still find their context, but drops anything older than
        _DISPATCH_CONTEXT_RETENTION attempts back — those are stale
        beyond any plausible late-result window.
        """
        cutoff = latest_dispatch_id - _DISPATCH_CONTEXT_RETENTION
        if cutoff <= 0:
            return
        stale = [
            (h, d) for (h, d) in self._dispatch_contexts
            if h == handle_id and d <= cutoff
        ]
        for key in stale:
            self._dispatch_contexts.pop(key, None)

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
                msg = await loop.run_in_executor(None, handle.resp.get, True, 0.5)
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
            if isinstance(msg, dict) and msg.get("op") == "mine_result":
                # Use the response's dispatch_id (not the handle's
                # current one) to look up the immutable context that was
                # dispatched alongside this attempt. A late result from
                # dispatch N must pair with dispatch N's context, even
                # if dispatch N+1 has already been issued.
                dispatch_id = msg.get("dispatch_id")
                key = (handle.miner_id, dispatch_id)
                context = self._dispatch_contexts.get(key)
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
                    continue
                await self._result_queue.put(
                    _ResultEnvelope(
                        result=result,
                        context=context,
                        handle_id=handle.miner_id,
                    )
                )
            elif isinstance(msg, dict) and msg.get("op") == "work_item_done":
                # Worker finished its mine_work_item loop with no result —
                # almost always because cancel() was observed. Surface it
                # tagged with the dispatch_id so _await_handle_done can
                # synchronize on cancellation of that specific dispatch,
                # not just any sentinel.
                self._done_queues[handle.miner_id].put_nowait(
                    msg.get("dispatch_id")
                )
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
                # Emit the same sentinel as work_item_done so
                # _await_handle_done can synchronize after a mining error.
                # Without this, every mining exception causes a guaranteed
                # timeout on the next cancel.
                self._done_queues[handle.miner_id].put_nowait(
                    msg.get("dispatch_id")
                )
            elif isinstance(msg, dict) and msg.get("op") == "stats":
                # Stats responses are pulled directly by callers of
                # handle.get_stats(); if one lands here it just means
                # nobody was listening — drop and continue. NOTE: while
                # the controller owns the handle, callers MUST NOT call
                # handle.get_stats() directly — the drainer dequeues the
                # response first and blocks the caller forever.
                pass
            else:
                logger.warning(
                    "handle %s sent unrecognized message type=%s op=%s; dropping",
                    handle.miner_id,
                    type(msg).__name__,
                    msg.get("op") if isinstance(msg, dict) else "n/a",
                )

    async def _subscribe_heads(self) -> None:
        """Subscribe to new best heads, with failover on validator drop.

        Loop semantics:
          - `subscribe_new_heads` returning normally → clean shutdown path.
          - `WebSocketException` / `ConnectionError` → trigger one
            `reconnect()` on the subscription client, then re-subscribe.
          - `NoValidatorReachable` from reconnect → fail-loud: record the
            structured attempt log in `stats.last_submission_error` and
            shut the controller down.
          - Any other exception (decoder bugs, RPC type errors, etc.) is
            treated as fatal and shuts down — those are not transient
            connection loss and should not be retried.
        """

        async def callback(block_hash: bytes, block_number: int) -> None:
            # Latest-only semantics: overwrite whatever's pending. If the
            # main loop is busy in submit_proof while heads pile up, we
            # don't want to dispatch against an old one when it returns.
            self._latest_head = (block_hash, block_number)
            self._head_signal.set()

        while not self._shutdown_event.is_set():
            try:
                await self._subscription_client.subscribe_new_heads(callback)
                return  # clean exit — subscribe returned without error
            except (WebSocketException, ConnectionError) as exc:
                logger.warning(
                    "head subscription dropped on %s (%s: %s); failing over",
                    getattr(self._subscription_client, "current_url", "<unknown>"),
                    type(exc).__name__,
                    exc,
                )
                try:
                    await self._subscription_client.reconnect()
                except NoValidatorReachable as fatal:
                    self.stats.last_submission_error = (
                        f"no validators reachable; head subscription cannot "
                        f"recover: {fatal}"
                    )
                    logger.error(
                        "subscription failover exhausted; triggering shutdown:\n%s",
                        fatal,
                    )
                    self.shutdown()
                    return
                # loop iteration: re-subscribe on the new validator
            except Exception as exc:
                self.stats.last_submission_error = (
                    f"head subscription crashed: {type(exc).__name__}: {exc}"
                )
                logger.exception("head subscription crashed; triggering shutdown")
                # We're already on the asyncio loop; no need for
                # call_soon_threadsafe (the subscription wrapper bridges its
                # background thread into our awaitable callback).
                self.shutdown()
                return

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
        # Slot clients are owned by the pool; do NOT close them here.
        # The pool's close() (called from the CLI's outer try/finally)
        # tears them down at process shutdown.


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
    logger.error("unrecognized submission error; classifying as fatal: %r", error)
    return SubmissionOutcome.FATAL


__all__ = [
    "ControllerStats",
    "STALE_SUBMISSION_ERRORS",
    "FATAL_SUBMISSION_ERRORS",
    "SubmissionOutcome",
    "SubstrateMinerController",
    "classify_submission",
]
