"""ValidatorPool — async RPC surface with hot-active swap.

The pool owns one active `ValidatorHandle` at a time. Every RPC call
routes to the active handle. On a connection-class error:
    * The active handle is shut down.
    * The URL failover rotates to the next URL.
    * A new handle is spawned on the next URL.
    * Idempotent ops (`query_*`, `get_*`) auto-retry up to
      `max_swap_retries` against the new handle.
    * `submit_extrinsic` is NOT auto-retried — it raises
      `ValidatorSwapped` so the caller (who has domain knowledge)
      decides. This avoids double-submit when finality lag means the
      original validator accepted the tx but the next one hasn't seen
      it yet.

Non-connection errors (e.g. a `RuntimeError` from the chain RPC
saying "no topology registered") pass through unchanged. The pool
only swaps on connection-class failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from websocket import WebSocketException

from substrate.sync_progress import SyncProgress
from substrate.url_failover import AllUrlsDown, SubstrateUrlFailover
from substrate.validator_handle import ValidatorHandle, ValidatorSwapped

logger = logging.getLogger(__name__)


# Errors that indicate the connection to the validator is broken, not
# that the chain returned a bad result. The pool swaps on these.
# ``ConnectionError`` already covers BrokenPipeError, ConnectionResetError,
# and ConnectionAbortedError. We intentionally do NOT catch generic
# ``OSError`` here — that would treat FileNotFoundError, PermissionError,
# etc. as connection failures and trigger spurious swaps.
#
# ``asyncio.CancelledError`` deliberately does NOT belong here. It derives
# from ``BaseException``, and the write path converts anything in this tuple
# into ``ValidatorSwapped`` — which would swallow a cancellation and break
# shutdown. `send_write` handles it in its own clause and re-raises.
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    # substrate-interface's transport raises this for a dropped or
    # half-open websocket; it is not an OSError, so it needs naming here
    # explicitly or a genuine socket failure propagates raw and the pool
    # never swaps.
    WebSocketException,
)


# Idempotent operations the pool may auto-retry across swaps.
# Anything not in this set raises ValidatorSwapped to the caller instead.
_IDEMPOTENT_OPS = frozenset(
    {
        "ensure_connected",
        "get_head",
        "get_sync_state",
        "get_block_number",
        "get_finalized_head",
        "get_mining_snapshot",
        "query_miner",
        "query_proofs_submitted",
        "query_difficulty",
        "query_current_difficulty",
        "query_mineable_topologies",
        "query_difficulty_for",
        "query_last_proof_block_number",
        "query_pow_constants",
        "query_balance",
        "query_solver",
        "query_job_order",
        "query_winning_solution",
        "query_winning_solution_count",
        "query_latest_qblock_id",
        "get_events_at",
    }
)


# Sync-wait tuning. The probe timeout is deliberately shorter than the
# handle's 10s rpc_call_timeout_s — a live-but-syncing node answers
# system_health in milliseconds, so a slow probe means a dead socket
# and real outages keep the existing swap behavior.
_SYNC_PROBE_TIMEOUT_S = 5.0
_SYNC_POLL_INTERVAL_S = 10.0


class NodeSyncing(ValidatorSwapped):
    """Active validator is in major sync — alive, but the chain is behind.

    Raised instead of swap-and-`ValidatorSwapped` for non-idempotent ops
    (a submit can't usefully reach a syncing node). Subclasses
    ``ValidatorSwapped`` so existing submit-retry callers keep working
    unchanged; new code may catch ``NodeSyncing`` specifically.
    """


class ValidatorPool:
    """Owns one active ValidatorHandle; routes RPCs; handles swap.

    Args:
        urls: List of validator URLs to rotate through.
        failover: `SubstrateUrlFailover` instance (allows shared
            configuration of backoff parameters).
        handle_factory: Function `(url) -> ValidatorHandle`. Production
            code passes `ValidatorHandle`; tests pass a fake factory.
        max_swap_retries: Maximum times an idempotent op retries across
            swaps before raising the underlying connection error.
        sync_probe_timeout_s: Seconds the get_sync_state probe may take
            before the node is treated as down; default 5.0.
        sync_poll_interval_s: Seconds between sync-wait progress probes;
            default 10.0.
    """

    def __init__(
        self,
        urls,
        failover: Optional[SubstrateUrlFailover] = None,
        handle_factory: Optional[Callable[[str], ValidatorHandle]] = None,
        max_swap_retries: int = 3,
        sync_probe_timeout_s: float = _SYNC_PROBE_TIMEOUT_S,
        sync_poll_interval_s: float = _SYNC_POLL_INTERVAL_S,
        reconnect_backoff_s: float = 0.5,
    ) -> None:
        # Accept tuples/lists; normalise to list for internal mutation.
        urls_list = list(urls) if urls is not None else []
        if not urls_list:
            raise ValueError("ValidatorPool requires at least one validator URL")
        self._urls = urls_list
        # Sensible defaults so legacy callers `ValidatorPool(urls=...)` keep
        # working. Tests inject custom failover/handle_factory for isolation.
        if failover is None:
            failover = SubstrateUrlFailover(urls_list)
        if handle_factory is None:

            def handle_factory(url: str) -> ValidatorHandle:  # noqa: E306
                return ValidatorHandle(url=url)

        self._failover = failover
        self._handle_factory = handle_factory
        self._max_swap_retries = max_swap_retries
        self._active: Optional[ValidatorHandle] = None
        self._swap_lock = asyncio.Lock()
        # Dedicated WRITE handle, isolated from the read/snapshot path. A
        # connection-class failure on a read op swaps only ``self._active``,
        # never this handle — so a high-frequency snapshot poll can no longer
        # tear an in-flight submit off its socket (QUI-829 / gh-18). Created
        # lazily on the first write; writes serialize on ``_write_lock`` so
        # only one submit at a time touches the single write handle.
        self._write_active: Optional[ValidatorHandle] = None
        self._write_lock = asyncio.Lock()
        # Set when a submit was cancelled mid-call, leaving the write child
        # still running the abandoned request. The next submit rebuilds the
        # handle instead of queueing behind it (QUI-899).
        self._write_handle_suspect: bool = False
        # Wall-clock time of the last landed write (submit). None until the
        # first success. Surfaced on /api/v1/status so a node that mines but
        # never lands a proof is diagnosable instead of silently stuck.
        self.last_successful_submission: Optional[float] = None
        self._sync_probe_timeout_s = float(sync_probe_timeout_s)
        self._sync_poll_interval_s = float(sync_poll_interval_s)
        # Short backoff before respawning the child on a single-URL
        # deployment. A one-node miner has nowhere to rotate, so a
        # transient client-socket blip is a fast reconnect, not the
        # escalating all-down backoff used when a real fleet is exhausted.
        self._reconnect_backoff_s = float(reconnect_backoff_s)
        # URL → last get_sync_state dict for URLs found syncing during the
        # current failure cycle. Cleared on any successful op.
        self._syncing_urls: dict[str, dict[str, Any]] = {}
        # Serializes sync-wait so concurrent send() callers don't run
        # overlapping probe loops.
        self._sync_wait_lock = asyncio.Lock()
        # Telemetry surface: last observed sync state (plus url/at); None
        # when healthy. The controller's stats snapshot writer reads this.
        self.last_sync_state: Optional[dict[str, Any]] = None

    @property
    def urls(self) -> tuple[str, ...]:
        """Legacy attribute: the immutable tuple of validator URLs."""
        return tuple(self._urls)

    @property
    def current_url(self) -> str:
        """Legacy attribute: the URL currently in use by the failover rotation."""
        return self._failover.current()

    def active_url(self) -> Optional[str]:
        return self._active.url if self._active is not None else None

    async def start(self) -> None:
        """Spawn the first validator handle on the first URL."""
        url = self._failover.current()
        self._active = self._handle_factory(url)
        self._active.start()

    async def shutdown(self) -> None:
        """Shut down the active read handle and the write handle (if any)."""
        if self._active is not None:
            await self._active.shutdown()
            self._active = None
        if self._write_active is not None:
            await self._write_active.shutdown()
            self._write_active = None

    async def send(self, op: str, args: dict[str, Any]) -> Any:
        """Route an RPC call to the active handle, with swap-on-failure.

        For idempotent ops, retries up to `max_swap_retries` across swaps.
        For non-idempotent ops (notably `submit_extrinsic`), raises
        `ValidatorSwapped` on a swap and leaves retry decision to caller.
        """
        attempts = 0
        while True:
            if self._active is None:
                raise RuntimeError("pool not started")
            handle = self._active
            try:
                result = await handle.send(op, args)
                self._failover.confirm_success()
                self._clear_sync_state()
                return result
            except ValidatorSwapped:
                # Our captured handle was shut down by a CONCURRENT swap
                # (another send()/force_swap already replaced self._active).
                # No swap is needed — retry idempotent ops against the now-
                # fresh active handle. Non-idempotent ops surface to the
                # caller, who must re-decide (a submit needs a fresh nonce).
                if op not in _IDEMPOTENT_OPS:
                    raise
                attempts += 1
                if attempts >= self._max_swap_retries:
                    logger.error(
                        "pool: idempotent op %s exhausted retries after "
                        "concurrent swaps (attempts=%d)",
                        op,
                        attempts,
                    )
                    raise
                continue
            except _CONNECTION_ERRORS as conn_exc:
                # A syncing node accepts connections but stalls runtime
                # calls — probe node-level RPCs to tell "down" from
                # "alive but syncing" before treating this as an outage.
                syncing = await self._record_sync_state(handle)
                if syncing and op not in _IDEMPOTENT_OPS:
                    # No swap: rotating away from a live node the caller
                    # must anyway wait out just churns child processes.
                    raise NodeSyncing(
                        f"validator {handle.url} is syncing; op {op} not submitted"
                    ) from conn_exc
                if not syncing:
                    logger.warning(
                        "pool: connection-class error on %s op=%s: %s; swapping",
                        handle.url,
                        op,
                        conn_exc,
                    )
                all_down = await self._swap_after_failure(handle)
                if op not in _IDEMPOTENT_OPS:
                    raise ValidatorSwapped(
                        f"swapped during non-idempotent op {op}; caller must decide"
                    ) from conn_exc
                # A single-URL pool never reports all_down (it fast-reconnects
                # instead), so include len==1 to route a still-syncing local
                # node into the quiet sync-wait rather than churning children.
                if (
                    self._syncing_urls
                    and (all_down or len(self._urls) == 1)
                    and await self._sync_wait()
                ):
                    # Sync finished — retry the op with a fresh budget.
                    attempts = 0
                    continue
                attempts += 1
                if all_down or attempts >= self._max_swap_retries:
                    logger.error(
                        "pool: idempotent op %s exhausted retries (all_down=%s attempts=%d)",
                        op,
                        all_down,
                        attempts,
                    )
                    raise
                # loop and try on the new active handle

    async def force_swap(self) -> None:
        """Kill the active handle and spawn the next URL (no retry logic)."""
        if self._active is None:
            return
        await self._swap_after_failure(self._active)

    # ------------------------------------------------------------------
    # Dedicated write path (submits) — isolated from read-path swaps.
    # ------------------------------------------------------------------

    async def send_write(self, op: str, args: dict[str, Any]) -> Any:
        """Route a non-idempotent write (submit) through a dedicated handle.

        The write handle never shares a socket with the read/snapshot path,
        so a snapshot-poll connection swap can no longer cancel an in-flight
        submit (QUI-829 / gh-18). Submits are rare, so the write handle is
        created lazily and its (often-idle, possibly-stale) socket is
        health-reconnected on demand before each write. Writes serialize on
        ``_write_lock`` — one submit at a time touches the single handle.

        On a connection-class failure the write handle is reconnected and
        ``ValidatorSwapped`` is raised: the caller re-composes with a fresh
        nonce and retries (same contract as the old shared-handle submit,
        which surfaced ``ValidatorSwapped`` on a mid-flight swap).
        """
        async with self._write_lock:
            handle = await self._ensure_write_handle_connected()
            try:
                result = await handle.send(op, args)
            except ValidatorSwapped:
                # The handle was shut down under us (only pool shutdown does
                # this to a write handle). Surface so the caller re-decides;
                # never silently retry a non-idempotent write.
                raise
            except _CONNECTION_ERRORS as conn_exc:
                await self._reconnect_write_handle(handle)
                raise ValidatorSwapped(
                    f"write handle swapped during {op}; caller must re-sign"
                ) from conn_exc
            except asyncio.CancelledError:
                # An outer `wait_for` gave up on this submit. The child is
                # still running the abandoned call, so the next submit must
                # not queue behind it (QUI-899). Flag for teardown rather
                # than awaiting cleanup here — awaiting inside a cancellation
                # is its own hazard — and re-raise, because converting a
                # cancellation into a normal error breaks shutdown. This is
                # also why `CancelledError` must never join
                # `_CONNECTION_ERRORS`: that path swallows it.
                self._write_handle_suspect = True
                raise
            self._failover.confirm_success()
            self._clear_sync_state()
            self.last_successful_submission = time.time()
            return result

    async def _ensure_write_handle_connected(self) -> ValidatorHandle:
        """Return the write handle with a freshly health-probed connection.

        Lazily spawns the write handle on the current failover URL. Because
        submits are rare the socket is usually idle/stale, so we drive the
        child's own ``ensure_connected`` (which reconnects a dead socket in
        place); if the child process itself is unreachable, respawn it once.
        Called only while holding ``_write_lock``.
        """
        if self._write_handle_suspect:
            # A previous submit was cancelled mid-call; that child may still
            # be busy with it. Rebuild before reusing (QUI-899).
            self._write_handle_suspect = False
            if self._write_active is not None:
                await self._reconnect_write_handle(self._write_active)
        if self._write_active is None:
            self._write_active = self._handle_factory(self._failover.current())
            self._write_active.start()
        handle = self._write_active
        try:
            await handle.send("ensure_connected", {})
        except (ValidatorSwapped, *_CONNECTION_ERRORS):
            await self._reconnect_write_handle(handle)
            handle = self._write_active
            await handle.send("ensure_connected", {})
        return handle

    async def _reconnect_write_handle(self, failed: ValidatorHandle) -> None:
        """Shut a dead write handle and respawn it on the current URL.

        Called only while holding ``_write_lock`` (writes are serialized), so
        no separate swap lock is needed. Identity-guarded so a stale reference
        can't tear down an already-respawned handle.
        """
        if self._write_active is not failed:
            return
        old = self._write_active
        self._write_active = None
        await old.shutdown()
        self._write_active = self._handle_factory(self._failover.current())
        self._write_active.start()

    async def _probe_sync_state(self, handle: ValidatorHandle) -> Optional[dict[str, Any]]:
        """Fetch node sync state from `handle`; None on any failure.

        None means "treat the node as genuinely down" — the normal swap
        path applies. Bounded by the short probe timeout so a dead
        socket can't stall failure handling.
        """
        try:
            state = await asyncio.wait_for(
                handle.send("get_sync_state", {}),
                timeout=self._sync_probe_timeout_s,
            )
        except Exception:  # noqa: BLE001 — any probe failure means "down"
            return None
        return state if isinstance(state, dict) else None

    async def _record_sync_state(self, handle: ValidatorHandle) -> bool:
        """Probe after a connection-class error; record + report syncing nodes."""
        state = await self._probe_sync_state(handle)
        if state is None or not state.get("is_syncing"):
            # Remove any stale entry: without this, a syncing record left by a
            # failed non-idempotent op (NodeSyncing path, no swap) leaks here
            # and mis-steers the AllUrlsDown branch into the quiet sync-wait
            # path when the node is genuinely down.
            self._syncing_urls.pop(handle.url, None)
            return False
        self._syncing_urls[handle.url] = state
        self._publish_sync_state(handle.url, state)
        logger.info(
            "pool: validator %s is alive but syncing (block %s/%s); not treating as down",
            handle.url,
            state.get("current_block"),
            state.get("highest_block"),
        )
        return True

    def _publish_sync_state(self, url: str, state: dict[str, Any]) -> None:
        self.last_sync_state = {**state, "url": url, "at": time.time()}

    def _clear_sync_state(self) -> None:
        if self._syncing_urls:
            self._syncing_urls.clear()
        self.last_sync_state = None

    def _best_syncing_url(self) -> str:
        """The syncing URL with the most-advanced chain (highest current_block)."""
        return max(
            self._syncing_urls.items(),
            key=lambda kv: int(kv[1].get("current_block") or 0),
        )[0]

    async def _sync_wait(self) -> bool:
        """Block until the active (syncing) validator catches up.

        Probes ``get_sync_state`` every ``sync_poll_interval_s``, logging
        one progress line per probe. This deliberately blocks the caller
        of ``send()`` — "mining paused until the node syncs" is the
        wanted semantics, and the event manager sits quietly awaiting it
        instead of tracebacking every poll.

        Serialized by ``_sync_wait_lock``: concurrent callers park on the
        lock; the winner runs the probe loop and clears ``_syncing_urls``
        on completion. A parked caller that acquires the lock after the
        winner finishes short-circuits immediately — the syncing records
        are already gone, so there is nothing left to wait for.

        Returns:
            True when sync completed (caller retries the original op with
            a fresh retry budget); False when the probe died mid-wait
            (caller falls back to normal failure accounting).
        """
        async with self._sync_wait_lock:
            # A caller that parked on the lock while another sync-wait ran
            # to completion has nothing left to wait for — the finisher
            # cleared the syncing records on its way out.
            if not self._syncing_urls:
                return True
            progress = SyncProgress()
            logger.warning(
                "validator node is syncing; mining paused until sync completes"
            )
            while True:
                handle = self._active
                if handle is None:
                    return False
                state = await self._probe_sync_state(handle)
                if state is None:
                    logger.warning(
                        "pool: sync probe failed on %s; resuming normal failover",
                        handle.url,
                    )
                    self._syncing_urls.pop(handle.url, None)
                    return False
                self._publish_sync_state(handle.url, state)
                if not state.get("is_syncing"):
                    logger.info(
                        "pool: validator %s finished syncing; resuming", handle.url
                    )
                    self._clear_sync_state()
                    return True
                logger.info("%s", progress.observe(state, time.monotonic()))
                await asyncio.sleep(self._sync_poll_interval_s)

    async def _swap_after_failure(self, failed_handle: ValidatorHandle) -> bool:
        """Internal: kill the current handle, rotate URL, spawn new handle.

        Takes the *handle object* the caller was using, not its URL, so the
        idempotency check is identity-based: a racing caller whose handle was
        already swapped out — even to the same URL, the single-URL case where
        URL comparison can never detect it — is a no-op and must not tear down
        the freshly-spawned handle.

        Returns:
            True if all URLs were exhausted (AllUrlsDown fired), False otherwise.
        """
        async with self._swap_lock:
            # Identity-based idempotency: only the caller still holding the
            # current active handle performs the swap.
            if self._active is None or self._active is not failed_handle:
                return False
            failed_url = failed_handle.url
            old = self._active
            all_down = False
            try:
                next_url = self._failover.advance_after_failure(failed_url)
            except AllUrlsDown as down:
                all_down = True
                if self._syncing_urls:
                    # A live-but-syncing node is not an outage: skip the
                    # ERROR + exponential backoff and hand the caller the
                    # most-advanced syncing URL to sync-wait against.
                    # (If that node dies mid-wait, advance_after_failure
                    # may log a current-URL mismatch warning once —
                    # harmless; the rotation still advances.)
                    next_url = self._best_syncing_url()
                    self._failover.reset_after_backoff()
                    logger.info(
                        "pool: no healthy validator (%d syncing); waiting on %s",
                        len(self._syncing_urls),
                        next_url,
                    )
                elif len(self._urls) == 1:
                    # Single-URL deployment: a failure means "respawn my child
                    # on the ONLY node", not a fleet-wide outage. Skip the
                    # escalating all-down backoff (a transient client-socket
                    # blip must not become a 60s stall, nor hold _swap_lock for
                    # a minute). all_down stays False so the idempotent caller
                    # retries the fresh handle within its max_swap_retries
                    # budget instead of being forced to raise.
                    all_down = False
                    self._failover.reset_after_backoff()
                    next_url = self._failover.current()
                    logger.info(
                        "pool: reconnecting single validator %s after "
                        "transient failure",
                        failed_url,
                    )
                    await asyncio.sleep(self._reconnect_backoff_s)
                else:
                    logger.error(
                        "pool: all validator URLs down; backing off %.2fs",
                        down.backoff_s,
                    )
                    await asyncio.sleep(down.backoff_s)
                    self._failover.reset_after_backoff()
                    next_url = self._failover.current()
            await old.shutdown()
            self._active = self._handle_factory(next_url)
            self._active.start()
            logger.info("pool: swapped to %s", next_url)
            return all_down
