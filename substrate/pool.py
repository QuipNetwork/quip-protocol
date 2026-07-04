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
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    # WebSocketException and substrate-interface's own connection errors
    # will be added here when wiring against the real client.
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
        self._sync_probe_timeout_s = float(sync_probe_timeout_s)
        self._sync_poll_interval_s = float(sync_poll_interval_s)
        # URL → last get_sync_state dict for URLs found syncing during the
        # current failure cycle. Cleared on any successful op.
        self._syncing_urls: dict[str, dict[str, Any]] = {}
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
        """Shut down the active handle (if any)."""
        if self._active is not None:
            await self._active.shutdown()
            self._active = None

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
                all_down = await self._swap_after_failure(handle.url)
                if op not in _IDEMPOTENT_OPS:
                    raise ValidatorSwapped(
                        f"swapped during non-idempotent op {op}; caller must decide"
                    ) from conn_exc
                if all_down and self._syncing_urls and await self._sync_wait():
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
        current_url = self._active.url
        await self._swap_after_failure(current_url)

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

        Returns:
            True when sync completed (caller retries the original op with
            a fresh retry budget); False when the probe died mid-wait
            (caller falls back to normal failure accounting).
        """
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

    async def _swap_after_failure(self, failed_url: str) -> bool:
        """Internal: kill the current handle, rotate URL, spawn new handle.

        Returns:
            True if all URLs were exhausted (AllUrlsDown fired), False otherwise.
        """
        async with self._swap_lock:
            # Idempotent within the lock — multiple racing callers
            # only swap once.
            if self._active is None or self._active.url != failed_url:
                return False
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
