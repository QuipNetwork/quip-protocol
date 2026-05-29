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
from typing import Any, Callable, Optional

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
_IDEMPOTENT_OPS = frozenset({
    "get_head",
    "get_block_number",
    "get_finalized_head",
    "get_mining_snapshot",
    "query_miner",
    "query_proofs_submitted",
    "query_difficulty",
    "query_current_difficulty",
    "query_last_proof_block_number",
    "query_pow_constants",
    "query_balance",
    "query_solver",
    "query_job_order",
    "query_winning_solution",
    "get_events_at",
})


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
    """

    def __init__(
        self,
        urls,
        failover: Optional[SubstrateUrlFailover] = None,
        handle_factory: Optional[Callable[[str], ValidatorHandle]] = None,
        max_swap_retries: int = 3,
    ) -> None:
        # Accept tuples/lists; normalise to list for internal mutation.
        urls_list = list(urls) if urls is not None else []
        if not urls_list:
            raise ValueError(
                "ValidatorPool requires at least one validator URL"
            )
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

    async def close(self) -> None:
        """Shut down the active validator handle. Idempotent."""
        await self.shutdown()

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
                return result
            except _CONNECTION_ERRORS as conn_exc:
                logger.warning(
                    "pool: connection-class error on %s op=%s: %s; swapping",
                    handle.url, op, conn_exc,
                )
                all_down = await self._swap_after_failure(handle.url)
                if op not in _IDEMPOTENT_OPS:
                    raise ValidatorSwapped(
                        f"swapped during non-idempotent op {op}; caller must decide"
                    ) from conn_exc
                attempts += 1
                if all_down or attempts >= self._max_swap_retries:
                    logger.error(
                        "pool: idempotent op %s exhausted retries (all_down=%s attempts=%d)",
                        op, all_down, attempts,
                    )
                    raise
                # loop and try on the new active handle

    async def force_swap(self) -> None:
        """Kill the active handle and spawn the next URL (no retry logic)."""
        if self._active is None:
            return
        current_url = self._active.url
        await self._swap_after_failure(current_url)

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
