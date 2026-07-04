"""Chain event manager — polls the pool, dedups, dispatches typed events.

The event manager is an asyncio task on the controller's main loop. It
owns:
    * a polling loop that calls `pool.send(snapshot_op, snapshot_args)`
      at adaptive cadence (`settled_poll_interval_s` in steady state,
      `catch_up_poll_interval_s` when overdue for a new head);
    * a subscriber dict mapping event_type → list of async callbacks;
    * a watchdog that calls `pool.force_swap()` when no state change
      has been observed within `dead_blocktime_multiplier × blocktime_s`.

Subscribers are registered at startup via `subscribe(event_type, cb)`.
Callbacks may be sync or async; exceptions raised by one callback are
caught and logged, never affecting siblings or the dispatch loop.

The poll loop and dispatch loop are independent asyncio tasks, both
started by `run()`. They communicate via an internal `asyncio.Queue`
(NOT IPC — same process). Backpressure: bounded queue drops oldest
on overflow (only happens if dispatch is wedged, which is a bug).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Hashable

logger = logging.getLogger(__name__)


class ChainEventManager:
    """Polls the validator pool, dedups, dispatches typed events.

    Args:
        pool: Anything with an async `send(op, args)` and async
            `force_swap()`. Production: `substrate.pool.ValidatorPool`.
            Tests: `_FakePool` in test_chain_event_manager.
        state_key: Callable that turns a snapshot payload into a
            hashable key for dedup. Same key → no event.
        snapshot_op: RPC op name to poll for. Typically
            `"get_mining_snapshot"`.
        snapshot_args: kwargs passed to the RPC call.
        blocktime_s: Known chain block production cadence.
        settled_poll_pct: Steady-state poll interval as a fraction of
            blocktime (e.g. 0.85 → ~5.1s for 6s blocktime).
        catch_up_poll_pct: Overdue-block poll interval (e.g. 0.10).
        stale_blocktime_multiplier: Warn if no state change in
            `multiplier × blocktime` (default 1.0).
        dead_blocktime_multiplier: Force-swap if no state change in
            `multiplier × blocktime` (default 3.0).
        event_q_size: Bound on the internal subscriber-dispatch queue.
            Default 64.
    """

    def __init__(
        self,
        pool: Any,
        state_key: Callable[[Any], Hashable],
        snapshot_op: str,
        snapshot_args: dict,
        blocktime_s: float = 6.0,
        settled_poll_pct: float = 0.85,
        catch_up_poll_pct: float = 0.10,
        stale_blocktime_multiplier: float = 1.0,
        dead_blocktime_multiplier: float = 3.0,
        event_q_size: int = 64,
    ) -> None:
        self._pool = pool
        self._state_key = state_key
        self._snapshot_op = snapshot_op
        self._snapshot_args = snapshot_args
        self._blocktime_s = blocktime_s
        self._settled_s = blocktime_s * settled_poll_pct
        self._catch_up_s = blocktime_s * catch_up_poll_pct
        self._stale_s = blocktime_s * stale_blocktime_multiplier
        self._dead_s = blocktime_s * dead_blocktime_multiplier
        self._subscribers: dict[str, list[Callable]] = {}
        self._event_q: asyncio.Queue = asyncio.Queue(maxsize=event_q_size)
        self._shutdown = asyncio.Event()
        self._last_state_key: Any = None
        self._last_change_at: float = time.monotonic()
        self._force_swap_inflight = False
        # Monotonic time of the last poll whose RPC succeeded (payload may
        # be None — that's a live validator with an unseeded chain). None
        # until the first success. Unlike ``_last_change_at`` this is a
        # POLL-liveness signal, not a state-change signal, and force_swap
        # does not reset it: the controller reads it to decide when the
        # validator is dark and local threshold decay should take over.
        self.last_successful_poll_monotonic: float | None = None

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a callback for `event_type`. May be sync or async."""
        self._subscribers.setdefault(event_type, []).append(callback)

    def request_shutdown(self) -> None:
        """Signal both loops to exit."""
        self._shutdown.set()

    async def run(self) -> None:
        """Start poll and dispatch loops; await shutdown."""
        await asyncio.gather(
            self._poll_loop(),
            self._dispatch_loop(),
            return_exceptions=False,
        )

    async def _poll_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                payload = await self._pool.send(self._snapshot_op, self._snapshot_args)
            except Exception:
                logger.exception("event manager: poll failed; will retry next interval")
                # The watchdog must run on the failure path too: a
                # validator whose every poll raises never reaches the
                # success path, and without this check force_swap would
                # never fire (observed 2026-07-04: 30 min of timeouts,
                # zero swaps).
                await self._check_watchdog(time.monotonic())
                await self._sleep_or_shutdown(self._settled_s)
                continue

            key = self._state_key(payload)
            now = time.monotonic()
            self.last_successful_poll_monotonic = now
            if key != self._last_state_key:
                self._last_state_key = key
                self._last_change_at = now
                self._enqueue_drop_oldest(("new_head", payload))

            await self._check_watchdog(now)

            # Adaptive cadence: if we're overdue for a new head, poll fast
            interval = (
                self._catch_up_s
                if time.monotonic() - self._last_change_at > self._settled_s
                else self._settled_s
            )
            await self._sleep_or_shutdown(interval)

    async def _check_watchdog(self, now: float) -> None:
        """Warn on stale state; force-swap the pool past the dead threshold.

        Runs on BOTH poll outcomes: success (chain may be stuck on the
        same head) and failure (validator may be up but not serving
        state). ``_last_change_at`` is the baseline either way — a
        failing poll can't change state, so staleness keeps accruing.
        """
        elapsed_since_change = now - self._last_change_at
        if elapsed_since_change > self._dead_s and not self._force_swap_inflight:
            logger.error(
                "event manager: no state change for %.2fs > dead threshold %.2fs; force-swapping",
                elapsed_since_change, self._dead_s,
            )
            self._force_swap_inflight = True
            try:
                await asyncio.wait_for(self._pool.force_swap(), timeout=self._dead_s)
            except asyncio.TimeoutError:
                logger.error(
                    "event manager: force_swap exceeded %.2fs timeout; will retry next watchdog tick",
                    self._dead_s,
                )
            finally:
                self._force_swap_inflight = False
            # Reset baseline so we don't immediately force_swap again
            self._last_change_at = time.monotonic()
        elif elapsed_since_change > self._stale_s:
            logger.warning(
                "event manager: no state change for %.2fs (stale threshold %.2fs)",
                elapsed_since_change, self._stale_s,
            )

    def _enqueue_drop_oldest(self, item: Any) -> None:
        """Put *item* on the event queue, dropping the oldest entry if full."""
        try:
            self._event_q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._event_q.get_nowait()
                self._event_q.put_nowait(item)
                logger.warning("event manager: event_q full; dropped oldest to make room")
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                logger.warning("event manager: could not recover from full queue; dropping event")

    async def _dispatch_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                event_type, payload = await asyncio.wait_for(
                    self._event_q.get(), timeout=0.5,
                )
            except asyncio.TimeoutError:
                continue
            for cb in self._subscribers.get(event_type, []):
                try:
                    result = cb(payload)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception(
                        "event manager: subscriber for %s raised; continuing",
                        event_type,
                    )

    async def _sleep_or_shutdown(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
