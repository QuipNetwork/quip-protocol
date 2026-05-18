"""
Async non-blocking sender for ``multiprocessing.Connection`` pipes.

The freeze captured in ``dump.txt`` was a bidirectional-duplex pipe
deadlock: both sides of each duplex socketpair did synchronous
``pipe.send`` on the asyncio event-loop thread, and when the kernel
buffers filled, the whole event loop stopped. See the plan at
``/Users/carback1/.claude/plans/include-all-3-original-zesty-toucan.md``
for the full mechanism.

``AsyncPipeSender`` decouples the event loop from ``pipe.send`` by
pushing messages through a bounded ``asyncio.Queue`` that a dedicated
drainer task pops and writes off-thread via ``asyncio.to_thread``.
Producers use two APIs:

* ``send_nowait`` — synchronous enqueue, drop-newest on full. Used for
  fire-and-forget commands where backpressure should not block the
  event loop. Caller must run on the event loop thread.
* ``send_blocking`` — async enqueue with timeout. Used for critical
  and RPC-reply messages where silent drops would leave a correlated
  future dangling until its own timeout.

Either API returns ``False`` (instead of raising) on a broken pipe,
full queue, or timeout, so callers can log and continue.

SPDX-License-Identifier: AGPL-3.0-or-later
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from typing import Any, Optional


_SENTINEL: Any = object()


class AsyncPipeSender:
    """Bounded asyncio-queue wrapper over ``multiprocessing.Connection``.

    Args:
        pipe: The parent-side or child-side pipe endpoint to write.
        maxsize: Queue capacity (default 256). Larger is safer for
            bursty workloads; smaller causes earlier drops under
            sustained coordinator-side slowness.
        name: Short identifier for log lines (e.g. peer address or
            ``"listener"``).
        logger: Logger instance; falls back to the module logger.
    """

    def __init__(
        self,
        pipe: "mp.connection.Connection",
        *,
        maxsize: int = 256,
        name: str = "",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._pipe = pipe
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._name = name
        self._logger = logger or logging.getLogger(__name__)
        self._dead = False
        self._closing = False
        self._dropped_full = 0
        self._dropped_last_logged = 0
        self._drain_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the drainer task. Call once, from the event loop thread.

        Optional — ``send_nowait`` and ``send_blocking`` lazy-start on
        first use so handles constructed in sync contexts (tests) do
        not fail; the drainer only needs a running event loop at send
        time.
        """
        self._ensure_drainer()

    def _ensure_drainer(self) -> bool:
        """Start the drainer task if not yet running. Returns True if a
        drainer is active, False if no event loop is running (in which
        case the caller should not enqueue).
        """
        if self._drain_task is not None:
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._drain_task = loop.create_task(
            self._drain(), name=f"pipe-sender-{self._name}",
        )
        return True

    @property
    def dead(self) -> bool:
        """True when the pipe is known-broken; further sends return False."""
        return self._dead

    @property
    def dropped(self) -> int:
        """Count of messages dropped due to full queue (cumulative)."""
        return self._dropped_full

    def send_nowait(self, msg: dict) -> bool:
        """Drop-newest enqueue. Returns False on full queue or dead pipe.

        Must be called from the event loop thread. Safe to call from a
        synchronous function that was itself invoked from a coroutine,
        because ``asyncio.Queue.put_nowait`` only manipulates in-memory
        state owned by the current loop.

        When called without a running event loop (tests, atexit sync
        shutdown), falls back to a direct synchronous ``pipe.send`` —
        the whole point of the async queue is to avoid blocking a
        running event loop, so without one, blocking is harmless.
        """
        if self._dead or self._closing:
            return False
        if not self._ensure_drainer():
            return self._direct_send(msg)
        try:
            self._queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            self._dropped_full += 1
            # First drop + every 100 subsequent drops emit a WARN so
            # backlog is visible without flooding logs. Report the
            # delta since the last log (not cumulative) so a spiking
            # drop rate is distinguishable from a flat lifetime total.
            if self._dropped_full == 1 or self._dropped_full % 100 == 0:
                delta = self._dropped_full - self._dropped_last_logged
                self._dropped_last_logged = self._dropped_full
                self._logger.warning(
                    "AsyncPipeSender[%s] queue full; dropped %d new "
                    "(total %d)",
                    self._name or "?", delta, self._dropped_full,
                )
            return False

    async def send_blocking(self, msg: dict, timeout: float = 5.0) -> bool:
        """Await enqueue with timeout. Use for critical / RPC-reply sends.

        Returns True on enqueue, False if the pipe is dead, the sender
        is closing, or the timeout expires (meaning the drainer is not
        keeping up — extreme backlog).
        """
        if self._dead or self._closing:
            return False
        if not self._ensure_drainer():
            return self._direct_send(msg)
        try:
            await asyncio.wait_for(self._queue.put(msg), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            self._logger.warning(
                "AsyncPipeSender[%s] put timed out after %ss (queue full)",
                self._name or "?", timeout,
            )
            return False

    def _direct_send(self, msg: dict) -> bool:
        """Synchronous pipe.send fallback when no event loop is running.

        Only reached from sync-context callers (tests, atexit). Marks
        the sender dead on pipe failure so subsequent sends fast-fail.
        """
        try:
            self._pipe.send(msg)
            return True
        except (BrokenPipeError, EOFError, OSError):
            self._dead = True
            return False

    async def _drain(self) -> None:
        """Background task: pop messages and write to pipe off-thread."""
        while True:
            msg = await self._queue.get()
            if msg is _SENTINEL:
                break
            try:
                await asyncio.to_thread(self._pipe.send, msg)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._logger.debug(
                    "AsyncPipeSender[%s] pipe broken: %s",
                    self._name or "?", exc,
                )
                self._dead = True
                break
            except Exception:
                # Keep the drainer alive on transient serialization
                # errors; one bad message must not freeze the sender.
                self._logger.exception(
                    "AsyncPipeSender[%s] unexpected send error",
                    self._name or "?",
                )

    async def close(self, timeout: float = 2.0) -> None:
        """Stop the drainer, waiting up to *timeout* seconds for flush.

        Subsequent ``send_*`` calls return False. Safe to call multiple
        times. Not raising on timeout — shutdown must make progress.
        """
        if self._drain_task is None:
            return
        self._closing = True
        try:
            self._queue.put_nowait(_SENTINEL)
        except asyncio.QueueFull:
            # Full queue: sentinel will be picked up after the drainer
            # works through the backlog, or we force-cancel below.
            pass
        try:
            await asyncio.wait_for(self._drain_task, timeout=timeout)
        except asyncio.TimeoutError:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        self._drain_task = None
        self._dead = True
