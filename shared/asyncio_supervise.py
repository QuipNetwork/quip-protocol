"""Asyncio task supervisor: turn silent task death into loud failure.

`supervise(coro, name, on_failure)` is the load-bearing primitive that
turns silent task death into loud controller shutdown. Every long-lived
asyncio task in the controller wraps in supervise() so an uncaught
exception triggers `on_failure()` (typically setting a shutdown event)
and re-raises so the caller's task itself dies loudly.

Cancellation is treated as a normal control-flow event and passes
through unchanged.

The original failure class fixed by this plan was a silently dying
asyncio task (the head subscription bridge). Wrapping every long-lived
task in supervise() makes that class of bug categorically impossible.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger(__name__)

OnFailure = Union[Callable[[], None], Callable[[], Awaitable[None]]]


async def supervise(
    coro: Awaitable[Any],
    name: str,
    on_failure: OnFailure,
) -> Any:
    """Await `coro`; if it raises, call `on_failure()` and re-raise.

    Args:
        coro: The awaitable to supervise (typically a long-lived task).
        name: Human-readable name used in log messages.
        on_failure: Callback invoked exactly once if `coro` raises a
            non-cancellation exception. May be sync or async. Errors
            raised by `on_failure` itself are logged but do not mask
            the original exception.

    Returns:
        Whatever `coro` returns on normal completion.

    Raises:
        asyncio.CancelledError: propagated unchanged.
        Exception: the original exception from `coro`, re-raised after
            `on_failure` has been invoked.
    """
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("supervised task %s crashed; escalating to on_failure", name)
        try:
            result = on_failure()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "on_failure callback for supervised task %s itself raised; "
                "original exception will still propagate",
                name,
            )
        raise
