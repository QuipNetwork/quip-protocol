"""Tests for shared.asyncio_supervise.supervise().

`supervise(coro, name, on_failure)` is the load-bearing primitive that
turns silent task death into loud controller shutdown. Every long-lived
asyncio task in the controller wraps in supervise() so an uncaught
exception triggers shutdown rather than dangling indefinitely.
"""
from __future__ import annotations

import asyncio

import pytest

from shared.asyncio_supervise import supervise


@pytest.mark.asyncio
async def test_normal_return_propagates_result():
    """If the supervised coro returns normally, supervise returns the same value."""
    async def coro():
        return 42

    on_failure_called = []
    result = await supervise(coro(), "ok-task", lambda: on_failure_called.append(True))

    assert result == 42
    assert on_failure_called == []


@pytest.mark.asyncio
async def test_exception_triggers_on_failure_and_reraises():
    """An uncaught exception calls on_failure() and re-raises (loud failure)."""
    class BoomError(Exception):
        pass

    async def coro():
        raise BoomError("simulated crash")

    on_failure_called = []
    with pytest.raises(BoomError, match="simulated crash"):
        await supervise(coro(), "boom-task", lambda: on_failure_called.append(True))

    assert on_failure_called == [True]


@pytest.mark.asyncio
async def test_cancellation_passes_through_without_on_failure():
    """CancelledError is a normal control-flow event, not a crash. on_failure stays unused."""
    async def coro():
        await asyncio.sleep(1.0)  # will be cancelled

    on_failure_called = []

    async def runner():
        await supervise(coro(), "cancellable", lambda: on_failure_called.append(True))

    task = asyncio.create_task(runner())
    await asyncio.sleep(0)  # let the task start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert on_failure_called == []


@pytest.mark.asyncio
async def test_on_failure_callable_can_be_sync_or_async():
    """on_failure may be a plain callable or a coroutine function; both work."""
    async def coro():
        raise RuntimeError("x")

    sync_called = []

    def sync_on_failure():
        sync_called.append(True)

    with pytest.raises(RuntimeError):
        await supervise(coro(), "sync-cb", sync_on_failure)
    assert sync_called == [True]

    async_called = []

    async def async_on_failure():
        async_called.append(True)

    async def coro2():
        raise RuntimeError("y")

    with pytest.raises(RuntimeError):
        await supervise(coro2(), "async-cb", async_on_failure)
    assert async_called == [True]


@pytest.mark.asyncio
async def test_on_failure_exception_does_not_mask_original():
    """If on_failure itself raises, the original exception is still what the caller sees."""
    async def coro():
        raise ValueError("original")

    def bad_on_failure():
        raise RuntimeError("nested failure handler bug")

    # The original ValueError must propagate; the RuntimeError from on_failure
    # must be logged but not mask the real cause.
    with pytest.raises(ValueError, match="original"):
        await supervise(coro(), "nested-bug", bad_on_failure)
