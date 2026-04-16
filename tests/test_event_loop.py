"""Tests for event loop helpers and uvloop integration."""

import asyncio

import pytest

from shared.event_loop import create_event_loop, _check_uvloop


def test_create_event_loop_returns_loop():
    """create_event_loop returns a usable event loop."""
    loop = create_event_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    loop.close()


def test_create_event_loop_runs_coroutine():
    """Event loop from create_event_loop can run async code."""
    loop = create_event_loop()

    async def hello():
        return 42

    result = loop.run_until_complete(hello())
    assert result == 42
    loop.close()


def test_uvloop_check_cached():
    """_check_uvloop result is cached between calls."""
    result1 = _check_uvloop()
    result2 = _check_uvloop()
    assert result1 == result2


@pytest.mark.skipif(
    not _check_uvloop(),
    reason="uvloop not installed"
)
def test_uvloop_loop_created():
    """When uvloop is available, create_event_loop uses it."""
    import uvloop
    loop = create_event_loop()
    assert isinstance(loop, uvloop.Loop)
    loop.close()
