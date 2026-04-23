"""Tests for ``AsyncPipeSender``.

The sender is the core primitive that eliminates the bidirectional-
pipe deadlock captured in the production ``dump.txt`` — it decouples
``pipe.send`` from the asyncio event loop via a bounded queue and an
off-thread drainer. These tests exercise the drop-newest, blocking-
with-timeout, broken-pipe, and close paths directly, without
spawning real child processes.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest

from shared.pipe_sender import AsyncPipeSender


def _make_pipe() -> tuple:
    """Return ``(parent, child)`` pipe ends for a duplex socketpair."""
    return mp.Pipe(duplex=True)


@pytest.mark.asyncio
async def test_send_nowait_roundtrip():
    """Enqueued messages reach the other end via the drainer."""
    parent, child = _make_pipe()
    sender = AsyncPipeSender(parent, maxsize=8, name="rt")
    sender.start()

    for i in range(4):
        assert sender.send_nowait({"i": i}) is True

    # Give the drainer a couple of event-loop turns to flush.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if child.poll(0):
            break

    received = []
    while child.poll(0):
        received.append(child.recv())
    assert len(received) == 4
    assert [m["i"] for m in received] == [0, 1, 2, 3]

    await sender.close(timeout=1.0)


@pytest.mark.asyncio
async def test_send_nowait_drops_on_full():
    """A full queue drops newest and increments the counter."""
    parent, _child = _make_pipe()
    # Use a small queue and hold the drainer by not awaiting so the
    # queue saturates before the first drain iteration.
    sender = AsyncPipeSender(parent, maxsize=2, name="drop")
    sender.start()

    accepted = 0
    rejected = 0
    for i in range(10):
        if sender.send_nowait({"i": i}):
            accepted += 1
        else:
            rejected += 1

    # With no awaits, the drainer hasn't run; queue holds at most
    # maxsize items.
    assert accepted >= 2
    assert rejected >= 1
    assert sender.dropped == rejected

    await sender.close(timeout=1.0)


@pytest.mark.asyncio
async def test_send_blocking_waits_for_capacity():
    """``send_blocking`` awaits space, then succeeds."""
    parent, child = _make_pipe()
    sender = AsyncPipeSender(parent, maxsize=1, name="blk")
    sender.start()

    assert sender.send_nowait({"pre": True}) is True
    # Now queue likely full or 1 slot; subsequent blocking put should
    # wait for drainer then enqueue.
    ok = await sender.send_blocking({"after": True}, timeout=2.0)
    assert ok is True

    # Give drainer time to flush.
    received = []
    for _ in range(50):
        await asyncio.sleep(0.01)
        while child.poll(0):
            received.append(child.recv())
        if len(received) >= 2:
            break
    assert {"pre": True} in received
    assert {"after": True} in received

    await sender.close(timeout=1.0)


@pytest.mark.asyncio
async def test_broken_pipe_marks_dead():
    """Drainer detects broken pipe and disables further sends."""
    parent, child = _make_pipe()
    sender = AsyncPipeSender(parent, maxsize=4, name="brk")
    sender.start()

    # Close the receiving end so the next send raises BrokenPipeError.
    child.close()

    # Push one message; drainer will try to send and fail.
    sender.send_nowait({"x": 1})
    # Allow drainer to observe the broken pipe.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if sender.dead:
            break
    assert sender.dead is True

    # Subsequent sends fast-fail.
    assert sender.send_nowait({"y": 2}) is False
    ok = await sender.send_blocking({"z": 3}, timeout=0.5)
    assert ok is False

    await sender.close(timeout=1.0)


@pytest.mark.asyncio
async def test_close_flushes_pending():
    """``close`` drains queued messages before marking dead."""
    parent, child = _make_pipe()
    sender = AsyncPipeSender(parent, maxsize=32, name="flush")
    sender.start()

    for i in range(20):
        sender.send_nowait({"i": i})

    await sender.close(timeout=2.0)
    assert sender.dead is True

    received = []
    while child.poll(0):
        received.append(child.recv())
    # All 20 queued messages should have flushed during close.
    assert len(received) == 20


def test_send_nowait_without_running_loop_falls_back():
    """No event loop → direct pipe.send (sync path for tests/atexit)."""
    parent, child = _make_pipe()
    sender = AsyncPipeSender(parent, maxsize=4, name="sync")

    # No running loop in this sync-level test function.
    assert sender.send_nowait({"hello": "sync"}) is True
    assert child.poll(1.0)
    assert child.recv() == {"hello": "sync"}
