"""Tests for substrate.event_manager.ChainEventManager.

The event manager polls the pool, dedups on state change, dispatches
typed events to async subscribers, and tells the pool to force_swap
when no state change has been observed within blocktime thresholds.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from substrate.event_manager import ChainEventManager


class _FakePool:
    """In-process stand-in for ValidatorPool. Each `get_mining_snapshot`
    call returns the next snapshot scripted by the test."""

    def __init__(self) -> None:
        self._snapshots: list[Any] = []
        self.force_swap_calls = 0
        self._last: Any = None

    def script(self, *snapshots: Any) -> None:
        self._snapshots.extend(snapshots)

    async def send(self, op: str, args: dict) -> Any:
        if op == "get_mining_snapshot":
            if not self._snapshots:
                # idle — return the last snapshot indefinitely
                await asyncio.sleep(0.001)
                return self._last
            self._last = self._snapshots.pop(0)
            return self._last
        raise NotImplementedError(op)

    async def force_swap(self) -> None:
        self.force_swap_calls += 1


def _snap(head_number: int, threshold_milli: int = -1000) -> dict:
    """Build a minimal mining_snapshot-shaped dict for tests."""
    return {
        "head_number": head_number,
        "last_proof_block_hash": b"\x00" * 32,
        "difficulty_milli": threshold_milli,
    }


def _state_key(snapshot: dict) -> tuple:
    return (
        snapshot["last_proof_block_hash"],
        snapshot["head_number"],
        snapshot["difficulty_milli"],
    )


@pytest.mark.asyncio
async def test_event_manager_emits_on_state_change():
    """A snapshot that differs from the previous emits 'new_head'."""
    pool = _FakePool()
    pool.script(_snap(10), _snap(11))
    received: list[dict] = []
    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,  # fast for tests
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=100.0,  # don't fire stale during this test
        dead_blocktime_multiplier=100.0,
    )
    em.subscribe("new_head", lambda payload: received.append(payload))
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.1)
    em.request_shutdown()
    await task
    # First snapshot is "new" → emitted. Second is different → emitted.
    head_numbers = [r["head_number"] for r in received]
    assert head_numbers == [10, 11]


@pytest.mark.asyncio
async def test_event_manager_dedups_identical_snapshots():
    """Two identical snapshots only produce one event."""
    pool = _FakePool()
    pool.script(_snap(10), _snap(10), _snap(10), _snap(11))
    received: list[dict] = []
    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=100.0,
        dead_blocktime_multiplier=100.0,
    )
    em.subscribe("new_head", lambda p: received.append(p))
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.1)
    em.request_shutdown()
    await task
    head_numbers = [r["head_number"] for r in received]
    assert head_numbers == [10, 11]  # repeats deduped


@pytest.mark.asyncio
async def test_event_manager_calls_force_swap_when_dead_threshold_exceeded():
    """If no state change is observed for dead_threshold_s, force_swap is invoked."""
    pool = _FakePool()
    pool.script(_snap(10))  # only one snapshot, then it returns the same forever
    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=1.0,
        dead_blocktime_multiplier=3.0,
    )
    em.subscribe("new_head", lambda p: None)
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.15)  # well beyond 3 × 0.01s
    em.request_shutdown()
    await task
    assert pool.force_swap_calls >= 1


@pytest.mark.asyncio
async def test_event_manager_per_callback_exception_isolation():
    """A buggy subscriber doesn't kill the dispatch loop or affect other subscribers."""
    pool = _FakePool()
    pool.script(_snap(10), _snap(11))

    received_a: list[dict] = []
    received_c: list[dict] = []

    def good_a(p):
        received_a.append(p)

    def bad_b(p):
        raise RuntimeError("subscriber bug")

    def good_c(p):
        received_c.append(p)

    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=100.0,
        dead_blocktime_multiplier=100.0,
    )
    em.subscribe("new_head", good_a)
    em.subscribe("new_head", bad_b)
    em.subscribe("new_head", good_c)
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.1)
    em.request_shutdown()
    await task
    # Both good_a and good_c receive both events; bad_b's exception is logged.
    assert [p["head_number"] for p in received_a] == [10, 11]
    assert [p["head_number"] for p in received_c] == [10, 11]


@pytest.mark.asyncio
async def test_event_manager_supports_async_callbacks():
    """async def callbacks are awaited correctly."""
    pool = _FakePool()
    pool.script(_snap(10))
    received: list[dict] = []

    async def async_callback(p):
        await asyncio.sleep(0.001)
        received.append(p)

    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=100.0,
        dead_blocktime_multiplier=100.0,
    )
    em.subscribe("new_head", async_callback)
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.05)
    em.request_shutdown()
    await task
    assert len(received) == 1
    assert received[0]["head_number"] == 10


@pytest.mark.asyncio
async def test_event_manager_force_swap_timeout_does_not_disable_watchdog():
    """If pool.force_swap() hangs, the watchdog still re-arms via timeout."""

    class _HangingSwapPool(_FakePool):
        async def force_swap(self) -> None:
            self.force_swap_calls += 1
            # hang forever — but the event manager must time us out
            await asyncio.sleep(3600)

    pool = _HangingSwapPool()
    pool.script(_snap(10))  # only one snapshot, then idle
    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=1.0,
        dead_blocktime_multiplier=3.0,  # 3 × 0.01s = 0.03s dead threshold
    )
    em.subscribe("new_head", lambda p: None)
    task = asyncio.create_task(em.run())
    # Let the watchdog fire multiple times — if force_swap hung the watchdog
    # would only count once. With timeout-re-arming, expect multiple swap attempts.
    await asyncio.sleep(0.25)
    em.request_shutdown()
    await task
    # Without the timeout, force_swap_calls would be 1 (stuck). With timeout
    # re-arming, we expect ≥ 2 attempts within 0.25s / (0.03s dead threshold).
    assert pool.force_swap_calls >= 2
