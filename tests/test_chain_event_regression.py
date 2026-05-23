"""Regression test for the original silent-head-subscription-death bug.

The pre-Plan-3 controller subscribed to new heads via substrate-interface.
That subscription could die silently (WebSocket reader thread stops
delivering notifications without raising). The controller would then
ride a stale ``last_proof_block_hash`` indefinitely: it never pushed
fresh ``live_threshold_milli`` to the workers as the chain's difficulty
decayed, and miners would idle on a threshold they couldn't satisfy.

The fix is the ChainEventManager: it polls the validator pool at
adaptive cadence and fires ``new_head`` whenever the snapshot's
``(last_proof_block_hash, max_energy_milli)`` changes. The watchdog
force-swaps the pool if no change is observed for
``dead_blocktime_multiplier × blocktime_s`` — but the more common case
is that the chain IS producing decays and we just need to surface them
quickly.

This test reproduces the pre-fix scenario: a chain that produces 100
decay steps with no proofs, monotonically shifting ``max_energy_milli``.
Without the fix, the controller would push the initial threshold once
and then never update. With the fix, ``set_live_threshold_milli`` is
called on every observed decay step.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import asyncio
import pytest

from substrate.event_manager import ChainEventManager


def _ctx(threshold_milli: int, last_proof_block_hash: bytes = b"\x00" * 32):
    return SimpleNamespace(
        last_proof_block_hash=last_proof_block_hash,
        topology_hash=b"\xab" * 32,
        difficulty=SimpleNamespace(max_energy_milli=threshold_milli),
    )


def _state_key(snapshot):
    return (
        snapshot.last_proof_block_hash,
        int(snapshot.difficulty.max_energy_milli),
    )


class _DecayPool:
    """Fake pool that scripts a sequence of decayed snapshots."""

    def __init__(self, snapshots: list[Any]) -> None:
        self._snapshots = list(snapshots)
        self._last = snapshots[0]
        self.force_swap_calls = 0

    async def send(self, op: str, args: dict) -> Any:
        assert op == "get_mining_snapshot"
        if self._snapshots:
            self._last = self._snapshots.pop(0)
        return self._last

    async def force_swap(self) -> None:
        self.force_swap_calls += 1


@pytest.mark.asyncio
async def test_decay_steps_drive_threshold_changes_through_event_manager():
    """100 monotonic decay steps must produce 100 threshold-change events.

    Without the original bug fix, the head subscription could die silently
    and the controller would push only the initial threshold. The event
    manager polls and dedups by state key — distinct
    ``max_energy_milli`` values are distinct keys, so each is observed.
    """
    # Initial threshold of -14842152 (matches the bug repro from the
    # original incident), decaying by ~21000 milli per "step". Stop at
    # ~-14630000 (the chain-side value observed when the bug surfaced).
    decay_steps = list(range(-14842152, -14630000, 2118))[:100]
    assert len(decay_steps) == 100
    pool = _DecayPool([_ctx(t) for t in decay_steps])

    received_thresholds: list[int] = []

    def record(ctx):
        received_thresholds.append(int(ctx.difficulty.max_energy_milli))

    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,  # fast for tests
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        # Disable watchdog for this test — we want a pure poll-decay run.
        stale_blocktime_multiplier=1000.0,
        dead_blocktime_multiplier=1000.0,
    )
    em.subscribe("new_head", record)
    task = asyncio.create_task(em.run())
    # Generous timeout so all 100 polls happen even on a busy CI.
    await asyncio.sleep(1.0)
    em.request_shutdown()
    await task

    # We must have seen every distinct decay value (state-key change → event).
    assert received_thresholds == decay_steps, (
        f"expected {len(decay_steps)} distinct threshold events; "
        f"got {len(received_thresholds)}"
    )


@pytest.mark.asyncio
async def test_no_state_change_for_3_blocktimes_triggers_force_swap():
    """The load-bearing watchdog: silent chain → force_swap.

    This is the failure mode the bug exhibited. The original
    substrate-interface subscription would freeze, so the controller
    never saw a head and never even tried to swap to a healthy
    validator. The new design's watchdog fires force_swap if the
    snapshot state hasn't changed for ``dead_blocktime_multiplier ×
    blocktime_s``.
    """
    # A pool that hands out the same snapshot forever.
    pool = _DecayPool([_ctx(threshold_milli=-1000)])
    em = ChainEventManager(
        pool=pool,
        state_key=_state_key,
        snapshot_op="get_mining_snapshot",
        snapshot_args={},
        blocktime_s=0.01,
        settled_poll_pct=0.5,
        catch_up_poll_pct=0.1,
        stale_blocktime_multiplier=1.0,
        dead_blocktime_multiplier=3.0,  # 3 × 0.01s = 30ms dead threshold
    )
    em.subscribe("new_head", lambda c: None)
    task = asyncio.create_task(em.run())
    await asyncio.sleep(0.2)  # well past the 30ms dead threshold
    em.request_shutdown()
    await task
    assert pool.force_swap_calls >= 1, (
        "watchdog did not call force_swap despite no state change "
        "for >> dead threshold"
    )
