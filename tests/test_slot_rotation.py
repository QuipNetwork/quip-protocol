# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Host-side slot bookkeeping for the CUDA self-feeding loop (gh-19 / QUI-828).

These run on any machine (no cupy): the throughput-decay bug lived entirely in
the pure rotation bookkeeping. The old rotation advanced ACTIVE into an empty
NEXT slot whenever the feeder was momentarily empty at a completion, dropping
the nonce forever and idling its GPU block — attempt rate fell block-by-block
over a long run until a restart. ``SlotState.rotate_on_completion`` now goes
idle (kept alive) instead, and free slots are derived so none can leak.
"""
from __future__ import annotations

from GPU.slot_rotation import SlotState


def test_free_slot_is_derived_and_conserved():
    ss = SlotState(slots_per_nonce=3)
    assert ss.free_slot() == 0
    ss.assign_active(0, "A")
    assert ss.free_slot() == 1
    ss.assign_next(1, "B")
    assert ss.free_slot() == 2  # third slot still free
    # All slots in {0,1,2}; active and next never collide.
    assert ss.active_slot != ss.next_slot


def test_rotate_with_next_promotes_it_to_active():
    ss = SlotState()
    ss.assign_active(0, "A")
    ss.assign_next(1, "B")
    ss.rotate_on_completion()
    assert ss.active_slot == 1 and ss.active_model == "B"
    assert ss.next_slot == -1 and ss.next_model is None
    assert not ss.is_idle


def test_rotate_without_next_goes_idle_not_dropped():
    """The regression: a completion with an empty feeder (no NEXT queued) must
    leave the nonce IDLE and revivable, never advancing ACTIVE into slot -1."""
    ss = SlotState()
    ss.assign_active(0, "A")  # feeder was empty, so no next was queued
    ss.rotate_on_completion()
    # Old code: active_slot=-1, active_model=None AND the nonce was skipped
    # forever. Now it is idle but fully revivable.
    assert ss.is_idle
    assert ss.active_slot == -1
    # Revive: a free slot exists and re-activates the nonce.
    slot = ss.free_slot()
    assert 0 <= slot < ss.slots_per_nonce
    ss.assign_active(slot, "B")
    assert not ss.is_idle and ss.active_model == "B"


def test_needs_next_and_is_idle_flags():
    ss = SlotState()
    assert ss.is_idle and not ss.needs_next()
    ss.assign_active(0, "A")
    assert not ss.is_idle and ss.needs_next()
    ss.assign_next(1, "B")
    assert not ss.needs_next()


def test_slots_never_leak_across_repeated_starvation():
    """Alternating starved/fed completions must never lose or double-book a
    slot — the old ``free_slot`` int was overwritten on a starved rotation,
    leaking a physical slot on the second consecutive starvation."""
    ss = SlotState(slots_per_nonce=3)
    ss.assign_active(0, "seed")
    for i in range(200):
        ss.rotate_on_completion()
        starved = i % 3 == 0
        if not starved:
            slot = ss.free_slot()
            assert slot >= 0
            # Fill active if idle, else next — mirrors the loop's _try_queue.
            if ss.is_idle:
                ss.assign_active(slot, f"m{i}")
            else:
                ss.assign_next(slot, f"m{i}")
        # Invariant every step: in-use slots are distinct and in range; a free
        # slot is always available unless both roles are filled.
        used = [s for s in (ss.active_slot, ss.next_slot) if s >= 0]
        assert len(used) == len(set(used)), f"duplicate slot at i={i}: {used}"
        assert all(0 <= s < ss.slots_per_nonce for s in used)
        if len(used) < ss.slots_per_nonce:
            assert ss.free_slot() >= 0


def _simulate_loop(num_nonces: int, ticks: int, feed, *, revive_idle: bool) -> list[int]:
    """Replay the loop's host bookkeeping (refill then complete-all).

    ``feed`` is a zero-arg callable returning a model or None (an intermittently
    empty feeder). ``revive_idle`` toggles the fix: when True, an idle nonce is
    re-activated once the feeder recovers (current loop); when False, idle
    nonces are never refilled — the legacy behaviour, where the loop skipped
    ``active_model is None`` and a starved nonce was dropped for good. Returns
    the per-tick count of ACTIVE (non-idle) nonces.
    """
    slots = [SlotState() for _ in range(num_nonces)]

    def try_queue(ss: SlotState) -> None:
        slot = ss.free_slot()
        if slot < 0:
            return
        m = feed()
        if m is None:
            return
        if ss.is_idle:
            ss.assign_active(slot, m)
        else:
            ss.assign_next(slot, m)

    # Cold start: active then next (blocking in the real loop; here the feeder
    # is generous enough at the start that both fill).
    for ss in slots:
        try_queue(ss)
    for ss in slots:
        if not ss.is_idle:
            try_queue(ss)

    active_counts: list[int] = []
    for _ in range(ticks):
        for ss in slots:
            if ss.needs_next() or (revive_idle and ss.is_idle):
                try_queue(ss)
        active_counts.append(sum(not ss.is_idle for ss in slots))
        for ss in slots:
            if not ss.is_idle:
                ss.rotate_on_completion()
    return active_counts


def _feed_empty_30pct():
    seq = iter(range(10_000_000))

    def feed():
        i = next(seq)
        return None if i % 10 < 3 else f"m{i}"  # empty ~30% of pulls

    return feed


def test_fix_holds_active_nonces_steady_under_starvation():
    """With the revive-idle rotation, a feeder empty ~30% of the time keeps the
    active-nonce count in a healthy stationary band — no downward drift."""
    counts = _simulate_loop(8, 400, _feed_empty_30pct(), revive_idle=True)
    first_q = counts[50:150]
    last_q = counts[-100:]
    assert min(last_q) >= 4, f"active nonces dipped too low: min={min(last_q)}"
    # Stationary: the tail mean is not materially below the early mean.
    assert sum(last_q) / len(last_q) >= sum(first_q) / len(first_q) - 0.5


def test_legacy_drop_behaviour_collapses_to_zero():
    """Contrast: without reviving idle nonces (the pre-fix loop), the same
    intermittent starvation drives the active-nonce count to zero — this is the
    gh-19 decay the fix removes."""
    counts = _simulate_loop(8, 400, _feed_empty_30pct(), revive_idle=False)
    assert counts[0] > counts[-1], "expected monotone-ish decay"
    assert counts[-1] == 0, f"legacy behaviour should collapse, got tail {counts[-5:]}"


def test_permanent_exhaustion_drains_all_nonces():
    """A genuinely dead feeder (always None) drains every nonce to idle — the
    finite/shutdown case must still terminate, not spin forever."""
    counts = _simulate_loop(4, 10, lambda: None, revive_idle=True)
    assert counts[-1] == 0
