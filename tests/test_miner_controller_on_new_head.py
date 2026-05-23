"""Verify SubstrateMinerController exposes an on_new_head async callback.

The callback receives a snapshot dict (as the event manager will deliver)
and performs the work today's _handle_head does: cancel handles on
work-key change, push live_threshold_milli to each handle, dispatch
fresh work on key change.
"""
from __future__ import annotations

import asyncio

import pytest

from substrate.miner_controller import SubstrateMinerController


class _FakeMinerHandle:
    def __init__(self, miner_id: str) -> None:
        self.miner_id = miner_id
        self.threshold_pushes: list[int] = []
        self.dispatched_contexts: list = []
        self.cancel_calls = 0
        self._active_dispatch_id = 0

    def set_live_threshold_milli(self, value: int) -> None:
        self.threshold_pushes.append(value)

    def mine_work_item(self, context) -> int:
        self.dispatched_contexts.append(context)
        self._active_dispatch_id += 1
        return self._active_dispatch_id

    def cancel(self) -> None:
        self.cancel_calls += 1


@pytest.fixture
def controller():
    """A controller wired with fakes only, no real pool/event manager."""
    handle = _FakeMinerHandle("miner-1")
    ctrl = SubstrateMinerController.__new__(SubstrateMinerController)
    ctrl.miner_handles = [handle]
    ctrl._current_work_key = None
    ctrl._current_context = None
    ctrl._last_pushed_threshold_milli = 0
    ctrl._closed_work_keys = {}
    ctrl._highest_handled_block = 0
    # Add anything else `on_new_head` reads from controller state.
    return ctrl, handle


@pytest.mark.asyncio
async def test_on_new_head_pushes_threshold_change(controller):
    """A snapshot with a new threshold value triggers set_live_threshold_milli on each handle."""
    ctrl, handle = controller
    snapshot = _make_snapshot(threshold_milli=-5000, head_number=10)
    await ctrl.on_new_head(snapshot)
    assert handle.threshold_pushes == [-5000]


@pytest.mark.asyncio
async def test_on_new_head_does_not_push_same_threshold(controller):
    """If threshold is unchanged, no push (matches existing controller behaviour)."""
    ctrl, handle = controller
    ctrl._last_pushed_threshold_milli = -5000
    snapshot = _make_snapshot(threshold_milli=-5000, head_number=10)
    await ctrl.on_new_head(snapshot)
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_on_new_head_dispatches_on_work_key_change(controller):
    """A snapshot with a new (last_proof_block_hash, topology_hash) dispatches work."""
    ctrl, handle = controller
    snapshot = _make_snapshot(
        threshold_milli=-5000,
        head_number=10,
        last_proof_block_hash=b"\x01" * 32,
    )
    await ctrl.on_new_head(snapshot)
    assert len(handle.dispatched_contexts) == 1


@pytest.mark.asyncio
async def test_on_new_head_skips_dispatch_on_same_work_key(controller):
    """A snapshot with the same work key as currently mining does not redispatch."""
    ctrl, handle = controller
    ctrl._current_work_key = (b"\x01" * 32, b"\xab" * 32)
    snapshot = _make_snapshot(
        threshold_milli=-5000,
        head_number=10,
        last_proof_block_hash=b"\x01" * 32,
        topology_hash=b"\xab" * 32,
    )
    handle._active_dispatch_id = 1  # currently mining
    await ctrl.on_new_head(snapshot)
    assert handle.dispatched_contexts == []  # no new dispatch


def _make_snapshot(
    *,
    threshold_milli: int,
    head_number: int,
    last_proof_block_hash: bytes = b"\x00" * 32,
    topology_hash: bytes = b"\xab" * 32,
) -> dict:
    """Construct a snapshot payload matching what get_mining_snapshot returns."""
    return {
        "head_number": head_number,
        "last_proof_block_hash": last_proof_block_hash,
        "topology_hash": topology_hash,
        "difficulty_milli": threshold_milli,
        # … other fields the controller reads
    }
