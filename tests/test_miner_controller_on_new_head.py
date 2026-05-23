"""Verify SubstrateMinerController exposes an on_new_head async callback.

The callback receives a `SubstrateMiningContext` (what `get_mining_snapshot`
returns through the event manager) and (in this narrow-scope commit)
performs a subset of what legacy `_handle_head` does: push
`live_threshold_milli` to each handle, dispatch fresh work on work-key
change, short-circuit on same work key.

Cancel-on-key-change and other legacy behaviors are deferred to later
tasks before `_handle_head` is deleted.
"""
from __future__ import annotations

from types import SimpleNamespace

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
    return ctrl, handle


def _make_context(
    *,
    threshold_milli: int,
    last_proof_block_hash: bytes = b"\x00" * 32,
    topology_hash: bytes = b"\xab" * 32,
) -> SimpleNamespace:
    """Build a SubstrateMiningContext-shaped object with attribute access.

    Tests use SimpleNamespace rather than the real dataclass to avoid the
    __post_init__ validation (32-byte length checks on hash fields, etc.)
    that's unrelated to the behavior under test.
    """
    return SimpleNamespace(
        last_proof_block_hash=last_proof_block_hash,
        topology_hash=topology_hash,
        difficulty=SimpleNamespace(max_energy_milli=threshold_milli),
    )


@pytest.mark.asyncio
async def test_on_new_head_pushes_threshold_change(controller):
    """A context with a new threshold value triggers set_live_threshold_milli on each handle."""
    ctrl, handle = controller
    ctx = _make_context(threshold_milli=-5000)
    await ctrl.on_new_head(ctx)
    assert handle.threshold_pushes == [-5000]


@pytest.mark.asyncio
async def test_on_new_head_does_not_push_same_threshold(controller):
    """If threshold is unchanged, no push (matches existing controller behaviour)."""
    ctrl, handle = controller
    ctrl._last_pushed_threshold_milli = -5000
    ctx = _make_context(threshold_milli=-5000)
    await ctrl.on_new_head(ctx)
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_on_new_head_dispatches_on_work_key_change(controller):
    """A context with a new (last_proof_block_hash, topology_hash) dispatches work."""
    ctrl, handle = controller
    ctx = _make_context(
        threshold_milli=-5000,
        last_proof_block_hash=b"\x01" * 32,
    )
    await ctrl.on_new_head(ctx)
    assert len(handle.dispatched_contexts) == 1


@pytest.mark.asyncio
async def test_on_new_head_skips_dispatch_on_same_work_key(controller):
    """A context with the same work key as currently mining does not redispatch."""
    ctrl, handle = controller
    ctrl._current_work_key = (b"\x01" * 32, b"\xab" * 32)
    ctx = _make_context(
        threshold_milli=-5000,
        last_proof_block_hash=b"\x01" * 32,
        topology_hash=b"\xab" * 32,
    )
    handle._active_dispatch_id = 1  # currently mining
    await ctrl.on_new_head(ctx)
    assert handle.dispatched_contexts == []  # no new dispatch


@pytest.mark.asyncio
async def test_on_new_head_skips_dispatch_on_closed_work_key(controller):
    """If we've already won this round, on_new_head must NOT redispatch.

    Both legacy `_handle_head` and the new `on_new_head` run concurrently
    during the migration. The legacy path adds the work key to
    `_closed_work_keys` on a successful submission; `on_new_head` would
    otherwise still try to dispatch when its next poll fires, producing
    a proof the chain would reject (stale round).
    """
    ctrl, handle = controller
    work_key = (b"\x01" * 32, b"\xab" * 32)
    # Mark the key as already-won — value can be anything; the check is
    # membership-only.
    ctrl._closed_work_keys[work_key] = object()
    ctx = _make_context(
        threshold_milli=-5000,
        last_proof_block_hash=work_key[0],
        topology_hash=work_key[1],
    )
    await ctrl.on_new_head(ctx)
    assert handle.dispatched_contexts == []
