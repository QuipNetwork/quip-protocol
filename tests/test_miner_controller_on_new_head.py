"""Verify SubstrateMinerController exposes an on_new_head async callback.

The callback receives a `SubstrateMiningContext` (what `get_mining_snapshot`
returns through the event manager) and runs the full guard chain:
push ``live_threshold_milli`` to each handle, dispatch fresh work on
work-key change, short-circuit on same work key, drop on zero-seed,
fail loud on topology mismatch.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

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

    def mine_work_item(self, context, *, solution_number=None) -> int:
        self.dispatched_contexts.append(context)
        self.last_solution_number = solution_number
        self._active_dispatch_id += 1
        return self._active_dispatch_id

    def cancel(self) -> None:
        self.cancel_calls += 1


@pytest.fixture
def controller():
    """A controller wired with fakes only, no real pool/event manager.

    Sets the minimum set of attributes ``on_new_head`` reads. The
    full-fat fields (drainer tasks, queues, signer, etc.) are not touched
    because the tests don't exercise paths that need them.
    """
    handle = _FakeMinerHandle("miner-1")
    ctrl = SubstrateMinerController.__new__(SubstrateMinerController)
    ctrl.miner_handles = [handle]
    ctrl._current_work_key = None
    ctrl._current_context = None
    ctrl._last_pushed_threshold_milli = 0
    ctrl._closed_work_keys = {}
    ctrl._highest_handled_block = 0
    ctrl._dispatch_contexts = {}
    ctrl.topology_hash = None
    ctrl.core = None
    # Anticipatory-submission state (Task 6b). on_new_head reads
    # `_latest_preview` on every head; these tests never store a preview,
    # so `_maybe_anticipatory_fire` short-circuits at the empty-store check
    # and the fire path is a clean no-op. The pool_client stub returns None
    # from every predictor query so that even if a preview were present,
    # `_anticipatory_inputs` would return None (no fire) rather than hit
    # the network.
    ctrl._latest_preview = {}
    ctrl._anticipatory_fired = set()
    ctrl._base_difficulty_by_key = {}
    ctrl._pow_constants = None
    # Per-round solution-number cache (the on-disk archive key). The stub
    # returns a fixed WinningSolutions count so dispatch resolves a stable
    # solution number without hitting the network.
    ctrl._solution_number_by_work_key = {}
    ctrl.pool_client = SimpleNamespace(
        query_difficulty=AsyncMock(return_value=None),
        query_last_proof_block_number=AsyncMock(return_value=None),
        query_pow_constants=AsyncMock(return_value=None),
        query_winning_solution_count=AsyncMock(return_value=195),
    )
    # Stats: SimpleNamespace stub matching the attrs on_new_head touches.
    ctrl.stats = SimpleNamespace(
        heads_observed=0,
        none_snapshots_seen=0,
        zero_seed_snapshots_dropped=0,
        heads_same_key_skipped=0,
        contexts_dispatched=0,
    )
    return ctrl, handle


def _make_context(
    *,
    threshold_milli: int,
    last_proof_block_hash: bytes = b"\x01" * 32,  # non-zero default (avoids zero-seed guard)
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
        # on_new_head logs len(nodes) / len(edges) on dispatch; provide empty
        # sequences so the log line doesn't crash.
        nodes=(),
        edges=(),
    )


@pytest.mark.asyncio
async def test_on_new_head_pushes_threshold_change(controller):
    """A context with a new threshold value triggers set_live_threshold_milli on each handle."""
    ctrl, handle = controller
    ctx = _make_context(threshold_milli=-5000)
    await ctrl.on_new_head(ctx)
    assert handle.threshold_pushes == [-5000]


@pytest.mark.asyncio
async def test_maybe_anticipatory_fire_noop_without_preview(controller):
    """No preview stored → clean no-op (no exception, no chain reads)."""
    ctrl, _ = controller
    await ctrl._maybe_anticipatory_fire(
        _make_context(threshold_milli=-5000), (b"\x01" * 32, b"\xab" * 32)
    )
    # Never reached the predictor-input queries.
    ctrl.pool_client.query_difficulty.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_anticipatory_fire_noop_when_constants_none(controller):
    """A preview present but ``query_pow_constants`` → None must short-circuit
    cleanly (no AttributeError on ``constants.curve_c_*``), keeping the
    preview, and must NOT even reach ``query_difficulty``."""
    ctrl, _ = controller
    ctx = _make_context(threshold_milli=-5000)
    ctx.block_number = 5
    key = (b"\x01" * 32, b"\xab" * 32)
    ctrl._latest_preview[key] = {"submit_floor_energy": -3.0}
    await ctrl._maybe_anticipatory_fire(ctx, key)
    # constants None → return before querying difficulty.
    ctrl.pool_client.query_pow_constants.assert_awaited()
    ctrl.pool_client.query_difficulty.assert_not_called()
    assert key in ctrl._latest_preview


@pytest.mark.asyncio
async def test_maybe_anticipatory_fire_noop_when_difficulty_none(controller):
    """Constants present but ``query_difficulty`` → None must be a clean
    no-op rather than raise; preview is kept for a later head."""
    ctrl, _ = controller
    ctrl._pow_constants = SimpleNamespace(
        epoch_length=5,
        curve_c_easy_milli=800,
        curve_c_knee_milli=750,
        curve_c_hard_milli=700,
    )
    ctx = _make_context(threshold_milli=-5000)
    ctx.block_number = 5
    key = (b"\x01" * 32, b"\xab" * 32)
    ctrl._latest_preview[key] = {"submit_floor_energy": -3.0}
    await ctrl._maybe_anticipatory_fire(ctx, key)
    # difficulty None → no fire, preview kept.
    ctrl.pool_client.query_difficulty.assert_awaited()
    assert key in ctrl._latest_preview


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

    ``_handle_result`` adds the work key to ``_closed_work_keys`` on a
    successful submission; without this guard, the next event-manager
    poll would dispatch again, producing a proof the chain would reject
    (stale round).
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


# ---------------------------------------------------------------------------
# Guards on the event-manager path: None snapshot, topology mismatch,
# zero seed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_new_head_none_snapshot_is_no_op(controller):
    """None snapshot (no topology registered) bumps a stat and returns."""
    ctrl, handle = controller
    await ctrl.on_new_head(None)
    assert handle.threshold_pushes == []
    assert handle.dispatched_contexts == []
    assert ctrl.stats.none_snapshots_seen == 1


@pytest.mark.asyncio
async def test_on_new_head_topology_mismatch_fails_loud(controller):
    """Configured topology_hash != snapshot.topology_hash → _OperatorFailLoud."""
    from substrate.miner_controller import _OperatorFailLoud

    ctrl, handle = controller
    ctrl.topology_hash = b"\xab" * 32  # expected
    ctx = _make_context(
        threshold_milli=-5000,
        topology_hash=b"\xcd" * 32,  # snapshot mismatches
    )
    with pytest.raises(_OperatorFailLoud, match="does not match snapshot"):
        await ctrl.on_new_head(ctx)


@pytest.mark.asyncio
async def test_on_new_head_zero_seed_guard_blocks_dispatch(controller):
    """A zero last_proof_block_hash with handled_block > 0 must skip dispatch."""
    ctrl, handle = controller
    ctrl._highest_handled_block = 100  # not genesis any more
    ctx = _make_context(
        threshold_milli=-5000,
        last_proof_block_hash=b"\x00" * 32,  # the transient zero state
    )
    await ctrl.on_new_head(ctx)
    assert handle.dispatched_contexts == []
    assert ctrl.stats.zero_seed_snapshots_dropped == 1


@pytest.mark.asyncio
async def test_on_new_head_zero_seed_allowed_during_bootstrap(controller):
    """Zero last_proof_block_hash is OK at genesis (highest_handled_block == 0)."""
    ctrl, handle = controller
    # _highest_handled_block stays at 0 (default) — bootstrap state.
    ctx = _make_context(
        threshold_milli=-5000,
        last_proof_block_hash=b"\x00" * 32,
    )
    await ctrl.on_new_head(ctx)
    # Threshold was pushed (it's a normal dispatch path) AND mine_work_item ran.
    assert handle.threshold_pushes == [-5000]
    assert len(handle.dispatched_contexts) == 1
