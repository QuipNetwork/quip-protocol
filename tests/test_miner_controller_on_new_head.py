"""Verify SubstrateMinerController exposes an on_new_head async callback.

The callback receives a `SubstrateMiningContext` (what `get_mining_snapshot`
returns through the event manager) and runs the full guard chain:
push ``live_threshold_milli`` to each handle, dispatch fresh work on
work-key change, short-circuit on same work key, drop on zero-seed,
fail loud on topology mismatch.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from substrate.miner_controller import SubstrateMinerController
from substrate.work_scheduler import WorkScheduler


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
    # T7: all dispatch goes through the WorkScheduler. A real (unstarted)
    # scheduler over the fake handle exercises the same dispatch_pow path
    # production uses; no drainer tasks are needed for these guards.
    ctrl._scheduler = WorkScheduler([handle])
    ctrl._current_work_key = None
    ctrl._last_logged_work_key = None
    ctrl._current_context = None
    ctrl._last_pushed_threshold_milli = 0
    ctrl._closed_work_keys = {}
    ctrl._highest_handled_block = 0
    ctrl._dispatch_contexts = {}
    ctrl.topology_hash = None
    ctrl.rebind_requested = False
    ctrl._shutdown_event = asyncio.Event()
    ctrl.core = None
    # Anticipatory-submission state. on_new_head no longer fires (the
    # free-running cadence timer owns that); it only stores previews and a
    # timing anchor. These tests never store a preview, so the active-key
    # state stays empty. The pool_client stub returns None from every
    # predictor query so dispatch's schedule build degrades cleanly.
    ctrl._latest_preview = {}
    ctrl._anticipatory_fired = set()
    ctrl._base_difficulty_by_key = {}
    ctrl._decay_schedule_by_key = {}
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
async def test_on_new_head_does_not_push_same_threshold(controller):
    """If threshold is unchanged, no push (matches existing controller behaviour)."""
    ctrl, handle = controller
    ctrl._last_pushed_threshold_milli = -5000
    ctx = _make_context(threshold_milli=-5000)
    await ctrl.on_new_head(ctx)
    assert handle.threshold_pushes == []


# ----------------------------------------------------------------------
# Local decay pushes (_maybe_push_local_decay) — the staleness fallback
# that keeps the live threshold easing while the validator pool is dark.
# ----------------------------------------------------------------------

_KEY = ("proof-hash", "topo-hash")
# max_energy_milli per decay step — monotonically easing (toward 0).
_SCHEDULE = [-10000, -9500, -9000, -8500]


def _seed_local_decay(
    ctrl,
    *,
    poll_stale_s: float = 100.0,
    est_block: int | None = 75,
    last_proof_block: int = 50,
    epoch_length: int = 10,
    last_pushed: int = -11000,
    schedule: list[int] | None = None,
) -> None:
    """Wire the minimum state ``_maybe_push_local_decay`` reads.

    Defaults put the round at step (75-50)//10 = 2 → schedule[2] = -9000,
    easing from a chain-pushed baseline of -11000.
    """
    now = time.monotonic()
    ctrl.events = SimpleNamespace(
        last_successful_poll_monotonic=now - poll_stale_s,
    )
    ctrl._timing = SimpleNamespace(
        estimate_block=lambda *, now_monotonic: est_block,
    )
    ctrl._current_work_key = _KEY
    ctrl._decay_schedule_by_key[_KEY] = (
        list(_SCHEDULE) if schedule is None else schedule,
        last_proof_block,
        epoch_length,
    )
    ctrl._last_pushed_threshold_milli = last_pushed
    ctrl._decay_horizon_logged_key = None


@pytest.mark.asyncio
async def test_local_decay_pushes_when_polls_stale(controller):
    ctrl, handle = controller
    _seed_local_decay(ctrl)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == [-9000]
    assert ctrl._last_pushed_threshold_milli == -9000
    # Same tick state again → no duplicate push.
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == [-9000]


@pytest.mark.asyncio
async def test_local_decay_never_regresses_tighter(controller):
    """A schedule step tighter than the last push must never be pushed —
    max() on negative milli keeps the looser value."""
    ctrl, handle = controller
    _seed_local_decay(ctrl, last_pushed=-8000)  # looser than schedule[2]=-9000
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []
    assert ctrl._last_pushed_threshold_milli == -8000


@pytest.mark.asyncio
async def test_local_decay_skips_when_polls_fresh(controller):
    ctrl, handle = controller
    _seed_local_decay(ctrl, poll_stale_s=1.0)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_local_decay_skips_without_schedule(controller):
    """A schedule-less round (dispatch-time RPC failure) never extrapolates."""
    ctrl, handle = controller
    _seed_local_decay(ctrl)
    del ctrl._decay_schedule_by_key[_KEY]
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_local_decay_skips_without_timing_anchor(controller):
    """estimate_block is None until a head has been observed — no basis."""
    ctrl, handle = controller
    _seed_local_decay(ctrl, est_block=None)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_local_decay_skips_at_genesis_sentinel(controller):
    ctrl, handle = controller
    _seed_local_decay(ctrl, last_proof_block=0)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_local_decay_skips_without_chain_baseline(controller):
    """_last_pushed_threshold_milli == 0 means no chain value was ever
    pushed — there is nothing trustworthy to ease from."""
    ctrl, handle = controller
    _seed_local_decay(ctrl, last_pushed=0)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == []


@pytest.mark.asyncio
async def test_local_decay_clamps_at_horizon(controller):
    """Past the schedule horizon, hold the last (loosest) value."""
    ctrl, handle = controller
    _seed_local_decay(ctrl, est_block=50_000)
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == [_SCHEDULE[-1]]


@pytest.mark.asyncio
async def test_chain_push_resumes_after_local_decay(controller):
    """After local decay, an equal chain value is a no-op and a tighter
    chain value (new round) pushes unclamped — the chain stays authoritative."""
    ctrl, handle = controller
    _seed_local_decay(ctrl)
    # Dispatch mechanics are not under test — a real scheduler would block
    # preempting the fake handle (it never acks cancels).
    ctrl._scheduler = SimpleNamespace(dispatch_pow=AsyncMock(return_value={}))
    ctrl._maybe_push_local_decay()
    assert handle.threshold_pushes == [-9000]

    # Poll recovers with the same value the local model reached: no dup.
    ctx_same = _make_context(threshold_milli=-9000)
    await ctrl.on_new_head(ctx_same)
    assert handle.threshold_pushes == [-9000]

    # New round ratchets tighter: chain-sourced push is unclamped.
    ctx_tighter = _make_context(
        threshold_milli=-12000, last_proof_block_hash=b"\x02" * 32,
    )
    await ctrl.on_new_head(ctx_tighter)
    assert handle.threshold_pushes == [-9000, -12000]


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
async def test_on_new_head_topology_change_requests_rebind(controller):
    """Bound topology_hash != snapshot.topology_hash → request rebind + shut
    down gracefully (the supervisor rebuilds against the new chain topology),
    instead of crashing the process."""
    ctrl, handle = controller
    ctrl.topology_hash = b"\xab" * 32  # bound at startup
    ctx = _make_context(
        threshold_milli=-5000,
        topology_hash=b"\xcd" * 32,  # chain changed its DefaultTopology
    )
    assert ctrl.rebind_requested is False
    await ctrl.on_new_head(ctx)  # must not raise
    assert ctrl.rebind_requested is True
    assert ctrl._shutdown_event.is_set()


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


@pytest.mark.asyncio
async def test_on_new_head_banner_logs_info_once_then_debug(controller, monkeypatch):
    """The new-head banner logs INFO on a new work key and DEBUG when the
    same work item is re-dispatched (worker went idle between heads)."""
    import substrate.miner_controller as mc
    from unittest.mock import MagicMock

    ctrl, handle = controller
    fake = MagicMock()
    monkeypatch.setattr(mc, "logger", fake)

    ctx = _make_context(threshold_milli=-5000, last_proof_block_hash=b"\x01" * 32)
    await ctrl.on_new_head(ctx)            # new key → INFO
    handle._active_dispatch_id = 0          # worker idle → re-dispatch same key
    await ctrl.on_new_head(ctx)            # unchanged key → DEBUG

    def _banners(method):
        return [
            c for c in method.call_args_list
            if c.args and isinstance(c.args[0], str)
            and c.args[0].startswith("new head (event manager)")
        ]

    assert len(_banners(fake.info)) == 1, "new work key should log INFO once"
    assert len(_banners(fake.debug)) == 1, "re-dispatch of same work logs DEBUG"
