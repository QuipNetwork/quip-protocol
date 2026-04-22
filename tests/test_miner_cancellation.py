"""Tests for MinerHandle.cancel() actually interrupting an active mine_block().

Regression coverage for the bug where cancel() pushed a "stop_mining" op onto
the worker's command queue, but the worker's command loop was blocked inside
miner.mine_block() and could not dequeue the stop signal until mining
finished on its own. That manifested as the node producing valid solutions
long after a peer had already won the block.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from shared.miner_service import MinerServiceHandle
from shared.miner_worker import MinerHandle


@dataclass
class _MockHeader:
    index: int = 0
    timestamp: int = 0


@dataclass
class _MockBlock:
    hash: bytes = b"\x00" * 32
    header: _MockHeader = field(default_factory=_MockHeader)


@dataclass
class _MockMinerInfo:
    miner_id: str = "test-miner"


@dataclass
class _MockRequirements:
    # SA on Z(9,2) reliably reaches -14000 with default params, so the
    # unsatisfiable lever is min_diversity: we ask for 99% average Hamming
    # distance across the selected solutions, which is geometrically
    # impossible for the Ising ground-state basin. That forces the miner
    # to loop until stop_event fires.
    difficulty_energy: float = -14000.0
    min_diversity: float = 0.99
    min_solutions: int = 10
    timeout_to_difficulty_adjustment_decay: int = 600
    h_values: tuple = (-1.0, 0.0, 1.0)


def _unreachable_job():
    return _MockBlock(), _MockMinerInfo(), _MockRequirements()


_CANCEL_RESPONSE_BUDGET_S = 5.0
"""Upper bound on how long cancel() may take to quiesce the worker.

One outer base_miner iteration (SA on Z(9,2)) dominates this budget; the
check exists to catch the old behaviour where cancel() was a no-op and the
worker ran until an unrelated timeout.
"""


@pytest.mark.timeout(20)
def test_miner_handle_cancel_interrupts_active_mining():
    """After cancel(), MinerHandle must become responsive within one iteration."""
    spec = {"id": "test-cancel-cpu", "kind": "cpu", "args": {}, "cfg": {}}
    handle = MinerHandle(spec)
    try:
        block, info, requirements = _unreachable_job()
        handle.mine(block, info, requirements, int(time.time()))

        # Let the worker pick up the mine_block op and enter the inner loop.
        time.sleep(1.0)

        t0 = time.time()
        handle.cancel()

        # get_stats() is queued on the same command queue as mine/cancel.
        # If cancel interrupted mining, the worker returns from mine_block,
        # loops back to cmd read, and answers the stats request promptly.
        # If cancel is a no-op, mining runs until its own completion
        # (never, at this difficulty) and get_stats() raises ValueError
        # from its 2s internal timeout.
        stats = None
        last_err = None
        while time.time() - t0 < _CANCEL_RESPONSE_BUDGET_S:
            try:
                stats = handle.get_stats()
                break
            except ValueError as exc:
                last_err = exc

        elapsed = time.time() - t0
        assert stats is not None, (
            f"cancel() did not interrupt mining within "
            f"{_CANCEL_RESPONSE_BUDGET_S}s (last error: {last_err})"
        )
        assert elapsed < _CANCEL_RESPONSE_BUDGET_S, (
            f"cancel() took {elapsed:.2f}s to quiesce the worker"
        )
    finally:
        handle.close()


@pytest.mark.timeout(20)
def test_miner_service_cancel_interrupts_active_mining():
    """Same guarantee for the long-lived MinerServiceHandle path."""
    spec = {"id": "test-cancel-svc", "kind": "cpu", "args": {}, "cfg": {}}
    handle = MinerServiceHandle(spec)
    try:
        block, info, requirements = _unreachable_job()
        # Wait for the service to finish building its miner before mining.
        time.sleep(1.0)
        handle.mine(block, info, requirements, int(time.time()))

        # Let mining get underway.
        time.sleep(1.0)

        t0 = time.time()
        handle.cancel()

        # After cancel, the service emits {"event": "stopped"}. Drain until
        # we see it, discarding any earlier MiningResult drops.
        saw_stopped = False
        while time.time() - t0 < _CANCEL_RESPONSE_BUDGET_S:
            try:
                msg = handle.result_queue.get(timeout=0.2)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("event") == "stopped":
                saw_stopped = True
                break

        elapsed = time.time() - t0
        assert saw_stopped, (
            f"cancel() did not produce a 'stopped' event within "
            f"{_CANCEL_RESPONSE_BUDGET_S}s (elapsed={elapsed:.2f}s)"
        )
    finally:
        handle.close()


@pytest.mark.timeout(20)
def test_cancel_on_idle_handle_is_safe():
    """cancel() must be safe to call when no mine_block is in flight.

    Node.mine_block's ``finally`` block now unconditionally cancels every
    handle, including paths where mining was never started this attempt.
    If cancel() ever raised on an idle worker, every shutdown and
    every "start mining failed validation" path would crash.
    """
    spec = {"id": "test-cancel-idle", "kind": "cpu", "args": {}, "cfg": {}}
    handle = MinerHandle(spec)
    try:
        # Let the worker enter its main cmd loop.
        time.sleep(0.5)

        handle.cancel()  # no mine in flight
        handle.cancel()  # double-cancel must be a no-op too

        # The worker must still be responsive to commands.
        stats = handle.get_stats()
        assert stats is not None, (
            "cancel() on idle handle left the worker unresponsive"
        )
    finally:
        handle.close()
