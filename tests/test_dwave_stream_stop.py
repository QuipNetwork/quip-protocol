"""Unit tests for DWaveMiner budget-gate helpers.

The legacy DWaveMiner.sample_ising_streaming (diagnostic path) was removed
when QPU switched to the generic StreamContext + DWaveSamplerWrapper.
Stop/cancel behaviour for the QPU streaming pump is covered by
tests/test_dwave_streaming_sampler.py.
"""
from __future__ import annotations

import logging

from QPU.dwave_miner import DWaveMiner, _PacingRateLimiter


class _FakeTimeManager:
    """Read-only get_stats() stand-in driving the in-loop budget decision."""

    def __init__(self, should_mine):
        # pool > 0 → mine; pool == 0 → drained.
        pool = 5.0 if should_mine else 0.0
        used = 35.0 if should_mine else 45.0
        self.end_burst_calls = 0
        self._stats = {
            "daily_budget_seconds": 1800.0,
            "cumulative_used_seconds": used,
            "pool_seconds": pool,
            "budget_remaining_seconds": pool,
            "pool_cap_seconds": 1800.0,
            "min_block_budget_seconds": 90.0,
            "burst_active": should_mine,
            "seconds_until_buffer": 0.0 if should_mine else 4080.0,
            "blocks_mined": 7,
            "blocks_skipped": 3,
        }

    def get_stats(self):
        return dict(self._stats)

    def end_burst(self):
        self.end_burst_calls += 1


def _dwave_with_time_manager(tm) -> DWaveMiner:
    miner = object.__new__(DWaveMiner)
    miner.time_manager = tm
    miner._inloop_pacing_rl = _PacingRateLimiter()
    miner.logger = logging.getLogger("miner.test-midstream-budget")
    return miner


def test_midstream_budget_ok_returns_none_without_time_manager():
    miner = _dwave_with_time_manager(None)
    assert miner._midstream_budget_ok(solution_number=1) is None


def test_midstream_budget_ok_passes_when_budget_available():
    miner = _dwave_with_time_manager(_FakeTimeManager(should_mine=True))
    status = miner._midstream_budget_ok(solution_number=1)
    assert status is not None
    assert status.should_mine is True
    # Stats carry the live (get_stats) snapshot for the log + telemetry push.
    assert status.stats["budget_remaining_seconds"] == 5.0
    assert status.stats["pool_seconds"] == 5.0
    assert status.stats["blocks_skipped"] == 3
    assert status.stats["daily_budget_seconds"] == 1800.0


def test_midstream_budget_ok_stalls_and_logs_when_exhausted(caplog):
    tm = _FakeTimeManager(should_mine=False)
    miner = _dwave_with_time_manager(tm)
    with caplog.at_level(logging.WARNING):
        status = miner._midstream_budget_ok(solution_number=1)
    assert status is not None
    assert status.should_mine is False
    # Drain must end the burst so the pool re-accumulates the full buffer.
    assert tm.end_burst_calls == 1
    assert any(
        "burst drained" in r.message and "re-accumulating" in r.message
        for r in caplog.records
    ), "drain did not emit the burst-drained WARNING"
