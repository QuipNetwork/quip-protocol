"""Unit tests for the QPU carry-over budget reservoir."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
import os
import importlib.util
import logging

import pytest

# Load qpu_time_manager module directly without triggering QPU/__init__.py
# This avoids the D-Wave dependency which may not be installed in test environments
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_module_path = os.path.join(_project_root, "QPU", "qpu_time_manager.py")
_spec = importlib.util.spec_from_file_location("qpu_time_manager", _module_path)
_module = importlib.util.module_from_spec(_spec)
# Register the module in sys.modules before execution (required for dataclass decorator)
sys.modules["qpu_time_manager"] = _module
_spec.loader.exec_module(_module)

parse_duration = _module.parse_duration
resolve_initial_budget = _module.resolve_initial_budget
QPUTimeConfig = _module.QPUTimeConfig
QPUTimeManager = _module.QPUTimeManager


def _tm(daily=86400.0, buffer=90.0, cap=None):
    """Reservoir manager with a pinned clock. daily=86400 => 1s pool/wall-s."""
    tm = QPUTimeManager(QPUTimeConfig(
        daily_budget_seconds=daily,
        min_block_budget_seconds=buffer,
        budget_cap_seconds=cap,
    ))
    tm.reset_clock(1_000_000.0)
    return tm


class TestParseDuration:
    """Tests for the parse_duration function."""

    def test_parse_seconds(self):
        assert parse_duration("30s") == 30.0
        assert parse_duration("1s") == 1.0
        assert parse_duration("0s") == 0.0
        assert parse_duration("3600s") == 3600.0

    def test_parse_minutes(self):
        assert parse_duration("5m") == 300.0
        assert parse_duration("1m") == 60.0
        assert parse_duration("20m") == 1200.0

    def test_parse_hours(self):
        assert parse_duration("1h") == 3600.0
        assert parse_duration("2h") == 7200.0
        assert parse_duration("0.5h") == 1800.0

    def test_parse_days(self):
        assert parse_duration("1d") == 86400.0
        assert parse_duration("2d") == 172800.0

    def test_parse_weeks(self):
        assert parse_duration("1w") == 604800.0
        assert parse_duration("2w") == 1209600.0

    def test_parse_raw_seconds(self):
        assert parse_duration("30") == 30.0
        assert parse_duration("3600") == 3600.0

    def test_parse_with_whitespace(self):
        assert parse_duration("  30s  ") == 30.0
        assert parse_duration("\t5m\n") == 300.0

    def test_parse_case_insensitive(self):
        assert parse_duration("30S") == 30.0
        assert parse_duration("5M") == 300.0
        assert parse_duration("1H") == 3600.0

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("30x")


class TestQPUTimeConfig:
    """Tests for QPUTimeConfig dataclass."""

    def test_default_reservoir_fields(self):
        cfg = QPUTimeConfig(daily_budget_seconds=1800.0)
        assert cfg.min_block_budget_seconds == 90.0
        assert cfg.budget_cap_seconds is None  # None => computed default

    def test_custom_values(self):
        config = QPUTimeConfig(
            daily_budget_seconds=120.0,
            min_blocks_for_estimation=10,
            ema_alpha=0.5,
            min_block_budget_seconds=30.0,
            budget_cap_seconds=300.0,
        )
        assert config.daily_budget_seconds == 120.0
        assert config.min_blocks_for_estimation == 10
        assert config.ema_alpha == 0.5
        assert config.min_block_budget_seconds == 30.0
        assert config.budget_cap_seconds == 300.0


class TestQPUTimeManager:
    """Initial state, recording, estimation."""

    def test_initial_state(self):
        manager = _tm(daily=1800.0)
        assert manager.cumulative_used_us == 0.0
        assert len(manager.block_times_us) == 0
        assert manager.blocks_mined == 0
        assert manager.blocks_skipped == 0
        assert manager.ema_estimate_us is None
        assert manager._pool_us == 0.0
        assert manager._burst_active is False

    def test_record_block_time_tracks_cumulative(self):
        manager = _tm(daily=1800.0)
        manager.record_block_time(5000.0)  # 5ms
        assert manager.cumulative_used_us == 5000.0
        assert len(manager.block_times_us) == 1
        assert manager.blocks_mined == 1

        manager.record_block_time(3000.0)
        assert manager.cumulative_used_us == 8000.0
        assert len(manager.block_times_us) == 2
        assert manager.blocks_mined == 2

    def test_estimate_no_data(self):
        manager = _tm(daily=1800.0)
        assert manager.estimate_next_block_time() == 10_000.0

    def test_estimate_insufficient_data(self):
        manager = _tm(daily=1800.0)
        manager.record_block_time(5000.0)
        manager.record_block_time(6000.0)
        manager.record_block_time(10000.0)
        # max(5000, 6000, 10000) * 1.5 = 15000
        assert manager.estimate_next_block_time() == 15000.0

    def test_estimate_with_ema(self):
        config = QPUTimeConfig(
            daily_budget_seconds=1800.0,
            min_blocks_for_estimation=3,
            ema_alpha=0.5,
        )
        manager = QPUTimeManager(config)
        for _ in range(3):
            manager.record_block_time(10000.0)
        # EMA 10000, *1.2 safety = 12000
        assert manager.estimate_next_block_time() == 12000.0

    def test_confidence_levels(self):
        manager = _tm(daily=1800.0, buffer=0.0)
        assert manager.should_mine_block(now=1_000_000.0).confidence == "low"
        for _ in range(5):
            manager.record_block_time(5000.0)
        assert manager.should_mine_block(now=1_000_000.0).confidence == "medium"
        for _ in range(5):
            manager.record_block_time(5000.0)
        assert manager.should_mine_block(now=1_000_000.0).confidence == "high"

    def test_reset(self):
        manager = _tm(daily=1800.0)
        manager.record_block_time(5000.0)
        manager.record_block_time(7000.0)
        manager.should_mine_block(now=1_000_500.0)
        manager.reset()
        assert manager.cumulative_used_us == 0.0
        assert len(manager.block_times_us) == 0
        assert manager.blocks_mined == 0
        assert manager.blocks_skipped == 0
        assert manager.ema_estimate_us is None
        assert manager._pool_us == 0.0
        assert manager._burst_active is False


class TestReservoir:
    """Accrual, cap, spend."""

    def test_pool_accrues_at_daily_rate(self):
        tm = _tm(daily=1800.0)  # 1800/86400 = 0.0208333 s pool/wall-s
        tm._accrue(1_000_000.0 + 3600.0)  # one hour
        # 3600 * (1800/86400) = 75s
        assert abs(tm._pool_us / 1e6 - 75.0) < 1e-6

    def test_pool_clamps_at_cap(self):
        tm = _tm(daily=86400.0, buffer=90.0, cap=120.0)  # 1s/wall-s
        tm._accrue(1_000_000.0 + 10_000.0)  # would be 10000s; clamp to 120
        assert abs(tm._pool_us / 1e6 - 120.0) < 1e-6

    def test_default_cap_is_max_of_daily_and_buffer(self):
        tm = _tm(daily=60.0, buffer=90.0)  # cap defaults to max(60, 90)=90
        assert abs(tm._pool_cap_us / 1e6 - 90.0) < 1e-6
        tm2 = _tm(daily=3600.0, buffer=90.0)  # max(3600, 90)=3600
        assert abs(tm2._pool_cap_us / 1e6 - 3600.0) < 1e-6

    def test_record_block_time_debits_pool(self):
        tm = _tm(daily=86400.0)  # 1s/wall-s
        tm._accrue(1_000_000.0 + 100.0)  # +100s
        tm.record_block_time(20_000_000.0, now=1_000_000.0 + 100.0)  # spend 20s
        assert abs(tm._pool_us / 1e6 - 80.0) < 1e-3
        assert abs(tm.cumulative_used_us / 1e6 - 20.0) < 1e-3
        assert tm.blocks_mined == 1


class TestHysteresis:
    """Start-at-buffer, continue-until-drained behavior."""

    def test_idle_blocks_until_buffer(self):
        tm = _tm()  # 1s/wall-s, buffer 90
        e = tm.should_mine_block(now=1_000_000.0 + 50.0)  # pool=50 < 90
        assert e.should_mine is False
        assert e.burst_active is False
        assert e.seconds_until_can_mine > 0
        e2 = tm.should_mine_block(now=1_000_000.0 + 95.0)  # pool=95 >= 90
        assert e2.should_mine is True
        assert e2.burst_active is True

    def test_burst_continues_below_buffer_until_drained(self):
        tm = _tm()
        tm.should_mine_block(now=1_000_000.0 + 90.0)  # start burst, pool ~90
        tm.record_block_time(50_000_000.0, now=1_000_000.0 + 90.0)  # -> ~40 (<90)
        e = tm.should_mine_block(now=1_000_000.0 + 90.0)  # same now; bursting
        assert e.should_mine is True                  # continues below buffer
        tm.record_block_time(45_000_000.0, now=1_000_000.0 + 90.0)  # pool <= 0
        e2 = tm.should_mine_block(now=1_000_000.0 + 90.0)
        assert e2.should_mine is False                # drained -> idle
        assert e2.burst_active is False

    def test_end_burst_forces_reaccumulation(self):
        tm = _tm()
        tm.should_mine_block(now=1_000_000.0 + 90.0)  # bursting
        tm.end_burst()
        tm.record_block_time(30_000_000.0, now=1_000_000.0 + 90.0)  # pool ~60 < 90
        e = tm.should_mine_block(now=1_000_000.0 + 90.0)
        assert e.should_mine is False                 # idle + below buffer => blocked

    def test_seconds_until_can_mine_math(self):
        tm = _tm(daily=86400.0, buffer=90.0)          # 1s/wall-s
        e = tm.should_mine_block(now=1_000_000.0 + 40.0)  # pool=40, need 90
        assert abs(e.seconds_until_can_mine - 50.0) < 1e-3
        assert e.should_mine is False


class TestGetStatsAndWarnings:
    def test_get_stats_reservoir_keys(self):
        tm = _tm()
        tm.should_mine_block(now=1_000_000.0 + 90.0)
        s = tm.get_stats()
        for k in ("daily_budget_seconds", "pool_seconds", "pool_cap_seconds",
                  "min_block_budget_seconds", "burst_active", "seconds_until_buffer",
                  "cumulative_used_seconds", "blocks_mined", "blocks_skipped",
                  "budget_remaining_seconds", "block_times_count",
                  "avg_block_time_seconds", "ema_estimate_seconds"):
            assert k in s
        assert s["budget_remaining_seconds"] == s["pool_seconds"]

    def test_startup_warns_when_buffer_exceeds_daily(self, caplog):
        with caplog.at_level(logging.WARNING):
            QPUTimeManager(QPUTimeConfig(daily_budget_seconds=60.0,
                                         min_block_budget_seconds=90.0))
        assert any("min_block_budget" in r.message and "day" in r.message.lower()
                   for r in caplog.records)

    def test_no_startup_warn_when_daily_exceeds_buffer(self, caplog):
        with caplog.at_level(logging.WARNING):
            QPUTimeManager(QPUTimeConfig(daily_budget_seconds=1800.0,
                                         min_block_budget_seconds=90.0))
        assert not any("min_block_budget" in r.message for r in caplog.records)

    def test_startup_warns_when_buffer_exceeds_cap(self, caplog):
        # budget_cap below min_block_budget: the pool caps under the burst
        # threshold, so a burst can never start — the most dangerous of the
        # two misconfigurations (total stall, not just slow fill).
        with caplog.at_level(logging.WARNING):
            QPUTimeManager(QPUTimeConfig(daily_budget_seconds=86400.0,
                                         min_block_budget_seconds=90.0,
                                         budget_cap_seconds=30.0))
        assert any("min_block_budget" in r.message and "NEVER start" in r.message
                   for r in caplog.records)

    def test_burst_never_starts_when_cap_below_buffer(self):
        # Behavioral counterpart to the warning: even after unbounded accrual
        # the pool clamps at the cap (30s) below the burst threshold (90s), so
        # should_mine_block never authorizes a burst.
        tm = QPUTimeManager(QPUTimeConfig(daily_budget_seconds=86400.0,
                                          min_block_budget_seconds=90.0,
                                          budget_cap_seconds=30.0))
        tm.reset_clock(1_000_000.0)
        # Far in the future — the reservoir has long since filled to its cap.
        e = tm.should_mine_block(now=1_000_000.0 + 10 * 86400.0)
        assert e.should_mine is False
        assert e.burst_active is False


class TestInitialBudgetSeeding:
    """Seeding the reservoir on boot so a fresh process mines immediately."""

    def test_default_starts_dry(self):
        # No initial_budget_seconds => historical behavior: pool starts at 0.
        tm = _tm()
        assert tm._pool_us == 0.0

    def test_seed_to_buffer_mines_immediately(self):
        # daily=40s/day would take ~2.25 days to accrue 90s; seeding skips that.
        tm = QPUTimeManager(QPUTimeConfig(
            daily_budget_seconds=40.0,
            min_block_budget_seconds=90.0,
            initial_budget_seconds=90.0,
        ))
        tm.reset_clock(1_000_000.0)
        e = tm.should_mine_block(now=1_000_000.0)  # no accrual yet
        assert e.should_mine is True
        assert e.burst_active is True

    def test_seed_clamps_to_cap(self):
        tm = QPUTimeManager(QPUTimeConfig(
            daily_budget_seconds=86400.0,
            min_block_budget_seconds=90.0,
            budget_cap_seconds=120.0,
            initial_budget_seconds=10_000.0,  # far above cap
        ))
        assert abs(tm._pool_us / 1e6 - 120.0) < 1e-6

    def test_negative_seed_floored_to_zero(self):
        tm = QPUTimeManager(QPUTimeConfig(
            daily_budget_seconds=86400.0,
            initial_budget_seconds=-50.0,
        ))
        assert tm._pool_us == 0.0


class TestResolveInitialBudget:
    """Operator keyword/duration -> seed-seconds resolution."""

    def test_min_keyword(self):
        assert resolve_initial_budget("min", 40.0, 90.0) == 90.0

    def test_daily_keyword(self):
        assert resolve_initial_budget("daily", 600.0, 90.0) == 600.0

    def test_cap_keyword_uses_explicit_cap(self):
        assert resolve_initial_budget("cap", 600.0, 90.0, 1200.0) == 1200.0

    def test_cap_keyword_derives_cap_when_none(self):
        # Mirrors the manager's derived cap = max(daily, buffer).
        assert resolve_initial_budget("cap", 600.0, 90.0, None) == 600.0
        assert resolve_initial_budget("cap", 40.0, 90.0, None) == 90.0

    def test_explicit_duration(self):
        assert resolve_initial_budget("10m", 600.0, 90.0) == 600.0
        assert resolve_initial_budget("300s", 600.0, 90.0) == 300.0

    def test_case_insensitive_and_whitespace(self):
        assert resolve_initial_budget("  Daily  ", 600.0, 90.0) == 600.0

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            resolve_initial_budget("banana", 600.0, 90.0)
