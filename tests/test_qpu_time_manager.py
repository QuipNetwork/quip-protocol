"""Unit tests for QPU time budget management."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
import os
import importlib.util

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
QPUTimeConfig = _module.QPUTimeConfig
QPUTimeEstimate = _module.QPUTimeEstimate
QPUTimeManager = _module.QPUTimeManager

import pytest


class TestParseDuration:
    """Tests for the parse_duration function."""

    def test_parse_seconds(self):
        """Test parsing seconds format."""
        assert parse_duration("30s") == 30.0
        assert parse_duration("1s") == 1.0
        assert parse_duration("0s") == 0.0
        assert parse_duration("3600s") == 3600.0

    def test_parse_minutes(self):
        """Test parsing minutes format."""
        assert parse_duration("5m") == 300.0
        assert parse_duration("1m") == 60.0
        assert parse_duration("20m") == 1200.0

    def test_parse_hours(self):
        """Test parsing hours format."""
        assert parse_duration("1h") == 3600.0
        assert parse_duration("2h") == 7200.0
        assert parse_duration("0.5h") == 1800.0

    def test_parse_days(self):
        """Test parsing days format."""
        assert parse_duration("1d") == 86400.0
        assert parse_duration("2d") == 172800.0

    def test_parse_weeks(self):
        """Test parsing weeks format."""
        assert parse_duration("1w") == 604800.0
        assert parse_duration("2w") == 1209600.0

    def test_parse_raw_seconds(self):
        """Test parsing raw numeric seconds."""
        assert parse_duration("30") == 30.0
        assert parse_duration("3600") == 3600.0

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        assert parse_duration("  30s  ") == 30.0
        assert parse_duration("\t5m\n") == 300.0

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_duration("30S") == 30.0
        assert parse_duration("5M") == 300.0
        assert parse_duration("1H") == 3600.0

    def test_parse_invalid_raises(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("30x")


class TestQPUTimeConfig:
    """Tests for QPUTimeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        assert config.daily_budget_seconds == 60.0
        assert config.min_blocks_for_estimation == 5
        assert config.ema_alpha == 0.3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QPUTimeConfig(
            daily_budget_seconds=120.0,
            min_blocks_for_estimation=10,
            ema_alpha=0.5,
        )
        assert config.daily_budget_seconds == 120.0
        assert config.min_blocks_for_estimation == 10
        assert config.ema_alpha == 0.5


class TestQPUTimeManager:
    """Tests for QPUTimeManager class."""

    def test_initial_state(self):
        """Test initial state of time manager."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        assert manager.cumulative_used_us == 0.0
        assert len(manager.block_times_us) == 0
        assert manager.blocks_mined == 0
        assert manager.blocks_skipped == 0
        assert manager.ema_estimate_us is None

    def test_record_block_time(self):
        """Test recording block times."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        manager.record_block_time(5000.0)  # 5ms
        assert manager.cumulative_used_us == 5000.0
        assert len(manager.block_times_us) == 1
        assert manager.blocks_mined == 1

        manager.record_block_time(3000.0)  # 3ms
        assert manager.cumulative_used_us == 8000.0
        assert len(manager.block_times_us) == 2
        assert manager.blocks_mined == 2

    def test_estimate_no_data(self):
        """Test estimation with no recorded data."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        estimate = manager.estimate_next_block_time()
        assert estimate == 10_000.0  # Default 10ms

    def test_estimate_insufficient_data(self):
        """Test estimation with fewer blocks than min_blocks_for_estimation."""
        config = QPUTimeConfig(daily_budget_seconds=60.0, min_blocks_for_estimation=5)
        manager = QPUTimeManager(config)

        # Record 3 blocks (less than 5)
        manager.record_block_time(5000.0)
        manager.record_block_time(6000.0)
        manager.record_block_time(10000.0)

        estimate = manager.estimate_next_block_time()
        # Should be max(5000, 6000, 10000) * 1.5 = 15000
        assert estimate == 15000.0

    def test_estimate_with_ema(self):
        """Test EMA estimation with sufficient data."""
        config = QPUTimeConfig(
            daily_budget_seconds=60.0,
            min_blocks_for_estimation=3,
            ema_alpha=0.5,
        )
        manager = QPUTimeManager(config)

        # Record 3 blocks to enable EMA
        manager.record_block_time(10000.0)
        manager.record_block_time(10000.0)
        manager.record_block_time(10000.0)

        # EMA should be initialized to average (10000)
        # Then updated: 0.5 * 10000 + 0.5 * 10000 = 10000
        # With 20% safety: 10000 * 1.2 = 12000
        estimate = manager.estimate_next_block_time()
        assert estimate == 12000.0

    def test_should_mine_sufficient_budget(self):
        """Test mining decision with sufficient budget."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)  # 60 seconds = 60,000,000 us
        manager = QPUTimeManager(config)

        result = manager.should_mine_block()
        assert result.should_mine is True
        assert result.daily_budget_us == 60_000_000.0
        # With proportional pacing, budget_remaining = proportional_limit - cumulative_used
        # Budget remaining should be positive (equal to proportional limit since cumulative is 0)
        assert result.budget_remaining_us == result.proportional_limit_us
        assert result.budget_remaining_us > 0

    def test_should_mine_insufficient_budget(self):
        """Test mining decision when budget is exhausted."""
        import time as time_module
        config = QPUTimeConfig(daily_budget_seconds=0.005)  # 5ms = 5000 us
        manager = QPUTimeManager(config)

        # Simulate being at end of day so proportional limit equals daily budget
        manager.day_start_timestamp = time_module.time() - 86400 + 60

        # Default estimate is 10ms (10000 us), but budget is only 5ms
        # cumulative (0) + estimated (10000) > budget (5000)
        result = manager.should_mine_block()
        assert result.should_mine is False
        assert manager.blocks_skipped == 1

    def test_should_mine_after_usage(self):
        """Test mining decision after some usage."""
        import time as time_module
        config = QPUTimeConfig(daily_budget_seconds=0.020)  # 20ms = 20000 us
        manager = QPUTimeManager(config)

        # Simulate being at end of day so proportional limit equals daily budget
        # Freeze time at 23h 59m into the day (86400 - 60 seconds elapsed)
        now = time_module.time()
        day_start = manager._calculate_day_start(now)
        manager.day_start_timestamp = day_start
        now = day_start + 86400 - 60  # 60 seconds before midnight

        # First check should pass (0 + 10000 <= 20000)
        result = manager.should_mine_block(now=now)
        assert result.should_mine is True

        # Record 15ms of usage
        manager.record_block_time(15000.0)

        # Second check should fail (15000 + 10000 > 20000)
        result = manager.should_mine_block(now=now)
        assert result.should_mine is False

    def test_confidence_levels(self):
        """Test confidence level calculation."""
        config = QPUTimeConfig(
            daily_budget_seconds=60.0,
            min_blocks_for_estimation=5,
        )
        manager = QPUTimeManager(config)

        # No blocks: low confidence
        result = manager.should_mine_block()
        assert result.confidence == "low"

        # Record 5 blocks (at threshold)
        for _ in range(5):
            manager.record_block_time(5000.0)

        # At min_blocks: medium confidence
        result = manager.should_mine_block()
        assert result.confidence == "medium"

        # Record 5 more blocks
        for _ in range(5):
            manager.record_block_time(5000.0)

        # 2x min_blocks: high confidence
        result = manager.should_mine_block()
        assert result.confidence == "high"

    def test_get_stats(self):
        """Test statistics retrieval."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        manager.record_block_time(5000.0)
        manager.record_block_time(7000.0)

        stats = manager.get_stats()
        assert stats["daily_budget_seconds"] == 60.0
        assert stats["cumulative_used_seconds"] == 0.012  # 12000 us = 0.012s
        assert stats["blocks_mined"] == 2
        assert stats["blocks_skipped"] == 0
        assert stats["block_times_count"] == 2
        assert stats["avg_block_time_seconds"] == 0.006  # 6000 us avg = 0.006s

    def test_reset(self):
        """Test resetting the time manager."""
        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # Record some data
        manager.record_block_time(5000.0)
        manager.record_block_time(7000.0)
        assert manager.cumulative_used_us > 0
        assert manager.blocks_mined == 2

        # Reset
        manager.reset()

        assert manager.cumulative_used_us == 0.0
        assert len(manager.block_times_us) == 0
        assert manager.blocks_mined == 0
        assert manager.blocks_skipped == 0
        assert manager.ema_estimate_us is None


class TestQPUTimeEstimate:
    """Tests for QPUTimeEstimate dataclass."""

    def test_estimate_creation(self):
        """Test creating a time estimate."""
        estimate = QPUTimeEstimate(
            estimated_block_time_us=10000.0,
            cumulative_used_us=50000.0,
            daily_budget_us=1000000.0,
            proportional_limit_us=500000.0,
            budget_remaining_us=450000.0,
            should_mine=True,
            confidence="high",
            elapsed_fraction=0.5,
            seconds_until_can_mine=0.0,
            is_pacing_limited=False,
        )

        assert estimate.estimated_block_time_us == 10000.0
        assert estimate.cumulative_used_us == 50000.0
        assert estimate.daily_budget_us == 1000000.0
        assert estimate.proportional_limit_us == 500000.0
        assert estimate.budget_remaining_us == 450000.0
        assert estimate.should_mine is True
        assert estimate.confidence == "high"
        assert estimate.elapsed_fraction == 0.5


class TestProportionalPacing:
    """Tests for proportional pacing feature."""

    def test_day_start_calculation(self):
        """Test that day start is calculated correctly."""
        import time as time_module
        from datetime import datetime, timezone

        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # day_start_timestamp should be UTC midnight today
        now = time_module.time()
        utc_now = datetime.fromtimestamp(now, tz=timezone.utc)
        expected_day_start = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

        assert manager.day_start_timestamp == expected_day_start.timestamp()

    def test_proportional_limit_early_day(self):
        """At 25% of day, proportional limit should be 25% of budget."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s budget
        manager = QPUTimeManager(config)

        # Manually set day_start to simulate being 6 hours (25%) into the day
        base_now = time_module.time()
        day_start = manager._calculate_day_start(base_now)
        manager.day_start_timestamp = day_start
        now = day_start + (6 * 3600)  # 6 hours after midnight

        result = manager.should_mine_block(now=now)

        # At 25% of day, limit should be ~10s (25% of 40s)
        # Allow some tolerance for time passing during test
        assert 9_000_000 <= result.proportional_limit_us <= 11_000_000
        assert 0.24 <= result.elapsed_fraction <= 0.26

    def test_proportional_limit_midday(self):
        """At 50% of day, proportional limit should be 50% of budget."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s budget
        manager = QPUTimeManager(config)

        # Manually set day_start to simulate being 12 hours (50%) into the day
        base_now = time_module.time()
        day_start = manager._calculate_day_start(base_now)
        manager.day_start_timestamp = day_start
        now = day_start + (12 * 3600)  # 12 hours after midnight

        result = manager.should_mine_block(now=now)

        # At 50% of day, limit should be ~20s (50% of 40s)
        assert 19_000_000 <= result.proportional_limit_us <= 21_000_000
        assert 0.49 <= result.elapsed_fraction <= 0.51

    def test_proportional_pacing_blocks_early_usage(self):
        """Should block mining if usage exceeds proportional limit."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s = 40,000,000 us
        manager = QPUTimeManager(config)

        # Simulate being 25% into the day (proportional limit = 10s)
        base_now = time_module.time()
        day_start = manager._calculate_day_start(base_now)
        manager.day_start_timestamp = day_start
        now = day_start + (6 * 3600)  # 6 hours after midnight

        # Record 12s of usage (exceeds 10s proportional limit)
        manager.cumulative_used_us = 12_000_000

        result = manager.should_mine_block(now=now)

        # Should not mine because 12s > 10s proportional limit
        assert result.should_mine is False
        assert manager.blocks_skipped == 1
        # Should be pacing limited (not budget exhausted)
        assert result.is_pacing_limited is True
        assert result.seconds_until_can_mine > 0

    def test_proportional_pacing_allows_within_limit(self):
        """Should allow mining if usage is within proportional limit."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s = 40,000,000 us
        manager = QPUTimeManager(config)

        # Simulate being 50% into the day (proportional limit = 20s)
        manager.day_start_timestamp = time_module.time() - (12 * 3600)

        # Record 5s of usage (well under 20s proportional limit)
        manager.cumulative_used_us = 5_000_000

        result = manager.should_mine_block()

        # Should mine because 5s + 10ms estimated << 20s proportional limit
        assert result.should_mine is True
        assert result.seconds_until_can_mine == 0.0
        assert result.is_pacing_limited is False

    def test_day_rollover_resets_counters(self):
        """Counters should reset when day changes."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # Record some usage
        manager.record_block_time(5000.0)
        manager.record_block_time(7000.0)
        assert manager.cumulative_used_us == 12000.0
        assert manager.blocks_mined == 2

        # Manually set day_start to yesterday (simulate day rollover)
        manager.day_start_timestamp = time_module.time() - 86400 - 100  # Yesterday

        # Call should_mine_block which triggers rollover check
        result = manager.should_mine_block()

        # Counters should be reset
        assert manager.cumulative_used_us == 0.0
        assert manager.blocks_mined == 0
        assert manager.blocks_skipped == 0
        # EMA should be preserved for estimation continuity
        assert result.should_mine is True

    def test_elapsed_fraction_capped_at_one(self):
        """Elapsed fraction should be capped at 1.0 when calculated value exceeds it.

        Note: The day rollover logic normally prevents this scenario, but
        the cap is a defensive measure that we test by verifying the min() call
        in the should_mine_block implementation.
        """
        import time as time_module
        from datetime import datetime, timezone

        config = QPUTimeConfig(daily_budget_seconds=40.0)
        manager = QPUTimeManager(config)

        # Set day_start to just after UTC midnight today (no rollover will occur)
        # This simulates being at the current time of day
        now = time_module.time()
        utc_now = datetime.fromtimestamp(now, tz=timezone.utc)
        today_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        manager.day_start_timestamp = today_midnight.timestamp()

        result = manager.should_mine_block()

        # Elapsed fraction should always be in valid range [0, 1]
        assert 0.0 <= result.elapsed_fraction <= 1.0
        # Proportional limit should be between 0 and daily budget
        assert 0.0 <= result.proportional_limit_us <= 40_000_000.0
        # If very late in day (>95%), proportional limit should be substantial
        if result.elapsed_fraction > 0.95:
            assert result.proportional_limit_us > 38_000_000.0

    def test_get_stats_includes_pacing_info(self):
        """get_stats should include proportional pacing information."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # Simulate being 50% into the day
        manager.day_start_timestamp = time_module.time() - (12 * 3600)

        stats = manager.get_stats()

        assert "proportional_limit_seconds" in stats
        assert "elapsed_fraction" in stats
        # At 50% of day, proportional limit should be ~30s
        assert 29.0 <= stats["proportional_limit_seconds"] <= 31.0
        assert 0.49 <= stats["elapsed_fraction"] <= 0.51

    def test_reset_updates_day_start(self):
        """reset() should update day_start_timestamp."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # Set day_start to yesterday
        old_day_start = time_module.time() - 86400
        manager.day_start_timestamp = old_day_start

        manager.reset()

        # day_start should be updated to today
        assert manager.day_start_timestamp > old_day_start

    def test_continuous_pacing_uses_last_block_time(self):
        """Wait time should be calculated using last block time, not estimate."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s budget
        manager = QPUTimeManager(config)

        # Simulate being 50% into the day (12 hours, proportional limit = 20s)
        base_now = time_module.time()
        day_start = manager._calculate_day_start(base_now)
        manager.day_start_timestamp = day_start
        now = day_start + (12 * 3600)  # 12 hours after midnight

        # Directly set cumulative to simulate already having mined
        # and record a last block of 8s
        manager.cumulative_used_us = 15_000_000  # 15s used
        manager.block_times_us = [7_000_000, 8_000_000]  # Last block was 8s

        # Need headroom = last block time = 8s
        # Need proportional_limit >= 15s + 8s = 23s
        # 23/40 = 57.5% of day = 13.8 hours from midnight
        # Currently at 12 hours, so wait = 1.8h = 6480s

        result = manager.should_mine_block(now=now)
        assert result.should_mine is False
        assert result.is_pacing_limited is True
        # Wait time should be ~1.8 hours = ~6480 seconds
        assert 6000 <= result.seconds_until_can_mine <= 7000

    def test_wait_past_midnight(self):
        """When cumulative + headroom > daily budget, wait extends past midnight."""
        import time as time_module
        from datetime import datetime, timezone

        config = QPUTimeConfig(daily_budget_seconds=40.0)  # 40s budget
        manager = QPUTimeManager(config)

        # Calculate today's midnight to avoid day rollover issues
        now = time_module.time()
        utc_now = datetime.fromtimestamp(now, tz=timezone.utc)
        today_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Set day_start to today's midnight, then set cumulative as if we're
        # 22 hours into the day
        manager.day_start_timestamp = today_midnight.timestamp()

        # Directly set state to simulate being 22 hours in with lots of usage
        manager.cumulative_used_us = 50_000_000  # 50s used (exceeds 40s budget!)
        manager.block_times_us = [15_000_000, 15_000_000, 20_000_000]  # Last was 20s

        # Need headroom = last block time = 20s
        # cumulative + headroom = 50s + 20s = 70s > 40s daily budget
        # Can't mine today! Wait until midnight, then wait for 20s proportional limit
        # 20/40 = 50% of next day = 12 hours after midnight

        result = manager.should_mine_block()
        assert result.should_mine is False
        assert result.is_pacing_limited is True
        # Wait time should extend past midnight
        # The exact value depends on current time, but should be > 12 hours
        # (time until midnight + 12 hours into next day)
        assert result.seconds_until_can_mine > 0
        # If we're past 12:00 UTC, wait is less than 24h; if before, could be more
        # Just verify it's a substantial wait (at least 12 hours = 43200s)
        # since we need to wait until 50% of next day
        assert result.seconds_until_can_mine >= 43200

    def test_can_mine_shows_zero_wait(self):
        """When can mine, seconds_until_can_mine should be 0."""
        import time as time_module

        config = QPUTimeConfig(daily_budget_seconds=60.0)
        manager = QPUTimeManager(config)

        # Simulate being 50% into the day (proportional limit = 30s)
        manager.day_start_timestamp = time_module.time() - (12 * 3600)

        # No usage yet
        result = manager.should_mine_block()

        assert result.should_mine is True
        assert result.seconds_until_can_mine == 0.0
        assert result.is_pacing_limited is False
