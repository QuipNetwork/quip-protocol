# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for MetalScheduler budget calculation and throttling.

No GPU required — tests exercise the scheduler logic with
mocked IOKit responses.
"""

import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_iokit_monitor():
    """Prevent IOKit polling on non-macOS platforms."""
    with patch(
        "GPU.metal_scheduler.MetalScheduler._start_iokit_monitor",
    ):
        yield


class TestMetalSchedulerBudget:
    """Core budget computation from utilization config."""

    def test_full_utilization(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert sched.get_core_budget() == 40

    def test_half_utilization(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=50,
            yielding=False,
        )
        assert sched.get_core_budget() == 20

    def test_minimum_budget_is_one(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=1,
            gpu_utilization_pct=1,
            yielding=False,
        )
        assert sched.get_core_budget() >= 1

    def test_small_percentage_rounds_down_but_floors_at_one(self):
        from GPU.metal_scheduler import MetalScheduler
        # 10 cores * 5% = 0.5 -> int(0.5) = 0 -> max(1, 0) = 1
        sched = MetalScheduler(
            gpu_core_count=10,
            gpu_utilization_pct=5,
            yielding=False,
        )
        assert sched.get_core_budget() == 1


class TestMetalSchedulerThrottle:
    """Throttle behavior in yielding vs non-yielding mode."""

    def test_no_throttle_when_not_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert sched.should_throttle() is False

    def test_throttle_when_external_load_high(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        # Simulate high external load
        sched._external_util_pct = 95
        assert sched.should_throttle() is True
        sched.stop()

    def test_no_throttle_when_external_load_low(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 50
        assert sched.should_throttle() is False
        sched.stop()


class TestMetalSchedulerTargetThreadgroups:
    """Target threadgroup computation."""

    def test_full_budget_when_not_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert sched.compute_target_threadgroups(
            max_tg=10, active_tg=5,
        ) == 10

    def test_reduced_when_external_load(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 80
        target = sched.compute_target_threadgroups(
            max_tg=10, active_tg=0,
        )
        # With 80% external, should reduce from 10
        assert 1 <= target < 10
        sched.stop()

    def test_minimum_target_is_one(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 99
        target = sched.compute_target_threadgroups(
            max_tg=10, active_tg=0,
        )
        assert target >= 1
        sched.stop()


class TestStableTargetHysteresis:
    """Hysteresis for stable target threadgroups."""

    def test_returns_none_on_first_call(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 80
        result = sched.check_stable_target_threadgroups(
            max_tg=10, active_tg=0,
        )
        assert result is None
        sched.stop()

    def test_returns_value_after_two_stable_calls(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 80
        sched.check_stable_target_threadgroups(10, 0)
        result = sched.check_stable_target_threadgroups(10, 0)
        assert result is not None
        assert 1 <= result <= 10
        sched.stop()

    def test_resets_on_target_change(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 80
        sched.check_stable_target_threadgroups(10, 0)
        # Change external load → different target
        sched._external_util_pct = 20
        result = sched.check_stable_target_threadgroups(10, 0)
        assert result is None  # Reset, need 2 stable again
        sched.stop()

    def test_returns_max_when_not_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        # Even without yielding, should return max after 2 calls
        sched.check_stable_target_threadgroups(10, 5)
        result = sched.check_stable_target_threadgroups(10, 5)
        assert result == 10


class TestIOKitQuery:
    """Test IOKit query function with graceful fallback."""

    def test_returns_int_in_range(self):
        from GPU.metal_scheduler import _query_iokit_gpu_utilization
        # On any platform, should return 0-100 (0 on non-macOS)
        result = _query_iokit_gpu_utilization()
        assert isinstance(result, int)
        assert 0 <= result <= 100

    def test_fallback_on_missing_library(self):
        from GPU.metal_scheduler import _query_iokit_gpu_utilization
        with patch("ctypes.cdll.LoadLibrary", side_effect=OSError):
            assert _query_iokit_gpu_utilization() == 0


class TestMetalSchedulerStop:
    """Verify clean shutdown."""

    def test_stop_without_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        # Should not raise
        sched.stop()

    def test_stop_with_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched.stop()
        assert sched._iokit_stop.is_set()


class TestMetalSchedulerCachedUtilization:
    """Test get_cached_utilization method."""

    def test_returns_zero_initially(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert sched.get_cached_utilization() == 0

    def test_returns_set_value(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._external_util_pct = 42
        assert sched.get_cached_utilization() == 42
        sched.stop()


class TestDutyCycleController:
    """DutyCycleController duty-cycle math and PI feedback."""

    def test_disabled_at_100_percent(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=100)
        assert dc.enabled is False

    def test_enabled_below_100(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        assert dc.enabled is True

    def test_compute_sleep_at_30_percent(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        sleep = dc.compute_sleep(0.1)
        # 0.1 * (1/0.3 - 1) ≈ 0.233
        assert 0.20 < sleep < 0.27

    def test_compute_sleep_at_50_percent(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=50)
        sleep = dc.compute_sleep(0.1)
        # 0.1 * (1/0.5 - 1) = 0.1
        assert 0.08 < sleep < 0.12

    def test_compute_sleep_returns_min_when_disabled(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=100)
        sleep = dc.compute_sleep(0.1)
        assert sleep == dc._MIN_SLEEP_S

    def test_compute_sleep_clamps_max(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=1)
        # 10s * (1/0.01 - 1) = 990s → clamped to 2s
        sleep = dc.compute_sleep(10.0)
        assert sleep <= dc._MAX_SLEEP_S

    def test_compute_sleep_clamps_min(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        sleep = dc.compute_sleep(0.0001)
        assert sleep >= dc._MIN_SLEEP_S

    def test_ema_smoothing_converges(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=50)
        # Feed 10 samples of 0.1s
        for _ in range(10):
            dc.compute_sleep(0.1)
        # EMA should be near 0.1
        assert 0.08 < dc._ema_compute_s < 0.12

    def test_ema_smoothing_filters_spike(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=50)
        # Steady state at 0.1s
        for _ in range(10):
            dc.compute_sleep(0.1)
        # Spike to 1.0s
        dc.compute_sleep(1.0)
        # EMA should be smoothed, not jump to 1.0
        assert dc._ema_compute_s < 0.5

    def test_feedback_increases_sleep_when_hot(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        initial_mult = dc._duty_multiplier
        # Report 60% utilization vs 30% target
        for _ in range(5):
            dc.feedback(60)
        assert dc._duty_multiplier > initial_mult

    def test_feedback_decreases_sleep_when_cool(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        # Start with inflated multiplier
        dc._duty_multiplier = 3.0
        # Report 10% utilization vs 30% target
        for _ in range(5):
            dc.feedback(10)
        assert dc._duty_multiplier < 3.0

    def test_feedback_noop_when_disabled(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=100)
        dc.feedback(90)
        assert dc._duty_multiplier == 1.0

    def test_feedback_clamps_multiplier(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        # Extreme positive error for many iterations
        for _ in range(1000):
            dc.feedback(100)
        assert dc._duty_multiplier <= 10.0
        # Extreme negative error
        for _ in range(1000):
            dc.feedback(0)
        assert dc._duty_multiplier >= 0.1

    def test_reset_clears_state(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=30)
        dc.compute_sleep(0.1)
        dc.feedback(90)
        dc.reset()
        assert dc._ema_compute_s == 0.0
        assert dc._ema_initialized is False
        assert dc._duty_multiplier == 1.0
        assert dc._integral == 0.0

    def test_target_pct_clamped(self):
        from GPU.metal_scheduler import DutyCycleController
        dc = DutyCycleController(target_pct=0)
        assert dc._target_pct == 1
        dc2 = DutyCycleController(target_pct=200)
        assert dc2._target_pct == 100
