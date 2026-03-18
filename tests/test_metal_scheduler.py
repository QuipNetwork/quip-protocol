# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for MetalScheduler budget calculation and throttling.

No GPU required — tests exercise the scheduler logic with
mocked IOKit responses.
"""

import sys
from unittest.mock import patch

import pytest


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
