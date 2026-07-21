# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for MetalScheduler budget calculation and the adaptive cap policy.

No GPU required — tests exercise the static core budget, the occupancy
thread-budget surface, and the pure tier policy (classify / hysteresis /
budget mapping) without spawning the sensor monitor.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_cap_monitor():
    """Stop the adaptive-cap monitor process from spawning in yielding tests."""
    with patch(
        "GPU.metal_scheduler.MetalScheduler._start_cap_monitor",
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

    def test_small_percentage_rounds_down_but_floors_at_one(self):
        from GPU.metal_scheduler import MetalScheduler
        # 10 cores * 5% = 0.5 -> int(0.5) = 0 -> max(1, 0) = 1
        sched = MetalScheduler(
            gpu_core_count=10,
            gpu_utilization_pct=5,
            yielding=False,
        )
        assert sched.get_core_budget() == 1


class TestThreadBudget:
    """Occupancy thread-budget surface (get_thread_budget)."""

    def test_uncapped_when_not_yielding(self):
        from GPU.metal_scheduler import UNCAPPED, MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        # No monitor → never caps occupancy.
        assert sched.get_thread_budget() == UNCAPPED

    def test_returns_published_budget_when_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            yielding=True,
            active_threads=2048,
        )
        # Monitor publishes the live occupancy cap; a fresh heartbeat means
        # the value is trusted as-is.
        sched._heartbeat_value.value = 1
        sched._budget_value.value = 512
        assert sched.get_thread_budget() == 512
        sched.stop()

    def test_stale_heartbeat_falls_back_to_active_cap(self):
        import time
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            yielding=True,
            active_threads=2048,
        )
        # Monitor said PAUSE (0), but if its heartbeat goes stale the
        # scheduler must fail safe to the ACTIVE cap (stay polite), never
        # to UNCAPPED.
        sched._budget_value.value = 0
        sched.get_thread_budget()  # establish heartbeat baseline
        sched._last_hb_time = time.monotonic() - 100  # heartbeat now stale
        assert sched.get_thread_budget() == 2048
        sched.stop()


class TestTargetDispatchMs:
    """The per-dispatch wall-time budget that gates Metal sweep-chunking.

    None ⇒ dispatch monolithically (no UI to protect); a float ⇒ split the
    sweep schedule. The occupancy machinery is covered by TestThreadBudget, so
    here get_thread_budget is stubbed to isolate the four policy branches.
    """

    def _sched(self, yielding: bool):
        from GPU.metal_scheduler import MetalScheduler
        return MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=yielding,
        )

    def test_none_when_yielding_off(self):
        # Yielding disabled → never chunk (peak throughput).
        sched = self._sched(yielding=False)
        assert sched.target_dispatch_ms() is None

    def test_none_when_uncapped(self):
        # IDLE/headless: occupancy uncapped → no user present → monolithic.
        from GPU.metal_scheduler import UNCAPPED
        sched = self._sched(yielding=True)
        sched.get_thread_budget = lambda: UNCAPPED
        assert sched.target_dispatch_ms() is None
        sched.stop()

    def test_none_when_paused(self):
        # PAUSE (budget 0): streaming loop handles PAUSE before dispatch, so
        # chunking must not engage here either.
        sched = self._sched(yielding=True)
        sched.get_thread_budget = lambda: 0
        assert sched.target_dispatch_ms() is None
        sched.stop()

    def test_target_ms_when_capping(self):
        # ACTIVE/LOW: a finite positive occupancy cap means a user is present →
        # split the schedule to the wall-time target.
        from GPU.metal_scheduler import ACTIVE_DISPATCH_TARGET_MS
        sched = self._sched(yielding=True)
        sched.get_thread_budget = lambda: 512
        assert sched.target_dispatch_ms() == ACTIVE_DISPATCH_TARGET_MS
        sched.stop()


class TestTierPolicy:
    """Pure adaptive-cap policy: classify → hysteresis → budget mapping."""

    @staticmethod
    def _signals(*, idle_s=0.0, thermal=None, on_battery=False, displays=1):
        from GPU import macos_sensors
        from GPU.metal_scheduler import Signals
        return Signals(
            idle_s=idle_s,
            thermal_state=thermal or macos_sensors.THERMAL_NOMINAL,
            on_battery=on_battery,
            active_displays=displays,
        )

    def test_active_when_user_present(self):
        from GPU.metal_scheduler import ACTIVE, CapConfig, HysteresisState, decide_tier
        out = decide_tier(self._signals(), CapConfig(), HysteresisState(ACTIVE, 0))
        assert out.tier == ACTIVE

    def test_battery_pauses(self):
        from GPU.metal_scheduler import ACTIVE, CapConfig, HysteresisState, PAUSE, decide_tier
        out = decide_tier(
            self._signals(on_battery=True), CapConfig(), HysteresisState(ACTIVE, 0),
        )
        assert out.tier == PAUSE

    def test_thermal_serious_drops_to_low(self):
        from GPU import macos_sensors
        from GPU.metal_scheduler import ACTIVE, CapConfig, HysteresisState, LOW, decide_tier
        out = decide_tier(
            self._signals(thermal=macos_sensors.THERMAL_SERIOUS),
            CapConfig(),
            HysteresisState(ACTIVE, 0),
        )
        assert out.tier == LOW

    def test_idle_when_away(self):
        from GPU.metal_scheduler import ACTIVE, CapConfig, HysteresisState, IDLE, decide_tier
        cfg = CapConfig(idle_after_s=60.0)
        # No HID input past the idle threshold → IDLE.
        out = decide_tier(
            self._signals(idle_s=120.0), cfg, HysteresisState(ACTIVE, 0),
        )
        assert out.tier == IDLE
        # Headless (no active displays) also reads as IDLE.
        out2 = decide_tier(
            self._signals(displays=0), cfg, HysteresisState(ACTIVE, 0),
        )
        assert out2.tier == IDLE

    def test_budget_mapping(self):
        from GPU.metal_scheduler import (
            ACTIVE, CapConfig, IDLE, LOW, PAUSE, UNCAPPED, tier_budget,
        )
        cfg = CapConfig(active_threads=2048)
        assert tier_budget(PAUSE, cfg) == 0
        assert tier_budget(LOW, cfg) == 1024
        assert tier_budget(IDLE, cfg) == UNCAPPED
        assert tier_budget(ACTIVE, cfg) == 2048

    def test_leaving_protective_tier_is_debounced(self):
        from GPU.metal_scheduler import (
            ACTIVE, CapConfig, HysteresisState, LEAVE_POLLS, PAUSE, decide_tier,
        )
        cfg = CapConfig()
        # Currently paused; the cause clears but leaving PAUSE needs
        # LEAVE_POLLS consecutive clean polls.
        state = HysteresisState(PAUSE, 0)
        for poll in range(1, LEAVE_POLLS):
            state = decide_tier(self._signals(), cfg, state)
            assert state.tier == PAUSE, f"left PAUSE too early on poll {poll}"
        state = decide_tier(self._signals(), cfg, state)
        assert state.tier == ACTIVE

    def test_entering_protective_tier_is_immediate(self):
        from GPU.metal_scheduler import ACTIVE, CapConfig, HysteresisState, PAUSE, decide_tier
        # No debounce on the way *into* protection — battery pauses at once.
        out = decide_tier(
            self._signals(on_battery=True), CapConfig(), HysteresisState(ACTIVE, 0),
        )
        assert out.tier == PAUSE


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

    def test_stop_with_yielding(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        # _start_cap_monitor is patched by the autouse fixture, so
        # _util_proc stays None and stop() is a no-op — just verify no raise.
        sched.stop()


class TestMetalSchedulerCachedUtilization:
    """Test get_measured_gpu method."""

    def test_returns_zero_initially(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=False,
        )
        assert sched.get_measured_gpu() == 0

    def test_returns_set_value(self):
        from GPU.metal_scheduler import MetalScheduler
        sched = MetalScheduler(
            gpu_core_count=40,
            gpu_utilization_pct=100,
            yielding=True,
        )
        sched._util_value.value = 42
        assert sched.get_measured_gpu() == 42
        sched.stop()
