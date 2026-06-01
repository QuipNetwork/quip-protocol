# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the Metal adaptive-cap policy + monitor (GPU/metal_scheduler.py).

The cap policy is Metal-only and entirely independent of the CUDA util
monitor. The tier decision is a pure function over signals + previous state,
exhaustively testable without hardware. The monitor loop is driven with mocked
sensors (no spawn) to assert the published target/measured/heartbeat and the
asymmetric hysteresis transitions.
"""
from __future__ import annotations

from unittest.mock import patch


from GPU import macos_sensors as ms
from GPU.metal_scheduler import (
    ACTIVE,
    CapConfig,
    HysteresisState,
    IDLE,
    LEAVE_POLLS,
    LOW,
    PAUSE,
    Signals,
    cap_monitor_main,
    decide_tier,
    tier_target_pct,
)


CFG = CapConfig(
    idle_after_s=60.0, active_util=30, idle_util=100, serious_util=20,
)


def _step(signals: Signals, state: HysteresisState) -> HysteresisState:
    return decide_tier(signals, CFG, state)


def _active() -> HysteresisState:
    return HysteresisState(tier=ACTIVE, leave_count=0)


def _sig(*, idle=0.0, thermal=ms.THERMAL_NOMINAL, battery=False, displays=1):
    return Signals(
        idle_s=idle, thermal_state=thermal,
        on_battery=battery, active_displays=displays,
    )


# ── target mapping ──────────────────────────────────────────────────────

class TestTargetMapping:
    def test_pause_is_zero(self):
        assert tier_target_pct(PAUSE, CFG) == 0

    def test_low_is_serious_util(self):
        assert tier_target_pct(LOW, CFG) == 20

    def test_idle_is_idle_util(self):
        assert tier_target_pct(IDLE, CFG) == 100

    def test_active_is_active_util(self):
        assert tier_target_pct(ACTIVE, CFG) == 30


# ── classify priority ───────────────────────────────────────────────────

class TestClassifyPriority:
    def test_battery_pauses_immediately(self):
        assert _step(_sig(battery=True), _active()).tier == PAUSE

    def test_critical_thermal_pauses_immediately(self):
        assert _step(_sig(thermal=ms.THERMAL_CRITICAL), _active()).tier == PAUSE

    def test_battery_outranks_serious_thermal(self):
        st = _step(_sig(battery=True, thermal=ms.THERMAL_SERIOUS), _active())
        assert st.tier == PAUSE

    def test_serious_thermal_is_low(self):
        assert _step(_sig(thermal=ms.THERMAL_SERIOUS), _active()).tier == LOW

    def test_headless_is_idle(self):
        assert _step(_sig(displays=0), _active()).tier == IDLE

    def test_long_idle_is_idle(self):
        assert _step(_sig(idle=120.0), _active()).tier == IDLE

    def test_user_present_is_active(self):
        assert _step(_sig(idle=5.0), _active()).tier == ACTIVE


# ── asymmetric hysteresis ───────────────────────────────────────────────

class TestHysteresis:
    def test_leave_pause_requires_n_consecutive_polls(self):
        st = _step(_sig(battery=True), _active())
        assert st.tier == PAUSE
        for _ in range(LEAVE_POLLS - 1):
            st = _step(_sig(idle=5.0), st)
            assert st.tier == PAUSE
        assert _step(_sig(idle=5.0), st).tier == ACTIVE

    def test_pause_leave_debounce_resets_on_reentry(self):
        st = _step(_sig(battery=True), _active())
        st = _step(_sig(idle=5.0), st)
        assert st.tier == PAUSE
        st = _step(_sig(battery=True), st)
        assert st.tier == PAUSE and st.leave_count == 0
        assert _step(_sig(idle=5.0), st).tier == PAUSE

    def test_enter_pause_from_low_is_immediate(self):
        st = _step(_sig(thermal=ms.THERMAL_SERIOUS), _active())
        assert st.tier == LOW
        assert _step(_sig(battery=True), st).tier == PAUSE

    def test_leave_low_requires_thermal_recovery_debounce(self):
        st = _step(_sig(thermal=ms.THERMAL_SERIOUS), _active())
        assert st.tier == LOW
        for _ in range(LEAVE_POLLS - 1):
            st = _step(_sig(thermal=ms.THERMAL_FAIR), st)
            assert st.tier == LOW
        assert _step(_sig(thermal=ms.THERMAL_FAIR), st).tier == ACTIVE

    def test_active_to_idle_is_immediate(self):
        assert _step(_sig(idle=120.0), _active()).tier == IDLE

    def test_idle_to_active_on_first_hid_event_is_immediate(self):
        st = _step(_sig(idle=120.0), _active())
        assert st.tier == IDLE
        assert _step(_sig(idle=0.0), st).tier == ACTIVE


# ── monitor loop with mocked sensors (no spawn) ─────────────────────────

class _V:
    def __init__(self, value=-1):
        self.value = value


class _Stop:
    """stop_event firing after ``after`` waits (bounds the loop)."""

    def __init__(self, after):
        self._after, self._n = after, 0

    def is_set(self):
        return self._n >= self._after

    def wait(self, _t):
        self._n += 1


class TestCapMonitorLoop:
    def test_publishes_active_cap_and_heartbeat(self):
        target, measured, heartbeat = _V(), _V(), _V()
        with patch.object(ms, "hid_idle_seconds", return_value=5.0), \
             patch.object(ms, "thermal_state", return_value=ms.THERMAL_NOMINAL), \
             patch.object(ms, "on_battery", return_value=False), \
             patch.object(ms, "active_display_count", return_value=1), \
             patch.object(ms, "gpu_active_residency", return_value=42):
            cap_monitor_main(
                target, measured, heartbeat, _Stop(after=3),
                0.0, 60.0, 30, 100, 20,
            )
        assert target.value == 30        # user present -> active_util
        assert measured.value == 42
        assert heartbeat.value >= 3

    def test_battery_publishes_pause(self):
        target, measured, heartbeat = _V(), _V(), _V()
        with patch.object(ms, "hid_idle_seconds", return_value=0.0), \
             patch.object(ms, "thermal_state", return_value=ms.THERMAL_NOMINAL), \
             patch.object(ms, "on_battery", return_value=True), \
             patch.object(ms, "active_display_count", return_value=1), \
             patch.object(ms, "gpu_active_residency", return_value=0):
            cap_monitor_main(
                target, measured, heartbeat, _Stop(after=4),
                0.0, 60.0, 30, 100, 20,
            )
        assert target.value == 0
        assert heartbeat.value == 4


# ── scheduler target readout + staleness fallback (no spawn) ────────────

class TestSchedulerTargetReadout:
    def _sched(self):
        from GPU.metal_scheduler import MetalScheduler
        return MetalScheduler(
            gpu_core_count=10, gpu_utilization_pct=100,
            yielding=False, active_util=30,
        )

    def test_yielding_off_is_flat_out(self):
        s = self._sched()
        assert s.get_target_pct() == 100

    def test_reads_published_target_when_fresh(self):
        s = self._sched()
        s._yielding = True
        s._heartbeat_value.value = 5
        s._target_value.value = 40
        assert s.get_target_pct() == 40

    def test_stale_heartbeat_falls_back_to_active_cap(self):
        s = self._sched()
        s._yielding = True
        s._heartbeat_value.value = 5
        s._target_value.value = 40
        assert s.get_target_pct() == 40        # first read seeds heartbeat
        s._stale_timeout_s = -1.0              # force "stale" on next read
        assert s.get_target_pct() == 30        # heartbeat unchanged -> active

    def test_measured_gpu_readout(self):
        s = self._sched()
        s._util_value.value = 73
        assert s.get_measured_gpu() == 73
