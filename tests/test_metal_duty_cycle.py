# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for DutyCycleController.set_target coordinated reset.

A runtime tier change must update the target AND reset the EMA + PI integral
together, so the next compute_sleep can't carry windup or an EMA from the old
cap (the verification-flagged reset/feedback desync).
"""
from __future__ import annotations

from GPU.metal_scheduler import DutyCycleController


class TestSetTarget:
    def test_updates_target_ratio_and_enabled(self):
        dc = DutyCycleController(target_pct=100)
        assert dc.enabled is False           # 100% -> disabled
        dc.set_target(30)
        assert dc.target_pct == 30
        assert dc.enabled is True
        assert dc._duty_ratio == 0.3

    def test_disables_at_100(self):
        dc = DutyCycleController(target_pct=30)
        assert dc.enabled is True
        dc.set_target(100)
        assert dc.enabled is False

    def test_resets_ema_and_pi_integral(self):
        dc = DutyCycleController(target_pct=30)
        # Build up EMA + integral windup.
        dc.compute_sleep(0.1)
        dc.feedback(95)
        assert dc._ema_initialized is True
        assert dc._integral != 0.0
        dc.set_target(50)
        assert dc._ema_initialized is False
        assert dc._ema_compute_s == 0.0
        assert dc._integral == 0.0
        assert dc._duty_multiplier == 1.0

    def test_no_step_jump_after_target_change(self):
        """First compute_sleep after set_target re-seeds EMA from the new
        sample (no carry-over from the previous cap)."""
        dc = DutyCycleController(target_pct=30)
        dc.compute_sleep(0.5)          # seed a large EMA at 30%
        dc.set_target(50)
        sleep = dc.compute_sleep(0.1)  # fresh sample at 50%
        # EMA re-seeded to 0.1 -> sleep = 0.1 * (1/0.5 - 1) = 0.1, not the
        # stale 0.5-based value.
        assert abs(sleep - 0.1) < 1e-6

    def test_idempotent_same_target_keeps_state(self):
        """Setting the same target must not reset accumulated state."""
        dc = DutyCycleController(target_pct=30)
        dc.compute_sleep(0.1)
        dc.feedback(95)
        integral = dc._integral
        dc.set_target(30)
        assert dc._integral == integral
        assert dc._ema_initialized is True
