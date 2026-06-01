# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for the macOS sensor wrappers (GPU/macos_sensors.py).

These tests mock the OS boundary (the per-sensor ``_raw_*`` helpers and the
IOKit query) so they run on any platform without Apple frameworks. A small
darwin-gated smoke section exercises the real ctypes paths when available.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from GPU import macos_sensors as ms


# ── safe-default policy: every sensor degrades, never raises ────────────

class TestSafeDefaults:
    """On a raw-layer failure each sensor returns its polite default."""

    def test_hid_idle_default_is_zero(self):
        with patch.object(ms, "_raw_hid_idle_seconds", side_effect=OSError):
            assert ms.hid_idle_seconds() == 0.0

    def test_thermal_default_is_nominal(self):
        with patch.object(ms, "_raw_thermal_state", side_effect=OSError):
            assert ms.thermal_state() == ms.THERMAL_NOMINAL

    def test_on_battery_default_is_false(self):
        with patch.object(ms, "_raw_on_battery", side_effect=OSError):
            assert ms.on_battery() is False

    def test_active_displays_default_is_one(self):
        with patch.object(ms, "_raw_active_display_count", side_effect=OSError):
            assert ms.active_display_count() == 1

    def test_residency_default_is_zero(self):
        with patch.object(ms, "_ioreport_residency", return_value=None), \
             patch.object(ms, "_query_iokit_gpu_utilization", side_effect=OSError):
            assert ms.gpu_active_residency() == 0


# ── pass-through / parsing of mocked raw values ─────────────────────────

class TestPassThrough:
    def test_hid_idle_passes_raw_value(self):
        with patch.object(ms, "_raw_hid_idle_seconds", return_value=42.5):
            assert ms.hid_idle_seconds() == 42.5

    @pytest.mark.parametrize("code,expected", [
        (0, ms.THERMAL_NOMINAL),
        (1, ms.THERMAL_FAIR),
        (2, ms.THERMAL_SERIOUS),
        (3, ms.THERMAL_CRITICAL),
        (99, ms.THERMAL_NOMINAL),  # unknown code -> safe nominal
    ])
    def test_thermal_maps_code_to_name(self, code, expected):
        with patch.object(ms, "_raw_thermal_state", return_value=code):
            assert ms.thermal_state() == expected

    def test_on_battery_true(self):
        with patch.object(ms, "_raw_on_battery", return_value=True):
            assert ms.on_battery() is True

    def test_active_displays_passes_count(self):
        with patch.object(ms, "_raw_active_display_count", return_value=3):
            assert ms.active_display_count() == 3


# ── residency: IOReport primary, IOKit fallback, clamping ───────────────

class TestResidencyFallback:
    def test_uses_ioreport_when_available(self):
        with patch.object(ms, "_ioreport_residency", return_value=73):
            assert ms.gpu_active_residency() == 73

    def test_falls_back_to_iokit_when_ioreport_none(self):
        with patch.object(ms, "_ioreport_residency", return_value=None), \
             patch.object(ms, "_query_iokit_gpu_utilization", return_value=55):
            assert ms.gpu_active_residency() == 55

    def test_clamps_out_of_range_high(self):
        with patch.object(ms, "_ioreport_residency", return_value=250):
            assert ms.gpu_active_residency() == 100

    def test_clamps_out_of_range_low(self):
        with patch.object(ms, "_ioreport_residency", return_value=-5):
            assert ms.gpu_active_residency() == 0


# ── darwin smoke: real ctypes paths must return sane values, never raise ─

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sensors")
class TestDarwinSmoke:
    def test_hid_idle_nonnegative(self):
        assert ms.hid_idle_seconds() >= 0.0

    def test_thermal_is_known(self):
        assert ms.thermal_state() in ms.THERMAL_STATES

    def test_on_battery_is_bool(self):
        assert isinstance(ms.on_battery(), bool)

    def test_active_displays_nonnegative(self):
        assert ms.active_display_count() >= 0

    def test_residency_in_range(self):
        assert 0 <= ms.gpu_active_residency() <= 100
