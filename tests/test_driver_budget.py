# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the QUI-867 stream-driver time-budget diagnostic."""

from __future__ import annotations

import json

import pytest

from GPU import driver_budget
from GPU.driver_budget import (
    DriverBudget,
    ENV_ENABLE,
    ENV_OUT,
    ENV_WINDOW,
    _NullBudget,
    record_throttle,
)


@pytest.fixture(autouse=True)
def _reset_throttle_accumulator():
    """Zero the module-level throttle accumulator between tests."""
    driver_budget._drain_throttle()
    yield
    driver_budget._drain_throttle()


class TestFromEnv:
    """The diagnostic is off unless explicitly enabled."""

    def test_absent_env_yields_inert_budget(self, monkeypatch):
        monkeypatch.delenv(ENV_ENABLE, raising=False)
        assert isinstance(DriverBudget.from_env(), _NullBudget)

    @pytest.mark.parametrize("val", ["0", "no", "", "off"])
    def test_falsey_env_yields_inert_budget(self, monkeypatch, val):
        monkeypatch.setenv(ENV_ENABLE, val)
        assert isinstance(DriverBudget.from_env(), _NullBudget)

    @pytest.mark.parametrize("val", ["1", "true", "yes"])
    def test_truthy_env_yields_real_budget(self, monkeypatch, val):
        monkeypatch.setenv(ENV_ENABLE, val)
        assert isinstance(DriverBudget.from_env(), DriverBudget)

    def test_malformed_window_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "not-a-number")
        assert DriverBudget.from_env()._window_s == 60.0


class TestNullBudget:
    """The inert path must satisfy the full surface without side effects."""

    def test_all_methods_are_safe_no_ops(self):
        n = _NullBudget()
        with n.phase("poll"):
            pass
        n.tick_result()
        n.tick_consumer(1.0)
        n.close()
        assert n.enabled is False


class TestWindowReport:
    """A window row must attribute wall-clock to the right buckets."""

    def _rows(self, path):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_emits_row_once_window_elapses(self, tmp_path, monkeypatch):
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "0")  # every tick closes a window
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()
        b.tick_result()
        b.close()

        rows = self._rows(out)
        assert len(rows) == 1
        assert rows[0]["results"] == 1
        assert rows[0]["window"] == 0

    def test_throttle_is_split_out_of_consumer(self, tmp_path, monkeypatch):
        """Throttle sleeps inside the consumer span; the row must separate them.

        This is the whole point of the diagnostic: "consumer" must report ring
        cost only, so a throttle-dominated window is unambiguous.
        """
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "0")
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()

        b.tick_consumer(1.0)     # 1.0s suspended at the yield...
        record_throttle(0.75)    # ...0.75s of which was the NVML sleep
        b.tick_result()
        b.close()

        row = self._rows(out)[0]
        assert row["throttle_events"] == 1
        # 0.75s attributed to throttle, leaving 0.25s of real consumer cost.
        assert row["throttle_ms_per_result"] == pytest.approx(750.0, abs=1.0)
        assert row["consumer_ms_per_result"] == pytest.approx(250.0, abs=1.0)

    def test_consumer_never_goes_negative_on_overlarge_throttle(
        self, tmp_path, monkeypatch,
    ):
        """A throttle exceeding the measured span must clamp, not go negative."""
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "0")
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()

        b.tick_consumer(0.1)
        record_throttle(0.5)  # larger than the recorded consumer span
        b.tick_result()
        b.close()

        assert self._rows(out)[0]["consumer_ms_per_result"] == 0.0

    def test_phase_accumulates_into_named_bucket(self, tmp_path, monkeypatch):
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "0")
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()

        with b.phase("download"):
            pass
        b.tick_result()
        b.close()

        row = self._rows(out)[0]
        assert "download_pct" in row and "download_ms_per_result" in row

    def test_phase_records_time_even_when_body_raises(self):
        """A raising body must still be attributed, or the budget under-counts."""
        b = DriverBudget(window_s=1e9)
        with pytest.raises(ValueError):
            with b.phase("poll"):
                raise ValueError("boom")
        assert b._totals["poll"] > 0.0

    def test_counters_reset_between_windows(self, tmp_path, monkeypatch):
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "0")
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()

        b.tick_result()
        b.tick_result()
        b.close()

        rows = self._rows(out)
        # Each row counts only its own window, never a running total.
        assert [r["results"] for r in rows] == [1, 1]
        assert [r["window"] for r in rows] == [0, 1]

    def test_close_emits_trailing_partial_window(self, tmp_path, monkeypatch):
        """Work in an unfinished window must not be silently discarded."""
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "1e9")  # never closes on its own
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()

        b.tick_result()
        assert self._rows(out) == []  # nothing emitted yet
        b.close()
        assert len(self._rows(out)) == 1

    def test_close_is_idempotent(self, tmp_path, monkeypatch):
        out = tmp_path / "budget.jsonl"
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WINDOW, "1e9")
        monkeypatch.setenv(ENV_OUT, str(out))
        b = DriverBudget.from_env()
        b.tick_result()
        b.close()
        b.close()  # must not raise on an already-closed file
        assert len(self._rows(out)) == 1


class TestThrottleAccumulator:
    """record_throttle is module-level; draining must be exact."""

    def test_drain_returns_and_zeroes(self):
        record_throttle(0.5)
        record_throttle(0.25)
        assert driver_budget._drain_throttle() == (0.75, 2)
        assert driver_budget._drain_throttle() == (0.0, 0)


class TestSchedulerIntegration:
    """throttle_if_busy must attribute its sleep to the budget."""

    def test_throttle_if_busy_records_when_throttling(self):
        from GPU.gpu_scheduler import throttle_if_busy

        class _Busy:
            def should_throttle(self):
                return True

        throttle_if_busy(_Busy(), sleep_fn=lambda _s: None, sleep_s=0.5)
        assert driver_budget._drain_throttle() == (0.5, 1)

    def test_throttle_if_busy_records_nothing_when_idle(self):
        from GPU.gpu_scheduler import throttle_if_busy

        class _Idle:
            def should_throttle(self):
                return False

        throttle_if_busy(_Idle(), sleep_fn=lambda _s: None, sleep_s=0.5)
        assert driver_budget._drain_throttle() == (0.0, 0)

    def test_none_scheduler_records_nothing(self):
        from GPU.gpu_scheduler import throttle_if_busy

        throttle_if_busy(None, sleep_fn=lambda _s: None, sleep_s=0.5)
        assert driver_budget._drain_throttle() == (0.0, 0)
