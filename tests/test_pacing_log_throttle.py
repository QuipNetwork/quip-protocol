# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for QPU pacing log rate-limiting and base_miner setup-abort throttling.

All tests use injected/fake clocks; no real DWaveMiner (which contacts D-Wave)
or real BaseMiner subclass with a live sampler is constructed here.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional

import pytest

from QPU.dwave_miner import _PacingRateLimiter
from shared.base_miner import _SetupAbortThrottle


# ---------------------------------------------------------------------------
# _PacingRateLimiter tests
# ---------------------------------------------------------------------------

class TestPacingRateLimiter:
    """Tests for the QPU pacing log rate-limiter."""

    def _make(self, interval: float = 60.0) -> _PacingRateLimiter:
        return _PacingRateLimiter(log_interval=interval)

    # ------------------------------------------------------------------
    # Entry into paced state logs immediately
    # ------------------------------------------------------------------

    def test_first_paced_call_logs(self):
        rl = self._make()
        assert rl.should_log(now=0.0, wait_bucket="2h") is True

    def test_immediate_repeat_same_bucket_suppressed(self):
        rl = self._make()
        rl.should_log(now=0.0, wait_bucket="2h")
        assert rl.should_log(now=0.0, wait_bucket="2h") is False

    def test_repeat_also_suppressed_before_interval(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        assert rl.should_log(now=59.9, wait_bucket="2h") is False

    # ------------------------------------------------------------------
    # Re-logs after interval elapses
    # ------------------------------------------------------------------

    def test_logs_again_after_interval(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        assert rl.should_log(now=60.0, wait_bucket="2h") is True

    def test_logs_again_just_over_interval(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        assert rl.should_log(now=60.001, wait_bucket="2h") is True

    def test_interval_is_relative_to_last_log(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")    # logged at t=0
        rl.should_log(now=60.0, wait_bucket="2h")   # logged at t=60
        assert rl.should_log(now=119.9, wait_bucket="2h") is False
        assert rl.should_log(now=120.0, wait_bucket="2h") is True

    # ------------------------------------------------------------------
    # Bucket change forces a log even before interval
    # ------------------------------------------------------------------

    def test_bucket_change_logs_immediately(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        # Only 1 second later but bucket changed
        assert rl.should_log(now=1.0, wait_bucket="1h") is True

    def test_bucket_change_updates_last_log_time(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        rl.should_log(now=1.0, wait_bucket="1h")   # logged; reset timer
        # Still within interval from t=1
        assert rl.should_log(now=30.0, wait_bucket="1h") is False

    # ------------------------------------------------------------------
    # Exiting paced state resets; re-entering logs again
    # ------------------------------------------------------------------

    def test_reset_clears_state(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        rl.reset()
        # After reset, next paced call should log as if fresh
        assert rl.should_log(now=1.0, wait_bucket="2h") is True

    def test_reset_then_immediate_repeat_suppressed(self):
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="2h")
        rl.reset()
        rl.should_log(now=1.0, wait_bucket="2h")  # first after reset
        assert rl.should_log(now=1.0, wait_bucket="2h") is False

    def test_paced_resume_paced_re_enters_correctly(self):
        """Simulate: paced → not-paced (reset) → paced again."""
        rl = self._make(interval=60.0)
        rl.should_log(now=0.0, wait_bucket="30m")   # enter paced
        rl.reset()                                   # mining resumed
        assert rl.should_log(now=10.0, wait_bucket="30m") is True  # re-enter paced


# ---------------------------------------------------------------------------
# _SetupAbortThrottle tests (base_miner work-tag rate-limiting)
# ---------------------------------------------------------------------------

class TestSetupAbortThrottle:
    """Tests for the base_miner _pre_mine_setup-False log throttle."""

    def _make(self, max_tags: int = 32) -> _SetupAbortThrottle:
        return _SetupAbortThrottle(max_tags=max_tags)

    def test_first_call_for_tag_logs(self):
        t = self._make()
        assert t.should_log("block=0xdeadbeef") is True

    def test_repeat_for_same_tag_suppressed(self):
        t = self._make()
        t.should_log("block=0xdeadbeef")
        assert t.should_log("block=0xdeadbeef") is False

    def test_new_tag_logs(self):
        t = self._make()
        t.should_log("block=0xdeadbeef")
        assert t.should_log("block=0xcafebabe") is True

    def test_multiple_tags_independent(self):
        t = self._make()
        assert t.should_log("tag-A") is True
        assert t.should_log("tag-B") is True
        assert t.should_log("tag-A") is False
        assert t.should_log("tag-B") is False

    # ------------------------------------------------------------------
    # Bounded size — oldest entry evicted when cap is reached
    # ------------------------------------------------------------------

    def test_bounded_at_max_tags(self):
        t = self._make(max_tags=3)
        for i in range(3):
            assert t.should_log(f"tag-{i}") is True
        # Adding a 4th tag should evict the oldest (tag-0)
        assert t.should_log("tag-3") is True
        # tag-0 was evicted, so it should log again
        assert t.should_log("tag-0") is True
        # Internal size never exceeds max_tags
        assert len(t) <= 3

    def test_size_never_exceeds_max(self):
        t = self._make(max_tags=5)
        for i in range(20):
            t.should_log(f"tag-{i}")
        assert len(t) <= 5

    def test_empty_throttle_has_zero_length(self):
        t = self._make()
        assert len(t) == 0
