# SPDX-License-Identifier: AGPL-3.0-or-later
from substrate.decay_timing import TimingTracker


def test_interval_and_lag_emas_from_timestamped_heads():
    t = TimingTracker(fallback_interval_s=6.0, lag_min_s=0.0, lag_max_s=12.0,
                      ema_alpha=0.5)
    t.observe_head(block_number=100, chain_ts_s=1000.0, monotonic_now=50.0,
                   wallclock_now=1002.0)
    assert t.interval_s == 6.0
    assert t.lag_s == 2.0
    t.observe_head(block_number=101, chain_ts_s=1006.2, monotonic_now=58.4,
                   wallclock_now=1008.4)
    assert 6.0 <= t.interval_s <= 6.3
    assert 2.0 <= t.lag_s <= 2.3


def test_lag_clamped_to_bounds():
    t = TimingTracker(fallback_interval_s=6.0, lag_min_s=0.0, lag_max_s=3.0,
                      ema_alpha=1.0)
    t.observe_head(block_number=1, chain_ts_s=0.0, monotonic_now=0.0,
                   wallclock_now=100.0)
    assert t.lag_s == 3.0
    t.observe_head(block_number=2, chain_ts_s=10.0, monotonic_now=6.0,
                   wallclock_now=5.0)
    assert t.lag_s == 0.0


def test_deadline_for_block_is_monotonic_anchored():
    t = TimingTracker(fallback_interval_s=6.0, lag_min_s=0.0, lag_max_s=12.0,
                      ema_alpha=1.0)
    t.observe_head(block_number=100, chain_ts_s=1000.0, monotonic_now=50.0,
                   wallclock_now=1000.0)
    assert t.fire_deadline_monotonic(b_star=105, now_monotonic=50.0) == 80.0


def test_deadline_none_without_anchor():
    t = TimingTracker(fallback_interval_s=6.0, lag_min_s=0.0, lag_max_s=12.0)
    assert t.fire_deadline_monotonic(b_star=105, now_monotonic=0.0) is None


def test_estimate_block_none_before_anchor():
    t = TimingTracker()
    assert t.estimate_block(now_monotonic=100.0) is None


def test_estimate_block_advances_with_interval():
    t = TimingTracker()
    t.observe_head(block_number=10, chain_ts_s=1000.0,
                   monotonic_now=500.0, wallclock_now=1000.0)
    t.observe_head(block_number=11, chain_ts_s=1006.0,
                   monotonic_now=506.0, wallclock_now=1006.0)
    # Anchor is block 11 @ monotonic 506, interval ~6s. +12s => +2 blocks => 13.
    assert t.estimate_block(now_monotonic=518.0) == 13
    # Exactly at the anchor returns the anchor block.
    assert t.estimate_block(now_monotonic=506.0) == 11
    # Before the anchor monotonic clamps to the anchor block (0 elapsed).
    assert t.estimate_block(now_monotonic=500.0) == 11
