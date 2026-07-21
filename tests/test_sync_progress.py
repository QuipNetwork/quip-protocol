"""Tests for substrate.sync_progress.SyncProgress — pure rate/ETA logic."""
from __future__ import annotations

from substrate.sync_progress import SyncProgress


def _state(current: int, highest: int, peers: int = 3) -> dict:
    return {"current_block": current, "highest_block": highest, "peers": peers}


def test_first_sample_reports_calculating():
    """With one sample there is no rate yet — say so instead of guessing."""
    progress = SyncProgress()
    line = progress.observe(_state(1_000, 31_000), now_s=0.0)
    assert "block 1,000 / 31,000" in line
    assert "3.2%" in line
    assert "calculating" in line


def test_two_samples_produce_rate_and_eta():
    """1000 blocks in 10s → 100 blk/s; 30000 remaining → 5m 0s."""
    progress = SyncProgress()
    progress.observe(_state(0, 30_000), now_s=0.0)
    line = progress.observe(_state(1_000, 31_000), now_s=10.0)
    assert progress.rate_blocks_per_s() == 100.0
    assert "~5m 0s remaining" in line
    assert "100 blk/s" in line
    assert "peers=3" in line


def test_zero_rate_reports_stalled():
    """Node connected but not advancing — surface the stall, keep waiting."""
    progress = SyncProgress()
    progress.observe(_state(5_000, 30_000), now_s=0.0)
    line = progress.observe(_state(5_000, 30_000), now_s=10.0)
    assert "stalled" in line


def test_zero_peers_reports_stalled():
    """0 peers can never finish syncing — call it out even with no rate yet."""
    progress = SyncProgress()
    line = progress.observe(_state(5_000, 30_000, peers=0), now_s=0.0)
    assert "stalled" in line
    assert "0 peers" in line


def test_zero_highest_block_does_not_divide_by_zero():
    progress = SyncProgress()
    line = progress.observe(_state(0, 0), now_s=0.0)
    assert "0.0%" in line


def test_window_slides_so_rate_tracks_recent_speed():
    """Old samples fall out of the window; the rate reflects the recent pace."""
    progress = SyncProgress(window=3)
    progress.observe(_state(0, 100_000), now_s=0.0)      # will be evicted
    progress.observe(_state(10, 100_000), now_s=10.0)
    progress.observe(_state(20, 100_000), now_s=20.0)
    progress.observe(_state(1_020, 100_000), now_s=30.0)
    # Window now spans t=10..30: (1020-10)/20 = 50.5 blk/s
    assert progress.rate_blocks_per_s() == 50.5


def test_duration_formats():
    """Cover the s / m+s / h+m formatting branches via the log line."""
    progress = SyncProgress()
    progress.observe(_state(0, 4_500), now_s=0.0)
    # 100 blk/s, 3500 remaining → 35s
    assert "~35s remaining" in progress.observe(_state(1_000, 4_500), now_s=10.0)

    progress2 = SyncProgress()
    progress2.observe(_state(0, 811_000), now_s=0.0)
    # 100 blk/s, 810000 remaining → 8100s → 2h 15m
    assert "~2h 15m remaining" in progress2.observe(_state(1_000, 811_000), now_s=10.0)


def test_slow_rate_shows_one_decimal():
    """Rates under 10 blk/s print one decimal so 0.5 doesn't render as 0."""
    progress = SyncProgress()
    progress.observe(_state(0, 10_000), now_s=0.0)
    line = progress.observe(_state(5, 10_000), now_s=10.0)
    assert "0.5 blk/s" in line
