# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for the ``SyncResult`` dataclass."""

from shared.block_synchronizer import SyncResult


def test_success_summary():
    result = SyncResult(
        success=True, requested=10, downloaded=10, elapsed=1.5,
    )
    assert "Synced 10/10" in result.summary()
    assert "1.5s" in result.summary()


def test_failure_with_failed_block():
    result = SyncResult(
        success=False, requested=10, downloaded=3,
        failed_block=4, elapsed=2.0,
    )
    summary = result.summary()
    assert "failed at block 4" in summary
    assert "3/10" in summary


def test_incomplete_without_failed_block():
    """Timeout with partial progress returns success=False, no failed_block."""
    result = SyncResult(
        success=False, requested=10, downloaded=5, elapsed=30.0,
    )
    summary = result.summary()
    assert "incomplete" in summary
    assert "5/10" in summary
    # Must not falsely claim success.
    assert result.success is False


def test_no_progress_failure():
    """Timeout with zero progress is a failure with failed_block set."""
    result = SyncResult(
        success=False, requested=10, downloaded=0,
        failed_block=0, elapsed=30.0,
    )
    assert result.success is False
    assert "failed at block 0" in result.summary()


def test_empty_range_is_success():
    """A zero-length sync range returns success with zero counts."""
    result = SyncResult(success=True)
    assert result.requested == 0
    assert result.downloaded == 0
    assert result.failed_block is None
