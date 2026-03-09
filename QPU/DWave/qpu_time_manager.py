"""QPU time budget management for controlling D-Wave QPU usage.

This module provides time budget tracking and estimation to prevent
excessive QPU consumption during mining operations. Uses proportional
pacing to spread usage evenly across the day.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


def parse_duration(duration_str: str) -> float:
    """Parse duration string to seconds.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 30.0 (seconds)
        "5m" -> 300.0
        "2h" -> 7200.0
        "1d" -> 86400.0
        "1w" -> 604800.0

    Args:
        duration_str: Duration string like "30s", "5m", "2h", "1d", "1w"

    Returns:
        Duration in seconds as float

    Raises:
        ValueError: If the duration string cannot be parsed
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60.0
    elif duration_str.endswith('h'):
        return float(duration_str[:-1]) * 3600.0
    elif duration_str.endswith('d'):
        return float(duration_str[:-1]) * 86400.0
    elif duration_str.endswith('w'):
        return float(duration_str[:-1]) * 604800.0
    else:
        # Try parsing as raw seconds
        return float(duration_str)


@dataclass
class QPUTimeConfig:
    """Configuration for QPU time budget management.

    The daily budget approach allows users to manually calculate their allocation.
    For example, with 20 minutes/month allocation:
        qpu_daily_budget = "40s"  (20 min / 30 days ≈ 40 seconds/day)
    """

    daily_budget_seconds: float
    """Daily QPU time budget in seconds. Mining stops when cumulative usage exceeds this."""

    min_blocks_for_estimation: int = 5
    """Minimum blocks mined before EMA estimation kicks in (default: 5)."""

    ema_alpha: float = 0.3
    """EMA decay factor (0.0-1.0). Higher = more weight to recent blocks."""


@dataclass
class QPUTimeEstimate:
    """Result of QPU time estimation for mining decision."""

    estimated_block_time_us: float
    """Estimated microseconds needed for the next block."""

    cumulative_used_us: float
    """Total QPU microseconds used so far today."""

    daily_budget_us: float
    """Daily budget in microseconds."""

    proportional_limit_us: float
    """Current proportional limit based on time of day (elapsed_fraction * daily_budget)."""

    budget_remaining_us: float
    """Budget remaining until proportional limit (proportional_limit - cumulative_used)."""

    should_mine: bool
    """True if there's sufficient time to mine the next block."""

    confidence: str
    """Confidence level: 'low', 'medium', or 'high' based on sample count."""

    elapsed_fraction: float
    """Fraction of day elapsed since UTC midnight (0.0-1.0)."""

    seconds_until_can_mine: float
    """Seconds until enough headroom accumulates (0 if can mine now). May extend past midnight."""

    is_pacing_limited: bool
    """True if blocked due to pacing, False if can mine. Always True when should_mine is False."""


class QPUTimeManager:
    """Manages QPU time budget and provides mining decisions based on usage estimates."""

    def __init__(self, config: QPUTimeConfig):
        """Initialize the time manager with budget configuration.

        Args:
            config: QPUTimeConfig with budget and estimation parameters
        """
        self.config = config
        self.block_times_us: List[float] = []
        self.cumulative_used_us: float = 0.0
        self.ema_estimate_us: Optional[float] = None
        self.blocks_mined: int = 0
        self.blocks_skipped: int = 0
        self.day_start_timestamp: float = self._calculate_day_start()

    def _calculate_day_start(self, now: Optional[float] = None) -> float:
        """Calculate Unix timestamp for start of current day (UTC midnight).

        Args:
            now: Optional Unix timestamp; uses current time if not provided

        Returns:
            Unix timestamp for UTC midnight of the current day
        """
        now = now or time.time()
        utc_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        day_start_dt = utc_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start_dt.timestamp()

    def _check_day_rollover(self, now: Optional[float] = None) -> bool:
        """Reset counters if day has changed (UTC midnight rollover).

        Args:
            now: Optional Unix timestamp; uses current time if not provided

        Returns:
            True if reset occurred, False otherwise
        """
        now = now or time.time()
        current_day_start = self._calculate_day_start(now)
        if current_day_start > self.day_start_timestamp:
            self.day_start_timestamp = current_day_start
            self.cumulative_used_us = 0.0
            self.blocks_mined = 0
            self.blocks_skipped = 0
            # Keep EMA estimate and block_times_us for estimation continuity
            return True
        return False

    def record_block_time(self, qpu_access_time_us: float) -> None:
        """Record QPU time used for a completed block.

        This should be called after each successful mining result is processed,
        passing the total QPU access time from the sampleset timing info.

        Args:
            qpu_access_time_us: QPU access time in microseconds
                (qpu_programming_time + qpu_sampling_time from D-Wave response)
        """
        self.block_times_us.append(qpu_access_time_us)
        self.cumulative_used_us += qpu_access_time_us
        self.blocks_mined += 1

        # Update EMA if we have enough samples
        if len(self.block_times_us) >= self.config.min_blocks_for_estimation:
            if self.ema_estimate_us is None:
                # Initialize EMA with average of recorded times
                self.ema_estimate_us = sum(self.block_times_us) / len(self.block_times_us)
            else:
                # Update EMA: new_ema = alpha * latest + (1 - alpha) * old_ema
                alpha = self.config.ema_alpha
                self.ema_estimate_us = alpha * qpu_access_time_us + (1 - alpha) * self.ema_estimate_us

    def estimate_next_block_time(self) -> float:
        """Estimate QPU time needed for the next block using EMA.

        Returns:
            Estimated microseconds for the next block with safety margin
        """
        if len(self.block_times_us) == 0:
            # No data: use conservative default estimate (10ms = 10,000 us)
            # This is typical for a single QPU job with modest reads
            return 10_000.0

        if len(self.block_times_us) < self.config.min_blocks_for_estimation:
            # Not enough data for EMA: use maximum observed + 50% safety margin
            return max(self.block_times_us) * 1.5

        # Use EMA estimate with 20% safety margin
        if self.ema_estimate_us is not None:
            return self.ema_estimate_us * 1.2
        else:
            # Fallback if EMA wasn't computed (shouldn't happen)
            return (sum(self.block_times_us) / len(self.block_times_us)) * 1.2

    def should_mine_block(self, now: Optional[float] = None) -> QPUTimeEstimate:
        """Determine if there's enough budget to mine the next block.

        Uses proportional pacing: at any point in the day, usage cannot exceed
        (elapsed_time / 24h) * daily_budget. This spreads QPU usage evenly
        across the day instead of exhausting the budget early.

        If usage exceeds the current proportional limit, mining is paused until
        the limit catches up (continuous pacing).

        Args:
            now: Optional Unix timestamp; uses current time if not provided

        Returns:
            QPUTimeEstimate with decision and supporting metrics
        """
        # Check for day rollover (resets counters at UTC midnight)
        self._check_day_rollover(now)

        now = now or time.time()
        estimated_time = self.estimate_next_block_time()
        daily_budget_us = self.config.daily_budget_seconds * 1_000_000

        # Calculate proportional limit based on time of day
        elapsed_today = now - self.day_start_timestamp
        elapsed_fraction = min(elapsed_today / 86400.0, 1.0)
        proportional_limit_us = daily_budget_us * elapsed_fraction

        # Budget remaining is based on proportional limit, not daily budget
        budget_remaining = proportional_limit_us - self.cumulative_used_us

        # Can mine if: cumulative + estimated <= proportional_limit
        needed_us = self.cumulative_used_us + estimated_time
        should_mine = needed_us <= proportional_limit_us

        # Calculate when we can mine again (continuous pacing)
        # Use last block time for headroom calculation to ensure we accumulate
        # enough time for a similar block before trying again
        if should_mine:
            seconds_until_can_mine = 0.0
            is_pacing_limited = False
        else:
            # Use last block time for headroom, fallback to estimate if no blocks yet
            headroom_needed = self.block_times_us[-1] if self.block_times_us else estimated_time
            wait_for_us = self.cumulative_used_us + headroom_needed

            if wait_for_us <= daily_budget_us:
                # Can mine today - calculate when proportional limit catches up
                elapsed_needed = (wait_for_us * 86400.0) / daily_budget_us
                can_mine_at = self.day_start_timestamp + elapsed_needed
                seconds_until_can_mine = max(0.0, can_mine_at - now)
            else:
                # Need to wait past midnight - cumulative resets to 0
                # Then wait for proportional_limit >= headroom_needed
                next_day_start = self.day_start_timestamp + 86400.0
                seconds_until_midnight = max(0.0, next_day_start - now)
                # After midnight reset, need proportional_limit >= headroom_needed
                elapsed_needed_tomorrow = (headroom_needed * 86400.0) / daily_budget_us
                seconds_until_can_mine = seconds_until_midnight + elapsed_needed_tomorrow

            is_pacing_limited = True

        # Determine confidence based on sample count
        n = len(self.block_times_us)
        if n < self.config.min_blocks_for_estimation:
            confidence = "low"
        elif n < self.config.min_blocks_for_estimation * 2:
            confidence = "medium"
        else:
            confidence = "high"

        if not should_mine:
            self.blocks_skipped += 1

        return QPUTimeEstimate(
            estimated_block_time_us=estimated_time,
            cumulative_used_us=self.cumulative_used_us,
            daily_budget_us=daily_budget_us,
            proportional_limit_us=proportional_limit_us,
            budget_remaining_us=max(0, budget_remaining),
            should_mine=should_mine,
            confidence=confidence,
            elapsed_fraction=elapsed_fraction,
            seconds_until_can_mine=seconds_until_can_mine,
            is_pacing_limited=is_pacing_limited,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return current time management statistics.

        Returns:
            Dictionary with budget status, usage, and estimation metrics
        """
        now = time.time()
        daily_budget_us = self.config.daily_budget_seconds * 1_000_000

        # Calculate proportional pacing info
        elapsed_today = now - self.day_start_timestamp
        elapsed_fraction = min(elapsed_today / 86400.0, 1.0)
        proportional_limit_us = daily_budget_us * elapsed_fraction
        budget_remaining = proportional_limit_us - self.cumulative_used_us

        return {
            "daily_budget_seconds": self.config.daily_budget_seconds,
            "cumulative_used_seconds": self.cumulative_used_us / 1_000_000,
            "proportional_limit_seconds": proportional_limit_us / 1_000_000,
            "budget_remaining_seconds": max(0, budget_remaining) / 1_000_000,
            "elapsed_fraction": elapsed_fraction,
            "blocks_mined": self.blocks_mined,
            "blocks_skipped": self.blocks_skipped,
            "ema_estimate_seconds": self.ema_estimate_us / 1_000_000 if self.ema_estimate_us else None,
            "block_times_count": len(self.block_times_us),
            "avg_block_time_seconds": (
                (sum(self.block_times_us) / len(self.block_times_us) / 1_000_000)
                if self.block_times_us else None
            ),
        }

    def reset(self) -> None:
        """Reset usage tracking (e.g., at start of new billing period)."""
        self.block_times_us.clear()
        self.cumulative_used_us = 0.0
        self.ema_estimate_us = None
        self.blocks_mined = 0
        self.blocks_skipped = 0
        self.day_start_timestamp = self._calculate_day_start()
