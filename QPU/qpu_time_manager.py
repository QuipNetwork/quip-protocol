"""QPU time budget management for controlling D-Wave QPU usage.

This module provides a **carry-over budget reservoir** that paces D-Wave QPU
usage. A pool of QPU-access-time budget accrues continuously at
``daily_budget / 86400`` per wall-second (unused budget banks across days,
clamped to ``budget_cap``) and is spent by ``record_block_time``.

Mining uses **start/continue hysteresis**: the miner stays idle until the pool
accrues to ``min_block_budget`` (the buffer), then bursts — continuing to mine
until the pool drains to 0 — at which point it idles and re-accumulates the full
buffer. This turns the old per-block dribble into accumulate-then-burst, which
amortizes D-Wave queue warmup on throughput-bound hardware.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


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


def resolve_initial_budget(
    raw: str,
    daily_budget_seconds: float,
    min_block_budget_seconds: float,
    budget_cap_seconds: Optional[float] = None,
) -> float:
    """Resolve an operator-facing initial-budget setting to seconds.

    Maps the three mental-model keywords plus explicit durations onto a seed
    value for the reservoir pool. The result is *not* clamped here; the manager
    clamps it to the pool cap at construction.

    Args:
        raw: ``"min"`` (one buffer), ``"daily"`` (a full day's budget),
            ``"cap"`` (fill the pool cap), or a duration string like ``"600s"``
            / ``"10m"`` (or raw seconds).
        daily_budget_seconds: Configured daily budget, for ``"daily"``.
        min_block_budget_seconds: Reservoir buffer, for ``"min"``.
        budget_cap_seconds: Configured cap, for ``"cap"``. ``None`` mirrors the
            manager's derived cap of ``max(daily, min_block)``.

    Returns:
        Seed value in QPU access seconds.

    Raises:
        ValueError: If ``raw`` is neither a known keyword nor a parseable
            duration.
    """
    key = raw.strip().lower()
    if key == "min":
        return min_block_budget_seconds
    if key == "daily":
        return daily_budget_seconds
    if key == "cap":
        if budget_cap_seconds is not None:
            return budget_cap_seconds
        return max(daily_budget_seconds, min_block_budget_seconds)
    return parse_duration(key)


@dataclass
class QPUTimeConfig:
    """Configuration for QPU time budget management.

    The daily budget approach allows users to manually calculate their allocation.
    For example, with 20 minutes/month allocation:
        qpu_daily_budget = "40s"  (20 min / 30 days ≈ 40 seconds/day)
    """

    daily_budget_seconds: float
    """Daily QPU time budget in seconds. The reservoir accrues at this rate
    (``daily_budget_seconds / 86400`` per wall-second)."""

    min_blocks_for_estimation: int = 5
    """Minimum blocks mined before EMA estimation kicks in (default: 5)."""

    ema_alpha: float = 0.3
    """EMA decay factor (0.0-1.0). Higher = more weight to recent blocks."""

    min_block_budget_seconds: float = 90.0
    """Reservoir buffer (QPU access seconds) that must accrue before a burst
    starts. Once mining, the burst continues until the pool drains to 0
    (start/continue hysteresis)."""

    budget_cap_seconds: Optional[float] = None
    """Max banked pool size (seconds). ``None`` => ``max(daily_budget_seconds,
    min_block_budget_seconds)`` so the buffer is always reachable and a
    post-downtime catch-up burst is bounded to one cap's worth of QPU."""

    initial_budget_seconds: Optional[float] = None
    """Pool balance to seed at process start (QPU access seconds). ``None``
    starts dry (the historical behavior) and waits to accrue the buffer before
    the first burst. Seeding to ``min_block_budget_seconds`` lets a fresh
    process mine one burst immediately instead of waiting hours-to-days for the
    pool to fill at ``daily_budget / 86400``. Clamped to the pool cap. This is
    a per-process grant, not persisted accounting: frequent restarts re-grant
    it, so keep the seed small (a buffer's worth) unless overspend is acceptable."""


@dataclass
class QPUTimeEstimate:
    """Result of a reservoir mining decision."""

    estimated_block_time_us: float
    """Estimated microseconds needed for the next block (EMA-based)."""

    cumulative_used_us: float
    """Total QPU microseconds spent since process start (lifetime)."""

    pool_us: float
    """Current reservoir pool balance in microseconds."""

    pool_cap_us: float
    """Maximum the pool can bank, in microseconds."""

    burst_active: bool
    """True if currently in a burst (mining down toward 0); False if idle and
    re-accumulating to the buffer."""

    should_mine: bool
    """True if there is sufficient budget to mine the next block."""

    confidence: str
    """Confidence level: 'low', 'medium', or 'high' based on sample count."""

    seconds_until_can_mine: float
    """Seconds until the pool accrues to the buffer (0 if mining now)."""


class QPUTimeManager:
    """Manages a carry-over QPU budget reservoir and mining decisions."""

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

        # Reservoir state.
        self._pool_us: float = 0.0
        self._burst_active: bool = False
        self._last_accrual_s: float = time.time()
        cap_s = config.budget_cap_seconds
        if cap_s is None:
            cap_s = max(config.daily_budget_seconds, config.min_block_budget_seconds)
        self._pool_cap_us: float = cap_s * 1_000_000
        self._accrual_rate_us_per_s: float = (
            config.daily_budget_seconds * 1_000_000 / 86400.0
        )

        # Seed the reservoir so a fresh process can mine without waiting to
        # accrue the buffer. Clamped to [0, cap]; ``None`` starts dry.
        if config.initial_budget_seconds is not None:
            self._pool_us = min(
                self._pool_cap_us,
                max(0.0, config.initial_budget_seconds * 1_000_000),
            )

        if config.min_block_budget_seconds > config.daily_budget_seconds > 0:
            eta_days = config.min_block_budget_seconds / config.daily_budget_seconds
            logger.warning(
                "min_block_budget (%.0fs) exceeds daily_budget (%.0fs): the "
                "reservoir fills only across ~%.1f days before the first burst.",
                config.min_block_budget_seconds,
                config.daily_budget_seconds,
                eta_days,
            )

    def reset_clock(self, now: float) -> None:
        """Test seam: pin the accrual clock to a fixed wall time."""
        self._last_accrual_s = now

    def _accrue(self, now: float) -> None:
        """Add elapsed accrual to the pool and clamp to the cap.

        Args:
            now: Wall-clock seconds; accrual since the last update is banked.
        """
        elapsed = now - self._last_accrual_s
        if elapsed > 0:
            self._pool_us = min(
                self._pool_cap_us,
                self._pool_us + self._accrual_rate_us_per_s * elapsed,
            )
            self._last_accrual_s = now

    def record_block_time(
        self, qpu_access_time_us: float, now: Optional[float] = None,
    ) -> None:
        """Record QPU time used for a completed block and debit the pool.

        This should be called after each successful mining result is processed,
        passing the total QPU access time from the sampleset timing info.

        Args:
            qpu_access_time_us: QPU access time in microseconds
                (qpu_programming_time + qpu_sampling_time from D-Wave response)
            now: Optional wall-clock timestamp for the accrual step; uses
                current time if not provided (test seam).
        """
        self._accrue(now if now is not None else time.time())
        self._pool_us -= qpu_access_time_us
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

    def _seconds_until_buffer(self) -> float:
        """Seconds for the pool to accrue back up to the min-block buffer.

        Caller-owned zero-guards (already-bursting / pool already at buffer)
        live at the call sites; this is only the non-trivial accrual branch.
        Assumes ``_accrue`` has already advanced ``_pool_us``.
        """
        deficit_us = max(
            0.0, self.config.min_block_budget_seconds * 1_000_000 - self._pool_us
        )
        rate = self._accrual_rate_us_per_s
        return deficit_us / rate if rate > 0 else float("inf")

    def should_mine_block(self, now: Optional[float] = None) -> QPUTimeEstimate:
        """Decide whether to mine, using start/continue reservoir hysteresis.

        - **Idle** (no active burst): mine only once the pool has accrued to
          ``min_block_budget`` (the buffer high-water mark).
        - **Bursting**: keep mining while the pool is above 0 (low-water mark),
          even below the buffer, so a started burst runs to completion.

        Draining to ``pool <= 0`` while bursting clears the burst flag, so the
        next idle decision requires the full buffer again.

        Args:
            now: Optional wall-clock timestamp; uses current time if not provided.

        Returns:
            QPUTimeEstimate with decision and supporting metrics.
        """
        now = now if now is not None else time.time()
        self._accrue(now)
        estimated_time = self.estimate_next_block_time()
        buffer_us = self.config.min_block_budget_seconds * 1_000_000
        pool_us = self._pool_us

        if self._burst_active:
            should_mine = pool_us > 0.0
        else:
            should_mine = pool_us >= buffer_us
        self._burst_active = should_mine

        if should_mine:
            seconds_until_can_mine = 0.0
        else:
            seconds_until_can_mine = self._seconds_until_buffer()
            self.blocks_skipped += 1

        n = len(self.block_times_us)
        if n < self.config.min_blocks_for_estimation:
            confidence = "low"
        elif n < self.config.min_blocks_for_estimation * 2:
            confidence = "medium"
        else:
            confidence = "high"

        return QPUTimeEstimate(
            estimated_block_time_us=estimated_time,
            cumulative_used_us=self.cumulative_used_us,
            pool_us=pool_us,
            pool_cap_us=self._pool_cap_us,
            burst_active=self._burst_active,
            should_mine=should_mine,
            confidence=confidence,
            seconds_until_can_mine=seconds_until_can_mine,
        )

    def end_burst(self) -> None:
        """Force re-accumulation: the next idle gate requires the full buffer.

        Called by the in-loop stop gate when the pool drains to 0, so a brief
        accrual back above 0 does not immediately restart a micro-burst.
        """
        self._burst_active = False

    def get_stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Return current reservoir statistics.

        Args:
            now: Optional wall-clock timestamp for the accrual step; uses
                current time if not provided. All callers on a single manager
                instance must share one clock domain (wall-clock) — the gates
                and ``record_block_time`` advance the same ``_last_accrual_s``.

        Returns:
            Dictionary with pool balance, cap, buffer, burst state, and
            estimation metrics. ``budget_remaining_seconds`` mirrors
            ``pool_seconds`` (the in-loop drain-to-0 gate reads it).
        """
        now = now if now is not None else time.time()
        self._accrue(now)
        pool_s = self._pool_us / 1_000_000
        buffer_s = self.config.min_block_budget_seconds
        if self._burst_active or self._pool_us >= buffer_s * 1_000_000:
            seconds_until_buffer = 0.0
        else:
            seconds_until_buffer = self._seconds_until_buffer()

        return {
            "daily_budget_seconds": self.config.daily_budget_seconds,
            "pool_seconds": pool_s,
            "budget_remaining_seconds": pool_s,
            "pool_cap_seconds": self._pool_cap_us / 1_000_000,
            "min_block_budget_seconds": buffer_s,
            "burst_active": self._burst_active,
            "seconds_until_buffer": seconds_until_buffer,
            "cumulative_used_seconds": self.cumulative_used_us / 1_000_000,
            "blocks_mined": self.blocks_mined,
            "blocks_skipped": self.blocks_skipped,
            "ema_estimate_seconds": (
                self.ema_estimate_us / 1_000_000 if self.ema_estimate_us else None
            ),
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
        self._pool_us = 0.0
        self._burst_active = False
        self._last_accrual_s = time.time()
