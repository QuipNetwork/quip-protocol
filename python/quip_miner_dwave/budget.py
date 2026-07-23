"""QPU budget reservoir: accumulate-then-burst pacing (ported from v0.2).

A pool of QPU-access-time budget accrues at ``daily_budget / 86400`` per
wall-second and is spent by :meth:`QPUTimeManager.record_access_time`. Mining
uses start/continue hysteresis: idle until the pool reaches
``min_block_budget``, then burst until the pool drains to 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from quip_miner_dwave.config import warn_unknown_fields

# Config keys the dwave backend recognizes in Configure.backend_toml. Anything
# else (outside SESSION_KEYS) is a typo and gets warned about, uniform with the
# Rust backends' unknown-field handling. Connection credentials (token/solver/
# region) are NOT here: they come from D-Wave's own config (dwave.conf + env),
# not the coordinator.
DWAVE_CONFIG_KEYS = frozenset(
    {
        "daily_budget",
        "daily_budget_seconds",
        "min_block_budget",
        "min_block_budget_seconds",
        "budget_cap",
        "budget_cap_seconds",
        "anneal_time_us",
        "num_reads",
    }
)


def warn_unknown_backend_keys(toml_text: str) -> None:
    """Parse ``Configure.backend_toml`` and warn on keys outside the dwave
    schema, matching the Rust backends' unknown-field warnings."""
    if not toml_text or not toml_text.strip():
        return
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py3.10
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(toml_text)
    except Exception:
        return
    warn_unknown_fields("dwave", data.keys(), DWAVE_CONFIG_KEYS)


def parse_duration(duration_str: str) -> float:
    """Parse a duration string (``30s``, ``5m``, ``2h``, ``1d``, ``1w``) to seconds."""
    s = duration_str.strip().lower()
    if s.endswith("s") and not s.endswith("ms"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 3600.0
    if s.endswith("d"):
        return float(s[:-1]) * 86400.0
    if s.endswith("w"):
        return float(s[:-1]) * 604800.0
    return float(s)


@dataclass
class QPUTimeConfig:
    """Reservoir configuration."""

    daily_budget_seconds: float
    min_block_budget_seconds: float = 90.0
    budget_cap_seconds: Optional[float] = None
    initial_budget_seconds: Optional[float] = None
    min_blocks_for_estimation: int = 5
    ema_alpha: float = 0.3


@dataclass
class QPUTimeEstimate:
    """Result of a reservoir mining decision."""

    should_mine: bool
    pool_us: float
    burst_active: bool
    seconds_until_can_mine: float
    estimated_block_time_us: float


class QPUTimeManager:
    """Carry-over QPU budget reservoir with start/continue hysteresis."""

    def __init__(self, config: QPUTimeConfig):
        self.config = config
        self.block_times_us: list[float] = []
        self.cumulative_used_us: float = 0.0
        self.ema_estimate_us: Optional[float] = None
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
        if config.initial_budget_seconds is not None:
            self._pool_us = min(
                self._pool_cap_us,
                max(0.0, config.initial_budget_seconds * 1_000_000),
            )

    def reset_clock(self, now: float) -> None:
        """Test seam: pin the accrual clock."""
        self._last_accrual_s = now

    def _accrue(self, now: float) -> None:
        elapsed = now - self._last_accrual_s
        if elapsed > 0:
            self._pool_us = min(
                self._pool_cap_us,
                self._pool_us + self._accrual_rate_us_per_s * elapsed,
            )
            self._last_accrual_s = now

    def record_access_time(
        self, qpu_access_time_us: float, now: Optional[float] = None
    ) -> None:
        """Debit the pool after a completed QPU job."""
        self._accrue(now if now is not None else time.time())
        self._pool_us -= qpu_access_time_us
        self.block_times_us.append(qpu_access_time_us)
        self.cumulative_used_us += qpu_access_time_us
        n = len(self.block_times_us)
        if n >= self.config.min_blocks_for_estimation:
            if self.ema_estimate_us is None:
                self.ema_estimate_us = sum(self.block_times_us) / n
            else:
                a = self.config.ema_alpha
                self.ema_estimate_us = (
                    a * qpu_access_time_us + (1 - a) * self.ema_estimate_us
                )

    def estimate_next_block_time(self) -> float:
        if not self.block_times_us:
            return 10_000.0
        if len(self.block_times_us) < self.config.min_blocks_for_estimation:
            return max(self.block_times_us) * 1.5
        if self.ema_estimate_us is not None:
            return self.ema_estimate_us * 1.2
        return (sum(self.block_times_us) / len(self.block_times_us)) * 1.2

    def should_mine(self, now: Optional[float] = None) -> QPUTimeEstimate:
        """Decide whether budget allows sampling (start/continue hysteresis)."""
        now = now if now is not None else time.time()
        self._accrue(now)
        estimated = self.estimate_next_block_time()
        buffer_us = self.config.min_block_budget_seconds * 1_000_000
        pool_us = self._pool_us

        if self._burst_active:
            may = pool_us > 0.0
        else:
            may = pool_us >= buffer_us
        self._burst_active = may

        if may:
            until = 0.0
        else:
            deficit = max(0.0, buffer_us - pool_us)
            rate = self._accrual_rate_us_per_s
            until = deficit / rate if rate > 0 else float("inf")

        return QPUTimeEstimate(
            should_mine=may,
            pool_us=pool_us,
            burst_active=self._burst_active,
            seconds_until_can_mine=until,
            estimated_block_time_us=estimated,
        )

    def end_burst(self) -> None:
        """Force re-accumulation to the full buffer."""
        self._burst_active = False

    def get_stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now if now is not None else time.time()
        self._accrue(now)
        return {
            "pool_seconds": self._pool_us / 1_000_000,
            "burst_active": self._burst_active,
            "daily_budget_seconds": self.config.daily_budget_seconds,
            "min_block_budget_seconds": self.config.min_block_budget_seconds,
            "cumulative_used_seconds": self.cumulative_used_us / 1_000_000,
        }


def budget_from_backend_toml(toml_text: str) -> Optional[QPUTimeManager]:
    """Build a manager from Configure.backend_toml, or None if no budget keys."""
    if not toml_text or not toml_text.strip():
        return None
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py3.10
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(toml_text)
    except Exception:
        return None

    raw = data.get("daily_budget") or data.get("daily_budget_seconds")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        daily = float(raw)
    else:
        daily = parse_duration(str(raw))

    min_block = data.get("min_block_budget") or data.get("min_block_budget_seconds")
    if min_block is None:
        min_s = min(90.0, daily) if daily > 0 else 0.0
    elif isinstance(min_block, (int, float)):
        min_s = float(min_block)
    else:
        min_s = parse_duration(str(min_block))

    cap_raw = data.get("budget_cap") or data.get("budget_cap_seconds")
    cap_s: Optional[float]
    if cap_raw is None:
        cap_s = None
    elif isinstance(cap_raw, (int, float)):
        cap_s = float(cap_raw)
    else:
        cap_s = parse_duration(str(cap_raw))

    init_raw = data.get("initial_budget") or data.get("initial_budget_seconds")
    init_s: Optional[float]
    if init_raw is None:
        # Default: seed one buffer so a fresh process can mine immediately.
        init_s = min_s
    elif isinstance(init_raw, (int, float)):
        init_s = float(init_raw)
    else:
        key = str(init_raw).strip().lower()
        if key == "min":
            init_s = min_s
        elif key == "daily":
            init_s = daily
        elif key == "cap":
            init_s = cap_s if cap_s is not None else max(daily, min_s)
        else:
            init_s = parse_duration(key)

    return QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=daily,
            min_block_budget_seconds=min_s,
            budget_cap_seconds=cap_s,
            initial_budget_seconds=init_s,
        )
    )
