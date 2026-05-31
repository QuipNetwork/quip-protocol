# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Wall-clock timing for anticipatory proof submission.

Tracks the chain's block interval and our network lag from on-chain block
timestamps, and converts a target block number into a monotonic-clock fire
deadline. Pure + deterministic (clocks are passed in) for unit testing; no
chain or asyncio dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingTracker:
    """EMA tracker for block interval + network lag, and block->deadline math.

    All times in seconds. ``interval_s`` is the smoothed seconds-per-block;
    ``lag_s`` is the smoothed (clamped) gap between a block's on-chain
    timestamp and our local observation of it -- used as the tx-lead.
    """

    fallback_interval_s: float = 6.0
    lag_min_s: float = 0.0
    lag_max_s: float = 12.0
    ema_alpha: float = 0.3

    interval_s: float = 0.0
    lag_s: float = 0.0
    _anchor_block: Optional[int] = None
    _anchor_chain_ts_s: float = 0.0
    _anchor_monotonic_s: float = 0.0
    _prev_block: Optional[int] = None
    _prev_chain_ts_s: float = 0.0
    _have_interval: bool = False

    def __post_init__(self) -> None:
        self.interval_s = self.fallback_interval_s

    def observe_head(
        self,
        *,
        block_number: int,
        chain_ts_s: float,
        monotonic_now: float,
        wallclock_now: float,
    ) -> None:
        """Fold one timestamped head into the interval + lag EMAs + anchor."""
        if (
            self._prev_block is not None
            and block_number > self._prev_block
            and chain_ts_s > self._prev_chain_ts_s
        ):
            observed = (chain_ts_s - self._prev_chain_ts_s) / (
                block_number - self._prev_block
            )
            if not self._have_interval:
                self.interval_s = observed
                self._have_interval = True
            else:
                self.interval_s = (
                    self.ema_alpha * observed
                    + (1.0 - self.ema_alpha) * self.interval_s
                )
        self._prev_block = block_number
        self._prev_chain_ts_s = chain_ts_s

        raw_lag = wallclock_now - chain_ts_s
        clamped = max(self.lag_min_s, min(self.lag_max_s, raw_lag))
        if self._anchor_block is None:
            self.lag_s = clamped
        else:
            self.lag_s = (
                self.ema_alpha * clamped + (1.0 - self.ema_alpha) * self.lag_s
            )

        self._anchor_block = block_number
        self._anchor_chain_ts_s = chain_ts_s
        self._anchor_monotonic_s = monotonic_now

    def fire_deadline_monotonic(
        self, *, b_star: int, now_monotonic: float
    ) -> Optional[float]:
        """Monotonic deadline at which to fire for target block ``b_star``.

        Returns ``anchor_monotonic + (b_star - anchor_block) * interval - lag``,
        or ``None`` if no head has been observed yet. May be <= now (fire now).
        """
        if self._anchor_block is None:
            return None
        chain_delta = (b_star - self._anchor_block) * self.interval_s
        target_monotonic = self._anchor_monotonic_s + chain_delta
        return target_monotonic - self.lag_s
