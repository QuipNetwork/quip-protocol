"""
Peer ban list with graduated backoff.

Tracks peers that repeatedly fail to connect. Early failures get
short cooldowns (30s); actual bans start after BAN_THRESHOLD
consecutive failures and escalate from 1 hour to 1 week with
randomized jitter.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# Failures below this threshold get a short cooldown, not a ban.
BAN_THRESHOLD = 5

# Cooldown for early failures (below threshold)
COOLDOWN_DURATION = 30.0        # 30 seconds

# Ban duration bounds (seconds) — applied at threshold and above
MIN_BAN_DURATION = 120.0        # 2 minutes
MAX_BAN_DURATION = 14400.0      # 4 hours
JITTER_RANGE = (0.75, 1.25)    # ±25% randomization


@dataclass
class _BanRecord:
    """Internal record for a banned peer."""
    failure_count: int = 0
    banned_until: float = 0.0
    last_failure: float = 0.0


class PeerBanList:
    """
    Track peer connection failures with graduated backoff.

    Failures 1 through BAN_THRESHOLD-1 apply a short cooldown (30s).
    At BAN_THRESHOLD and above, real bans kick in starting at 1 hour
    and doubling up to 1 week. Successful connections reset everything.
    """

    def __init__(
        self,
        min_duration: float = MIN_BAN_DURATION,
        max_duration: float = MAX_BAN_DURATION,
        ban_threshold: int = BAN_THRESHOLD,
        cooldown: float = COOLDOWN_DURATION,
        logger: Optional[logging.Logger] = None,
    ):
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._ban_threshold = ban_threshold
        self._cooldown = cooldown
        self._records: dict[str, _BanRecord] = {}
        self.logger = logger or logging.getLogger(__name__)

    def is_banned(self, peer: str) -> bool:
        """Check if a peer is currently banned (or in cooldown)."""
        record = self._records.get(peer)
        if record is None:
            return False
        if time.monotonic() >= record.banned_until:
            return False
        return True

    def time_remaining(self, peer: str) -> float:
        """Seconds remaining on a peer's ban, or 0 if not banned."""
        record = self._records.get(peer)
        if record is None:
            return 0.0
        remaining = record.banned_until - time.monotonic()
        return max(0.0, remaining)

    def failure_count(self, peer: str) -> int:
        """Number of recorded failures for a peer."""
        record = self._records.get(peer)
        return record.failure_count if record else 0

    def record_failure(self, peer: str, reason: str = "") -> float:
        """Record a connection failure.

        Below the ban threshold, applies a short cooldown.
        At the threshold and above, applies escalating bans.

        Args:
            peer: Peer address (host:port).
            reason: Human-readable failure reason for logging.

        Returns:
            Cooldown/ban duration in seconds.
        """
        now = time.monotonic()
        record = self._records.get(peer)
        if record is None:
            record = _BanRecord()
            self._records[peer] = record

        record.failure_count += 1
        record.last_failure = now

        if record.failure_count < self._ban_threshold:
            # Short cooldown — not a real ban
            duration = self._cooldown
            record.banned_until = now + duration
            reason_str = f" — {reason}" if reason else ""
            self.logger.debug(
                "Cooldown peer %s for %ds (failure #%d%s)",
                peer, int(duration), record.failure_count, reason_str,
            )
        else:
            # Real ban with exponential backoff
            exponent = record.failure_count - self._ban_threshold
            raw_duration = self._min_duration * (2 ** exponent)
            capped_duration = min(raw_duration, self._max_duration)
            jitter = random.uniform(*JITTER_RANGE)
            duration = capped_duration * jitter
            record.banned_until = now + duration
            reason_str = f" — {reason}" if reason else ""
            self.logger.warning(
                "Banned peer %s for %s (failure #%d%s)",
                peer, self._format_duration(duration),
                record.failure_count, reason_str,
            )

        return duration

    def record_success(self, peer: str) -> None:
        """Record a successful connection, resetting the failure count."""
        if peer in self._records:
            del self._records[peer]

    def clear_ban(self, peer: str) -> None:
        """Manually clear a peer's ban and failure history."""
        self._records.pop(peer, None)

    def clear_all(self) -> int:
        """Clear all bans. Returns number of records cleared."""
        count = len(self._records)
        self._records.clear()
        return count

    def banned_peers(self) -> dict[str, float]:
        """Return dict of currently banned peers and seconds remaining."""
        now = time.monotonic()
        result = {}
        for peer, record in self._records.items():
            remaining = record.banned_until - now
            if remaining > 0:
                result[peer] = remaining
        return result

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds < 3600:
            return f"{seconds / 60:.0f}m"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"
