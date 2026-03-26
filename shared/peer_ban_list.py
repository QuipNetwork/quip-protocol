"""
Peer ban list with exponential backoff.

Tracks peers that repeatedly fail to connect and bans them for
increasing durations, from 1 hour up to 1 week, with randomized
jitter to prevent thundering herd reconnection storms.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# Ban duration bounds (seconds)
MIN_BAN_DURATION = 3600.0       # 1 hour
MAX_BAN_DURATION = 604800.0     # 1 week
JITTER_RANGE = (0.75, 1.25)    # ±25% randomization


@dataclass
class _BanRecord:
    """Internal record for a banned peer."""
    failure_count: int = 0
    banned_until: float = 0.0
    last_failure: float = 0.0


class PeerBanList:
    """
    Track peer connection failures and apply exponential backoff bans.

    Each failure doubles the ban duration (with jitter), starting at
    1 hour and capping at 1 week. Successful connections reset the
    failure count.
    """

    def __init__(
        self,
        min_duration: float = MIN_BAN_DURATION,
        max_duration: float = MAX_BAN_DURATION,
        logger: Optional[logging.Logger] = None,
    ):
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._records: dict[str, _BanRecord] = {}
        self.logger = logger or logging.getLogger(__name__)

    def is_banned(self, peer: str) -> bool:
        """Check if a peer is currently banned."""
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
        """Record a connection failure and ban the peer.

        Args:
            peer: Peer address (host:port).
            reason: Human-readable failure reason for logging.

        Returns:
            Ban duration in seconds.
        """
        now = time.monotonic()
        record = self._records.get(peer)
        if record is None:
            record = _BanRecord()
            self._records[peer] = record

        record.failure_count += 1
        record.last_failure = now

        # Exponential backoff: min_dur * 2^(n-1), capped at max_dur
        raw_duration = self._min_duration * (2 ** (record.failure_count - 1))
        capped_duration = min(raw_duration, self._max_duration)

        # Apply jitter
        jitter = random.uniform(*JITTER_RANGE)
        duration = capped_duration * jitter

        record.banned_until = now + duration

        reason_str = f" — {reason}" if reason else ""
        self.logger.warning(
            f"Banned peer {peer} for {self._format_duration(duration)} "
            f"(failure #{record.failure_count}{reason_str})"
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
