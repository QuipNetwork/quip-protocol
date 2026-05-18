"""
Peer ban list with graduated backoff.

Tracks peers that repeatedly fail to connect. Early failures get
short cooldowns (30s); actual bans start after BAN_THRESHOLD
consecutive failures and escalate from 2 minutes to 4 hours.

State machine:
  - Cooldown/ban active → record_failure is a no-op
  - Cooldown/ban expired (or no entry) → increment and apply next level
  - Any state + success → clear entry entirely
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional


# Failures below this threshold get a short cooldown, not a ban.
BAN_THRESHOLD = 5

# Cooldown for early failures (below threshold)
COOLDOWN_DURATION = 30.0        # 30 seconds

# Ban duration bounds (seconds) — applied at threshold and above
MIN_BAN_DURATION = 120.0        # 2 minutes
MAX_BAN_DURATION = 14400.0      # 4 hours
JITTER_RANGE = (0.75, 1.25)    # ±25% randomization

# Capacity-rejection cooldowns — peer is healthy but full. Distinct
# from failure graduation: we want to back off, but not escalate into
# a long ban since the peer is not misbehaving.
CAPACITY_COOLDOWN_MIN = 60.0    # 1 minute
CAPACITY_COOLDOWN_MAX = 900.0   # 15 minutes


@dataclass
class _BanRecord:
    """Internal record for a peer's ban history."""
    failure_count: int = 0
    ban_count: int = 0
    banned_until: float = 0.0
    last_failure: float = 0.0
    capacity_count: int = 0


class PeerBanList:
    """
    Track peer connection failures with graduated backoff.

    Failures 1 through BAN_THRESHOLD-1 apply a short cooldown (30s).
    At BAN_THRESHOLD and above, real bans kick in starting at 2 min
    and doubling to 4 hours.

    Key invariant: record_failure is a no-op while a cooldown or ban
    is still active. This prevents runaway escalation from retry loops.
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
        return time.monotonic() < record.banned_until

    def time_remaining(self, peer: str) -> float:
        """Seconds remaining on a peer's ban, or 0 if not banned."""
        record = self._records.get(peer)
        if record is None:
            return 0.0
        return max(0.0, record.banned_until - time.monotonic())

    def failure_count(self, peer: str) -> int:
        """Number of recorded failures for a peer."""
        record = self._records.get(peer)
        return record.failure_count if record else 0

    def ban_count(self, peer: str) -> int:
        """Number of real bans (not cooldowns) applied to a peer."""
        record = self._records.get(peer)
        return record.ban_count if record else 0

    def record_failure(self, peer: str, reason: str = "") -> float:
        """Record a connection failure.

        No-op if the peer is still in an active cooldown or ban.
        Otherwise increments failure count and applies the next
        level of cooldown/ban.

        Args:
            peer: Peer address (host:port).
            reason: Human-readable failure reason for logging.

        Returns:
            Cooldown/ban duration in seconds, or 0 if no-op.
        """
        now = time.monotonic()
        record = self._records.get(peer)

        # Still active — don't escalate
        if record is not None and now < record.banned_until:
            return 0.0

        if record is None:
            record = _BanRecord()
            self._records[peer] = record

        record.failure_count += 1
        record.last_failure = now
        reason_str = f" — {reason}" if reason else ""

        if record.failure_count < self._ban_threshold:
            # Short cooldown — not a real ban
            duration = self._cooldown
            record.banned_until = now + duration
            self.logger.debug(
                "Cooldown peer %s for %ds (failure #%d%s)",
                peer, int(duration), record.failure_count, reason_str,
            )
        else:
            # Real ban with exponential backoff based on ban_count
            record.ban_count += 1
            raw_duration = self._min_duration * (2 ** (record.ban_count - 1))
            capped_duration = min(raw_duration, self._max_duration)
            jitter = random.uniform(*JITTER_RANGE)
            duration = capped_duration * jitter
            record.banned_until = now + duration
            self.logger.warning(
                "Banned peer %s for %s (ban #%d, failure #%d%s)",
                peer, self._format_duration(duration),
                record.ban_count, record.failure_count, reason_str,
            )

        return duration

    def record_capacity_rejection(self, peer: str) -> float:
        """Record an ``at_capacity`` rejection from a peer.

        Separate from ``record_failure`` because capacity rejection is
        a transient resource signal, not misbehavior. Applies a short
        cooldown that doubles on repeats (1 min → 2 min → 4 min …) and
        is capped at ``CAPACITY_COOLDOWN_MAX`` (15 min).

        Does not contribute to ``failure_count`` or the long-ban
        graduation ladder; a peer that rejects us for capacity then
        later accepts is recorded via ``record_success`` as normal.

        No-op while the peer is already in cooldown/ban — prevents a
        retry loop from inflating the backoff.

        Returns:
            Cooldown duration in seconds, or 0 if no-op.
        """
        now = time.monotonic()
        record = self._records.get(peer)

        if record is not None and now < record.banned_until:
            return 0.0

        if record is None:
            record = _BanRecord()
            self._records[peer] = record

        record.capacity_count += 1
        raw = CAPACITY_COOLDOWN_MIN * (2 ** (record.capacity_count - 1))
        duration = min(raw, CAPACITY_COOLDOWN_MAX)
        record.banned_until = now + duration
        self.logger.debug(
            "Capacity cooldown peer %s for %s (capacity #%d)",
            peer, self._format_duration(duration), record.capacity_count,
        )
        return duration

    def record_success(self, peer: str) -> None:
        """Record a successful connection — clears entry entirely."""
        self._records.pop(peer, None)

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
