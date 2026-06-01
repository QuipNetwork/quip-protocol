"""Token-bucket per-peer rate limiter.

Each peer gets a bucket that refills at a steady rate. Requests are
allowed as long as the bucket has tokens. This prevents any single
peer from flooding the node with messages.

Usage::

    limiter = PeerRateLimiter(tokens_per_second=10.0, max_burst=20)
    if limiter.allow("1.2.3.4:20049"):
        handle(message)
    else:
        drop(message)
"""

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _Bucket:
    """Internal token bucket state for one peer."""
    tokens: float
    last_refill: float


@dataclass
class PeerRateLimiter:
    """Token-bucket rate limiter keyed by peer address.

    Args:
        tokens_per_second: Refill rate per peer.
        max_burst: Maximum tokens a bucket can hold (burst capacity).
    """
    tokens_per_second: float = 10.0
    max_burst: int = 20
    _buckets: Dict[str, _Bucket] = field(default_factory=dict, repr=False)

    def allow(self, peer: str) -> bool:
        """Check whether *peer* may send a message right now.

        Returns True and consumes one token if allowed, False otherwise.
        """
        now = time.monotonic()
        bucket = self._buckets.get(peer)
        if bucket is None:
            bucket = _Bucket(tokens=self.max_burst, last_refill=now)
            self._buckets[peer] = bucket

        # Refill tokens based on elapsed time
        elapsed = now - bucket.last_refill
        if elapsed > 0:
            bucket.tokens = min(
                self.max_burst,
                bucket.tokens + elapsed * self.tokens_per_second,
            )
            bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    def remove_peer(self, peer: str) -> None:
        """Remove tracking state for a disconnected peer."""
        self._buckets.pop(peer, None)

    def prune(self, max_idle: float = 300.0) -> int:
        """Remove buckets that have been idle for more than *max_idle* seconds.

        Returns the number of buckets removed.
        """
        now = time.monotonic()
        stale = [
            peer for peer, bucket in self._buckets.items()
            if now - bucket.last_refill > max_idle
        ]
        for peer in stale:
            del self._buckets[peer]
        return len(stale)

    def peer_count(self) -> int:
        """Number of peers currently being tracked."""
        return len(self._buckets)
