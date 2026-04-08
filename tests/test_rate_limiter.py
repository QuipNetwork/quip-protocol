"""Tests for PeerRateLimiter token-bucket implementation."""

import time
from unittest.mock import patch

import pytest

from shared.rate_limiter import PeerRateLimiter


def test_first_request_allowed():
    """First request from a new peer is always allowed."""
    limiter = PeerRateLimiter(tokens_per_second=10.0, max_burst=20)
    assert limiter.allow("peer1") is True


def test_burst_capacity():
    """Peer can send up to max_burst requests immediately."""
    limiter = PeerRateLimiter(tokens_per_second=1.0, max_burst=5)
    results = [limiter.allow("peer1") for _ in range(5)]
    assert all(results), "All burst requests should be allowed"
    assert limiter.allow("peer1") is False, "Exceeding burst should be denied"


def test_refill_after_time():
    """Tokens refill based on elapsed time."""
    limiter = PeerRateLimiter(tokens_per_second=10.0, max_burst=5)
    # Drain the bucket
    for _ in range(5):
        limiter.allow("peer1")
    assert limiter.allow("peer1") is False

    # Advance time by 0.5s -> 5 tokens refilled
    with patch("shared.rate_limiter.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 0.5
        assert limiter.allow("peer1") is True


def test_independent_peer_buckets():
    """Each peer has its own independent token bucket."""
    limiter = PeerRateLimiter(tokens_per_second=1.0, max_burst=2)
    # Drain peer1
    limiter.allow("peer1")
    limiter.allow("peer1")
    assert limiter.allow("peer1") is False

    # peer2 should still have tokens
    assert limiter.allow("peer2") is True


def test_tokens_capped_at_max_burst():
    """Tokens never exceed max_burst even after long idle."""
    limiter = PeerRateLimiter(tokens_per_second=100.0, max_burst=5)
    # Use one token
    limiter.allow("peer1")

    # Even after a long time, max is still max_burst
    with patch("shared.rate_limiter.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 1000
        # Should allow 5, not more
        results = [limiter.allow("peer1") for _ in range(5)]
        assert all(results)
        assert limiter.allow("peer1") is False


def test_remove_peer():
    """Removing a peer clears its bucket state."""
    limiter = PeerRateLimiter(tokens_per_second=1.0, max_burst=2)
    limiter.allow("peer1")
    limiter.allow("peer1")
    assert limiter.allow("peer1") is False

    limiter.remove_peer("peer1")
    # Fresh bucket after removal
    assert limiter.allow("peer1") is True


def test_remove_nonexistent_peer():
    """Removing a peer that doesn't exist is a no-op."""
    limiter = PeerRateLimiter()
    limiter.remove_peer("nonexistent")  # Should not raise


def test_prune_idle_buckets():
    """Prune removes buckets idle for more than max_idle."""
    limiter = PeerRateLimiter(tokens_per_second=10.0, max_burst=20)
    limiter.allow("peer1")
    limiter.allow("peer2")
    assert limiter.peer_count() == 2

    with patch("shared.rate_limiter.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 400
        removed = limiter.prune(max_idle=300.0)
        assert removed == 2
        assert limiter.peer_count() == 0


def test_prune_keeps_active_buckets():
    """Prune keeps buckets that were recently active."""
    limiter = PeerRateLimiter(tokens_per_second=10.0, max_burst=20)
    limiter.allow("peer1")
    # peer1 bucket was just created, should not be pruned
    removed = limiter.prune(max_idle=300.0)
    assert removed == 0
    assert limiter.peer_count() == 1


def test_peer_count():
    """peer_count reflects the number of tracked peers."""
    limiter = PeerRateLimiter()
    assert limiter.peer_count() == 0
    limiter.allow("a")
    limiter.allow("b")
    limiter.allow("c")
    assert limiter.peer_count() == 3


def test_high_rate_sustained():
    """Sustained high-rate traffic is throttled after burst."""
    limiter = PeerRateLimiter(tokens_per_second=2.0, max_burst=3)
    # Burst of 3
    assert limiter.allow("peer1")
    assert limiter.allow("peer1")
    assert limiter.allow("peer1")
    # No more tokens
    assert limiter.allow("peer1") is False
