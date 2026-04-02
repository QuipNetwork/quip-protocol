"""Tests for PeerBanList graduated backoff logic."""

import time
from unittest.mock import patch

import pytest

from shared.peer_ban_list import (
    PeerBanList,
    BAN_THRESHOLD,
    COOLDOWN_DURATION,
    MIN_BAN_DURATION,
    MAX_BAN_DURATION,
)


@pytest.fixture
def ban_list():
    return PeerBanList()


def _exhaust_cooldowns(bl, peer="peer:1"):
    """Record failures until the next one would be a real ban."""
    for _ in range(BAN_THRESHOLD - 1):
        bl.record_failure(peer, "timeout")


def test_not_banned_initially(ban_list):
    assert not ban_list.is_banned("1.2.3.4:20049")
    assert ban_list.time_remaining("1.2.3.4:20049") == 0.0
    assert ban_list.failure_count("1.2.3.4:20049") == 0


def test_cooldown_after_first_failure(ban_list):
    """First failure applies a short cooldown, not a real ban."""
    duration = ban_list.record_failure("1.2.3.4:20049", "handshake timeout")
    assert duration == COOLDOWN_DURATION
    assert ban_list.is_banned("1.2.3.4:20049")
    assert ban_list.failure_count("1.2.3.4:20049") == 1


def test_cooldown_below_threshold(ban_list):
    """All failures below threshold get the short cooldown."""
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for i in range(BAN_THRESHOLD - 1):
            duration = ban_list.record_failure("peer:1", "timeout")
            assert duration == COOLDOWN_DURATION, f"failure #{i+1}"


def test_ban_at_threshold(ban_list):
    """Ban kicks in at the threshold failure."""
    _exhaust_cooldowns(ban_list)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        duration = ban_list.record_failure("peer:1", "timeout")
    assert duration == MIN_BAN_DURATION


def test_exponential_escalation():
    """Each failure past threshold doubles the ban duration."""
    bl = PeerBanList()
    _exhaust_cooldowns(bl)
    durations = []
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for _ in range(5):
            d = bl.record_failure("peer:1", "timeout")
            durations.append(d)

    assert durations[0] == MIN_BAN_DURATION          # 1h
    assert durations[1] == MIN_BAN_DURATION * 2       # 2h
    assert durations[2] == MIN_BAN_DURATION * 4       # 4h
    assert durations[3] == MIN_BAN_DURATION * 8       # 8h
    assert durations[4] == MIN_BAN_DURATION * 16      # 16h


def test_max_duration_cap():
    """Ban duration should cap at MAX_BAN_DURATION (1 week)."""
    bl = PeerBanList()
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for _ in range(30):
            duration = bl.record_failure("peer:1", "timeout")
    assert duration == MAX_BAN_DURATION


def test_jitter_applied():
    """Ban duration should vary with jitter once past threshold."""
    bl = PeerBanList()
    _exhaust_cooldowns(bl)
    with patch("shared.peer_ban_list.random.uniform", return_value=0.75):
        low = bl.record_failure("peer:1", "timeout")

    bl2 = PeerBanList()
    _exhaust_cooldowns(bl2, "peer:2")
    with patch("shared.peer_ban_list.random.uniform", return_value=1.25):
        high = bl2.record_failure("peer:2", "timeout")

    assert low == MIN_BAN_DURATION * 0.75
    assert high == MIN_BAN_DURATION * 1.25
    assert low < high


def test_success_resets_failure_count(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    assert ban_list.failure_count("peer:1") == 1

    ban_list.record_success("peer:1")
    assert ban_list.failure_count("peer:1") == 0
    assert not ban_list.is_banned("peer:1")


def test_clear_ban(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    assert ban_list.is_banned("peer:1")

    ban_list.clear_ban("peer:1")
    assert not ban_list.is_banned("peer:1")
    assert ban_list.failure_count("peer:1") == 0


def test_clear_all(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    ban_list.record_failure("peer:2", "timeout")
    assert ban_list.clear_all() == 2
    assert not ban_list.is_banned("peer:1")
    assert not ban_list.is_banned("peer:2")


def test_cooldown_expires():
    """Cooldown should expire quickly."""
    bl = PeerBanList(cooldown=0.1)
    bl.record_failure("peer:1", "timeout")
    assert bl.is_banned("peer:1")

    time.sleep(0.15)
    assert not bl.is_banned("peer:1")


def test_ban_expires():
    """Real ban should expire after the duration passes."""
    bl = PeerBanList(min_duration=0.1, max_duration=1.0, ban_threshold=1)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        bl.record_failure("peer:1", "timeout")
    assert bl.is_banned("peer:1")

    time.sleep(0.15)
    assert not bl.is_banned("peer:1")


def test_banned_peers_dict(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    ban_list.record_failure("peer:2", "timeout")
    banned = ban_list.banned_peers()
    assert "peer:1" in banned
    assert "peer:2" in banned
    assert banned["peer:1"] > 0
    assert banned["peer:2"] > 0


def test_independent_peer_tracking(ban_list):
    """Failures for one peer should not affect another."""
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        ban_list.record_failure("peer:1", "timeout")
        ban_list.record_failure("peer:1", "timeout")
        ban_list.record_failure("peer:2", "timeout")

    assert ban_list.failure_count("peer:1") == 2
    assert ban_list.failure_count("peer:2") == 1


def test_format_duration():
    assert PeerBanList._format_duration(1800) == "30m"
    assert PeerBanList._format_duration(3600) == "1.0h"
    assert PeerBanList._format_duration(86400) == "1.0d"
    assert PeerBanList._format_duration(604800) == "7.0d"


def test_clear_nonexistent_peer(ban_list):
    """Clearing a peer that was never banned should not raise."""
    ban_list.clear_ban("nonexistent:1")
    ban_list.record_success("nonexistent:2")


def test_escalation_steps_to_max():
    """Verify the number of ban escalation steps to reach max."""
    bl = PeerBanList()
    _exhaust_cooldowns(bl)
    ban_steps = 0
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        while True:
            duration = bl.record_failure("peer:1", "timeout")
            ban_steps += 1
            if duration == MAX_BAN_DURATION:
                break
            assert ban_steps < 20, "should reach max within 20 steps"
    # 1h * 2^(n-1) >= 604800 → n >= 9
    assert ban_steps == 9
