"""Tests for PeerBanList graduated backoff state machine."""

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
    """Record failures until the next one would be a real ban.

    Uses a short-cooldown list so expired checks pass immediately.
    """
    bl2 = PeerBanList(cooldown=0.0)
    for _ in range(BAN_THRESHOLD - 1):
        bl2.record_failure(peer, "timeout")
    # Copy the record into the target ban list
    bl._records[peer] = bl2._records[peer]
    bl._records[peer].banned_until = 0.0  # ensure expired


# --- Basic state ---

def test_not_banned_initially(ban_list):
    assert not ban_list.is_banned("1.2.3.4:20049")
    assert ban_list.time_remaining("1.2.3.4:20049") == 0.0
    assert ban_list.failure_count("1.2.3.4:20049") == 0
    assert ban_list.ban_count("1.2.3.4:20049") == 0


# --- Cooldown phase ---

def test_cooldown_after_first_failure(ban_list):
    duration = ban_list.record_failure("peer:1", "timeout")
    assert duration == COOLDOWN_DURATION
    assert ban_list.is_banned("peer:1")
    assert ban_list.failure_count("peer:1") == 1
    assert ban_list.ban_count("peer:1") == 0


def test_cooldown_below_threshold():
    bl = PeerBanList(cooldown=0.0)  # instant expiry
    for i in range(BAN_THRESHOLD - 1):
        duration = bl.record_failure("peer:1", "timeout")
        assert duration == 0.0  # cooldown=0
        assert bl.ban_count("peer:1") == 0, f"failure #{i+1}"


# --- No-op while active ---

def test_noop_while_cooldown_active(ban_list):
    """record_failure is a no-op while cooldown is still active."""
    ban_list.record_failure("peer:1", "timeout")
    assert ban_list.failure_count("peer:1") == 1

    # Call again while still in cooldown
    duration = ban_list.record_failure("peer:1", "timeout again")
    assert duration == 0.0  # no-op
    assert ban_list.failure_count("peer:1") == 1  # not incremented


def test_noop_while_ban_active(ban_list):
    """record_failure is a no-op while a real ban is active."""
    _exhaust_cooldowns(ban_list)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        ban_list.record_failure("peer:1", "timeout")
    assert ban_list.ban_count("peer:1") == 1

    # Call again while ban is active
    duration = ban_list.record_failure("peer:1", "still failing")
    assert duration == 0.0
    assert ban_list.ban_count("peer:1") == 1  # not incremented


# --- Ban phase ---

def test_ban_at_threshold(ban_list):
    _exhaust_cooldowns(ban_list)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        duration = ban_list.record_failure("peer:1", "timeout")
    assert duration == MIN_BAN_DURATION
    assert ban_list.ban_count("peer:1") == 1


def test_exponential_escalation_by_ban_count():
    """Each ban doubles duration, based on ban_count not failure_count."""
    bl = PeerBanList(cooldown=0.0, min_duration=0.0)
    _exhaust_cooldowns(bl)
    ban_durations = []
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for _ in range(5):
            d = bl.record_failure("peer:1", "timeout")
            ban_durations.append(d)
            bl._records["peer:1"].banned_until = 0.0  # expire for next

    # Durations based on ban_count 1,2,3,4,5 → 2^0, 2^1, 2^2, 2^3, 2^4
    # With min_duration=0.0, all are 0. Use real min_duration instead:
    bl2 = PeerBanList(cooldown=0.0)
    _exhaust_cooldowns(bl2)
    durations = []
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for _ in range(5):
            d = bl2.record_failure("peer:1", "timeout")
            durations.append(d)
            bl2._records["peer:1"].banned_until = 0.0

    assert durations[0] == MIN_BAN_DURATION          # 2m
    assert durations[1] == MIN_BAN_DURATION * 2       # 4m
    assert durations[2] == MIN_BAN_DURATION * 4       # 8m
    assert durations[3] == MIN_BAN_DURATION * 8       # 16m
    assert durations[4] == MIN_BAN_DURATION * 16      # 32m


def test_max_duration_cap():
    bl = PeerBanList(cooldown=0.0)
    _exhaust_cooldowns(bl)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        for _ in range(30):
            bl.record_failure("peer:1", "timeout")
            bl._records["peer:1"].banned_until = 0.0
    # Last ban should be capped
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        duration = bl.record_failure("peer:1", "timeout")
    assert duration == MAX_BAN_DURATION


def test_jitter_applied():
    bl = PeerBanList(cooldown=0.0)
    _exhaust_cooldowns(bl)
    with patch("shared.peer_ban_list.random.uniform", return_value=0.75):
        low = bl.record_failure("peer:1", "timeout")

    bl2 = PeerBanList(cooldown=0.0)
    _exhaust_cooldowns(bl2, "peer:2")
    with patch("shared.peer_ban_list.random.uniform", return_value=1.25):
        high = bl2.record_failure("peer:2", "timeout")

    assert low == MIN_BAN_DURATION * 0.75
    assert high == MIN_BAN_DURATION * 1.25
    assert low < high


# --- Recovery ---

def test_success_clears_entirely(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    assert ban_list.failure_count("peer:1") == 1

    ban_list.record_success("peer:1")
    assert ban_list.failure_count("peer:1") == 0
    assert ban_list.ban_count("peer:1") == 0
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


# --- Expiry ---

def test_cooldown_expires():
    bl = PeerBanList(cooldown=0.1)
    bl.record_failure("peer:1", "timeout")
    assert bl.is_banned("peer:1")
    time.sleep(0.15)
    assert not bl.is_banned("peer:1")


def test_ban_expires():
    bl = PeerBanList(min_duration=0.1, max_duration=1.0, ban_threshold=1)
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        bl.record_failure("peer:1", "timeout")
    assert bl.is_banned("peer:1")
    time.sleep(0.15)
    assert not bl.is_banned("peer:1")


# --- Inspection ---

def test_banned_peers_dict(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    ban_list.record_failure("peer:2", "timeout")
    banned = ban_list.banned_peers()
    assert "peer:1" in banned
    assert "peer:2" in banned


def test_independent_peer_tracking(ban_list):
    ban_list.record_failure("peer:1", "timeout")
    assert ban_list.failure_count("peer:1") == 1

    ban_list.record_failure("peer:2", "timeout")
    assert ban_list.failure_count("peer:2") == 1
    assert ban_list.failure_count("peer:1") == 1


def test_format_duration():
    assert PeerBanList._format_duration(1800) == "30m"
    assert PeerBanList._format_duration(3600) == "1.0h"
    assert PeerBanList._format_duration(86400) == "1.0d"
    assert PeerBanList._format_duration(604800) == "7.0d"


def test_clear_nonexistent_peer(ban_list):
    ban_list.clear_ban("nonexistent:1")
    ban_list.record_success("nonexistent:2")


# --- Escalation steps ---

def test_escalation_steps_to_max():
    bl = PeerBanList(cooldown=0.0)
    _exhaust_cooldowns(bl)
    ban_steps = 0
    with patch("shared.peer_ban_list.random.uniform", return_value=1.0):
        while True:
            duration = bl.record_failure("peer:1", "timeout")
            bl._records["peer:1"].banned_until = 0.0  # expire for next
            ban_steps += 1
            if duration == MAX_BAN_DURATION:
                break
            assert ban_steps < 20
    # 120 * 2^(n-1) >= 14400 → n=8
    assert ban_steps == 8
