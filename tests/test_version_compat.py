"""Tests for version compatibility logic."""

from unittest.mock import patch

import pytest

from shared.version import (
    is_version_compatible,
    MIN_COMPATIBLE_VERSION,
    select_compatible_peers,
)


@pytest.fixture(autouse=True)
def _local_version_pinned():
    """Pin local version to MIN_COMPATIBLE_VERSION for deterministic tests."""
    with patch("shared.version.get_version", return_value=MIN_COMPATIBLE_VERSION):
        yield


def test_same_version():
    assert is_version_compatible(MIN_COMPATIBLE_VERSION) is True


def test_newer_patch():
    # Local is "0.1.1"; a "0.1.2" peer shares major.minor and is >= MIN.
    assert is_version_compatible("0.1.2") is True


def test_older_patch():
    """0.1.0 is below MIN_COMPATIBLE_VERSION (0.1.1), so incompatible."""
    assert is_version_compatible("0.1.0") is False


def test_even_older_patch():
    """0.0.x is far below MIN_COMPATIBLE_VERSION, incompatible."""
    assert is_version_compatible("0.0.8") is False


def test_different_minor_newer():
    assert is_version_compatible("0.2.0") is False


def test_different_minor_older():
    with patch("shared.version.get_version", return_value="0.2.0"):
        assert is_version_compatible("0.1.0") is False


def test_different_major():
    assert is_version_compatible("1.0.0") is False


def test_below_min_compatible_version():
    with patch("shared.version.MIN_COMPATIBLE_VERSION", "0.1.1"):
        assert is_version_compatible("0.0.9") is False
        assert is_version_compatible("0.1.0") is False
        assert is_version_compatible("0.1.1") is True
        assert is_version_compatible("0.1.5") is True


def test_dev_suffix_different_minor():
    """Dev pre-releases follow the same major.minor rule."""
    assert is_version_compatible("0.2.0-dev") is False
    assert is_version_compatible("0.2.0.dev1") is False


def test_dev_suffix_same_minor():
    # 0.1.1.dev1 is a pre-release of 0.1.1 and parses as < 0.1.1, so it
    # fails MIN_COMPATIBLE_VERSION. This is the desired behavior — dev
    # builds of the first acceptable release are still rejected.
    assert is_version_compatible("0.1.1.dev1") is False


def test_dev_suffix_above_min():
    """Pre-release of the next patch passes MIN and shares major.minor."""
    assert is_version_compatible("0.1.2.dev1") is True


def test_default_min_rejects_old_versions():
    """MIN_COMPATIBLE_VERSION 0.1.1 rejects all 0.1.0 and 0.0.x peers."""
    assert MIN_COMPATIBLE_VERSION == "0.1.1"
    assert is_version_compatible("0.0.8") is False
    assert is_version_compatible("0.1.0") is False
    assert is_version_compatible("0.1.1") is True


def test_missing_version_rejected():
    """Peers that don't announce a version cannot bypass the gate."""
    assert is_version_compatible(None) is False
    assert is_version_compatible("") is False


def test_select_compatible_peers_excludes_unknown_version():
    """Peers with no recorded version must be excluded from sync.

    Regression guard for commit ``e4ec0b2`` ("Exclude unknown-version
    peers from block sync"). Local version is pinned to 0.1.1.
    """
    peers = {"a": "info-a", "b": "info-b", "c": "info-c", "d": "info-d"}
    peer_versions = {
        "a": "0.1.1",   # compatible
        "b": "0.1.0",   # below MIN
        "c": None,      # unknown → exclude
        # "d" missing entirely → exclude
    }
    result = select_compatible_peers(peers, peer_versions)
    assert set(result.keys()) == {"a"}


def test_select_compatible_peers_empty_input():
    assert select_compatible_peers({}, {}) == {}


def test_select_compatible_peers_all_unknown():
    peers = {"a": "info", "b": "info"}
    assert select_compatible_peers(peers, {}) == {}


def test_malformed_version_rejected():
    """Unparseable version strings are rejected rather than crashing."""
    assert is_version_compatible("banana") is False
    assert is_version_compatible("not.a.version") is False
    assert is_version_compatible("...") is False
    # Missing-patch strings like "0.1" parse but fall below MIN (0.1.1).
    assert is_version_compatible("0.1") is False
