"""Tests for version compatibility logic."""

from unittest.mock import patch

import pytest

from shared.version import (
    is_version_compatible,
    MIN_COMPATIBLE_VERSION,
    select_compatible_peers,
    version_from_descriptor,
)


@pytest.fixture(autouse=True)
def _local_version():
    """Pin local version to match MIN_COMPATIBLE_VERSION for deterministic tests."""
    with patch("shared.version.get_version", return_value="0.1.2"):
        yield


def test_same_version():
    assert is_version_compatible("0.1.2") is True


def test_newer_patch():
    assert is_version_compatible("0.1.3") is True


def test_older_patch():
    """0.1.1 is below MIN_COMPATIBLE_VERSION (0.1.2), so incompatible."""
    assert is_version_compatible("0.1.1") is False


def test_below_minor_rejected():
    """0.0.x is below both MIN and the local minor."""
    assert is_version_compatible("0.0.8") is False


def test_different_minor_newer():
    assert is_version_compatible("0.2.0") is False


def test_different_minor_older():
    with patch("shared.version.get_version", return_value="0.2.0"):
        assert is_version_compatible("0.1.2") is False


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
    # PEP 440 orders X.dev1 < X, so the dev must be newer than MIN's patch.
    assert is_version_compatible("0.1.3.dev1") is True


def test_default_min_rejects_old_versions():
    """MIN_COMPATIBLE_VERSION 0.1.2 rejects all 0.1.1 and earlier peers."""
    assert MIN_COMPATIBLE_VERSION == "0.1.2"
    assert is_version_compatible("0.0.8") is False
    assert is_version_compatible("0.1.1") is False
    assert is_version_compatible("0.1.2") is True


def test_missing_version_rejected():
    """Peers that don't announce a version cannot bypass the gate."""
    assert is_version_compatible(None) is False
    assert is_version_compatible("") is False


def test_malformed_version_rejected():
    """Unparseable version strings are rejected rather than crashing."""
    assert is_version_compatible("banana") is False
    assert is_version_compatible("not.a.version") is False
    assert is_version_compatible("...") is False
    # Missing-patch strings like "0.1" parse but fall below MIN (0.1.2).
    assert is_version_compatible("0.1") is False


def test_select_compatible_peers_excludes_unknown_version():
    """Peers with no recorded version must be excluded from sync.

    Regression guard for commit ``e4ec0b2`` ("Exclude unknown-version
    peers from block sync"). Local version is pinned to 0.1.2 via the
    autouse fixture.
    """
    peers = {"a": "info-a", "b": "info-b", "c": "info-c", "d": "info-d"}
    peer_versions = {
        "a": "0.1.2",   # compatible
        "b": "0.1.1",   # below MIN
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


# ---------------------------------------------------------------------------
# version_from_descriptor
# ---------------------------------------------------------------------------

def test_version_from_descriptor_extracts_runtime_version():
    desc = {"runtime": {"quip_version": "0.1.11"}, "node_id": "n1"}
    assert version_from_descriptor(desc) == "0.1.11"


def test_version_from_descriptor_none_when_descriptor_none():
    assert version_from_descriptor(None) is None


def test_version_from_descriptor_none_when_empty():
    assert version_from_descriptor({}) is None


def test_version_from_descriptor_none_when_runtime_missing():
    assert version_from_descriptor({"node_id": "n1"}) is None


def test_version_from_descriptor_none_when_runtime_empty():
    assert version_from_descriptor({"runtime": {}}) is None


def test_version_from_descriptor_none_when_version_empty_string():
    assert version_from_descriptor({"runtime": {"quip_version": ""}}) is None


def test_version_from_descriptor_tolerates_non_dict_runtime():
    # A peer sending a malformed descriptor shouldn't raise — we treat
    # the version as unknown and let is_version_compatible reject it.
    assert version_from_descriptor({"runtime": None}) is None
    assert version_from_descriptor({"runtime": "not-a-dict"}) is None
