"""Tests for version compatibility logic."""

from unittest.mock import patch

import pytest

from shared.version import is_version_compatible, MIN_COMPATIBLE_VERSION


@pytest.fixture(autouse=True)
def _local_version_008():
    """Pin local version to 0.0.8 for deterministic tests."""
    with patch("shared.version.get_version", return_value="0.0.8"):
        yield


def test_same_version():
    assert is_version_compatible("0.0.8") is True


def test_newer_patch():
    assert is_version_compatible("0.0.9") is True


def test_older_patch():
    assert is_version_compatible("0.0.7") is True


def test_different_minor_newer():
    assert is_version_compatible("0.1.0") is False


def test_different_minor_older():
    with patch("shared.version.get_version", return_value="0.1.0"):
        assert is_version_compatible("0.0.8") is False


def test_different_major():
    assert is_version_compatible("1.0.0") is False


def test_below_min_compatible_version():
    with patch("shared.version.MIN_COMPATIBLE_VERSION", "0.0.5"):
        assert is_version_compatible("0.0.4") is False
        assert is_version_compatible("0.0.5") is True
        assert is_version_compatible("0.0.8") is True


def test_dev_suffix_different_minor():
    """Dev pre-releases follow the same major.minor rule."""
    assert is_version_compatible("0.1.0-dev") is False
    assert is_version_compatible("0.1.0.dev1") is False


def test_dev_suffix_same_minor():
    assert is_version_compatible("0.0.9.dev1") is True


def test_default_min_is_permissive():
    """Default MIN_COMPATIBLE_VERSION accepts all 0.0.x peers."""
    assert MIN_COMPATIBLE_VERSION == "0.0.1"
    assert is_version_compatible("0.0.1") is True
