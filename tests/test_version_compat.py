"""Tests for version compatibility logic."""

from unittest.mock import patch

import pytest

from shared.version import is_version_compatible, MIN_COMPATIBLE_VERSION


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
    with patch("shared.version.MIN_COMPATIBLE_VERSION", "0.1.0"):
        assert is_version_compatible("0.0.9") is False
        assert is_version_compatible("0.1.0") is True
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
