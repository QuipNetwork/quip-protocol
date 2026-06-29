"""
Version management for QuIP Protocol.

This module provides centralized access to the package version and protocol version.
"""

import importlib.metadata
from typing import Optional

from packaging.version import parse as parse_version

__version__: Optional[str] = None

# Protocol version - increment when making breaking changes to:
# - Block serialization format
# - Network message format
# - Consensus rules
PROTOCOL_VERSION = 2

# Minimum application version reported by ``get_version_info``.
MIN_COMPATIBLE_VERSION = "0.1.2"


def get_version() -> str:
    """
    Get the current version of the QuIP Protocol package.

    Returns:
        Version string (e.g., "0.1.0")
    """
    global __version__

    if __version__ is None:
        try:
            # Try to get version from package metadata (when installed)
            __version__ = importlib.metadata.version("quip-protocol")
        except importlib.metadata.PackageNotFoundError:
            # Fallback when running from a source checkout (not pip-installed).
            # Keep in lockstep with pyproject.toml's version so a source-run
            # miner advertises the same major.minor and isn't rejected by the
            # version-compat gate.
            __version__ = "0.2.1rc27"

    return __version__


def get_version_info() -> dict:
    """
    Get detailed version information.

    Returns:
        Dictionary with version details
    """
    ver = get_version()
    parsed = parse_version(ver)
    release = parsed.release + (0, 0, 0)
    return {
        "version": ver,
        "major": release[0],
        "minor": release[1],
        "patch": release[2],
        "is_dev": parsed.is_devrelease or parsed.is_prerelease,
        "min_compatible_version": MIN_COMPATIBLE_VERSION,
    }