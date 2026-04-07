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
PROTOCOL_VERSION = 1

# Minimum application version we will accept from peers.
# Bump this when a release introduces changes that older nodes cannot handle.
MIN_COMPATIBLE_VERSION = "0.0.1"


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
            # Fallback to development version
            __version__ = "0.1.0-dev"

    return __version__


def _major_minor(version_str: str) -> str:
    """Extract 'major.minor' prefix from a version string."""
    return ".".join(version_str.split(".")[:2])


def is_version_compatible(peer_version_str: str) -> bool:
    """Check whether a peer's application version is compatible.

    Compatibility rules (evaluated in order):
    1. Peer version must be >= MIN_COMPATIBLE_VERSION.
    2. Peer must share the same major.minor as the local version
       (e.g. 0.0.8 and 0.0.9 are compatible; 0.0.x and 0.1.x are not).

    Args:
        peer_version_str: Semantic version string reported by the peer.

    Returns:
        True if the peer is compatible, False otherwise.
    """
    peer_ver = parse_version(peer_version_str)
    min_ver = parse_version(MIN_COMPATIBLE_VERSION)
    if peer_ver < min_ver:
        return False
    local_mm = _major_minor(get_version())
    peer_mm = _major_minor(peer_version_str)
    return local_mm == peer_mm


def get_version_info() -> dict:
    """
    Get detailed version information.

    Returns:
        Dictionary with version details
    """
    ver = get_version()
    return {
        "version": ver,
        "major": int(ver.split(".")[0]) if ver != "0.1.0-dev" else 0,
        "minor": int(ver.split(".")[1]) if ver != "0.1.0-dev" else 1,
        "patch": int(ver.split(".")[2].split("-")[0]) if ver != "0.1.0-dev" else 0,
        "is_dev": ver.endswith("-dev"),
        "min_compatible_version": MIN_COMPATIBLE_VERSION,
    }