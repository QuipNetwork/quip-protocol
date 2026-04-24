"""
Version management for QuIP Protocol.

This module provides centralized access to the package version and protocol version.
"""

import importlib.metadata
from typing import Any, Dict, Mapping, Optional, TypeVar

from packaging.version import InvalidVersion, parse as parse_version

_PeerInfo = TypeVar("_PeerInfo")

__version__: Optional[str] = None

# Protocol version - increment when making breaking changes to:
# - Block serialization format
# - Network message format
# - Consensus rules
PROTOCOL_VERSION = 2

# Minimum application version we will accept from peers.
# Bump this when a release introduces changes that older nodes cannot handle.
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
            # Fallback to development version
            __version__ = "0.1.20"

    return __version__


def _major_minor(version_str: str) -> str:
    """Extract 'major.minor' prefix from a version string."""
    return ".".join(version_str.split(".")[:2])


def is_version_compatible(peer_version_str: Optional[str]) -> bool:
    """Check whether a peer's application version is compatible.

    Compatibility rules (evaluated in order):
    1. Peer must report a parseable version string.
    2. Peer version must be >= MIN_COMPATIBLE_VERSION.
    3. Peer must share the same major.minor as the local version
       (e.g. 0.0.8 and 0.0.9 are compatible; 0.0.x and 0.1.x are not).

    Args:
        peer_version_str: Semantic version string reported by the peer.

    Returns:
        True if the peer is compatible, False otherwise. Missing or
        unparseable values are treated as incompatible — a peer that
        can't (or won't) announce a version cannot bypass the gate.
    """
    if not peer_version_str:
        return False
    try:
        peer_ver = parse_version(peer_version_str)
    except InvalidVersion:
        return False
    min_ver = parse_version(MIN_COMPATIBLE_VERSION)
    if peer_ver < min_ver:
        return False
    local_mm = _major_minor(get_version())
    peer_mm = _major_minor(peer_version_str)
    return local_mm == peer_mm


def version_from_descriptor(
    descriptor: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Pull ``runtime.quip_version`` out of a NodeDescriptor dict.

    Callers that receive a descriptor (JOIN response, STATUS response,
    peer_versions map) use this to seed ``peer_versions`` without
    waiting for the next inbound heartbeat.

    Returns None for missing, empty, or malformed descriptors — the
    version gate will then treat the peer as unknown until a heartbeat
    lands.
    """
    if not descriptor:
        return None
    runtime = descriptor.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    version_str = runtime.get("quip_version")
    if not version_str:
        return None
    return version_str


def select_compatible_peers(
    peers: Mapping[str, _PeerInfo],
    peer_versions: Mapping[str, Optional[str]],
) -> Dict[str, _PeerInfo]:
    """Return the subset of ``peers`` whose recorded version is compatible.

    Peers whose version has not yet been observed (e.g. just after a
    JOIN, before the first heartbeat) are excluded — we must not sync
    blocks from a peer we haven't positively version-gated.

    Args:
        peers: Mapping from peer address to peer info.
        peer_versions: Mapping from peer address to the last-seen
            version string (or None if not yet observed).

    Returns:
        A new dict containing only the compatible peers.
    """
    return {
        peer: info for peer, info in peers.items()
        if is_version_compatible(peer_versions.get(peer))
    }


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