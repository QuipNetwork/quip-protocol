"""
Version management for QuIP Protocol.

This module provides centralized access to the package version.
"""

import importlib.metadata
from typing import Optional

__version__: Optional[str] = None


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


def get_version_info() -> dict:
    """
    Get detailed version information.

    Returns:
        Dictionary with version details
    """
    version = get_version()
    return {
        "version": version,
        "major": int(version.split(".")[0]) if version != "0.1.0-dev" else 0,
        "minor": int(version.split(".")[1]) if version != "0.1.0-dev" else 1,
        "patch": int(version.split(".")[2].split("-")[0]) if version != "0.1.0-dev" else 0,
        "is_dev": version.endswith("-dev")
    }