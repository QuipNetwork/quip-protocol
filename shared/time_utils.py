"""UTC time utilities for quantum blockchain.

This module provides centralized UTC time functions to ensure all nodes
use consistent timestamps regardless of their local timezone.
"""

from datetime import datetime, timezone


def utc_timestamp() -> int:
    """Get current UTC timestamp as integer seconds since epoch.

    Returns:
        int: UTC timestamp in seconds since Unix epoch
    """
    return int(datetime.now(timezone.utc).timestamp())
