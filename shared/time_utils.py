"""UTC time utilities for quantum blockchain.

This module provides centralized UTC time functions to ensure all nodes
use consistent timestamps regardless of their local timezone.
"""

import time
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Time synchronization constants
MAX_CLOCK_DRIFT_SECONDS = 300  # 5 minutes maximum allowed drift
TIMESTAMP_TOLERANCE_SECONDS = 60  # 1 minute tolerance for block timestamps
NETWORK_TIME_SYNC_INTERVAL = 300  # 5 minutes between network time sync checks


def utc_timestamp() -> int:
    """Get current UTC timestamp as integer seconds since epoch.
    
    Returns:
        int: UTC timestamp in seconds since Unix epoch
    """
    return int(datetime.now(timezone.utc).timestamp())


def utc_timestamp_float() -> float:
    """Get current UTC timestamp as float seconds since epoch.
    
    Returns:
        float: UTC timestamp in seconds since Unix epoch with sub-second precision
    """
    return datetime.now(timezone.utc).timestamp()


def utc_timestamp_ms() -> int:
    """Get current UTC timestamp in milliseconds.
    
    Returns:
        int: UTC timestamp in milliseconds since Unix epoch
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def format_utc_timestamp(timestamp: int) -> str:
    """Format a UTC timestamp for human-readable display.
    
    Args:
        timestamp: UTC timestamp in seconds since epoch
        
    Returns:
        str: Formatted timestamp string in ISO format
    """
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def validate_timestamp(timestamp: int, reference_time: Optional[int] = None) -> bool:
    """Validate that a timestamp is within acceptable bounds.
    
    Args:
        timestamp: The timestamp to validate
        reference_time: Reference time to compare against (defaults to current UTC time)
        
    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    if reference_time is None:
        reference_time = utc_timestamp()
    
    # Check if timestamp is too far in the future
    if timestamp > reference_time + TIMESTAMP_TOLERANCE_SECONDS:
        logger.warning(f"Timestamp {timestamp} is too far in the future (reference: {reference_time})")
        return False
    
    # Check if timestamp is too far in the past (allow reasonable historical blocks)
    # We allow up to 24 hours in the past for blockchain synchronization
    if timestamp < reference_time - (24 * 3600):
        logger.warning(f"Timestamp {timestamp} is too far in the past (reference: {reference_time})")
        return False
    
    return True


def validate_block_timestamp(timestamp: int, previous_timestamp: int, 
                           current_time: Optional[int] = None) -> bool:
    """Validate a block timestamp against blockchain rules.
    
    Args:
        timestamp: The block timestamp to validate
        previous_timestamp: Timestamp of the previous block
        current_time: Current UTC time (defaults to current UTC time)
        
    Returns:
        bool: True if timestamp is valid for blockchain, False otherwise
    """
    if current_time is None:
        current_time = utc_timestamp()
    
    # Block timestamp must be after previous block
    if timestamp <= previous_timestamp:
        logger.info(f"Block timestamp {timestamp} is not after previous block timestamp {previous_timestamp}")
        return False
    
    # Block timestamp cannot be too far in the future
    if timestamp > current_time + TIMESTAMP_TOLERANCE_SECONDS:
        logger.info(f"Block timestamp {timestamp} is too far in the future (current: {current_time})")
        return False
    
    return True


def get_network_time_offset(peer_timestamps: list[int]) -> int:
    """Calculate network time offset based on peer timestamps.
    
    This function helps detect if the local clock is significantly out of sync
    with the network by comparing against peer timestamps.
    
    Args:
        peer_timestamps: List of recent timestamps from network peers
        
    Returns:
        int: Estimated offset in seconds (positive means local clock is ahead)
    """
    if not peer_timestamps:
        return 0
    
    local_time = utc_timestamp()
    
    # Calculate median peer time to avoid outliers
    peer_timestamps.sort()
    n = len(peer_timestamps)
    if n % 2 == 0:
        median_peer_time = (peer_timestamps[n//2 - 1] + peer_timestamps[n//2]) // 2
    else:
        median_peer_time = peer_timestamps[n//2]
    
    offset = local_time - median_peer_time
    
    if abs(offset) > MAX_CLOCK_DRIFT_SECONDS:
        logger.warning(f"Large clock drift detected: {offset} seconds from network median")
    
    return offset


def is_clock_synchronized(peer_timestamps: list[int]) -> bool:
    """Check if local clock is synchronized with the network.
    
    Args:
        peer_timestamps: List of recent timestamps from network peers
        
    Returns:
        bool: True if clock is synchronized within acceptable bounds
    """
    if not peer_timestamps:
        return True  # No peers to compare against
    
    offset = get_network_time_offset(peer_timestamps)
    return abs(offset) <= MAX_CLOCK_DRIFT_SECONDS


def sync_time_with_network(peer_timestamps: list[int]) -> Optional[int]:
    """Get network-synchronized time estimate.
    
    This function provides a time estimate that's synchronized with the network
    based on peer timestamps. Use this for critical blockchain operations.
    
    Args:
        peer_timestamps: List of recent timestamps from network peers
        
    Returns:
        Optional[int]: Network-synchronized timestamp, or None if insufficient data
    """
    if len(peer_timestamps) < 3:
        # Not enough peers for reliable synchronization
        return None
    
    offset = get_network_time_offset(peer_timestamps)
    
    # If offset is within tolerance, use local time
    if abs(offset) <= TIMESTAMP_TOLERANCE_SECONDS:
        return utc_timestamp()
    
    # Otherwise, adjust local time by the offset
    adjusted_time = utc_timestamp() - offset
    logger.info(f"Using network-synchronized time with offset: {-offset} seconds")
    return adjusted_time


# Backward compatibility functions that log warnings
def deprecated_time_time() -> float:
    """Deprecated: Use utc_timestamp_float() instead."""
    logger.warning("Using deprecated time.time() - switch to utc_timestamp_float() for UTC consistency")
    return time.time()


def deprecated_int_time_time() -> int:
    """Deprecated: Use utc_timestamp() instead."""
    logger.warning("Using deprecated int(time.time()) - switch to utc_timestamp() for UTC consistency")
    return int(time.time())
