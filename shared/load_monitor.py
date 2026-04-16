"""Node load monitoring for connection management decisions.

Tracks connection count, system CPU, memory, and queue depths to
determine when a node is overloaded and how many connections to shed.

Usage::

    monitor = LoadMonitor(max_connections=50)
    monitor.update(connection_count=45, block_queue=800, gossip_queue=600)

    if monitor.is_overloaded():
        n = monitor.connections_to_shed()
        # shed n connections via MIGRATE
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeLoad:
    """Snapshot of node load metrics at a point in time."""
    connection_count: int = 0
    max_connections: int = 50
    cpu_load_avg: float = 0.0
    memory_percent: float = 0.0
    block_queue_depth: int = 0
    gossip_queue_depth: int = 0
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "connection_count": self.connection_count,
            "max_connections": self.max_connections,
            "cpu_load_avg": round(self.cpu_load_avg, 2),
            "memory_percent": round(self.memory_percent, 1),
            "block_queue_depth": self.block_queue_depth,
            "gossip_queue_depth": self.gossip_queue_depth,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeLoad':
        return cls(
            connection_count=d.get("connection_count", 0),
            max_connections=d.get("max_connections", 50),
            cpu_load_avg=d.get("cpu_load_avg", 0.0),
            memory_percent=d.get("memory_percent", 0.0),
            block_queue_depth=d.get("block_queue_depth", 0),
            gossip_queue_depth=d.get("gossip_queue_depth", 0),
            timestamp=d.get("timestamp", 0.0),
        )


class LoadMonitor:
    """Monitors node load and decides when to shed connections.

    Uses high/low watermarks to create hysteresis — once overloaded,
    the node sheds connections until utilization drops below the low
    watermark, preventing rapid oscillation.

    Args:
        max_connections: Maximum peer connections allowed.
        high_watermark: Fraction (0-1) above which the node is overloaded.
        low_watermark: Fraction (0-1) below which shedding stops.
        cpu_threshold: 1-min load average above which CPU is overloaded.
        queue_threshold: Queue depth above which queues are overloaded.
    """

    def __init__(
        self,
        max_connections: int = 50,
        high_watermark: float = 0.8,
        low_watermark: float = 0.5,
        cpu_threshold: float = 0.0,
        queue_threshold: int = 800,
    ):
        self.max_connections = max_connections
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.cpu_threshold = cpu_threshold or os.cpu_count() or 4
        self.queue_threshold = queue_threshold

        self._connection_count: int = 0
        self._block_queue_depth: int = 0
        self._gossip_queue_depth: int = 0
        self._shedding: bool = False

    def update(
        self,
        connection_count: int,
        block_queue: int = 0,
        gossip_queue: int = 0,
    ) -> None:
        """Update current load metrics."""
        self._connection_count = connection_count
        self._block_queue_depth = block_queue
        self._gossip_queue_depth = gossip_queue

    def snapshot(self) -> NodeLoad:
        """Capture current load metrics as a snapshot."""
        cpu_load = _get_cpu_load()
        mem_pct = _get_memory_percent()
        return NodeLoad(
            connection_count=self._connection_count,
            max_connections=self.max_connections,
            cpu_load_avg=cpu_load,
            memory_percent=mem_pct,
            block_queue_depth=self._block_queue_depth,
            gossip_queue_depth=self._gossip_queue_depth,
        )

    def connection_utilization(self) -> float:
        """Current connection utilization as a fraction (0-1)."""
        if self.max_connections <= 0:
            return 0.0
        return self._connection_count / self.max_connections

    def is_overloaded(self) -> bool:
        """Whether the node is currently overloaded.

        Uses hysteresis: becomes overloaded when any metric exceeds
        the high watermark, stays overloaded until all metrics drop
        below the low watermark.
        """
        conn_util = self.connection_utilization()

        if self._shedding:
            # Stay in shedding mode until below low watermark
            if conn_util <= self.low_watermark:
                self._shedding = False
            return self._shedding

        # Check if any metric exceeds high watermark
        if conn_util >= self.high_watermark:
            self._shedding = True
            return True

        # Check CPU load
        cpu_load = _get_cpu_load()
        if cpu_load > self.cpu_threshold:
            self._shedding = True
            return True

        # Check queue depths
        if (self._block_queue_depth > self.queue_threshold
                or self._gossip_queue_depth > self.queue_threshold):
            self._shedding = True
            return True

        return False

    def connections_to_shed(self) -> int:
        """How many connections to drop to reach the low watermark.

        Returns 0 if not overloaded.
        """
        if not self._shedding:
            return 0
        target = int(self.max_connections * self.low_watermark)
        excess = self._connection_count - target
        return max(0, excess)

    def should_accept_join(self) -> bool:
        """Whether we should accept a new JOIN request."""
        return (
            self._connection_count < self.max_connections
            and not self._shedding
        )


def _get_cpu_load() -> float:
    """Get 1-minute CPU load average. Returns 0.0 on unsupported platforms."""
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0


def _get_memory_percent() -> float:
    """Get memory usage as a percentage. Best-effort, returns 0.0 on failure."""
    try:
        import resource
    except ImportError:
        return 0.0  # Expected on Windows
    try:
        import sys
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # rusage gives maxrss in bytes (macOS) or KB (Linux)
        rss_bytes = usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024

        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        total = pages * page_size
        if total > 0:
            return (rss_bytes / total) * 100.0
        return 0.0
    except (OSError, ValueError):
        return 0.0
