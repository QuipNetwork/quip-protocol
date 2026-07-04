"""Sync progress tracking + human-readable ETA for a syncing validator node.

Pure logic, no I/O: the pool feeds ``observe()`` one ``get_sync_state``
sample per probe and logs the returned line. The rate is computed over a
sliding window of recent samples so the ETA adapts as sync speeds up or
stalls, and ``highest_block`` is re-read each sample so the ETA tracks
chain growth during the sync.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Optional

# Below this rate the node is effectively not advancing — report a stall
# instead of an absurd multi-year ETA.
_STALL_RATE_BLOCKS_PER_S = 0.01


class SyncProgress:
    """Blocks/sec + ETA estimator over a sliding sample window.

    Args:
        window: Number of ``(time, current_block)`` samples the rate is
            computed across. At the pool's 10s probe cadence the default
            6 gives a ~1-minute moving average.
    """

    def __init__(self, window: int = 6) -> None:
        self._samples: deque[tuple[float, int]] = deque(maxlen=window)

    def observe(self, state: Mapping[str, Any], now_s: float) -> str:
        """Record one sync-state sample and return the progress log line.

        Args:
            state: ``get_sync_state`` dict — reads ``current_block``,
                ``highest_block``, ``peers``.
            now_s: Monotonic timestamp of the sample (``time.monotonic()``).

        Returns:
            A line like ``validator syncing: block 812,340 / 900,102
            (90.2%) — ~4m 30s remaining (312 blk/s, peers=12)``.
        """
        current = int(state.get("current_block") or 0)
        highest = int(state.get("highest_block") or 0)
        peers = int(state.get("peers") or 0)
        self._samples.append((now_s, current))
        pct = (100.0 * current / highest) if highest > 0 else 0.0
        return (
            f"validator syncing: block {current:,} / {highest:,} "
            f"({pct:.1f}%) — {self._eta_text(highest, peers)}"
        )

    def rate_blocks_per_s(self) -> Optional[float]:
        """Blocks/sec across the sample window; None with <2 samples."""
        if len(self._samples) < 2:
            return None
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        if t1 <= t0:
            return None
        return (b1 - b0) / (t1 - t0)

    def _eta_text(self, highest: int, peers: int) -> str:
        if peers == 0:
            return "sync stalled (0 peers)"
        rate = self.rate_blocks_per_s()
        if rate is None:
            return "~calculating… remaining"
        if rate <= _STALL_RATE_BLOCKS_PER_S:
            return f"sync stalled ({peers} peers, 0 blk/s)"
        remaining_s = max(0, highest - self._samples[-1][1]) / rate
        rate_txt = f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}"
        return (
            f"~{_format_duration(remaining_s)} remaining "
            f"({rate_txt} blk/s, peers={peers})"
        )


def _format_duration(seconds: float) -> str:
    """Compact human duration: ``45s`` / ``4m 30s`` / ``2h 15m``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"
