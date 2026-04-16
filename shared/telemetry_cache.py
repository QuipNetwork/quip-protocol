"""
Read-only cache of telemetry directory contents.

Periodically scans the telemetry directory written by TelemetryManager
and serves cached data to REST and QUIC telemetry endpoints.

SPDX-License-Identifier: AGPL-3.0-or-later
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class EpochInfo:
    """Cached metadata for one epoch directory."""

    epoch: str
    block_count: int = 0
    first_block: int = 0
    last_block: int = 0


class TelemetryCache:
    """Read-only cache of telemetry directory contents.

    Refreshes periodically by scanning the telemetry directory.
    All accessor methods return cached data with no filesystem I/O.

    Args:
        telemetry_dir: Path to the telemetry directory.
        refresh_interval: Seconds between filesystem scans.
        logger: Logger instance.
    """

    def __init__(
        self,
        telemetry_dir: str = "telemetry",
        refresh_interval: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ):
        self._dir = Path(telemetry_dir)
        self._refresh_interval = refresh_interval
        self._logger = logger or logging.getLogger(__name__)
        self._refresh_task: Optional[asyncio.Task] = None

        # Cached state
        self._epochs: List[str] = []
        self._epoch_info: Dict[str, EpochInfo] = {}
        self._latest_epoch: str = ""
        self._latest_block_index: int = 0
        self._total_blocks: int = 0
        self._nodes_data: Optional[dict] = None
        self._nodes_mtime: float = 0.0
        self._latest_block_data: Optional[dict] = None

        # Block LRU cache: (epoch, block_index) -> parsed JSON
        self._block_cache: Dict[tuple, dict] = {}
        self._block_cache_max = 64

        # SSE callbacks
        self.on_new_block: Optional[Callable[[str, int, dict], None]] = None
        self.on_nodes_changed: Optional[Callable[[dict], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic refresh loop."""
        await self._refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        self._logger.info(
            "TelemetryCache started (dir=%s, interval=%.1fs)",
            self._dir,
            self._refresh_interval,
        )

    async def stop(self) -> None:
        """Cancel the refresh loop."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        """Background loop that calls refresh() periodically."""
        try:
            while True:
                await asyncio.sleep(self._refresh_interval)
                try:
                    await self._refresh()
                except Exception as exc:
                    self._logger.warning("TelemetryCache refresh failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _refresh(self) -> None:
        """Scan the telemetry directory and update cached state."""
        if not self._dir.is_dir():
            return

        self._refresh_nodes()
        self._refresh_epochs()

    def _refresh_nodes(self) -> None:
        """Re-read nodes.json if its mtime changed."""
        nodes_path = self._dir / "nodes.json"
        if not nodes_path.is_file():
            return
        try:
            mtime = os.path.getmtime(nodes_path)
        except OSError:
            return
        if mtime == self._nodes_mtime:
            return

        old_data = self._nodes_data
        self._nodes_data = self._read_json(nodes_path)
        self._nodes_mtime = mtime

        if (
            self.on_nodes_changed
            and self._nodes_data
            and self._nodes_data != old_data
        ):
            self.on_nodes_changed(self._nodes_data)

    def _refresh_epochs(self) -> None:
        """Scan epoch directories and detect new blocks."""
        epochs: List[str] = []
        epoch_info: Dict[str, EpochInfo] = {}
        total_blocks = 0

        try:
            entries = os.listdir(self._dir)
        except OSError:
            return

        for entry in entries:
            entry_path = self._dir / entry
            if not entry_path.is_dir() or not entry.isdigit():
                continue
            epochs.append(entry)

            block_indices = self._list_block_indices(entry_path)
            if block_indices:
                info = EpochInfo(
                    epoch=entry,
                    block_count=len(block_indices),
                    first_block=min(block_indices),
                    last_block=max(block_indices),
                )
            else:
                info = EpochInfo(epoch=entry)
            epoch_info[entry] = info
            total_blocks += info.block_count

        epochs.sort()
        self._epochs = epochs
        self._epoch_info = epoch_info
        self._total_blocks = total_blocks

        # Determine latest block
        prev_epoch = self._latest_epoch
        prev_index = self._latest_block_index

        if epochs:
            latest_ep = epochs[-1]
            self._latest_epoch = latest_ep
            self._latest_block_index = epoch_info[latest_ep].last_block
        else:
            self._latest_epoch = ""
            self._latest_block_index = 0

        # Detect new block
        is_new = (
            self._latest_epoch != prev_epoch
            or self._latest_block_index != prev_index
        ) and self._latest_epoch and self._latest_block_index > 0

        if is_new or self._latest_block_data is None:
            self._load_latest_block(is_new)

    def _load_latest_block(self, notify: bool) -> None:
        """Read the latest block file and optionally fire the callback."""
        if not self._latest_epoch or self._latest_block_index <= 0:
            return

        block_path = (
            self._dir / self._latest_epoch / f"{self._latest_block_index}.json"
        )
        data = self._read_json(block_path)
        if data is None:
            return

        self._latest_block_data = data
        key = (self._latest_epoch, self._latest_block_index)
        self._block_cache[key] = data
        self._evict_block_cache()

        if notify and self.on_new_block:
            self.on_new_block(
                self._latest_epoch, self._latest_block_index, data,
            )

    # ------------------------------------------------------------------
    # Accessors (no I/O, return cached data)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return telemetry status summary."""
        nodes_updated = ""
        node_count = 0
        active_count = 0
        if self._nodes_data:
            nodes_updated = self._nodes_data.get("updated_at", "")
            node_count = self._nodes_data.get("node_count", 0)
            active_count = self._nodes_data.get("active_count", 0)

        return {
            "epochs": list(self._epochs),
            "latest_epoch": self._latest_epoch,
            "latest_block_index": self._latest_block_index,
            "total_blocks": self._total_blocks,
            "node_count": node_count,
            "active_node_count": active_count,
            "nodes_updated_at": nodes_updated,
        }

    def get_nodes(self) -> Optional[dict]:
        """Return parsed nodes.json data."""
        return self._nodes_data

    def get_epochs(self) -> List[dict]:
        """Return epoch listing with block counts."""
        return [
            {
                "epoch": info.epoch,
                "block_count": info.block_count,
                "first_block": info.first_block,
                "last_block": info.last_block,
            }
            for info in (self._epoch_info.get(e) for e in self._epochs)
            if info is not None
        ]

    def get_block(self, epoch: str, block_index: int) -> Optional[dict]:
        """Return a single block's telemetry data.

        Uses an LRU cache; reads from disk on cache miss.
        """
        key = (epoch, block_index)
        cached = self._block_cache.get(key)
        if cached is not None:
            return cached

        if epoch not in self._epoch_info:
            return None

        block_path = self._dir / epoch / f"{block_index}.json"
        data = self._read_json(block_path)
        if data is not None:
            self._block_cache[key] = data
            self._evict_block_cache()
        return data

    def get_latest(self) -> Optional[dict]:
        """Return the most recent block telemetry with metadata."""
        if not self._latest_block_data:
            return None
        return {
            "epoch": self._latest_epoch,
            "block_index": self._latest_block_index,
            "block": self._latest_block_data,
        }

    # ------------------------------------------------------------------
    # ETag generation
    # ------------------------------------------------------------------

    def status_etag(self) -> str:
        """ETag for the status endpoint."""
        return (
            f"{self._latest_epoch}:{self._latest_block_index}:{self._total_blocks}"
        )

    def nodes_etag(self) -> str:
        """ETag for the nodes endpoint."""
        if self._nodes_data:
            return self._nodes_data.get("updated_at", "")
        return ""

    @staticmethod
    def block_etag(block_data: dict) -> str:
        """ETag for a block endpoint (uses block_hash, immutable)."""
        return block_data.get("block_hash", "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: Path) -> Optional[dict]:
        """Read and parse a JSON file, returning None on failure."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.debug("Failed to read %s: %s", path, exc)
            return None

    @staticmethod
    def _list_block_indices(epoch_dir: Path) -> List[int]:
        """List block indices (from filenames like 1.json) in an epoch dir."""
        indices = []
        try:
            for name in os.listdir(epoch_dir):
                if name.endswith(".json"):
                    stem = name[:-5]
                    if stem.isdigit():
                        indices.append(int(stem))
        except OSError:
            pass
        return indices

    def _evict_block_cache(self) -> None:
        """Evict oldest entries if block cache exceeds max size."""
        while len(self._block_cache) > self._block_cache_max:
            oldest_key = next(iter(self._block_cache))
            del self._block_cache[oldest_key]
