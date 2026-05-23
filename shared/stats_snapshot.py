"""Atomic stats snapshot file for cross-process telemetry.

The controller process owns one `StatsSnapshotWriter` that periodically
serializes a snapshot dict to a JSON file via tmp-file + `os.replace`.
The telemetry sibling process calls `read_snapshot(path)` on each
incoming stats request. No live IPC; the file is the channel.

Atomic semantics:
    * Write goes to `<path>.tmp` first.
    * `os.replace(<path>.tmp, <path>)` is atomic on POSIX (and on
      Windows for files on the same filesystem). A reader never sees
      a partial file.

Robustness:
    * `read_snapshot` returns `None` for missing or corrupt files —
      both are normal transient states (controller starting up, reader caught
      a partial write on a non-atomic filesystem, etc.). Callers should treat
      None as "stats unavailable right now" and respond with a generic
      503/stale indicator.
    * The writer catches exceptions in `get_snapshot()`, logs them,
      and continues. One buggy stats collector doesn't break the
      writer loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StatsSnapshotWriter:
    """Periodically write a stats dict atomically to `path`.

    Args:
        path: Target file location.
        get_snapshot: Callable returning a JSON-serializable dict.
        interval_s: How often to write. Default 1.0s.
    """

    def __init__(
        self,
        path: os.PathLike,
        get_snapshot: Callable[[], dict[str, Any]],
        interval_s: float = 1.0,
    ) -> None:
        self._path = Path(path)
        self._tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        self._get_snapshot = get_snapshot
        self._interval_s = float(interval_s)

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Run the writer loop until `shutdown_event` is set."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while not shutdown_event.is_set():
            try:
                snapshot = self._get_snapshot()
                self._write_atomic(snapshot)
            except Exception:
                logger.exception("stats snapshot write failed; will retry next interval")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=self._interval_s)
                return
            except asyncio.TimeoutError:
                pass

    def _write_atomic(self, snapshot: dict[str, Any]) -> None:
        with open(self._tmp_path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh)
        os.replace(self._tmp_path, self._path)


def read_snapshot(path: os.PathLike) -> Optional[dict[str, Any]]:
    """Read and parse `path`; return None on any error.

    Missing files and corrupt JSON both return None — both are
    expected transient states (controller starting up, reader caught
    a partial write on a non-atomic filesystem). Callers should treat
    None as "stats unavailable right now" and respond with a generic
    503/stale indicator.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    except OSError:
        # Other I/O errors (permissions, FS unmounted, etc.) — log loudly
        # since these usually indicate misconfiguration the operator must see.
        logger.exception("read_snapshot: unexpected I/O error reading %s", path)
        return None
