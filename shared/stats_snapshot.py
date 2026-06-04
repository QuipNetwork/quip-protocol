"""Atomic stats snapshot file for cross-process telemetry.

The controller process owns one `StatsSnapshotWriter` that periodically
serializes a snapshot dict to a JSON file via tmp-file + `os.replace`.
The telemetry sibling process calls `read_snapshot(path)` on each
incoming stats request. No live IPC; the file is the channel.

Multi-process aggregator support (v0.2 Docker entrypoint): each child
quip-miner writes to a per-kind file (`telemetry-stats-cpu.json`,
`telemetry-stats-gpu.json`, `telemetry-stats-qpu.json`) inside a
shared snapshot directory. The single telemetry sibling reads all of
them via `read_all_snapshots(dir)` + `merge_snapshots(...)` and
exposes a unified `/api/v1` surface to dashboards.

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
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Filename prefix the aggregator reader globs for. Each child controller
# picks a unique suffix (typically the miner_kind: cpu/gpu/qpu) and
# writes to `<dir>/<SNAPSHOT_FILENAME_PREFIX><suffix>.json`.
SNAPSHOT_FILENAME_PREFIX = "telemetry-stats-"
SNAPSHOT_FILENAME_SUFFIX = ".json"


def snapshot_filename_for(kind: str) -> str:
    """Return the canonical snapshot filename for a given kind label.

    `kind` should be a stable identifier per controller process
    (typically `cpu` / `gpu` / `qpu`). Multiple processes writing the
    same kind would race the same file; callers must keep kinds
    distinct within a snapshot directory.
    """
    safe = "".join(c for c in kind.lower() if c.isalnum() or c == "-")
    if not safe:
        safe = "default"
    return f"{SNAPSHOT_FILENAME_PREFIX}{safe}{SNAPSHOT_FILENAME_SUFFIX}"


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


def read_all_snapshots(snapshot_dir: os.PathLike) -> list[dict[str, Any]]:
    """Read every `telemetry-stats-*.json` file in `snapshot_dir`.

    Returns a list of parsed snapshot dicts in deterministic order
    (sorted by filename, which sorts by kind alphabetically: cpu, gpu,
    qpu). Missing / corrupt files are skipped silently — the merge
    step degrades gracefully when a child is starting up or its
    writer is mid-replace.

    Returns `[]` when the directory doesn't exist or contains no
    matching files; the caller treats that the same as "no snapshots
    available" (stale 503 from the API surface).
    """
    snap_dir = Path(snapshot_dir)
    if not snap_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    pattern = f"{SNAPSHOT_FILENAME_PREFIX}*{SNAPSHOT_FILENAME_SUFFIX}"
    for path in sorted(snap_dir.glob(pattern)):
        snap = read_snapshot(path)
        if snap is not None:
            out.append(snap)
    return out


# Controller counter keys that are summed across all snapshots. Each
# child has its own pool, its own dispatch loop, and its own counter
# tally; the operator-facing total is the sum.
_SUMMED_CONTROLLER_COUNTERS: tuple[str, ...] = (
    "heads_observed",
    "contexts_dispatched",
    "results_received",
    "proofs_submitted",
    "stale_drops",
    "submission_errors",
    "zero_seed_snapshots_dropped",
    "heads_same_key_skipped",
    "none_snapshots_seen",
    "duplicate_result_drops",
    "proofs_unverified",
)


def merge_snapshots(snapshots: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Combine N per-kind snapshots into a single unified view.

    Merge rules:

      * `controller.<counter>` (heads_observed, contexts_dispatched,
        proofs_submitted, …) — SUMMED across snapshots. Operators care
        about total work performed by the container, not per-mode.
      * `controller.active_url` — taken from the first snapshot that
        supplies one (it's per-pool; each child owns its own pool and
        the value isn't meaningful to aggregate).
      * `node_id`, `ss58_address`, `account_id_hex`, `descriptor`,
        `miner_survey`, `attempts_dir` — taken from the first snapshot
        that supplies a non-empty value. These describe the container
        as a whole (same signer, same hardware host, same JSONL store)
        and merging beyond first-wins would invent contradictions.
      * `miners` — unioned by `id`. Each child reports only its own
        worker handles; the merged view is the full inventory.
      * New `modes` field — per-mode breakdown so the dashboard can
        distinguish "cpu mining produced 4 of 10 proofs" from
        "everything came from gpu". Keyed by the snapshot's `mode`
        field (set by the writer); missing mode falls back to a
        synthetic `unknown.<i>` slot.

    Returns `None` when `snapshots` is empty — telemetry callers
    should surface this as a 503 just like the single-snapshot path
    returns when the file is missing or corrupt.
    """
    snaps = [s for s in snapshots if s]
    if not snaps:
        return None

    summed: dict[str, int] = {k: 0 for k in _SUMMED_CONTROLLER_COUNTERS}
    first_active_url: Optional[str] = None
    for s in snaps:
        c = s.get("controller") or {}
        for k in _SUMMED_CONTROLLER_COUNTERS:
            v = c.get(k)
            if isinstance(v, (int, float)):
                summed[k] += int(v)
        if first_active_url is None and c.get("active_url"):
            first_active_url = c.get("active_url")

    merged_controller = dict(summed)
    merged_controller["active_url"] = first_active_url

    def _first_nonempty(field: str, default: Any = None) -> Any:
        for s in snaps:
            v = s.get(field)
            if v not in (None, "", {}, []):
                return v
        return default

    seen_ids: set[str] = set()
    union_miners: list[dict[str, str]] = []
    for s in snaps:
        for m in s.get("miners") or []:
            mid = m.get("id")
            if isinstance(mid, str) and mid and mid not in seen_ids:
                seen_ids.add(mid)
                union_miners.append(dict(m))

    modes: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(snaps):
        mode = s.get("mode") or f"unknown.{i}"
        modes[mode] = {
            "controller": s.get("controller") or {},
            "miners": s.get("miners") or [],
        }

    return {
        "controller": merged_controller,
        "node_id": _first_nonempty("node_id"),
        "ss58_address": _first_nonempty("ss58_address"),
        "account_id_hex": _first_nonempty("account_id_hex"),
        "miners": union_miners,
        "descriptor": _first_nonempty("descriptor", default={}),
        "miner_survey": _first_nonempty("miner_survey", default={}),
        "attempts_dir": _first_nonempty("attempts_dir"),
        "modes": modes,
    }
