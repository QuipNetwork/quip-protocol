# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time-budget accounting for the CUDA stream-driver loop (QUI-867 diagnostic).

The driver loop in :mod:`GPU.base_cuda_sampler` is single-threaded and yields
from inside its poll loop, so it can only mark a slot READY (``upload_slot``)
while it is actually running. Since rc49 the kernel waits indefinitely for a
READY slot, so any wall-clock the driver spends elsewhere converts directly into
GPU spin-wait. QUI-867's decay signature -- attempt rate falling while the core
clock stays pinned and board power *follows* throughput down -- is what that
looks like from outside.

This records where the driver's wall-clock actually goes, bucketed, and reports
it per window. The question it answers is not "what is slow" but "which bucket
GROWS as att/s falls", which is what names the root cause.

Disabled unless ``QUIP_DRIVER_BUDGET=1``, so it costs one attribute lookup in
the hot loop when off. Set ``QUIP_DRIVER_BUDGET_WINDOW`` to change the report
interval (default 60s) and ``QUIP_DRIVER_BUDGET_OUT`` to write JSONL.

Usage (from the driver loop)::

    budget = DriverBudget.from_env()
    with budget.phase("poll"):
        ctrl = cp.asnumpy(self._d_sf_ctrl)
    ...
    budget.tick_result()
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

ENV_ENABLE = "QUIP_DRIVER_BUDGET"
ENV_WINDOW = "QUIP_DRIVER_BUDGET_WINDOW"
ENV_OUT = "QUIP_DRIVER_BUDGET_OUT"

# Every bucket the driver's wall-clock can land in. "consumer" is the one the
# loop cannot measure directly from a phase() block -- it is the span the
# generator sits suspended at its yield while the stream driver writes to the
# ring -- so tick_consumer() records it separately.
PHASES = ("poll", "download", "upload", "consumer", "throttle", "spin")


# The NVML throttle sleeps inside throttled_stream, i.e. while the driver
# generator is suspended at its yield -- so its cost already lands in the
# "consumer" span and cannot be timed with a phase() block from the loop.
# throttle_if_busy adds to this module-level accumulator instead, and the
# window report splits it back out of "consumer". Module-level because the
# scheduler has no handle on the budget instance.
_throttle_accum_s = 0.0
_throttle_events = 0


def record_throttle(seconds: float) -> None:
    """Record one NVML throttle sleep. Called by GPU.gpu_scheduler."""
    global _throttle_accum_s, _throttle_events
    _throttle_accum_s += seconds
    _throttle_events += 1


def _drain_throttle() -> tuple[float, int]:
    """Take and zero the accumulated throttle time and event count."""
    global _throttle_accum_s, _throttle_events
    s, n = _throttle_accum_s, _throttle_events
    _throttle_accum_s, _throttle_events = 0.0, 0
    return s, n


class _NullBudget:
    """No-op stand-in used when the diagnostic is off.

    Matches DriverBudget's surface so the driver loop never branches on None.
    """

    enabled = False

    @contextmanager
    def phase(self, _name: str):
        yield

    def tick_result(self) -> None:
        pass

    def tick_consumer(self, _seconds: float) -> None:
        pass

    def close(self) -> None:
        pass


class DriverBudget:
    """Accumulates per-phase wall-clock and emits a window report.

    Args:
        window_s: Seconds per reported window.
        out_path: Optional JSONL path; rows are also logged either way.
    """

    enabled = True

    def __init__(self, window_s: float = 60.0, out_path: Optional[str] = None):
        self._window_s = window_s
        self._out_path = out_path
        self._fh = open(out_path, "a", encoding="utf-8") if out_path else None
        self._t_start = time.monotonic()
        self._win_start = self._t_start
        self._win_index = 0
        self._totals = {p: 0.0 for p in PHASES}
        self._results = 0

    @classmethod
    def from_env(cls):
        """Build from QUIP_DRIVER_BUDGET* env, or a no-op when disabled."""
        if os.environ.get(ENV_ENABLE, "").strip() not in ("1", "true", "yes"):
            return _NullBudget()
        try:
            window = float(os.environ.get(ENV_WINDOW, "60"))
        except ValueError:
            window = 60.0
        out = os.environ.get(ENV_OUT) or None
        logger.info(
            "QUI-867 driver budget ENABLED (window=%.0fs, out=%s)", window, out,
        )
        return cls(window_s=window, out_path=out)

    @contextmanager
    def phase(self, name: str):
        """Time a block into bucket *name*."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._totals[name] += time.monotonic() - t0

    def tick_consumer(self, seconds: float) -> None:
        """Record time the generator sat suspended at its yield."""
        self._totals["consumer"] += seconds

    def tick_result(self) -> None:
        """Count one completed result and emit a window report if due."""
        self._results += 1
        now = time.monotonic()
        if now - self._win_start >= self._window_s:
            self._emit(now)

    def _emit(self, now: float) -> None:
        elapsed = now - self._win_start
        # Throttle slept inside the consumer span; move it to its own bucket so
        # "consumer" reports real ring-write cost and the two are separable.
        throttle_s, throttle_n = _drain_throttle()
        self._totals["throttle"] = throttle_s
        self._totals["consumer"] = max(0.0, self._totals["consumer"] - throttle_s)
        accounted = sum(self._totals.values())
        row = {
            "window": self._win_index,
            "uptime_min": round((now - self._t_start) / 60.0, 2),
            "elapsed_s": round(elapsed, 2),
            "att_per_s": round(self._results / elapsed, 4) if elapsed else 0.0,
            "results": self._results,
            "throttle_events": throttle_n,
        }
        # Percent of wall-clock per bucket: the shape that matters. A bucket
        # whose share climbs while att_per_s falls is the accumulator.
        for p in PHASES:
            row[f"{p}_pct"] = round(100.0 * self._totals[p] / elapsed, 2) if elapsed else 0.0
            row[f"{p}_ms_per_result"] = (
                round(1000.0 * self._totals[p] / self._results, 3)
                if self._results else 0.0
            )
        # Unaccounted = wall-clock the loop spent outside every phase() block.
        # A large or growing value means the instrumentation is missing the
        # real cost centre -- treat it as a finding, not noise.
        row["unaccounted_pct"] = round(100.0 * (elapsed - accounted) / elapsed, 2) if elapsed else 0.0

        logger.info(
            "[QUI-867 budget] win=%d up=%.1fmin att/s=%.2f | poll=%.1f%% "
            "dl=%.1f%% ul=%.1f%% consumer=%.1f%% throttle=%.1f%% spin=%.1f%% "
            "unacct=%.1f%%",
            row["window"], row["uptime_min"], row["att_per_s"],
            row["poll_pct"], row["download_pct"], row["upload_pct"],
            row["consumer_pct"], row["throttle_pct"], row["spin_pct"],
            row["unaccounted_pct"],
        )
        if self._fh:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

        self._win_index += 1
        self._win_start = now
        self._results = 0
        self._totals = {p: 0.0 for p in PHASES}

    def close(self) -> None:
        """Emit a final partial window and close the output file."""
        try:
            if self._results:
                self._emit(time.monotonic())
        finally:
            if self._fh:
                self._fh.close()
                self._fh = None


__all__ = ["DriverBudget", "PHASES", "ENV_ENABLE", "ENV_WINDOW", "ENV_OUT"]
