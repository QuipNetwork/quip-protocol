#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolate the PoW producer and measure whether its throughput decays with uptime.

QUI-867 diagnostic. The CUDA miner's attempt rate falls ~30% over ~4h while the
GPU clock stays pinned at boost and board power *follows* throughput down. Since
rc49 (``GPU/cuda_sa.cu``) the kernel waits indefinitely for a READY slot, so any
feeder starvation converts directly into GPU spin-wait -- i.e. exactly that
signature. That makes the producer the thing to rule in or out first.

This runs :class:`shared.ising_feeder.RandomIsingFeeder` with NO GPU, no kernel,
no worker, and no chain attached, and reports models/sec per time window. The
feeder is pure-CPU (a spawn ``ProcessPoolExecutor`` deriving nonces into h/J
dicts), so this reproduces on any machine -- no RTX 5090 required.

Reading the result:

* Rate stays FLAT  -> the feeder is healthy. The decay is downstream: the worker
  slows, the fixed-size SampleView ring fills, the stream driver blocks on the
  ring write, stops calling ``_try_queue``, no slot goes READY, kernel spins.
* Rate DECAYS      -> the feeder itself is the accumulator. Look inside the pool.

Usage:
    python tools/feeder_decay_probe.py --minutes 240
    python tools/feeder_decay_probe.py --minutes 240 --rate 3.0   # paced
    python tools/feeder_decay_probe.py --minutes 5 --nodes-limit 500  # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shared.allowed_value_spec import AllowedValueSet
from shared.ising_feeder import RandomIsingFeeder, _default_max_workers

# Mirrors GPU/gpu_miner.py:47 -- the buffer depth the CUDA path actually runs.
GPU_FEEDER_BUFFER_SIZE = 16

# The chain's ternary field spec (h in {-1, 0, +1}). build_feeder() rejects a
# None on the PoW path; the GPU path passes the topology's real spec, and
# ternary is the shape that exercises the expensive scalar dict build.
TERNARY_H = AllowedValueSet((-1, 0, 1))


def _rss_bytes(pid: int) -> int:
    """Resident set size for *pid* in bytes, or 0 if it can't be read.

    Uses psutil when available and falls back to `ps`, so the probe runs on a
    bare interpreter without adding a dependency for a diagnostic.
    """
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip() or 0) * 1024
    except Exception:
        return 0


def _pool_rss_bytes(feeder: RandomIsingFeeder) -> tuple[int, int]:
    """(total RSS of the generator worker processes, worker count).

    Reaches into the executor's private ``_processes`` because the probe needs
    to see pool-side growth that no public API exposes. Tolerates the attribute
    being absent or None (it flips to None once the executor shuts down).
    """
    procs = getattr(feeder._pool, "_processes", None) or {}
    total = 0
    n = 0
    for proc in list(procs.values()):
        if proc.pid is not None:
            total += _rss_bytes(proc.pid)
            n += 1
    return total, n


@dataclass
class Window:
    """One reporting window's accumulated observations."""

    index: int
    started_at: float
    popped: int = 0
    pop_waits_s: list[float] = field(default_factory=list)

    def summarize(self, now: float, feeder: RandomIsingFeeder,
                  drained_delta: int) -> dict:
        elapsed = now - self.started_at
        waits = sorted(self.pop_waits_s)
        pool_rss, workers = _pool_rss_bytes(feeder)
        stats = feeder.stats()

        def pct(p: float) -> float:
            if not waits:
                return 0.0
            k = min(len(waits) - 1, int(p * len(waits)))
            return waits[k] * 1000.0

        return {
            "window": self.index,
            "elapsed_s": round(elapsed, 2),
            "models_per_s": round(self.popped / elapsed, 4) if elapsed else 0.0,
            "popped": self.popped,
            "pop_ms_p50": round(pct(0.50), 2),
            "pop_ms_p99": round(pct(0.99), 2),
            "ready": stats["ready"],
            "pending": stats["pending"],
            "drained_delta": drained_delta,
            "parent_rss_mb": round(_rss_bytes(os.getpid()) / 1e6, 1),
            "pool_rss_mb": round(pool_rss / 1e6, 1),
            "workers": workers,
        }


def build_probe_feeder(nodes_limit: Optional[int],
                       buffer_size: int,
                       max_workers: Optional[int]) -> RandomIsingFeeder:
    """Construct a RandomIsingFeeder over the production topology.

    Args:
        nodes_limit: Truncate the topology to this many nodes for a fast smoke
            run. None uses the full Advantage2_system1 graph (4577 nodes /
            41515 edges) that the miner actually runs.
        buffer_size: Ready + in-flight target. Defaults to the CUDA path's 16.
        max_workers: Generator processes. None auto-scales to cpu_count-1.

    Returns:
        A started feeder (its constructor already kicked off the first _fill).
    """
    from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_TOPOLOGY as topo

    nodes = list(topo.nodes)
    edges = list(topo.edges)
    if nodes_limit is not None:
        keep = set(nodes[:nodes_limit])
        nodes = [n for n in nodes if n in keep]
        edges = [(u, v) for (u, v) in edges if u in keep and v in keep]

    return RandomIsingFeeder(
        last_proof_block_hash=b"qui867-probe".ljust(32, b"\x00"),
        miner_bytes=b"qui867-miner".ljust(32, b"\x00"),
        nodes=nodes,
        edges=edges,
        buffer_size=buffer_size,
        max_workers=max_workers,
        allowed_h=TERNARY_H,
    )


def run_probe(minutes: float, window_s: float, rate: Optional[float],
              nodes_limit: Optional[int], buffer_size: int,
              max_workers: Optional[int], out_path: Path) -> int:
    """Drive the feeder for *minutes* and emit one JSONL row per window.

    Returns:
        Process exit code (0 on a clean run).
    """
    feeder = build_probe_feeder(nodes_limit, buffer_size, max_workers)
    nodes_desc = "full" if nodes_limit is None else f"first {nodes_limit}"
    pacing = "flat-out" if rate is None else f"paced {rate}/s"
    print(
        f"# QUI-867 feeder decay probe\n"
        f"#   topology : {nodes_desc}\n"
        f"#   buffer   : {buffer_size}   workers: "
        f"{max_workers or _default_max_workers()}   cpus: {os.cpu_count()}\n"
        f"#   pacing   : {pacing}\n"
        f"#   duration : {minutes} min   window: {window_s}s\n"
        f"#   out      : {out_path}\n",
        flush=True,
    )

    deadline = time.monotonic() + minutes * 60.0
    win = Window(index=0, started_at=time.monotonic())
    last_drained = feeder.stats()["drained_count"]
    interval = (1.0 / rate) if rate else 0.0
    next_pop = time.monotonic()
    rows = 0

    hdr = (f"{'win':>4} {'elap':>7} {'models/s':>9} {'pop_p50':>8} "
           f"{'pop_p99':>8} {'ready':>6} {'pend':>5} {'drain':>6} "
           f"{'rss_mb':>7} {'pool_mb':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    try:
        with out_path.open("w", encoding="utf-8") as fh:
            while time.monotonic() < deadline:
                if rate:
                    sleep_for = next_pop - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    next_pop += interval

                t0 = time.monotonic()
                model = feeder.pop_blocking()
                win.pop_waits_s.append(time.monotonic() - t0)
                win.popped += 1
                # Drop the reference immediately: holding models would make the
                # probe itself the leak we are hunting for.
                del model

                now = time.monotonic()
                if now - win.started_at >= window_s:
                    stats = feeder.stats()
                    drained_delta = stats["drained_count"] - last_drained
                    last_drained = stats["drained_count"]
                    row = win.summarize(now, feeder, drained_delta)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        f"{row['window']:>4} {row['elapsed_s']:>7.1f} "
                        f"{row['models_per_s']:>9.3f} {row['pop_ms_p50']:>8.1f} "
                        f"{row['pop_ms_p99']:>8.1f} {row['ready']:>6} "
                        f"{row['pending']:>5} {row['drained_delta']:>6} "
                        f"{row['parent_rss_mb']:>7.1f} {row['pool_rss_mb']:>8.1f}",
                        flush=True,
                    )
                    rows += 1
                    win = Window(index=win.index + 1, started_at=now)
    except KeyboardInterrupt:
        print("\n# interrupted -- shutting down pool", flush=True)
    finally:
        feeder.stop()

    print(f"\n# wrote {rows} windows to {out_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="QUI-867: measure RandomIsingFeeder throughput vs uptime.",
    )
    ap.add_argument("--minutes", type=float, default=240.0,
                    help="run duration (default 240 = 4h, the observed decay window)")
    ap.add_argument("--window", type=float, default=60.0,
                    help="seconds per reported window (default 60)")
    ap.add_argument("--rate", type=float, default=None,
                    help="pace pops to N models/s (default: flat-out)")
    ap.add_argument("--nodes-limit", type=int, default=None,
                    help="truncate topology to N nodes for a fast smoke run")
    ap.add_argument("--buffer-size", type=int, default=GPU_FEEDER_BUFFER_SIZE,
                    help=f"feeder buffer depth (default {GPU_FEEDER_BUFFER_SIZE}, the CUDA path)")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="generator processes (default: cpu_count-1)")
    ap.add_argument("--out", type=Path,
                    default=Path("feeder_decay_probe.jsonl"),
                    help="JSONL output path")
    args = ap.parse_args()

    return run_probe(
        minutes=args.minutes,
        window_s=args.window,
        rate=args.rate,
        nodes_limit=args.nodes_limit,
        buffer_size=args.buffer_size,
        max_workers=args.max_workers,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
