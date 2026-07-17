#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-stack soak: real Metal miner through the real driver/ring/worker path.

QUI-867 discriminator #1, run locally. The standalone GPU probes exonerated the
producer in a single continuous round, but they bypass everything the reporter's
miner actually runs: the stream-driver process, the SampleView ring, desc_q, the
worker's mine_work_item loop, attempt logging, and round switching. This probe
runs ALL of that — a real ``MetalMiner`` dispatched round after round exactly as
``miner_worker`` would — with no chain attached, and reports per window:

* att/s counted where production counts it (the worker's acquire loop)
* ring_drops (the shared counter from c16b514) — the drop-path discriminator
* worker + driver RSS, attempt-log bytes on disk

Reading the result:
* att/s decays here (native macOS/APFS, no WSL2) -> the decay is in the
  codebase's host plumbing; the drop/worker path is the accumulator.
* flat -> the codebase is clean end-to-end on clean hardware, pointing the
  remaining suspicion at the reporter's WSL2/Docker environment.

Round lengths are sampled from the measured chain distribution (median ~2 min,
p90 ~1.5-2.3 h) via the wins JSONL when present, so the switch path and the
per-solution attempt files are exercised at production cadence.

Usage:
    python tools/stack_soak_probe.py --minutes 240 --util 40
    python tools/stack_soak_probe.py --minutes 6 --round-cap 120  # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# reads pins the RING PAYLOAD to the reporter's production shape (the
# 200 x 4577 int8 sample matrix crossing the SampleView ring per attempt) —
# that is the suspect surface. sweeps is pure GPU compute, which every prior
# measurement has exonerated, so it is slashed to keep attempts flowing fast
# enough for per-window statistics: at the reporter's 1000 sweeps a single
# Metal attempt under the fair-share cap takes minutes on an in-use Mac.
PIN_NUM_READS = 200
PIN_NUM_SWEEPS = 64


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip() or 0) * 1024
    except Exception:
        return 0


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        pass
    return total


def _round_lengths_s(seed: int) -> "random.Random":
    """RNG over empirical round lengths (seconds) from the chain wins data.

    Falls back to a synthetic short/long mix matching the measured shape
    (median ~2 min, heavy tail to ~3 h) when the wins file is absent.
    """
    gaps_blocks: list[int] = []
    for name in ("quip_wins.wins.jsonl", "quip_recent.wins.jsonl"):
        p = Path(name)
        if p.exists():
            blocks = sorted(
                json.loads(line)["block_number"] for line in p.open()
            )
            gaps_blocks = [b2 - b1 for b1, b2 in zip(blocks, blocks[1:]) if b2 > b1]
            break
    rng = random.Random(seed)
    if gaps_blocks:
        def draw() -> float:
            return rng.choice(gaps_blocks) * 6.0  # 6s blocktime
    else:
        def draw() -> float:
            return rng.uniform(60, 240) if rng.random() < 0.7 else rng.uniform(1200, 9000)
    rng.draw_round_s = draw  # type: ignore[attr-defined]
    return rng


def build_probe_miner(util: int):
    """Construct a real MetalMiner with the production topology.

    The workload is pinned (PIN_NUM_READS / PIN_NUM_SWEEPS) by overriding
    _adapt_mining_params on the instance's class.
    """
    from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_TOPOLOGY as topo
    from GPU.metal_miner import MetalMiner

    class _SoakMetalMiner(MetalMiner):
        def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
            params = super()._adapt_mining_params(requirements, nodes, edges)
            params["num_reads"] = PIN_NUM_READS
            params["num_sweeps"] = PIN_NUM_SWEEPS
            return params

    # yielding=False is production-faithful for QUI-867: the reporter's
    # 3.17 att/s exceeds yielding mode's 2/s ceiling, so their rig runs
    # yielding off. It also stops the sensor governor from pausing the
    # producer (budget=0 sleep loop, metal_sa.py:205) whenever this Mac is
    # in active use, which otherwise gates att/s on user activity instead
    # of the plumbing under test.
    return _SoakMetalMiner(
        "Metal-soak", topology=topo, utilization=util, yielding=False,
    ), topo


def make_context(topo, round_index: int):
    """Synthetic SubstrateMiningContext for one round. No chain involved.

    The threshold is unreachably strict so no attempt ever qualifies: the
    round only ends when the probe preempts it, exactly like a production
    round the miner never wins.
    """
    from shared.allowed_value_spec import AllowedValueSet
    from substrate.types import SubstrateDifficulty, SubstrateMiningContext

    seed = hashlib.blake2b(
        round_index.to_bytes(8, "big"), digest_size=32,
    ).digest()
    return SubstrateMiningContext(
        last_proof_block_hash=seed,
        topology_hash=b"\xcd" * 32,
        nodes=list(topo.nodes),
        edges=list(topo.edges),
        difficulty=SubstrateDifficulty(
            min_solutions=70,
            max_energy_milli=-(10 ** 12),  # unreachable: never ends a round
            min_diversity_milli=200,
        ),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=AllowedValueSet((-1000, 0, 1000)),
        allowed_j_values=AllowedValueSet((-1000, 1000)),
        allowed_spin_values=AllowedValueSet((-1000, 1000)),
        block_hash=seed[::-1],
        block_number=round_index + 1,
    )


def monitor(miner, out_path: Path, log_path: Path, runtime_dir: Path,
            window_s: float, stop: threading.Event, state: dict) -> None:
    """Window reporter thread: samples worker-side counters every window_s."""
    t0 = time.monotonic()
    last_attempts = 0
    win = 0
    with out_path.open("w", encoding="utf-8") as fh:
        while not stop.wait(window_s):
            attempts = miner.timing_stats.get("blocks_attempted", 0)
            drops = miner.ring_drops
            driver = getattr(miner, "_driver_proc", None)
            row = {
                "window": win,
                "uptime_min": round((time.monotonic() - t0) / 60.0, 2),
                "att_per_s": round((attempts - last_attempts) / window_s, 4),
                "attempts_total": attempts,
                "ring_drops": drops,
                "round_index": state.get("round_index"),
                "rounds_started": state.get("rounds_started"),
                "worker_rss_mb": round(_rss_bytes(os.getpid()) / 1e6, 1),
                "driver_rss_mb": round(
                    _rss_bytes(driver.pid) / 1e6, 1,
                ) if driver is not None and driver.is_alive() else None,
                "attempt_log_mb": round(_dir_bytes(runtime_dir) / 1e6, 2),
                "probe_log_mb": round(
                    log_path.stat().st_size / 1e6, 2,
                ) if log_path.exists() else 0.0,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"win={row['window']:>4} up={row['uptime_min']:>7.1f}m "
                f"att/s={row['att_per_s']:>7.3f} drops={row['ring_drops']} "
                f"round={row['round_index']} wRSS={row['worker_rss_mb']}MB "
                f"dRSS={row['driver_rss_mb']}MB "
                f"alog={row['attempt_log_mb']}MB",
                flush=True,
            )
            last_attempts = attempts
            win += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="QUI-867 full-stack Metal soak")
    ap.add_argument("--minutes", type=float, default=240.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--util", type=int, default=40,
                    help="Metal utilization pct (default 40: polite overnight)")
    ap.add_argument("--round-cap", type=float, default=None,
                    help="cap round length in seconds (smoke runs)")
    ap.add_argument("--seed", type=int, default=867)
    ap.add_argument("--out-dir", type=Path, default=Path("stack_soak_out"),
                    help="directory for the JSONL, log, and attempt-log tree")
    args = ap.parse_args()

    out_dir = args.out_dir
    runtime_dir = out_dir / "soak_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QUIP_RUNTIME_DIR"] = str(runtime_dir)

    # Unbounded plain-file log sink on purpose: mirrors the shipped Docker
    # json-file capture (no rotation) so log-growth cost is part of the soak.
    import logging
    log_path = out_dir / "stack_soak.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path)],
    )

    miner, topo = build_probe_miner(args.util)

    # Drain the stream-driver's logs the way production does (miner_core owns
    # log_queue + log_writer_main). Without this the driver process has no
    # handlers and its diagnostics vanish.
    import logging.handlers
    log_q = mp.get_context("spawn").Queue()
    listener = logging.handlers.QueueListener(
        log_q, *logging.getLogger().handlers, respect_handler_level=False,
    )
    listener.start()
    miner._log_queue = log_q

    out_path = out_dir / "stack_soak.jsonl"
    print(
        f"# QUI-867 full-stack Metal soak\n"
        f"#   topology  : {len(list(topo.nodes))} nodes\n"
        f"#   workload  : reads={PIN_NUM_READS} sweeps={PIN_NUM_SWEEPS} "
        f"util={args.util}%\n"
        f"#   duration  : {args.minutes} min, window {args.window}s\n"
        f"#   runtime   : {runtime_dir}\n#   out       : {out_path}\n",
        flush=True,
    )

    rng = _round_lengths_s(args.seed)
    deadline = time.monotonic() + args.minutes * 60.0
    mon_stop = threading.Event()
    state = {"round_index": 0, "rounds_started": 0}
    mon = threading.Thread(
        target=monitor,
        args=(miner, out_path, log_path, runtime_dir, args.window,
              mon_stop, state),
        daemon=True,
    )
    mon.start()

    ctx_mp = mp.get_context("spawn")
    round_index = 0
    try:
        while time.monotonic() < deadline:
            # Floor at 30s: the empirical distribution has 6-30s gaps, but a
            # round shorter than Metal's first-completion latency produces
            # zero attempts and only adds switch noise.
            round_s = max(30.0, rng.draw_round_s())  # type: ignore[attr-defined]
            if args.round_cap:
                round_s = min(round_s, args.round_cap)
            round_s = min(round_s, max(5.0, deadline - time.monotonic()))
            state["round_index"] = round_index
            state["rounds_started"] = round_index + 1

            # Replicate miner_worker's pre-dispatch state.
            miner._current_dispatch_id = round_index
            miner._current_solution_number = round_index + 1

            stop_event = ctx_mp.Event()
            timer = threading.Timer(round_s, stop_event.set)
            timer.daemon = True
            timer.start()
            result = miner.mine_work_item(
                make_context(topo, round_index), stop_event,
            )
            timer.cancel()
            if result is not None:
                # Impossible threshold: a result here is itself a finding.
                logging.getLogger("soak").warning(
                    "round %d returned a result despite impossible "
                    "threshold: %r", round_index, result,
                )
            round_index += 1
    except KeyboardInterrupt:
        print("# interrupted", flush=True)
    finally:
        mon_stop.set()
        mon.join(timeout=5)
        try:
            miner._close_driver()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass
    print(f"# done: {round_index} rounds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
