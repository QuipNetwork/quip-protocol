#!/usr/bin/env python3
"""Live-fire probe: confirm per-attempt consumer cost on the real QPU loop.

Runs the *exact* production streaming pipeline
(``DWaveMiner.sample_ising_streaming``, ``queue_depth`` deep, backed by a
``RandomIsingFeeder``) against the live D-Wave QPU, and for every yielded
sampleset measures — separately — the three things that make up a mining
iteration's wall time:

  * ``t_next``  — time blocked in ``next(stream)``: the QPU pipeline /
    cloud round-trip wait. This is what dominates when the pipeline is
    NOT overlapping (effective queue depth ~1).
  * ``t_meta``  — ``compute_solution_meta`` (runs on every attempt, even
    no-hope ones, to populate the attempt log).
  * ``t_eval``  — ``evaluate_sampleset`` in the substrate-ratchet config
    (``strict_energy=False`` + live threshold). Production gates this
    behind a precheck; here we force it on every attempt to measure its
    cost on *real* QPU samplesets (defects + reconstruction included).

The verdict separates the two competing explanations for slow production
throughput: a large ``t_next`` with tiny consumer times means the loop is
QPU-stream-bound (look at queue_depth / region); a small ``t_next`` with a
large consumer time means it is consumer-CPU-bound (the path this probe's
companion optimizations target).

Credentials come from ``.env`` (DWAVE_API_KEY / solver / region), exactly
like the canary. Reads QPU budget: ``--n`` submissions × per-submission
QPU access (~60ms at 112 reads / 80us) — e.g. ``--n 40`` ≈ 2.5s of QPU
access. No chain contact.

Example:
    python tools/qpu_consumer_livefire.py --n 40 --num-reads 112 --anneal 80
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # noqa: BLE001 — .env is optional if env is already set
    pass

import multiprocessing  # noqa: E402 — after sys.path / load_dotenv setup above

from QPU.dwave_miner import DWaveMiner  # noqa: E402
from shared.ising_feeder import RandomIsingFeeder  # noqa: E402
from shared.miner_types import BlockRequirements  # noqa: E402
from shared.quantum_proof_of_work import compute_solution_meta  # noqa: E402


def _summary(name: str, xs: List[float]) -> str:
    """One-line min/median/mean/max summary for a millisecond series."""
    if not xs:
        return f"  {name}: (no samples)"
    return (
        f"  {name:14s}: n={len(xs):3d}  min={min(xs):8.2f}  "
        f"median={statistics.median(xs):8.2f}  "
        f"mean={statistics.fmean(xs):8.2f}  max={max(xs):8.2f}  (ms)"
    )


def run(args: argparse.Namespace) -> int:
    """Drive the production stream and report the per-attempt breakdown."""
    miner = DWaveMiner(miner_id="livefire", queue_depth=args.queue_depth)
    nodes = miner.sampler.nodes
    edges = miner.sampler.edges
    print(
        f"[livefire] sampler ready: {len(nodes)} nodes, {len(edges)} edges, "
        f"queue_depth={args.queue_depth}, reads={args.num_reads}, "
        f"anneal={args.anneal}us, n={args.n}",
        file=sys.stderr,
    )

    feeder = RandomIsingFeeder(
        last_proof_block_hash=os.urandom(32),
        miner_bytes=os.urandom(32),
        nodes=nodes,
        edges=edges,
        buffer_size=args.feeder_buffer_size,
    )
    miner._stop_event = multiprocessing.Event()

    requirements = BlockRequirements(
        difficulty_energy=args.energy_threshold,
        min_diversity=args.min_diversity,
        min_solutions=args.min_solutions,
        timeout_to_difficulty_adjustment_decay=2**31 - 1,
    )

    t_next: List[float] = []
    t_meta: List[float] = []
    t_eval: List[float] = []
    qpu_ms: List[float] = []
    errors = 0

    overall_start = time.perf_counter()
    try:
        stream = miner.sample_ising_streaming(
            feeder,
            num_reads=args.num_reads,
            annealing_time=args.anneal,
            queue_depth=args.queue_depth,
            energy_threshold=args.energy_threshold,
        )
        it = iter(stream)
        for i in range(args.n):
            t0 = time.perf_counter()
            try:
                model, sampleset = next(it)
            except StopIteration:
                break
            t_next.append((time.perf_counter() - t0) * 1000.0)

            try:
                t1 = time.perf_counter()
                compute_solution_meta(sampleset, args.energy_threshold)
                t_meta.append((time.perf_counter() - t1) * 1000.0)

                # evaluate_sampleset needs a full-topology sampleset (its
                # energy recompute maps spins onto all `nodes`). Production
                # only ever calls it on promising candidates, which the
                # stream reconstructs to full width; non-promising attempts
                # are yielded raw (active qubits only) and the precheck
                # skips evaluate. Mirror that: only time t_eval when the
                # sampleset is full-width, else it stays a no-hope skip.
                if sampleset.record.sample.shape[1] == len(nodes):
                    t2 = time.perf_counter()
                    miner.evaluate_sampleset(
                        sampleset, requirements, nodes, edges,
                        model.nonce, model.salt, prev_timestamp=0,
                        start_time=overall_start, strict_energy=False,
                        live_threshold_energy=args.live_threshold,
                    )
                    t_eval.append((time.perf_counter() - t2) * 1000.0)

                timing = sampleset.info.get("timing", {}) if sampleset.info else {}
                qpu = timing.get("qpu_programming_time", 0) + timing.get(
                    "qpu_sampling_time", 0,
                )
                qpu_ms.append(qpu / 1000.0)
            except Exception as exc:  # noqa: BLE001 — log + continue the run
                errors += 1
                print(f"[livefire] attempt {i} error: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

            if (i + 1) % args.log_every == 0:
                print(f"[livefire] {i + 1}/{args.n} attempts...", file=sys.stderr)
    finally:
        miner._stop_event.set()
        feeder.stop()

    wall = time.perf_counter() - overall_start
    n_done = len(t_next)
    print("\n=== live-fire consumer breakdown (real QPU) ===")
    print(_summary("t_next", t_next))
    print(_summary("t_meta", t_meta))
    print(_summary("t_eval", t_eval))
    print(_summary("qpu_access", qpu_ms))
    consumer = [
        (t_meta[i] if i < len(t_meta) else 0) + (t_eval[i] if i < len(t_eval) else 0)
        for i in range(n_done)
    ]
    print(_summary("consumer(m+e)", consumer))
    if n_done:
        thru = n_done / wall
        med_next = statistics.median(t_next)
        med_consumer = statistics.median(consumer) if consumer else 0.0
        print(f"\n  throughput   : {thru:.2f} submissions/sec over {wall:.1f}s")
        print(f"  errors       : {errors}")
        bound = "QPU-stream-bound" if med_next > med_consumer else "consumer-bound"
        print(f"  VERDICT      : {bound} "
              f"(median t_next={med_next:.0f}ms vs consumer={med_consumer:.0f}ms)")
        print(f"  50ms target  : consumer median {med_consumer:.1f}ms "
              f"{'PASS' if med_consumer <= 50 else 'FAIL'}")
    return 0


def main() -> int:
    """Parse CLI args and run the probe."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n", type=int, default=40,
                   help="submissions to measure (default 40)")
    p.add_argument("--num-reads", type=int, default=112,
                   help="QPU reads per submission (default 112, production)")
    p.add_argument("--anneal", type=float, default=80.0,
                   help="annealing time us (default 80, production)")
    p.add_argument("--queue-depth", type=int, default=30,
                   help="in-flight submissions (default 30, production)")
    p.add_argument("--feeder-buffer-size", type=int, default=60)
    p.add_argument("--energy-threshold", type=float, default=-14850.0,
                   help="difficulty_energy / reconstruction gate")
    p.add_argument("--live-threshold", type=float, default=-14900.0,
                   help="live decayed threshold for the ratchet eval")
    p.add_argument("--min-solutions", type=int, default=5)
    p.add_argument("--min-diversity", type=float, default=0.2)
    p.add_argument("--log-every", type=int, default=10)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
