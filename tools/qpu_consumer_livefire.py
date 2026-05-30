#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Live-fire / production smoke check for the QPU mining loop.

Two modes (``--mode``):

``driver`` (default) — **production smoke check.** Drives the *real*
production path the live miner uses: ``BaseMiner._start_result_pump`` spawns
the ``QPU.stream_driver`` subprocess running ``build_production_stream``
against the live QPU; the consumer reads samplesets through the
``SharedSampleRing`` via ``_acquire_result``; ``_teardown_dispatch`` reaps it.
This exercises the parts the in-process probe cannot:

  * **Spawn** — the driver starts NON-DAEMON and builds the feeder's
    ``ProcessPoolExecutor`` with no ``daemonic processes are not allowed to
    have children`` AssertionError (the bug a daemon driver would hit).
  * **Liveness** — (``--fault-injection``) a driver is hard-killed mid-stream
    and the consumer must end the dispatch instead of hanging on an empty
    queue (the silent "miner looks stuck" failure mode).
  * **Teardown** — no ``BufferError`` on ring close, no orphaned child
    processes, and ``/dev/shm`` segments returned to baseline (no leak).

``inprocess`` — the consumer-cost probe: drives
``DWaveMiner.sample_ising_streaming`` in-process and measures, per attempt,
``t_next`` (QPU pipeline wait), ``t_meta`` (``compute_solution_meta``), and
``t_eval`` (``evaluate_sampleset``). Use this to confirm the per-attempt
consumer work stays under the 50ms target; it does NOT spawn the driver
subprocess.

Credentials come from ``.env`` (DWAVE_API_KEY / solver / region). QPU budget:
``--n`` submissions × per-submission QPU access (~60ms at 112 reads / 80us);
the driver mode's optional liveness phase opens one extra short connection. No
chain contact.

Examples:
    python tools/qpu_consumer_livefire.py --n 40 --num-reads 112 --anneal 80
    python tools/qpu_consumer_livefire.py --mode inprocess --n 40
"""
from __future__ import annotations

import argparse
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from dwave_topologies import DEFAULT_TOPOLOGY  # noqa: E402
from shared.base_miner import (  # noqa: E402
    _ACQUIRE_CONTINUE,
    _ACQUIRE_DONE,
    _ACQUIRE_OK,
    _ACQUIRE_STOP,
)
from shared.ising_feeder import RandomIsingFeeder  # noqa: E402
from shared.miner_types import BlockRequirements  # noqa: E402
from shared.proc_util import terminate_join  # noqa: E402
from shared.quantum_proof_of_work import compute_solution_meta  # noqa: E402

# A hard-killed driver must let the consumer end the dispatch within this many
# seconds (it should be near-instant: one queue.Empty tick + is_alive check).
_LIVENESS_BOUND_S = 10.0
_CONSUMER_TARGET_MS = 50.0


def _summary(name: str, xs: List[float]) -> str:
    """One-line min/median/mean/max summary for a millisecond series."""
    if not xs:
        return f"  {name:14s}: (no samples)"
    return (
        f"  {name:14s}: n={len(xs):3d}  min={min(xs):8.2f}  "
        f"median={statistics.median(xs):8.2f}  "
        f"mean={statistics.fmean(xs):8.2f}  max={max(xs):8.2f}  (ms)"
    )


def _count_shm() -> Optional[int]:
    """POSIX shared-memory segment count (Linux ``/dev/shm``), else None.

    Used to detect leaked ``SharedSampleRing`` segments after teardown.
    macOS has no ``/dev/shm`` so the check is skipped there.
    """
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir):
        try:
            return len(os.listdir(shm_dir))
        except OSError:
            return None
    return None


class _BudgetSpy:
    """Stand-in for ``QPUTimeManager``: counts qpu_us fed via the driver path.

    Confirms the daily-budget accounting still works end to end — the driver
    process can't reach the worker's time_manager, so the per-attempt QPU
    access time rides the descriptor and is recorded by ``_acquire_result``.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.total_us = 0

    def record_block_time(self, us: int) -> None:
        self.calls += 1
        self.total_us += int(us)


def _make_requirements(args: argparse.Namespace) -> BlockRequirements:
    """Build the substrate-ratchet requirements used by ``evaluate_sampleset``."""
    return BlockRequirements(
        difficulty_energy=args.energy_threshold,
        min_diversity=args.min_diversity,
        min_solutions=args.min_solutions,
        timeout_to_difficulty_adjustment_decay=2**31 - 1,
    )


def _build_consumer(args: argparse.Namespace) -> DWaveMiner:
    """Connection-less worker miner (the one QPU connection lives in the driver).

    ``connect=False`` mirrors the production worker: no sampler here; the
    stream-driver process builds its own connected ``DWaveMiner``. solver /
    region / token flow through env (DWAVE_API_KEY etc.), exactly as the
    in-process probe relied on them.
    """
    consumer = DWaveMiner(
        miner_id="smoke",
        queue_depth=args.queue_depth,
        solver_name=os.environ.get("DWAVE_SOLVER") or None,
        region=os.environ.get("DWAVE_REGION") or None,
        connect=False,
    )
    consumer.time_manager = _BudgetSpy()
    return consumer


def _make_sample_ctx(nodes, edges, args: argparse.Namespace) -> Dict[str, Any]:
    """The per-dispatch context ``_start_result_pump`` forwards to the driver."""
    return {
        "miner_id": "smoke",
        "num_reads": args.num_reads,
        "num_sweeps": 1,
        "nodes": nodes,
        "edges": edges,
        "annealing_time": args.anneal,
        "energy_threshold": args.energy_threshold,
        "last_proof_block_hash": os.urandom(32),
        "miner_bytes": os.urandom(32),
        "feeder_buffer_size": args.feeder_buffer_size,
        "extra": {},
    }


def _release(consumer: DWaveMiner, acquired) -> None:
    """Return the ring slot and drop the zero-copy view before teardown."""
    if acquired.ring_slot is not None and consumer._ring is not None:
        consumer._ring.release(acquired.ring_slot)
    acquired.sampleset = None


def _child_pids(pid: int) -> List[int]:
    """Direct child PIDs of ``pid`` via pgrep (best-effort, POSIX only)."""
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                             text=True, timeout=5)
        return [int(x) for x in out.stdout.split()]
    except Exception:  # noqa: BLE001 — pgrep absent → orphans simply leak
        return []


def _hard_kill(pids: List[int]) -> None:
    """SIGKILL each pid, ignoring those already gone."""
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def _consume(consumer, desc_q, driver_proc, sample_ctx, nodes, edges,
             requirements, args) -> Dict[str, Any]:
    """Read ``args.n`` results through the real ``_acquire_result`` path."""
    stop = multiprocessing.Event()
    t_acq: List[float] = []
    t_meta: List[float] = []
    t_eval: List[float] = []
    qpu_ms: List[float] = []
    errors = 0
    ended_early = False
    start = time.perf_counter()
    for i in range(args.n):
        t0 = time.perf_counter()
        acquired = consumer._acquire_result(
            stop, desc_q, t0, sample_ctx=sample_ctx, driver_proc=driver_proc)
        if acquired.action == _ACQUIRE_DONE:
            ended_early = True
            print("[smoke] stream ended early (driver DONE)", file=sys.stderr)
            break
        if acquired.action == _ACQUIRE_STOP:
            break
        if acquired.action == _ACQUIRE_CONTINUE:
            errors += 1
            continue
        t_acq.append((time.perf_counter() - t0) * 1000.0)
        ss = acquired.sampleset
        try:
            t1 = time.perf_counter()
            compute_solution_meta(ss, args.energy_threshold)
            t_meta.append((time.perf_counter() - t1) * 1000.0)
            # evaluate_sampleset needs a full-topology sampleset; production
            # only reconstructs promising candidates to full width, so mirror
            # that and only time eval when the matrix spans all nodes.
            if ss.record.sample.shape[1] == len(nodes):
                t2 = time.perf_counter()
                consumer.evaluate_sampleset(
                    ss, requirements, nodes, edges, acquired.nonce,
                    acquired.salt, prev_timestamp=0, start_time=start,
                    strict_energy=False, live_threshold_energy=args.live_threshold)
                t_eval.append((time.perf_counter() - t2) * 1000.0)
            if acquired.qpu_access_time_us:
                qpu_ms.append(acquired.qpu_access_time_us / 1000.0)
        except Exception as exc:  # noqa: BLE001 — count + continue the run
            errors += 1
            print(f"[smoke] attempt {i} error: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        finally:
            _release(consumer, acquired)
        if (i + 1) % args.log_every == 0:
            print(f"[smoke] {i + 1}/{args.n} consumed...", file=sys.stderr)
    return {
        "t_acq": t_acq, "t_meta": t_meta, "t_eval": t_eval, "qpu_ms": qpu_ms,
        "errors": errors, "ended_early": ended_early,
        "wall": time.perf_counter() - start,
    }


def _liveness_check(consumer, nodes, edges, args) -> Dict[str, Any]:
    """Hard-kill a fresh driver mid-stream; the consumer must end, not hang.

    Validates the ``_acquire_result`` liveness path: a driver that dies
    without enqueuing the ``None`` sentinel (SIGKILL here, OOM/segfault in the
    wild) must be detected via ``driver_proc.is_alive()`` so the consumer
    returns DONE rather than draining an empty queue forever.
    """
    sample_ctx = _make_sample_ctx(nodes, edges, args)
    ring, desc_q, proc, dstop = consumer._start_result_pump(sample_ctx)
    stop = multiprocessing.Event()
    try:
        # Confirm the stream is actually flowing before we kill it.
        first = consumer._acquire_result(
            stop, desc_q, time.perf_counter(), sample_ctx=sample_ctx,
            driver_proc=proc)
        flowing = first.action == _ACQUIRE_OK
        _release(consumer, first)
        # Capture the driver's feeder ProcessPoolExecutor workers before the
        # kill: SIGKILL skips the driver's cleanup, orphaning them (reparented
        # to init). Reap them explicitly so the smoke check leaves nothing
        # behind — a real hard crash would leak these until the OS reaps them.
        workers = _child_pids(proc.pid)
        # Simulate a hard crash: kill without setting the stop event so no
        # sentinel is ever sent.
        proc.kill()
        proc.join(timeout=5.0)
        _hard_kill(workers)
        # The consumer must reach DONE within the bound. It may first drain a
        # few buffered descriptors (release them); a hang would blow the bound.
        t0 = time.perf_counter()
        reached_done = False
        while time.perf_counter() - t0 < _LIVENESS_BOUND_S:
            acq = consumer._acquire_result(
                stop, desc_q, time.perf_counter(), sample_ctx=sample_ctx,
                driver_proc=proc)
            if acq.action == _ACQUIRE_DONE:
                reached_done = True
                break
            if acq.action == _ACQUIRE_STOP:
                break
            _release(consumer, acq)
        elapsed = time.perf_counter() - t0
        return {"flowing": flowing, "reached_done": reached_done,
                "elapsed": elapsed}
    finally:
        consumer._teardown_dispatch(dstop, proc)


def _report_consume(res: Dict[str, Any]) -> bool:
    """Print the consume-phase breakdown; return True if the 50ms target holds."""
    print("\n=== consume breakdown (real driver subprocess) ===")
    print(_summary("t_acquire", res["t_acq"]))
    print(_summary("t_meta", res["t_meta"]))
    print(_summary("t_eval", res["t_eval"]))
    print(_summary("qpu_access", res["qpu_ms"]))
    consumer_ms = [
        (res["t_meta"][i] if i < len(res["t_meta"]) else 0)
        + (res["t_eval"][i] if i < len(res["t_eval"]) else 0)
        for i in range(len(res["t_acq"]))
    ]
    print(_summary("consumer(m+e)", consumer_ms))
    med = statistics.median(consumer_ms) if consumer_ms else 0.0
    n = len(res["t_acq"])
    if n and res["wall"] > 0:
        print(f"\n  throughput   : {n / res['wall']:.2f} submissions/sec "
              f"over {res['wall']:.1f}s")
    print(f"  errors       : {res['errors']}")
    ok = med <= _CONSUMER_TARGET_MS
    print(f"  50ms target  : consumer median {med:.1f}ms "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_driver(args: argparse.Namespace) -> int:
    """Production smoke check: spawn the real driver, consume, fault-test, reap."""
    nodes = list(DEFAULT_TOPOLOGY.nodes)
    edges = list(DEFAULT_TOPOLOGY.edges)
    print(f"[smoke] topology: {len(nodes)} nodes, {len(edges)} edges; "
          f"queue_depth={args.queue_depth}, reads={args.num_reads}, "
          f"anneal={args.anneal}us, n={args.n}", file=sys.stderr)

    consumer = _build_consumer(args)
    requirements = _make_requirements(args)
    sample_ctx = _make_sample_ctx(nodes, edges, args)

    base_children = {c.pid for c in multiprocessing.active_children()}
    base_shm = _count_shm()

    # --- Spawn: this is build_production_stream in a real subprocess ---
    ring, desc_q, driver_proc, driver_stop = consumer._start_result_pump(
        sample_ctx)
    spawn_ok = driver_proc is not None and driver_proc.daemon is False
    print(f"[smoke] driver spawned: pid={getattr(driver_proc, 'pid', None)} "
          f"daemon={getattr(driver_proc, 'daemon', None)} "
          f"(non-daemon required) -> {'OK' if spawn_ok else 'FAIL'}",
          file=sys.stderr)

    # --- Consume through the real _acquire_result path ---
    res = _consume(consumer, desc_q, driver_proc, sample_ctx, nodes, edges,
                   requirements, args)

    # --- Clean teardown (stop event -> driver finally -> close_unlink) ---
    buffererror = False
    try:
        consumer._teardown_dispatch(driver_stop, driver_proc)
    except BufferError:
        buffererror = True

    time.sleep(0.5)  # let reaped children settle before the orphan scan
    leftover = [c for c in multiprocessing.active_children()
                if c.pid not in base_children]
    orphans_ok = not leftover
    shm_after = _count_shm()
    shm_ok = base_shm is None or shm_after is None or shm_after <= base_shm

    consume_ok = _report_consume(res)

    # --- Optional liveness fault-injection (fresh short-lived driver) ---
    live = None
    if args.fault_injection:
        print("\n[smoke] fault-injection: hard-killing a driver mid-stream...",
              file=sys.stderr)
        live = _liveness_check(consumer, nodes, edges, args)

    # Best-effort reap of anything we still own so the CLI exits cleanly.
    for child in multiprocessing.active_children():
        terminate_join(child, 2.0)

    return _emit_verdict(spawn_ok, consume_ok, res, buffererror, orphans_ok,
                         leftover, shm_ok, base_shm, shm_after, live)


def _emit_verdict(spawn_ok, consume_ok, res, buffererror, orphans_ok,
                  leftover, shm_ok, base_shm, shm_after, live) -> int:
    """Print the smoke-check checklist and return 0 (PASS) / 1 (FAIL)."""
    print("\n=== production smoke check ===")
    checks = [
        ("driver spawned non-daemon", spawn_ok),
        ("errors == 0", res["errors"] == 0),
        (f"consumer <= {_CONSUMER_TARGET_MS:.0f}ms", consume_ok),
        ("no BufferError on teardown", not buffererror),
        ("no orphaned child processes", orphans_ok),
        ("no /dev/shm leak", shm_ok),
    ]
    if live is not None:
        checks.append(("driver-death -> DONE (no hang)",
                       bool(live["reached_done"])))
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not orphans_ok:
        print(f"      leftover children: {[c.name for c in leftover]}")
    if base_shm is not None and shm_after is not None:
        print(f"      /dev/shm segments: {base_shm} -> {shm_after}")
    if live is not None:
        print(f"      liveness: flowing={live['flowing']} "
              f"reached_done={live['reached_done']} "
              f"in {live['elapsed']:.2f}s (bound {_LIVENESS_BOUND_S:.0f}s)")
    all_ok = all(ok for _, ok in checks)
    print(f"\n  VERDICT      : {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def run_inprocess(args: argparse.Namespace) -> int:
    """Consumer-cost probe: drive the stream in-process (no driver subprocess)."""
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
    requirements = _make_requirements(args)

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
              f"{'PASS' if med_consumer <= _CONSUMER_TARGET_MS else 'FAIL'}")
    return 0


def main() -> int:
    """Parse CLI args and run the selected mode."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("driver", "inprocess"), default="driver",
                   help="driver = production smoke check (default); "
                        "inprocess = consumer-cost probe only")
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
    p.add_argument("--no-fault-injection", dest="fault_injection",
                   action="store_false",
                   help="skip the hard-kill driver-liveness phase (driver mode)")
    p.set_defaults(fault_injection=True)
    args = p.parse_args()
    return run_driver(args) if args.mode == "driver" else run_inprocess(args)


if __name__ == "__main__":
    raise SystemExit(main())
