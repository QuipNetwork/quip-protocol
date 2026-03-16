#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Pipeline integration benchmark for SA and Gibbs multi-nonce mining.

Tests:
  1. SA at 25/50/75/100% GPU utilization
  2. Gibbs at 25/50/75/100% utilization
  3. SA + Gibbs co-existence with yielding=True

Energy target: -14600 (easy, fast turnaround).
Timeout: 120s per single trial, 180s for co-existence.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_pipeline.py
"""

import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

sys.path.append(str(Path(__file__).parent.parent))

from shared.block import BlockRequirements, create_genesis_block


@dataclass
class NodeInfo:
    """Minimal node info for mining."""
    miner_id: str


ENERGY_TARGET = -14750.0
REQUIREMENTS = BlockRequirements(
    difficulty_energy=ENERGY_TARGET,
    min_diversity=0.2,
    min_solutions=5,
    timeout_to_difficulty_adjustment_decay=0,
)

# Fast kernel launches: ~200 betas × 64 reads = seconds/batch
OVERRIDE_PARAMS = {
    'num_sweeps': 200,
    'num_reads': 64,
    'num_sweeps_per_beta': 1,
}

SINGLE_TIMEOUT = 120  # seconds per single-miner trial
COEXIST_TIMEOUT = 180  # seconds for co-existence test


def _get_nonces_per_attempt(miner):
    """Read how many nonces the scheduler dispatches per attempt."""
    scheduler = getattr(miner, '_scheduler', None)
    if scheduler is None:
        return 1
    return scheduler.get_sm_budget()


def run_trial(label, miner_cls, device, utilization, yielding):
    """Run a single mining trial. Returns trial metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  {label}  (util={utilization}%, yielding={yielding})")
    print(f"{'=' * 60}")

    miner = miner_cls(
        miner_id=f"bench-{label}",
        device=device,
        yielding=yielding,
        gpu_utilization=utilization,
    )

    # Override adaptive params for fast kernel launches
    miner._adapt_mining_params = lambda *a, **kw: dict(OVERRIDE_PARAMS)

    nonces_per = _get_nonces_per_attempt(miner)

    prev_block = create_genesis_block()
    prev_block.next_block_requirements = REQUIREMENTS
    node_info = NodeInfo(miner_id=miner.miner_id)
    stop_event = multiprocessing.Event()

    start = time.time()
    result = miner.mine_block(
        prev_block=prev_block,
        node_info=node_info,
        requirements=REQUIREMENTS,
        prev_timestamp=prev_block.header.timestamp,
        stop_event=stop_event,
    )
    elapsed = time.time() - start

    attempts = miner.timing_stats['blocks_attempted']
    total_nonces = attempts * nonces_per
    nonces_s = total_nonces / elapsed if elapsed > 0 else 0.0

    if result:
        print(
            f"  MINED in {elapsed:.1f}s | "
            f"energy={result.energy:.1f} | "
            f"attempts={attempts} | "
            f"nonces/attempt={nonces_per} | "
            f"nonces/s={nonces_s:.1f}"
        )
    else:
        print(f"  FAILED after {elapsed:.1f}s")

    # Free GPU memory between trials
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    return {
        'label': label,
        'elapsed': elapsed,
        'result': result,
        'blocks_attempted': attempts,
        'nonces_per_attempt': nonces_per,
        'nonces_s': nonces_s,
        'success': result is not None,
    }


def _coexist_worker(miner_cls_name, label, device, result_file):
    """Subprocess target for co-existence test.

    Writes results to a JSON file instead of using Queue/pickle.
    """
    import json

    # Import miner class by name to avoid pickling issues
    from GPU.cuda_miner import CudaMiner

    if miner_cls_name == "CudaGibbsMiner":
        def cls(*a, **kw):
            return CudaMiner(*a, update_mode="gibbs", **kw)
    else:
        cls = CudaMiner

    try:
        trial = run_trial(
            label, cls, device,
            utilization=50, yielding=True,
        )
        data = {
            'label': trial['label'],
            'elapsed': trial['elapsed'],
            'blocks_attempted': trial['blocks_attempted'],
            'nonces_per_attempt': trial['nonces_per_attempt'],
            'nonces_s': float(trial['nonces_s']),
            'success': trial['success'],
            'energy': (
                float(trial['result'].energy)
                if trial['result'] else None
            ),
        }
    except Exception as exc:
        data = {'label': label, 'error': str(exc)}

    Path(result_file).write_text(json.dumps(data))


def run_coexistence_test(device):
    """Run SA + Gibbs concurrently on the same GPU.

    Uses 'spawn' context so each child gets a fresh CUDA runtime
    instead of inheriting the parent's (already-initialized) context.
    """
    import json
    import tempfile

    print(f"\n{'#' * 60}")
    print("  CO-EXISTENCE: SA + Gibbs @ 50% yielding=True")
    print(f"{'#' * 60}")

    ctx = multiprocessing.get_context('spawn')

    tmpdir = Path(tempfile.mkdtemp())
    sa_file = str(tmpdir / "sa_result.json")
    gibbs_file = str(tmpdir / "gibbs_result.json")

    procs = [
        ctx.Process(
            target=_coexist_worker,
            args=("CudaMiner", "coexist-SA", device, sa_file),
        ),
        ctx.Process(
            target=_coexist_worker,
            args=(
                "CudaGibbsMiner", "coexist-Gibbs",
                device, gibbs_file,
            ),
        ),
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join(timeout=COEXIST_TIMEOUT)
        if p.is_alive():
            print(f"  TIMEOUT: {p.name} exceeded {COEXIST_TIMEOUT}s")
            p.terminate()
            p.join(timeout=5)

    results = []
    for fpath in [sa_file, gibbs_file]:
        path = Path(fpath)
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            label = "SA" if "sa_" in fpath else "Gibbs"
            results.append({'label': f"coexist-{label}", 'error': "no output"})

    # Cleanup temp files
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()

    return results


def main():
    device = "0"

    from functools import partial
    from GPU.cuda_miner import CudaMiner

    def GibbsMiner(*a, **kw):
        return CudaMiner(*a, update_mode="gibbs", **kw)

    all_results = []

    # --- Single-miner trials ---
    for miner_cls, name in [
        (CudaMiner, "SA"),
        (GibbsMiner, "Gibbs"),
    ]:
        for util in [25, 50, 75, 100]:
            label = f"{name}-{util}pct"
            trial = run_trial(
                label, miner_cls, device,
                utilization=util, yielding=False,
            )
            if trial['elapsed'] > SINGLE_TIMEOUT:
                print(f"  WARNING: {label} exceeded {SINGLE_TIMEOUT}s")
            all_results.append(trial)

    # --- Co-existence test ---
    coexist_results = run_coexistence_test(device)

    # --- Summary ---
    print(f"\n{'=' * 76}")
    print(f"  PIPELINE BENCHMARK SUMMARY  (target={ENERGY_TARGET})")
    print(f"{'=' * 76}")
    print(
        f"  {'Trial':<20s} {'Time':>6s} {'Att':>4s} "
        f"{'N/att':>5s} {'Nonces/s':>9s} {'Status':>7s}"
    )
    print(f"  {'-' * 55}")

    for r in all_results:
        status = "OK" if r['success'] else "FAIL"
        print(
            f"  {r['label']:<20s} {r['elapsed']:>5.1f}s "
            f"{r['blocks_attempted']:>4d} "
            f"{r['nonces_per_attempt']:>5d} "
            f"{r['nonces_s']:>9.1f} {status:>7s}"
        )

    print(f"  {'-' * 55}")
    for r in coexist_results:
        if 'error' in r:
            print(f"  {r['label']:<20s} {'ERROR':>7s}  {r['error']}")
        else:
            status = "OK" if r['success'] else "FAIL"
            nonces_s = r.get('nonces_s', 0.0)
            print(
                f"  {r['label']:<20s} {r['elapsed']:>5.1f}s "
                f"{r['blocks_attempted']:>4d} "
                f"{r.get('nonces_per_attempt', '?'):>5} "
                f"{nonces_s:>9.1f} {status:>7s}"
            )

    # --- Scaling check ---
    print(f"\n  Scaling (nonces/s should increase with util%):")
    for name in ["SA", "Gibbs"]:
        rates = [
            r['nonces_s'] for r in all_results
            if r['label'].startswith(name) and r['success']
        ]
        if len(rates) >= 2:
            monotonic = all(
                rates[i] <= rates[i + 1]
                for i in range(len(rates) - 1)
            )
            tag = "PASS" if monotonic else "WARN"
            print(
                f"    {name}: "
                f"{' < '.join(f'{r:.1f}' for r in rates)}"
                f"  [{tag}]"
            )

    # --- Pass/fail ---
    failures = [r for r in all_results if not r['success']]
    coexist_failures = [
        r for r in coexist_results
        if 'error' in r or not r.get('success')
    ]
    total_fail = len(failures) + len(coexist_failures)

    print(f"\n  Result: {len(all_results) + len(coexist_results)} "
          f"trials, {total_fail} failures")

    if total_fail > 0:
        print("  BENCHMARK FAILED")
        sys.exit(1)
    else:
        print("  BENCHMARK PASSED")


if __name__ == "__main__":
    main()
