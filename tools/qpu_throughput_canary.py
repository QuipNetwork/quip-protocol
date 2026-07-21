#!/usr/bin/env python3
"""Canary + parameter-sweep tool for QPU throughput tuning.

Two modes:

  ``--mode canary`` (default)
      Submit a small batch of production-size Ising problems at a fixed
      reference (num_reads, annealing_time) and report current QPU
      responsiveness. The headline number is ``effective_qpu_fraction``
      = sum(qpu_access_time) / total wall time. 1.0 means no D-Wave
      queue contention; lower values mean other tenants are using the
      QPU. The canary is a **normalization** instrument — it tells you
      whether the QPU is busy *right now*, not which parameters give
      the best throughput.

  ``--mode sweep``
      Run a grid over ``(num_reads, annealing_time)``, with a canary
      block before each grid point so every datapoint carries its load
      context. For each point, computes per-submission wall, per-
      submission QPU access time, and (if ``--energy-threshold`` is set
      or derivable) a normalized time-to-solution estimate. Prints a
      recommendation line and a CSV body for further analysis.

The tool reuses ``QPU.dwave_sampler.DWaveSamplerWrapper`` (so it picks
up the same solver / region / token plumbing as the miner) and
``shared.quantum_proof_of_work.generate_ising_model_from_nonce`` for
problem generation.

Run manually — per repository policy QPU benchmarks are never invoked
in the background.

Example:

    python tools/qpu_throughput_canary.py --mode canary --n 30
    python tools/qpu_throughput_canary.py --mode sweep \
        --num-reads-grid 64,128,256,512 \
        --annealing-time-grid 5,10,20,40,80,120 \
        --m 10 --canary-n 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make repo modules importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    # dotenv is optional — if absent, env vars must already be set.
    pass

from shared.proc_util import terminate_join
from shared.quantum_proof_of_work import (
    compute_solution_meta,
    evaluate_sampleset,
    pack_spins_hex,
)
from shared.miner_types import BlockRequirements
from shared.ising_feeder import RandomIsingFeeder
from QPU.dwave_miner import DWaveMiner
from QPU.stream_driver import qpu_access_time_us


# ---------------------------------------------------------------------
# Submission primitives
# ---------------------------------------------------------------------


def _make_chain_validator(
    requirements: BlockRequirements,
    nodes: List[int],
    edges: List[Tuple[int, int]],
) -> Callable:
    """Return a per-submission validator using ``evaluate_sampleset``.

    The returned callable replays the chain's full three-gate check on
    each sampleset: energy < difficulty_energy AND num_unique_below_target
    >= min_solutions AND diversity_of_selected_K >= min_diversity. Skip
    validation is True since the canary derived the (h, J) itself and
    doesn't need a per-solution recompute.
    """
    def validate(sampleset, h, J, nonce, salt) -> bool:
        try:
            result = evaluate_sampleset(
                sampleset, requirements, nodes, edges,
                nonce=nonce, salt=salt,
                prev_timestamp=0, start_time=0.0,
                miner_id="canary", miner_type="QPU",
                h=h, J=J,
                skip_validation=True, strict_energy=True,
            )
        except Exception:  # noqa: BLE001 — treat any blow-up as "did not win"
            return False
        return result is not None
    return validate


def _make_energy_only_validator(threshold: float) -> Callable:
    """Legacy energy-only validator for when chain requirements are absent.

    Ignores num_solutions and diversity — produces an inflated p_success
    versus what the chain would accept. Used only when neither
    ``--ws-url`` nor explicit ``--min-solutions``/``--min-diversity``
    are supplied, and the tool prints a warning so the operator knows
    the result is optimistic.
    """
    def validate(sampleset, h, J, nonce, salt) -> bool:
        try:
            best = float(min(sampleset.record.energy))
        except (AttributeError, ValueError):
            return False
        return best < threshold
    return validate



def _run_batch(
    miner: DWaveMiner,
    *,
    num_reads: int,
    annealing_time_us: float,
    n_submissions: int,
    queue_depth: int,
    feeder_buffer_size: int,
    energy_threshold: float,
    label_prefix: str,
    validator: Callable,
    log_every: int = 50,
    stored_solutions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run n submissions through ``DWaveSamplerWrapper.sample_ising_streaming``.

    Uses the same sampler as the production pipeline: background
    ``RandomIsingFeeder`` (ProcessPool) for problem derivation, async D-Wave
    submissions pipelined ``queue_depth`` deep. The new pump yields raw
    reduced samplesets with ``sampleset.info['defect_info']`` set; this
    function reconstructs the full topology for each result so energies and
    the chain validator see the full spin vector.

    Per-submission metrics: ``qpu_access_us`` (from sampleset.info),
    ``best_energy``, ``valid`` (chain or energy-only). Per-submission
    *wall* time isn't meaningful in pipelined mode (submissions overlap);
    overall throughput is reported as ``submissions_per_sec`` instead.

    Progress lines print to stderr every ``log_every`` completions so
    long runs don't go silent.
    """
    nodes = miner.sampler.nodes
    edges = miner.sampler.edges
    # Synthetic 32-byte seeds for the feeder — the canary doesn't care
    # about specific nonces, just that the feeder pumps fresh (h, J)
    # pairs. Production seeds derive from chain head + signer; here
    # urandom keeps each canary run independent.
    feeder = RandomIsingFeeder(
        last_proof_block_hash=os.urandom(32),
        miner_bytes=os.urandom(32),
        nodes=nodes,
        edges=edges,
        buffer_size=feeder_buffer_size,
    )
    # stop_event is passed into the pump so it can cancel in-flight D-Wave
    # futures when we reach n_submissions or on KeyboardInterrupt.
    stop_event = multiprocessing.Event()
    miner._stop_event = stop_event  # kept for legacy callers that check this attr

    per_submission: List[Dict[str, Any]] = []
    overall_start = time.monotonic()
    last_log = overall_start
    completed = 0
    valid_count = 0
    qpu_us_running = 0

    print(
        f"[canary {label_prefix}] reads={num_reads} anneal={annealing_time_us}us "
        f"queue_depth={queue_depth} feeder_buf={feeder_buffer_size} "
        f"target={n_submissions}",
        file=sys.stderr,
    )

    try:
        stream = miner.sampler.sample_ising_streaming(
            feeder,
            num_reads=num_reads,
            annealing_time=annealing_time_us,
            queue_depth=queue_depth,
            energy_threshold_milli=int(energy_threshold * 1000),
            stop_event=stop_event,
        )
        for model, sampleset in stream:
            # Reconstruct the full topology sampleset from the raw reduced
            # result so energies and the chain validator see all variables.
            defect_info = sampleset.info.get("defect_info")
            if defect_info is not None:
                sampleset = miner.sampler.reconstruct_full_sampleset(
                    sampleset, defect_info,
                )
            # Wrap the sampleset-info access too: the future is lazy,
            # so accessing .info or .record can raise a transient
            # D-Wave SolverError (502, 503, network) that would
            # otherwise crash the whole sweep. Mark the attempt as
            # errored and move on — the next iteration's future was
            # already submitted by sample_ising_streaming.
            try:
                qpu_us = qpu_access_time_us(sampleset)
                best_energy = float(min(sampleset.record.energy))
            except (AttributeError, ValueError):
                qpu_us = 0
                best_energy = float("nan")
            except Exception as exc:  # noqa: BLE001
                per_submission.append({
                    "idx": completed,
                    "error": f"sampleset: {type(exc).__name__}: {exc}",
                })
                completed += 1
                if completed >= n_submissions:
                    break
                continue
            try:
                valid = validator(
                    sampleset, model.h, model.J, model.nonce, model.salt,
                )
            except Exception as exc:  # noqa: BLE001
                per_submission.append({
                    "idx": completed,
                    "error": f"validator: {type(exc).__name__}: {exc}",
                })
                completed += 1
                if completed >= n_submissions:
                    break
                continue
            meta, top_5_sols, top_5_es = compute_solution_meta(
                sampleset, energy_threshold,
            )
            per_submission.append({
                "idx": completed,
                "qpu_access_us": qpu_us,
                "best_energy": best_energy,
                "valid": valid,
                "solution_meta": meta,
            })
            # Sidecar capture for "stored" attempts — defined as
            # producing at least one sample below threshold. These are
            # the candidates worth re-analyzing offline: their top-5
            # solutions + the nonce that derived (h, J) let an operator
            # reproduce the BQM and re-run any sampler against it.
            if (
                stored_solutions_path is not None
                and best_energy == best_energy  # NaN filter
                and best_energy < energy_threshold
            ):
                record = {
                    "ts_ns": time.time_ns(),
                    "label": label_prefix,
                    "idx": completed,
                    "params": {
                        "num_reads": num_reads,
                        "annealing_time_us": annealing_time_us,
                    },
                    "nonce_hex": model.nonce.hex(),
                    "salt_hex": model.salt.hex(),
                    "best_energy": best_energy,
                    "qpu_access_us": qpu_us,
                    "valid": valid,
                    "top_5_energies": top_5_es,
                    "top_5_solutions_hex": [
                        pack_spins_hex(s) for s in top_5_sols
                    ],
                }
                try:
                    with open(stored_solutions_path, "a") as fh:
                        fh.write(json.dumps(record) + "\n")
                except OSError as exc:
                    print(
                        f"[canary] WARNING: failed to append stored "
                        f"solution to {stored_solutions_path}: {exc}",
                        file=sys.stderr,
                    )
            completed += 1
            qpu_us_running += qpu_us
            if valid:
                valid_count += 1

            now = time.monotonic()
            if (
                completed % log_every == 0
                or now - last_log > 30.0
                or completed == n_submissions
            ):
                elapsed = now - overall_start
                sub_per_s = completed / elapsed if elapsed > 0 else 0.0
                p_so_far = valid_count / completed if completed else 0.0
                qpu_mean_ms = (qpu_us_running / completed) / 1000.0
                pct = 100.0 * completed / n_submissions
                eta_s = (n_submissions - completed) / sub_per_s if sub_per_s > 0 else 0.0
                # Show the just-completed attempt's energy/valid + rolling
                # aggregates. Matches how mining attempt logs scroll in
                # production miner stdout.
                e_str = (
                    f"{best_energy:.0f}"
                    if best_energy == best_energy else "NaN"
                )
                v_str = "✓" if valid else "·"
                print(
                    f"[canary {label_prefix}] "
                    f"#{completed:>5d}/{n_submissions} "
                    f"e={e_str:>7s} {v_str} qpu={qpu_us/1000:.1f}ms "
                    f"| agg: {sub_per_s:.2f}/s qpu_mean={qpu_mean_ms:.1f}ms "
                    f"valid={valid_count} p={p_so_far:.4f} "
                    f"pct={pct:.1f}% eta={eta_s/60:.1f}min",
                    file=sys.stderr,
                    flush=True,
                )
                last_log = now

            if completed >= n_submissions:
                # Signal the generator to wind down. The pump sees
                # stop_event set, cancels in-flight D-Wave futures, and
                # returns cleanly (no hanging SDK threads).
                stop_event.set()
                break
    finally:
        # Ensure the pump sees stop_event so in-flight D-Wave futures
        # are cancelled and the SDK releases its poll threads.
        stop_event.set()
        try:
            for _ in stream:
                pass
        except Exception:  # noqa: BLE001
            pass
        feeder.stop()

    overall_wall = time.monotonic() - overall_start
    return _summarize(
        per_submission,
        overall_wall,
        num_reads=num_reads,
        annealing_time_us=annealing_time_us,
        n_requested=n_submissions,
        concurrency=queue_depth,
    )


def _moments(xs: List[float]) -> Dict[str, Optional[float]]:
    """Mean, median, stdev, stderr — central tendency + spread + CI input.

    Returns Nones when the sample is too small for the stat to be
    meaningful (variance/stderr need n >= 2). Each setting's own
    variance is the basis for whether differences between settings are
    statistically distinguishable — see ``_tts_bootstrap_ci``.
    """
    if not xs:
        return {
            "n": 0, "mean": None, "median": None,
            "stdev": None, "stderr": None,
        }
    if len(xs) == 1:
        return {
            "n": 1, "mean": float(xs[0]), "median": float(xs[0]),
            "stdev": 0.0, "stderr": None,
        }
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    return {
        "n": len(xs), "mean": mean,
        "median": statistics.median(xs),
        "stdev": sd, "stderr": sd / math.sqrt(len(xs)),
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than normal-approx at small n and extreme p — which
    is exactly the regime we hit when p_success is near 0 or 1.
    Returns (lower, upper) on [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (
        z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _tts_bootstrap_ci(
    cost_samples: List[float],
    valid_flags: List[bool],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Optional[Tuple[float, float, float]]:
    """Bootstrap 95% CI for TTS = mean(cost) / p_success.

    ``cost_samples`` is whatever per-submission cost we're optimizing
    (typically ``qpu_access_us`` so the metric is contention-invariant
    — D-Wave-billed compute per win, independent of cloud queue
    behavior at sample time).

    Jointly resamples (cost, valid) pairs so the correlation between
    per-submission cost and success is preserved. Returns
    (point_estimate, lower_2.5%, upper_97.5%) — None when no
    successes exist in any resample.
    """
    n = len(cost_samples)
    if n == 0 or len(valid_flags) != n:
        return None
    valid_count = sum(valid_flags)
    if valid_count == 0:
        return None
    point_p = valid_count / n
    point_cost = statistics.fmean(cost_samples)
    point_tts = point_cost / point_p

    rng = random.Random(seed)
    tts_samples: List[float] = []
    indices = list(range(n))
    for _ in range(n_resamples):
        boot = [rng.choice(indices) for _ in range(n)]
        bw = [cost_samples[i] for i in boot]
        bv = sum(valid_flags[i] for i in boot)
        if bv == 0:
            continue
        bp = bv / n
        bm = statistics.fmean(bw)
        tts_samples.append(bm / bp)
    if not tts_samples:
        return None
    tts_samples.sort()
    lo = tts_samples[int(0.025 * len(tts_samples))]
    hi = tts_samples[int(0.975 * len(tts_samples)) - 1]
    return (point_tts, lo, hi)


def _summarize(
    per_submission: List[Dict[str, Any]],
    overall_wall_s: float,
    *,
    num_reads: int,
    annealing_time_us: float,
    n_requested: int,
    concurrency: int,
) -> Dict[str, Any]:
    """Reduce per-submission records to headline metrics + variance."""
    ok = [r for r in per_submission if "error" not in r]
    errors = [r for r in per_submission if "error" in r]
    qpu_samples = [r["qpu_access_us"] for r in ok]
    energy_samples = [
        r["best_energy"] for r in ok
        if r["best_energy"] == r["best_energy"]  # NaN filter
    ]
    valid_flags = [bool(r.get("valid")) for r in ok]

    total_qpu_us = sum(qpu_samples)
    effective_fraction = (
        (total_qpu_us / 1e6) / overall_wall_s
        if overall_wall_s > 0 else 0.0
    )

    valid_count = sum(valid_flags)
    p_success = (valid_count / len(ok)) if ok else 0.0
    p_lo, p_hi = _wilson_ci(valid_count, len(ok))

    # Aggregate the per-submission solution_meta into distributions.
    # Raw top-5 solutions are NOT aggregated (and aren't present here
    # either — they're written to the sidecar JSONL on stored attempts).
    sm_aggregate: Dict[str, Any] = {}
    meta_records = [
        r["solution_meta"] for r in ok
        if isinstance(r.get("solution_meta"), dict)
        and r["solution_meta"]
    ]
    scalar_keys = (
        "n_unique_total", "n_unique_below_threshold",
        "top_5_diversity", "top_5_energy_ceiling",
    )
    if meta_records:
        for k in scalar_keys:
            values = [
                d[k] for d in meta_records
                if d.get(k) is not None
            ]
            sm_aggregate[k] = _moments(values) if values else {"n": 0}

    return {
        "params": {
            "num_reads": num_reads,
            "annealing_time_us": annealing_time_us,
            "concurrency": concurrency,
            "submissions_requested": n_requested,
        },
        "summary": {
            "submissions_completed": len(ok),
            "submissions_errored": len(errors),
            "submissions_valid": valid_count,
            "p_success_chain": p_success,
            "p_success_ci95": [p_lo, p_hi],
            "total_wall_s": overall_wall_s,
            "submissions_per_sec": (
                len(ok) / overall_wall_s if overall_wall_s > 0 else 0.0
            ),
            "qpu_access_per_submission_us": _moments(qpu_samples),
            "best_energy": _moments(energy_samples),
            "best_energy_min": (
                min(energy_samples) if energy_samples else None
            ),
            "total_qpu_access_us": total_qpu_us,
            "effective_qpu_fraction": effective_fraction,
            "solution_meta": sm_aggregate,
        },
        "qpu_samples": qpu_samples,
        "valid_flags": valid_flags,
        "per_submission": per_submission,
        "errors": [e["error"] for e in errors],
    }


# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------


def _build_miner(args) -> DWaveMiner:
    """Construct a DWaveMiner using CLI overrides + .env defaults.

    The miner instance is reused across all sweep points so we hit the
    same sampler / connection state production would. Each ``_run_batch``
    call spins up its own ``RandomIsingFeeder`` and tears it down.
    """
    kwargs: Dict[str, Any] = {
        "miner_id": "canary",
        "queue_depth": args.queue_depth,
    }
    if args.solver:
        kwargs["solver_name"] = args.solver
    if args.region:
        kwargs["region"] = args.region
    if args.token:
        kwargs["token"] = args.token
    return DWaveMiner(**kwargs)


def cmd_canary(
    miner: DWaveMiner, args, validator: Callable, energy_threshold: float,
) -> Dict[str, Any]:
    """Single batch at the reference params."""
    return _run_batch(
        miner,
        num_reads=args.ref_num_reads,
        annealing_time_us=args.ref_annealing_time_us,
        n_submissions=args.n,
        queue_depth=args.queue_depth,
        feeder_buffer_size=args.feeder_buffer_size,
        energy_threshold=energy_threshold,
        label_prefix="canary",
        validator=validator,
        log_every=args.log_every,
        stored_solutions_path=args.stored_solutions_path,
    )


def cmd_sweep(
    miner: DWaveMiner, args, validator: Callable, energy_threshold: float,
) -> Dict[str, Any]:
    """Grid sweep over (num_reads, annealing_time) with interleaved canaries."""
    reads_grid = _parse_int_list(args.num_reads_grid)
    anneal_grid = _parse_float_list(args.annealing_time_grid)

    sweep_results: List[Dict[str, Any]] = []
    canary_results: List[Dict[str, Any]] = []

    completed_configs = getattr(args, "completed_configs", set()) or set()
    for ai, anneal in enumerate(anneal_grid):
        for ri, reads in enumerate(reads_grid):
            if (reads, anneal) in completed_configs:
                print(
                    f"[canary sweep-a{ai}r{ri}] SKIPPED "
                    f"(--resume, reads={reads} anneal={anneal}us "
                    f"already complete in sidecar)",
                    file=sys.stderr,
                )
                continue
            # Canary first so its load reading reflects the moment
            # immediately before the measured point — operators can
            # compare each sweep point to its preceding canary.
            if args.canary_n > 0:
                canary = _run_batch(
                    miner,
                    num_reads=args.ref_num_reads,
                    annealing_time_us=args.ref_annealing_time_us,
                    n_submissions=args.canary_n,
                    queue_depth=args.queue_depth,
                    feeder_buffer_size=args.feeder_buffer_size,
                    energy_threshold=energy_threshold,
                    label_prefix=f"canary-a{ai}r{ri}",
                    validator=validator,
                    log_every=args.log_every,
                    stored_solutions_path=args.stored_solutions_path,
                )
            else:
                canary = {"summary": {"effective_qpu_fraction": 0.0}}
            canary_results.append(canary)
            point = _run_batch(
                miner,
                num_reads=reads,
                annealing_time_us=anneal,
                n_submissions=args.m,
                queue_depth=args.queue_depth,
                feeder_buffer_size=args.feeder_buffer_size,
                energy_threshold=energy_threshold,
                label_prefix=f"sweep-a{ai}r{ri}",
                validator=validator,
                log_every=args.log_every,
                stored_solutions_path=args.stored_solutions_path,
            )
            point["canary_ref"] = canary["summary"]
            sweep_results.append(point)

    for point in sweep_results:
        p_success = point["summary"]["p_success_chain"]
        qpu_stats = point["summary"]["qpu_access_per_submission_us"]
        mean_qpu_us = qpu_stats.get("mean")

        # QPU TTS = mean per-submission QPU access time / p_success.
        # This is the contention-invariant metric — D-Wave bills us
        # for qpu_access_time, so this measures QPU compute paid per
        # winning attempt. Independent of cloud queue contention,
        # comparable across runs taken at different times.
        if p_success > 0 and mean_qpu_us is not None:
            qpu_tts_us = mean_qpu_us / p_success
        else:
            qpu_tts_us = None

        # 95% bootstrap CI on the QPU TTS — combines the
        # per-submission QPU variance with the binomial uncertainty
        # in p_success. This is the headline "statistically
        # distinguishable from another setting" number.
        tts_ci = _tts_bootstrap_ci(
            point["qpu_samples"], point["valid_flags"],
        )
        if tts_ci is not None:
            qpu_tts_boot, qpu_tts_lo, qpu_tts_hi = tts_ci
        else:
            qpu_tts_boot = qpu_tts_lo = qpu_tts_hi = None

        point["throughput_analysis"] = {
            "p_success": p_success,
            "p_success_ci95": point["summary"]["p_success_ci95"],
            "qpu_tts_us": qpu_tts_us,
            "qpu_tts_us_bootstrap": qpu_tts_boot,
            "qpu_tts_us_ci95": (
                [qpu_tts_lo, qpu_tts_hi]
                if qpu_tts_boot is not None else None
            ),
            # Reported as diagnostic only — not used for ranking.
            "point_canary_qpu_fraction": (
                point["canary_ref"]["effective_qpu_fraction"]
            ),
        }
        # Drop the raw sample arrays — they were only needed for the
        # bootstrap and would bloat the output JSON otherwise.
        point.pop("qpu_samples", None)
        point.pop("valid_flags", None)

    # Rank by QPU TTS (contention-invariant). Tie-break on the upper
    # bound of the CI so a setting that's nominally best but with
    # huge variance doesn't beat a slightly-higher-but-tight one.
    rated = [
        p for p in sweep_results
        if p["throughput_analysis"]["qpu_tts_us"] is not None
    ]
    if rated:
        best = min(
            rated,
            key=lambda p: (
                p["throughput_analysis"]["qpu_tts_us"],
                p["throughput_analysis"]["qpu_tts_us_ci95"][1]
                if p["throughput_analysis"]["qpu_tts_us_ci95"] else 1e15,
            ),
        )
        ta = best["throughput_analysis"]
        recommendation = {
            "num_reads": best["params"]["num_reads"],
            "annealing_time_us": best["params"]["annealing_time_us"],
            "qpu_tts_us": ta["qpu_tts_us"],
            "qpu_tts_us_ci95": ta["qpu_tts_us_ci95"],
            "p_success": ta["p_success"],
            "p_success_ci95": ta["p_success_ci95"],
        }
    else:
        recommendation = None

    return {
        "mode": "sweep",
        "recommendation": recommendation,
        "points": sweep_results,
        "canaries": canary_results,
    }


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    p.add_argument(
        "--mode", choices=("canary", "sweep"), default="canary",
        help="canary (default): single batch at ref params; "
             "sweep: grid over (num_reads, annealing_time)",
    )
    p.add_argument(
        "--n", type=int, default=30,
        help="canary mode: total submissions (default 30)",
    )
    p.add_argument(
        "--m", type=int, default=15,
        help="sweep mode: submissions per grid point (default 15)",
    )
    p.add_argument(
        "--canary-n", type=int, default=5,
        help="sweep mode: canary probes between grid points (default 5)",
    )
    p.add_argument(
        "--queue-depth", type=int, default=30,
        help="in-flight D-Wave submissions (default 30 — matches "
             "DWaveMiner.queue_depth production value). Controls the "
             "concurrency the streaming sampler keeps in flight.",
    )
    p.add_argument(
        "--feeder-buffer-size", type=int, default=60,
        help="RandomIsingFeeder buffer (default 60 — matches "
             "DWaveMiner.FEEDER_BUFFER_SIZE). Background ProcessPool "
             "pre-derives this many (h, J) pairs.",
    )
    p.add_argument(
        "--num-reads-grid", default="64,128,256,512",
        help="sweep mode: comma-separated num_reads values",
    )
    p.add_argument(
        "--annealing-time-grid", default="5,10,20,40,80,120",
        help="sweep mode: comma-separated annealing_time_us values",
    )
    p.add_argument(
        "--ref-num-reads", type=int, default=128,
        help="reference num_reads for canary submissions (default 128)",
    )
    p.add_argument(
        "--ref-annealing-time-us", type=float, default=20.0,
        help="reference annealing_time for canary submissions "
             "(default 20.0 us)",
    )
    p.add_argument(
        "--ws-url", default=None,
        help="substrate websocket URL "
             "(e.g. wss://qpu-1.nodes.quip.network/rpc) — when set, "
             "fetch chain BlockRequirements (difficulty_energy, "
             "min_solutions, min_diversity) at startup so p_success "
             "matches what the chain actually accepts. Without it the "
             "tool falls back to an energy-only check, which "
             "over-counts wins.",
    )
    p.add_argument(
        "--energy-threshold", type=float, default=None,
        help="override difficulty_energy. Required when --ws-url is "
             "absent; ignored otherwise unless explicitly provided.",
    )
    p.add_argument(
        "--min-solutions", type=int, default=None,
        help="override chain's min_solutions for validation.",
    )
    p.add_argument(
        "--min-diversity", type=float, default=None,
        help="override chain's min_diversity (Hamming-distance floor "
             "in [0, 1]) for validation.",
    )
    p.add_argument("--solver", default=None, help="override solver name")
    p.add_argument("--region", default=None, help="override D-Wave region")
    p.add_argument("--token", default=None, help="override API token")
    p.add_argument(
        "--output", default=None,
        help="write JSON to this path (default: stdout)",
    )
    p.add_argument("--seed", type=int, default=None, help="seed Python RNG")
    p.add_argument(
        "--resume", action="store_true",
        help="If the stored-solutions sidecar from a previous run exists "
             "(default path: <output>.stored.jsonl), skip any sweep grid "
             "points whose stored records indicate >= --m completed "
             "attempts. Partial configs are NOT resumed mid-stream — "
             "they re-run from scratch. Without --resume, the sidecar "
             "is truncated at start (current default).",
    )
    p.add_argument(
        "--store-solutions", default=None,
        help="JSONL sidecar path for archiving 'stored' attempts (best "
             "energy < threshold). Each line records nonce + salt + "
             "params + top-5 solutions in packed hex. Reproducible: "
             "feed nonce to generate_ising_model_from_nonce to recover "
             "the BQM. Default: <output>.stored.jsonl when --output is "
             "set; off when --output is stdout.",
    )
    p.add_argument(
        "--log-every", type=int, default=50,
        help="emit a [canary] aggregate line every N completions "
             "(default 50). Independent of the miner's own logging — "
             "DWaveMiner.sample_ising_streaming prints its [QPU] stream "
             "depth line every 100 completions regardless.",
    )
    return p


def _build_validator(
    args, sampler_nodes: List[int], sampler_edges: List[Tuple[int, int]],
) -> Tuple[Callable, Dict[str, Any]]:
    """Build the per-submission validator + return its metadata.

    Resolves chain requirements with this precedence:
      1. ``--ws-url`` provided → query chain for difficulty
         (max_energy, min_solutions, min_diversity); CLI overrides
         (``--energy-threshold`` etc.) take precedence over chain values.
      2. No ``--ws-url`` but all three CLI overrides present → use them.
      3. ``--ws-url`` absent, partial overrides → error: insufficient
         info for chain validation.
      4. Only ``--energy-threshold`` → energy-only fallback validator;
         tool emits a warning that p_success will be over-counted.
    """
    metadata: Dict[str, Any] = {"source": None}
    chain_energy = chain_min_solutions = chain_min_diversity = None
    if args.ws_url:
        import asyncio
        from substrate.client import SubstrateClient

        async def _fetch():
            cl = SubstrateClient(url=args.ws_url)
            await cl.connect()
            try:
                return await cl.query_difficulty()
            finally:
                await cl.close()

        diff = asyncio.run(_fetch())
        if diff is None:
            print(
                f"[canary] WARNING: --ws-url={args.ws_url} returned no "
                "Difficulty (chain not yet seeded?); using CLI overrides "
                "or energy-only fallback.",
                file=sys.stderr,
            )
        else:
            chain_energy = diff.max_energy
            chain_min_solutions = diff.min_solutions
            chain_min_diversity = diff.min_diversity
            metadata["source"] = "chain"
            metadata["chain_difficulty"] = {
                "max_energy": chain_energy,
                "min_solutions": chain_min_solutions,
                "min_diversity": chain_min_diversity,
            }
            print(
                f"[canary] chain difficulty: max_energy={chain_energy:.2f} "
                f"min_solutions={chain_min_solutions} "
                f"min_diversity={chain_min_diversity:.3f}",
                file=sys.stderr,
            )

    energy = args.energy_threshold if args.energy_threshold is not None else chain_energy
    min_solutions = args.min_solutions if args.min_solutions is not None else chain_min_solutions
    min_diversity = args.min_diversity if args.min_diversity is not None else chain_min_diversity

    if energy is None:
        raise SystemExit(
            "no --energy-threshold and chain query did not yield one; "
            "supply --ws-url or --energy-threshold"
        )

    if min_solutions is None or min_diversity is None:
        print(
            "[canary] WARNING: missing min_solutions/min_diversity — "
            "falling back to energy-only validation. p_success will "
            "over-count vs chain acceptance. Supply --ws-url for full "
            "chain semantics.",
            file=sys.stderr,
        )
        metadata["source"] = metadata.get("source") or "energy_only"
        metadata["validator"] = "energy_only"
        metadata["energy_threshold"] = energy
        return _make_energy_only_validator(energy), metadata

    requirements = BlockRequirements(
        difficulty_energy=energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=2**31 - 1,
    )
    metadata["validator"] = "chain"
    metadata["requirements"] = {
        "difficulty_energy": energy,
        "min_solutions": min_solutions,
        "min_diversity": min_diversity,
    }
    return _make_chain_validator(requirements, sampler_nodes, sampler_edges), metadata


def main() -> int:
    args = _build_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Route every Python logger through stderr — this is what makes
    # production miner output visible (DWaveMiner's "[QPU] stream
    # depth: ..." line every 100 completions from Phase 1, DWave SDK
    # info logs). Without this the canary tool runs the production
    # pipeline silently.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    miner = _build_miner(args)
    print(
        f"[canary] miner ready: {len(miner.sampler.nodes)} nodes, "
        f"{len(miner.sampler.edges)} edges, queue_depth={args.queue_depth}, "
        f"feeder_buf={args.feeder_buffer_size}",
        file=sys.stderr,
    )
    validator, val_meta = _build_validator(
        args, miner.sampler.nodes, miner.sampler.edges,
    )
    # Resolve stored-solutions sidecar path. Defaults to
    # <output>.stored.jsonl when --output is set, off otherwise (we
    # can't pair a sidecar with stdout output).
    stored_path: Optional[Path] = None
    if args.store_solutions:
        stored_path = Path(args.store_solutions)
    elif args.output:
        stored_path = Path(args.output).with_suffix(
            Path(args.output).suffix + ".stored.jsonl"
        )
    completed_configs: set = set()
    if stored_path is not None:
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and stored_path.exists():
            # Scan the existing sidecar for fully-completed configs.
            # A config is "complete" when stored records show max_idx
            # >= args.m - 1 (we ran at least m attempts). Skipped configs
            # don't produce new sweep output; user merges JSONs manually
            # if they want a unified view across runs.
            max_idx: Dict[Tuple[int, float], int] = {}
            with open(stored_path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (
                        rec["params"]["num_reads"],
                        rec["params"]["annealing_time_us"],
                    )
                    idx = rec.get("idx", -1)
                    if idx > max_idx.get(key, -1):
                        max_idx[key] = idx
            threshold_m = (
                args.m if args.mode == "sweep" else args.n
            ) - 1
            completed_configs = {
                key for key, mi in max_idx.items() if mi >= threshold_m
            }
            print(
                f"[canary] resume: {len(completed_configs)} configs "
                f"already complete in {stored_path}; will skip them",
                file=sys.stderr,
            )
        elif not args.resume:
            # Truncate at start so each run produces a fresh sidecar.
            stored_path.write_text("")
        print(
            f"[canary] stored-solutions sidecar: {stored_path}",
            file=sys.stderr,
        )
    args.stored_solutions_path = stored_path
    args.completed_configs = completed_configs
    # Stream API requires the threshold to be passed for fast-path
    # reconstruction; mirror our validator's energy threshold so
    # non-promising samplesets get correctly skipped.
    energy_threshold = (
        val_meta.get("requirements", {}).get("difficulty_energy")
        or val_meta.get("energy_threshold")
    )

    if args.mode == "canary":
        result = cmd_canary(miner, args, validator, energy_threshold)
        s = result["summary"]
        print(
            f"[canary] {s['submissions_completed']}/{args.n} ok in "
            f"{s['total_wall_s']:.2f}s "
            f"({s['submissions_per_sec']:.2f}/s) | "
            f"effective_qpu_fraction={s['effective_qpu_fraction']:.3f} | "
            f"qpu_mean={s['qpu_access_per_submission_us']['mean']:.0f}us "
            f"(sd={s['qpu_access_per_submission_us']['stdev']:.0f}) | "
            f"p_success={s['p_success_chain']:.4f} "
            f"({s['submissions_valid']}/{s['submissions_completed']})",
            file=sys.stderr,
        )
    else:
        result = cmd_sweep(miner, args, validator, energy_threshold)
        rec = result["recommendation"]
        if rec:
            lo, hi = rec["qpu_tts_us_ci95"] or (None, None)
            ci_str = (
                f" 95%CI=[{lo/1e3:.0f}, {hi/1e3:.0f}]ms"
                if lo is not None else ""
            )
            print(
                f"[sweep] recommendation: num_reads={rec['num_reads']} "
                f"annealing_time_us={rec['annealing_time_us']} | "
                f"qpu_tts={rec['qpu_tts_us']/1e3:.0f}ms{ci_str} "
                f"p_succ={rec['p_success']:.2f}",
                file=sys.stderr,
            )
        else:
            print(
                "[sweep] no rated grid points — every (num_reads, "
                "annealing_time) point either errored or had zero "
                "successes against the energy threshold.",
                file=sys.stderr,
            )

    result["validator"] = val_meta
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(payload)
        print(f"[canary] wrote {args.output}", file=sys.stderr)
    else:
        print(payload)

    _force_shutdown(miner)
    return 0


def _force_shutdown(miner) -> None:
    """Tear down resources that otherwise block interpreter exit.

    Two stubborn holders keep the canary alive after its work is done:

    1. The D-Wave cloud client's non-daemon submission/polling thread —
       closed via ``DWaveSamplerWrapper.close()``. Without this the
       process sleeps forever at exit waiting on that thread.
    2. Leftover spawn-context feeder workers — each ``_run_batch``
       stops its own feeder, but a worker caught mid-derivation can
       ignore SIGTERM and survive. We reap any remaining children of
       THIS process here.

    After cleanup we ``os._exit(0)`` rather than returning, because the
    D-Wave SDK and multiprocessing both register atexit hooks that can
    re-block; all output is already flushed to disk by this point, so a
    hard exit is safe and deterministic.
    """
    try:
        miner.sampler.close()
    except Exception:  # noqa: BLE001
        pass

    children = multiprocessing.active_children()
    for child in children:
        try:
            terminate_join(child, 2)
        except Exception:  # noqa: BLE001
            pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
