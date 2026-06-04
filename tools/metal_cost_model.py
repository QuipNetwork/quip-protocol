#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Measure the Metal SA GPU cost surface to tune the sweep-yielding strategy.

The Metal miner derives ``(num_reads, num_sweeps)`` from the block's target
energy (``MetalMiner.adapt_parameters``). A single monolithic command buffer
that runs all ``num_sweeps`` can't be preempted by macOS, so it freezes the UI
(see ``GPU/metal_sa.py`` sweep-chunking). To size the chunks well we need the
*physical* cost: how long a dispatch takes and how much of the GPU it occupies
as a function of ``(num_reads, num_sweeps)``.

This tool runs real Metal dispatches (monolithic, ``target_dispatch_ms=None``)
over (a) the realistic operating points for a set of target energies and (b) a
grid that varies sweeps and reads independently, recording per-cell wall-time
and (optionally) GPU active-residency. It writes a CSV and prints a derived
model: the per-beta cost, the chunk size that hits a target dispatch time, and
the lockup magnitude of each realistic point.

It deliberately saturates the GPU — run it when you can tolerate UI jank (the
very condition we're characterizing). Metal/macOS only.

Example:
  .quip/bin/python tools/metal_cost_model.py \
      --energies -14900,-15000,-15250 --repeats 5 --residency \
      --out metal_cost_model.csv
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dwave_topologies import DEFAULT_TOPOLOGY  # noqa: E402
from GPU import macos_sensors  # noqa: E402
from GPU.metal_miner import MetalMiner  # noqa: E402
from GPU.metal_sa import MetalSASampler  # noqa: E402
from GPU.metal_utils import compute_beta_schedule  # noqa: E402
from shared.ising_model import IsingModel  # noqa: E402
from shared.quantum_proof_of_work import (  # noqa: E402
    derive_nonce,
    generate_ising_model_from_nonce,
)

# Target per-command-buffer wall-time used to translate the measured cost into a
# recommended chunk size. Mirrors GPU.metal_scheduler.ACTIVE_DISPATCH_TARGET_MS.
DEFAULT_TARGET_DISPATCH_MS = 8.0
_CSV_COLUMNS = (
    "kind", "target_energy", "num_reads", "num_sweeps", "n_betas",
    "wall_ms", "ms_per_beta", "us_per_read_beta", "gpu_residency_pct",
)


def _build_model(sampler: MetalSASampler, seed: int) -> IsingModel:
    """One deterministic IsingModel over the sampler's topology (per-cell fixed)."""
    rng = np.random.RandomState(seed)
    salt = rng.bytes(32)
    miner_bytes = b"cost-model".ljust(32, b"\x00")
    last_proof_block_hash = b"cost_model_padding_to_32_bytes!!"
    nonce = derive_nonce(last_proof_block_hash, miner_bytes, salt)
    h, j = generate_ising_model_from_nonce(nonce, sampler.nodes, sampler.edges)
    return IsingModel(h=h, J=j, nonce=nonce, salt=salt)


def _measure_cell(
    sampler: MetalSASampler,
    model: IsingModel,
    *,
    num_reads: int,
    num_sweeps: int,
    repeats: int,
    residency_secs: float,
    seed: int,
) -> Tuple[float, Optional[int]]:
    """Median monolithic-dispatch wall-ms and (optional) GPU residency %.

    Runs one warmup dispatch (discarded — pipeline/buffer pool warmup), then
    ``repeats`` timed dispatches. When ``residency_secs > 0`` it then runs
    back-to-back dispatches for that long, sampling the GPU residency between
    them, and returns the median sample.
    """
    beta_arr, beta_range = compute_beta_schedule(
        model.h, model.J, num_sweeps, 1, None, "geometric", None,
    )
    common = dict(
        num_reads=num_reads, beta_schedule_arr=beta_arr, beta_range=beta_range,
        beta_schedule_type="geometric", num_sweeps_per_beta=1, seed=seed,
        target_dispatch_ms=None,
    )
    sampler._dispatch_batch([model], **common)  # warmup, discarded

    samples_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sampler._dispatch_batch([model], **common)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_ms = statistics.median(samples_ms)

    residency: Optional[int] = None
    if residency_secs > 0:
        reads: List[int] = []
        deadline = time.perf_counter() + residency_secs
        while time.perf_counter() < deadline:
            sampler._dispatch_batch([model], **common)
            reads.append(macos_sensors.gpu_active_residency())
        residency = int(statistics.median(reads)) if reads else None

    return wall_ms, residency


def _realistic_points(
    energies: List[float], min_diversity: float, min_solutions: int,
    num_nodes: int, num_edges: int,
) -> List[Tuple[str, float, int, int]]:
    """(kind, energy, num_reads, num_sweeps) the miner would actually run."""
    points = []
    for energy in energies:
        params = MetalMiner.adapt_parameters(
            energy, min_diversity, min_solutions,
            num_nodes=num_nodes, num_edges=num_edges,
        )
        points.append(
            ("realistic", energy,
             int(params["num_reads"]), int(params["num_sweeps"])),
        )
    return points


def _grid_cells(
    realistic: List[Tuple[str, float, int, int]],
    reads_grid: List[int], sweeps_grid: List[int],
) -> List[Tuple[str, float, int, int]]:
    """Realistic points + a sweeps-axis sweep + a reads-axis sweep, deduped.

    The realistic points anchor the model to production; the two axis sweeps
    (vary sweeps at a fixed reads, vary reads at a fixed sweeps) let a linear
    fit separate per-beta cost from fixed overhead and the reads/occupancy term.
    """
    # List form so an empty ``realistic`` (e.g. ``--energies ""``) still has the
    # ADAPT_MAX_READS element rather than ``max(int)`` raising TypeError.
    ref_reads = max([MetalMiner.ADAPT_MAX_READS, *(r for _, _, r, _ in realistic)])
    ref_sweeps = sweeps_grid[len(sweeps_grid) // 2]
    cells = list(realistic)
    cells += [("sweeps-axis", 0.0, ref_reads, s) for s in sweeps_grid]
    cells += [("reads-axis", 0.0, r, ref_sweeps) for r in reads_grid]
    seen: set = set()
    deduped = []
    for cell in cells:
        key = (cell[2], cell[3])
        if key not in seen:
            seen.add(key)
            deduped.append(cell)
    return deduped


def _chunk_sweep(
    sampler: MetalSASampler,
    model: IsingModel,
    *,
    num_reads: int,
    num_sweeps: int,
    targets: List[float],
    repeats: int,
    residency_secs: float,
    seed: int,
) -> List[Dict[str, object]]:
    """Total wall-time + GPU residency for one point across target_dispatch_ms.

    Drives the REAL chunked path (``_dispatch_batch`` with the self-calibrating
    controller) at each target (0 ⇒ monolithic). Captures the two things the
    monolithic grid can't: per-command-buffer overhead (total time vs target)
    and whether chunking lets the compositor breathe (GPU residency vs target).
    """
    beta_arr, beta_range = compute_beta_schedule(
        model.h, model.J, num_sweeps, 1, None, "geometric", None,
    )
    base = dict(
        num_reads=num_reads, beta_schedule_arr=beta_arr, beta_range=beta_range,
        beta_schedule_type="geometric", num_sweeps_per_beta=1, seed=seed,
    )

    def _one(target_ms: Optional[float]) -> None:
        sampler._dispatch_batch([model], target_dispatch_ms=target_ms, **base)

    rows: List[Dict[str, object]] = []
    for tgt in targets:
        target_ms = None if tgt <= 0 else float(tgt)
        # Reset calibration ONCE per target so the warmup converges the chunk
        # size for this target; the timed repeats then measure steady-state
        # chunked throughput, not the cold-start re-calibration cost.
        sampler._betas_per_chunk = None
        _one(target_ms)  # warmup
        samples_ms: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _one(target_ms)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = statistics.median(samples_ms)
        chunk = "mono" if target_ms is None else sampler._betas_per_chunk
        residency: Optional[int] = None
        if residency_secs > 0:
            res: List[int] = []
            deadline = time.perf_counter() + residency_secs
            while time.perf_counter() < deadline:
                _one(target_ms)
                res.append(macos_sensors.gpu_active_residency())
            residency = int(statistics.median(res)) if res else None
        rows.append({
            "target_ms": "mono" if target_ms is None else tgt,
            "chunk_betas": chunk,
            "wall_ms": round(wall_ms, 1),
            "gpu_residency_pct": "" if residency is None else residency,
        })
    return rows


def _print_chunk_sweep(
    rows: List[Dict[str, object]], num_reads: int, num_sweeps: int,
) -> None:
    mono = next((r for r in rows if r["target_ms"] == "mono"), None)
    base_ms = float(mono["wall_ms"]) if mono else 0.0
    print(
        f"\n=== chunk-overhead + residency sweep "
        f"(reads={num_reads}, sweeps={num_sweeps}) ===",
    )
    for r in rows:
        wall = float(r["wall_ms"])
        overhead = (wall - base_ms) / base_ms * 100 if base_ms else 0.0
        res = r["gpu_residency_pct"]
        print(
            f"  target={str(r['target_ms']):>4} ms  chunk={str(r['chunk_betas']):>4} "
            f"betas  wall={wall:8.1f} ms  (+{overhead:5.1f}% vs mono)  "
            f"GPU={res if res != '' else '-':>3}%",
        )
    print(
        "Read: low overhead% ⇒ chunking is cheap (fixed cost is per-dispatch, "
        "not per-buffer); GPU-residency dropping below 100 as the target shrinks "
        "⇒ the compositor is getting windows (smaller/no sleep needed).",
    )


def _row(
    kind: str, energy: float, reads: int, sweeps: int,
    wall_ms: float, residency: Optional[int],
) -> Dict[str, object]:
    return {
        "kind": kind,
        "target_energy": f"{energy:.0f}" if kind == "realistic" else "",
        "num_reads": reads,
        "num_sweeps": sweeps,
        "n_betas": sweeps,
        "wall_ms": round(wall_ms, 3),
        "ms_per_beta": round(wall_ms / sweeps, 5),
        "us_per_read_beta": round(wall_ms * 1000.0 / (sweeps * reads), 4),
        "gpu_residency_pct": "" if residency is None else residency,
    }


def _write_csv(rows: List[Dict[str, object]], out: Path) -> None:
    import csv

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _print_model(rows: List[Dict[str, object]], target_ms: float) -> None:
    """Print the derived cost model + the recommended chunking strategy."""
    sweeps_axis = [r for r in rows if r["kind"] == "sweeps-axis"]
    print("\n=== derived cost model ===")
    if len(sweeps_axis) >= 2:
        xs = np.array([float(r["num_sweeps"]) for r in sweeps_axis])
        ys = np.array([float(r["wall_ms"]) for r in sweeps_axis])
        slope, intercept = np.polyfit(xs, ys, 1)
        print(
            f"wall_ms ≈ {intercept:.2f} + {slope:.5f} * num_sweeps "
            f"(at {sweeps_axis[0]['num_reads']} reads) — "
            f"{slope * 1000:.2f} us/beta, {intercept:.2f} ms fixed overhead",
        )
        if slope > 0:
            chunk = max(1, int((target_ms - intercept) / slope))
            print(
                f"chunk size for {target_ms:.0f} ms/buffer ≈ {chunk} betas "
                "(after overhead)",
            )
    print("\n=== realistic operating points (lockup magnitude) ===")
    for r in rows:
        if r["kind"] != "realistic":
            continue
        wall = float(r["wall_ms"])
        per_beta = float(r["ms_per_beta"])
        chunk = max(1, int(target_ms / per_beta)) if per_beta > 0 else r["n_betas"]
        n_chunks = (int(r["n_betas"]) + chunk - 1) // chunk
        res = r["gpu_residency_pct"]
        print(
            f"  E={r['target_energy']:>7}  reads={r['num_reads']:>3} "
            f"sweeps={r['num_sweeps']:>4}  wall={wall:8.1f} ms  "
            f"GPU={res if res != '' else '-':>3}%  -> {target_ms:.0f}ms chunks: "
            f"{chunk} betas x {n_chunks} buffers",
        )
    print(
        "\nStrategy: ACTIVE/LOW splits each dispatch to ~the chunk size above "
        f"(target {target_ms:.0f} ms); IDLE runs monolithic. Insert a sleep "
        "between chunks to cap GPU%-residency below 100 if it stays saturated.",
    )


def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the Metal SA GPU cost surface for sweep-yielding.",
    )
    parser.add_argument("--energies", default="-14900,-15000,-15250")
    parser.add_argument("--reads-grid", default="32,64,128,256")
    parser.add_argument("--sweeps-grid", default="64,128,256,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-solutions", type=int, default=70)
    parser.add_argument("--min-diversity", type=float, default=0.0)
    parser.add_argument(
        "--residency", action="store_true",
        help="also measure GPU active-residency (slower, more saturating)",
    )
    parser.add_argument("--residency-secs", type=float, default=1.0)
    parser.add_argument(
        "--target-ms", type=float, default=DEFAULT_TARGET_DISPATCH_MS,
    )
    parser.add_argument(
        "--chunk-sweep", action="store_true",
        help="measure total wall-time + GPU residency on the real chunked path "
             "across --chunk-targets (instead of the monolithic cost grid)",
    )
    parser.add_argument(
        "--chunk-targets", default="0,33,16,8,4,2",
        help="target_dispatch_ms values for --chunk-sweep (0 = monolithic)",
    )
    parser.add_argument(
        "--chunk-energy", type=float, default=None,
        help="target energy whose (reads,sweeps) the chunk-sweep uses "
             "(default: the first --energies entry)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="metal_cost_model.csv")
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny grid for a quick functional check (no full sweep)",
    )
    args = parser.parse_args()

    energies = _parse_float_list(args.energies)
    reads_grid = _parse_int_list(args.reads_grid)
    sweeps_grid = _parse_int_list(args.sweeps_grid)
    if args.smoke:
        energies = energies[:1]
        reads_grid = [32]
        sweeps_grid = [64, 128]

    graph = DEFAULT_TOPOLOGY.graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    sampler = MetalSASampler()
    sampler.prepare_topology()
    model = _build_model(sampler, args.seed)

    realistic = _realistic_points(
        energies, args.min_diversity, args.min_solutions, num_nodes, num_edges,
    )

    if args.chunk_sweep:
        energy = args.chunk_energy if args.chunk_energy is not None else energies[0]
        params = MetalMiner.adapt_parameters(
            energy, args.min_diversity, args.min_solutions,
            num_nodes=num_nodes, num_edges=num_edges,
        )
        reads, sweeps = int(params["num_reads"]), int(params["num_sweeps"])
        print(
            f"topology: {num_nodes} nodes, {num_edges} edges | chunk-sweep at "
            f"E={energy:.0f} -> reads={reads}, sweeps={sweeps}",
        )
        sweep_rows = _chunk_sweep(
            sampler, model, num_reads=reads, num_sweeps=sweeps,
            targets=_parse_float_list(args.chunk_targets), repeats=args.repeats,
            residency_secs=args.residency_secs if args.residency else 0.0,
            seed=args.seed,
        )
        _print_chunk_sweep(sweep_rows, reads, sweeps)
        if args.out != parser.get_default("out"):
            print(
                "note: --chunk-sweep is print-only; --out is not written",
                file=sys.stderr,
            )
        return 0

    cells = _grid_cells(realistic, reads_grid, sweeps_grid)

    print(
        f"topology: {num_nodes} nodes, {num_edges} edges | "
        f"{len(cells)} cells x {args.repeats} repeats"
        + (" + residency bursts" if args.residency else ""),
    )
    rows: List[Dict[str, object]] = []
    for kind, energy, reads, sweeps in cells:
        wall_ms, residency = _measure_cell(
            sampler, model, num_reads=reads, num_sweeps=sweeps,
            repeats=args.repeats,
            residency_secs=args.residency_secs if args.residency else 0.0,
            seed=args.seed,
        )
        row = _row(kind, energy, reads, sweeps, wall_ms, residency)
        rows.append(row)
        print(
            f"  [{kind:>11}] reads={reads:>3} sweeps={sweeps:>4} -> "
            f"{wall_ms:8.1f} ms ({row['ms_per_beta']} ms/beta)"
            + (f", GPU {residency}%" if residency is not None else ""),
        )

    out = Path(args.out)
    _write_csv(rows, out)
    print(f"\nwrote {len(rows)} rows -> {out}")
    _print_model(rows, args.target_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
