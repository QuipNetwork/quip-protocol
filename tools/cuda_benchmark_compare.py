#!/usr/bin/env python3
"""Comparative CUDA kernel profiling benchmark suite.

Runs Gibbs and SA kernels on identical Ising models and produces
flamegraphs, comparison tables, and charts.

Models are pipelined into each kernel for realistic throughput:
- Gibbs: batches of GIBBS_BATCH_SIZE via sample_ising(list, list)
- SA: batches of SA_BATCH_SIZE via ring buffer pipelining

Usage:
    python tools/cuda_benchmark_compare.py test1 [options]
    python tools/cuda_benchmark_compare.py test2 [options]
    python tools/cuda_benchmark_compare.py both  [options]

Options:
    --target-energy FLOAT   Energy target (default: -14900.0)
    --num-models INT        Models for test1 CLT (default: 50)
    --seed-start INT        Starting nonce seed (default: 0)
    --max-attempts INT      Max mining attempts for test2 (default: 500)
    --output-dir PATH       Output directory (default: benchmarks/compare)
    --gpu INT               GPU device (default: 1)
    --json                  Also dump raw JSON data
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cupy as cp
import numpy as np

# Project imports
from dwave_topologies import DEFAULT_TOPOLOGY
from GPU.cuda_gibbs_sa import CudaGibbsSampler
from GPU.cuda_kernel import CudaKernelRealSA
from shared.beta_schedule import _default_ising_beta_range
from shared.energy_utils import energy_to_difficulty
from shared.quantum_proof_of_work import generate_ising_model_from_nonce
from tools.cuda_profile_regions import (
    SA_REGION_NAMES,
    GIBBS_REGION_NAMES,
    generate_flamegraph_html,
    get_gpu_info,
    get_topology_stats,
    print_gibbs_profile,
    print_precise_ticks,
    print_sa_profile,
)


# Pipeline batch sizes matching hardware capacity
GIBBS_BATCH_SIZE = 12
SA_BATCH_SIZE = 48  # One job per SM on A4000

# Adaptive parameter bounds for CUDA miners
ADAPT_MIN_SWEEPS = 256
ADAPT_MAX_SWEEPS = 2048
ADAPT_MIN_READS = 64
ADAPT_MAX_READS = 256


def compute_adaptive_params(target_energy):
    """Compute sweeps/reads from target energy via difficulty curve.

    Returns:
        Dict with keys: difficulty, num_sweeps, num_reads,
        num_betas, sweeps_per_beta.
    """
    difficulty = energy_to_difficulty(target_energy)
    num_sweeps = int(
        ADAPT_MIN_SWEEPS
        + difficulty * (ADAPT_MAX_SWEEPS - ADAPT_MIN_SWEEPS)
    )
    num_reads = int(
        ADAPT_MIN_READS
        + difficulty * (ADAPT_MAX_READS - ADAPT_MIN_READS)
    )
    return {
        "difficulty": difficulty,
        "num_sweeps": num_sweeps,
        "num_reads": num_reads,
        "num_betas": num_sweeps,
        "sweeps_per_beta": 1,
    }


def generate_models(num_models, seed_start):
    """Generate deterministic Ising models from topology + nonce seeds.

    Returns:
        List of (seed, h_dict, J_dict) tuples.
    """
    graph = DEFAULT_TOPOLOGY.graph
    nodes = sorted(graph.nodes())
    edges = list(graph.edges())
    models = []
    for i in range(num_models):
        seed = seed_start + i
        h, J = generate_ising_model_from_nonce(seed, nodes, edges)
        models.append((seed, h, J))
    return models


# ── Batched kernel dispatch ─────────────────────────────────


def run_gibbs_batch(models, num_reads, num_sweeps, profile):
    """Dispatch a batch of models through Gibbs in a single kernel.

    Args:
        models: List of (seed, h_dict, J_dict) tuples.
        num_reads: Reads per model.
        num_sweeps: Total sweeps.
        profile: Whether to collect profiling data.

    Returns:
        (energies, wall_time, profile_data_or_None)
    """
    sampler = CudaGibbsSampler(profile=profile)
    h_list = [h for _, h, _ in models]
    J_list = [J for _, _, J in models]
    t0 = time.perf_counter()
    results = sampler.sample_ising(
        h_list, J_list,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
    )
    wall = time.perf_counter() - t0
    energies = [float(r.first.energy) for r in results]
    data = sampler.get_profile_data() if profile else None
    return energies, wall, data


def run_sa_batch(models, num_reads, num_betas,
                 sweeps_per_beta, profile):
    """Pipeline a batch of models through SA ring buffer.

    Enqueues all jobs, signals batch ready, dequeues all results.

    Args:
        models: List of (seed, h_dict, J_dict) tuples.
        num_reads: Reads per model.
        num_betas: Beta schedule steps.
        sweeps_per_beta: Sweeps per beta.
        profile: Whether to collect profiling data.

    Returns:
        (energies, wall_time, profile_data_or_None)
    """
    batch_size = len(models)
    ring_size = max(batch_size, 16)
    kernel = CudaKernelRealSA(
        profile=profile, verbose=False,
        ring_size=ring_size,
    )
    capped_reads = min(num_reads, 256)

    t0 = time.perf_counter()
    for i, (seed, h, J) in enumerate(models):
        beta_range = _default_ising_beta_range(h, J)
        kernel.enqueue_job(
            job_id=i, h=h, J=J,
            num_reads=capped_reads,
            num_betas=num_betas,
            num_sweeps_per_beta=sweeps_per_beta,
            beta_range=beta_range,
        )
    kernel.signal_batch_ready()

    # Dequeue all results
    results = {}
    deadline = time.time() + 1200.0
    while len(results) < batch_size and time.time() < deadline:
        r = kernel.try_dequeue_result()
        if r is not None:
            results[r["job_id"]] = r
        else:
            time.sleep(0.01)

    wall = time.perf_counter() - t0
    kernel.stop_immediate()
    data = kernel.get_profile_data() if profile else None

    energies = [
        float(results[i]["min_energy"]) if i in results
        else float("nan")
        for i in range(batch_size)
    ]
    return energies, wall, data


# ── Profile aggregation ─────────────────────────────────────


def build_aggregate_profile(all_profiles, kernel_name):
    """Average profile data across multiple batches.

    Concatenates all active rows from all batch profile arrays
    and computes the grand mean.

    Args:
        all_profiles: List of raw profile arrays (one per batch).
        kernel_name: 'gibbs' or 'sa'.

    Returns:
        Synthetic profile array suitable for generate_flamegraph_html.
    """
    num_regions = (
        12 if kernel_name == "gibbs"
        else CudaKernelRealSA.SA_NUM_REGIONS
    )
    active_rows = []
    for data in all_profiles:
        active = data[data[:, 0] > 0]
        if len(active) > 0:
            active_rows.append(active)

    assert len(active_rows) > 0, (
        f"No active threads across all {kernel_name} profiles"
    )
    all_active = np.concatenate(active_rows, axis=0)
    grand_avg = all_active.mean(axis=0)
    return grand_avg.reshape(1, num_regions)


# ── Plotting ─────────────────────────────────────────────────


def plot_comparison_table(rows, output_path):
    """Render per-model energy comparison as a table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = [
        "Model", "Gibbs E", "SA E", "Delta",
    ]
    cell_text = []
    for r in rows:
        delta = r["sa_energy"] - r["gibbs_energy"]
        cell_text.append([
            str(r["model_id"]),
            f'{r["gibbs_energy"]:.0f}',
            f'{r["sa_energy"]:.0f}',
            f'{delta:+.0f}',
        ])

    fig_height = max(4, 0.4 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=headers,
        loc="center", cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for col in range(len(headers)):
        table[0, col].set_facecolor("#2c3e50")
        table[0, col].set_text_props(color="white", weight="bold")
    plt.title("Gibbs vs SA Per-Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Table saved to {output_path}")


def plot_energy_comparison(rows, output_path):
    """Bar chart comparing best energies per model per kernel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r["model_id"] for r in rows]
    gibbs = [r["gibbs_energy"] for r in rows]
    sa = [r["sa_energy"] for r in rows]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, gibbs, width, label="Gibbs", color="#e74c3c")
    ax.bar(x + width / 2, sa, width, label="SA", color="#3498db")
    ax.set_xlabel("Model ID")
    ax.set_ylabel("Best Energy")
    ax.set_title("Gibbs vs SA Best Energy Per Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, fontsize=7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Energy chart saved to {output_path}")


def plot_energy_timeline(gibbs_attempts, sa_attempts, target, output_path):
    """Plot attempt # vs best_energy for both kernels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    if gibbs_attempts:
        ax.plot(
            [a["attempt"] for a in gibbs_attempts],
            [a["best_energy"] for a in gibbs_attempts],
            "o-", color="#e74c3c", label="Gibbs", markersize=3,
        )
    if sa_attempts:
        ax.plot(
            [a["attempt"] for a in sa_attempts],
            [a["best_energy"] for a in sa_attempts],
            "s-", color="#3498db", label="SA", markersize=3,
        )
    ax.axhline(
        target, color="#2ecc71", linestyle="--",
        label=f"Target ({target})",
    )
    ax.set_xlabel("Attempt #")
    ax.set_ylabel("Best Energy")
    ax.set_title("Mining Energy Timeline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Timeline saved to {output_path}")


def plot_overhead_comparison(
    gibbs_inst, gibbs_uninst, sa_inst, sa_uninst, output_path
):
    """Bar chart comparing instrumented vs uninstrumented wall times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Gibbs", "SA"]
    inst = [gibbs_inst, sa_inst]
    uninst = [gibbs_uninst, sa_uninst]
    overhead_pct = [
        (i - u) / u * 100 if u > 0 else 0
        for i, u in zip(inst, uninst)
    ]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2, uninst, width,
        label="Uninstrumented", color="#2ecc71",
    )
    bars2 = ax.bar(
        x + width / 2, inst, width,
        label="Instrumented", color="#e67e22",
    )

    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax.text(
            b2.get_x() + b2.get_width() / 2,
            b2.get_height() + 0.01,
            f"+{overhead_pct[i]:.1f}%",
            ha="center", fontsize=10,
        )

    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title("Profiling Instrumentation Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Overhead chart saved to {output_path}")


def plot_mining_summary(gibbs_summary, sa_summary, output_path):
    """Table image summarizing mining test results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = [
        "Kernel", "Attempts", "Total Time (s)",
        "Overhead %", "Final Energy", "Mined?",
    ]
    cell_text = []
    for name, s in [("Gibbs", gibbs_summary), ("SA", sa_summary)]:
        cell_text.append([
            name,
            str(s["attempts"]),
            f'{s["total_time"]:.2f}',
            f'{s["overhead_pct"]:.1f}%',
            f'{s["final_energy"]:.0f}',
            "Yes" if s["mined"] else "No",
        ])

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=headers,
        loc="center", cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for col in range(len(headers)):
        table[0, col].set_facecolor("#2c3e50")
        table[0, col].set_text_props(color="white", weight="bold")
    plt.title("Mining Test Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary saved to {output_path}")


# ── Test orchestration ───────────────────────────────────────


def run_test1(num_models, target_energy, output_dir, dump_json):
    """Test 1: CLT comparison across N models with pipelined dispatch."""
    params = compute_adaptive_params(target_energy)
    print(f"\n{'=' * 60}")
    print(f"TEST 1: CLT Comparison ({num_models} models)")
    print(f"{'=' * 60}")
    print(f"  Target energy: {target_energy}")
    print(f"  Difficulty:    {params['difficulty']:.3f}")
    print(f"  Sweeps:        {params['num_sweeps']}")
    print(f"  Reads:         {params['num_reads']}")
    print(f"  Gibbs batch:   {GIBBS_BATCH_SIZE}")
    print(f"  SA batch:      {SA_BATCH_SIZE}")
    print()

    gpu_name, clock_khz, _ = get_gpu_info()
    models = generate_models(num_models, seed_start=0)

    # ── Gibbs: batch dispatch ──
    print("  Gibbs (pipelined)...")
    gibbs_energies = []
    gibbs_profiles = []
    gibbs_total_wall = 0.0

    for start in range(0, num_models, GIBBS_BATCH_SIZE):
        batch = models[start:start + GIBBS_BATCH_SIZE]
        label = f"{start}-{start + len(batch) - 1}"
        print(f"    Batch [{label}]...", end=" ", flush=True)
        energies, wall, data = run_gibbs_batch(
            batch, params["num_reads"], params["num_sweeps"],
            profile=True,
        )
        gibbs_energies.extend(energies)
        gibbs_profiles.append(data)
        gibbs_total_wall += wall
        e_min = min(energies)
        e_max = max(energies)
        print(
            f"done ({wall:.2f}s)  "
            f"E: [{e_min:.0f}, {e_max:.0f}]"
        )

    # ── SA: ring buffer pipelining ──
    print("\n  SA (pipelined)...")
    sa_energies = []
    sa_profiles = []
    sa_total_wall = 0.0

    for start in range(0, num_models, SA_BATCH_SIZE):
        batch = models[start:start + SA_BATCH_SIZE]
        label = f"{start}-{start + len(batch) - 1}"
        print(f"    Batch [{label}]...", end=" ", flush=True)
        energies, wall, data = run_sa_batch(
            batch, params["num_reads"],
            params["num_betas"], params["sweeps_per_beta"],
            profile=True,
        )
        sa_energies.extend(energies)
        sa_profiles.append(data)
        sa_total_wall += wall
        e_min = min(energies)
        e_max = max(energies)
        print(
            f"done ({wall:.2f}s)  "
            f"E: [{e_min:.0f}, {e_max:.0f}]"
        )

    # ── Summary ──
    print(f"\n  Gibbs total: {gibbs_total_wall:.2f}s  "
          f"({gibbs_total_wall / num_models:.2f}s/model)")
    print(f"  SA total:    {sa_total_wall:.2f}s  "
          f"({sa_total_wall / num_models:.2f}s/model)")

    # Build per-model table rows
    table_rows = []
    for i, (seed, _, _) in enumerate(models):
        table_rows.append({
            "model_id": seed,
            "gibbs_energy": gibbs_energies[i],
            "sa_energy": sa_energies[i],
        })

    # ── Aggregate flamegraphs ──
    print("\nBuilding aggregate flamegraphs...")
    g_agg = build_aggregate_profile(gibbs_profiles, "gibbs")
    s_agg = build_aggregate_profile(sa_profiles, "sa")

    os.makedirs(output_dir, exist_ok=True)

    generate_flamegraph_html(
        g_agg, clock_khz, "gibbs",
        params["num_sweeps"], gpu_name,
        os.path.join(output_dir, "flamegraph_gibbs_aggregate.html"),
    )
    generate_flamegraph_html(
        s_agg, clock_khz, "sa",
        params["num_sweeps"], gpu_name,
        os.path.join(output_dir, "flamegraph_sa_aggregate.html"),
    )

    # ── Plots ──
    print("\nGenerating plots...")
    plot_comparison_table(
        table_rows,
        os.path.join(output_dir, "comparison_table.png"),
    )
    plot_energy_comparison(
        table_rows,
        os.path.join(output_dir, "energy_comparison.png"),
    )

    if dump_json:
        json_path = os.path.join(output_dir, "test1_results.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "params": params,
                    "rows": table_rows,
                    "gibbs_total_wall": gibbs_total_wall,
                    "sa_total_wall": sa_total_wall,
                },
                f, indent=2, default=str,
            )
        print(f"  JSON saved to {json_path}")

    print(f"\nTest 1 complete. Outputs in {output_dir}/")


def run_test2(target_energy, max_attempts, output_dir, dump_json):
    """Test 2: Mining simulation with pipelined batch dispatch."""
    params = compute_adaptive_params(target_energy)
    print(f"\n{'=' * 60}")
    print(f"TEST 2: Mining Test (max {max_attempts} attempts)")
    print(f"{'=' * 60}")
    print(f"  Target energy: {target_energy}")
    print(f"  Difficulty:    {params['difficulty']:.3f}")
    print(f"  Sweeps:        {params['num_sweeps']}")
    print(f"  Reads:         {params['num_reads']}")
    print(f"  Gibbs batch:   {GIBBS_BATCH_SIZE}")
    print(f"  SA batch:      {SA_BATCH_SIZE}")
    print()

    gpu_name, clock_khz, _ = get_gpu_info()
    graph = DEFAULT_TOPOLOGY.graph
    nodes = sorted(graph.nodes())
    edges = list(graph.edges())

    os.makedirs(output_dir, exist_ok=True)

    def _mine_batched(kernel_name, batch_size, profile):
        """Run batched mining attempts until target hit or exhausted.

        Returns:
            (attempts_list, profiles_list, total_wall)
        """
        run_fn = run_gibbs_batch if kernel_name == "gibbs" else run_sa_batch
        attempts = []
        profiles = []
        total_wall = 0.0
        mined = False

        for start in range(0, max_attempts, batch_size):
            if mined:
                break
            end = min(start + batch_size, max_attempts)
            batch_models = []
            for nonce in range(start, end):
                h, J = generate_ising_model_from_nonce(
                    nonce, nodes, edges,
                )
                batch_models.append((nonce, h, J))

            if kernel_name == "gibbs":
                energies, wall, data = run_fn(
                    batch_models, params["num_reads"],
                    params["num_sweeps"], profile=profile,
                )
            else:
                energies, wall, data = run_fn(
                    batch_models, params["num_reads"],
                    params["num_betas"],
                    params["sweeps_per_beta"],
                    profile=profile,
                )

            total_wall += wall
            if data is not None:
                profiles.append(data)

            for i, e in enumerate(energies):
                nonce = start + i
                attempts.append({
                    "attempt": nonce,
                    "nonce": nonce,
                    "best_energy": e,
                    "wall_time": wall / len(batch_models),
                })
                if e <= target_energy:
                    print(
                        f"    MINED at attempt {nonce}! "
                        f"E={e:.0f}"
                    )
                    mined = True
                    break

            best_in_batch = min(energies)
            print(
                f"    Batch [{start}-{start + len(batch_models) - 1}]: "
                f"best={best_in_batch:.0f} ({wall:.2f}s)"
            )

        return attempts, profiles, total_wall

    # Instrumented runs
    print("  Running GIBBS with profiling...")
    gibbs_attempts, gibbs_profiles, gibbs_inst_time = (
        _mine_batched("gibbs", GIBBS_BATCH_SIZE, profile=True)
    )

    print("\n  Running SA with profiling...")
    sa_attempts, sa_profiles, sa_inst_time = (
        _mine_batched("sa", SA_BATCH_SIZE, profile=True)
    )

    # Uninstrumented runs for overhead comparison
    print("\n  Running uninstrumented for overhead measurement...")
    num_gibbs = len(gibbs_attempts)
    num_sa = len(sa_attempts)

    # Gibbs uninstrumented (same nonce count)
    gibbs_uninst_models = []
    for nonce in range(num_gibbs):
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        gibbs_uninst_models.append((nonce, h, J))

    gibbs_uninst_time = 0.0
    for start in range(0, num_gibbs, GIBBS_BATCH_SIZE):
        batch = gibbs_uninst_models[start:start + GIBBS_BATCH_SIZE]
        _, wall, _ = run_gibbs_batch(
            batch, params["num_reads"], params["num_sweeps"],
            profile=False,
        )
        gibbs_uninst_time += wall

    # SA uninstrumented (same nonce count)
    sa_uninst_models = []
    for nonce in range(num_sa):
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        sa_uninst_models.append((nonce, h, J))

    sa_uninst_time = 0.0
    for start in range(0, num_sa, SA_BATCH_SIZE):
        batch = sa_uninst_models[start:start + SA_BATCH_SIZE]
        _, wall, _ = run_sa_batch(
            batch, params["num_reads"],
            params["num_betas"], params["sweeps_per_beta"],
            profile=False,
        )
        sa_uninst_time += wall

    # Build aggregate flamegraphs
    print("\nBuilding aggregate flamegraphs...")
    if gibbs_profiles:
        g_agg = build_aggregate_profile(gibbs_profiles, "gibbs")
        generate_flamegraph_html(
            g_agg, clock_khz, "gibbs",
            params["num_sweeps"], gpu_name,
            os.path.join(output_dir, "flamegraph_gibbs_mining.html"),
        )
    if sa_profiles:
        s_agg = build_aggregate_profile(sa_profiles, "sa")
        generate_flamegraph_html(
            s_agg, clock_khz, "sa",
            params["num_sweeps"], gpu_name,
            os.path.join(output_dir, "flamegraph_sa_mining.html"),
        )

    # Plots
    print("\nGenerating plots...")
    plot_energy_timeline(
        gibbs_attempts, sa_attempts, target_energy,
        os.path.join(output_dir, "energy_timeline.png"),
    )

    plot_overhead_comparison(
        gibbs_inst_time, gibbs_uninst_time,
        sa_inst_time, sa_uninst_time,
        os.path.join(output_dir, "overhead_comparison.png"),
    )

    gibbs_summary = {
        "attempts": num_gibbs,
        "total_time": gibbs_inst_time,
        "overhead_pct": (
            (gibbs_inst_time - gibbs_uninst_time)
            / gibbs_uninst_time * 100
            if gibbs_uninst_time > 0 else 0
        ),
        "final_energy": (
            gibbs_attempts[-1]["best_energy"]
            if gibbs_attempts else float("nan")
        ),
        "mined": (
            gibbs_attempts[-1]["best_energy"] <= target_energy
            if gibbs_attempts else False
        ),
    }
    sa_summary = {
        "attempts": num_sa,
        "total_time": sa_inst_time,
        "overhead_pct": (
            (sa_inst_time - sa_uninst_time)
            / sa_uninst_time * 100
            if sa_uninst_time > 0 else 0
        ),
        "final_energy": (
            sa_attempts[-1]["best_energy"]
            if sa_attempts else float("nan")
        ),
        "mined": (
            sa_attempts[-1]["best_energy"] <= target_energy
            if sa_attempts else False
        ),
    }
    plot_mining_summary(
        gibbs_summary, sa_summary,
        os.path.join(output_dir, "mining_summary.png"),
    )

    if dump_json:
        json_path = os.path.join(output_dir, "test2_results.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "params": params,
                    "target_energy": target_energy,
                    "gibbs_attempts": gibbs_attempts,
                    "sa_attempts": sa_attempts,
                    "gibbs_summary": gibbs_summary,
                    "sa_summary": sa_summary,
                },
                f, indent=2, default=str,
            )
        print(f"  JSON saved to {json_path}")

    print(f"\nTest 2 complete. Outputs in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative CUDA kernel profiling benchmark",
    )
    parser.add_argument(
        "test", choices=["test1", "test2", "both"],
        help="Which test to run",
    )
    parser.add_argument(
        "--target-energy", type=float, default=-14900.0,
        help="Energy target (default: -14900.0)",
    )
    parser.add_argument(
        "--num-models", type=int, default=50,
        help="Number of models for test1 (default: 50)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Starting nonce seed (default: 0)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=500,
        help="Max mining attempts for test2 (default: 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/compare",
        help="Output directory (default: benchmarks/compare)",
    )
    parser.add_argument(
        "--gpu", type=int, default=1,
        help="GPU device (default: 1)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also dump raw JSON data",
    )
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gpu_name, clock_khz, clock_mhz = get_gpu_info()
    print(f"GPU: {gpu_name}, SM clock: {clock_mhz:.0f} MHz")

    if args.test in ("test1", "both"):
        run_test1(
            args.num_models, args.target_energy,
            args.output_dir, args.json,
        )
    if args.test in ("test2", "both"):
        run_test2(
            args.target_energy, args.max_attempts,
            args.output_dir, args.json,
        )


if __name__ == "__main__":
    main()
