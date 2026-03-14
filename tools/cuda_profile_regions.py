#!/usr/bin/env python3
"""CUDA kernel region profiling via clock64() instrumentation.

Runs SA or Gibbs kernel with PROFILE_REGIONS enabled, reads back
per-region cycle counters, and outputs a hierarchical breakdown
showing where GPU time is spent inside the kernel.

Usage:
    python tools/cuda_profile_regions.py [--kernel gibbs|sa] \
        [--sweeps N] [--reads N] [--plot] [--flame] \
        [--flame-html]
"""

import argparse
import colorsys
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import cupy as cp
import numpy as np


# Region names for each kernel
SA_REGION_NAMES = [
    "SA_TOTAL",         # 0
    "BETA_OVERHEAD",    # 1
    "SWEEP_TOTAL",      # 2
    "VAR_SCAN",         # 3 (derived: 4+5+6)
    "THRESHOLD_SKIP",   # 4
    "ACCEPT_DECIDE",    # 5
    "FLIP_TOTAL",       # 6
    "NEIGHBOR_LOOP",    # 7
    "FLIP_COUNT",       # 8 (count, not cycles)
    "SKIP_COUNT",       # 9 (count, not cycles)
]

GIBBS_REGION_NAMES = [
    "ANNEALING_TOTAL",  # 0
    "BETA_ITER_TOTAL",  # 1
    "SWEEP_ITER_TOTAL", # 2
    "COLOR_ITER_TOTAL", # 3
    "COLOR_SETUP",      # 4
    "NODE_LOOP_TOTAL",  # 5
    "FIELD_COMPUTE",    # 6
    "SPIN_UPDATE",      # 7
    "SYNC_COLOR",       # 8
    "BETA_COUNT",       # 9  (count)
    "SWEEP_COUNT",      # 10 (count)
    "COLOR_COUNT",      # 11 (count)
]

# Region index -> (start_line, end_line) in .cu source
GIBBS_SOURCE_MAP = {
    0: (319, 425),  # ANNEALING_TOTAL
    1: (322, 420),  # BETA_ITER_TOTAL
    2: (328, 413),  # SWEEP_ITER_TOTAL
    3: (332, 407),  # COLOR_ITER_TOTAL
    4: (335, 344),  # COLOR_SETUP
    5: (347, 394),  # NODE_LOOP_TOTAL
    6: (359, 369),  # FIELD_COMPUTE
    7: (373, 389),  # SPIN_UPDATE
    8: (397, 402),  # SYNC_COLOR
}

SA_SOURCE_MAP = {
    0: (442, 503),  # SA_TOTAL
    1: (444, 447),  # BETA_OVERHEAD
    2: (450, 500),  # SWEEP_TOTAL
    4: (455, 458),  # THRESHOLD_SKIP
    5: (461, 472),  # ACCEPT_DECIDE
    6: (475, 496),  # FLIP_TOTAL
    7: (484, 491),  # NEIGHBOR_LOOP
}

# Project root for resolving .cu file paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_gpu_info():
    """Query GPU name and SM clock rate."""
    dev = cp.cuda.Device()
    attrs = dev.attributes
    clock_khz = attrs['ClockRate']
    clock_mhz = clock_khz / 1000.0
    name = cp.cuda.runtime.getDeviceProperties(dev.id)['name']
    if isinstance(name, bytes):
        name = name.decode()
    return name, clock_khz, clock_mhz


def cycles_to_time(cycles, clock_khz):
    """Convert cycles to seconds."""
    return cycles / (clock_khz * 1000.0)


def fmt_time(seconds):
    """Format seconds as human-readable time."""
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 0.001:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds * 1e6:.2f} us"


def fmt_cycles(cycles):
    """Format cycle count with commas."""
    return f"{int(cycles):,}"


def build_ising_problem():
    """Build a standard Advantage topology Ising problem."""
    from dwave_topologies import DEFAULT_TOPOLOGY

    graph = DEFAULT_TOPOLOGY.graph
    nodes = sorted(graph.nodes())
    edges = list(graph.edges())

    rng = np.random.default_rng(42)
    h = {n: float(rng.choice([-1, 0, 1])) for n in nodes}
    J = {}
    for u, v in edges:
        J[(u, v)] = float(rng.choice([-1, 1]))

    return h, J


def profile_gibbs(num_reads, num_sweeps):
    """Run Gibbs kernel with profiling and return data."""
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    h, J = build_ising_problem()
    sampler = CudaGibbsSampler(profile=True)
    sampler.sample_ising(
        [h], [J],
        num_reads=num_reads,
        num_sweeps=num_sweeps,
    )
    return sampler.get_profile_data()


def profile_sa(num_reads, num_sweeps):
    """Run SA kernel with profiling and return data.

    Uses CudaKernelRealSA persistent kernel: enqueue one job,
    poll for result, then read profiling counters.
    """
    import time
    from GPU.cuda_kernel import CudaKernelRealSA
    from shared.beta_schedule import _default_ising_beta_range

    h_dict, J_dict = build_ising_problem()

    num_betas = num_sweeps
    num_sweeps_per_beta = 1
    beta_range = _default_ising_beta_range(h_dict, J_dict)

    kernel = CudaKernelRealSA(
        profile=True, verbose=False,
    )

    kernel.enqueue_job(
        job_id=0,
        h=h_dict,
        J=J_dict,
        num_reads=min(num_reads, 256),
        num_betas=num_betas,
        num_sweeps_per_beta=num_sweeps_per_beta,
        beta_range=beta_range,
    )
    kernel.signal_batch_ready()

    # Poll for result
    deadline = time.time() + 30.0
    result = None
    while time.time() < deadline:
        result = kernel.try_dequeue_result()
        if result is not None:
            break
        time.sleep(0.01)

    # Stop the persistent kernel BEFORE reading profile data.
    # get_profile_data() calls stream.synchronize(), which
    # blocks forever if the kernel is still running.
    kernel.stop_immediate()
    data = kernel.get_profile_data()

    if result is None:
        print(
            "WARNING: SA persistent kernel did not produce "
            "a result within 30s. Profile data may be empty."
        )

    return data


def print_sa_profile(data, clock_khz, num_reads, num_sweeps):
    """Print SA kernel profile breakdown."""
    # Filter to threads that actually did work (SA_TOTAL > 0)
    active = data[data[:, 0] > 0]
    if len(active) == 0:
        print("No active threads found in profile data.")
        return

    n_active = len(active)
    avg = active.mean(axis=0)
    std = active.std(axis=0)
    mn = active.min(axis=0)
    mx = active.max(axis=0)

    # Derive VAR_SCAN = THRESHOLD_SKIP + ACCEPT_DECIDE + FLIP_TOTAL
    avg[3] = avg[4] + avg[5] + avg[6]

    total = avg[0]
    num_betas = num_sweeps  # num_sweeps = num_betas in default config

    print(f"\n{'=' * 60}")
    print(f"SA Kernel Profile ({n_active} active threads)")
    print(f"{'=' * 60}")

    cycle_regions = [
        (0, "SA_TOTAL", None),
        (1, "  BETA_OVERHEAD", 0),
        (2, "  SWEEP_TOTAL", 0),
        (3, "  VAR_SCAN (derived)", 0),
        (4, "    THRESHOLD_SKIP", 0),
        (5, "    ACCEPT_DECIDE", 0),
        (6, "    FLIP_TOTAL", 0),
        (7, "      NEIGHBOR_LOOP", 0),
    ]

    print(f"\n{'Region':<28} {'Avg Cycles':>14} "
          f"{'Avg Time':>10} {'%':>7} {'StdDev':>14}")
    print("-" * 75)
    for idx, name, parent in cycle_regions:
        c = avg[idx]
        t = cycles_to_time(c, clock_khz)
        pct = (c / total * 100) if total > 0 else 0
        sd = std[idx] if idx != 3 else 0
        print(f"{name:<28} {fmt_cycles(c):>14} "
              f"{fmt_time(t):>10} {pct:>6.1f}% "
              f"{fmt_cycles(sd):>14}")

    # Count-based metrics
    flip_count = avg[8]
    skip_count = avg[9]
    print(f"\n{'Derived Metrics':}")
    print(f"  Flip count (avg):  {flip_count:,.0f}")
    print(f"  Skip count (avg):  {skip_count:,.0f}")
    total_vars = flip_count + skip_count
    if total_vars > 0:
        print(f"  Flip rate:         "
              f"{flip_count / total_vars * 100:.1f}%")
        print(f"  Skip rate:         "
              f"{skip_count / total_vars * 100:.1f}%")
    if flip_count > 0:
        avg_flip_cycles = avg[6] / flip_count
        print(f"  Avg cycles/flip:   {avg_flip_cycles:,.0f}")
        avg_neighbor_cycles = avg[7] / flip_count
        print(f"  Avg cycles/neighbor_loop: "
              f"{avg_neighbor_cycles:,.0f}")

    # Load imbalance
    print(f"\n{'Load Imbalance (SA_TOTAL)':}")
    print(f"  Min: {fmt_cycles(mn[0]):>14}  "
          f"({fmt_time(cycles_to_time(mn[0], clock_khz))})")
    print(f"  Max: {fmt_cycles(mx[0]):>14}  "
          f"({fmt_time(cycles_to_time(mx[0], clock_khz))})")
    print(f"  Std: {fmt_cycles(std[0]):>14}  "
          f"({fmt_time(cycles_to_time(std[0], clock_khz))})")
    if avg[0] > 0:
        cv = std[0] / avg[0] * 100
        print(f"  CV:  {cv:.1f}%")


def print_gibbs_profile(data, clock_khz, num_reads, num_sweeps):
    """Print Gibbs kernel profile breakdown."""
    # Filter to work units that completed (ANNEALING_TOTAL > 0)
    active = data[data[:, 0] > 0]
    if len(active) == 0:
        print("No active work units found in profile data.")
        return

    n_active = len(active)
    avg = active.mean(axis=0)
    std = active.std(axis=0)
    mn = active.min(axis=0)
    mx = active.max(axis=0)

    total = avg[0]

    print(f"\n{'=' * 60}")
    print(f"Gibbs Kernel Profile ({n_active} work units)")
    print(f"{'=' * 60}")

    cycle_regions = [
        (0, "ANNEALING_TOTAL", None),
        (1, "  BETA_ITER_TOTAL", 0),
        (2, "    SWEEP_ITER_TOTAL", 0),
        (3, "      COLOR_ITER_TOTAL", 0),
        (4, "        COLOR_SETUP", 0),
        (5, "        NODE_LOOP_TOTAL", 0),
        (6, "          FIELD_COMPUTE", 0),
        (7, "          SPIN_UPDATE", 0),
        (8, "        SYNC_COLOR", 0),
    ]

    print(f"\n{'Region':<28} {'Avg Cycles':>14} "
          f"{'Avg Time':>10} {'%':>7} {'StdDev':>14}")
    print("-" * 75)
    for idx, name, parent in cycle_regions:
        c = avg[idx]
        t = cycles_to_time(c, clock_khz)
        pct = (c / total * 100) if total > 0 else 0
        sd = std[idx]
        print(f"{name:<28} {fmt_cycles(c):>14} "
              f"{fmt_time(t):>10} {pct:>6.1f}% "
              f"{fmt_cycles(sd):>14}")

    # Iteration counts
    beta_count = avg[9]
    sweep_count = avg[10]
    color_count = avg[11]
    print(f"\n{'Iteration Counts (avg)':}")
    print(f"  Beta iterations:   {beta_count:,.0f}")
    print(f"  Sweep iterations:  {sweep_count:,.0f}")
    print(f"  Color iterations:  {color_count:,.0f}")

    # Per-sweep breakdown
    if sweep_count > 0:
        print(f"\n{'Per-Sweep Breakdown':}")
        per_sweep_regions = [
            (4, "COLOR_SETUP"),
            (6, "FIELD_COMPUTE"),
            (7, "SPIN_UPDATE"),
            (8, "SYNC_COLOR"),
        ]
        sweep_total_cycles = (
            avg[2] / sweep_count if sweep_count > 0 else 0
        )
        print(f"  {'Region':<20} {'Cycles/sweep':>14} "
              f"{'Time/sweep':>10} {'%':>7}")
        print(f"  {'-' * 55}")
        for idx, name in per_sweep_regions:
            c = avg[idx] / sweep_count
            t = cycles_to_time(c, clock_khz)
            pct = (c / sweep_total_cycles * 100
                   if sweep_total_cycles > 0 else 0)
            print(f"  {name:<20} {fmt_cycles(c):>14} "
                  f"{fmt_time(t):>10} {pct:>6.1f}%")
        print(
            f"  {'TOTAL':<20} "
            f"{fmt_cycles(sweep_total_cycles):>14} "
            f"{fmt_time(cycles_to_time(sweep_total_cycles, clock_khz)):>10}"
        )

    # Consistency checks
    print(f"\n{'Consistency Checks':}")
    overhead = avg[0] - avg[1]
    print(f"  TOTAL - BETA_ITER = "
          f"{fmt_cycles(overhead)} "
          f"({overhead / total * 100:.2f}% overhead)")
    node_parts = avg[6] + avg[7]
    node_total = avg[5]
    if node_total > 0:
        print(f"  FIELD+SPIN / NODE_LOOP = "
              f"{node_parts / node_total * 100:.1f}% "
              f"(rest = load imbalance)")

    # Load imbalance
    print(f"\n{'Load Imbalance (ANNEALING_TOTAL)':}")
    print(f"  Min: {fmt_cycles(mn[0]):>14}  "
          f"({fmt_time(cycles_to_time(mn[0], clock_khz))})")
    print(f"  Max: {fmt_cycles(mx[0]):>14}  "
          f"({fmt_time(cycles_to_time(mx[0], clock_khz))})")
    print(f"  Std: {fmt_cycles(std[0]):>14}  "
          f"({fmt_time(cycles_to_time(std[0], clock_khz))})")
    if avg[0] > 0:
        cv = std[0] / avg[0] * 100
        print(f"  CV:  {cv:.1f}%")


def plot_profile(data, region_names, clock_khz, kernel_name,
                 output_path):
    """Generate profile visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    active = data[data[:, 0] > 0]
    if len(active) == 0:
        return

    avg = active.mean(axis=0)

    if kernel_name == "sa":
        # Per-sweep pie chart
        labels = ["THRESHOLD_SKIP", "ACCEPT_DECIDE",
                   "FLIP_TOTAL", "Other"]
        sizes = [avg[4], avg[5], avg[6],
                 max(0, avg[2] - avg[4] - avg[5] - avg[6])]
    else:
        labels = ["COLOR_SETUP", "FIELD_COMPUTE",
                   "SPIN_UPDATE", "SYNC_COLOR", "Other"]
        sizes = [avg[4], avg[6], avg[7], avg[8],
                 max(0, avg[3] - avg[4] - avg[6]
                     - avg[7] - avg[8])]

    # Filter zero-size slices
    nonzero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not nonzero:
        return
    labels, sizes = zip(*nonzero)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%')
    axes[0].set_title(
        f"{kernel_name.upper()} Per-Sweep Breakdown"
    )

    # Histogram of total cycles across work units
    totals = active[:, 0]
    times_ms = totals / (clock_khz * 1.0)
    axes[1].hist(times_ms, bins=min(50, len(active)),
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Total time (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"{kernel_name.upper()} Total Time Distribution"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def get_topology_stats():
    """Query topology for node/edge/degree stats.

    Returns:
        Tuple of (num_nodes, num_edges, avg_degree,
                  nodes_per_color, num_colors).
    """
    from dwave_topologies import DEFAULT_TOPOLOGY

    graph = DEFAULT_TOPOLOGY.graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = 2.0 * num_edges / num_nodes
    num_colors = 4
    nodes_per_color = math.ceil(num_nodes / num_colors)
    return (num_nodes, num_edges, avg_degree,
            nodes_per_color, num_colors)


def print_precise_ticks(data, clock_khz, num_sweeps,
                        kernel_name, gpu_name):
    """Print per-operation tick breakdown from profiling data.

    Normalizes aggregate cycle counters by topology constants
    and iteration counts to produce per-node, per-neighbor,
    and per-global-load cycle estimates.
    """
    active = data[data[:, 0] > 0]
    if len(active) == 0:
        print("No active data for precise ticks.")
        return

    avg = active.mean(axis=0)
    clock_mhz = clock_khz / 1000.0
    topo = get_topology_stats()
    num_nodes, num_edges, avg_deg, npc, num_colors = topo
    block_dim = 256  # standard CUDA block size

    print(f"\n{'=' * 60}")
    print(f"Precise Per-Operation Ticks ({kernel_name.upper()})")
    print(f"{'=' * 60}")
    print(f"GPU: {gpu_name} @ {clock_mhz:.0f} MHz")
    print(f"Topology: {num_nodes} nodes, {num_edges} edges, "
          f"avg degree {avg_deg:.1f}")
    print(f"Colors: {num_colors}, "
          f"nodes/color: ~{npc}")

    if kernel_name == "gibbs":
        _print_gibbs_ticks(
            avg, clock_khz, num_sweeps,
            num_nodes, avg_deg, npc, num_colors, block_dim,
        )
    else:
        _print_sa_ticks(
            avg, clock_khz, num_sweeps,
            num_nodes, avg_deg,
        )


def _print_gibbs_ticks(avg, clock_khz, num_sweeps,
                       num_nodes, avg_deg, npc,
                       num_colors, block_dim):
    """Print Gibbs-specific per-operation ticks."""
    # Thread 0 processes ceil(npc / blockDim) nodes per color
    nodes_per_color_t0 = math.ceil(npc / block_dim)
    total_field_calls = (
        num_colors * nodes_per_color_t0 * num_sweeps
    )
    # Fixed loads: h_bias, row_ptr[node], row_ptr[node+1]
    # Per-neighbor: col_idx + coupling = 2
    loads_per_node = 3 + 2 * avg_deg

    sweep_count = avg[10] if avg[10] > 0 else num_sweeps
    beta_count = avg[9] if avg[9] > 0 else 1

    field_total = avg[6]
    spin_total = avg[7]
    sync_total = avg[8]
    sweep_total = avg[2]

    cyc_per_field = field_total / total_field_calls
    cyc_per_neighbor = cyc_per_field / avg_deg
    cyc_per_load = cyc_per_field / loads_per_node
    cyc_per_spin = spin_total / total_field_calls
    cyc_per_barrier = sync_total / (num_colors * sweep_count)
    cyc_per_sweep = sweep_total / sweep_count
    beta_overhead = (avg[1] - avg[2]) / beta_count

    print(f"\nThread 0 measurements "
          f"(representative due to __syncthreads):")
    print(f"  nodes_per_color_t0 = ceil({npc}/{block_dim})"
          f" = {nodes_per_color_t0}")
    print(f"  total field calls  = {num_colors} x "
          f"{nodes_per_color_t0} x {num_sweeps}"
          f" = {total_field_calls:,}")

    print(f"\n  Per field computation:")
    print(f"    FIELD_COMPUTE / calls = "
          f"{fmt_cycles(field_total)} / {total_field_calls:,}")
    print(f"    = {fmt_cycles(cyc_per_field)} cycles/node"
          f"  ({fmt_time(cycles_to_time(cyc_per_field, clock_khz))})")
    print(f"    Per neighbor iteration: "
          f"{fmt_cycles(cyc_per_field)} / {avg_deg:.1f}"
          f" = {fmt_cycles(cyc_per_neighbor)} cycles/neighbor")
    print(f"    Per global mem load:    "
          f"{fmt_cycles(cyc_per_field)} / {loads_per_node:.1f}"
          f" = {fmt_cycles(cyc_per_load)} cycles/load")

    print(f"\n  Per spin update:")
    print(f"    SPIN_UPDATE / calls = "
          f"{fmt_cycles(spin_total)} / {total_field_calls:,}"
          f" = {fmt_cycles(cyc_per_spin)} cycles/node")

    print(f"\n  Per color barrier:")
    print(f"    SYNC_COLOR / (colors x sweeps) = "
          f"{fmt_cycles(sync_total)} / "
          f"{int(num_colors * sweep_count):,}"
          f" = {fmt_cycles(cyc_per_barrier)} cycles/barrier")

    print(f"\n  Per sweep:")
    print(f"    SWEEP_TOTAL / sweeps = "
          f"{fmt_cycles(cyc_per_sweep)} cycles"
          f"  ({fmt_time(cycles_to_time(cyc_per_sweep, clock_khz))})")

    print(f"\n  Per beta step overhead:")
    print(f"    (BETA_ITER - SWEEP_ITER) / betas = "
          f"{fmt_cycles(beta_overhead)} cycles"
          f"  ({fmt_time(cycles_to_time(beta_overhead, clock_khz))})")

    # Cross-checks
    reconstructed = cyc_per_field * total_field_calls
    print(f"\n  Cross-checks:")
    print(f"    cycles/field x calls = "
          f"{fmt_cycles(reconstructed)}")
    print(f"    FIELD_COMPUTE        = "
          f"{fmt_cycles(field_total)}")
    err = abs(reconstructed - field_total) / field_total * 100
    print(f"    error: {err:.2f}%")

    neighbor_recon = cyc_per_neighbor * avg_deg
    print(f"    cycles/neighbor x degree = "
          f"{fmt_cycles(neighbor_recon)}")
    print(f"    cycles/field             = "
          f"{fmt_cycles(cyc_per_field)}")
    err2 = (
        abs(neighbor_recon - cyc_per_field)
        / cyc_per_field * 100
    )
    print(f"    error: {err2:.2f}%")

    load_recon = cyc_per_load * loads_per_node
    print(f"    cycles/load x loads      = "
          f"{fmt_cycles(load_recon)}")
    print(f"    cycles/field             = "
          f"{fmt_cycles(cyc_per_field)}")
    err3 = (
        abs(load_recon - cyc_per_field)
        / cyc_per_field * 100
    )
    print(f"    error: {err3:.2f}%")


def _print_sa_ticks(avg, clock_khz, num_sweeps,
                    num_nodes, avg_deg):
    """Print SA-specific per-operation ticks."""
    num_betas = num_sweeps  # 1 sweep per beta in default config
    total_vars = num_nodes * num_betas

    flip_count = avg[8]
    skip_count = avg[9]
    flip_total = avg[6]
    neighbor_total = avg[7]
    sweep_total = avg[2]

    print(f"\n  Total variable scans: "
          f"{num_nodes} x {num_betas} = {total_vars:,}")

    if flip_count > 0:
        cyc_per_flip = flip_total / flip_count
        cyc_per_neighbor = neighbor_total / flip_count
        print(f"\n  Per accepted flip:")
        print(f"    FLIP_TOTAL / flips = "
              f"{fmt_cycles(flip_total)} / {flip_count:,.0f}"
              f" = {fmt_cycles(cyc_per_flip)} cycles/flip")
        print(f"    NEIGHBOR_LOOP / flips = "
              f"{fmt_cycles(neighbor_total)} / {flip_count:,.0f}"
              f" = {fmt_cycles(cyc_per_neighbor)} "
              f"cycles/neighbor_update")
        if avg_deg > 0:
            per_neighbor = cyc_per_neighbor / avg_deg
            print(f"    Per individual neighbor: "
                  f"{fmt_cycles(per_neighbor)} cycles")

    cyc_per_sweep = sweep_total / num_betas
    print(f"\n  Per sweep:")
    print(f"    SWEEP_TOTAL / sweeps = "
          f"{fmt_cycles(cyc_per_sweep)} cycles"
          f"  ({fmt_time(cycles_to_time(cyc_per_sweep, clock_khz))})")


def generate_flamegraph(data, clock_khz, kernel_name,
                        output_path):
    """Generate flamegraph SVG from profiling data.

    Draws a flame chart with horizontal bars proportional to
    cycle counts, stacked by nesting depth. Saved as SVG for
    lossless scaling.

    Args:
        data: Raw profiling array (work_units x regions).
        clock_khz: GPU clock rate in kHz.
        kernel_name: 'gibbs' or 'sa'.
        output_path: Output SVG file path.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available, skipping flamegraph.")
        return

    active = data[data[:, 0] > 0]
    if len(active) == 0:
        print("No active data for flamegraph.")
        return

    avg = active.mean(axis=0)
    total = avg[0]
    if total <= 0:
        return

    if kernel_name == "gibbs":
        rows = _build_gibbs_flame_rows(avg, total)
    else:
        rows = _build_sa_flame_rows(avg, total)

    num_rows = max(r[1] for r in rows) + 1
    fig_height = max(3.0, num_rows * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    _draw_flame_bars(ax, rows, clock_khz)
    _style_flame_axes(
        ax, fig, num_rows, kernel_name, total, clock_khz,
    )

    plt.tight_layout()
    fig.savefig(output_path, format='svg')
    plt.close(fig)
    print(f"\nFlamegraph saved to {output_path}")


def _draw_flame_bars(ax, rows, clock_khz):
    """Draw labeled rectangles for each flame bar."""
    from matplotlib.patches import Rectangle

    bar_height = 0.7
    for bar in rows:
        name, row, x_start, width, color, cycles, pct = bar
        rect = Rectangle(
            (x_start, row), width, bar_height,
            facecolor=color, edgecolor='white',
            linewidth=1.5,
        )
        ax.add_patch(rect)

        label = f"{name}  {fmt_cycles(cycles)}  ({pct:.1f}%)"
        short = f"{name} ({pct:.1f}%)"
        display_label = label if width > 0.25 else short
        if width < 0.05:
            display_label = ""

        ax.text(
            x_start + width / 2, row + bar_height / 2,
            display_label,
            ha='center', va='center',
            fontsize=8 if width > 0.15 else 6,
            fontweight='bold', color='white',
            clip_on=True,
        )


def _style_flame_axes(ax, fig, num_rows, kernel_name,
                      total, clock_khz):
    """Apply dark-theme styling to flamegraph axes."""
    bar_height = 0.7
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.3, num_rows + 0.3)
    ax.set_xlabel("Fraction of total cycles", fontsize=11)
    ax.set_ylabel("Nesting depth", fontsize=11)
    ax.set_yticks(
        [i + bar_height / 2 for i in range(num_rows)]
    )
    ax.set_yticklabels(
        [f"depth {i}" for i in range(num_rows)]
    )
    ax.set_title(
        f"{kernel_name.upper()} Kernel Flamegraph "
        f"({fmt_cycles(total)} total cycles, "
        f"{fmt_time(cycles_to_time(total, clock_khz))})",
        fontsize=13, fontweight='bold',
    )
    bg = '#2c3e50'
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.tick_params(colors='white')
    for attr in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        attr.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_color('white')


def _build_gibbs_flame_rows(avg, total):
    """Build flame bar specs for Gibbs kernel.

    Returns list of (name, row, x_start, width, color, cycles,
    pct).
    """
    c_compute = '#e74c3c'
    c_compute2 = '#e67e22'
    c_sync = '#3498db'
    c_setup = '#95a5a6'
    c_outer = '#9b59b6'
    c_loop = '#2ecc71'

    def frac(c):
        return c / total if total > 0 else 0

    def pct(c):
        return c / total * 100 if total > 0 else 0

    bars = []
    # Row 0: ANNEALING_TOTAL
    bars.append((
        "ANNEALING_TOTAL", 0, 0.0, 1.0,
        c_outer, avg[0], 100.0,
    ))
    # Row 1: BETA_ITER_TOTAL
    w1 = frac(avg[1])
    bars.append((
        "BETA_ITER_TOTAL", 1, 0.0, w1,
        c_outer, avg[1], pct(avg[1]),
    ))
    # Row 2: SWEEP_ITER_TOTAL
    w2 = frac(avg[2])
    bars.append((
        "SWEEP_ITER_TOTAL", 2, 0.0, w2,
        c_loop, avg[2], pct(avg[2]),
    ))
    # Row 3: COLOR_ITER_TOTAL
    w3 = frac(avg[3])
    bars.append((
        "COLOR_ITER_TOTAL", 3, 0.0, w3,
        c_loop, avg[3], pct(avg[3]),
    ))
    # Row 4: COLOR_SETUP | NODE_LOOP | SYNC_COLOR
    x = 0.0
    w_setup = frac(avg[4])
    bars.append((
        "SETUP", 4, x, w_setup,
        c_setup, avg[4], pct(avg[4]),
    ))
    x += w_setup
    w_node = frac(avg[5])
    bars.append((
        "NODE_LOOP", 4, x, w_node,
        c_compute2, avg[5], pct(avg[5]),
    ))
    x += w_node
    w_sync = frac(avg[8])
    bars.append((
        "SYNC", 4, x, w_sync,
        c_sync, avg[8], pct(avg[8]),
    ))
    # Row 5: FIELD_COMPUTE | SPIN_UPDATE (within NODE_LOOP)
    x_node_start = frac(avg[4])  # starts after SETUP
    w_field = frac(avg[6])
    bars.append((
        "FIELD_COMPUTE", 5, x_node_start, w_field,
        c_compute, avg[6], pct(avg[6]),
    ))
    w_spin = frac(avg[7])
    bars.append((
        "SPIN_UPDATE", 5, x_node_start + w_field, w_spin,
        c_compute2, avg[7], pct(avg[7]),
    ))
    return bars


def _build_sa_flame_rows(avg, total):
    """Build flame bar specs for SA kernel.

    Returns list of (name, row, x_start, width, color, cycles,
    pct).
    """
    c_compute = '#e74c3c'
    c_compute2 = '#e67e22'
    c_skip = '#95a5a6'
    c_outer = '#9b59b6'
    c_loop = '#2ecc71'

    def frac(c):
        return c / total if total > 0 else 0

    def pct(c):
        return c / total * 100 if total > 0 else 0

    bars = []
    # Row 0: SA_TOTAL
    bars.append((
        "SA_TOTAL", 0, 0.0, 1.0,
        c_outer, avg[0], 100.0,
    ))
    # Row 1: SWEEP_TOTAL
    w_sweep = frac(avg[2])
    bars.append((
        "SWEEP_TOTAL", 1, 0.0, w_sweep,
        c_loop, avg[2], pct(avg[2]),
    ))
    # Row 2: THRESHOLD_SKIP | ACCEPT_DECIDE | FLIP_TOTAL |
    #         overhead
    x = 0.0
    w_skip = frac(avg[4])
    bars.append((
        "SKIP", 2, x, w_skip,
        c_skip, avg[4], pct(avg[4]),
    ))
    x += w_skip
    w_accept = frac(avg[5])
    bars.append((
        "ACCEPT", 2, x, w_accept,
        c_compute2, avg[5], pct(avg[5]),
    ))
    x += w_accept
    w_flip = frac(avg[6])
    bars.append((
        "FLIP_TOTAL", 2, x, w_flip,
        c_compute, avg[6], pct(avg[6]),
    ))
    overhead = max(0, avg[2] - avg[4] - avg[5] - avg[6])
    if overhead > 0:
        x += w_flip
        w_oh = frac(overhead)
        bars.append((
            "overhead", 2, x, w_oh,
            c_skip, overhead, pct(overhead),
        ))
    # Row 3: NEIGHBOR_LOOP (within FLIP_TOTAL)
    flip_start = frac(avg[4]) + frac(avg[5])
    w_neighbor = frac(avg[7])
    bars.append((
        "NEIGHBOR_LOOP", 3, flip_start, w_neighbor,
        c_compute, avg[7], pct(avg[7]),
    ))
    return bars


# ── Interactive HTML flamegraph ─────────────────────────────


def _jitter_color(name):
    """Generate a warm Gregg-style color from a bar name.

    Hue 0-60° (red→yellow), saturation 70-100%,
    lightness 50-70%. Deterministic hash jitter gives
    visual variety without semantic overload.
    """
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h % 60) / 360.0          # 0°–60°
    sat = 0.7 + (h >> 8 & 0xFF) / 850.0   # 0.70–1.0
    lit = 0.5 + (h >> 16 & 0xFF) / 1275.0  # 0.50–0.70
    r, g, b = colorsys.hls_to_rgb(hue, lit, sat)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _read_cu_source(kernel_name):
    """Read CUDA source file for the given kernel.

    Returns source text or empty string if file not found.
    """
    filename = f"cuda_{kernel_name}.cu"
    path = _PROJECT_ROOT / "GPU" / filename
    if not path.exists():
        print(f"WARNING: {path} not found, "
              "source panel will be empty.")
        return ""
    return path.read_text(encoding="utf-8")


def _highlight_cuda_source(source):
    """Apply regex-based syntax highlighting to CUDA source.

    Returns list of HTML-escaped, highlighted lines.
    """
    keywords = (
        r'\b(if|else|for|while|return|break|continue|'
        r'int|float|bool|void|const|char|signed|'
        r'unsigned|long|short|double|int8_t|'
        r'__global__|__device__|__shared__|'
        r'__syncthreads|__ldg|__expf|__int2float_rn|'
        r'__uint2float_rn|threadIdx|blockIdx|blockDim|'
        r'clock64)\b'
    )
    types_re = (
        r'\b(curandState|dim3|size_t)\b'
    )

    highlighted = []
    for line in source.splitlines():
        # Escape HTML entities first
        esc = (line
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
        # Comments (// ...)
        esc = re.sub(
            r'(//.*)',
            r'<span class="cm">\1</span>', esc,
        )
        # Preprocessor directives
        esc = re.sub(
            r'^(\s*#\w+.*)',
            r'<span class="pp">\1</span>', esc,
        )
        # String literals
        esc = re.sub(
            r'(&quot;[^&]*&quot;|"[^"]*")',
            r'<span class="st">\1</span>', esc,
        )
        # Numbers
        esc = re.sub(
            r'\b(\d+\.?\d*[fFL]?)\b',
            r'<span class="nu">\1</span>', esc,
        )
        # Keywords
        esc = re.sub(
            keywords,
            r'<span class="kw">\1</span>', esc,
        )
        # Types
        esc = re.sub(
            types_re,
            r'<span class="ty">\1</span>', esc,
        )
        highlighted.append(esc)
    return highlighted


def _compute_direct_lines(source_map):
    """Compute direct lines for each region.

    Direct lines = region's line range minus union of all
    child regions' ranges. This lets click-to-highlight
    show only lines belonging directly to a region,
    excluding nested children.

    Returns dict mapping region_idx -> sorted list of ints.
    """
    direct = {}
    for idx, (start, end) in source_map.items():
        own = set(range(start, end + 1))
        # Subtract all children (regions whose range
        # is strictly contained within this one)
        for other_idx, (cs, ce) in source_map.items():
            if other_idx == idx:
                continue
            if cs >= start and ce <= end and (
                cs > start or ce < end
            ):
                own -= set(range(cs, ce + 1))
        direct[idx] = sorted(own)
    return direct


def _build_source_html(highlighted_lines, source_map,
                        line_offset):
    """Build source code HTML with data-regions attributes.

    Each <span class="src-line"> gets data-regions listing
    which region indices cover that line.

    Args:
        highlighted_lines: Syntax-highlighted line strings.
        source_map: Region idx -> (start, end) line mapping.
        line_offset: First line number in the source extract.
    """
    # Build reverse map: line_number -> set of region indices
    line_regions = {}
    for idx, (start, end) in source_map.items():
        for ln in range(start, end + 1):
            line_regions.setdefault(ln, set()).add(idx)

    parts = []
    for i, html_line in enumerate(highlighted_lines):
        ln = line_offset + i
        regions = line_regions.get(ln, set())
        regions_attr = ",".join(
            str(r) for r in sorted(regions)
        )
        parts.append(
            f'<span class="src-line" '
            f'data-line="{ln}" '
            f'data-regions="{regions_attr}">'
            f'<span class="ln">{ln:4d}</span> '
            f'{html_line}</span>'
        )
    return "\n".join(parts)


def _derive_bar_hierarchy(bars):
    """Set parent/children on bars from row layout.

    A bar at row N spanning [x, x+w] is parent of bars
    at row N+1 whose x range falls within [x, x+w].
    """
    eps = 1e-9
    for bar_i in bars:
        for bar_j in bars:
            if bar_j["row"] != bar_i["row"] + 1:
                continue
            if (bar_j["x"] >= bar_i["x"] - eps
                    and bar_j["x"] + bar_j["w"]
                    <= bar_i["x"] + bar_i["w"] + eps):
                bar_j["parent"] = bar_i["name"]
                bar_i["children"].append(bar_j["name"])


def _build_profile_json(data, clock_khz, kernel_name,
                         num_sweeps, gpu_name):
    """Build JSON profile data for the HTML template.

    Includes bars list with source mappings, direct_lines,
    and per-operation metrics.
    """
    active = data[data[:, 0] > 0]
    assert len(active) > 0, (
        "No active work units — cannot build profile"
    )

    avg = active.mean(axis=0)
    total = float(avg[0])
    source_map = (GIBBS_SOURCE_MAP if kernel_name == "gibbs"
                  else SA_SOURCE_MAP)
    region_names = (
        GIBBS_REGION_NAMES if kernel_name == "gibbs"
        else SA_REGION_NAMES
    )
    direct_lines = _compute_direct_lines(source_map)

    if kernel_name == "gibbs":
        rows = _build_gibbs_flame_rows(avg, total)
    else:
        rows = _build_sa_flame_rows(avg, total)

    # Map bar names back to region indices
    name_to_idx = {}
    for idx, name in enumerate(region_names):
        name_to_idx[name] = idx
    # Handle shortened names used in flame bars
    short_names = {
        "SETUP": "COLOR_SETUP",
        "NODE_LOOP": "NODE_LOOP_TOTAL",
        "SYNC": "SYNC_COLOR",
        "SKIP": "THRESHOLD_SKIP",
        "ACCEPT": "ACCEPT_DECIDE",
    }

    bars = []
    for bar in rows:
        name, row, x_start, width, color, cycles, pct = bar
        full_name = short_names.get(name, name)
        region_idx = name_to_idx.get(full_name)
        src_lines = None
        direct = None
        if region_idx is not None and region_idx in source_map:
            s, e = source_map[region_idx]
            src_lines = [s, e]
            direct = direct_lines.get(region_idx, [])
        bars.append({
            "name": name,
            "fullName": full_name,
            "row": row,
            "x": x_start,
            "w": width,
            "color": _jitter_color(name),
            "cycles": int(cycles),
            "pct": round(pct, 2),
            "time": fmt_time(
                cycles_to_time(cycles, clock_khz)
            ),
            "sourceLines": src_lines,
            "directLines": direct,
            "parent": None,
            "children": [],
        })

    _derive_bar_hierarchy(bars)
    topo = get_topology_stats()

    return {
        "kernel": kernel_name,
        "gpu": gpu_name,
        "clockMhz": round(clock_khz / 1000.0),
        "totalCycles": int(total),
        "totalTime": fmt_time(
            cycles_to_time(total, clock_khz)
        ),
        "numSweeps": num_sweeps,
        "bars": bars,
        "topology": {
            "nodes": topo[0],
            "edges": topo[1],
            "avgDegree": round(topo[2], 1),
            "nodesPerColor": topo[3],
            "numColors": topo[4],
        },
    }


def _render_flame_svg(bars, viewbox_w=1200,
                       bar_height=18, pad=1):
    """Build inline SVG with viewBox, <g> groups, and clipPath.

    Uses viewBox for zoom support. Each bar is a <g> with a
    <rect> and <text> clipped to the rect width.
    """
    max_row = max(b["row"] for b in bars)
    svg_h = (max_row + 1) * (bar_height + pad) + pad

    parts = [
        f'<svg id="flame-svg" width="100%" '
        f'height="{svg_h}" '
        f'viewBox="0 0 {viewbox_w} {svg_h}" '
        f'preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
    ]

    # Emit clipPath defs for each bar
    for idx, b in enumerate(bars):
        x = b["x"] * viewbox_w
        w = b["w"] * viewbox_w
        y = ((max_row - b["row"])
             * (bar_height + pad) + pad)
        parts.append(
            f'<clipPath id="cp-{idx}">'
            f'<rect x="{x:.2f}" y="{y}" '
            f'width="{max(w, 0.5):.2f}" '
            f'height="{bar_height}"/>'
            f'</clipPath>'
        )
    parts.append('</defs>')

    # Emit bar groups
    for idx, b in enumerate(bars):
        x = b["x"] * viewbox_w
        w = b["w"] * viewbox_w
        y = ((max_row - b["row"])
             * (bar_height + pad) + pad)
        src = (json.dumps(b["sourceLines"])
               if b["sourceLines"] else "null")
        direct = (json.dumps(b["directLines"])
                  if b["directLines"] else "null")

        parts.append(
            f'<g class="flame-bar" '
            f'data-idx="{idx}" '
            f'data-name="{b["name"]}" '
            f'data-full-name="{b["fullName"]}" '
            f'data-cycles="{b["cycles"]}" '
            f'data-pct="{b["pct"]}" '
            f'data-time="{b["time"]}" '
            f"data-source-lines='{src}' "
            f"data-direct-lines='{direct}'>"
        )
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" '
            f'width="{max(w, 0.5):.2f}" '
            f'height="{bar_height}" '
            f'fill="{b["color"]}"/>'
        )
        label = b["name"]
        parts.append(
            f'<text clip-path="url(#cp-{idx})" '
            f'x="{x + 3:.2f}" '
            f'y="{y + bar_height - 4}" '
            f'font-size="12" fill="#000" '
            f'font-family="monospace" '
            f'pointer-events="none">'
            f'{label}</text>'
        )
        parts.append('</g>')

    parts.append('</svg>')
    return "\n".join(parts)


def _build_metrics_html(data, clock_khz, kernel_name,
                         num_sweeps, gpu_name, topo):
    """Build metrics table HTML for the collapsible panel."""
    active = data[data[:, 0] > 0]
    avg = active.mean(axis=0)
    total = float(avg[0])
    region_names = (
        GIBBS_REGION_NAMES if kernel_name == "gibbs"
        else SA_REGION_NAMES
    )

    if kernel_name == "gibbs":
        cycle_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        cycle_indices = [0, 1, 2, 4, 5, 6, 7]

    rows = []
    for idx in cycle_indices:
        c = avg[idx]
        t = cycles_to_time(c, clock_khz)
        pct = (c / total * 100) if total > 0 else 0
        name = region_names[idx]
        rows.append(
            f'<tr><td>{name}</td>'
            f'<td class="r">{fmt_cycles(c)}</td>'
            f'<td class="r">{fmt_time(t)}</td>'
            f'<td class="r">{pct:.1f}%</td></tr>'
        )

    topo_html = (
        f'<p class="topo">Topology: {topo["nodes"]} nodes, '
        f'{topo["edges"]} edges, '
        f'avg degree {topo["avgDegree"]}, '
        f'{topo["numColors"]} colors, '
        f'~{topo["nodesPerColor"]} nodes/color</p>'
    )

    return (
        f'{topo_html}'
        f'<table class="metrics">'
        f'<tr><th>Region</th><th>Avg Cycles</th>'
        f'<th>Time</th><th>%</th></tr>'
        f'{"".join(rows)}'
        f'</table>'
    )


# Self-contained HTML template with CSS + JS.
# Uses sentinel replacement to avoid brace-escaping.
# All data comes from our own profiling output — no
# external/user input flows into the DOM.
HTML_TEMPLATE = '''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title><!--KERNEL_NAME--> Kernel Flamegraph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;
font-family:'Fira Code','Consolas',monospace;
font-size:13px}
.header{background:#16213e;padding:12px 20px;
border-bottom:1px solid #0f3460;
font-size:16px;font-weight:bold;color:#e94560}
.header span{color:#888;font-size:13px;
font-weight:normal;margin-left:12px}
.toolbar{display:flex;align-items:center;gap:8px;
padding:6px 20px;background:#16213e;
border-bottom:1px solid #0f3460;height:30px}
.toolbar button{background:#0f3460;color:#e0e0e0;
border:1px solid #e94560;border-radius:4px;
padding:2px 10px;cursor:pointer;font-size:12px;
font-family:inherit}
.toolbar button:disabled{opacity:0.4;cursor:default}
.toolbar button:not(:disabled):hover{
background:#e94560;color:#fff}
.toolbar input{background:#0f0f23;color:#e0e0e0;
border:1px solid #333;border-radius:4px;
padding:2px 8px;font-size:12px;font-family:inherit;
width:200px}
.toolbar input:focus{outline:none;
border-color:#e94560}
#search-match{color:#888;font-size:12px}
.container{display:flex;
height:calc(100vh - 140px)}
.flame-panel{flex:0 0 60%;padding:16px;
overflow:auto;border-right:1px solid #0f3460;
position:relative}
.source-panel{flex:0 0 40%;padding:16px;
overflow:auto;background:#0f0f23}
.source-panel pre{line-height:1.6;white-space:pre;
tab-size:4}
.src-line{display:block;padding:0 8px;
border-left:3px solid transparent;
transition:background .15s,border-color .15s}
.src-line:hover{background:#1a1a4e}
.src-line.hl{background:#2a2a5e;
border-left-color:#e94560}
.src-line.hl-direct{background:#3a2a3e;
border-left-color:#ff6b9d}
.ln{color:#555;user-select:none;
display:inline-block;width:4ch;text-align:right;
margin-right:8px}
.kw{color:#c678dd} .ty{color:#e5c07b}
.cm{color:#5c6370;font-style:italic}
.pp{color:#d19a66} .st{color:#98c379}
.nu{color:#d19a66}
.flame-bar{cursor:pointer}
.flame-bar rect{stroke:none;
transition:opacity .15s}
.flame-bar:hover rect{stroke:#000;
stroke-width:0.5}
.flame-bar.selected rect{stroke:#e94560;
stroke-width:2}
.flame-bar.matched rect{fill:#ee0 !important}
.flame-bar.dimmed rect{opacity:0.3}
#info-bar{height:18px;background:#16213e;
padding:0 12px;font-size:12px;color:#e0e0e0;
font-family:monospace;line-height:18px;
border-top:1px solid #0f3460;overflow:hidden;
white-space:nowrap}
.metrics-panel{border-top:1px solid #0f3460;
padding:12px 20px;background:#16213e}
.metrics-panel summary{cursor:pointer;
color:#e94560;font-weight:bold;font-size:14px;
user-select:none}
.metrics-panel summary:hover{color:#ff6b9d}
.metrics{width:100%;border-collapse:collapse;
margin-top:8px;font-size:12px}
.metrics th{text-align:left;color:#888;
border-bottom:1px solid #333;padding:4px 12px}
.metrics td{padding:4px 12px;
border-bottom:1px solid #222}
.metrics .r{text-align:right;font-variant-numeric:
tabular-nums}
.topo{color:#888;margin-bottom:8px;font-size:12px}
</style>
</head>
<body>
<div class="header"><!--KERNEL_NAME--> Kernel Profile
<span><!--GPU_INFO--></span></div>
<div class="toolbar">
<button id="reset-zoom" disabled>Reset Zoom</button>
<input id="search-input" type="text"
 placeholder="Search regex...">
<span id="search-match"></span>
</div>
<div class="container">
<div class="flame-panel">
<!--FLAME_SVG-->
</div>
<div class="source-panel">
<pre><!--SOURCE_HTML--></pre>
</div>
</div>
<div id="info-bar">&nbsp;</div>
<div class="metrics-panel">
<details>
<summary>Metrics &amp; Topology</summary>
<!--METRICS_HTML-->
</details>
</div>
<script>
(function() {
"use strict";
var profile = <!--PROFILE_JSON-->;
var svg = document.getElementById("flame-svg");
var barEls = document.querySelectorAll(".flame-bar");
var srcLines = document.querySelectorAll(".src-line");
var infoBar = document.getElementById("info-bar");
var resetBtn = document.getElementById("reset-zoom");
var searchInput = document.getElementById("search-input");
var searchMatch = document.getElementById("search-match");
var rNames = profile.kernel === "gibbs"
  ? ["ANNEALING_TOTAL","BETA_ITER_TOTAL",
     "SWEEP_ITER_TOTAL","COLOR_ITER_TOTAL",
     "COLOR_SETUP","NODE_LOOP_TOTAL",
     "FIELD_COMPUTE","SPIN_UPDATE","SYNC_COLOR"]
  : ["SA_TOTAL","BETA_OVERHEAD","SWEEP_TOTAL",
     "VAR_SCAN","THRESHOLD_SKIP","ACCEPT_DECIDE",
     "FLIP_TOTAL","NEIGHBOR_LOOP"];

var origViewBox = svg.getAttribute("viewBox");
var zoomStack = [];
var selectedBar = null;

function clearSourceHL() {
  for (var j = 0; j < srcLines.length; j++)
    srcLines[j].classList.remove("hl","hl-direct");
}

function clearBarSelection() {
  for (var i = 0; i < barEls.length; i++)
    barEls[i].classList.remove("selected");
  selectedBar = null;
}

function highlightSourceRange(start, end) {
  for (var i = 0; i < srcLines.length; i++) {
    var ln = parseInt(srcLines[i].dataset.line, 10);
    if (ln >= start && ln <= end)
      srcLines[i].classList.add("hl");
  }
  var first = document.querySelector(".src-line.hl");
  if (first) first.scrollIntoView(
    {behavior:"smooth",block:"center"});
}

function highlightDirectLines(lines) {
  for (var i = 0; i < srcLines.length; i++) {
    var ln = parseInt(srcLines[i].dataset.line, 10);
    if (lines.indexOf(ln) !== -1)
      srcLines[i].classList.add("hl-direct");
  }
  var first = document.querySelector(
    ".src-line.hl-direct");
  if (first) first.scrollIntoView(
    {behavior:"smooth",block:"center"});
}

function highlightBarsForRegions(regionStr) {
  if (!regionStr) return;
  var regions = regionStr.split(",").map(Number);
  for (var i = 0; i < barEls.length; i++) {
    var bd = profile.bars[i];
    if (!bd) continue;
    for (var r = 0; r < regions.length; r++) {
      var ri = regions[r];
      if (ri < rNames.length &&
          bd.fullName === rNames[ri])
        barEls[i].classList.add("selected");
    }
  }
}

function setInfo(text) {
  infoBar.textContent = text || "\u00a0";
}

/* ── Zoom ─────────────────────────────────── */

function zoomToBar(barData) {
  var vbW = 1200;
  var x = barData.x * vbW;
  var w = barData.w * vbW;
  var vb = svg.getAttribute("viewBox").split(" ");
  zoomStack.push(vb.join(" "));
  svg.setAttribute("viewBox",
    x + " 0 " + w + " " + vb[3]);
  resetBtn.disabled = false;
  setInfo("Zoomed: " + barData.name +
    " (" + barData.pct + "%, " + barData.time + ")");
}

function resetZoom() {
  svg.setAttribute("viewBox", origViewBox);
  zoomStack = [];
  resetBtn.disabled = true;
  setInfo("");
}

resetBtn.addEventListener("click", function() {
  resetZoom();
});

/* ── Search ───────────────────────────────── */

searchInput.addEventListener("input", function() {
  var q = this.value.trim();
  if (!q) {
    for (var i = 0; i < barEls.length; i++) {
      barEls[i].classList.remove("matched","dimmed");
    }
    searchMatch.textContent = "";
    setInfo("");
    return;
  }
  var re;
  try { re = new RegExp(q, "i"); }
  catch(e) { return; }
  var matched = 0;
  for (var i = 0; i < barEls.length; i++) {
    var name = barEls[i].dataset.name;
    if (re.test(name)) {
      barEls[i].classList.add("matched");
      barEls[i].classList.remove("dimmed");
      matched++;
    } else {
      barEls[i].classList.remove("matched");
      barEls[i].classList.add("dimmed");
    }
  }
  var msg = matched + " of " + barEls.length +
    " bars match";
  searchMatch.textContent = msg;
  setInfo(msg);
});

/* ── Hover (info bar) ─────────────────────── */

for (var i = 0; i < barEls.length; i++) {
  (function(bar, idx) {
    bar.addEventListener("mouseenter", function() {
      var bd = profile.bars[idx];
      if (!bd) return;
      setInfo(bd.name + "  |  " +
        parseInt(bd.cycles,10).toLocaleString() +
        " cycles  |  " + bd.time +
        "  |  " + bd.pct + "%");
      if (!selectedBar) {
        clearSourceHL();
        var src = JSON.parse(
          this.getAttribute("data-source-lines"));
        if (src) highlightSourceRange(src[0], src[1]);
      }
    });
    bar.addEventListener("mouseleave", function() {
      if (zoomStack.length > 0) {
        var top = profile.bars[parseInt(
          barEls[barEls.length-1].dataset.idx,10)];
        setInfo("");
      } else {
        setInfo("");
      }
      if (!selectedBar) clearSourceHL();
    });
    bar.addEventListener("click", function(e) {
      e.stopPropagation();
      var bd = profile.bars[idx];
      if (!bd) return;
      clearBarSelection();
      clearSourceHL();
      // Zoom
      zoomToBar(bd);
      // Highlight source
      selectedBar = this;
      this.classList.add("selected");
      var direct = JSON.parse(
        this.getAttribute("data-direct-lines"));
      if (direct && direct.length > 0) {
        highlightDirectLines(direct);
      } else {
        var src = JSON.parse(
          this.getAttribute("data-source-lines"));
        if (src) highlightSourceRange(src[0], src[1]);
      }
    });
  })(barEls[i], parseInt(barEls[i].dataset.idx, 10));
}

/* ── Source line click → highlight bars ───── */

for (var j = 0; j < srcLines.length; j++) {
  (function(line) {
    line.addEventListener("click", function() {
      clearBarSelection();
      clearSourceHL();
      this.classList.add("hl");
      highlightBarsForRegions(this.dataset.regions);
    });
  })(srcLines[j]);
}

/* ── Click background to deselect ─────────── */

document.querySelector(".flame-panel")
  .addEventListener("click", function(e) {
    if (e.target.closest(".flame-bar")) return;
    clearBarSelection();
    clearSourceHL();
  });
})();
</script>
</body>
</html>
'''


def generate_flamegraph_html(data, clock_khz, kernel_name,
                              num_sweeps, gpu_name,
                              output_path):
    """Generate interactive HTML flamegraph with source linking.

    Reads the CUDA source file, builds profile JSON, renders
    inline SVG, and assembles a self-contained HTML file.

    Args:
        data: Raw profiling array (work_units x regions).
        clock_khz: GPU clock rate in kHz.
        kernel_name: 'gibbs' or 'sa'.
        num_sweeps: Number of sweeps used in profiling.
        gpu_name: GPU device name string.
        output_path: Output HTML file path.
    """
    source_map = (GIBBS_SOURCE_MAP if kernel_name == "gibbs"
                  else SA_SOURCE_MAP)

    # Determine source line range to extract
    all_starts = [s for s, _ in source_map.values()]
    all_ends = [e for _, e in source_map.values()]
    line_start = min(all_starts) - 5  # context padding
    line_end = max(all_ends) + 5

    # Read and extract source
    full_source = _read_cu_source(kernel_name)
    if full_source:
        all_lines = full_source.splitlines()
        extract = all_lines[line_start - 1:line_end]
    else:
        extract = ["// Source file not found"]
        line_start = 1

    highlighted = _highlight_cuda_source("\n".join(extract))
    source_html = _build_source_html(
        highlighted, source_map, line_start,
    )

    profile = _build_profile_json(
        data, clock_khz, kernel_name, num_sweeps, gpu_name,
    )
    flame_svg = _render_flame_svg(profile["bars"])
    metrics_html = _build_metrics_html(
        data, clock_khz, kernel_name, num_sweeps,
        gpu_name, profile["topology"],
    )

    clock_mhz = clock_khz / 1000.0
    gpu_info = f"{gpu_name} @ {clock_mhz:.0f} MHz"

    html = (HTML_TEMPLATE
            .replace('<!--PROFILE_JSON-->',
                     json.dumps(profile))
            .replace('<!--SOURCE_HTML-->', source_html)
            .replace('<!--FLAME_SVG-->', flame_svg)
            .replace('<!--KERNEL_NAME-->',
                     kernel_name.upper())
            .replace('<!--GPU_INFO-->', gpu_info)
            .replace('<!--METRICS_HTML-->', metrics_html))

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\nInteractive flamegraph saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA kernel region profiling"
    )
    parser.add_argument(
        "--kernel", choices=["gibbs", "sa"],
        default="gibbs",
        help="Which kernel to profile (default: gibbs)",
    )
    parser.add_argument(
        "--sweeps", type=int, default=1000,
        help="Number of sweeps (default: 1000)",
    )
    parser.add_argument(
        "--reads", type=int, default=100,
        help="Number of reads (default: 100)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib visualization",
    )
    parser.add_argument(
        "--flame", action="store_true",
        help="Generate flamegraph SVG visualization",
    )
    parser.add_argument(
        "--flame-html", action="store_true",
        help="Generate interactive HTML flamegraph "
             "with source linking",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for --flame-html "
             "(default: flamegraphs/flamegraph_KERNEL.html)",
    )
    args = parser.parse_args()

    gpu_name, clock_khz, clock_mhz = get_gpu_info()
    print(f"GPU: {gpu_name}, SM clock: {clock_mhz:.0f} MHz")
    print(f"Kernel: {args.kernel}, sweeps={args.sweeps}, "
          f"reads={args.reads}")

    # Create output directory for flamegraphs if needed
    if args.flame or args.flame_html:
        os.makedirs("flamegraphs", exist_ok=True)

    if args.kernel == "gibbs":
        data = profile_gibbs(args.reads, args.sweeps)
        print_gibbs_profile(
            data, clock_khz, args.reads, args.sweeps
        )
        print_precise_ticks(
            data, clock_khz, args.sweeps,
            "gibbs", gpu_name,
        )
        if args.plot:
            plot_profile(
                data, GIBBS_REGION_NAMES, clock_khz,
                "gibbs", "profile_gibbs.png",
            )
        if args.flame:
            generate_flamegraph(
                data, clock_khz,
                "gibbs",
                "flamegraphs/flamegraph_gibbs.svg",
            )
        if args.flame_html:
            out = (args.output
                   or "flamegraphs/flamegraph_gibbs.html")
            generate_flamegraph_html(
                data, clock_khz, "gibbs",
                args.sweeps, gpu_name, out,
            )
    else:
        data = profile_sa(args.reads, args.sweeps)
        print_sa_profile(
            data, clock_khz, args.reads, args.sweeps
        )
        print_precise_ticks(
            data, clock_khz, args.sweeps,
            "sa", gpu_name,
        )
        if args.plot:
            plot_profile(
                data, SA_REGION_NAMES, clock_khz,
                "sa", "profile_sa.png",
            )
        if args.flame:
            generate_flamegraph(
                data, clock_khz,
                "sa",
                "flamegraphs/flamegraph_sa.svg",
            )
        if args.flame_html:
            out = (args.output
                   or "flamegraphs/flamegraph_sa.html")
            generate_flamegraph_html(
                data, clock_khz, "sa",
                args.sweeps, gpu_name, out,
            )


if __name__ == "__main__":
    main()
