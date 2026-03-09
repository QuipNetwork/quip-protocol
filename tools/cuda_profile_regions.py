#!/usr/bin/env python3
"""CUDA kernel region profiling via clock64() instrumentation.

Runs SA or Gibbs kernel with PROFILE_REGIONS enabled, reads back
per-region cycle counters, and outputs a hierarchical breakdown
showing where GPU time is spent inside the kernel.

Usage:
    python tools/cuda_profile_regions.py [--kernel gibbs|sa] \
        [--sweeps N] [--reads N] [--plot]
"""

import argparse
import sys

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

    data = kernel.get_profile_data()
    kernel.stop_immediate()

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
        sweep_total_cycles = avg[2] / sweep_count if sweep_count > 0 else 0
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
        print(f"  {'TOTAL':<20} "
              f"{fmt_cycles(sweep_total_cycles):>14} "
              f"{fmt_time(cycles_to_time(sweep_total_cycles, clock_khz)):>10}")

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
    args = parser.parse_args()

    gpu_name, clock_khz, clock_mhz = get_gpu_info()
    print(f"GPU: {gpu_name}, SM clock: {clock_mhz:.0f} MHz")
    print(f"Kernel: {args.kernel}, sweeps={args.sweeps}, "
          f"reads={args.reads}")

    if args.kernel == "gibbs":
        data = profile_gibbs(args.reads, args.sweeps)
        print_gibbs_profile(
            data, clock_khz, args.reads, args.sweeps
        )
        if args.plot:
            plot_profile(
                data, GIBBS_REGION_NAMES, clock_khz,
                "gibbs", "profile_gibbs.png",
            )
    else:
        data = profile_sa(args.reads, args.sweeps)
        print_sa_profile(
            data, clock_khz, args.reads, args.sweeps
        )
        if args.plot:
            plot_profile(
                data, SA_REGION_NAMES, clock_khz,
                "sa", "profile_sa.png",
            )


if __name__ == "__main__":
    main()
