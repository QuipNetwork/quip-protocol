#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Profile Gibbs GPU pipeline to diagnose bursty utilization.

Three modes:
  raw          — sampler launch/sync/download cycle only.
                 No Python pipeline overhead. Isolates kernel
                 launch latency and H2D/D2H transfer time.
  pipeline     — full IsingPipeline with ProcessPoolExecutor +
                 generator thread. Captures end-to-end overhead.
  self-feeding — self-feeding kernel with 3-slot rotating buffers.
                 Kernel stays resident; host polls completions.

Usage:
  # Raw sampler (nsys captures GPU timeline)
  nsys profile -o /tmp/gibbs_raw --force-overwrite \\
    -t cuda,nvtx --stats=true \\
    .quip/bin/python3 tools/profile_gibbs_pipeline.py --mode raw

  # Self-feeding kernel
  nsys profile -o /tmp/gibbs_sf --force-overwrite \\
    -t cuda,nvtx --stats=true \\
    .quip/bin/python3 tools/profile_gibbs_pipeline.py \\
      --mode self-feeding

  # Quick check without nsys (prints wall-clock times)
  .quip/bin/python3 tools/profile_gibbs_pipeline.py --mode raw

MPS (Multi-Process Service) for true SM sharing:
  # Start MPS for true SM sharing between processes:
  #   sudo nvidia-cuda-mps-control -d
  #
  # Stop MPS:
  #   echo quit | sudo nvidia-cuda-mps-control
"""

import argparse
import logging
import time

import cupy as cp
import numpy as np


NUM_ITERATIONS = 10
READS_PER_NONCE = 32
NUM_SWEEPS = 1000
SMS_PER_NONCE = 4


def build_ising_problems(num_nonces):
    """Build Ising problems using the real mining path.

    Uses generate_ising_model_from_nonce with fake block
    context so buffer sizes, sparsity, and value distributions
    match production exactly.
    """
    import random as stdlib_random

    from dwave_topologies import DEFAULT_TOPOLOGY
    from shared.quantum_proof_of_work import (
        generate_ising_model_from_nonce,
        ising_nonce_from_block,
    )

    graph = DEFAULT_TOPOLOGY.graph
    nodes = sorted(graph.nodes())
    edges = list(graph.edges())

    prev_hash = b"\x00" * 32
    miner_id = "profile-test"
    cur_index = 0

    h_list, J_list = [], []
    for _ in range(num_nonces):
        salt = stdlib_random.randbytes(32)
        nonce = ising_nonce_from_block(
            prev_hash,
            miner_id,
            cur_index,
            salt,
        )
        h, J = generate_ising_model_from_nonce(
            nonce,
            nodes,
            edges,
        )
        h_list.append(h)
        J_list.append(J)
    return h_list, J_list, nodes, edges


def profile_raw(num_nonces, iterations):
    """Profile raw sampler: launch → sync → launch → download.

    No pipeline threads, no process pool. Pure GPU cycle.
    """
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    h_list, J_list, _, _ = build_ising_problems(num_nonces)
    # Second batch for preload alternation
    h_list2, J_list2, _, _ = build_ising_problems(num_nonces)

    sampler = CudaGibbsSampler()
    sampler.prepare(
        num_reads=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        max_nonces=num_nonces,
    )

    mn_params = dict(
        reads_per_nonce=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        sms_per_nonce=SMS_PER_NONCE,
    )

    # Warm up — JIT compile + cache population.
    # harvest_multi_nonce = harvest_sync + download (consumes
    # pending state), so we do a full cycle here.
    sampler.launch_multi_nonce(h_list, J_list, **mn_params)
    _ = sampler.harvest_multi_nonce()

    # Kick off a kernel so the loop can harvest_sync it
    sampler.launch_multi_nonce(h_list, J_list, **mn_params)

    times = {
        "preload": [],
        "harvest_sync": [],
        "launch": [],
        "download": [],
        "total": [],
    }

    # Profiled iterations: preload → sync → launch → download
    # Mirrors IsingPipeline.next_batch() exactly.
    for i in range(iterations):
        t_total = time.perf_counter()

        # Alternate h/J to exercise both staging buffers
        cur_h = h_list if i % 2 == 0 else h_list2
        cur_J = J_list if i % 2 == 0 else J_list2

        t0 = time.perf_counter()
        sampler.preload_multi_nonce(
            cur_h,
            cur_J,
            **mn_params,
        )
        times["preload"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pending = sampler.harvest_sync()
        times["harvest_sync"].append(
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        sampler.launch_multi_nonce([], [], **mn_params)
        times["launch"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = sampler.download_results(pending)
        times["download"].append(time.perf_counter() - t0)

        times["total"].append(
            time.perf_counter() - t_total,
        )

    sampler.close()
    return times


def profile_pipeline(num_nonces, iterations):
    """Profile full IsingPipeline end-to-end.

    Uses the real mining pipeline: ProcessPoolExecutor for
    CPU model generation + generator thread for H2D staging.
    """
    from GPU.cuda_gibbs_sa import CudaGibbsSampler
    from GPU.cuda_gibbs_miner import IsingPipeline
    from GPU.gpu_scheduler import KernelScheduler

    _, _, nodes, edges = build_ising_problems(1)

    dev_id = 0
    cp.cuda.Device(dev_id).use()
    device_sms = cp.cuda.Device(dev_id).attributes["MultiProcessorCount"]
    scheduler = KernelScheduler(
        device_id=dev_id,
        device_sms=device_sms,
        gpu_utilization_pct=100,
        yielding=False,
    )
    sm_budget = scheduler.get_sm_budget()
    if num_nonces == 0:
        num_nonces = max(1, sm_budget // SMS_PER_NONCE)
    max_nonces = 2 * num_nonces

    sampler = CudaGibbsSampler(max_sms=sm_budget)
    sampler.prepare(
        num_reads=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        max_nonces=max_nonces,
    )

    # Fake mining context for model generation
    prev_hash = b"\x00" * 32
    miner_id = "profile-test"
    cur_index = 0

    mn_params = dict(
        reads_per_nonce=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        sms_per_nonce=SMS_PER_NONCE,
    )

    pipeline = IsingPipeline(
        sampler,
        prev_hash,
        miner_id,
        cur_index,
        nodes,
        edges,
        num_nonces,
        mn_params,
    )

    t0 = time.perf_counter()
    pipeline.start()
    cold_start = time.perf_counter() - t0

    batch_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = pipeline.next_batch()
        batch_times.append(time.perf_counter() - t0)

    pipeline.stop()
    scheduler.stop()
    sampler.close()

    return {
        "cold_start": cold_start,
        "batch": batch_times,
    }


_YIELD_CHECK_INTERVAL = 2.0

logger = logging.getLogger(__name__)


def _check_yield_profiler(
    scheduler,
    sampler,
    num_betas,
    active_count,
    max_nonces,
    h_list,
    J_list,
    h_list2,
    J_list2,
):
    """Scale nonces using scheduler's fair-share formula.

    Returns the new active_count, or current if no change.
    """
    target = scheduler.check_stable_target(
        max_nonces, active_count,
    )
    if target is None or target == active_count:
        return active_count

    if target < active_count:
        for nid in range(target, active_count):
            sampler.signal_nonce_exit(nid)
        logger.info(
            "Yielding: %d -> %d nonces",
            active_count, target,
        )
        return target

    # Scale up: requires kernel restart
    sampler.signal_exit()
    sampler._d_sf_ctrl[:] = 0
    for k in range(target):
        sampler.upload_slot(
            k, 0, h_list[k], J_list[k],
        )
        sampler.upload_slot(
            k, 1, h_list2[k], J_list2[k],
        )
    sampler.launch_self_feeding(
        num_betas=num_betas,
        active_nonce_count=target,
    )
    logger.info(
        "Reclaiming: %d -> %d nonces",
        active_count, target,
    )
    return target


def profile_self_feeding(num_nonces, iterations, yielding=False):
    """Profile self-feeding kernel with 3-slot rotating buffers.

    Kernel stays resident — host uploads to free slots and polls
    completions. Measures total throughput and per-model latency.

    Args:
        num_nonces: Nonce groups to launch.
        iterations: Number of batch iterations.
        yielding: Enable dynamic nonce yielding via NVML.
    """
    from GPU.cuda_gibbs_sa import CudaGibbsSampler
    from GPU.gpu_scheduler import KernelScheduler

    h_list, J_list, _, _ = build_ising_problems(num_nonces)
    h_list2, J_list2, _, _ = build_ising_problems(num_nonces)

    sampler = CudaGibbsSampler()
    sampler.prepare(
        num_reads=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        max_nonces=num_nonces * 2,
    )
    sampler.prepare_self_feeding(
        num_nonces=num_nonces,
        reads_per_nonce=READS_PER_NONCE,
        num_sweeps=NUM_SWEEPS,
        sms_per_nonce=SMS_PER_NONCE,
    )

    # Upload beta schedule
    num_betas, _ = sampler.upload_beta_schedule(
        h_list[0],
        J_list[0],
        NUM_SWEEPS,
    )

    # Upload slot 0 and slot 1 for all nonces
    for k in range(num_nonces):
        sampler.upload_slot(k, 0, h_list[k], J_list[k])
        sampler.upload_slot(k, 1, h_list2[k], J_list2[k])

    # Launch kernel
    sampler.launch_self_feeding(num_betas=num_betas)

    # Set up yielding scheduler
    scheduler = None
    if yielding:
        dev_id = 0
        device_sms = cp.cuda.Device(dev_id).attributes["MultiProcessorCount"]
        scheduler = KernelScheduler(
            device_id=dev_id,
            device_sms=device_sms,
            gpu_utilization_pct=100,
            yielding=True,
        )
        mps = scheduler.is_mps_active()
        print(f"  MPS: {'active' if mps else 'not active'}")

    batch_times = []
    models_in_batch = 0
    batch_idx = 2
    target_models = iterations * num_nonces
    active_count = num_nonces
    last_yield_check = 0.0
    models_completed = 0

    t_start = time.perf_counter()
    t_batch = t_start

    while models_completed < target_models:
        # Periodic yield check
        if scheduler is not None:
            now = time.monotonic()
            if now - last_yield_check >= _YIELD_CHECK_INTERVAL:
                last_yield_check = now
                prev_active = active_count
                active_count = _check_yield_profiler(
                    scheduler,
                    sampler,
                    num_betas,
                    active_count,
                    num_nonces,
                    h_list,
                    J_list,
                    h_list2,
                    J_list2,
                )
                if active_count != prev_active:
                    # Reset batch boundary after scale change
                    models_in_batch = 0
                    t_batch = time.perf_counter()

        completed = sampler.poll_completions()
        if not completed:
            time.sleep(0.0001)
            continue

        for nonce_id, slot_id in completed:
            # Skip completions from yielded nonces
            if nonce_id >= active_count:
                continue
            sampler.download_slot(nonce_id, slot_id)
            models_completed += 1
            models_in_batch += 1

            # Refill with new data
            src = h_list if batch_idx % 2 == 0 else h_list2
            src_j = J_list if batch_idx % 2 == 0 else J_list2
            idx = nonce_id % len(src)
            sampler.upload_slot(
                nonce_id,
                slot_id,
                src[idx],
                src_j[idx],
            )
            batch_idx += 1

        # Record batch time based on active nonce count
        if models_in_batch >= active_count:
            t_now = time.perf_counter()
            batch_times.append(t_now - t_batch)
            t_batch = t_now
            models_in_batch = 0

    sampler.signal_exit()
    if scheduler is not None:
        scheduler.stop()
    sampler.close()

    return {
        "batch": batch_times,
        "total": time.perf_counter() - t_start,
    }


def report_self_feeding(times, num_nonces):
    """Print self-feeding mode timing report."""
    print(f"\n{'=' * 55}")
    print(f"Self-feeding profile ({num_nonces} nonces)")
    print(f"{'=' * 55}")
    ms = [t * 1000 for t in times["batch"]]
    print_stats("batch", ms)
    throughput = num_nonces / (np.mean(ms) / 1000)
    print(f"\n  Total elapsed: {times['total'] * 1000:.1f} ms")
    print(f"  Throughput:    {throughput:.1f} nonces/sec")


def print_stats(label, values_ms):
    """Print min/median/max/mean for a list of ms values."""
    arr = np.array(values_ms)
    print(
        f"  {label:15s}: "
        f"min={arr.min():7.2f}  "
        f"med={np.median(arr):7.2f}  "
        f"max={arr.max():7.2f}  "
        f"mean={arr.mean():7.2f} ms"
    )


def report_raw(times, num_nonces):
    """Print raw mode timing report."""
    print(f"\n{'=' * 55}")
    print(f"Raw sampler profile ({num_nonces} nonces)")
    print(f"{'=' * 55}")
    for key in ["preload", "harvest_sync", "launch", "download", "total"]:
        ms = [t * 1000 for t in times[key]]
        print_stats(key, ms)

    totals = [t * 1000 for t in times["total"]]
    gpu_busy = [t * 1000 for t in times["harvest_sync"]]
    overhead = [t - g for t, g in zip(totals, gpu_busy)]
    print(f"\n  GPU kernel (harvest_sync): {np.mean(gpu_busy):.2f} ms avg")
    print(f"  Pipeline overhead:         {np.mean(overhead):.2f} ms avg")
    pct = np.mean(gpu_busy) / np.mean(totals) * 100
    print(f"  GPU utilization:           {pct:.1f}%")


def report_pipeline(times, num_nonces):
    """Print pipeline mode timing report."""
    print(f"\n{'=' * 55}")
    print(f"Pipeline profile ({num_nonces} nonces)")
    print(f"{'=' * 55}")
    print(f"  Cold start: {times['cold_start'] * 1000:.2f} ms")
    ms = [t * 1000 for t in times["batch"]]
    print_stats("next_batch", ms)
    throughput = num_nonces / (np.mean(ms) / 1000)
    print(f"\n  Throughput: {throughput:.1f} nonces/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Gibbs GPU pipeline",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "raw",
            "pipeline",
            "self-feeding",
            "both",
        ],
        default="both",
        help="Profiling mode (default: both)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Profiling iterations (default: {NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--nonces",
        type=int,
        default=0,
        help="Nonces per batch (0 = auto from SM budget)",
    )
    parser.add_argument(
        "--utilization",
        type=int,
        default=100,
        help="GPU utilization percent (default: 100)",
    )
    parser.add_argument(
        "--yielding",
        action="store_true",
        help=("Enable dynamic nonce yielding (self-feeding mode only)"),
    )
    args = parser.parse_args()

    if args.yielding:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(message)s",
        )
        from GPU.gpu_scheduler import configure_mps_thread_limit
        configure_mps_thread_limit(
            args.utilization, 0, yielding=True,
        )

    dev = cp.cuda.Device(0)
    device_sms = dev.attributes["MultiProcessorCount"]
    sm_budget = max(
        1,
        int(device_sms * args.utilization / 100),
    )
    num_nonces = args.nonces
    if num_nonces == 0:
        num_nonces = max(1, sm_budget // SMS_PER_NONCE)
    print(
        f"Device: GPU 0 ({device_sms} SMs), "
        f"{sm_budget} SM budget ({args.utilization}%), "
        f"{num_nonces} nonces, "
        f"{args.iterations} iterations"
    )

    if args.mode in ("raw", "both"):
        times = profile_raw(num_nonces, args.iterations)
        report_raw(times, num_nonces)

    if args.mode in ("pipeline", "both"):
        times = profile_pipeline(
            num_nonces,
            args.iterations,
        )
        report_pipeline(times, num_nonces)

    if args.mode == "self-feeding":
        times = profile_self_feeding(
            num_nonces,
            args.iterations,
            yielding=args.yielding,
        )
        report_self_feeding(times, num_nonces)


if __name__ == "__main__":
    main()
