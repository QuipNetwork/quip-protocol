#!/usr/bin/env python3
"""Test QPU performance with varying parameters on fixed problem instances.

This tool tests how QPU parameters affect solution quality by:
1. Generating fixed problem instances from seed(s)
2. Running CPU canary baseline (64 sweeps, 32 reads)
3. Running QPU with varying num_reads and annealing_time parameters
4. Optionally interleaving multiple seeds to prevent QPU from settling on known solutions
5. Allowing configurable intervals between QPU queries
6. Async streaming (queue_depth > 1) for interval=0 to maximize throughput

The goal is to understand the relationship between QPU parameters and solution quality.
"""
import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np

from shared.block import create_genesis_block, BlockRequirements
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import load_topology


def parse_duration(duration_str: str) -> float:
    """Parse duration string to seconds.

    Supports: 30s, 5m, 2h, 1d, 1w
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60.0
    elif duration_str.endswith('h'):
        return float(duration_str[:-1]) * 3600.0
    elif duration_str.endswith('d'):
        return float(duration_str[:-1]) * 86400.0
    elif duration_str.endswith('w'):
        return float(duration_str[:-1]) * 604800.0
    else:
        return float(duration_str)


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in s.split(',')]


def parse_duration_list(s: str) -> List[float]:
    """Parse comma-separated duration strings to list of seconds."""
    return [parse_duration(x.strip()) for x in s.split(',')]


# Conservative constants for Advantage2 QPU time estimation.
# Derived from actual D-Wave timing: 2048 reads x 320us -> 1,014,436us.
QPU_MAX_ACCESS_TIME_US = 1_000_000
QPU_PROGRAMMING_TIME_US = 15_000
QPU_PER_SAMPLE_OVERHEAD_US = 180  # readout + thermalization (conservative)


def exceeds_qpu_time_limit(num_reads: int, annealing_time: float) -> bool:
    """Check if a parameter combination would exceed QPU time limit."""
    estimated = (QPU_PROGRAMMING_TIME_US
                 + num_reads * (annealing_time + QPU_PER_SAMPLE_OVERHEAD_US))
    return estimated > QPU_MAX_ACCESS_TIME_US


def generate_nonce(seed: int, topology) -> Tuple[str, Dict]:
    """Generate a nonce and Ising model from a seed."""
    random.seed(seed)
    np.random.seed(seed)

    prev_block = create_genesis_block()
    salt = random.randbytes(32)
    nonce = ising_nonce_from_block(
        prev_block.hash,
        f"qpu-test-{seed}",
        1,
        salt
    )

    nodes = list(topology.nodes)
    edges = list(topology.edges)
    h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

    return nonce, {'h': h, 'J': J, 'salt': salt.hex()}


def run_cpu_baseline(cpu_miner, h, J, num_sweeps: int, num_reads: int) -> Dict:
    """Run CPU baseline test."""
    start_time = time.time()
    sampleset = cpu_miner.sampler.sample_ising(
        h, J,
        num_reads=num_reads,
        num_sweeps=num_sweeps
    )
    elapsed = time.time() - start_time

    energies = sampleset.record.energy

    return {
        'energy_min': float(np.min(energies)),
        'energy_max': float(np.max(energies)),
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'time': elapsed,
        'num_sweeps': num_sweeps,
        'num_reads': num_reads,
        'all_energies': [float(e) for e in energies]
    }


def extract_result(sampleset, num_reads: int, annealing_time: float,
                   elapsed: float) -> Dict:
    """Extract result dict from a resolved sampleset."""
    energies = sampleset.record.energy

    result = {
        'energy_min': float(np.min(energies)),
        'energy_max': float(np.max(energies)),
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'time': elapsed,
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'all_energies': [float(e) for e in energies]
    }

    if hasattr(sampleset, 'info') and sampleset.info:
        timing = sampleset.info.get('timing', {})
        if timing:
            result['qpu_timing'] = {
                'qpu_access_time': timing.get('qpu_access_time'),
                'qpu_anneal_time_per_sample': timing.get('qpu_anneal_time_per_sample'),
                'qpu_programming_time': timing.get('qpu_programming_time'),
                'qpu_readout_time_per_sample': timing.get('qpu_readout_time_per_sample'),
                'total_post_processing_time': timing.get('total_post_processing_time'),
            }

    return result


def run_qpu_test(qpu_miner, h, J, nonce: int, num_reads: int,
                 annealing_time: float) -> Dict:
    """Run QPU test synchronously (for interval > 0)."""
    start_time = time.time()

    topology_label = qpu_miner.sampler.job_label
    nonce_hex = hex(nonce)[2:][:8]
    job_label = f"{topology_label}_{nonce_hex}"

    sampleset = qpu_miner.sampler.sample_ising(
        h, J,
        num_reads=num_reads,
        annealing_time=annealing_time,
        label=job_label
    )
    elapsed = time.time() - start_time

    return extract_result(sampleset, num_reads, annealing_time, elapsed)


@dataclass
class PendingJob:
    """Tracks a pending async QPU job."""
    future: Any
    seed: int
    interval_seconds: float
    num_reads: int
    annealing_time: float
    submit_time: float
    test_index: int


def save_results_incremental(output_file: str, output_data: Dict):
    """Save current results to JSON (overwrites on each call)."""
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def build_output_data(seeds, topology, num_reads_list, annealing_time_list,
                      interval_list, cpu_baseline_sweeps, cpu_baseline_reads,
                      cpu_baselines, qpu_results, queue_depth):
    """Build the full output data dict for JSON serialization."""
    qpu_results_serializable = {}
    for (seed, interval_seconds), results in qpu_results.items():
        key_str = f"seed_{seed}_interval_{interval_seconds}"
        qpu_results_serializable[key_str] = {
            'seed': seed,
            'interval': interval_seconds,
            'results': results
        }

    return {
        'seeds': seeds,
        'topology': topology.solver_name,
        'num_reads_tested': num_reads_list,
        'annealing_time_tested': annealing_time_list,
        'interval_tested': interval_list,
        'queue_depth': queue_depth,
        'cpu_baseline': {
            'num_sweeps': cpu_baseline_sweeps,
            'num_reads': cpu_baseline_reads,
            'results': cpu_baselines
        },
        'qpu_results': qpu_results_serializable,
        'timestamp': time.time()
    }


def run_streaming(qpu_miner, nonces_and_models, num_reads_list,
                  annealing_time_list, interval_seconds, qpu_results,
                  queue_depth, total_tests, test_counter, output_file,
                  output_data_builder):
    """Run QPU tests with async streaming for maximum throughput.

    Maintains up to queue_depth jobs in-flight simultaneously.
    Saves results incrementally after each completed job.
    """
    # Build work queue, skipping combos that exceed QPU time limit
    work_items = []
    skipped_combos = 0
    for num_reads in num_reads_list:
        for annealing_time in annealing_time_list:
            if exceeds_qpu_time_limit(num_reads, annealing_time):
                skipped_combos += 1
                continue
            for seed, nonce, model in nonces_and_models:
                work_items.append((seed, nonce, model, num_reads, annealing_time))

    if skipped_combos:
        skipped_jobs = skipped_combos * len(nonces_and_models)
        print(f"  Skipped {skipped_combos} param combo(s) exceeding "
              f"{QPU_MAX_ACCESS_TIME_US / 1e6:.0f}s QPU limit "
              f"({skipped_jobs} jobs)")

    topology_label = qpu_miner.sampler.job_label
    pending: Dict[Any, PendingJob] = {}
    work_index = 0
    completed_count = 0
    total_work = len(work_items)

    def submit_job(idx: int) -> PendingJob:
        seed, nonce, model, num_reads, annealing_time = work_items[idx]
        nonce_hex = hex(nonce)[2:][:8]
        job_label = f"{topology_label}_{nonce_hex}"

        future = qpu_miner.sampler.sample_ising_async(
            model['h'], model['J'],
            num_reads=num_reads,
            annealing_time=annealing_time,
            label=job_label
        )
        return PendingJob(
            future=future,
            seed=seed,
            interval_seconds=interval_seconds,
            num_reads=num_reads,
            annealing_time=annealing_time,
            submit_time=time.time(),
            test_index=test_counter[0] + idx + 1
        )

    # Fill initial queue
    while len(pending) < queue_depth and work_index < total_work:
        try:
            job = submit_job(work_index)
            pending[id(job.future)] = job
            work_index += 1
        except Exception as e:
            print(f"\n  ⚠️  Submit error (work_index={work_index}): {e}")
            work_index += 1

    # Process completed jobs and refill
    while pending:
        completed_future_id = None

        while completed_future_id is None:
            for fid, job in pending.items():
                if job.future.done():
                    completed_future_id = fid
                    break
            if completed_future_id is None:
                time.sleep(0.05)

        job = pending.pop(completed_future_id)
        completed_count += 1

        try:
            sampleset = job.future.sampleset
            elapsed = time.time() - job.submit_time
            result = extract_result(
                sampleset, job.num_reads, job.annealing_time, elapsed
            )
            result['interval'] = job.interval_seconds

            key = (job.seed, job.interval_seconds)
            if key not in qpu_results:
                qpu_results[key] = []
            qpu_results[key].append(result)

            global_idx = test_counter[0] + completed_count
            print(f"[{global_idx:4d}/{total_tests}] Seed={job.seed}, "
                  f"num_reads={job.num_reads}, "
                  f"annealing_time={job.annealing_time}μs | "
                  f"energy_min={result['energy_min']:.1f}, "
                  f"time={elapsed:.1f}s | "
                  f"in_flight={len(pending)}", flush=True)

            # Incremental save
            output_data = output_data_builder()
            save_results_incremental(output_file, output_data)

        except Exception as e:
            global_idx = test_counter[0] + completed_count
            print(f"[{global_idx:4d}/{total_tests}] ⚠️  Error: Seed={job.seed}, "
                  f"num_reads={job.num_reads}, "
                  f"annealing_time={job.annealing_time}μs: {e}",
                  flush=True)

        # Refill queue
        while len(pending) < queue_depth and work_index < total_work:
            try:
                job = submit_job(work_index)
                pending[id(job.future)] = job
                work_index += 1
            except Exception as e:
                print(f"\n  ⚠️  Submit error (work_index={work_index}): {e}")
                work_index += 1

    test_counter[0] += total_work


def run_sync(qpu_miner, nonces_and_models, num_reads_list,
             annealing_time_list, interval_seconds, qpu_results,
             total_tests, test_counter, output_file, output_data_builder):
    """Run QPU tests synchronously with sleep intervals between queries."""
    work_items = []
    skipped_combos = 0
    for num_reads in num_reads_list:
        for annealing_time in annealing_time_list:
            if exceeds_qpu_time_limit(num_reads, annealing_time):
                skipped_combos += 1
                continue
            for seed, nonce, model in nonces_and_models:
                work_items.append((seed, nonce, model, num_reads, annealing_time))

    if skipped_combos:
        skipped_jobs = skipped_combos * len(nonces_and_models)
        print(f"  Skipped {skipped_combos} param combo(s) exceeding "
              f"{QPU_MAX_ACCESS_TIME_US / 1e6:.0f}s QPU limit "
              f"({skipped_jobs} jobs)")

    for i, (seed, nonce, model, num_reads, annealing_time) in enumerate(work_items):
        test_counter[0] += 1
        idx = test_counter[0]

        print(f"[{idx}/{total_tests}] Seed={seed}, num_reads={num_reads}, "
              f"annealing_time={annealing_time}μs, interval={interval_seconds}s...",
              end='', flush=True)

        try:
            result = run_qpu_test(
                qpu_miner, model['h'], model['J'],
                nonce, num_reads, annealing_time
            )
            result['interval'] = interval_seconds

            key = (seed, interval_seconds)
            if key not in qpu_results:
                qpu_results[key] = []
            qpu_results[key].append(result)

            print(f" energy_min={result['energy_min']:.1f}, "
                  f"time={result['time']:.3f}s")

            # Incremental save
            output_data = output_data_builder()
            save_results_incremental(output_file, output_data)

        except Exception as e:
            print(f" ⚠️  Error: {e}")

        if interval_seconds > 0 and i < len(work_items) - 1:
            time.sleep(interval_seconds)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test QPU performance with varying parameters on fixed problem instances'
    )
    parser.add_argument(
        '--seed',
        type=int,
        action='append',
        help='Random seed(s) for reproducible nonce generation. Can be specified multiple times.'
    )
    parser.add_argument(
        '--num-reads',
        type=str,
        default='32,64,128,256,512,1024,2048',
        help='Comma-separated list of num_reads values to test (default: 32,64,128,256,512,1024,2048)'
    )
    parser.add_argument(
        '--annealing-time',
        type=str,
        default='5,10,20,40,80,160,320',
        help='Comma-separated list of annealing times in microseconds (default: 5,10,20,40,80,160,320)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='0',
        help='Comma-separated sleep durations between QPU queries (default: 0). Examples: "0,5s,10s", "1m,5m,10m"'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Topology name (default: DEFAULT_TOPOLOGY=Z(9,2)). Examples: "Z(9,2)", "Advantage2_system1.12"'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--cpu-baseline-sweeps',
        type=int,
        default=64,
        help='Number of sweeps for CPU baseline (default: 64)'
    )
    parser.add_argument(
        '--cpu-baseline-reads',
        type=int,
        default=32,
        help='Number of reads for CPU baseline (default: 32)'
    )
    parser.add_argument(
        '--queue-depth',
        type=int,
        default=30,
        help='Number of QPU jobs to keep in-flight for async streaming (default: 30). '
             'Only used when interval=0.'
    )

    args = parser.parse_args()

    if not args.seed:
        print("❌ At least one --seed must be specified")
        return 1

    seeds = args.seed

    try:
        num_reads_list = parse_int_list(args.num_reads)
        annealing_time_list = parse_float_list(args.annealing_time)
        interval_list = parse_duration_list(args.interval)
    except (ValueError, IndexError) as e:
        print(f"❌ Failed to parse parameters: {e}")
        return 1

    if args.topology:
        try:
            topology = load_topology(args.topology)
            print(f"✅ Loaded topology: {topology.solver_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Failed to load topology '{args.topology}': {e}")
            return 1
    else:
        topology = DEFAULT_TOPOLOGY

    queue_depth = args.queue_depth

    print("🔬 QPU Parameter Testing Tool")
    print("=" * 60)
    print(f"Topology: {topology.solver_name} ({len(topology.nodes)} nodes, {len(topology.edges)} edges)")
    print(f"Seeds: {seeds}")
    print(f"num_reads values: {num_reads_list}")
    print(f"annealing_time values: {annealing_time_list}")
    print(f"Interval values: {interval_list}")
    print(f"Queue depth: {queue_depth} (async streaming for interval=0)")
    print(f"CPU baseline: {args.cpu_baseline_sweeps} sweeps, {args.cpu_baseline_reads} reads")
    print()

    # Determine output file early for incremental saves
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"qpu_test_{timestamp}.json"

    # Initialize miners
    print("Initializing miners...")
    from CPU.sa_miner import SimulatedAnnealingMiner
    from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
    from QPU.dwave_miner import DWaveMiner

    cpu_sampler = SimulatedAnnealingStructuredSampler(topology=topology)
    cpu_miner = SimulatedAnnealingMiner(miner_id="qpu-test-cpu", sampler=cpu_sampler)
    qpu_miner = DWaveMiner(miner_id="qpu-test-qpu", topology=topology, qpu_timeout=0.0)

    print("✅ CPU miner initialized")
    print("✅ QPU miner initialized")
    print()

    # Generate nonces for all seeds
    print(f"Generating {len(seeds)} problem instance(s)...")
    nonces_and_models = []
    for seed in seeds:
        nonce, model = generate_nonce(seed, topology)
        nonces_and_models.append((seed, nonce, model))
        nonce_hex = format(nonce, '064x')
        print(f"  Seed {seed}: nonce={nonce_hex[:16]}...")
    print()

    # Run CPU baseline for each nonce
    print(f"Running CPU baseline ({args.cpu_baseline_sweeps} sweeps, {args.cpu_baseline_reads} reads)...")
    cpu_baselines = {}
    for seed, nonce, model in nonces_and_models:
        print(f"  Seed {seed}...", end='', flush=True)
        result = run_cpu_baseline(
            cpu_miner,
            model['h'],
            model['J'],
            args.cpu_baseline_sweeps,
            args.cpu_baseline_reads
        )
        cpu_baselines[seed] = result
        print(f" energy_min={result['energy_min']:.1f}, time={result['time']:.3f}s")
    print()

    # Build list of all QPU tests, accounting for skipped combos
    skipped_combos = sum(
        1 for nr in num_reads_list for at in annealing_time_list
        if exceeds_qpu_time_limit(nr, at)
    )
    valid_combos = len(num_reads_list) * len(annealing_time_list) - skipped_combos
    total_tests = valid_combos * len(nonces_and_models) * len(interval_list)
    print(f"Running {total_tests} QPU tests...")
    print(f"  {len(num_reads_list)} num_reads × {len(annealing_time_list)} annealing_time × {len(interval_list)} interval × {len(seeds)} seed(s)")
    if skipped_combos:
        skipped_jobs = skipped_combos * len(nonces_and_models) * len(interval_list)
        print(f"  {skipped_combos} param combo(s) exceed {QPU_MAX_ACCESS_TIME_US / 1e6:.0f}s QPU limit ({skipped_jobs} jobs skipped)")

    # Show mode per interval
    for interval in interval_list:
        mode = f"async streaming (queue_depth={queue_depth})" if interval == 0 else "sync (sequential)"
        print(f"  interval={interval}s: {mode}")
    print()

    qpu_results = {}
    # Mutable counter so nested functions can update it
    test_counter = [0]

    def make_output_data():
        return build_output_data(
            seeds, topology, num_reads_list, annealing_time_list,
            interval_list, args.cpu_baseline_sweeps, args.cpu_baseline_reads,
            cpu_baselines, qpu_results, queue_depth
        )

    # Run tests per interval
    for interval_seconds in interval_list:
        if interval_seconds == 0 and queue_depth > 1:
            print(f"\n{'=' * 60}")
            print(f"STREAMING: interval=0, queue_depth={queue_depth}")
            print(f"{'=' * 60}")
            run_streaming(
                qpu_miner, nonces_and_models, num_reads_list,
                annealing_time_list, interval_seconds, qpu_results,
                queue_depth, total_tests, test_counter, output_file,
                make_output_data
            )
        else:
            print(f"\n{'=' * 60}")
            print(f"SYNC: interval={interval_seconds}s")
            print(f"{'=' * 60}")
            run_sync(
                qpu_miner, nonces_and_models, num_reads_list,
                annealing_time_list, interval_seconds, qpu_results,
                total_tests, test_counter, output_file, make_output_data
            )

    print()
    print("=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    # Print summary for each seed and interval combination
    for seed in seeds:
        for interval_seconds in interval_list:
            key = (seed, interval_seconds)
            if key not in qpu_results:
                continue

            print(f"\n{'=' * 80}")
            print(f"Seed {seed} | Interval: {interval_seconds}s")
            print(f"CPU Baseline: energy_min={cpu_baselines[seed]['energy_min']:.1f}, "
                  f"energy_mean={cpu_baselines[seed]['energy_mean']:.1f}")
            print(f"{'=' * 80}")

            results_dict = {}
            for result in qpu_results[key]:
                lookup_key = (result['num_reads'], result['annealing_time'])
                results_dict[lookup_key] = result['energy_min']

            header = f"{'num_reads':<12}"
            for at in annealing_time_list:
                header += f"{at:>10.0f}μs"
            print(header)
            print("-" * (12 + 10 * len(annealing_time_list)))

            for nr in num_reads_list:
                row = f"{nr:<12}"
                for at in annealing_time_list:
                    energy = results_dict.get((nr, at))
                    if energy is not None:
                        row += f"{energy:>10.1f}"
                    else:
                        row += f"{'N/A':>10}"
                print(row)

            print()

            qpu_energies = [r['energy_min'] for r in qpu_results[key]]
            print(f"QPU Statistics ({len(qpu_results[key])} tests):")
            print(f"  Best energy: {min(qpu_energies):.1f}")
            print(f"  Worst energy: {max(qpu_energies):.1f}")
            print(f"  Mean energy: {np.mean(qpu_energies):.1f}")
            print(f"  Std dev: {np.std(qpu_energies):.1f}")

            best_result = min(qpu_results[key], key=lambda r: r['energy_min'])
            print(f"  Best params: num_reads={best_result['num_reads']}, "
                  f"annealing_time={best_result['annealing_time']}μs")

    # Final save
    output_data = make_output_data()
    save_results_incremental(output_file, output_data)

    total_results = sum(len(v) for v in qpu_results.values())
    print(f"\n💾 Results saved to {output_file} ({total_results} results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
