#!/usr/bin/env python3
"""Test QPU performance with varying parameters on fixed problem instances.

This tool tests how QPU parameters affect solution quality by:
1. Generating fixed problem instances from seed(s)
2. Running CPU canary baseline (64 sweeps, 32 reads)
3. Running QPU with varying num_reads and annealing_time parameters
4. Optionally interleaving multiple seeds to prevent QPU from settling on known solutions
5. Allowing configurable intervals between QPU queries

The goal is to understand the relationship between QPU parameters and solution quality.
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    """
    Parse duration string to seconds.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 30.0 (seconds)
        "5m" -> 300.0
        "2h" -> 7200.0
        "1d" -> 86400.0
        "1w" -> 604800.0
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
        # Try parsing as raw seconds
        return float(duration_str)


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in s.split(',')]


def parse_duration_list(s: str) -> List[float]:
    """Parse comma-separated duration strings to list of seconds.

    Examples:
        "0,5s,10s" -> [0.0, 5.0, 10.0]
        "1m,5m,10m" -> [60.0, 300.0, 600.0]
    """
    return [parse_duration(x.strip()) for x in s.split(',')]


def generate_nonce(seed: int, topology) -> Tuple[str, Dict]:
    """Generate a nonce and Ising model from a seed.

    Args:
        seed: Random seed for reproducible nonce generation
        topology: Topology to use for model generation

    Returns:
        Tuple of (nonce, model_dict) where model_dict contains h, J
    """
    # Initialize random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create genesis block as prev_block
    prev_block = create_genesis_block()

    # Generate random salt for nonce generation (reproducible via seed)
    salt = random.randbytes(32)

    # Generate nonce deterministically
    nonce = ising_nonce_from_block(
        prev_block.hash,
        f"qpu-test-{seed}",
        1,  # block index
        salt
    )

    # Generate Ising model
    nodes = list(topology.nodes)
    edges = list(topology.edges)
    h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

    return nonce, {'h': h, 'J': J, 'salt': salt.hex()}


def run_cpu_baseline(cpu_miner, h, J, num_sweeps: int, num_reads: int) -> Dict:
    """Run CPU baseline test.

    Args:
        cpu_miner: CPU SA miner instance
        h, J: Ising model parameters
        num_sweeps: Number of sweeps for SA
        num_reads: Number of reads

    Returns:
        Dictionary with energy, time, and samples
    """
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


def run_qpu_test(qpu_miner, h, J, nonce: int, num_reads: int, annealing_time: float) -> Dict:
    """Run QPU test with specific parameters.

    Args:
        qpu_miner: QPU miner instance
        h, J: Ising model parameters
        nonce: Nonce value (for job labeling)
        num_reads: Number of reads
        annealing_time: Annealing time in microseconds

    Returns:
        Dictionary with energy, time, and QPU timing info
    """
    start_time = time.time()

    # Generate job label with topology and nonce (matching dwave_miner.py behavior)
    topology_label = qpu_miner.sampler.job_label  # e.g., "Quip_Z9_T2"
    nonce_hex = hex(nonce)[2:][:8]  # First 8 hex chars of nonce
    job_label = f"{topology_label}_{nonce_hex}"

    sampleset = qpu_miner.sampler.sample_ising(
        h, J,
        num_reads=num_reads,
        annealing_time=annealing_time,
        label=job_label
    )
    elapsed = time.time() - start_time

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

    # Add QPU timing info if available
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

    args = parser.parse_args()

    # Validate seeds
    if not args.seed:
        print("❌ At least one --seed must be specified")
        return 1

    seeds = args.seed

    # Parse parameter lists
    try:
        num_reads_list = parse_int_list(args.num_reads)
        annealing_time_list = parse_float_list(args.annealing_time)
        interval_list = parse_duration_list(args.interval)
    except (ValueError, IndexError) as e:
        print(f"❌ Failed to parse parameters: {e}")
        return 1

    # Parse topology if specified
    if args.topology:
        try:
            topology = load_topology(args.topology)
            print(f"✅ Loaded topology: {topology.solver_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Failed to load topology '{args.topology}': {e}")
            return 1
    else:
        topology = DEFAULT_TOPOLOGY

    print("🔬 QPU Parameter Testing Tool")
    print("=" * 60)
    print(f"Topology: {topology.solver_name} ({len(topology.nodes)} nodes, {len(topology.edges)} edges)")
    print(f"Seeds: {seeds}")
    print(f"num_reads values: {num_reads_list}")
    print(f"annealing_time values: {annealing_time_list}")
    print(f"Interval values: {interval_list}")
    print(f"CPU baseline: {args.cpu_baseline_sweeps} sweeps, {args.cpu_baseline_reads} reads")
    print()

    # Initialize miners
    print("Initializing miners...")
    from CPU.sa_miner import SimulatedAnnealingMiner
    from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
    from QPU.dwave_miner import DWaveMiner

    # Create CPU sampler with matching topology
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
        # nonce is an int, convert to hex for display
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

    # Build list of all QPU tests (interleaved by seed)
    total_tests = len(nonces_and_models) * len(num_reads_list) * len(annealing_time_list) * len(interval_list)
    print(f"Running {total_tests} QPU tests...")
    print(f"  {len(num_reads_list)} num_reads × {len(annealing_time_list)} annealing_time × {len(interval_list)} interval × {len(seeds)} seed(s)")
    print()

    qpu_results = {}
    test_num = 0

    # Interleave tests: for each (num_reads, annealing_time, interval) triple, cycle through all seeds
    for interval_seconds in interval_list:
        for num_reads in num_reads_list:
            for annealing_time in annealing_time_list:
                for seed, nonce, model in nonces_and_models:
                    test_num += 1

                    print(f"[{test_num}/{total_tests}] Seed={seed}, num_reads={num_reads}, "
                          f"annealing_time={annealing_time}μs, interval={interval_seconds}s...", end='', flush=True)

                    # Run QPU test
                    result = run_qpu_test(
                        qpu_miner,
                        model['h'],
                        model['J'],
                        nonce,
                        num_reads,
                        annealing_time
                    )
                    result['interval'] = interval_seconds

                    # Store result
                    key = (seed, interval_seconds)
                    if key not in qpu_results:
                        qpu_results[key] = []
                    qpu_results[key].append(result)

                    print(f" energy_min={result['energy_min']:.1f}, time={result['time']:.3f}s")

                    # Sleep between QPU queries if interval specified
                    if interval_seconds > 0 and test_num < total_tests:
                        time.sleep(interval_seconds)

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

            # Build table: rows=num_reads, cols=annealing_time
            # Create dictionary for quick lookup
            results_dict = {}
            for result in qpu_results[key]:
                lookup_key = (result['num_reads'], result['annealing_time'])
                results_dict[lookup_key] = result['energy_min']

            # Print header
            header = f"{'num_reads':<12}"
            for at in annealing_time_list:
                header += f"{at:>10.0f}μs"
            print(header)
            print("-" * (12 + 10 * len(annealing_time_list)))

            # Print rows
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

            # Print overall statistics
            qpu_energies = [r['energy_min'] for r in qpu_results[key]]
            print(f"QPU Statistics ({len(qpu_results[key])} tests):")
            print(f"  Best energy: {min(qpu_energies):.1f}")
            print(f"  Worst energy: {max(qpu_energies):.1f}")
            print(f"  Mean energy: {np.mean(qpu_energies):.1f}")
            print(f"  Std dev: {np.std(qpu_energies):.1f}")

            # Find best parameters
            best_result = min(qpu_results[key], key=lambda r: r['energy_min'])
            print(f"  Best params: num_reads={best_result['num_reads']}, "
                  f"annealing_time={best_result['annealing_time']}μs")

    # Save results - need to flatten the nested dict structure for JSON
    qpu_results_serializable = {}
    for (seed, interval_seconds), results in qpu_results.items():
        key_str = f"seed_{seed}_interval_{interval_seconds}"
        qpu_results_serializable[key_str] = {
            'seed': seed,
            'interval': interval_seconds,
            'results': results
        }

    output_data = {
        'seeds': seeds,
        'topology': topology.solver_name,
        'num_reads_tested': num_reads_list,
        'annealing_time_tested': annealing_time_list,
        'interval_tested': interval_list,
        'cpu_baseline': {
            'num_sweeps': args.cpu_baseline_sweeps,
            'num_reads': args.cpu_baseline_reads,
            'results': cpu_baselines
        },
        'qpu_results': qpu_results_serializable,
        'timestamp': time.time()
    }

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"qpu_test_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
