#!/usr/bin/env python3
"""Compare mining rates: cpu vs cpu-filtered at fixed difficulty.

Spawns parallel workers using the same multiprocessing pattern as
compare_mining_rates.py. Measures blocks/minute, success rate, and
energy distributions for each miner type.
"""
import argparse
import json
import logging
import multiprocessing
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set spawn method before any other multiprocessing code
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from dwave_topologies import DEFAULT_TOPOLOGY
from shared.block import BlockRequirements, create_genesis_block
from shared.time_utils import utc_timestamp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class NodeInfo:
    """Simple node info for testing."""

    miner_id: str


def parse_duration(duration_str: str) -> float:
    """Parse duration string to minutes.

    Supports: 30s, 5m, 2h, 1d, 1w, or raw minute values.
    """
    duration_str = duration_str.strip().lower()
    if duration_str.endswith('s'):
        return int(duration_str[:-1]) / 60.0
    elif duration_str.endswith('m'):
        return float(duration_str[:-1])
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 60.0
    elif duration_str.endswith('d'):
        return int(duration_str[:-1]) * 1440.0
    elif duration_str.endswith('w'):
        return int(duration_str[:-1]) * 10080.0
    return float(duration_str)


def aggregate_results(
    miner_results: List[Dict],
    total_time: float,
) -> Dict:
    """Aggregate results from all miners into unified statistics."""
    total_blocks = sum(r.get('blocks_found', 0) for r in miner_results)
    total_attempts = sum(r.get('attempts', 0) for r in miner_results)

    all_energies: List[float] = []
    all_diversities: List[float] = []
    all_solution_counts: List[int] = []
    all_mining_times: List[float] = []

    for r in miner_results:
        all_energies.extend(r.get('energies', []))
        all_diversities.extend(r.get('diversities', []))
        all_solution_counts.extend(r.get('solution_counts', []))
        all_mining_times.extend(r.get('mining_times', []))

    def _stats(vals):
        if not vals:
            return {'min': None, 'max': None, 'mean': None}
        return {
            'min': min(vals),
            'max': max(vals),
            'mean': sum(vals) / len(vals),
        }

    return {
        'num_miners': len(miner_results),
        'per_miner_stats': miner_results,
        'total_blocks_found': total_blocks,
        'total_attempts': total_attempts,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'success_rate': (
            total_blocks / total_attempts if total_attempts > 0 else 0
        ),
        'blocks_per_minute': (
            total_blocks / (total_time / 60) if total_time > 0 else 0
        ),
        'energy_stats': {
            **_stats(all_energies),
            'all_energies': all_energies,
        },
        'diversity_stats': _stats(all_diversities),
        'solution_count_stats': _stats(all_solution_counts),
        'mining_time_stats': {
            **_stats(all_mining_times),
            'all_times': all_mining_times,
        },
    }


def mine_worker(
    miner_spec: Dict,
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    result_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
):
    """Worker function for parallel mining.

    Args:
        miner_spec: Dict with 'kind' and 'id' for miner creation.
        difficulty_energy: Fixed difficulty threshold.
        min_diversity: Minimum solution diversity.
        min_solutions: Minimum solutions required.
        result_queue: Queue to send results back.
        stop_event: Shared event to signal stop.
    """
    kind = miner_spec['kind']
    miner_id = miner_spec['id']
    topology = DEFAULT_TOPOLOGY

    try:
        if kind == 'cpu':
            from CPU.sa_miner import SimulatedAnnealingMiner
            miner = SimulatedAnnealingMiner(
                miner_id=miner_id, topology=topology,
            )
        elif kind == 'cpu-filtered':
            from CPU.sa_filtered_miner import SAFilteredMiner
            miner = SAFilteredMiner(
                miner_id=miner_id, topology=topology,
            )
        else:
            result_queue.put({
                'error': f'Unknown miner kind: {kind}',
                'miner_id': miner_id,
            })
            return
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({'error': str(e), 'miner_id': miner_id})
        return

    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=0,
    )

    node_info = NodeInfo(miner_id=miner_id)
    blocks_found = []
    attempts = 0
    start_time = time.time()
    prev_block = create_genesis_block()
    prev_block.next_block_requirements = requirements

    last_progress_time = start_time
    progress_interval = 60

    def submit_results():
        total_time = time.time() - start_time
        result_queue.put({
            'miner_id': miner_id,
            'miner_type': kind,
            'blocks_found': len(blocks_found),
            'attempts': attempts,
            'total_time': total_time,
            'energies': [b.energy for b in blocks_found],
            'diversities': [b.diversity for b in blocks_found],
            'solution_counts': [b.num_valid for b in blocks_found],
            'mining_times': [
                b.mining_time for b in blocks_found if b.mining_time
            ],
        })

    while not stop_event.is_set():
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed_min = (current_time - start_time) / 60
            rate = (
                len(blocks_found) / elapsed_min
                if elapsed_min > 0
                else 0
            )
            print(
                f"   [{miner_id}] {elapsed_min:.1f} min, "
                f"Blocks: {len(blocks_found)}, "
                f"Attempts: {attempts}, "
                f"Rate: {rate:.2f}/min",
            )
            last_progress_time = current_time

        attempts += 1
        result = miner.mine_block(
            prev_block=prev_block,
            node_info=node_info,
            requirements=requirements,
            prev_timestamp=prev_block.header.timestamp,
            stop_event=stop_event,
        )

        if result:
            blocks_found.append(result)
            print(
                f"   [{miner_id}] Block {len(blocks_found)}! "
                f"Energy: {result.energy:.1f}, "
                f"Diversity: {result.diversity:.3f}, "
                f"Solutions: {result.num_valid}",
            )

        submit_results()

    submit_results()


def build_specs(
    miner_type: str,
    num_cpus: int,
) -> List[Dict]:
    """Build miner spec dicts for the requested type(s).

    Args:
        miner_type: 'cpu', 'cpu-filtered', or 'both'.
        num_cpus: Number of workers per miner type.

    Returns:
        List of spec dicts with 'kind' and 'id'.
    """
    specs = []
    types = (
        ['cpu', 'cpu-filtered']
        if miner_type == 'both'
        else [miner_type]
    )
    for kind in types:
        for i in range(num_cpus):
            specs.append({
                'kind': kind,
                'id': f'{kind}-{i}',
            })
    return specs


def run_benchmark(args) -> int:
    """Run the mining rate benchmark."""
    try:
        duration_minutes = parse_duration(args.duration)
    except (ValueError, IndexError):
        print(f"Invalid duration: '{args.duration}'")
        return 1

    specs = build_specs(args.miner_type, args.num_cpus)
    if not specs:
        print("No miners configured")
        return 1

    print("Prefilter Mining Rate Benchmark")
    print("=" * 50)
    print(f"Miner type: {args.miner_type}")
    print(f"Difficulty: {args.difficulty_energy:.1f}")
    print(f"Duration: {args.duration} ({duration_minutes:.1f} min)")
    print(f"Workers per type: {args.num_cpus}")
    print(f"\nSpawning {len(specs)} worker(s):")
    for s in specs:
        print(f"   - {s['id']}")

    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    processes = []
    for spec in specs:
        p = multiprocessing.Process(
            target=mine_worker,
            args=(
                spec,
                args.difficulty_energy,
                args.min_diversity,
                args.min_solutions,
                result_queue,
                stop_event,
            ),
        )
        p.start()
        processes.append(p)

    print(f"\nStarted at {time.strftime('%H:%M:%S')}")
    print(f"Running for {duration_minutes:.1f} minutes...")

    def timer_thread():
        time.sleep(duration_minutes * 60)
        print(f"\nDuration reached ({args.duration})")
        stop_event.set()

    timer = threading.Thread(target=timer_thread, daemon=True)
    timer.start()

    start_time = time.time()
    results_by_id: Dict[str, Dict] = {}

    def drain_queue():
        while True:
            try:
                r = result_queue.get_nowait()
                mid = r.get('miner_id')
                if mid:
                    results_by_id[mid] = r
            except Exception:
                break

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
            drain_queue()
            if stop_event.is_set():
                shutdown_start = time.time()
                while any(p.is_alive() for p in processes):
                    drain_queue()
                    if time.time() - shutdown_start > 180:
                        print("   Timeout, forcing shutdown...")
                        drain_queue()
                        for p in processes:
                            if p.is_alive():
                                p.terminate()
                        for p in processes:
                            p.join(timeout=2.0)
                            if p.is_alive():
                                p.kill()
                        break
                    time.sleep(0.5)
                break
        drain_queue()
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...")
        stop_event.set()
        shutdown_start = time.time()
        while any(p.is_alive() for p in processes):
            drain_queue()
            if time.time() - shutdown_start > 60:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                for p in processes:
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.kill()
                break
            time.sleep(0.5)
        drain_queue()

    total_time = time.time() - start_time
    all_results = list(results_by_id.values())

    errors = [r for r in all_results if 'error' in r]
    for err in errors:
        print(f"ERROR {err.get('miner_id')}: {err['error']}")

    valid = [r for r in all_results if 'error' not in r]

    # Group by miner type for side-by-side comparison
    by_type: Dict[str, List[Dict]] = {}
    for r in valid:
        mt = r['miner_type']
        by_type.setdefault(mt, []).append(r)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    type_stats = {}
    for mt, results in sorted(by_type.items()):
        stats = aggregate_results(results, total_time)
        type_stats[mt] = stats

        print(f"\n--- {mt.upper()} ---")
        print(f"  Miners: {stats['num_miners']}")
        print(f"  Blocks: {stats['total_blocks_found']}")
        print(f"  Attempts: {stats['total_attempts']}")
        print(
            f"  Success rate: "
            f"{stats['success_rate'] * 100:.1f}%",
        )
        print(f"  Rate: {stats['blocks_per_minute']:.3f} blocks/min")

        if stats['total_blocks_found'] > 0:
            es = stats['energy_stats']
            print(f"  Energy: min={es['min']:.1f}, "
                  f"max={es['max']:.1f}, mean={es['mean']:.1f}")
            ds = stats['diversity_stats']
            print(f"  Diversity: min={ds['min']:.3f}, "
                  f"max={ds['max']:.3f}, mean={ds['mean']:.3f}")

    # Comparison summary when running both
    if len(type_stats) == 2 and all(
        s['blocks_per_minute'] > 0 for s in type_stats.values()
    ):
        cpu_rate = type_stats.get('cpu', {}).get('blocks_per_minute', 0)
        filt_rate = type_stats.get(
            'cpu-filtered', {},
        ).get('blocks_per_minute', 0)
        if cpu_rate > 0:
            speedup = filt_rate / cpu_rate
            print(f"\n--- COMPARISON ---")
            print(f"  cpu-filtered / cpu rate: {speedup:.2f}x")

    # Save JSON
    output_data = {
        'miner_type': args.miner_type,
        'difficulty_energy': args.difficulty_energy,
        'duration_spec': args.duration,
        'duration_minutes': duration_minutes,
        'min_diversity': args.min_diversity,
        'min_solutions': args.min_solutions,
        'per_type_stats': type_stats,
        'errors': [e.get('error') for e in errors],
        'timestamp': utc_timestamp(),
    }

    output_file = args.output
    if not output_file:
        ts = int(time.time())
        output_file = (
            f"prefilter_mining_{args.miner_type}_{ts}.json"
        )

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_file}")
    return 0 if valid else 1


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            'Compare mining rates: cpu vs cpu-filtered '
            'at fixed difficulty'
        ),
    )
    parser.add_argument(
        '--miner-type',
        type=str,
        choices=['cpu', 'cpu-filtered', 'both'],
        default='both',
        help='Miner type to test (default: both)',
    )
    parser.add_argument(
        '--difficulty-energy',
        type=float,
        default=-14900.0,
        help='Fixed difficulty energy threshold (default: -14900.0)',
    )
    parser.add_argument(
        '--duration',
        type=str,
        default='10m',
        help='Mining duration (default: 10m). Examples: 30s, 5m, 2h',
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.15,
        help='Minimum solution diversity (default: 0.15)',
    )
    parser.add_argument(
        '--min-solutions',
        type=int,
        default=5,
        help='Minimum number of solutions (default: 5)',
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=2,
        help='Number of workers per miner type (default: 2)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results',
    )
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
