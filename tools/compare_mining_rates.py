#!/usr/bin/env python3
"""Compare mining rates across different hardware at fixed difficulty."""
import argparse
import json
import logging
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging to capture miner logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # Force reconfiguration if already configured
)

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from dataclasses import dataclass

from shared.block import Block, BlockHeader, BlockRequirements, create_genesis_block
from shared.miner_types import MiningResult
from shared.time_utils import utc_timestamp
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import load_topology


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to minutes.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 0.5 (minutes)
        "5m" -> 5.0
        "2h" -> 120.0
        "1d" -> 1440.0
        "1w" -> 10080.0
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
    else:
        # Try parsing as raw minutes
        return float(duration_str)


@dataclass
class NodeInfo:
    """Simple node info for testing."""
    miner_id: str


def mine_continuous(
    miner,
    difficulty_energy: float,
    duration_minutes: float,
    min_diversity: float = 0.15,
    min_solutions: int = 5,
    log_file = None
) -> Dict:
    """Mine continuously for specified duration at fixed difficulty.

    Args:
        miner: Miner instance (CPU, CUDA, QPU, etc.)
        difficulty_energy: Fixed difficulty energy threshold
        duration_minutes: How long to mine (in minutes)
        min_diversity: Minimum solution diversity requirement
        min_solutions: Minimum number of solutions required
        log_file: Optional file object to write logs to

    Returns:
        Dictionary with mining statistics
    """
    def log(msg):
        """Print to console and optionally to log file."""
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"\n⛏️  Starting continuous mining:")
    log(f"   Duration: {duration_minutes} minutes")
    log(f"   Difficulty: {difficulty_energy:.1f}")
    log(f"   Min diversity: {min_diversity}")
    log(f"   Min solutions: {min_solutions}")

    # Setup
    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=0  # Disable decay
    )

    node_info = NodeInfo(miner_id=f"rate-test-{miner.miner_type}-0")
    blocks_found: List[MiningResult] = []
    attempts = 0
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    # QPU time tracking
    total_qpu_time_us = 0.0  # Total QPU time in microseconds

    # Progress tracking
    last_progress_time = start_time
    progress_interval = 60  # Print progress every minute

    log(f"\n⏱️  Mining started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    log(f"   Will run until {time.strftime('%H:%M:%S', time.localtime(start_time + duration_seconds))}")

    # Create genesis block as prev_block
    prev_block = create_genesis_block()
    # Update requirements for testing
    prev_block.next_block_requirements = requirements

    # Create stop event that persists across mining attempts
    stop_event = multiprocessing.Event()

    # Start a timer thread to set stop_event after duration expires
    def timer_thread():
        time.sleep(duration_seconds)
        log(f"\n✅ Duration limit reached ({duration_minutes} min)")
        stop_event.set()

    timer = threading.Thread(target=timer_thread, daemon=True)
    timer.start()

    while True:
        # Check if stop event was set by timer
        if stop_event.is_set():
            break

        # Progress update
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed = current_time - start_time
            elapsed_min = elapsed / 60
            blocks_per_min = len(blocks_found) / elapsed_min if elapsed_min > 0 else 0
            log(f"   [{elapsed_min:.1f}/{duration_minutes:.0f} min] "
                  f"Blocks: {len(blocks_found)}, "
                  f"Attempts: {attempts}, "
                  f"Rate: {blocks_per_min:.2f} blocks/min")
            last_progress_time = current_time

        attempts += 1
        prev_timestamp = prev_block.header.timestamp

        # Mine the block (using shared stop_event)
        attempt_start = time.time()
        result = miner.mine_block(
            prev_block=prev_block,
            node_info=node_info,
            requirements=requirements,
            prev_timestamp=prev_timestamp,
            stop_event=stop_event
        )
        attempt_time = time.time() - attempt_start

        # Check if stop event was set during mining
        if stop_event.is_set():
            break

        # Track QPU time for this attempt (if available)
        attempt_qpu_time_us = 0.0
        if hasattr(miner, 'timing_stats') and 'qpu_access_time' in miner.timing_stats:
            # Get the most recent QPU access time
            if miner.timing_stats['qpu_access_time']:
                attempt_qpu_time_us = miner.timing_stats['qpu_access_time'][-1]
                total_qpu_time_us += attempt_qpu_time_us

        if result:
            blocks_found.append(result)
            qpu_time_msg = f", QPU: {attempt_qpu_time_us / 1e6:.3f}s, Total QPU: {total_qpu_time_us / 1e6:.2f}s" if attempt_qpu_time_us > 0 else ""
            log(f"   ✅ Block {len(blocks_found)} found! "
                  f"Energy: {result.energy:.1f}, "
                  f"Time: {attempt_time:.1f}s, "
                  f"Diversity: {result.diversity:.3f}, "
                  f"Solutions: {result.num_valid}"
                  f"{qpu_time_msg}")

    total_time = time.time() - start_time

    # Compute statistics
    energies = [b.energy for b in blocks_found]
    diversities = [b.diversity for b in blocks_found]
    solution_counts = [b.num_valid for b in blocks_found]
    mining_times = [b.mining_time for b in blocks_found if b.mining_time]

    stats = {
        'total_blocks_found': len(blocks_found),
        'total_attempts': attempts,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'success_rate': len(blocks_found) / attempts if attempts > 0 else 0,
        'blocks_per_minute': len(blocks_found) / (total_time / 60) if total_time > 0 else 0,
        'energy_stats': {
            'min': min(energies) if energies else None,
            'max': max(energies) if energies else None,
            'mean': sum(energies) / len(energies) if energies else None,
            'all_energies': energies
        },
        'diversity_stats': {
            'min': min(diversities) if diversities else None,
            'max': max(diversities) if diversities else None,
            'mean': sum(diversities) / len(diversities) if diversities else None
        },
        'solution_count_stats': {
            'min': min(solution_counts) if solution_counts else None,
            'max': max(solution_counts) if solution_counts else None,
            'mean': sum(solution_counts) / len(solution_counts) if solution_counts else None
        },
        'mining_time_stats': {
            'min': min(mining_times) if mining_times else None,
            'max': max(mining_times) if mining_times else None,
            'mean': sum(mining_times) / len(mining_times) if mining_times else None,
            'all_times': mining_times
        },
        'qpu_time_stats': {
            'total_qpu_time_us': total_qpu_time_us,
            'total_qpu_time_seconds': total_qpu_time_us / 1e6,
            'avg_qpu_time_per_attempt_us': total_qpu_time_us / attempts if attempts > 0 else 0
        } if total_qpu_time_us > 0 else None
    }

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare mining rates at fixed difficulty'
    )
    parser.add_argument(
        '--miner-type',
        type=str,
        choices=['cpu', 'cuda', 'metal', 'qpu'],
        required=True,
        help='Miner type to test'
    )
    parser.add_argument(
        '--difficulty-energy',
        type=float,
        required=True,
        help='Fixed difficulty energy threshold (e.g., -15450.0)'
    )
    parser.add_argument(
        '--duration',
        type=str,
        default='10m',
        help='Mining duration (default: 10m). Examples: 30s, 5m, 2h, 1d, 1w'
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.15,
        help='Minimum solution diversity (default: 0.15)'
    )
    parser.add_argument(
        '--min-solutions',
        type=int,
        default=5,
        help='Minimum number of solutions (default: 5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device ID (for cuda miner, default: 0)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Topology name (default: DEFAULT_TOPOLOGY=Z(9,2)). Examples: "Z(9,2)", "Z(10,2)", "Advantage2_system1.8"'
    )

    args = parser.parse_args()

    # Parse duration
    try:
        duration_minutes = parse_duration(args.duration)
    except (ValueError, IndexError):
        print(f"❌ Invalid duration format: '{args.duration}'. Use formats like: 30s, 5m, 2h, 1d, 1w")
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

    print("🔬 Mining Rate Comparison Tool")
    print("=" * 50)
    print(f"Miner type: {args.miner_type.upper()}")
    print(f"Topology: {topology.solver_name} ({len(topology.nodes)} nodes, {len(topology.edges)} edges)")
    print(f"Difficulty: {args.difficulty_energy:.1f}")
    print(f"Duration: {args.duration} ({duration_minutes:.1f} minutes)")

    # Initialize miner
    miner = None
    if args.miner_type == 'cpu':
        from CPU.sa_miner import SimulatedAnnealingMiner
        miner = SimulatedAnnealingMiner(miner_id="rate-test-cpu", topology=topology)
    elif args.miner_type == 'cuda':
        from GPU.cuda_miner import CudaMiner
        miner = CudaMiner(miner_id="rate-test-cuda", device=args.device, topology=topology)
    elif args.miner_type == 'metal':
        from GPU.metal_miner import MetalMiner
        miner = MetalMiner(miner_id="rate-test-metal", topology=topology)
    elif args.miner_type == 'qpu':
        from QPU.dwave_miner import DWaveMiner
        miner = DWaveMiner(miner_id="rate-test-qpu", topology=topology, qpu_timeout=0.0)

    if not miner:
        print(f"❌ Failed to initialize {args.miner_type} miner")
        return 1

    print(f"✅ {args.miner_type.upper()} miner initialized")

    # Open log file
    log_file = None
    if args.output:
        log_filename = args.output.replace('.json', '.log')
        log_file = open(log_filename, 'w')
        print(f"📝 Logging to {log_filename}")

    # Run continuous mining
    start_time = time.time()
    try:
        stats = mine_continuous(
            miner=miner,
            difficulty_energy=args.difficulty_energy,
            duration_minutes=duration_minutes,
            min_diversity=args.min_diversity,
            min_solutions=args.min_solutions,
            log_file=log_file
        )
    finally:
        if log_file:
            log_file.close()
    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"✅ Mining completed:")
    print(f"   Total time: {stats['total_time_minutes']:.1f} min")
    print(f"   Blocks found: {stats['total_blocks_found']}")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"   Mining rate: {stats['blocks_per_minute']:.3f} blocks/min")

    if stats['total_blocks_found'] > 0:
        print(f"\n📈 Energy distribution:")
        print(f"   Min: {stats['energy_stats']['min']:.1f}")
        print(f"   Max: {stats['energy_stats']['max']:.1f}")
        print(f"   Mean: {stats['energy_stats']['mean']:.1f}")

        print(f"\n🌈 Diversity distribution:")
        print(f"   Min: {stats['diversity_stats']['min']:.3f}")
        print(f"   Max: {stats['diversity_stats']['max']:.3f}")
        print(f"   Mean: {stats['diversity_stats']['mean']:.3f}")

        if stats['mining_time_stats']['mean']:
            print(f"\n⏱️  Mining time per block:")
            print(f"   Min: {stats['mining_time_stats']['min']:.1f}s")
            print(f"   Max: {stats['mining_time_stats']['max']:.1f}s")
            print(f"   Mean: {stats['mining_time_stats']['mean']:.1f}s")

    # QPU time reporting
    if stats.get('qpu_time_stats'):
        qpu_stats = stats['qpu_time_stats']
        print(f"\n🔮 QPU Time Usage:")
        print(f"   Total QPU time: {qpu_stats['total_qpu_time_seconds']:.2f}s ({qpu_stats['total_qpu_time_us'] / 1e6:.6f}s)")
        print(f"   Average per attempt: {qpu_stats['avg_qpu_time_per_attempt_us'] / 1e6:.4f}s")
        if stats['total_blocks_found'] > 0:
            qpu_time_per_block = qpu_stats['total_qpu_time_seconds'] / stats['total_blocks_found']
            print(f"   Average per block found: {qpu_time_per_block:.2f}s")

    # Save results
    output_data = {
        'miner_type': args.miner_type,
        'difficulty_energy': args.difficulty_energy,
        'duration_spec': args.duration,
        'duration_minutes': duration_minutes,
        'min_diversity': args.min_diversity,
        'min_solutions': args.min_solutions,
        'statistics': stats,
        'timestamp': utc_timestamp()
    }

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"mining_rate_{args.miner_type}_{args.duration}min_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    return 0


if __name__ == "__main__":
    from typing import Tuple
    sys.exit(main())
