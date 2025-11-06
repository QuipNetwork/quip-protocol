#!/usr/bin/env python3
"""Compare mining rates across different hardware at fixed difficulty."""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

from shared.block import Block, BlockHeader, BlockRequirements, create_genesis_block
from shared.miner_types import MiningResult
from shared.time_utils import utc_timestamp


@dataclass
class NodeInfo:
    """Simple node info for testing."""
    miner_id: str


def mine_continuous(
    miner,
    difficulty_energy: float,
    duration_minutes: float,
    min_diversity: float = 0.3,
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

    # Progress tracking
    last_progress_time = start_time
    progress_interval = 60  # Print progress every minute

    log(f"\n⏱️  Mining started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    log(f"   Will run until {time.strftime('%H:%M:%S', time.localtime(start_time + duration_seconds))}")

    # Create genesis block as prev_block
    prev_block = create_genesis_block()
    # Update requirements for testing
    prev_block.next_block_requirements = requirements

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            log(f"\n✅ Duration limit reached ({duration_minutes} min)")
            break

        # Progress update
        if time.time() - last_progress_time >= progress_interval:
            elapsed_min = elapsed / 60
            blocks_per_min = len(blocks_found) / elapsed_min if elapsed_min > 0 else 0
            log(f"   [{elapsed_min:.1f}/{duration_minutes:.0f} min] "
                  f"Blocks: {len(blocks_found)}, "
                  f"Attempts: {attempts}, "
                  f"Rate: {blocks_per_min:.2f} blocks/min")
            last_progress_time = time.time()

        attempts += 1
        prev_timestamp = prev_block.header.timestamp

        # Create stop event
        stop_event = multiprocessing.Event()

        # Mine the block
        attempt_start = time.time()
        result = miner.mine_block(
            prev_block=prev_block,
            node_info=node_info,
            requirements=requirements,
            prev_timestamp=prev_timestamp,
            stop_event=stop_event
        )
        attempt_time = time.time() - attempt_start

        if result:
            blocks_found.append(result)
            log(f"   ✅ Block {len(blocks_found)} found! "
                  f"Energy: {result.energy:.1f}, "
                  f"Time: {attempt_time:.1f}s, "
                  f"Diversity: {result.diversity:.3f}, "
                  f"Solutions: {result.num_valid}")

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
        }
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
        type=float,
        default=10.0,
        help='Mining duration in minutes (default: 10.0)'
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.3,
        help='Minimum solution diversity (default: 0.3)'
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
        help='QPU topology name (for qpu miner, e.g., "Z(10,2)")'
    )

    args = parser.parse_args()

    print("🔬 Mining Rate Comparison Tool")
    print("=" * 50)
    print(f"Miner type: {args.miner_type.upper()}")
    print(f"Difficulty: {args.difficulty_energy:.1f}")
    print(f"Duration: {args.duration} minutes")

    # Initialize miner
    miner = None
    if args.miner_type == 'cpu':
        from CPU.sa_miner import SimulatedAnnealingMiner
        miner = SimulatedAnnealingMiner(miner_id="rate-test-cpu")
    elif args.miner_type == 'cuda':
        from GPU.cuda_miner import CudaMiner
        miner = CudaMiner(miner_id="rate-test-cuda", device=args.device)
    elif args.miner_type == 'metal':
        from GPU.metal_miner import MetalMiner
        miner = MetalMiner(miner_id="rate-test-metal")
    elif args.miner_type == 'qpu':
        from QPU.dwave_miner import DWaveMiner
        miner = DWaveMiner(miner_id="rate-test-qpu", topology_name=args.topology, qpu_timeout=0.0)

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
            duration_minutes=args.duration,
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

    # Save results
    output_data = {
        'miner_type': args.miner_type,
        'difficulty_energy': args.difficulty_energy,
        'duration_minutes': args.duration,
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
