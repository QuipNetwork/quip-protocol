#!/usr/bin/env python3
"""Find difficulty threshold where mining takes target time (e.g., 10 minutes)."""
import argparse
import json
import logging
import multiprocessing
import random
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

# Enable unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Configure logging to show miner output
logging.basicConfig(
    level=logging.INFO,
    format='    %(message)s',  # Indent miner logs slightly
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataclasses import dataclass

from shared.block import Block, BlockHeader, BlockRequirements, create_genesis_block
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import calc_energy_range, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from shared.time_utils import utc_timestamp
from dwave_topologies.topologies.json_loader import load_topology


@dataclass
class NodeInfo:
    """Simple node info for testing."""
    miner_id: str


def _mine_with_timeout_worker(miner, prev_block, node_info, requirements, prev_timestamp, stop_event, result_queue):
    """Worker function for mining with timeout (must be top-level for pickling)."""
    import sys
    # Ensure output is flushed immediately
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    result = miner.mine_block(
        prev_block=prev_block,
        node_info=node_info,
        requirements=requirements,
        prev_timestamp=prev_timestamp,
        stop_event=stop_event
    )
    result_queue.put(result)


def test_mining_time(
    miner,
    difficulty_energy: float,
    num_attempts: int = 3,
    timeout_per_attempt: Optional[float] = None
) -> Tuple[float, int]:
    """Test average mining time at given difficulty.

    Args:
        miner: Miner instance
        difficulty_energy: Difficulty threshold to test
        num_attempts: Number of mining attempts
        timeout_per_attempt: Max time per attempt in seconds (None = no timeout)

    Returns:
        (avg_time, successful_attempts)
    """
    timeout_str = f", timeout={timeout_per_attempt:.0f}s" if timeout_per_attempt else ""
    print(f"  Testing difficulty_energy={difficulty_energy:.1f} ({num_attempts} attempts{timeout_str})")

    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=0.2,
        min_solutions=5,
        timeout_to_difficulty_adjustment_decay=0  # Disable decay
    )

    times = []
    node_info = NodeInfo(miner_id=f"test-{miner.miner_type}-0")

    # Create genesis block as prev_block
    prev_block = create_genesis_block()
    # Update requirements for testing
    prev_block.next_block_requirements = requirements

    for attempt in range(num_attempts):
        prev_timestamp = prev_block.header.timestamp

        # Create stop event
        stop_event = multiprocessing.Event()

        # Mine the block with timeout
        start_time = time.time()

        if timeout_per_attempt:
            # Run mining in a separate process with timeout
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_mine_with_timeout_worker,
                args=(miner, prev_block, node_info, requirements, prev_timestamp, stop_event, result_queue)
            )
            process.start()
            process.join(timeout=timeout_per_attempt)

            if process.is_alive():
                # Timeout - terminate the process
                stop_event.set()
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                elapsed = time.time() - start_time
                print(f"    Attempt {attempt + 1}: Timeout after {elapsed:.1f}s")
                result = None
            else:
                # Completed successfully or failed
                elapsed = time.time() - start_time
                try:
                    result = result_queue.get_nowait()
                except:
                    result = None
        else:
            # No timeout - run directly
            result = miner.mine_block(
                prev_block=prev_block,
                node_info=node_info,
                requirements=requirements,
                prev_timestamp=prev_timestamp,
                stop_event=stop_event
            )
            elapsed = time.time() - start_time

        if result:
            times.append(elapsed)
            print(f"    Attempt {attempt + 1}: {elapsed:.1f}s (energy={result.energy:.1f})")
        elif not timeout_per_attempt or elapsed < timeout_per_attempt * 0.9:
            print(f"    Attempt {attempt + 1}: Failed (no solution)")

    if not times:
        return float('inf'), 0

    avg_time = sum(times) / len(times)
    return avg_time, len(times)


def sequential_adaptive_threshold(
    miner,
    target_time_minutes: float,
    tolerance: float,
    target_samples: int,
    max_total_attempts: int,
    output_file: Optional[str] = None,
    timeout_multiplier: Optional[float] = None,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
    min_diversity: float = 0.25,
    min_solutions: int = 5
) -> Optional[dict]:
    """Sequential testing with adaptive difficulty adjustment.

    Algorithm:
    1. Start at initial difficulty
    2. Keep mining until we get target_samples within acceptable range [lower, upper]
    3. If too many samples below lower → increase difficulty
    4. If too many samples above upper → decrease difficulty
    5. Once we have target_samples in range → found threshold

    Args:
        miner: Miner instance
        target_time_minutes: Target block time in minutes
        tolerance: Acceptable deviation (e.g., 0.35 = ±35%)
        target_samples: Number of in-range samples needed (e.g., 35 for CLT)
        max_total_attempts: Maximum total mining attempts before giving up
        output_file: File to write progress to
        timeout_multiplier: Timeout multiplier (default: 1 + tolerance)

    Returns:
        Dictionary with results, or None if failed
    """
    target_time_seconds = target_time_minutes * 60
    lower_bound = target_time_seconds * (1.0 - tolerance)
    upper_bound = target_time_seconds * (1.0 + tolerance)

    if timeout_multiplier is None:
        timeout_multiplier = 1.0 + tolerance
    timeout_per_attempt = target_time_seconds * timeout_multiplier

    # Get energy range from calibration or use provided values
    if energy_min is None or energy_max is None:
        # Use miner's topology for energy calculation
        auto_min, knee_energy, auto_max = calc_energy_range(
            num_nodes=len(miner.nodes),
            num_edges=len(miner.edges)
        )
        min_energy = energy_min if energy_min is not None else auto_min
        max_energy = energy_max if energy_max is not None else auto_max
    else:
        min_energy = energy_min
        max_energy = energy_max

    print(f"\n🔍 Sequential adaptive search for {target_time_minutes:.1f} minute block time")
    print(f"   Topology: {len(miner.nodes)} nodes, {len(miner.edges)} edges")
    print(f"   Target range: [{lower_bound:.0f}s, {upper_bound:.0f}s] (±{tolerance * 100:.0f}%)")
    print(f"   Need {target_samples} samples in range (CLT)")
    print(f"   Max attempts: {max_total_attempts}")
    print(f"   Timeout per attempt: {timeout_per_attempt:.0f}s")
    print(f"   Energy search range: [{min_energy:.1f}, {max_energy:.1f}]")

    # Binary search bounds for adaptive difficulty
    search_left = int(min_energy)  # Hardest (most negative)
    search_right = int(max_energy)  # Easiest (least negative)
    current_energy = int((search_left + search_right) / 2.0)
    previous_energy = current_energy
    last_adjustment_direction = None  # Track if we're oscillating

    in_range_times = []
    below_range_count = 0
    above_range_count = 0
    total_attempts = 0

    # Track all energies achieved for statistics
    all_energies = []

    # Track last 5 outcomes for consistency check
    recent_outcomes = []  # Will store 'fast', 'in_range', or 'slow'

    node_info = NodeInfo(miner_id=f"test-{miner.miner_type}-0")

    # Create genesis block
    prev_block = create_genesis_block()

    while len(in_range_times) < target_samples and total_attempts < max_total_attempts:
        # Update requirements with current difficulty
        requirements = BlockRequirements(
            difficulty_energy=current_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=0
        )
        prev_block.next_block_requirements = requirements
        prev_timestamp = prev_block.header.timestamp

        # Announce next attempt
        print(f"\n▶️  Starting attempt {total_attempts + 1} (GSE target={current_energy:.0f})")

        # Mine one block
        stop_event = multiprocessing.Event()
        start_time = time.time()

        if timeout_per_attempt:
            # Use threading instead of multiprocessing to preserve stdout/stderr
            result_container = [None]

            def mine_thread():
                result_container[0] = miner.mine_block(
                    prev_block=prev_block,
                    node_info=node_info,
                    requirements=requirements,
                    prev_timestamp=prev_timestamp,
                    stop_event=stop_event
                )

            thread = threading.Thread(target=mine_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_per_attempt)

            elapsed = time.time() - start_time

            if thread.is_alive():
                # Timeout - signal stop
                stop_event.set()
                thread.join(timeout=5)
                result = result_container[0]  # May be None or partial result
            else:
                # Completed
                result = result_container[0]
        else:
            result = miner.mine_block(
                prev_block=prev_block,
                node_info=node_info,
                requirements=requirements,
                prev_timestamp=prev_timestamp,
                stop_event=stop_event
            )
            elapsed = time.time() - start_time

        total_attempts += 1

        # Track energy if result was successful
        if result:
            all_energies.append(result.energy)

        # Categorize result and track outcome
        outcome = None
        if result and elapsed <= upper_bound:
            if elapsed >= lower_bound:
                # In range!
                in_range_times.append(elapsed)
                outcome = 'in_range'
                print(f"  ✅ Sample {len(in_range_times)}/{target_samples}: {elapsed:.1f}s, GSE={result.energy:.0f}, diversity={result.diversity:.3f}, solutions={result.num_valid}")
            else:
                # Too fast
                below_range_count += 1
                outcome = 'fast'
                print(f"  ⬇️  Too fast: {elapsed:.1f}s < {lower_bound:.0f}s, GSE={result.energy:.0f}, diversity={result.diversity:.3f}, solutions={result.num_valid}")
        else:
            # Too slow or timeout
            above_range_count += 1
            outcome = 'slow'
            status = "timeout" if elapsed >= timeout_per_attempt * 0.9 else "slow"
            if result:
                print(f"  ⬆️  Too slow ({status}): {elapsed:.1f}s > {upper_bound:.0f}s, GSE={result.energy:.0f}, diversity={result.diversity:.3f}, solutions={result.num_valid}")
            else:
                print(f"  ⬆️  Too slow ({status}): {elapsed:.1f}s > {upper_bound:.0f}s (no valid solution found)")

        # Track last 5 outcomes
        if outcome:
            recent_outcomes.append(outcome)
            if len(recent_outcomes) > 5:
                recent_outcomes.pop(0)  # Keep only last 5

        # Print statistics table after each attempt
        if all_energies:
            import statistics
            print(f"\n  📊 GSE Statistics (n={len(all_energies)}):")
            print(f"     Min: {min(all_energies):.1f}")
            print(f"     Max: {max(all_energies):.1f}")
            print(f"     Mean: {statistics.mean(all_energies):.1f}")
            print(f"     Median: {statistics.median(all_energies):.1f}")
            if len(all_energies) > 1:
                print(f"     Variance: {statistics.variance(all_energies):.1f}")
                print(f"     StdDev: {statistics.stdev(all_energies):.1f}")
            try:
                mode_val = statistics.mode(all_energies)
                print(f"     Mode: {mode_val:.1f}")
            except statistics.StatisticsError:
                print(f"     Mode: N/A (no unique mode)")
            print(f"     Difficulty range: [{min_energy:.1f}, {max_energy:.1f}]")
            print(f"     Current target: {current_energy:.1f}")
            if previous_energy != current_energy:
                delta = current_energy - previous_energy
                print(f"     Δ from previous: {delta:+.2f}\n")
            else:
                print()

        # Write progress
        if output_file:
            try:
                progress_data = {
                    'status': 'in_progress',
                    'current_difficulty': current_energy,
                    'in_range_samples': len(in_range_times),
                    'target_samples': target_samples,
                    'below_range': below_range_count,
                    'above_range': above_range_count,
                    'total_attempts': total_attempts,
                    'timestamp': utc_timestamp()
                }
                with open(output_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                print(f"  ⚠️  Warning: Could not write progress: {e}")

        # Check consistency after we have 5 outcomes
        if len(recent_outcomes) == 5 and len(in_range_times) < target_samples:
            # Count outcomes in last 5
            fast_count = recent_outcomes.count('fast')
            slow_count = recent_outcomes.count('slow')
            in_range_count = recent_outcomes.count('in_range')

            # Check if behavior is consistent (all or most outcomes are the same)
            if fast_count >= 4:
                # Consistently too fast → make harder
                previous_energy = current_energy
                search_right = current_energy
                current_energy = int((search_left + search_right) / 2.0)
                print(f"  🔧 Adjusting HARDER: {current_energy} (5 recent: {fast_count} fast, {in_range_count} in-range, {slow_count} slow)")
                recent_outcomes.clear()
                below_range_count = 0
                above_range_count = 0
            elif slow_count >= 4:
                # Consistently too slow → make easier
                previous_energy = current_energy
                search_left = current_energy
                current_energy = int((search_left + search_right) / 2.0)
                print(f"  🔧 Adjusting EASIER: {current_energy} (5 recent: {fast_count} fast, {in_range_count} in-range, {slow_count} slow)")
                recent_outcomes.clear()
                below_range_count = 0
                above_range_count = 0
            else:
                # Mixed behavior → stop adjusting, we've found the right difficulty
                print(f"  ✋ Behavior is mixed (5 recent: {fast_count} fast, {in_range_count} in-range, {slow_count} slow)")
                print(f"  ✅ Converged at difficulty {current_energy} - stopping adjustments")
                print(f"  📊 Collecting remaining samples at this difficulty...")
                # Don't adjust anymore - just collect samples

            # Ensure we stay within bounds
            current_energy = int(max(min_energy, min(max_energy, current_energy)))

    # Always return results, even if we didn't reach target
    import statistics
    if len(in_range_times) > 0:
        avg_time = sum(in_range_times) / len(in_range_times)
        stdev_time = statistics.stdev(in_range_times) if len(in_range_times) > 1 else 0.0
        variance_time = statistics.variance(in_range_times) if len(in_range_times) > 1 else 0.0
    else:
        # No in-range samples, use all samples that succeeded
        all_times = []
        # We don't have access to all times here, so we'll report as failed
        return None

    return {
        'difficulty_energy': current_energy,
        'avg_time_seconds': avg_time,
        'avg_time_minutes': avg_time / 60.0,
        'stdev_seconds': stdev_time,
        'variance_seconds': variance_time,
        'in_range_samples': len(in_range_times),
        'below_range_samples': below_range_count,
        'above_range_samples': above_range_count,
        'total_attempts': total_attempts,
        'successful_attempts': len(in_range_times),
        'converged': len(in_range_times) >= target_samples,
        'success': len(in_range_times) >= target_samples
    }


def binary_search_threshold(
    miner,
    target_time_minutes: float,
    tolerance: float,
    max_iterations: int,
    num_attempts_per_test: int,
    output_file: Optional[str] = None,
    timeout_multiplier: Optional[float] = None
) -> Optional[dict]:
    """Binary search for difficulty that yields target mining time.

    Args:
        miner: Miner instance (CPU, CUDA, etc.)
        target_time_minutes: Target block time in minutes
        tolerance: Acceptable deviation from target (fraction, e.g., 0.35 = ±35%)
        max_iterations: Maximum binary search iterations
        num_attempts_per_test: Number of mining attempts to average per difficulty level
        output_file: File to write progress to
        timeout_multiplier: Timeout = target_time × this multiplier (default: None = uses 1 + tolerance)

    Returns:
        Dictionary with results, or None if failed
    """
    target_time_seconds = target_time_minutes * 60

    # Default timeout: upper bound of acceptable range (target × (1 + tolerance))
    # Anything beyond this is outside our desired variance, so no point continuing
    if timeout_multiplier is None:
        timeout_multiplier = 1.0 + tolerance

    timeout_per_attempt = target_time_seconds * timeout_multiplier

    # Get energy range from calibration using miner's topology
    min_energy, knee_energy, max_energy = calc_energy_range(
        num_nodes=len(miner.nodes),
        num_edges=len(miner.edges)
    )

    print(f"\n🔍 Binary search for {target_time_minutes:.1f} minute block time")
    print(f"   Topology: {len(miner.nodes)} nodes, {len(miner.edges)} edges")
    print(f"   Energy range: [{min_energy:.1f}, {max_energy:.1f}]")
    print(f"   Tolerance: ±{tolerance * 100:.0f}%")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Timeout per attempt: {timeout_per_attempt:.0f}s ({timeout_multiplier}× target)")

    # Binary search bounds (in energy space)
    left = min_energy  # Hardest (most negative)
    right = max_energy  # Easiest (least negative)

    best_result = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Test midpoint
        mid_energy = (left + right) / 2.0

        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        print(f"Range: [{left:.1f}, {right:.1f}], testing {mid_energy:.1f}")

        avg_time, successful = test_mining_time(
            miner, mid_energy,
            num_attempts=num_attempts_per_test,
            timeout_per_attempt=timeout_per_attempt
        )

        if successful == 0:
            # Too hard - make easier
            print(f"  ⚠️  All attempts failed, making easier")
            left = mid_energy
            continue

        avg_time_minutes = avg_time / 60.0
        deviation = abs(avg_time - target_time_seconds) / target_time_seconds

        print(f"  📊 Average time: {avg_time_minutes:.1f} min (target: {target_time_minutes:.1f} min)")
        print(f"  📏 Deviation: {deviation * 100:.1f}% (tolerance: {tolerance * 100:.0f}%)")

        # Track best result
        if best_result is None or deviation < best_result['deviation']:
            best_result = {
                'difficulty_energy': mid_energy,
                'avg_time_seconds': avg_time,
                'avg_time_minutes': avg_time_minutes,
                'successful_attempts': successful,
                'total_attempts': num_attempts_per_test,
                'deviation': deviation,
                'iteration': iteration
            }

            # Write progress to file after each improvement
            if output_file:
                try:
                    progress_data = {
                        'status': 'in_progress',
                        'current_best': best_result,
                        'iteration': iteration,
                        'max_iterations': max_iterations,
                        'timestamp': utc_timestamp()
                    }
                    with open(output_file, 'w') as f:
                        json.dump(progress_data, f, indent=2)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not write progress file: {e}")

        # Check if within tolerance
        if deviation <= tolerance:
            print(f"  ✅ Found threshold within tolerance!")
            best_result['converged'] = True
            break

        # Adjust search bounds
        if avg_time < target_time_seconds:
            # Too fast - make harder (more negative)
            print(f"  ⬇️  Too fast, making harder")
            right = mid_energy
        else:
            # Too slow - make easier (less negative)
            print(f"  ⬆️  Too slow, making easier")
            left = mid_energy

        # Check convergence
        if abs(right - left) < 10.0:  # Energy units
            print(f"  ℹ️  Search range converged (width: {abs(right - left):.1f})")
            break

    if best_result:
        best_result['converged'] = best_result.get('converged', False)
        best_result['final_range'] = [left, right]

    return best_result


def parse_time(time_str: str) -> float:
    """Parse human-readable time string to minutes.

    Supports formats like:
    - "10" or "10m" -> 10 minutes
    - "30s" -> 0.5 minutes
    - "2h" -> 120 minutes
    - "1.5h" -> 90 minutes

    Args:
        time_str: Time string to parse

    Returns:
        Time in minutes
    """
    time_str = time_str.strip().lower()

    # Parse value and unit
    if time_str.endswith('s'):
        value = float(time_str[:-1])
        return value / 60.0  # Convert seconds to minutes
    elif time_str.endswith('m'):
        return float(time_str[:-1])
    elif time_str.endswith('h'):
        value = float(time_str[:-1])
        return value * 60.0  # Convert hours to minutes
    else:
        # Assume minutes if no unit specified
        return float(time_str)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Find difficulty threshold for target block time'
    )
    parser.add_argument(
        '--miner-type',
        type=str,
        choices=['cpu', 'cuda', 'metal'],
        required=True,
        help='Miner type to test'
    )
    parser.add_argument(
        '--target-time',
        type=str,
        default='10m',
        help='Target block time (e.g., "10m", "30s", "2h") (default: 10m)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.35,
        help='Acceptable deviation from target (default: 0.35 = ±35%%)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum binary search iterations (default: 10)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['binary', 'sequential'],
        default='sequential',
        help='Search algorithm: binary (fixed attempts per test) or sequential (adaptive) (default: sequential)'
    )
    parser.add_argument(
        '--attempts-per-test',
        type=int,
        default=35,
        help='[Binary only] Mining attempts per difficulty level. [Sequential] Target in-range samples (default: 35, satisfies CLT)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=200,
        help='[Sequential only] Maximum total attempts before giving up (default: 200)'
    )
    parser.add_argument(
        '--energy-min',
        type=float,
        default=None,
        help='[Sequential only] Minimum (hardest/most negative) energy to search (default: auto-detect from topology)'
    )
    parser.add_argument(
        '--energy-max',
        type=float,
        default=None,
        help='[Sequential only] Maximum (easiest/least negative) energy to search (default: auto-detect from topology)'
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.25,
        help='Minimum solution diversity required (default: 0.25)'
    )
    parser.add_argument(
        '--min-solutions',
        type=int,
        default=5,
        help='Minimum number of valid solutions required (default: 5)'
    )
    parser.add_argument(
        '--timeout-multiplier',
        type=float,
        default=None,
        help='Timeout per attempt = target_time × this value (default: None = 1 + tolerance, i.e., upper bound of acceptable range)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Path to topology/embedding JSON file (default: use Z(9,2) perfect topology)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    # Parse target time
    target_time_minutes = parse_time(args.target_time)

    # Load topology if specified
    topology = None
    if args.topology:
        topology = load_topology(args.topology)

    print("🔬 Block Time Threshold Finder")
    print("=" * 50)
    print(f"Miner type: {args.miner_type.upper()}")
    print(f"Target time: {target_time_minutes} minutes")
    print(f"Tolerance: ±{args.tolerance * 100:.0f}%")

    # Initialize miner
    miner = None
    if args.miner_type == 'cpu':
        from CPU.sa_miner import SimulatedAnnealingMiner
        from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
        sampler = SimulatedAnnealingStructuredSampler(topology=topology)
        miner = SimulatedAnnealingMiner(
            miner_id="threshold-test-cpu",
            sampler=sampler
        )
    elif args.miner_type == 'cuda':
        from GPU.cuda_miner import CudaMiner
        miner = CudaMiner(
            miner_id="threshold-test-cuda",
            device="0",
            topology=topology
        )
    elif args.miner_type == 'metal':
        from GPU.metal_miner import MetalMiner
        miner = MetalMiner(
            miner_id="threshold-test-metal",
            topology=topology
        )

    if not miner:
        print(f"❌ Failed to initialize {args.miner_type} miner")
        return 1

    print(f"✅ {args.miner_type.upper()} miner initialized")

    # Determine output file
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        # Create safe filename from target time
        safe_time = args.target_time.replace('.', '_')
        output_file = f"threshold_{args.miner_type}_{safe_time}_{timestamp}.json"

    # Run search algorithm
    start_time = time.time()
    if args.algorithm == 'sequential':
        result = sequential_adaptive_threshold(
            miner=miner,
            target_time_minutes=target_time_minutes,
            tolerance=args.tolerance,
            target_samples=args.attempts_per_test,
            max_total_attempts=args.max_attempts,
            output_file=output_file,
            timeout_multiplier=args.timeout_multiplier,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
            min_diversity=args.min_diversity,
            min_solutions=args.min_solutions
        )
    else:  # binary
        result = binary_search_threshold(
            miner=miner,
            target_time_minutes=target_time_minutes,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            num_attempts_per_test=args.attempts_per_test,
            output_file=output_file,
            timeout_multiplier=args.timeout_multiplier
        )
    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)

    if result:
        print(f"✅ Best threshold found:")
        print(f"   Difficulty energy: {result['difficulty_energy']:.1f}")
        print(f"   Average block time: {result['avg_time_minutes']:.2f} min ({result['avg_time_seconds']:.1f}s)")

        # Show variance/stdev if available (sequential algorithm)
        if 'stdev_seconds' in result:
            print(f"   Standard deviation: {result['stdev_seconds']:.1f}s")
            print(f"   Variance: {result['variance_seconds']:.1f}s²")

        # Show deviation if available (binary search algorithm)
        if 'deviation' in result:
            print(f"   Deviation: {result['deviation'] * 100:.1f}%")

        print(f"   Successful attempts: {result['successful_attempts']}/{result['total_attempts']}")
        print(f"   Converged: {'Yes' if result.get('converged', False) else 'No'}")
        print(f"   Total search time: {total_time / 60:.1f} min")

        # Save final results
        output_data = {
            'status': 'completed',
            'miner_type': args.miner_type,
            'target_time_minutes': target_time_minutes,
            'tolerance': args.tolerance,
            'result': result,
            'total_search_time_seconds': total_time,
            'timestamp': utc_timestamp()
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to {output_file}")
        return 0
    else:
        print("❌ Failed to find threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())
