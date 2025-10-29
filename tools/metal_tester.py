#!/usr/bin/env python3
"""Metal SA performance tester with known optimal energies."""

import sys
import time
import argparse
from pathlib import Path

from basic_ising_problems import BASIC_ISING_PROBLEMS

try:
    from GPU.metal_sa import MetalSASampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def test_metal_sampler(problem_idx, num_reads=256, num_sweeps=512, timeout_seconds=20.0,
                      max_retries=3):
    """Test Metal SA sampler on a specific problem with retry logic."""

    if problem_idx >= len(BASIC_ISING_PROBLEMS):
        print(f"❌ Problem {problem_idx} not found")
        return None

    h, J, optimal_energy, description = BASIC_ISING_PROBLEMS[problem_idx]

    print(f"🧪 Problem {problem_idx}: {description}")
    print(f"   Variables: {len(h)}, Couplings: {len(J)}, Optimal GSE: {optimal_energy}")
    print(f"   📊 Config: {num_reads} reads, {num_sweeps} sweeps, {timeout_seconds}s timeout")

    if not METAL_AVAILABLE:
        print(f"   ❌ Metal not available")
        return None

    results = {}

    print(f"   🔄 Testing SA...")
    for attempt in range(max_retries):
        try:
            sampler = MetalSASampler()

            start_time = time.time()
            # Task 6: Wrap single problem in lists for batched API
            samplesets = sampler.sample_ising(
                [h], [J],
                num_reads=num_reads,
                num_sweeps=num_sweeps
            )
            sampleset = samplesets[0]  # Extract single result
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                print(f"      ⏰ Timeout after {elapsed:.1f}s (attempt {attempt + 1})")
                continue

            energies = sampleset.record.energy
            min_energy = float(min(energies))
            success = abs(min_energy - optimal_energy) < 1e-6

            print(f"      ⏱️  {elapsed:.3f}s, Min energy: {min_energy}")
            print(f"      ✅ Success: {success}")
            print(f"      🔍 Debug: sampleset has {len(sampleset)} samples")
            print(f"      🔍 Debug: energy range {min(energies)} to {max(energies)}")

            results['sa'] = {
                'time': elapsed,
                'min_energy': min_energy,
                'success': success,
                'attempt': attempt + 1
            }
            break

        except Exception as e:
            print(f"      ❌ Error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                results['sa'] = {
                    'time': None,
                    'min_energy': None,
                    'success': False,
                    'error': str(e)
                }

    return results


def main():
    parser = argparse.ArgumentParser(description='Metal SA performance tester')
    parser.add_argument('--problem', type=int, help='Test specific problem index')
    parser.add_argument('--reads', type=int, default=256, help='Number of reads')
    parser.add_argument('--sweeps', type=int, default=1000, help='Number of sweeps')
    parser.add_argument('--timeout', type=float, default=20.0, help='Timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Max retries per test')

    args = parser.parse_args()

    print(f"🔬 Metal SA Performance Tester")
    print("=" * 70)
    print("✅ Metal sampler now uses the provided h,J problems from basic_ising_problems.py")
    print("✅ Metal SA kernel sampler ready")
    print(f"🎯 Total problems available: {len(BASIC_ISING_PROBLEMS)}")
    print(f"🔄 Retry limit: {args.retries} attempts per test")

    # Parameter override message
    override_params = []
    if args.reads != 256:
        override_params.append(f"reads={args.reads}")
    if args.sweeps != 1000:
        override_params.append(f"sweeps={args.sweeps}")

    if override_params:
        print(f"⚙️  Parameter overrides: {', '.join(override_params)}")

    # Test specific problem or all
    if args.problem is not None:
        print(f"🎯 Testing only problem {args.problem}")
        problem_indices = [args.problem]
    else:
        print(f"🎯 Testing all {len(BASIC_ISING_PROBLEMS)} problems")
        problem_indices = list(range(len(BASIC_ISING_PROBLEMS)))

    print()

    all_results = {}
    for problem_idx in problem_indices:
        results = test_metal_sampler(
            problem_idx,
            num_reads=args.reads,
            num_sweeps=args.sweeps,
            timeout_seconds=args.timeout,
            max_retries=args.retries
        )
        if results:
            all_results[problem_idx] = results
        print()

    # Summary
    print("📊 Summary:")
    print("=" * 50)

    if all_results:
        print(f"Problems tested: {len(all_results)}")

        # Get last problem info
        last_problem_idx = max(all_results.keys())
        last_problem_desc = BASIC_ISING_PROBLEMS[last_problem_idx][3]
        print(f"Last problem: {last_problem_desc}")

        # Analyze SA results
        sa_times = []
        sa_successes = 0

        for results in all_results.values():
            if 'sa' in results:
                sa_result = results['sa']
                if sa_result['time'] is not None:
                    sa_times.append(sa_result['time'])
                if sa_result['success']:
                    sa_successes += 1

        # Display results
        if sa_times:
            avg_time = sum(sa_times) / len(sa_times)
            last_time = sa_times[-1] if sa_times else 0
            last_success = all_results[last_problem_idx]['sa']['success']

            print(f"SA: {last_time:.1f}s, Success: {last_success}")
            print(f"Success rate - SA: {sa_successes}/{len(all_results)}")
            print(f"Average time - SA: {avg_time:.1f}s")

        total_solved = sum(1 for results in all_results.values()
                          if results.get('sa', {}).get('success', False))
        print(f"Total problems solved: {total_solved}/{len(all_results)}")
    else:
        print("No results to display")

    print()
    print("✅ Metal SA tester complete!")


if __name__ == "__main__":
    main()
