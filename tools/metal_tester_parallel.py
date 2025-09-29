#!/usr/bin/env python3
"""Metal Parallel Tempering performance tester with known optimal energies."""

import sys
import time
import argparse
from pathlib import Path

from basic_ising_problems import BASIC_ISING_PROBLEMS
import dimod

try:
    from GPU.metal_sampler_parallel import MetalKernelDimodSampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def _is_ea_compatible(h, J):
    """Check if a problem is compatible with 3D Edwards-Anderson Metal sampler."""
    try:
        # Test BQM creation
        bqm = dimod.BinaryQuadraticModel(h, J, 'SPIN')

        # Check EA requirements
        # 1. All linear biases must be zero
        has_nonzero_h = any(abs(bias) > 1e-10 for bias in bqm.linear.values())

        # 2. Problem size must be cubic
        N = bqm.num_variables
        L = round(N ** (1/3))
        is_cubic = (L ** 3 == N)

        # 3. All couplings must be ±1
        valid_couplings = all(abs(abs(coupling) - 1.0) <= 1e-10 for coupling in bqm.quadratic.values())

        # 4. Must use SPIN variables
        is_spin = (bqm.vartype == dimod.SPIN)

        return not has_nonzero_h and is_cubic and valid_couplings and is_spin

    except Exception:
        return False


# Filter problems to only EA-compatible ones
EA_COMPATIBLE_PROBLEMS = []
for idx, (h, J, optimal_energy, description) in enumerate(BASIC_ISING_PROBLEMS):
    if _is_ea_compatible(h, J):
        EA_COMPATIBLE_PROBLEMS.append((idx, h, J, optimal_energy, description))

EA_COMPATIBLE_INDICES = [prob[0] for prob in EA_COMPATIBLE_PROBLEMS]


def test_metal_sampler(problem_idx, num_reads=256, num_sweeps=1000, timeout_seconds=20.0,
                      max_retries=3, num_replicas=None, sample_interval=None):
    """Test Metal sampler on a specific problem with retry logic."""

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

    print(f"   🔄 Testing Parallel Tempering...")
    for attempt in range(max_retries):
        try:
            sampler = MetalKernelDimodSampler()

            start_time = time.time()
            sampleset = sampler.sample_ising(
                h, J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                num_replicas=num_replicas,
                sample_interval=sample_interval
            )
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                print(f"      ⏰ Timeout after {elapsed:.1f}s (attempt {attempt + 1})")
                continue

            energies = sampleset.record.energy
            min_energy = float(min(energies))
            success = min_energy <= optimal_energy

            print(f"      ⏱️  {elapsed:.3f}s, Min energy: {min_energy}")
            print(f"      ✅ Success: {success}")
            print(f"      🔍 Debug: sampleset has {len(sampleset)} samples")
            print(f"      🔍 Debug: energy range {min(energies)} to {max(energies)}")

            results['parallel_tempering'] = {
                'time': elapsed,
                'min_energy': min_energy,
                'success': success,
                'attempt': attempt + 1
            }
            break

        except Exception as e:
            print(f"      ❌ Error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                results['parallel_tempering'] = {
                    'time': None,
                    'min_energy': None,
                    'success': False,
                    'error': str(e)
                }

    return results


def main():
    parser = argparse.ArgumentParser(description='Metal Parallel Tempering performance tester')
    parser.add_argument('--problem', type=int, help='Test specific problem index')
    parser.add_argument('--reads', type=int, default=256, help='Number of reads')
    parser.add_argument('--sweeps', type=int, default=1000, help='Number of sweeps')
    parser.add_argument('--timeout', type=float, default=20.0, help='Timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Max retries per test')
    parser.add_argument('--num-replicas', type=int, help='Number of replicas')
    parser.add_argument('--sample-interval', type=int, help='Sample interval')

    args = parser.parse_args()

    print("🔬 Metal Parallel Tempering Performance Tester")
    print("=" * 70)
    print("✅ Metal sampler now uses the provided h,J problems from basic_ising_problems.py")
    print("✅ Metal Parallel Tempering kernel sampler ready")
    print(f"🎯 Total problems available: {len(BASIC_ISING_PROBLEMS)}")
    print(f"🔍 EA-compatible problems: {len(EA_COMPATIBLE_PROBLEMS)}")
    print(f"🔧 Compatible problem indices: {EA_COMPATIBLE_INDICES}")
    print(f"🔄 Retry limit: {args.retries} attempts per test")

    # Parameter override message
    override_params = []
    if args.reads != 256:
        override_params.append(f"reads={args.reads}")
    if args.sweeps != 1000:
        override_params.append(f"sweeps={args.sweeps}")
    if args.sample_interval:
        override_params.append(f"sample_interval={args.sample_interval}")

    if override_params:
        print(f"⚙️  Parameter overrides: {', '.join(override_params)}")

    # Test specific problem or EA-compatible ones
    if args.problem is not None:
        if args.problem not in EA_COMPATIBLE_INDICES:
            print(f"⚠️  Problem {args.problem} is not EA-compatible")
        print(f"🎯 Testing only problem {args.problem} (EA-compatible)")
        problem_indices = [args.problem]
    else:
        print(f"🎯 Testing all {len(EA_COMPATIBLE_PROBLEMS)} EA-compatible problems")
        problem_indices = EA_COMPATIBLE_INDICES

    print()

    all_results = {}
    for problem_idx in problem_indices:
        results = test_metal_sampler(
            problem_idx,
            num_reads=args.reads,
            num_sweeps=args.sweeps,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
            num_replicas=args.num_replicas,
            sample_interval=args.sample_interval
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

        # Analyze parallel tempering results
        pt_times = []
        pt_successes = 0

        for results in all_results.values():
            if 'parallel_tempering' in results:
                pt_result = results['parallel_tempering']
                if pt_result['time'] is not None:
                    pt_times.append(pt_result['time'])
                if pt_result['success']:
                    pt_successes += 1

        # Display results
        if pt_times:
            avg_time = sum(pt_times) / len(pt_times)
            last_time = pt_times[-1] if pt_times else 0
            last_success = all_results[last_problem_idx]['parallel_tempering']['success']

            print(f"Parallel Tempering: {last_time:.1f}s, Success: {last_success}")
            print(f"Success rate - Parallel Tempering: {pt_successes}/{len(all_results)}")
            print(f"Average time - Parallel Tempering: {avg_time:.1f}s")

        total_solved = sum(1 for results in all_results.values()
                          if results.get('parallel_tempering', {}).get('success', False))
        print(f"Total problems solved: {total_solved}/{len(all_results)}")
    else:
        print("No results to display")

    print()
    print("✅ Metal tester complete!")


if __name__ == "__main__":
    main()