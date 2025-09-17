#!/usr/bin/env python3
"""Metal hierarchical vs original p-bit tester with known optimal energies."""

import sys
import time
import argparse
from pathlib import Path

from basic_ising_problems import BASIC_ISING_PROBLEMS

try:
    from GPU.metal_kernel_sampler import MetalKernelDimodSampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def test_metal_sampler(skip=0, retry=3, reads=None, sweeps=None):
    """Test Metal sampler hierarchical vs original on known problems."""
    print("🔬 Metal Hierarchical vs Original P-bit Tester")
    print("=" * 60)

    # Initialize samplers
    metal_sampler = None

    # Check availability and initialize
    metal_available = 'METAL_AVAILABLE' in globals() and METAL_AVAILABLE

    if metal_available:
        try:
            metal_sampler = MetalKernelDimodSampler("mps")
            print("✅ Metal P-bit kernel sampler ready")
        except Exception as e:
            print(f"❌ Metal P-bit sampler failed: {e}")
            metal_available = False

    if not metal_available:
        print("❌ Metal sampler not available")
        return

    # Test parameters - scale with problem size or use overrides
    def get_test_params(num_variables):
        # Use override values if provided
        if reads is not None and sweeps is not None:
            # Scale timeout based on problem size even with overrides
            if num_variables <= 16:
                timeout = 5.0
            elif num_variables <= 64:
                timeout = 10.0
            elif num_variables <= 128:
                timeout = 15.0
            else:  # 256+
                timeout = 30.0
            return reads, sweeps, timeout
        elif reads is not None:
            # Override reads only, scale sweeps and timeout
            if num_variables <= 16:
                return reads, 1000, 5.0
            elif num_variables <= 64:
                return reads, 500, 10.0
            elif num_variables <= 128:
                return reads, 250, 15.0
            else:  # 256+
                return reads, 100, 30.0
        elif sweeps is not None:
            # Override sweeps only, scale reads and timeout
            if num_variables <= 16:
                return 100, sweeps, 5.0
            elif num_variables <= 64:
                return 50, sweeps, 10.0
            elif num_variables <= 128:
                return 25, sweeps, 15.0
            else:  # 256+
                return 10, sweeps, 30.0
        else:
            # Default scaling
            if num_variables <= 16:
                return 100, 1000, 5.0
            elif num_variables <= 64:
                return 50, 500, 10.0
            elif num_variables <= 128:
                return 25, 250, 15.0
            else:  # 256+
                return 10, 100, 30.0

    print(f"🎯 Problems: {len(BASIC_ISING_PROBLEMS)} (with scaled parameters)")
    if skip > 0:
        print(f"⏭️  Skipping first {skip} problems")
    if retry > 0:
        print(f"🔄 Retry limit: {retry} attempts per test")
    if reads is not None or sweeps is not None:
        override_info = []
        if reads is not None:
            override_info.append(f"reads={reads}")
        if sweeps is not None:
            override_info.append(f"sweeps={sweeps}")
        print(f"⚙️  Parameter overrides: {', '.join(override_info)}")

    results = []

    def test_hierarchical(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy):
        """Test hierarchical method. Returns (success, runtime) tuple."""
        try:
            start_time = time.time()
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                use_hierarchical=True
            )
            runtime = time.time() - start_time

            min_energy = float(min(sampleset.record.energy))
            success = abs(min_energy - optimal_energy) < 1e-6  # Exact match

            print(f"      ⏱️  {runtime:.1f}s, Min energy: {min_energy:.1f}")
            print(f"      ✅ Success: {success}")

            if runtime > timeout_seconds:
                print(f"      ⏰ Timeout exceeded ({timeout_seconds}s)")
                return False, runtime

            return success, runtime

        except Exception as e:
            print(f"      ❌ Hierarchical error: {e}")
            return False, 0.0

    def test_original(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy):
        """Test original p-bit method. Returns (success, runtime) tuple."""
        try:
            start_time = time.time()
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                use_hierarchical=False
            )
            runtime = time.time() - start_time

            min_energy = float(min(sampleset.record.energy))
            success = abs(min_energy - optimal_energy) < 1e-6  # Exact match

            print(f"      ⏱️  {runtime:.1f}s, Min energy: {min_energy:.1f}")
            print(f"      ✅ Success: {success}")

            if runtime > timeout_seconds:
                print(f"      ⏰ Timeout exceeded ({timeout_seconds}s)")
                return False, runtime

            return success, runtime

        except Exception as e:
            print(f"      ❌ Original p-bit error: {e}")
            return False, 0.0

    for idx, (h, J, optimal_energy, description) in enumerate(BASIC_ISING_PROBLEMS):
        if idx < skip:
            continue
        num_variables = len(h)
        num_reads, num_sweeps, timeout_seconds = get_test_params(num_variables)

        print(f"\n🧪 Problem {idx}: {description}")
        print(f"   Variables: {num_variables}, Couplings: {len(J)}, Optimal GSE: {optimal_energy}")
        print(f"   📊 Config: {num_reads} reads, {num_sweeps} sweeps, {timeout_seconds}s timeout")

        problem_results = {
            'problem_idx': idx,
            'description': description,
            'num_variables': num_variables,
            'num_couplings': len(J),
            'optimal_energy': optimal_energy,
            'hierarchical': None,
            'original': None
        }

        # Test original p-bit first
        original_success = False
        original_runtime = 0.0
        if metal_available and metal_sampler is not None:
            print("   🔄 Testing original p-bit...")
            for attempt in range(retry + 1):  # +1 for initial attempt
                if attempt > 0:
                    print(f"   🔄 Original retry {attempt}/{retry}...")
                success, runtime = test_original(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy)
                if success:
                    original_success = True
                    original_runtime = runtime
                    break

            if not original_success:
                print(f"   ❌ Original p-bit failed after {retry + 1} attempts")

        # Test hierarchical
        hierarchical_success = False
        hierarchical_runtime = 0.0
        if metal_available and metal_sampler is not None:
            print("   🔄 Testing hierarchical...")
            for attempt in range(retry + 1):  # +1 for initial attempt
                if attempt > 0:
                    print(f"   🔄 Hierarchical retry {attempt}/{retry}...")
                success, runtime = test_hierarchical(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy)
                if success:
                    hierarchical_success = True
                    hierarchical_runtime = runtime
                    break

            if not hierarchical_success:
                print(f"   ❌ Hierarchical failed after {retry + 1} attempts")
                break

        # Store results with timing information
        problem_results['hierarchical'] = {'success': hierarchical_success, 'runtime': hierarchical_runtime}
        problem_results['original'] = {'success': original_success, 'runtime': original_runtime}

        results.append(problem_results)

        # Check if both succeeded
        if not (problem_results['hierarchical']['success'] and problem_results['original']['success']):
            print(f"   ❌ One or both methods failed on problem {idx}")
            break

    # Summary
    print(f"\n📊 Summary:")
    print("=" * 50)
    print(f"Problems tested: {len(results)}")

    if results:
        last_result = results[-1]
        print(f"Last problem: {last_result['description']}")
        print(f"Hierarchical: {last_result['hierarchical']['runtime']:.1f}s, Success: {last_result['hierarchical']['success']}")
        print(f"Original: {last_result['original']['runtime']:.1f}s, Success: {last_result['original']['success']}")

        # Success rate summary
        total_problems = len(results)
        hier_successes = sum(1 for r in results if r['hierarchical']['success'])
        orig_successes = sum(1 for r in results if r['original']['success'])
        print(f"Success rates - Hierarchical: {hier_successes}/{total_problems}, Original: {orig_successes}/{total_problems}")

        # Performance comparison (only for successful runs)
        successful_problems = [r for r in results if r['hierarchical']['success'] and r['original']['success']]
        if successful_problems:
            total_hier_time = sum(r['hierarchical']['runtime'] for r in successful_problems)
            total_orig_time = sum(r['original']['runtime'] for r in successful_problems)
            avg_hier_time = total_hier_time / len(successful_problems)
            avg_orig_time = total_orig_time / len(successful_problems)

            print(f"Average times - Hierarchical: {avg_hier_time:.1f}s, Original: {avg_orig_time:.1f}s")
            if avg_orig_time > 0:
                speedup = avg_orig_time / avg_hier_time
                print(f"Speedup: {speedup:.2f}x (hierarchical is {speedup:.0f}% faster)")
    # Clean up
    if metal_available and metal_sampler is not None:
        metal_sampler.close()

    print("\n✅ Metal tester complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Metal sampler hierarchical vs original on known problems")
    parser.add_argument("--skip", type=int, default=0,
                       help="Number of problems to skip (default: 0)")
    parser.add_argument("--retry", type=int, default=3,
                       help="Number of retry attempts per test (default: 3)")
    parser.add_argument("--reads", type=int, default=None,
                       help="Override number of reads (default: auto-scaled)")
    parser.add_argument("--sweeps", type=int, default=None,
                       help="Override number of sweeps (default: auto-scaled)")
    args = parser.parse_args()

    test_metal_sampler(skip=args.skip, retry=args.retry, reads=args.reads, sweeps=args.sweeps)