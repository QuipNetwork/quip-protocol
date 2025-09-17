#!/usr/bin/env python3
"""Metal hierarchical vs original p-bit tester with known optimal energies."""

import sys
import time
import argparse
from pathlib import Path

from basic_ising_problems import BASIC_ISING_PROBLEMS

try:
    from GPU.metal_sampler import MetalSampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def test_metal_sampler(skip=0, retry=3, reads=None, sweeps=None, debug=False, problem=None, block_size=None, use_sparse_updates=True, timing_variance=0.1, intensity_variance=0.1, offset_variance=0.1, spins_per_block=96, beta_start=0.01, beta_end=15.0, max_flips_per_block=None, initial_temperature=None, temperature_decay_rate=None, annealing_schedule_type="logspace", num_trotters=None, gamma=1.0, total_annealing_steps=None):
    """Test Metal sampler hierarchical vs original on known problems."""
    print("🔬 Metal Hierarchical vs Original P-bit Tester")
    print("=" * 60)

    # Initialize samplers
    metal_sampler = None

    # Check availability and initialize
    metal_available = 'METAL_AVAILABLE' in globals() and METAL_AVAILABLE

    if metal_available:
        try:
            metal_sampler = MetalSampler("mps")
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

    if problem is not None:
        print(f"🎯 Testing only problem {problem}")

    results = []

    def test_hierarchical(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug=False):
        """Test hierarchical method. Returns (success, runtime) tuple."""
        try:
            start_time = time.time()
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                use_hierarchical=True,
                block_size=block_size,
                use_sparse_updates=use_sparse_updates,
                timing_variance=timing_variance,
                intensity_variance=intensity_variance,
                offset_variance=offset_variance,
                spins_per_block=spins_per_block,
                beta_start=beta_start,
                beta_end=beta_end,
                max_flips_per_block=max_flips_per_block,
                initial_temperature=initial_temperature,
                temperature_decay_rate=temperature_decay_rate,
                annealing_schedule_type=annealing_schedule_type,
                num_trotters=num_trotters,
                gamma=gamma,
                total_annealing_steps=total_annealing_steps
            )
            runtime = time.time() - start_time

            min_energy = float(min(sampleset.record.energy))
            success = abs(min_energy - optimal_energy) < 1e-6  # Exact match

            print(f"      ⏱️  {runtime:.1f}s, Min energy: {min_energy:.1f}")
            print(f"      ✅ Success: {success}")

            if debug and hasattr(sampleset, 'record'):
                energies = sampleset.record.energy
                if len(energies) > 0:
                    print(f"      🔍 Energy range: {min(energies):.1f} to {max(energies):.1f}")
                    print(f"      🔍 Unique energies: {len(set(energies))}")
                    # Show distribution of top 5 energies
                    sorted_energies = sorted(set(energies))[:5]
                    print(f"      🔍 Best energies: {sorted_energies}")

            if runtime > timeout_seconds:
                print(f"      ⏰ Timeout exceeded ({timeout_seconds}s)")
                return False, runtime

            return success, runtime

        except Exception as e:
            print(f"      ❌ Hierarchical error: {e}")
            return False, 0.0

    def test_original(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug=False):
        """Test original p-bit method. Returns (success, runtime) tuple."""
        try:
            start_time = time.time()
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                use_hierarchical=False,
                block_size=block_size,
                use_sparse_updates=use_sparse_updates,
                timing_variance=timing_variance,
                intensity_variance=intensity_variance,
                offset_variance=offset_variance,
                spins_per_block=spins_per_block,
                beta_start=beta_start,
                beta_end=beta_end,
                max_flips_per_block=max_flips_per_block,
                initial_temperature=initial_temperature,
                temperature_decay_rate=temperature_decay_rate,
                annealing_schedule_type=annealing_schedule_type,
                num_trotters=num_trotters,
                gamma=gamma,
                total_annealing_steps=total_annealing_steps
            )
            runtime = time.time() - start_time

            min_energy = float(min(sampleset.record.energy))
            success = abs(min_energy - optimal_energy) < 1e-6  # Exact match

            print(f"      ⏱️  {runtime:.1f}s, Min energy: {min_energy:.1f}")
            print(f"      ✅ Success: {success}")

            if debug and hasattr(sampleset, 'record'):
                energies = sampleset.record.energy
                if len(energies) > 0:
                    print(f"      🔍 Energy range: {min(energies):.1f} to {max(energies):.1f}")
                    print(f"      🔍 Unique energies: {len(set(energies))}")
                    # Show distribution of top 5 energies
                    sorted_energies = sorted(set(energies))[:5]
                    print(f"      🔍 Best energies: {sorted_energies}")

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
        if problem is not None and idx != problem:
            continue
        num_variables = len(h)
        num_reads, num_sweeps, timeout_seconds = get_test_params(num_variables)

        print(f"\n🧪 Problem {idx}: {description}")
        print(f"   Variables: {num_variables}, Couplings: {len(J)}, Optimal GSE: {optimal_energy}")
        print(f"   📊 Config: {num_reads} reads, {num_sweeps} sweeps, {timeout_seconds}s timeout")

        if debug:
            print(f"   🔍 Debug: h={dict(list(h.items())[:5])}..." if len(h) > 5 else f"   🔍 Debug: h={h}")
            print(f"   🔍 Debug: J has {len(J)} couplings")
            if J:
                sample_couplings = list(J.items())[:3]
                print(f"   🔍 Debug: Sample couplings: {sample_couplings}")

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
                success, runtime = test_original(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug)
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
                success, runtime = test_hierarchical(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug)
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
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debugging output")
    parser.add_argument("--problem", type=int, default=None,
                        help="Test only specific problem number (0-based indexing)")

    # New Metal performance tuning parameters
    parser.add_argument("--block-size", type=int, default=None,
                        help="Block size for hierarchical updates (default: auto-scaled)")
    parser.add_argument("--use-sparse-updates", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help="Enable sparse incremental field updates (default: True)")
    parser.add_argument("--timing-variance", type=float, default=0.1,
                        help="Timing variance for P-bit updates (default: 0.1)")
    parser.add_argument("--intensity-variance", type=float, default=0.1,
                        help="Intensity variance for P-bit updates (default: 0.1)")
    parser.add_argument("--offset-variance", type=float, default=0.1,
                        help="Offset variance for P-bit updates (default: 0.1)")
    parser.add_argument("--spins-per-block", type=int, default=96,
                        help="Spins per block for P-bit parallel updates (default: 96)")
    parser.add_argument("--beta-start", type=float, default=0.01,
                        help="Starting inverse temperature for annealing (default: 0.01)")
    parser.add_argument("--beta-end", type=float, default=15.0,
                        help="Ending inverse temperature for annealing (default: 15.0)")
    parser.add_argument("--max-flips-per-block", type=int, default=None,
                        help="Maximum expected spin flips per block (default: auto-calculated)")

    # New SQA convergence parameters from literature review
    parser.add_argument("--initial-temperature", type=float, default=None,
                        help="Initial temperature for annealing (default: auto-scaled)")
    parser.add_argument("--temperature-decay-rate", type=float, default=None,
                        help="Temperature decay rate per MC step (default: auto)")
    parser.add_argument("--annealing-schedule-type", type=str, default="logspace",
                        choices=["linear", "exponential", "cosine", "logspace"],
                        help="Type of annealing schedule (default: logspace)")
    parser.add_argument("--num-trotters", type=int, default=None,
                        help="Number of trotters/replicas (default: auto from num_reads)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Transverse field strength for quantum annealing (default: 1.0)")
    parser.add_argument("--total-annealing-steps", type=int, default=None,
                        help="Total annealing steps (default: num_sweeps)")

    args = parser.parse_args()

    test_metal_sampler(
        skip=args.skip, retry=args.retry, reads=args.reads, sweeps=args.sweeps,
        debug=args.debug, problem=args.problem, block_size=args.block_size,
        use_sparse_updates=args.use_sparse_updates, timing_variance=args.timing_variance,
        intensity_variance=args.intensity_variance, offset_variance=args.offset_variance,
        spins_per_block=args.spins_per_block, beta_start=args.beta_start,
        beta_end=args.beta_end, max_flips_per_block=args.max_flips_per_block,
        initial_temperature=args.initial_temperature,
        temperature_decay_rate=args.temperature_decay_rate,
        annealing_schedule_type=args.annealing_schedule_type,
        num_trotters=args.num_trotters,
        gamma=args.gamma,
        total_annealing_steps=args.total_annealing_steps
    )