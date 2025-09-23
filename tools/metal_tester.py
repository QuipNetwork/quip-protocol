#!/usr/bin/env python3
"""Metal Parallel Tempering performance tester with known optimal energies."""

import sys
import time
import argparse
from pathlib import Path

from basic_ising_problems import BASIC_ISING_PROBLEMS
import dimod

try:
    from GPU.metal_sampler import MetalSampler
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


def test_metal_sampler(skip=0, retry=3, reads=None, sweeps=None, debug=False, problem=None, num_replicas=None, swap_interval=15, T_min=0.1, T_max=5.0, cooling_factor=0.999, spin_updates_per_sweep=None, parallel_spin_updates=True, synthetic_regular_n=None, synthetic_regular_degree=None, sample_interval=None):
    """Test Metal Parallel Tempering sampler on EA-compatible problems only."""
    print("🔬 Metal Parallel Tempering Performance Tester")
    print("=" * 70)
    print("✅ Metal sampler now uses the provided h,J problems from basic_ising_problems.py")

    # Initialize samplers
    metal_sampler = None

    # Check availability and initialize
    metal_available = METAL_AVAILABLE

    if metal_available:
        try:
            metal_sampler = MetalSampler("mps")
            print("✅ Metal Parallel Tempering kernel sampler ready")
        except Exception as e:
            print(f"❌ Metal PT sampler failed: {e}")
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
                timeout = 20.0  # Increased for PT evaluation
            elif num_variables <= 64:
                timeout = 30.0  # Increased for PT evaluation
            elif num_variables <= 128:
                timeout = 45.0
            else:  # 256+
                timeout = 60.0
            return reads, sweeps, timeout
        elif reads is not None:
            # Override reads only, scale sweeps and timeout
            if num_variables <= 16:
                return reads, 1000, 20.0
            elif num_variables <= 64:
                return reads, 500, 30.0
            elif num_variables <= 128:
                return reads, 250, 45.0
            else:  # 256+
                return reads, 100, 60.0
        elif sweeps is not None:
            # Override sweeps only, scale reads and timeout
            if num_variables <= 16:
                return 100, sweeps, 20.0
            elif num_variables <= 64:
                return 50, sweeps, 30.0
            elif num_variables <= 128:
                return 25, sweeps, 45.0
            else:  # 256+
                return 10, sweeps, 60.0
        else:
            # Default scaling
            if num_variables <= 16:
                return 100, 1000, 20.0
            elif num_variables <= 64:
                return 50, 500, 30.0
            elif num_variables <= 128:
                return 25, 250, 45.0
            else:  # 256+
                return 10, 100, 60.0

    print(f"🎯 Total problems available: {len(BASIC_ISING_PROBLEMS)}")
    print(f"🔍 EA-compatible problems: {len(EA_COMPATIBLE_PROBLEMS)}")

    if synthetic_regular_n is not None and synthetic_regular_degree is not None:
        print(f"🧪 Using synthetic regular graph: N={synthetic_regular_n}, degree={synthetic_regular_degree}")
        # No EA checks needed for synthetic test
        ea_indices = []
    else:
        if len(EA_COMPATIBLE_PROBLEMS) == 0:
            print("❌ No EA-compatible problems found! Metal sampler requires:")
            print("   - Cubic lattice sizes (N = L³): 8, 27, 64, 125, 216...")
            print("   - All linear biases h = 0")
            print("   - All couplings J = ±1")
            print("   - SPIN variables")
            return
        ea_indices = [idx for idx, _, _, _, _ in EA_COMPATIBLE_PROBLEMS]
        print(f"🔧 Compatible problem indices: {ea_indices}")

    if skip > 0:
        print(f"⏭️  Skipping first {skip} problems")
    if retry > 0:
        print(f"🔄 Retry limit: {retry} attempts per test")
    if any(v is not None for v in (reads, sweeps, sample_interval)):
        override_info = []
        if reads is not None:
            override_info.append(f"reads={reads}")
        if sweeps is not None:
            override_info.append(f"sweeps={sweeps}")
        if sample_interval is not None:
            override_info.append(f"sample_interval={sample_interval}")
        print(f"⚙️  Parameter overrides: {', '.join(override_info)}")

    if problem is not None and not (synthetic_regular_n is not None and synthetic_regular_degree is not None):
        # Check if the specified problem is EA-compatible
        if problem in ea_indices:
            print(f"🎯 Testing only problem {problem} (EA-compatible)")
        else:
            print(f"❌ Problem {problem} is not EA-compatible with Metal sampler")
            return

    results = []

    def test_parallel_tempering(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug=False, spin_updates_override=None):
        """Test Parallel Tempering method. Returns (success, runtime) tuple."""
        try:
            start_time = time.time()

            # Use default sample_ising method (unified GPU kernel)
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                num_replicas=num_replicas,
                swap_interval=swap_interval,
                T_min=T_min,
                T_max=T_max,
                sample_interval=sample_interval
            )

            runtime = time.time() - start_time

            min_energy = float(min(sampleset.record.energy))
            success = True if optimal_energy is None else (abs(min_energy - optimal_energy) < 1e-6)

            print(f"      ⏱️  {runtime:.3f}s, Min energy: {min_energy:.1f}")
            print(f"      ✅ Success: {success}")
            print(f"      🔍 Debug: sampleset has {len(sampleset.record.energy)} samples")
            print(f"      🔍 Debug: energy range {min(sampleset.record.energy):.1f} to {max(sampleset.record.energy):.1f}")

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
            print(f"      ❌ Parallel Tempering error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0


    # Synthetic regular graph generator (simple configuration model with retries)
    def _generate_random_regular_spin_bqm(n, d, max_attempts=50):
        import random
        if (n * d) % 2 != 0:
            raise ValueError("n*d must be even for a d-regular graph")
        for _ in range(max_attempts):
            stubs = [i for i in range(n) for _ in range(d)]
            random.shuffle(stubs)
            edges = set()
            ok = True
            while stubs:
                a = stubs.pop()
                # find a partner b that is not a and not already connected
                found = False
                for idx in range(len(stubs) - 1, -1, -1):
                    b = stubs[idx]
                    if b == a or (min(a, b), max(a, b)) in edges:
                        continue
                    # pair a-b
                    stubs.pop(idx)
                    edges.add((min(a, b), max(a, b)))
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if ok and len(edges) == (n * d) // 2:
                h = {i: 0.0 for i in range(n)}
                import random as _r
                J = {e: float(_r.choice([-1, 1])) for e in edges}
                return h, J
        raise RuntimeError("Failed to generate d-regular simple graph after retries")

    # Choose problem set
    if synthetic_regular_n is not None and synthetic_regular_degree is not None:
        try:
            h_syn, J_syn = _generate_random_regular_spin_bqm(synthetic_regular_n, synthetic_regular_degree)
        except Exception as e:
            print(f"❌ Synthetic graph generation failed: {e}")
            return
        problems_to_test = [(-1, h_syn, J_syn, None, f"{synthetic_regular_degree}-regular random graph (N={synthetic_regular_n})")]
    else:
        problems_to_test = EA_COMPATIBLE_PROBLEMS
        if problem is not None:
            # Filter to only the specified problem if it's EA-compatible
            problems_to_test = [(idx, h, J, opt_e, desc) for idx, h, J, opt_e, desc in EA_COMPATIBLE_PROBLEMS if idx == problem]
            if not problems_to_test:
                return  # Already handled above

    for test_idx, (original_idx, h, J, optimal_energy, description) in enumerate(problems_to_test):
        if test_idx < skip:
            continue

        num_variables = len(h)
        num_reads, num_sweeps, timeout_seconds = get_test_params(num_variables)

        print(f"\n🧪 Problem {original_idx}: {description}")
        print(f"   Variables: {num_variables}, Couplings: {len(J)}, Optimal GSE: {optimal_energy}")
        print(f"   📊 Config: {num_reads} reads, {num_sweeps} sweeps, {timeout_seconds}s timeout")

        if debug:
            print(f"   🔍 Debug: h={dict(list(h.items())[:5])}..." if len(h) > 5 else f"   🔍 Debug: h={h}")
            print(f"   🔍 Debug: J has {len(J)} couplings")
            if J:
                sample_couplings = list(J.items())[:3]
                print(f"   🔍 Debug: Sample couplings: {sample_couplings}")

        problem_results = {
            'problem_idx': original_idx,
            'description': description,
            'num_variables': num_variables,
            'num_couplings': len(J),
            'optimal_energy': optimal_energy,
            'parallel_tempering': None
        }

        # Test Parallel Tempering
        pt_success = False
        pt_runtime = 0.0
        if metal_available and metal_sampler is not None:
            print("   🔄 Testing Parallel Tempering...")
            for attempt in range(retry + 1):  # +1 for initial attempt
                if attempt > 0:
                    print(f"   🔄 PT retry {attempt}/{retry}...")
                success, runtime = test_parallel_tempering(h, J, num_reads, num_sweeps, timeout_seconds, optimal_energy, debug, spin_updates_override=spin_updates_per_sweep)
                if success:
                    pt_success = True
                    pt_runtime = runtime
                    break

            if not pt_success:
                print(f"   ❌ Parallel Tempering failed after {retry + 1} attempts")
                break

        # Store results with timing information
        problem_results['parallel_tempering'] = {'success': pt_success, 'runtime': pt_runtime}

        results.append(problem_results)

        # Check if PT succeeded
        if not problem_results['parallel_tempering']['success']:
            print(f"   ❌ Parallel Tempering failed on problem {original_idx}")
            break

    # Summary
    print(f"\n📊 Summary:")
    print("=" * 50)
    print(f"Problems tested: {len(results)}")

    if results:
        last_result = results[-1]
        print(f"Last problem: {last_result['description']}")
        print(f"Parallel Tempering: {last_result['parallel_tempering']['runtime']:.1f}s, Success: {last_result['parallel_tempering']['success']}")

        # Success rate summary
        total_problems = len(results)
        pt_successes = sum(1 for r in results if r['parallel_tempering']['success'])
        print(f"Success rate - Parallel Tempering: {pt_successes}/{total_problems}")

        # Performance summary for successful runs
        successful_problems = [r for r in results if r['parallel_tempering']['success']]
        if successful_problems:
            total_pt_time = sum(r['parallel_tempering']['runtime'] for r in successful_problems)
            avg_pt_time = total_pt_time / len(successful_problems)
            print(f"Average time - Parallel Tempering: {avg_pt_time:.1f}s")
            print(f"Total problems solved: {len(successful_problems)}/{len(results)}")
    # Clean up
    if metal_available and metal_sampler is not None:
        metal_sampler.close()

    print("\n✅ Metal tester complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Metal Parallel Tempering sampler on known problems")
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

    # Parallel Tempering performance tuning parameters
    parser.add_argument("--num-replicas", type=int, default=None,
                        help="Number of temperature replicas (default: auto-selected based on num_reads)")
    parser.add_argument("--swap-interval", type=int, default=15,
                        help="Steps between replica exchanges (default: 15)")
    parser.add_argument("--T-min", type=float, default=0.1,
                        help="Minimum temperature (default: 0.1)")
    parser.add_argument("--T-max", type=float, default=5.0,
                        help="Maximum temperature (default: 5.0)")
    parser.add_argument("--cooling-factor", type=float, default=0.999,
                        help="Temperature cooling factor per step (default: 0.999)")
    parser.add_argument("--sample-interval", type=int, default=None,
                        help="Collect a sample every K sweeps (default: heuristic if omitted)")

    parser.add_argument("--spin-updates-per-sweep", type=int, default=None,
                        help="Number of spin updates per sweep (default: N)")
    parser.add_argument("--parallel-spin-updates", action="store_true", default=True,
                        help="Use parallel spin updates with double-buffering (default: True)")
    parser.add_argument("--sequential-spin-updates", dest="parallel_spin_updates", action="store_false",
                        help="Use sequential spin updates (disable parallel optimization)")


    # Synthetic graph options
    parser.add_argument("--synthetic-regular-n", type=int, default=None,
                        help="Generate a synthetic d-regular random graph with N nodes")
    parser.add_argument("--synthetic-regular-degree", type=int, default=None,
                        help="Degree d for the synthetic regular random graph")

    args = parser.parse_args()

    test_metal_sampler(
        skip=args.skip, retry=args.retry, reads=args.reads, sweeps=args.sweeps,
        debug=args.debug, problem=args.problem, num_replicas=args.num_replicas,
        swap_interval=args.swap_interval, T_min=args.T_min, T_max=args.T_max,
        cooling_factor=args.cooling_factor, spin_updates_per_sweep=args.spin_updates_per_sweep,
        parallel_spin_updates=args.parallel_spin_updates,
        synthetic_regular_n=args.synthetic_regular_n, synthetic_regular_degree=args.synthetic_regular_degree,
        sample_interval=args.sample_interval
    )