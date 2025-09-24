#!/usr/bin/env python3
"""Metal GPU baseline parameter testing tool."""
import argparse
import sys
import time
import json
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce, evaluate_sampleset, calculate_diversity
from shared.block_requirements import BlockRequirements

try:
    from GPU.metal_kernel_sampler import MetalKernelDimodSampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# Optional: new parallel sampler for A/B testing
try:
    from GPU.metal_kernel_sampler_parallel import MetalKernelDimodSamplerParallel
    PARALLEL_AVAILABLE = True
except Exception:
    PARALLEL_AVAILABLE = False


def create_large_problem(size=4593):
    """Create a large Ising problem to test memory usage."""
    # Create a problem with specified size that approximates the topology used in production
    num_vars = size

    # Create a grid-like structure with some random couplings to make it more interesting
    h = {i: 0.0 for i in range(num_vars)}

    # Create a pattern that mimics real-world quantum annealing problems
    J = {}

    # Add some grid-like couplings (simplified)
    for i in range(num_vars):
        # Connect to neighbors in a grid pattern (simplified)
        if i + 1 < num_vars:
            J[(i, i+1)] = -1.0
        if i + 64 < num_vars:  # Connect to row below (approximate grid)
            J[(i, i+64)] = -1.0

    # Add some random couplings to make it more complex
    import random
    for _ in range(1000):  # Add some random couplings
        i = random.randint(0, num_vars - 1)
        j = random.randint(0, num_vars - 1)
        if i != j and (i,j) not in J and (j,i) not in J:
            # Use binary ±1 couplings to conform to Ising model requirements
            J[(i, j)] = float(random.choice([-1, 1]))

    return h, J


def metal_baseline_test(timeout_minutes=10.0, output_file=None, num_replicas=None, swap_interval=15, T_min=0.1, T_max=5.0, large_problem=False, only_label=None, sampler_choice: str = "original"):
    """Test Metal GPU performance with CPU baseline format and evaluation logic."""
    print("🔬 Metal GPU Baseline Parameter Test (Kernel-Only)")
    print("=" * 50)
    print(f"⏰ Timeout: {timeout_minutes} minutes")

    # Select sampler (production or parallel)
    metal_sampler = None
    sampler_type = "unknown"

    if sampler_choice == "parallel":
        if not PARALLEL_AVAILABLE:
            print("❌ Parallel sampler not available")
            return None
        try:
            metal_sampler = MetalKernelDimodSamplerParallel("mps")
            nodes = metal_sampler.nodes
            edges = metal_sampler.edges
            sampler_type = "metal-parallel"
            print("✅ Parallel sampler ready (new 2D kernel path)")
        except Exception as e:
            print(f"❌ Parallel sampler failed: {e}")
            return None
    else:
        if METAL_AVAILABLE:
            try:
                metal_sampler = MetalKernelDimodSampler("mps")
                nodes = metal_sampler.nodes
                edges = metal_sampler.edges
                sampler_type = "metal-pt"
                print("✅ Metal Parallel Tempering sampler ready (production kernel)")
            except Exception as e:
                print(f"❌ Metal Parallel Tempering sampler failed: {e}")
                return None
        else:
            print("❌ Metal Parallel Tempering sampler not available")
            return None

    # Choose problem size
    if large_problem:
        print("🧪 Using large 4593-variable problem for memory testing")
        h, J = create_large_problem(4593)
    else:
        # Initial problem setup to show problem size
        seed = 12345  # Fixed seed for reproducible results
        h, J = generate_ising_model_from_nonce(seed, nodes, edges)

    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")

    # Test configurations - optimized for Metal Parallel Tempering performance
    if large_problem:
        # For memory testing with larger problems, use more aggressive configurations
        test_configs = [
            (512, 32, "Small Large PT"),      # Start with smaller sweep count for memory testing
            (1024, 32, "Medium Large PT"),
            (2048, 32, "Large PT"),
        ]
    else:
        # Standard configurations for small problems
        test_configs = [
            (256, 64, "Light"),
            (512, 100, "Low"),
            (1024, 100, "Medium"),
            (2048, 150, "High"),
            (4096, 200, "Very High"),
            (8192, 200, "Max")
        ]
        # Optional filter: run only the requested label (e.g., "Light")
        available_labels = [desc for _, _, desc in test_configs]
        if only_label:
            filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
            if not filtered:
                print(f"⚠️ No test config matched --only {only_label!r}; available: {available_labels}")
                return None
            test_configs = filtered


    print(f"\n🧪 Testing Metal Parallel Tempering configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': sampler_type,
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': 12345 if not large_problem else "large_problem"
        },
        'tests': []
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()

    for sweeps, reads, desc in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads")

        try:
            # Generate the Ising model first (like the real miners do)
            if large_problem:
                # For large problems, we want to make sure they're deterministic
                nonce = 1234567890  # Fixed for reproducible large problem tests
                h, J = create_large_problem(4593)

                # Validate binary couplings (±1)
                if any(not (abs(v - 1.0) < 1e-8 or abs(v + 1.0) < 1e-8) for v in J.values()):
                    raise ValueError("Non-binary coupling detected in baseline J; expected only ±1.")

            else:
                nonce = random.randint(0, 2**32 - 1)
                h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

                # Validate binary couplings (±1)
                if any(not (abs(v - 1.0) < 1e-8 or abs(v + 1.0) < 1e-8) for v in J.values()):
                    raise ValueError("Non-binary coupling detected in baseline J; expected only ±1.")


            start_time = time.time()
            sampleset = metal_sampler.sample_ising(
                h=h, J=J,
                num_reads=reads,
                num_sweeps=sweeps,
                num_replicas=num_replicas,
                swap_interval=swap_interval,
                T_min=T_min,
                T_max=T_max
            )
            runtime = time.time() - start_time

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"  🎯 min_energy = {min_energy:.1f}")
            print(f"  📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")

            # Use evaluate_sampleset to get diversity and num_solutions (same as CPU)
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty (allow positive energies)
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600  # 10 minutes
            )

            # Use the same nonce and generate test salt for evaluation
            salt = b"test_salt_metal_baseline"
            prev_timestamp = int(time.time()) - 600  # 10 minutes ago

            # Evaluate the sampleset
            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonce, salt,
                prev_timestamp, start_time, f"metal-baseline-{sweeps}-{reads}", "Metal"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Calculate diversity of top 10 solutions by energy (same as CPU)
            solutions = list(sampleset.record.sample)
            energies = list(sampleset.record.energy)

            # Sort solutions by energy and take top 10
            solution_energy_pairs = list(zip(solutions, energies))
            solution_energy_pairs.sort(key=lambda x: x[1])  # Sort by energy (ascending = better)
            top_10_solutions = [sol for sol, _ in solution_energy_pairs[:10]]

            top_10_diversity = calculate_diversity(top_10_solutions)
            print(f"  🌈 diversity (top 10) = {top_10_diversity:.3f}")

            if mining_result:
                diversity = mining_result.diversity
                num_solutions = mining_result.num_valid
                meets_requirements = True
                print(f"  🔢 num_solutions = {num_solutions}")
                print(f"  ✅ Meets mining requirements!")
            else:
                print(f"  ❌ Does not meet mining requirements")

            # Energy target analysis (same as CPU)
            target_reached = "none"
            if min_energy <= -15650:
                target_reached = "excellent"
            elif min_energy <= -15500:
                target_reached = "very_good"
            elif min_energy <= -15400:
                target_reached = "good"
            elif min_energy <= -15300:
                target_reached = "fair"

            if target_reached != "none":
                print(f"  🎖️  Quality: {target_reached}")

            test_result = {
                'description': desc,
                'num_sweeps': int(sweeps),
                'num_reads': int(reads),
                'runtime_seconds': float(runtime),
                'runtime_minutes': float(runtime / 60),
                'min_energy': min_energy,
                'avg_energy': avg_energy,
                'std_energy': std_energy,
                'target_reached': target_reached,
                'diversity': float(diversity),
                'diversity_top_10': float(top_10_diversity),
                'num_solutions': int(num_solutions),
                'meets_requirements': bool(meets_requirements)
            }
            results['tests'].append(test_result)

            # Individual test timeout check
            if runtime > timeout_seconds * 0.8:  # 80% of total timeout
                print(f"  ⏰ Single test approaching timeout, stopping further tests")
                break

        except Exception as e:
            print(f"  ❌ Error: {e}")
            break


    # Summary (same as CPU)
    total_runtime = time.time() - total_start_time
    print(f"\n📊 Parallel Tempering Metal Baseline Summary (total time: {total_runtime/60:.1f} min):")
    print("=" * 50)

    if results['tests']:
        # Best energy achieved
        best_result = min(results['tests'], key=lambda r: r['min_energy'])
        print(f"🏆 Best energy: {best_result['min_energy']:.1f}")
        print(f"   Required: {best_result['num_sweeps']} sweeps, {best_result['runtime_minutes']:.1f} min")

        # Time vs energy analysis
        print(f"\n⏱️ Time vs Energy Performance:")
        for result in results['tests']:
            quality = f"({result['target_reached']})" if result['target_reached'] != 'none' else ""
            print(f"  {result['runtime_minutes']:5.1f} min: {result['min_energy']:7.1f} energy {quality}")

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Parallel Tempering Metal GPU baseline parameter testing tool')
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=10.0,
        help='Timeout in minutes (default: 10.0)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (3 minute timeout)'
    )
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Extended test mode (30 minute timeout)'
    )
    parser.add_argument(
        '--large-problem',
        action='store_true',
        help='Use large 4593-variable problem for memory testing'
    )

    # Parallel Tempering algorithm control flags
    parser.add_argument(
        '--num-replicas',
        type=int,
        default=None,
        help='Number of temperature replicas (default: auto-selected based on num_reads)'
    )
    parser.add_argument(
        '--swap-interval',
        type=int,
        default=15,
        help='Steps between replica exchanges (default: 15)'
    )
    parser.add_argument(
        '--T-min',
        type=float,
        default=0.1,
        help='Minimum temperature (default: 0.1)'
    )
    parser.add_argument(
        '--T-max',
        type=float,
        default=5.0,
        help='Maximum temperature (default: 5.0)'
    )
    parser.add_argument(
        '--only',
        type=str,
        help='Run only the config with this description (e.g., "Light")'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        choices=['original', 'parallel'],
        default='original',
        help='Select sampler implementation (original production kernel or new parallel kernel)'
    )


    args = parser.parse_args()

    # Handle preset timeouts
    if args.quick:
        timeout = 3.0
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"metal_baseline_results_{timestamp}.json"

    # Run test
    metal_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        num_replicas=args.num_replicas,
        swap_interval=args.swap_interval,
        T_min=args.T_min,
        T_max=args.T_max,
        large_problem=args.large_problem,
        only_label=args.only,
        sampler_choice=args.sampler
    )

    print(f"\n✅ Parallel Tempering Metal baseline test complete!")


if __name__ == "__main__":
    main()
