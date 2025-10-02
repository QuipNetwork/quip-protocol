#!/usr/bin/env python3
"""Metal Pure SA baseline parameter testing tool."""
import argparse
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce, evaluate_sampleset, calculate_diversity
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY

import importlib.util
spec = importlib.util.spec_from_file_location("metal_sa_pure", Path(__file__).parent.parent / "GPU" / "metal_sa_pure.py")
metal_sa_pure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metal_sa_pure)
PureMetalSASampler = metal_sa_pure.PureMetalSASampler


def metal_baseline_test(timeout_minutes=10.0, output_file=None, only_label=None, experimental=False):
    """Test Metal Pure SA performance with baseline format and evaluation logic.

    Args:
        experimental: If True, use latest experimental phase (currently Phase 5: multiple replicas)
    """
    mode_str = "Experimental (Phase 6: Replica Exchange)" if experimental else "Baseline (Sequential)"
    print(f"🔬 Metal Pure SA Baseline Parameter Test ({mode_str})")
    print("=" * 50)
    print(f"⏰ Timeout: {timeout_minutes} minutes")

    # Initialize sampler
    try:
        metal_sampler = PureMetalSASampler()
        print(f"✅ Metal Pure SA sampler ready ({mode_str})")
    except Exception as e:
        print(f"❌ Metal Pure SA sampler failed: {e}")
        return None

    # Get topology
    topology_graph = DEFAULT_TOPOLOGY.graph
    nodes = list(topology_graph.nodes())
    edges = list(topology_graph.edges())

    # Generate test problem
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges)

    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")

    # Test configurations - optimized for GPU Pure SA performance
    # GPU favors more reads, fewer sweeps (opposite of CPU!)
    # Optimal: 256-512 reads, 256-512 sweeps for best throughput
    test_configs = [
        (256, 256, "Light"),    # Fast, good quality (7.9s, -14254 energy)
        (512, 256, "Medium"),   # Optimal balance (7.7s, -14260 energy) ⭐ RECOMMENDED
        (1024, 256, "Heavy"),   # Best quality (12.0s, -14332 energy)
        (2048, 128, "Ultra"),   # Maximum quality (22.2s, -14336 energy)
    ]

    # Optional filter: run only the requested label
    if only_label:
        available_labels = [desc for _, _, desc in test_configs]
        filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
        if not filtered:
            print(f"⚠️ No test config matched --only {only_label!r}; available: {available_labels}")
            return None
        test_configs = filtered

    print(f"\n🧪 Testing Metal Pure SA configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': 'metal-pure-sa',
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': 12345
        },
        'tests': []
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()

    # Use deterministic seed sequence for reproducible comparisons
    import random
    random.seed(42)
    test_nonces = [random.randint(0, 2**32 - 1) for _ in range(len(test_configs))]

    for idx, (sweeps, reads, desc) in enumerate(test_configs):
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads")

        try:
            # Generate problem with deterministic nonce
            nonce = test_nonces[idx]
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            start_time = time.time()
            if experimental:
                # Use latest experimental phase (currently Phase 6: replica exchange)
                # Use num_replicas=reads to match the number of samples
                sampleset = metal_sampler.sample_ising_with_replica_exchange(
                    h=h, J=J,
                    num_replicas=reads,
                    num_sweeps=16,  # sweeps per exchange
                    num_exchanges=sweeps // 16  # total exchanges
                )
            else:
                # Use baseline (sequential)
                sampleset = metal_sampler.sample_ising(
                    h=h, J=J,
                    num_reads=reads,
                    num_sweeps=sweeps
                )
            runtime = time.time() - start_time

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"  🎯 min_energy = {min_energy:.1f}")
            print(f"  📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")

            # Use evaluate_sampleset to get diversity and num_solutions
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600
            )

            salt = b"test_salt_metal_baseline_pure_sa"
            prev_timestamp = int(time.time()) - 600

            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonce, salt,
                prev_timestamp, start_time, f"metal-pure-sa-{sweeps}-{reads}", "Metal"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Calculate diversity of top 10 solutions by energy
            solutions = list(sampleset.record.sample)
            energies = list(sampleset.record.energy)

            solution_energy_pairs = list(zip(solutions, energies))
            solution_energy_pairs.sort(key=lambda x: x[1])
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

            # Energy target analysis
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
            if runtime > timeout_seconds * 0.8:
                print(f"  ⏰ Single test approaching timeout, stopping further tests")
                break

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    total_runtime = time.time() - total_start_time
    print(f"\n📊 Pure SA Metal Baseline Summary (total time: {total_runtime/60:.1f} min):")
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
    parser = argparse.ArgumentParser(description='Metal Pure SA baseline parameter testing tool')
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
        '--only',
        type=str,
        help='Run only the config with this description (e.g., "Light")'
    )
    parser.add_argument(
        '--experimental',
        action='store_true',
        help='Use experimental features (currently Phase 5: multiple replicas)'
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
        mode_suffix = "_experimental" if args.experimental else ""
        output_file = f"metal_pure_sa_baseline_results{mode_suffix}_{timestamp}.json"

    # Run test
    metal_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        only_label=args.only,
        experimental=args.experimental
    )

    print(f"\n✅ Metal Pure SA baseline test complete!")


if __name__ == "__main__":
    main()
