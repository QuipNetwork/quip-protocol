#!/usr/bin/env python3
"""Metal SA baseline parameter testing tool."""
import json
import random
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce
from tools.baseline_utils import (
    build_baseline_argparser,
    classify_energy,
    evaluate_baseline_sampleset,
    filter_configs_by_label,
    load_baseline_topology,
    print_problem_summary,
    print_results_summary,
)

from GPU.metal_sa import MetalSASampler
from GPU.metal_miner import get_gpu_core_count


def metal_baseline_test(timeout_minutes=10.0, output_file=None, only_label=None, h_values=None, num_models=1, topology=None):
    """Test Metal SA performance with baseline format and evaluation logic.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        only_label: Run only specific config (e.g., "Light Metal")
        h_values: List of allowed h field values
        num_models: Number of parallel models to run
        topology: Topology to use. Can be:
                  - Z(m,t) format for perfect Zephyr topology (e.g., "Z(9,2)")
                  - Hardware name (e.g., "Advantage2_system1")
                  - File path to topology JSON (e.g., "path/to/topology.json.gz")
                  - File path to embedding (e.g., "path/to/*.embed.json.gz") - auto-detected
                  Default: Advantage2_system1
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

    print(f"🔬 Metal SA Baseline Parameter Test")
    print("=" * 50)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎲 h_values: {h_values}")

    # Initialize sampler
    try:
        metal_sampler = MetalSASampler()
        print(f"✅ Metal SA sampler ready")
    except Exception as e:
        print(f"❌ Metal SA sampler failed: {e}")
        return None

    # Get topology
    nodes, edges, topology_desc = load_baseline_topology(
        topology_arg=topology,
    )
    print(f"📐 Topology: {topology_desc}")

    # Generate test problem with h_values
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

    # Show h distribution
    print_problem_summary(h, J)

    # Test configurations - matching CPU baseline for fair comparison
    test_configs = [
        (256, 64, "Light Metal"),
        (512, 100, "Low Metal"),
        (1024, 100, "Medium Metal"),
        (2048, 150, "High Metal"),
        (4096, 200, "Very High Metal"),
        (8192, 200, "Max Metal")
    ]

    # Optional filter: run only the requested label
    test_configs = filter_configs_by_label(test_configs, only_label)
    if test_configs is None:
        return None

    print(f"\n🧪 Testing Metal SA configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': 'metal-sa',
        'topology': topology_desc,
        'topology_arg': topology if topology else "default",
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
    random.seed(42)
    test_nonces = [random.randint(0, 2**32 - 1) for _ in range(len(test_configs))]

    for idx, (sweeps, reads, desc) in enumerate(test_configs):
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads, {num_models} models")

        try:
            # Generate problems with deterministic nonces
            h_list = []
            J_list = []
            nonces = []

            for _ in range(num_models):
                nonce = test_nonces[idx]
                nonces.append(nonce)
                h, J = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)
                h_list.append(h)
                J_list.append(J)

            start_time = time.time()
            # Process models in batch
            samplesets = metal_sampler.sample_ising(
                h=h_list, J=J_list,
                num_reads=reads,
                num_sweeps=sweeps
            )
            runtime = time.time() - start_time
            throughput = num_models / runtime  # models per second

            # Collect stats from all models; capture first sampleset's energies for reuse
            all_min_energies = []
            all_avg_energies = []
            energies = []
            for i, ss in enumerate(samplesets):
                e = list(ss.record.energy)
                if i == 0:
                    energies = e
                all_min_energies.append(float(min(e)))
                all_avg_energies.append(float(sum(e) / len(e)))

            # Use first sampleset for detailed analysis
            sampleset = samplesets[0]
            # energies already set from loop above
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime:.2f}s ({num_models} models)")
            if num_models > 1:
                print(f"  🚀 Throughput: {throughput:.2f} models/second")
                print(f"  🎯 Best energy: {min(all_min_energies):.1f} (across {num_models} models)")
            else:
                print(f"  🎯 min_energy = {min_energy:.1f}")
            print(f"  📊 Avg energy (first model): {avg_energy:.1f} (±{std_energy:.1f})")

            # Evaluate the sampleset
            eval_fields = evaluate_baseline_sampleset(
                sampleset, nodes, edges, nonces[0],
                start_time, f"metal-sa-{sweeps}-{reads}", "Metal",
                b"test_salt_metal_baseline_sa",
            )

            # Energy target analysis
            target_reached = classify_energy(min_energy)
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
                **eval_fields,
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
    results['_total_runtime_seconds'] = total_runtime
    print_results_summary(results, "Metal SA Baseline Summary")

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


def main():
    """Main function with command line argument parsing."""
    parser = build_baseline_argparser(
        'Metal SA baseline parameter testing tool',
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Number of models to process in parallel (default: auto-detect GPU cores, typically 40)',
    )

    args = parser.parse_args()

    # Auto-detect GPU core count if not specified
    if args.num_models is None:
        try:
            num_models = get_gpu_core_count()
        except Exception as e:
            print(f"⚠️ Could not detect GPU cores ({e}), defaulting to 40 models")
            num_models = 40
    else:
        num_models = args.num_models

    # Parse h_values
    h_values = [float(v.strip()) for v in args.h_values.split(',')]

    # Handle preset timeouts and filters
    only_label = args.only
    if args.quick:
        timeout = 10.0
        only_label = "Light Metal"  # Force Light test only
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"metal_sa_baseline_results_{timestamp}.json"

    # Run test
    metal_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        only_label=only_label,
        h_values=h_values,
        num_models=num_models,
        topology=args.topology
    )

    print(f"\n✅ Metal SA baseline test complete!")


if __name__ == "__main__":
    main()
