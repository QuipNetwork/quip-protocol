#!/usr/bin/env python3
"""Metal SA baseline parameter testing tool."""
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
from dwave_topologies.topologies.json_loader import load_topology
from dwave_topologies.embedded_topology import create_embedded_topology

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
    if topology:
        # Auto-detect embedding files by .embed.json.gz extension
        if topology.endswith('.embed.json.gz'):
            print(f"🔗 Loading embedded topology: {topology}")
            # Parse Z(m,t) from filename like "zephyr_z9_t2.embed.json.gz"
            import os
            filename = os.path.basename(topology)
            if filename.startswith("zephyr_z"):
                parts = filename.replace("zephyr_z", "").replace(".embed.json.gz", "").split("_t")
                topology_name = f"Z({parts[0]},{parts[1]})"
                embedded_topo = create_embedded_topology(topology_name)
                nodes = embedded_topo.nodes
                edges = embedded_topo.edges
                topology_desc = f"{topology_name} embedded ({len(nodes)} qubits, {len(edges)} couplers)"
            else:
                raise ValueError(f"Cannot parse embedding filename: {filename}")
        else:
            print(f"📂 Loading topology: {topology}")
            topo_obj = load_topology(topology)
            nodes = list(topo_obj.graph.nodes) if hasattr(topo_obj, 'graph') else topo_obj.nodes
            edges = list(topo_obj.graph.edges) if hasattr(topo_obj, 'graph') else topo_obj.edges
            topology_name = getattr(topo_obj, 'solver_name', 'unknown')
            topology_desc = f"{topology_name} ({len(nodes)} nodes, {len(edges)} edges)"
    else:
        print(f"✨ Using default topology (Advantage2_system1)")
        topo_obj = DEFAULT_TOPOLOGY
        nodes = list(topo_obj.graph.nodes) if hasattr(topo_obj, 'graph') else topo_obj.nodes
        edges = list(topo_obj.graph.edges) if hasattr(topo_obj, 'graph') else topo_obj.edges
        topology_desc = f"{topo_obj.solver_name} ({len(nodes)} nodes, {len(edges)} edges)"

    print(f"📐 Topology: {topology_desc}")

    # Generate test problem with h_values
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

    # Show h distribution
    h_vals_set = sorted(set(h.values()))
    h_counts = {v: list(h.values()).count(v) for v in h_vals_set}
    h_dist_str = ", ".join([f"{v}: {h_counts[v]} ({100*h_counts[v]/len(h):.1f}%)" for v in h_vals_set])
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    print(f"   h distribution: {h_dist_str}")

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
    if only_label:
        available_labels = [desc for _, _, desc in test_configs]
        filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
        if not filtered:
            print(f"⚠️ No test config matched --only {only_label!r}; available: {available_labels}")
            return None
        test_configs = filtered

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
    import random
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

            # Collect stats from all models
            all_min_energies = []
            all_avg_energies = []
            for sampleset in samplesets:
                energies = list(sampleset.record.energy)
                all_min_energies.append(float(min(energies)))
                all_avg_energies.append(float(sum(energies) / len(energies)))

            # Use first sampleset for detailed analysis
            sampleset = samplesets[0]
            energies = list(sampleset.record.energy)
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

            # Use evaluate_sampleset to get diversity and num_solutions
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600
            )

            salt = b"test_salt_metal_baseline_sa"
            prev_timestamp = int(time.time()) - 600

            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonces[0], salt,
                prev_timestamp, start_time, f"metal-sa-{sweeps}-{reads}", "Metal"
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
    print(f"\n📊 Metal SA Baseline Summary (total time: {total_runtime/60:.1f} min):")
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
    parser = argparse.ArgumentParser(description='Metal SA baseline parameter testing tool')
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
        help='Quick test mode (only Light test)'
    )
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Extended test mode (30 minute timeout)'
    )
    parser.add_argument(
        '--only',
        type=str,
        help='Run only the config with this description (e.g., "Light Metal")'
    )
    parser.add_argument(
        '--h-values',
        type=str,
        default='-1,0,1',
        help='Comma-separated h field values (default: -1,0,1). Use "0" for h=0 baseline.'
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Number of models to process in parallel (default: auto-detect GPU cores, typically 40)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        help='Topology: Z(9,2), Advantage2_system1, file path, or *.embed.json.gz for embedded. '
             'Default: Advantage2_system1'
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
