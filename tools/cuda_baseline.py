#!/usr/bin/env python3
"""CUDA GPU baseline parameter testing tool using persistent kernel."""
import argparse
import sys
import time
import json
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from shared.quantum_proof_of_work import generate_ising_model_from_nonce, evaluate_sampleset, calculate_diversity
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.embedded_topology import create_embedded_topology
from dwave_topologies.topologies.json_loader import load_topology

try:
    from GPU.cuda_kernel import CudaKernelRealSA
    from GPU.cuda_sa import CudaSASamplerAsync, CudaKernelAdapter
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def cuda_baseline_test(timeout_minutes=10.0, output_file=None, only_label=None, h_values=None, use_embedding=None, topology_path=None):
    """Test CUDA GPU performance using CudaSASamplerAsync.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        only_label: Run only specific config (e.g., "Light CUDA")
        h_values: List of allowed h field values
        use_embedding: If specified, use embedded hardware topology instead of perfect topology.
                      Format: "Z(9,2)" for Z(9,2) embedding
        topology_path: Path to topology file (JSON or JSON.gz). Takes precedence over use_embedding.
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

    print("🔬 CUDA GPU Baseline Parameter Test (CudaSASamplerAsync)")
    print("=" * 60)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎲 h_values: {h_values}")

    if not CUDA_AVAILABLE:
        print("❌ CUDA not available")
        return None

    try:
        print("📦 Initializing CUDA sampler...")
        # Use larger ring size to support many parallel jobs
        # Max jobs = num_SMs * jobs_per_SM = ~48 * 4 = 192 for small jobs
        # Disable verbose output for production use
        kernel = CudaKernelRealSA(
            ring_size=256,
            max_threads_per_job=256,
            max_N=5000,
            debug_verbose=0,
            debug_kernel=0,
            debug_workers=0,
            verbose=False
        )
        adapter = CudaKernelAdapter(kernel)
        sampler = CudaSASamplerAsync(adapter)
        print("✅ CUDA sampler ready")
    except Exception as e:
        print(f"❌ CUDA sampler initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get topology
    if topology_path:
        print(f"📂 Loading topology from file: {topology_path}")
        topology = load_topology(topology_path)
        nodes = list(topology.graph.nodes) if hasattr(topology, 'graph') else topology.nodes
        edges = list(topology.graph.edges) if hasattr(topology, 'graph') else topology.edges
        topology_name = getattr(topology, 'solver_name', None) or getattr(topology, 'topology_shape', Path(topology_path).stem)
        topology_desc = f"{topology_name} from file ({len(nodes)} nodes, {len(edges)} edges)"
    elif use_embedding:
        print(f"🔗 Using embedded hardware topology: {use_embedding}")
        embedded_topo = create_embedded_topology(use_embedding)
        nodes = embedded_topo.nodes
        edges = embedded_topo.edges
        topology_desc = f"{use_embedding} embedded ({len(nodes)} qubits, {len(edges)} couplers)"
    else:
        print(f"✨ Using perfect topology (default)")
        nodes = list(DEFAULT_TOPOLOGY.graph.nodes)
        edges = list(DEFAULT_TOPOLOGY.graph.edges)
        topology_desc = f"perfect Z(9,2) ({len(nodes)} nodes, {len(edges)} edges)"

    sampler_type = "persistent-kernel"
    print(f"📐 Topology: {topology_desc}")

    # Initial problem setup to show problem size
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

    # Show h distribution
    h_vals_set = sorted(set(h.values()))
    h_counts = {v: list(h.values()).count(v) for v in h_vals_set}
    h_dist_str = ", ".join([f"{v}: {h_counts[v]} ({100*h_counts[v]/len(h):.1f}%)" for v in h_vals_set])
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    print(f"   h distribution: {h_dist_str}")
    
    # Test configurations - matching CPU baseline exactly
    test_configs = [
        (256, 64, "Light CUDA"),
        (512, 100, "Low CUDA"),
        (1024, 100, "Medium CUDA"),
        (2048, 150, "High CUDA"),
        (4096, 200, "Very High CUDA"),
        (8192, 200, "Max CUDA")
    ]

    # Optional filter: run only the requested label
    if only_label:
        available_labels = [desc for _, _, desc in test_configs]
        filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
        if not filtered:
            print(f"⚠️ No test config matched --only {only_label!r}; available: {available_labels}")
            return None
        test_configs = filtered

    print(f"\n🧪 Testing CUDA configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': sampler_type,
        'topology': topology_desc,
        'use_embedding': use_embedding if use_embedding else "none",
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': seed
        },
        'tests': []
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()

    # Query GPU capabilities
    num_sms = sampler.get_num_sms()
    print(f"🔧 GPU has {num_sms} streaming multiprocessors (SMs)")

    for sweeps, reads, desc in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        # Batch size = number of SMs (each SM processes one job at a time)
        # The persistent kernel has 48 blocks (1 per SM), and blocks process jobs sequentially
        num_models = num_sms
        problem_size = len(nodes)

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads, {num_models} models in parallel ({problem_size} nodes)")

        try:
            # Generate multiple Ising problems
            h_list = []
            J_list = []
            nonces = []

            for _ in range(num_models):
                nonce = random.randint(0, 2**32 - 1)
                nonces.append(nonce)
                h_dict, J_dict = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)

                # Compute N as max node ID + 1 (topology has sparse node IDs: 4593 nodes numbered 0-4799)
                N = max(max(nodes), max(max(i, j) for i, j in edges)) + 1

                # Convert h_dict to array indexed directly by node ID (NO remapping needed!)
                # This ensures alignment with edge node IDs
                h = np.zeros(N, dtype=np.float32)
                for node, val in h_dict.items():
                    h[node] = val

                # Convert J_dict to array indexed by edges
                J = np.zeros(len(edges), dtype=np.float32)
                edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
                for (i, j), val in J_dict.items():
                    if (i, j) in edge_to_idx:
                        J[edge_to_idx[(i, j)]] = val
                    elif (j, i) in edge_to_idx:
                        J[edge_to_idx[(j, i)]] = val

                h_list.append(h)
                J_list.append(J)

            start_time = time.time()

            # Use sample_ising API (synchronous, returns list of SampleSets)
            samplesets = sampler.sample_ising(
                h_list=h_list,
                J_list=J_list,
                num_reads=reads,
                num_betas=sweeps,
                num_sweeps_per_beta=1,
                edges=edges
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

            # Extract energies from sampleset
            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime:.2f}s ({num_models} models)")
            print(f"  🚀 Throughput: {throughput:.2f} models/second")
            print(f"  🎯 Best energy: {min(all_min_energies):.1f} (across {num_models} models)")
            print(f"  📊 Avg energy (first model): {avg_energy:.1f} (±{std_energy:.1f})")

            # Filter samples to only include actual topology nodes
            # Samples have length N=4800 (max node ID + 1) but only 4593 nodes exist
            # We need to extract only the values at the actual node indices
            filtered_samples = []
            for sample in sampleset.record.sample:
                # Extract only the values at node indices that exist in the topology
                filtered_sample = np.array([sample[node] for node in nodes], dtype=np.int8)
                filtered_samples.append(filtered_sample)

            # Create new SampleSet with filtered samples (correct length for validation)
            import dimod
            filtered_sampleset = dimod.SampleSet.from_samples(
                filtered_samples,
                vartype='SPIN',
                energy=sampleset.record.energy,
                info=sampleset.info
            )

            # Use evaluate_sampleset to get diversity and num_solutions (same as CPU)
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty (allow positive energies)
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600  # 10 minutes
            )

            # Use the nonce and generate test salt for evaluation
            salt = b"test_salt_cuda_baseline"
            prev_timestamp = int(time.time()) - 600  # 10 minutes ago

            # Evaluate the filtered sampleset
            mining_result = evaluate_sampleset(
                filtered_sampleset, requirements, nodes, edges, nonces[0], salt,
                prev_timestamp, start_time, f"cuda-baseline-{sweeps}-{reads}", "CUDA"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Calculate diversity of top 10 solutions by energy (same as CPU)
            solutions = list(filtered_sampleset.record.sample)
            energies = list(filtered_sampleset.record.energy)

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
    print(f"\n📊 CUDA Baseline Summary (total time: {total_runtime/60:.1f} min):")
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

    # Stop the persistent kernel
    kernel.stop_immediate()

    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='CUDA GPU baseline parameter testing tool')
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
        help='Run only the config with this description (e.g., "Light CUDA")'
    )
    parser.add_argument(
        '--h-values',
        type=str,
        default='-1,0,1',
        help='Comma-separated h field values (default: -1,0,1). Use "0" for h=0 baseline.'
    )
    parser.add_argument(
        '--embedding',
        type=str,
        help='Use embedded hardware topology instead of perfect topology (e.g., "Z(9,2)")'
    )
    parser.add_argument(
        '--topology',
        type=str,
        help='Topology to use. Can be: file path (e.g., "dwave_topologies/topologies/advantage2_system1_7.json.gz"), '
             'hardware name (e.g., "Advantage2_system1.8"), or Zephyr format (e.g., "Z(12,4)"). '
             'Takes precedence over --embedding.'
    )
    args = parser.parse_args()

    # Parse h_values
    h_values = [float(v.strip()) for v in args.h_values.split(',')]

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"cuda_baseline_results_{timestamp}.json"

    # Handle preset timeouts and filters
    only_label = args.only
    if args.quick:
        timeout = 10.0
        only_label = "Light CUDA"  # Force Light test only
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    # Run baseline test
    cuda_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        only_label=only_label,
        h_values=h_values,
        use_embedding=args.embedding,
        topology_path=args.topology
    )

    print(f"\n✅ CUDA baseline test complete!")


if __name__ == "__main__":
    main()