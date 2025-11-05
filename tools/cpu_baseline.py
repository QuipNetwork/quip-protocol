#!/usr/bin/env python3
"""CPU baseline parameter testing tool.

This tool tests CPU simulated annealing performance with various sweep/read configurations.

EMBEDDING TESTING:
------------------
You can test embedding quality before using QPU time:

1. Perfect mode (default):
   python tools/cpu_baseline.py --topology <embedding_file>
   - Solves on ideal Z(m,t) topology
   - Establishes baseline performance

2. Embedded mode:
   python tools/cpu_baseline.py --topology <embedding_file> --use-embedding
   - Applies embedding and solves on Advantage2 hardware topology
   - Tests embedding structure and problem generation pipeline

IMPORTANT: Embedded mode shows significant energy degradation (~16%) due to chain breaking
in classical SA. This is a TESTING ARTIFACT - QPU quantum annealing maintains chain integrity
much better. Use embedded mode to verify embedding structure, NOT to predict QPU performance.
"""
import argparse
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
from shared.quantum_proof_of_work import generate_ising_model_from_nonce, evaluate_sampleset
from shared.block_requirements import BlockRequirements
from dwave_topologies import ZEPHYR_Z9_T2_TOPOLOGY
from dwave_topologies.embedded_topology import create_embedded_topology
from dwave_topologies.embedding_loader import load_embedding
import random
import dimod


def cpu_baseline_test(timeout_minutes=10.0, output_file=None, h_values=None, only_label=None, topology_file=None, use_embedding=False):
    """Test CPU performance with configurable timeout.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        h_values: List of allowed h field values
        only_label: Run only specific config (e.g., "Light CPU")
        topology_file: Path to embedding file (e.g., "dwave_topologies/embeddings/.../zephyr_z9_t2.embed.json.gz")
        use_embedding: If True, apply embedding and solve on hardware topology.
                      If False (default), solve on perfect logical topology.
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

    print("🔬 CPU Baseline Parameter Test")
    print("=" * 40)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎲 h_values: {h_values}")

    # Load topology and embedding if specified
    embedding_data = None
    embedding_dict = None
    source_topology = None
    hardware_topology = None

    if topology_file:
        # Parse topology name from file path (e.g., "zephyr_z9_t2")
        import os
        filename = os.path.basename(topology_file)
        if filename.startswith("zephyr_z") and filename.endswith(".embed.json.gz"):
            # Extract m, t from filename like "zephyr_z9_t2.embed.json.gz"
            parts = filename.replace("zephyr_z", "").replace(".embed.json.gz", "").split("_t")
            topology_name = f"Z({parts[0]},{parts[1]})"

            # Load embedding data
            print(f"📁 Loading embedding from: {topology_file}")
            embedding_data = load_embedding(topology_name, "Advantage2_system1.6")
            if not embedding_data:
                print(f"❌ Failed to load embedding file")
                return None

            embedding_dict = {int(k): v for k, v in embedding_data['embedding'].items()}
            print(f"✅ Loaded embedding: {len(embedding_dict)} logical vars → avg chain length {embedding_data['statistics']['avg_chain_length']:.2f}")

            # Get source (logical) topology
            from dwave_topologies.topologies.zephyr import zephyr
            m, t = int(parts[0]), int(parts[1])
            source_topology = zephyr(m, t)

            # Get hardware topology
            from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_6_TOPOLOGY
            hardware_topology = ADVANTAGE2_SYSTEM1_6_TOPOLOGY
        else:
            print(f"❌ Unsupported topology file format: {filename}")
            return None

    # Initialize topology and sampler based on mode
    if topology_file and use_embedding:
        print(f"🔗 Mode: EMBEDDED - Solving on Advantage2 hardware with embedding")
        nodes = hardware_topology.nodes
        edges = hardware_topology.edges
        logical_nodes = source_topology.nodes
        logical_edges = source_topology.edges
        topology_desc = f"Advantage2 hardware ({len(nodes)} qubits, {len(edges)} couplers) via embedding"

        # Create sampler with hardware topology
        from dwave.samplers import SimulatedAnnealingSampler
        from dwave.system.testing import MockDWaveSampler
        substitute = SimulatedAnnealingSampler()
        cpu_sampler = MockDWaveSampler(
            nodelist=nodes,
            edgelist=edges,
            properties=hardware_topology.properties,
            substitute_sampler=substitute
        )
        cpu_sampler.mocked_parameters.add('num_sweeps')
        cpu_sampler.parameters.update(substitute.parameters)

    elif topology_file:
        print(f"✨ Mode: PERFECT - Solving on perfect logical topology")
        nodes = source_topology.nodes
        edges = source_topology.edges
        logical_nodes = nodes
        logical_edges = edges
        topology_desc = f"Perfect {topology_name} ({len(nodes)} nodes, {len(edges)} edges)"
        cpu_sampler = SimulatedAnnealingStructuredSampler()

    else:
        print(f"✨ Using default topology")
        cpu_sampler = SimulatedAnnealingStructuredSampler()
        nodes = cpu_sampler.nodes
        edges = cpu_sampler.edges
        logical_nodes = nodes
        logical_edges = edges
        topology_desc = f"Default Z(9,2) ({len(nodes)} nodes, {len(edges)} edges)"

    print(f"📐 Topology: {topology_desc}")

    # Generate problem on LOGICAL topology (always)
    seed = 12345  # Fixed seed for reproducible results
    if topology_file:
        logical_h, logical_J = generate_ising_model_from_nonce(seed, logical_nodes, logical_edges, h_values=h_values)
        print(f"📊 Generated problem: {len(logical_nodes)} logical variables, {len(logical_edges)} couplings")
    else:
        logical_h, logical_J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)
        logical_nodes = nodes
        logical_edges = edges

    # If using embedding, embed the problem onto hardware
    if topology_file and use_embedding:
        print(f"🔗 Embedding logical problem onto hardware...")
        # Embed h and J from logical to hardware
        from dwave.embedding import embed_ising
        # Convert edge list to adjacency dict for embed_ising
        target_adj = {u: set() for u in nodes}
        for u, v in edges:
            target_adj[u].add(v)
            target_adj[v].add(u)
        h, J = embed_ising(logical_h, logical_J, embedding_dict, target_adj)
        print(f"✅ Embedded: {len(h)} hardware qubits, {len(J)} hardware couplers")
    else:
        h = logical_h
        J = logical_J

    # Show h distribution
    h_vals_set = sorted(set(h.values()))
    h_counts = {v: list(h.values()).count(v) for v in h_vals_set}
    h_dist_str = ", ".join([f"{v}: {h_counts[v]} ({100*h_counts[v]/len(h):.1f}%)" for v in h_vals_set])
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    print(f"   h distribution: {h_dist_str}")
    
    # Test configurations - from light to heavy
    test_configs = [
        (256, 64, "Light CPU"),
        (512, 100, "Low CPU"),
        (1024, 100, "Medium CPU"),
        (2048, 150, "High CPU"),
        (4096, 200, "Very High CPU"),
        (8192, 200, "Max CPU")
    ]

    # Optional filter: run only the requested label
    if only_label:
        available_labels = [desc for _, _, desc in test_configs]
        filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
        if not filtered:
            print(f"⚠️ No test config matched --only {only_label!r}; available: {available_labels}")
            return None
        test_configs = filtered
    
    print(f"\n🧪 Testing CPU configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
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

    # Use deterministic seed sequence for reproducible comparisons
    random.seed(42)
    test_nonces = [random.randint(0, 2**32 - 1) for _ in range(len(test_configs))]

    for idx, (sweeps, reads, desc) in enumerate(test_configs):
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads")

        try:
            # Generate the Ising model with deterministic nonce on LOGICAL topology
            nonce = test_nonces[idx]
            if topology_file:
                logical_h_test, logical_J_test = generate_ising_model_from_nonce(nonce, logical_nodes, logical_edges, h_values=h_values)
            else:
                logical_h_test, logical_J_test = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)

            # Embed if needed
            if topology_file and use_embedding:
                from dwave.embedding import embed_ising, unembed_sampleset
                h_test, J_test = embed_ising(logical_h_test, logical_J_test, embedding_dict, target_adj)
            else:
                h_test, J_test = logical_h_test, logical_J_test

            start_time = time.time()
            sampleset = cpu_sampler.sample_ising(
                h=h_test, J=J_test,
                num_reads=reads,
                num_sweeps=sweeps
            )

            # Unembed if needed
            if topology_file and use_embedding:
                from dwave.embedding import unembed_sampleset
                sampleset = unembed_sampleset(sampleset, embedding_dict, source_bqm=dimod.BinaryQuadraticModel.from_ising(logical_h_test, logical_J_test))

            runtime = time.time() - start_time

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"  🎯 min_energy = {min_energy:.1f}")
            print(f"  📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")

            # Use evaluate_sampleset to get diversity and num_solutions
            # Create test requirements (using dummy values to ensure they pass)
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty (allow positive energies)
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600  # 10 minutes
            )

            # Use the same nonce and generate test salt for evaluation
            salt = b"test_salt_cpu_baseline"
            prev_timestamp = int(time.time()) - 600  # 10 minutes ago

            # Evaluate the sampleset (always use logical topology for validation)
            eval_nodes = logical_nodes if topology_file else nodes
            eval_edges = logical_edges if topology_file else edges
            mining_result = evaluate_sampleset(
                sampleset, requirements, eval_nodes, eval_edges, nonce, salt,
                prev_timestamp, start_time, f"cpu-baseline-{sweeps}-{reads}", "CPU"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Calculate diversity of top 10 solutions by energy
            from shared.quantum_proof_of_work import calculate_diversity
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
            if runtime > timeout_seconds * 0.8:  # 80% of total timeout
                print(f"  ⏰ Single test approaching timeout, stopping further tests")
                break
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break
    
    # Summary
    total_runtime = time.time() - total_start_time
    print(f"\n📊 CPU Baseline Summary (total time: {total_runtime/60:.1f} min):")
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
    parser = argparse.ArgumentParser(description='CPU baseline parameter testing tool')
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
        help='Run only the config with this description (e.g., "Light CPU")'
    )
    parser.add_argument(
        '--h-values',
        type=str,
        default='-1,0,1',
        help='Comma-separated h field values (default: -1,0,1). Use "0" for h=0 baseline.'
    )
    parser.add_argument(
        '--topology',
        type=str,
        help='Path to embedding file (e.g., "dwave_topologies/embeddings/Advantage2_system1.6/zephyr_z9_t2.embed.json.gz")'
    )
    parser.add_argument(
        '--use-embedding',
        action='store_true',
        help='Apply embedding and solve on hardware topology (requires --topology)'
    )

    args = parser.parse_args()

    # Parse h_values
    h_values = [float(v.strip()) for v in args.h_values.split(',')]

    # Handle preset timeouts and filters
    only_label = args.only
    if args.quick:
        timeout = 10.0
        only_label = "Light CPU"  # Force Light test only
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"cpu_baseline_results_{timestamp}.json"

    # Run test
    cpu_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        h_values=h_values,
        only_label=only_label,
        topology_file=args.topology,
        use_embedding=args.use_embedding
    )

    print(f"\n✅ CPU baseline test complete!")


if __name__ == "__main__":
    main()