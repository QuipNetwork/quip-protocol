#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Block Gibbs baseline parameter testing tool.

Output format matches cuda_baseline.py exactly. Runs 12
models in parallel (4 SMs per model × 12 = 48 SMs).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dimod
import numpy as np

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    evaluate_sampleset,
    calculate_diversity,
)
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.json_loader import (
    load_topology,
)
from dwave_topologies.embedded_topology import (
    create_embedded_topology,
)

from GPU.cuda_gibbs_sa import CudaGibbsSampler

# 4 SMs per Gibbs model
SMs_PER_MODEL = 4


def cuda_gibbs_baseline_test(
    timeout_minutes=10.0,
    output_file=None,
    only_label=None,
    h_values=None,
    topology=None,
    update_mode="gibbs",
    parallel=True,
    num_models=12,
):
    """Test CUDA Block Gibbs with output matching cuda_baseline.

    Args:
        timeout_minutes: Test timeout in minutes.
        output_file: Path to save JSON results.
        only_label: Run only specific config label.
        h_values: List of allowed h field values.
        topology: Topology specification.
        update_mode: "gibbs" or "metropolis".
        parallel: Chromatic parallel (True) or sequential.
        num_models: Models to run in parallel.
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    mode_name = (
        "Gibbs" if update_mode == "gibbs"
        else "Metropolis"
    )
    kernel_type = (
        "Chromatic" if parallel else "Sequential"
    )
    print(
        f"🔬 CUDA Block {mode_name} ({kernel_type}) "
        f"Baseline Test"
    )
    print("=" * 60)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎲 h_values: {h_values}")

    try:
        print("📦 Initializing CUDA Gibbs sampler...")
        sampler = CudaGibbsSampler(
            update_mode=update_mode, parallel=parallel,
        )
        print("✅ CUDA Gibbs sampler ready")
    except Exception as e:
        print(f"❌ CUDA Gibbs sampler failed: {e}")
        return None

    # Get topology
    if topology:
        if topology.endswith('.embed.json.gz'):
            print(
                f"🔗 Loading embedded topology: {topology}"
            )
            import os
            filename = os.path.basename(topology)
            parts = filename.replace(
                "zephyr_z", "",
            ).replace(".embed.json.gz", "").split("_t")
            topology_name = f"Z({parts[0]},{parts[1]})"
            embedded_topo = create_embedded_topology(
                topology_name,
            )
            nodes = embedded_topo.nodes
            edges = embedded_topo.edges
            topology_desc = (
                f"{topology_name} embedded "
                f"({len(nodes)} qubits, "
                f"{len(edges)} couplers)"
            )
        else:
            print(f"📂 Loading topology: {topology}")
            topo_obj = load_topology(topology)
            nodes = (
                list(topo_obj.graph.nodes)
                if hasattr(topo_obj, 'graph')
                else topo_obj.nodes
            )
            edges = (
                list(topo_obj.graph.edges)
                if hasattr(topo_obj, 'graph')
                else topo_obj.edges
            )
            topology_name = getattr(
                topo_obj, 'solver_name', 'unknown',
            )
            topology_desc = (
                f"{topology_name} "
                f"({len(nodes)} nodes, "
                f"{len(edges)} edges)"
            )
    else:
        print("✨ Using perfect topology (default)")
        topo_obj = DEFAULT_TOPOLOGY
        nodes = list(topo_obj.graph.nodes)
        edges = list(topo_obj.graph.edges)
        topology_desc = (
            f"perfect Z(9,2) "
            f"({len(nodes)} nodes, {len(edges)} edges)"
        )

    print(f"📐 Topology: {topology_desc}")

    seed = 12345
    h, J = generate_ising_model_from_nonce(
        seed, nodes, edges, h_values=h_values,
    )

    h_vals_set = sorted(set(h.values()))
    h_counts = {
        v: list(h.values()).count(v) for v in h_vals_set
    }
    h_dist_str = ", ".join([
        f"{v}: {h_counts[v]} "
        f"({100 * h_counts[v] / len(h):.1f}%)"
        for v in h_vals_set
    ])
    print(
        f"📊 Problem: {len(h)} variables, "
        f"{len(J)} couplings"
    )
    print(f"   h distribution: {h_dist_str}")

    test_configs = [
        (256, 64, f"Light {mode_name}"),
        (512, 100, f"Low {mode_name}"),
        (1024, 100, f"Medium {mode_name}"),
        (2048, 150, f"High {mode_name}"),
        (4096, 200, f"Very High {mode_name}"),
        (8192, 200, f"Max {mode_name}"),
    ]

    if only_label:
        available = [desc for _, _, desc in test_configs]
        filtered = [
            cfg for cfg in test_configs
            if cfg[2].lower() == only_label.lower()
        ]
        if not filtered:
            print(
                f"⚠️ No config matched --only "
                f"{only_label!r}; available: {available}"
            )
            return None
        test_configs = filtered

    print(f"\n🧪 Testing CUDA Block {mode_name} configs:")
    print(
        f"🔧 {num_models} models in parallel "
        f"({SMs_PER_MODEL} SMs/model)"
    )

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': f'cuda-{update_mode}',
        'kernel_type': kernel_type.lower(),
        'topology': topology_desc,
        'topology_arg': topology if topology else "default",
        'update_mode': update_mode,
        'num_models': num_models,
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': seed,
        },
        'tests': [],
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()
    problem_size = len(nodes)

    for sweeps, reads, desc in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(
                f"\n⏰ Total timeout "
                f"({timeout_minutes} min) reached"
            )
            break

        print(
            f"\n{desc}: {sweeps} sweeps, {reads} reads, "
            f"{num_models} models in parallel "
            f"({problem_size} nodes)"
        )

        try:
            # Generate multiple Ising problems
            h_list = []
            J_list = []
            nonces = []
            for _ in range(num_models):
                nonce = random.randint(0, 2**32 - 1)
                nonces.append(nonce)
                h_i, J_i = generate_ising_model_from_nonce(
                    nonce, nodes, edges,
                    h_values=h_values,
                )
                h_list.append(h_i)
                J_list.append(J_i)

            start_time = time.time()

            samplesets = sampler.sample_ising(
                h=h_list, J=J_list,
                num_reads=reads,
                num_sweeps=sweeps,
            )

            runtime = time.time() - start_time
            throughput = num_models / runtime

            # Stats across all models
            all_min_energies = []
            for ss in samplesets:
                energies = list(ss.record.energy)
                all_min_energies.append(float(min(energies)))

            # First model for detailed analysis
            sampleset = samplesets[0]
            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(np.mean(energies))
            std_energy = float(np.std(energies))

            print(
                f"  ⏱️  {runtime:.2f}s "
                f"({num_models} models)"
            )
            print(
                f"  🚀 Throughput: "
                f"{throughput:.2f} models/second"
            )
            print(
                f"  🎯 Best energy: "
                f"{min(all_min_energies):.1f} "
                f"(across {num_models} models)"
            )
            print(
                f"  📊 Avg energy (first model): "
                f"{avg_energy:.1f} (±{std_energy:.1f})"
            )

            # Diversity of top 10 solutions (first model)
            solutions = list(sampleset.record.sample)
            pairs = sorted(
                zip(solutions, energies),
                key=lambda x: x[1],
            )
            top_10 = [sol for sol, _ in pairs[:10]]
            top_10_diversity = calculate_diversity(top_10)
            print(
                f"  🌈 diversity (top 10) = "
                f"{top_10_diversity:.3f}"
            )

            # Evaluate against mining requirements
            requirements = BlockRequirements(
                difficulty_energy=0.0,
                min_diversity=0.1,
                min_solutions=1,
                timeout_to_difficulty_adjustment_decay=600,
            )
            salt = b"test_salt_cuda_gibbs_baseline"
            prev_timestamp = int(time.time()) - 600

            mining_result = evaluate_sampleset(
                sampleset, requirements,
                nodes, edges, nonces[0], salt,
                prev_timestamp, start_time,
                f"cuda-{update_mode}-{sweeps}-{reads}",
                "CUDA",
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            if mining_result:
                diversity = mining_result.diversity
                num_solutions = mining_result.num_valid
                meets_requirements = True
                print(f"  🔢 num_solutions = {num_solutions}")
                print(f"  ✅ Meets mining requirements!")
            else:
                print(f"  ❌ Does not meet mining requirements")

            # Quality tier
            best_across = min(all_min_energies)
            target_reached = "none"
            if best_across <= -15650:
                target_reached = "excellent"
            elif best_across <= -15500:
                target_reached = "very_good"
            elif best_across <= -15400:
                target_reached = "good"
            elif best_across <= -15300:
                target_reached = "fair"

            if target_reached != "none":
                print(f"  🎖️  Quality: {target_reached}")

            test_result = {
                'description': desc,
                'num_sweeps': int(sweeps),
                'num_reads': int(reads),
                'num_models': num_models,
                'runtime_seconds': float(runtime),
                'runtime_minutes': float(runtime / 60),
                'min_energy': float(best_across),
                'avg_energy': avg_energy,
                'std_energy': std_energy,
                'throughput': throughput,
                'target_reached': target_reached,
                'diversity': float(diversity),
                'diversity_top_10': float(
                    top_10_diversity,
                ),
                'num_solutions': int(num_solutions),
                'meets_requirements': bool(
                    meets_requirements,
                ),
            }
            results['tests'].append(test_result)

            if runtime > timeout_seconds * 0.8:
                print(
                    f"  ⏰ Approaching timeout, stopping"
                )
                break

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    total_runtime = time.time() - total_start_time
    print(
        f"\n📊 CUDA Block {mode_name} Summary "
        f"(total time: {total_runtime / 60:.1f} min):"
    )
    print("=" * 50)

    if results['tests']:
        best = min(
            results['tests'], key=lambda r: r['min_energy'],
        )
        print(f"🏆 Best energy: {best['min_energy']:.1f}")
        print(
            f"   Required: {best['num_sweeps']} sweeps, "
            f"{best['runtime_minutes']:.1f} min"
        )

        print(f"\n⏱️ Time vs Energy Performance:")
        for r in results['tests']:
            quality = (
                f"({r['target_reached']})"
                if r['target_reached'] != 'none' else ""
            )
            print(
                f"  {r['runtime_minutes']:5.1f} min: "
                f"{r['min_energy']:7.1f} energy {quality}"
            )

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            'CUDA Block Gibbs baseline testing tool'
        ),
    )
    parser.add_argument(
        '--timeout', '-t', type=float, default=10.0,
        help='Timeout in minutes (default: 10.0)',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (only Light test)',
    )
    parser.add_argument(
        '--extended', action='store_true',
        help='Extended test mode (30 minute timeout)',
    )
    parser.add_argument(
        '--only', type=str,
        help='Run only this config (e.g., "Light Gibbs")',
    )
    parser.add_argument(
        '--h-values', type=str, default='-1,0,1',
        help=(
            'Comma-separated h values '
            '(default: -1,0,1)'
        ),
    )
    parser.add_argument(
        '--topology', type=str,
        help=(
            'Topology: Z(9,2), hardware name, '
            'or file path'
        ),
    )
    parser.add_argument(
        '--update-mode', type=str,
        choices=['gibbs', 'metropolis'],
        default='gibbs',
        help='Update mode (default: gibbs)',
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help=(
            'Use sequential kernel instead of '
            'Chromatic parallel'
        ),
    )
    parser.add_argument(
        '--num-models', type=int, default=12,
        help=(
            'Number of models in parallel '
            '(default: 12, i.e. 4 SMs × 12 = 48 SMs)'
        ),
    )

    args = parser.parse_args()

    h_values = [
        float(v.strip()) for v in args.h_values.split(',')
    ]

    mode_name = (
        "Gibbs" if args.update_mode == "gibbs"
        else "Metropolis"
    )
    only_label = args.only
    if args.quick:
        timeout = 10.0
        only_label = f"Light {mode_name}"
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        kernel = "seq" if args.sequential else "par"
        output_file = (
            f"cuda_{args.update_mode}_{kernel}_"
            f"baseline_results_{timestamp}.json"
        )

    cuda_gibbs_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        only_label=only_label,
        h_values=h_values,
        topology=args.topology,
        update_mode=args.update_mode,
        parallel=not args.sequential,
        num_models=args.num_models,
    )

    kernel_type = (
        "Sequential" if args.sequential
        else "Chromatic"
    )
    print(
        f"\n✅ CUDA Block {mode_name} ({kernel_type}) "
        f"baseline test complete!"
    )


if __name__ == "__main__":
    main()
