#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CPU Block Gibbs baseline parameter testing tool.

Uses dwave's SimulatedAnnealingSampler with Zephyr four-color variable reordering
to achieve block Gibbs sampling on CPU. Variables are reordered so that same-color
nodes (which share no edges) are consecutive, making sequential Gibbs updates
equivalent to block-parallel updates.

Requires a Zephyr topology (default: Advantage2_system1.13).

Usage:
    python tools/sa_gibbs_baseline.py --quick
    python tools/sa_gibbs_baseline.py --quick --update-mode metropolis
    python tools/sa_gibbs_baseline.py --timeout 10
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave_networkx import zephyr_four_color, zephyr_coordinates

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    evaluate_sampleset,
    calculate_diversity,
)
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.json_loader import load_topology


def _extract_zephyr_params(topo_obj):
    """Extract Zephyr (m, t) from a topology object.

    Returns (m, t) or raises ValueError if not a Zephyr topology.
    """
    # ZephyrTopology instances have .m and .t directly
    if hasattr(topo_obj, 'm') and hasattr(topo_obj, 't'):
        return topo_obj.m, topo_obj.t

    # Loaded topologies: check properties
    props = getattr(topo_obj, 'properties', {})
    topo_info = props.get('topology', {})
    if topo_info.get('type') == 'zephyr':
        shape = topo_info['shape']
        return int(shape[0]), int(shape[1])

    raise ValueError(
        f"Topology is not Zephyr (type={topo_info.get('type', 'unknown')}). "
        "Block Gibbs color reordering requires a Zephyr topology."
    )


def _build_color_ordering(nodes, m, t):
    """Build Zephyr four-color variable ordering map.

    Returns:
        standard_to_bg: dict mapping original node index -> color-ordered index
        bg_to_standard: dict mapping color-ordered index -> original node index
        block_counts: dict mapping color -> count of nodes in that color block
    """
    to_tuple = zephyr_coordinates(m, t).linear_to_zephyr
    ordered_nodes = sorted(nodes, key=lambda n: zephyr_four_color(to_tuple(n)))
    standard_to_bg = {n: idx for idx, n in enumerate(ordered_nodes)}
    bg_to_standard = {idx: n for idx, n in enumerate(ordered_nodes)}

    # Count nodes per color block
    block_counts = {}
    for n in nodes:
        color = zephyr_four_color(to_tuple(n))
        block_counts[color] = block_counts.get(color, 0) + 1

    return standard_to_bg, bg_to_standard, block_counts


def _reorder_ising(h, J, standard_to_bg):
    """Remap h and J dicts from standard to color-block ordering."""
    h_bg = {standard_to_bg[n]: v for n, v in h.items()}
    J_bg = {(standard_to_bg[n1], standard_to_bg[n2]): v for (n1, n2), v in J.items()}
    return h_bg, J_bg


def _remap_sampleset(sampleset, bg_to_standard):
    """Remap sampleset variables from color-block ordering back to standard."""
    return sampleset.relabel_variables(bg_to_standard)


def sa_gibbs_baseline_test(
    timeout_minutes=10.0,
    output_file=None,
    h_values=None,
    only_label=None,
    topology=None,
    update_mode="gibbs",
):
    """Test CPU Block Gibbs performance with baseline format and evaluation logic.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        h_values: List of allowed h field values
        only_label: Run only specific config (e.g., "Light Gibbs")
        topology: Topology to use (must be Zephyr). Can be:
                  - Z(m,t) format (e.g., "Z(9,2)")
                  - Hardware name (e.g., "Advantage2_system1.13")
                  - File path to topology JSON
                  Default: Advantage2_system1.13
        update_mode: "gibbs" or "metropolis" (default: "gibbs")
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    mode_name = "Gibbs" if update_mode == "gibbs" else "Metropolis"
    acceptance = "Gibbs" if update_mode == "gibbs" else "Metropolis"

    print(f"CPU Block {mode_name} Baseline Parameter Test")
    print("=" * 50)
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"h_values: {h_values}")
    print(f"Update mode: {update_mode}")

    # Load topology
    if topology:
        print(f"Loading topology: {topology}")
        topo_obj = load_topology(topology)
        nodes = list(topo_obj.graph.nodes) if hasattr(topo_obj, 'graph') else topo_obj.nodes
        edges = list(topo_obj.graph.edges) if hasattr(topo_obj, 'graph') else topo_obj.edges
        topology_desc = f"{getattr(topo_obj, 'solver_name', 'unknown')} ({len(nodes)} nodes, {len(edges)} edges)"
    else:
        print("Using default topology (Advantage2_system1.13)")
        topo_obj = DEFAULT_TOPOLOGY
        nodes = list(topo_obj.graph.nodes) if hasattr(topo_obj, 'graph') else topo_obj.nodes
        edges = list(topo_obj.graph.edges) if hasattr(topo_obj, 'graph') else topo_obj.edges
        topology_desc = f"{topo_obj.solver_name} ({len(nodes)} nodes, {len(edges)} edges)"

    print(f"Topology: {topology_desc}")

    # Extract Zephyr parameters and build color ordering
    m, t = _extract_zephyr_params(topo_obj)
    standard_to_bg, bg_to_standard, block_counts = _build_color_ordering(nodes, m, t)
    print(f"Zephyr({m},{t}) four-color block sizes: {block_counts}")

    # Initialize sampler
    sampler = SimulatedAnnealingSampler()

    # Generate test problem
    seed = 12345
    h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

    # Show h distribution
    h_vals_set = sorted(set(h.values()))
    h_counts = {v: list(h.values()).count(v) for v in h_vals_set}
    h_dist_str = ", ".join([f"{v}: {h_counts[v]} ({100*h_counts[v]/len(h):.1f}%)" for v in h_vals_set])
    print(f"Problem: {len(h)} variables, {len(J)} couplings")
    print(f"   h distribution: {h_dist_str}")

    # Test configurations
    test_configs = [
        (256, 64, f"Light {mode_name}"),
        (512, 100, f"Low {mode_name}"),
        (1024, 100, f"Medium {mode_name}"),
        (2048, 150, f"High {mode_name}"),
        (4096, 200, f"Very High {mode_name}"),
        (8192, 200, f"Max {mode_name}"),
    ]

    # Optional filter
    if only_label:
        available_labels = [desc for _, _, desc in test_configs]
        filtered = [cfg for cfg in test_configs if cfg[2].lower() == only_label.lower()]
        if not filtered:
            print(f"No test config matched --only {only_label!r}; available: {available_labels}")
            return None
        test_configs = filtered

    print(f"\nTesting CPU Block {mode_name} configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': f'cpu-sa-{update_mode}',
        'topology': topology_desc,
        'topology_arg': topology if topology else "default",
        'update_mode': update_mode,
        'zephyr_params': {'m': m, 't': t},
        'color_block_sizes': {str(k): v for k, v in block_counts.items()},
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': seed,
        },
        'tests': [],
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()

    # Deterministic seed sequence for reproducible comparisons
    random.seed(42)
    test_nonces = [random.randint(0, 2**32 - 1) for _ in range(len(test_configs))]

    for idx, (sweeps, reads, desc) in enumerate(test_configs):
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\nTotal timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads")

        try:
            # Generate problem with deterministic nonce
            nonce = test_nonces[idx]
            h_test, J_test = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)

            # Reorder to color-block ordering
            h_bg, J_bg = _reorder_ising(h_test, J_test, standard_to_bg)

            start_time = time.time()
            sampleset_bg = sampler.sample_ising(
                h=h_bg,
                J=J_bg,
                num_reads=reads,
                num_sweeps=sweeps,
                proposal_acceptance_criteria=acceptance,
            )
            runtime = time.time() - start_time

            # Remap back to standard ordering for evaluation
            sampleset = _remap_sampleset(sampleset_bg, bg_to_standard)

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float(np.std(energies))

            print(f"  Runtime: {runtime:.2f}s ({runtime/60:.1f} min)")
            print(f"  min_energy = {min_energy:.1f}")
            print(f"  avg_energy = {avg_energy:.1f} (+/-{std_energy:.1f})")

            # Evaluate with mining requirements
            requirements = BlockRequirements(
                difficulty_energy=0.0,
                min_diversity=0.1,
                min_solutions=1,
                timeout_to_difficulty_adjustment_decay=600,
            )

            salt = b"test_salt_sa_gibbs_baseline"
            prev_timestamp = int(time.time()) - 600

            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonce, salt,
                prev_timestamp, start_time, f"cpu-sa-{update_mode}-{sweeps}-{reads}", "CPU"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Diversity of top 10 solutions by energy
            solutions = list(sampleset.record.sample)
            energies_arr = list(sampleset.record.energy)

            solution_energy_pairs = list(zip(solutions, energies_arr))
            solution_energy_pairs.sort(key=lambda x: x[1])
            top_10_solutions = [sol for sol, _ in solution_energy_pairs[:10]]

            top_10_diversity = calculate_diversity(top_10_solutions)
            print(f"  diversity (top 10) = {top_10_diversity:.3f}")

            if mining_result:
                diversity = mining_result.diversity
                num_solutions = mining_result.num_valid
                meets_requirements = True
                print(f"  num_solutions = {num_solutions}")
                print(f"  Meets mining requirements!")
            else:
                print(f"  Does not meet mining requirements")

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
                print(f"  Quality: {target_reached}")

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
                'meets_requirements': bool(meets_requirements),
            }
            results['tests'].append(test_result)

            # Individual test timeout check
            if runtime > timeout_seconds * 0.8:
                print(f"  Single test approaching timeout, stopping further tests")
                break

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    total_runtime = time.time() - total_start_time
    print(f"\nCPU Block {mode_name} Baseline Summary (total time: {total_runtime/60:.1f} min):")
    print("=" * 50)

    if results['tests']:
        best_result = min(results['tests'], key=lambda r: r['min_energy'])
        print(f"Best energy: {best_result['min_energy']:.1f}")
        print(f"   Required: {best_result['num_sweeps']} sweeps, {best_result['runtime_minutes']:.1f} min")

        print(f"\nTime vs Energy Performance:")
        for result in results['tests']:
            quality = f"({result['target_reached']})" if result['target_reached'] != 'none' else ""
            print(f"  {result['runtime_minutes']:5.1f} min: {result['min_energy']:7.1f} energy {quality}")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='CPU Block Gibbs baseline parameter testing tool'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=10.0,
        help='Timeout in minutes (default: 10.0)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (only Light test)',
    )
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Extended test mode (30 minute timeout)',
    )
    parser.add_argument(
        '--only',
        type=str,
        help='Run only the config with this description (e.g., "Light Gibbs")',
    )
    parser.add_argument(
        '--h-values',
        type=str,
        default='-1,0,1',
        help='Comma-separated h field values (default: -1,0,1). Use "0" for h=0 baseline.',
    )
    parser.add_argument(
        '--topology',
        type=str,
        help='Zephyr topology: Z(9,2), Advantage2_system1.13, or file path. '
             'Default: Advantage2_system1.13',
    )
    parser.add_argument(
        '--update-mode',
        type=str,
        choices=['gibbs', 'metropolis'],
        default='gibbs',
        help='Acceptance criteria: gibbs (default) or metropolis',
    )

    args = parser.parse_args()

    # Parse h_values
    h_values = [float(v.strip()) for v in args.h_values.split(',')]

    # Handle preset timeouts and filters
    only_label = args.only
    mode_name = "Gibbs" if args.update_mode == "gibbs" else "Metropolis"
    if args.quick:
        timeout = 10.0
        only_label = f"Light {mode_name}"
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"sa_{args.update_mode}_baseline_results_{timestamp}.json"

    sa_gibbs_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        h_values=h_values,
        only_label=only_label,
        topology=args.topology,
        update_mode=args.update_mode,
    )

    print(f"\nCPU Block {mode_name} baseline test complete!")


if __name__ == "__main__":
    main()
