#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Metal Block Gibbs baseline parameter testing tool."""

import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

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

from GPU.metal_gibbs_sa import MetalGibbsSampler
from GPU.metal_miner import get_gpu_core_count


def metal_gibbs_baseline_test(
    timeout_minutes=10.0,
    output_file=None,
    only_label=None,
    h_values=None,
    num_models=1,
    topology=None,
    update_mode="gibbs",
    parallel=False
):
    """Test Metal Block Gibbs performance with baseline format and evaluation logic.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        only_label: Run only specific config (e.g., "Light Gibbs")
        h_values: List of allowed h field values
        num_models: Number of parallel models to run
        topology: Topology to use. Can be:
                  - Z(m,t) format for perfect Zephyr topology (e.g., "Z(9,2)")
                  - Hardware name (e.g., "Advantage2_system1")
                  - File path to topology JSON (e.g., "path/to/topology.json.gz")
                  - File path to embedding (e.g., "path/to/*.embed.json.gz") - auto-detected
                  Default: Advantage2_system1
        update_mode: "gibbs" or "metropolis" (default: "gibbs")
        parallel: Use parallel kernel with threadgroup barriers (default: False)
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    mode_name = "Gibbs" if update_mode == "gibbs" else "Metropolis"
    parallel_str = " (Parallel)" if parallel else ""
    print(f"Metal Block {mode_name}{parallel_str} Baseline Parameter Test")
    print("=" * 50)
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"h_values: {h_values}")
    print(f"Update mode: {update_mode}")
    print(f"Parallel mode: {parallel}")

    # Initialize sampler
    try:
        gibbs_sampler = MetalGibbsSampler(update_mode=update_mode, parallel=parallel)
        print(f"Metal Block {mode_name}{parallel_str} sampler ready")
        print(f"Color block sizes: {gibbs_sampler.block_counts}")
    except Exception as e:
        print(f"Metal Block {mode_name} sampler failed: {e}")
        return None

    # Get topology
    nodes, edges, topology_desc = load_baseline_topology(
        topology_arg=topology,
    )
    print(f"Topology: {topology_desc}")

    # Generate test problem with h_values
    seed = 12345
    h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

    # Show h distribution
    print_problem_summary(h, J)

    # Test configurations - matching other baselines
    test_configs = [
        (256, 64, f"Light {mode_name}"),
        (512, 100, f"Low {mode_name}"),
        (1024, 100, f"Medium {mode_name}"),
        (2048, 150, f"High {mode_name}"),
        (4096, 200, f"Very High {mode_name}"),
        (8192, 200, f"Max {mode_name}")
    ]

    # Optional filter
    test_configs = filter_configs_by_label(test_configs, only_label)
    if test_configs is None:
        return None

    print(f"\nTesting Metal Block {mode_name} configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': f'metal-{update_mode}' + ('-parallel' if parallel else ''),
        'topology': topology_desc,
        'topology_arg': topology if topology else "default",
        'update_mode': update_mode,
        'parallel': parallel,
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
            print(f"\nTotal timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n{desc}: {sweeps} sweeps, {reads} reads, {num_models} models")

        try:
            # Generate one problem from the deterministic nonce; every model
            # in the batch shares it (the sampler reads h/J without mutating).
            nonce = test_nonces[idx]
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges, h_values=h_values)
            nonces = [nonce] * num_models
            h_list = [h] * num_models
            J_list = [J] * num_models

            start_time = time.time()
            # Process models in batch
            samplesets = gibbs_sampler.sample_ising(
                h=h_list, J=J_list,
                num_reads=reads,
                num_sweeps=sweeps
            )
            runtime = time.time() - start_time
            throughput = num_models / runtime

            # Collect stats from all models
            all_min_energies = []
            for sampleset in samplesets:
                energies = list(sampleset.record.energy)
                all_min_energies.append(float(min(energies)))

            # Use first sampleset for detailed analysis
            sampleset = samplesets[0]
            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(np.mean(energies))
            std_energy = float(np.std(energies))

            print(f"  Runtime: {runtime:.2f}s ({num_models} models)")
            if num_models > 1:
                print(f"  Throughput: {throughput:.2f} models/second")
                print(f"  Best energy: {min(all_min_energies):.1f} (across {num_models} models)")
            else:
                print(f"  min_energy = {min_energy:.1f}")
            print(f"  Avg energy (first model): {avg_energy:.1f} (+/-{std_energy:.1f})")

            # Evaluate the sampleset
            eval_fields = evaluate_baseline_sampleset(
                sampleset, nodes, edges, nonces[0],
                start_time, f"metal-{update_mode}-{sweeps}-{reads}", "Metal",
                b"test_salt_metal_gibbs_baseline",
            )

            # Energy target analysis
            target_reached = classify_energy(min_energy)
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
                **eval_fields,
            }
            results['tests'].append(test_result)

            # Individual test timeout check
            if runtime > timeout_seconds * 0.8:
                print(f"  Single test approaching timeout, stopping further tests")
                break

        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            break

    # Summary
    total_runtime = time.time() - total_start_time
    results['_total_runtime_seconds'] = total_runtime
    print_results_summary(
        results, f"Metal Block {mode_name} Baseline Summary",
    )

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    """Main function with command line argument parsing."""
    parser = build_baseline_argparser(
        'Metal Block Gibbs baseline parameter testing tool',
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=None,
        help='Number of models to process in parallel (default: auto-detect GPU cores)',
    )
    parser.add_argument(
        '--update-mode',
        type=str,
        choices=['gibbs', 'metropolis'],
        default='gibbs',
        help='Update mode: gibbs (default) or metropolis',
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel kernel with threadgroup barriers for true parallel color block updates',
    )

    args = parser.parse_args()

    # Auto-detect GPU core count if not specified
    if args.num_models is None:
        try:
            num_models = get_gpu_core_count()
        except Exception as e:
            print(f"Could not detect GPU cores ({e}), defaulting to 40 models")
            num_models = 40
    else:
        num_models = args.num_models

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
        parallel_suffix = "_parallel" if args.parallel else ""
        output_file = f"metal_{args.update_mode}{parallel_suffix}_baseline_results_{timestamp}.json"

    # Run test
    metal_gibbs_baseline_test(
        timeout_minutes=timeout,
        output_file=output_file,
        only_label=only_label,
        h_values=h_values,
        num_models=num_models,
        topology=args.topology,
        update_mode=args.update_mode,
        parallel=args.parallel
    )

    parallel_str = " (Parallel)" if args.parallel else ""
    print(f"\nMetal Block {mode_name}{parallel_str} baseline test complete!")


if __name__ == "__main__":
    main()
