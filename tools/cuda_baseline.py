#!/usr/bin/env python3
"""CUDA GPU baseline parameter testing tool using self-feeding SA kernel."""
import json
import random
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

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

try:
    from GPU.cuda_sa import CudaSASampler
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def cuda_baseline_test(
    timeout_minutes=10.0,
    output_file=None,
    only_label=None,
    h_values=None,
    use_embedding=None,
    topology_path=None,
):
    """Test CUDA GPU performance using CudaSASampler.

    Args:
        timeout_minutes: Test timeout in minutes
        output_file: Path to save JSON results
        only_label: Run only specific config (e.g., "Light CUDA")
        h_values: List of allowed h field values
        use_embedding: If specified, use embedded hardware topology.
                      Format: "Z(9,2)" for Z(9,2) embedding
        topology_path: Path to topology file (JSON or JSON.gz).
                      Takes precedence over use_embedding.
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    print("🔬 CUDA GPU Baseline Parameter Test (CudaSASampler)")
    print("=" * 60)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎲 h_values: {h_values}")

    if not CUDA_AVAILABLE:
        print("❌ CUDA not available")
        return None

    try:
        print("📦 Initializing CUDA sampler...")
        sampler = CudaSASampler()
        print("✅ CUDA sampler ready")
    except Exception as e:
        print(f"❌ CUDA sampler initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get topology
    nodes, edges, topology_desc = load_baseline_topology(
        topology_arg=topology_path,
        embedding_arg=use_embedding,
    )

    sampler_type = "self-feeding-sa"
    print(f"📐 Topology: {topology_desc}")

    # Initial problem setup to show problem size
    seed = 12345
    h, J = generate_ising_model_from_nonce(
        seed, nodes, edges, h_values=h_values,
    )

    # Show h distribution
    print_problem_summary(h, J)

    # Test configurations - matching CPU baseline exactly
    test_configs = [
        (256, 64, "Light CUDA"),
        (512, 100, "Low CUDA"),
        (1024, 100, "Medium CUDA"),
        (2048, 150, "High CUDA"),
        (4096, 200, "Very High CUDA"),
        (8192, 200, "Max CUDA"),
    ]

    # Optional filter: run only the requested label
    test_configs = filter_configs_by_label(test_configs, only_label)
    if test_configs is None:
        return None

    print(f"\n🧪 Testing CUDA configurations:")

    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': sampler_type,
        'topology': topology_desc,
        'use_embedding': (
            use_embedding if use_embedding else "none"
        ),
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': seed,
        },
        'tests': [],
    }

    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()

    # Query GPU capabilities
    import cupy as cp
    dev = cp.cuda.Device()
    num_sms = dev.attributes['MultiProcessorCount']
    print(
        f"🔧 GPU has {num_sms} streaming multiprocessors "
        f"(SMs)"
    )

    for sweeps, reads, desc in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(
                f"\n⏰ Total timeout "
                f"({timeout_minutes} min) reached, stopping"
            )
            break

        # SA: 1 SM per nonce → num_sms models in parallel
        num_models = num_sms
        problem_size = len(nodes)

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
                h_dict, J_dict = generate_ising_model_from_nonce(
                    nonce, nodes, edges, h_values=h_values,
                )
                h_list.append(h_dict)
                J_list.append(J_dict)

            start_time = time.time()

            samplesets = sampler.sample_ising(
                h=h_list,
                J=J_list,
                num_reads=reads,
                num_sweeps=sweeps,
            )

            runtime = time.time() - start_time
            throughput = num_models / runtime

            # Collect stats from all models
            all_min_energies = []
            all_avg_energies = []
            for sampleset in samplesets:
                energies = list(sampleset.record.energy)
                all_min_energies.append(
                    float(min(energies))
                )
                all_avg_energies.append(
                    float(sum(energies) / len(energies))
                )

            # Use first sampleset for detailed analysis
            sampleset = samplesets[0]

            # Extract energies from sampleset
            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(
                sum(energies) / len(energies)
            )
            std_energy = float(
                (sum(
                    (e - avg_energy) ** 2 for e in energies
                ) / len(energies)) ** 0.5
            )

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

            # Evaluate the sampleset
            eval_fields = evaluate_baseline_sampleset(
                sampleset, nodes, edges, nonces[0],
                start_time, f"cuda-baseline-{sweeps}-{reads}", "CUDA",
                b"test_salt_cuda_baseline",
            )

            # Energy target analysis (same as CPU)
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
                print(
                    "  ⏰ Single test approaching timeout, "
                    "stopping further tests"
                )
                break

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary (same as CPU)
    total_runtime = time.time() - total_start_time
    results['_total_runtime_seconds'] = total_runtime
    print_results_summary(results, "CUDA Baseline Summary")

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    # Clean up sampler
    sampler.close()

    return results


def main():
    """Main function with command line argument parsing."""
    parser = build_baseline_argparser(
        'CUDA GPU baseline parameter testing tool',
    )
    parser.add_argument(
        '--embedding', type=str,
        help='Use embedded hardware topology instead of '
        'perfect topology (e.g., "Z(9,2)")',
    )
    args = parser.parse_args()

    # Parse h_values
    h_values = [
        float(v.strip()) for v in args.h_values.split(',')
    ]

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = (
            f"cuda_baseline_results_{timestamp}.json"
        )

    # Handle preset timeouts and filters
    only_label = args.only
    if args.quick:
        timeout = 10.0
        only_label = "Light CUDA"
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
        topology_path=args.topology,
    )

    print(f"\n✅ CUDA baseline test complete!")


if __name__ == "__main__":
    main()
