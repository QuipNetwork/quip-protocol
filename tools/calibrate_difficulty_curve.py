#!/usr/bin/env python3
"""Experimental tool to calibrate difficulty curve parameters.

This tool maps out the actual difficulty curve by running SA at different
computational effort levels (num_sweeps) and measuring achieved energies.

The goal is to identify empirical min/knee/max energy values for
adjust_energy_along_curve() instead of using guessed values.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
from dwave_topologies import DEFAULT_TOPOLOGY

# Try to import GPU samplers
try:
    from GPU.metal_sa import MetalSASampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

try:
    from GPU.cuda_sa_sampler import CudaSASampler
    CUDA_AVAILABLE = False  # Disabled for now (requires CUDA machine)
except ImportError:
    CUDA_AVAILABLE = False


def calibrate_curve(
    topology_name: str = "advantage2",
    h_values: List[float] = None,
    sweeps_range: List[int] = None,
    reads_range: List[int] = None,
    time_per_test: float = 600.0,  # 10 minutes per test configuration
    output_file: str = None,
    use_gpu: bool = True
) -> Dict:
    """Run SA at different sweep/reads combinations to map out difficulty curve.

    Args:
        topology_name: Topology to use (default: advantage2)
        h_values: List of h field values (default: [-1, 0, 1])
        sweeps_range: List of num_sweeps values to test
        reads_range: List of num_reads values to test (default: [64, 128, 256])
        time_per_test: Time budget (seconds) per test config (default: 600s = 10 min)
        output_file: Output JSON file path
        use_gpu: Auto-detect and use GPU if available (default: True)

    Returns:
        Dictionary with calibration results
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]

    if sweeps_range is None:
        # Always test practical mining ranges: 64, 128, 256
        sweeps_range = [64, 128, 256, 512, 1024, 2048, 4096]

    if reads_range is None:
        # Test practical reads ranges
        reads_range = [64, 128, 256]

    print("=" * 70)
    print("🔬 Difficulty Curve Calibration Tool")
    print("=" * 70)
    print(f"Topology: {topology_name}")
    print(f"h_values: {h_values}")
    print(f"Sweep range: {sweeps_range}")
    print(f"Reads range: {reads_range}")
    print(f"Time per test config: {time_per_test:.0f}s ({time_per_test/60:.1f} min)")
    total_tests = len(sweeps_range) * len(reads_range)
    total_time = total_tests * time_per_test
    print(f"Total tests: {total_tests} ({total_time/60:.0f} min estimated)")
    print()

    # Auto-detect and initialize sampler
    sampler_type = "CPU"
    if use_gpu and METAL_AVAILABLE:
        print("🚀 GPU (Metal) detected - using Metal SA")
        sampler = MetalSASampler()
        sampler_type = "Metal"
        # Get topology for Metal (different API)
        topology_graph = DEFAULT_TOPOLOGY.graph
        nodes = list(topology_graph.nodes())
        edges = list(topology_graph.edges())
    elif use_gpu and CUDA_AVAILABLE:
        print("🚀 GPU (CUDA) detected - using CUDA SA")
        sampler = CudaSASampler()
        sampler_type = "CUDA"
        nodes = sampler.nodes
        edges = sampler.edges
    else:
        if use_gpu:
            print("⚠️  No GPU available - falling back to CPU")
        else:
            print("💻 Using CPU SA (GPU disabled)")
        sampler = SimulatedAnnealingStructuredSampler()
        sampler_type = "CPU"
        nodes = sampler.nodes
        edges = sampler.edges

    print(f"Problem size: {len(nodes)} nodes, {len(edges)} edges")
    print()

    # Results storage
    results = {
        'topology': topology_name,
        'sampler_type': sampler_type,
        'h_values': h_values,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'time_per_test_seconds': time_per_test,
        'sweeps_range': sweeps_range,
        'reads_range': reads_range,
        'curve_data': []
    }

    # Run experiments for each (sweeps, reads) combination
    for sweeps in sweeps_range:
        for num_reads in reads_range:
            print(f"Testing num_sweeps={sweeps}, num_reads={num_reads}...")
            start_time = time.time()

            energies = []
            sample_count = 0

            # Run samples until time budget is exhausted
            seed = 0
            while (time.time() - start_time) < time_per_test:
                # Generate problem with deterministic nonce
                h, J = generate_ising_model_from_nonce(
                    seed,
                    nodes,
                    edges,
                    h_values=h_values
                )

                # Sample with SA (handle different APIs)
                if sampler_type == "Metal":
                    # Metal uses batched API
                    samplesets = sampler.sample_ising(
                        h=[h],
                        J=[J],
                        num_reads=num_reads,
                        num_sweeps=sweeps
                    )
                    sampleset = samplesets[0]
                else:
                    # CPU/CUDA use single problem API
                    sampleset = sampler.sample_ising(
                        h=h,
                        J=J,
                        num_reads=num_reads,
                        num_sweeps=sweeps
                    )

                # Get best energy from this sample
                best_energy = float(sampleset.first.energy)
                energies.append(best_energy)
                sample_count += 1
                seed += 1

            runtime = time.time() - start_time

            # Calculate statistics
            energies_array = np.array(energies)
            avg_energy = float(np.mean(energies_array))
            std_energy = float(np.std(energies_array))
            min_energy = float(np.min(energies_array))
            max_energy = float(np.max(energies_array))
            median_energy = float(np.median(energies_array))

            print(f"  ⏱️  Runtime: {runtime:.1f}s ({runtime/60:.1f} min) - {sample_count} samples")
            print(f"  📊 Avg energy: {avg_energy:.1f} (±{std_energy:.1f})")
            print(f"  🎯 Min energy: {min_energy:.1f}")
            print(f"  📈 Median: {median_energy:.1f}")
            print()

            # Store results
            curve_point = {
                'sweeps': int(sweeps),
                'num_reads': int(num_reads),
                'avg_energy': avg_energy,
                'std_energy': std_energy,
                'min_energy': min_energy,
                'max_energy': max_energy,
                'median_energy': median_energy,
                'runtime_seconds': runtime,
                'samples': sample_count,
                'all_energies': energies  # Store all for analysis
            }
            results['curve_data'].append(curve_point)

    # Analyze curve to find calibration parameters
    print("=" * 70)
    print("📊 Calibration Analysis")
    print("=" * 70)

    curve_data = results['curve_data']

    # Find minimum energy (highest computational effort)
    min_point = min(curve_data, key=lambda x: x['avg_energy'])
    calibrated_min_energy = min_point['avg_energy']

    # Find maximum energy (lowest computational effort)
    max_point = max(curve_data, key=lambda x: x['avg_energy'])
    calibrated_max_energy = max_point['avg_energy']

    # Find knee point (diminishing returns)
    # Use derivative approximation: find where improvement rate drops below threshold
    knee_sweeps = None
    knee_energy = None

    if len(curve_data) >= 3:
        # Group by sweeps and take best energy for each sweep level
        sweeps_to_best = {}
        for point in curve_data:
            s = point['sweeps']
            e = point['avg_energy']
            if s not in sweeps_to_best or e < sweeps_to_best[s]:
                sweeps_to_best[s] = e

        # Sort by sweeps
        sorted_sweeps = sorted(sweeps_to_best.keys())

        if len(sorted_sweeps) >= 3:
            # Calculate energy improvement rate (delta_energy / delta_sweeps)
            improvements = []
            for i in range(1, len(sorted_sweeps)):
                delta_energy = sweeps_to_best[sorted_sweeps[i]] - sweeps_to_best[sorted_sweeps[i-1]]
                delta_sweeps = sorted_sweeps[i] - sorted_sweeps[i-1]
                if delta_sweeps > 0:  # Avoid division by zero
                    improvement_rate = delta_energy / delta_sweeps  # Negative = improvement
                    improvements.append((sorted_sweeps[i], improvement_rate))

            # Find where improvement rate drops to 20% of initial rate
            # (i.e., 80% reduction in marginal benefit)
            if improvements:
                initial_rate = improvements[0][1]
                threshold_rate = initial_rate * 0.2

                for sweeps, rate in improvements:
                    if rate > threshold_rate:  # Less improvement (more negative becomes less negative)
                        knee_sweeps = sweeps
                        knee_energy = sweeps_to_best[sweeps]
                        break

                # If no knee found, use middle point
                if knee_sweeps is None:
                    mid_idx = len(sorted_sweeps) // 2
                    knee_sweeps = sorted_sweeps[mid_idx]
                    knee_energy = sweeps_to_best[knee_sweeps]
        else:
            # Not enough unique sweep values, use middle point
            mid_idx = len(curve_data) // 2
            knee_sweeps = curve_data[mid_idx]['sweeps']
            knee_energy = curve_data[mid_idx]['avg_energy']

    # Store calibration
    calibration = {
        'min_energy': calibrated_min_energy,
        'min_energy_sweeps': min_point['sweeps'],
        'knee_energy': knee_energy,
        'knee_sweeps': knee_sweeps,
        'max_energy': calibrated_max_energy,
        'max_energy_sweeps': max_point['sweeps']
    }
    results['calibration'] = calibration

    print(f"Calibrated parameters:")
    print(f"  🎯 Min energy: {calibrated_min_energy:.1f} (at {min_point['sweeps']} sweeps)")
    if knee_energy:
        print(f"  📍 Knee point: {knee_energy:.1f} (at {knee_sweeps} sweeps)")
    print(f"  📈 Max energy: {calibrated_max_energy:.1f} (at {max_point['sweeps']} sweeps)")
    print()

    # Print suggested code update
    print("💡 Suggested update for adjust_energy_along_curve():")
    print(f"   min_energy = {calibrated_min_energy:.1f}  # Hardest difficulty")
    if knee_energy:
        print(f"   knee_energy = {knee_energy:.1f}  # Mid-range difficulty")
    print(f"   max_energy = {calibrated_max_energy:.1f}  # Easiest difficulty")
    print()

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")

    return results


def plot_curve(results: Dict, output_file: str = None):
    """Plot the difficulty curve."""
    curve_data = results['curve_data']

    sweeps = [point['sweeps'] for point in curve_data]
    avg_energies = [point['avg_energy'] for point in curve_data]
    std_energies = [point['std_energy'] for point in curve_data]
    min_energies = [point['min_energy'] for point in curve_data]

    # Set style
    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Energy vs Sweeps
    ax1.errorbar(sweeps, avg_energies, yerr=std_energies,
                 marker='o', capsize=5, label='Average ± Std')
    ax1.plot(sweeps, min_energies, marker='s', linestyle='--',
             label='Best observed', alpha=0.7)

    # Mark calibration points
    calibration = results.get('calibration', {})
    if calibration.get('knee_sweeps'):
        ax1.axvline(calibration['knee_sweeps'], color='red', linestyle=':',
                   label=f"Knee ({calibration['knee_sweeps']} sweeps)")

    ax1.set_xlabel('Number of Sweeps', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Difficulty Curve: Energy vs Computational Effort', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement Rate
    if len(curve_data) >= 2:
        improvement_rates = []
        sweep_midpoints = []
        for i in range(1, len(curve_data)):
            delta_energy = curve_data[i]['avg_energy'] - curve_data[i-1]['avg_energy']
            delta_sweeps = curve_data[i]['sweeps'] - curve_data[i-1]['sweeps']
            rate = delta_energy / delta_sweeps
            improvement_rates.append(rate)
            sweep_midpoints.append((curve_data[i]['sweeps'] + curve_data[i-1]['sweeps']) / 2)

        ax2.plot(sweep_midpoints, improvement_rates, marker='o', color='green')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Number of Sweeps', fontsize=12)
        ax2.set_ylabel('Improvement Rate (ΔE / ΔSweeps)', fontsize=12)
        ax2.set_title('Marginal Improvement Rate', fontsize=14)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Experimental difficulty curve calibration tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Quick calibration with default settings
  python tools/calibrate_difficulty_curve.py

  # Full calibration with extended sweep range
  python tools/calibrate_difficulty_curve.py \\
      --sweeps-range 64,128,256,512,1024,2048,4096,8192 \\
      --samples 100 \\
      --output curve_calibration.json

  # Test different h distributions
  python tools/calibrate_difficulty_curve.py --h-values 0
  python tools/calibrate_difficulty_curve.py --h-values -1,1
  python tools/calibrate_difficulty_curve.py --h-values -1,0,1
        """
    )

    parser.add_argument(
        '--topology',
        type=str,
        default='advantage2',
        help='Topology name (default: advantage2)'
    )
    parser.add_argument(
        '--h-values',
        type=str,
        default='-1,0,1',
        help='Comma-separated h field values (default: -1,0,1)'
    )
    parser.add_argument(
        '--sweeps-range',
        type=str,
        default='64,128,256,512,1024,2048,4096',
        help='Comma-separated num_sweeps values to test (default: 64,128,256,512,1024,2048,4096)'
    )
    parser.add_argument(
        '--time-per-test',
        type=float,
        default=600.0,
        help='Time budget in seconds per test configuration (default: 600s = 10 min)'
    )
    parser.add_argument(
        '--reads-range',
        type=str,
        default='64,128,256',
        help='Comma-separated num_reads values to test (default: 64,128,256)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: auto-generated)'
    )
    parser.add_argument(
        '--plot',
        type=str,
        help='Save plot to file (e.g., curve.png)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU auto-detection, use CPU only'
    )

    args = parser.parse_args()

    # Parse h_values
    h_values = [float(v.strip()) for v in args.h_values.split(',')]

    # Parse sweeps_range
    sweeps_range = [int(v.strip()) for v in args.sweeps_range.split(',')]

    # Parse reads_range
    reads_range = [int(v.strip()) for v in args.reads_range.split(',')]

    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        h_str = "_".join(str(int(v)) for v in h_values)
        output_file = f"curve_calibration_h{h_str}_{timestamp}.json"

    # Run calibration
    results = calibrate_curve(
        topology_name=args.topology,
        h_values=h_values,
        sweeps_range=sweeps_range,
        reads_range=reads_range,
        time_per_test=args.time_per_test,
        output_file=output_file,
        use_gpu=not args.no_gpu
    )

    # Generate plot
    if not args.no_plot:
        plot_output = args.plot
        if not plot_output:
            # Auto-generate plot filename based on JSON filename
            plot_output = output_file.replace('.json', '.png')

        try:
            plot_curve(results, plot_output)
        except Exception as e:
            print(f"⚠️  Failed to generate plot: {e}")

    print()
    print("✅ Calibration complete!")


if __name__ == "__main__":
    main()
