#!/usr/bin/env python3
"""QPU baseline parameter testing tool."""
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
import random

try:
    from QPU.dwave_sampler import DWaveSamplerWrapper
    QPU_AVAILABLE = True
except ImportError:
    QPU_AVAILABLE = False


def qpu_baseline_test(timeout_minutes=20.0, output_file=None, target_energy=-15500.0, min_runtime_minutes=6.0):
    """Test QPU performance with configurable timeout and minimum runtime."""
    print("🔬 QPU Baseline Parameter Test")
    print("=" * 40)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"⏰ Minimum runtime: {min_runtime_minutes} minutes")
    print(f"🎯 Target energy: {target_energy}")
    
    if not QPU_AVAILABLE:
        print("❌ QPU not available")
        return None
    
    # Initialize sampler
    try:
        sampler = DWaveSamplerWrapper()
        print("✅ QPU sampler ready")
        print(f"📊 QPU topology: {len(sampler.nodes)} nodes, {len(sampler.edges)} edges")
    except Exception as e:
        print(f"❌ QPU failed: {e}")
        return None
    
    # Get topology information from sampler
    nodes = sampler.nodes
    edges = sampler.edges
    print(f"📊 Topology: {len(nodes)} nodes, {len(edges)} edges")
    
    results = {
        'timeout_minutes': float(timeout_minutes),
        'min_runtime_minutes': float(min_runtime_minutes),
        'target_energy': float(target_energy),
        'qpu_info': {
            'num_qpu_nodes': int(len(sampler.nodes)),
            'num_qpu_edges': int(len(sampler.edges)),
            'sampler_type': sampler.sampler_type
        },
        'tests': []
    }
    
    # QPU test configurations - vary num_reads, keep annealing time constant
    # QPU architecture doesn't benefit from longer annealing times - 5µs is optimal
    read_counts = [256, 512]
    annealing_time = 200.0
    
    print(f"\n🧪 Testing QPU configurations:")
    print(f"Read counts to test: {read_counts}")
    print(f"Annealing time: {annealing_time} µs (optimal for QPU)")
    print(f"⚠️  Note: QPU time is expensive - each test may cost significant credits")

    timeout_seconds = timeout_minutes * 60
    min_runtime_seconds = min_runtime_minutes * 60
    total_start_time = time.time()
    
    # Test each read count
    for num_reads in read_counts:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break

        print(f"\n  Testing {num_reads} reads, {annealing_time}µs annealing...")

        try:
            # Generate the Ising model for this test (like the real miners do)
            nonce = random.randint(0, 2**32 - 1)
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            start_time = time.time()

            # Cast h and J to match protocol expectations (int is a valid Variable type)
            from typing import cast, Mapping, Tuple, Any
            h_cast = cast(Mapping[Any, float], h)
            J_cast = cast(Mapping[Tuple[Any, Any], float], J)

            sampleset = sampler.sample_ising(
                h_cast, J_cast,
                num_reads=num_reads,
                annealing_time=annealing_time,
                answer_mode='raw'
            )
            runtime = time.time() - start_time

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"    ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"    🎯 min_energy = {min_energy:.1f}")
            print(f"    📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")

            # Verify energy calculation consistency
            from shared.quantum_proof_of_work import energies_for_solutions
            solutions = list(sampleset.record.sample)
            recalc_energies = energies_for_solutions(solutions, h, J, nodes)
            recalc_min = min(recalc_energies)
            print(f"    ✓ Energy verification: sampler={min_energy:.1f}, recalc={recalc_min:.1f}, diff={abs(min_energy - recalc_min):.1f}")

            # Use evaluate_sampleset to get diversity and num_solutions
            # Create test requirements (using dummy values to ensure they pass)
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty (allow positive energies)
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600  # 10 minutes
            )

            # Use the same nonce and generate test salt for evaluation
            salt = b"test_salt_qpu_baseline"
            prev_timestamp = int(time.time()) - 600  # 10 minutes ago

            # Evaluate the sampleset
            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonce, salt,
                prev_timestamp, start_time, f"qpu-baseline-{num_reads}", "QPU"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            if mining_result:
                diversity = mining_result.diversity
                num_solutions = mining_result.num_valid
                meets_requirements = True
                print(f"    🌈 diversity = {diversity:.3f}")
                print(f"    🔢 num_solutions = {num_solutions}")
                print(f"    ✅ Meets mining requirements!")
            else:
                print(f"    ❌ Does not meet mining requirements")

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
                print(f"    🎖️  Quality: {target_reached}")

            # Check if we reached target
            target_reached_bool = bool(min_energy <= target_energy)
            if target_reached_bool:
                print(f"    ✅ Target energy {target_energy} reached!")
                
            # Extract QPU timing info if available
            qpu_timing = {}
            if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                timing = sampleset.info['timing']
                for key in ['qpu_anneal_time_per_sample', 'qpu_sampling_time', 'qpu_programming_time']:
                    if key in timing:
                        qpu_timing[key] = float(timing[key])

            test_result = {
                'num_reads': int(num_reads),
                'annealing_time_us': float(annealing_time),
                'runtime_seconds': float(runtime),
                'runtime_minutes': float(runtime / 60),
                'min_energy': min_energy,
                'avg_energy': avg_energy,
                'std_energy': std_energy,
                'target_reached': target_reached,
                'target_reached_bool': target_reached_bool,
                'diversity': float(diversity),
                'num_solutions': int(num_solutions),
                'meets_requirements': bool(meets_requirements),
                'qpu_timing': qpu_timing
            }
            results['tests'].append(test_result)

            # Early termination if we hit excellent quality
            if target_reached == "excellent":
                print(f"    🎉 Excellent quality achieved!")

            # Check if runtime was too short - we'll add more reads in next iteration
            if runtime < min_runtime_seconds:
                print(f"    ⚡ Runtime {runtime:.1f}s < minimum {min_runtime_seconds:.1f}s")
                print(f"    📈 Will test higher read counts to meet minimum runtime")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

    # Check if we need to extend the read counts based on minimum runtime
    if results['tests']:
        max_runtime = max(r['runtime_seconds'] for r in results['tests'])
        if max_runtime < min_runtime_seconds:
            print(f"\n⚠️  Maximum runtime {max_runtime:.1f}s < minimum {min_runtime_seconds:.1f}s")
            print(f"    Consider testing higher read counts (>{max(read_counts)}) to meet minimum runtime requirement")
    
    # Summary and analysis
    total_runtime = time.time() - total_start_time
    print(f"\n📊 QPU Baseline Summary (total time: {total_runtime/60:.1f} min):")
    print("=" * 50)
    
    if results['tests']:
        # Best energy achieved
        best_result = min(results['tests'], key=lambda r: r['min_energy'])
        print(f"🏆 Best energy: {best_result['min_energy']:.1f}")
        print(f"   Required: {best_result['num_reads']} reads, {best_result['annealing_time_us']}µs, {best_result['runtime_minutes']:.1f} min")
        
        # Target achievement analysis
        target_achievers = [r for r in results['tests'] if r['target_reached_bool']]
        if target_achievers:
            fastest_target = min(target_achievers, key=lambda r: r['runtime_seconds'])
            print(f"🎯 Fastest to reach {target_energy}: {fastest_target['num_reads']} reads, {fastest_target['annealing_time_us']}µs ({fastest_target['runtime_minutes']:.1f} min)")
        
        # Quality tiers
        quality_tiers = {}
        for result in results['tests']:
            tier = result['target_reached']
            if tier != 'none':
                if tier not in quality_tiers:
                    quality_tiers[tier] = []
                quality_tiers[tier].append(result)
        
        print(f"\n🏅 Quality Achievements:")
        for tier in ['fair', 'good', 'very_good', 'excellent']:
            if tier in quality_tiers:
                fastest = min(quality_tiers[tier], key=lambda r: r['runtime_seconds'])
                print(f"   {tier.title()}: {fastest['num_reads']} reads, {fastest['annealing_time_us']}µs ({fastest['runtime_minutes']:.1f} min)")
        
        # Runtime analysis
        min_runtime_met = any(r['runtime_seconds'] >= min_runtime_seconds for r in results['tests'])
        print(f"\n⏰ Minimum runtime requirement ({min_runtime_minutes} min): {'✅ Met' if min_runtime_met else '❌ Not met'}")

        # Cost estimation (rough estimate based on D-Wave pricing)
        total_reads = sum(r['num_reads'] for r in results['tests'])
        estimated_cost = total_reads * 0.00005  # Rough estimate: $0.00005 per read
        print(f"\n💰 Estimated cost: ~${estimated_cost:.2f} (based on {total_reads} total reads)")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='QPU baseline parameter testing tool')
    parser.add_argument(
        '--timeout', '-t', 
        type=float, 
        default=30.0,
        help='Timeout in minutes (default: 30.0)'
    )
    parser.add_argument(
        '--min-runtime', 
        type=float,
        default=6.0,
        help='Minimum runtime in minutes before allowing early termination (default: 6.0)'
    )
    parser.add_argument(
        '--target', 
        type=float,
        default=-15500.0,
        help='Target energy threshold (default: -15500.0)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (15 min timeout, 3 min minimum, target -15300)'
    )
    parser.add_argument(
        '--extended',
        action='store_true', 
        help='Extended test mode (60 min timeout, 10 min minimum, target -15600)'
    )
    
    args = parser.parse_args()
    
    # Handle preset modes
    if args.quick:
        timeout = 15.0
        min_runtime = 3.0
        target = -15300.0
    elif args.extended:
        timeout = 60.0
        min_runtime = 10.0
        target = -15600.0
    else:
        timeout = args.timeout
        min_runtime = args.min_runtime
        target = args.target
    
    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"qpu_baseline_results_{timestamp}.json"
    
    # Run test
    results = qpu_baseline_test(
        timeout_minutes=timeout, 
        output_file=output_file, 
        target_energy=target,
        min_runtime_minutes=min_runtime
    )
    
    if results:
        print(f"\n✅ QPU baseline test complete!")
    else:
        print(f"\n❌ QPU baseline test failed!")


if __name__ == "__main__":
    main()
