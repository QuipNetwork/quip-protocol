#!/usr/bin/env python3
"""CUDA GPU baseline parameter testing tool."""
import argparse
import sys
import time
import json
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce, evaluate_sampleset, calculate_diversity
from shared.block_requirements import BlockRequirements

try:
    from GPU.cuda_sa import CudaSASampler
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def cuda_baseline_test(timeout_minutes=10.0, output_file=None):
    """Test CUDA GPU performance with CPU baseline format and evaluation logic."""
    print("🔬 CUDA GPU Baseline Parameter Test (CuPy RawKernel)")
    print("=" * 50)
    print(f"⏰ Timeout: {timeout_minutes} minutes")

    # Initialize CUDA SA sampler
    cuda_sampler = None
    sampler_type = "unknown"

    if CUDA_AVAILABLE:
        try:
            cuda_sampler = CudaSASampler()
            nodes = cuda_sampler.nodes
            edges = cuda_sampler.edges
            sampler_type = "cuda-sa"
            print("✅ CUDA SA sampler ready (CuPy RawKernel)")
        except Exception as e:
            print(f"⚠️ CUDA SA sampler failed: {e}")
            cuda_sampler = None

    if cuda_sampler is None:
        try:
            cuda_sampler = OptimizedChunkCUDASampler("mps")
            nodes = cuda_sampler.nodes
            edges = cuda_sampler.edges
            sampler_type = "optimized-chunks"
            print("✅ CUDA optimized sampler ready (PyTorch MPS fallback)")
        except Exception as e:
            print(f"❌ CUDA optimized failed: {e}")
            return None
    
    if cuda_sampler is None:
        print("❌ No CUDA samplers available")
        return None
    
    # Initial problem setup to show problem size
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges)
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    
    # Test configurations - matching CPU baseline exactly
    test_configs = [
        (256, 64, "Light CUDA"),
        (512, 100, "Low CUDA"),
        (1024, 100, "Medium CUDA"),
        (2048, 150, "High CUDA"),
        (4096, 200, "Very High CUDA"),
        (8192, 200, "Max CUDA")
    ]
    
    print(f"\n🧪 Testing CUDA configurations:")
    
    results = {
        'timeout_minutes': timeout_minutes,
        'sampler_type': sampler_type,
        'problem_info': {
            'num_variables': len(h),
            'num_couplings': len(J),
            'seed': seed
        },
        'tests': []
    }
    
    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()
    
    for sweeps, reads, desc in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break
            
        print(f"\n{desc}: {sweeps} sweeps, {reads} reads")
        
        try:
            # Generate the Ising model first (like the real miners do)
            nonce = random.randint(0, 2**32 - 1)
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            start_time = time.time()
            sampleset = cuda_sampler.sample_ising(
                h=h, J=J,
                num_reads=reads,
                num_sweeps=sweeps
            )
            runtime = time.time() - start_time

            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))
            avg_energy = float(sum(energies) / len(energies))
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)

            print(f"  ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"  🎯 min_energy = {min_energy:.1f}")
            print(f"  📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")

            # Use evaluate_sampleset to get diversity and num_solutions (same as CPU)
            requirements = BlockRequirements(
                difficulty_energy=0.0,       # Very lenient difficulty (allow positive energies)
                min_diversity=0.1,           # Low diversity requirement
                min_solutions=1,             # Low solution count requirement
                timeout_to_difficulty_adjustment_decay=600  # 10 minutes
            )

            # Use the same nonce and generate test salt for evaluation
            salt = b"test_salt_cuda_baseline"
            prev_timestamp = int(time.time()) - 600  # 10 minutes ago

            # Evaluate the sampleset
            mining_result = evaluate_sampleset(
                sampleset, requirements, nodes, edges, nonce, salt,
                prev_timestamp, start_time, f"cuda-baseline-{sweeps}-{reads}", "CUDA"
            )

            diversity = 0.0
            num_solutions = 0
            meets_requirements = False

            # Calculate diversity of top 10 solutions by energy (same as CPU)
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
        help='Quick test mode (3 minute timeout)'
    )
    parser.add_argument(
        '--extended',
        action='store_true', 
        help='Extended test mode (30 minute timeout)'
    )
    
    args = parser.parse_args()
    
    # Handle preset timeouts
    if args.quick:
        timeout = 3.0
    elif args.extended:
        timeout = 30.0
    else:
        timeout = args.timeout
    
    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"cuda_baseline_results_{timestamp}.json"
    
    # Run test
    cuda_baseline_test(timeout_minutes=timeout, output_file=output_file)

    print(f"\n✅ CUDA baseline test complete!")


if __name__ == "__main__":
    main()