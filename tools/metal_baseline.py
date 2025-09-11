#!/usr/bin/env python3
"""Metal GPU baseline parameter testing tool."""
import argparse
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
from shared.quantum_proof_of_work import generate_ising_model_from_nonce

try:
    from GPU.metal_sampler import MetalSampler
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

try:
    from GPU.metal_sampler_optimized_chunks import OptimizedChunkMetalSampler
    OPTIMIZED_METAL_AVAILABLE = True
except ImportError:
    OPTIMIZED_METAL_AVAILABLE = False


def metal_optimized_comparison_test(timeout_minutes=10.0, output_file=None, target_energy=-15500.0):
    """Compare baseline vs optimized Metal GPU performance."""
    print("🚀 Metal GPU Optimization Comparison Test")
    print("=" * 50)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎯 Target energy: {target_energy}")
    
    # Check availability
    if not METAL_AVAILABLE:
        print("❌ Original Metal GPU sampler not available")
        original_sampler = None
    else:
        try:
            original_sampler = MetalSampler("mps")
            print("✅ Original Metal sampler ready")
        except Exception as e:
            print(f"❌ Original Metal failed: {e}")
            original_sampler = None
    
    if not OPTIMIZED_METAL_AVAILABLE:
        print("❌ Optimized Metal GPU sampler not available")
        optimized_sampler = None
    else:
        try:
            optimized_sampler = OptimizedChunkMetalSampler("mps")
            print("✅ Optimized Metal sampler ready")
        except Exception as e:
            print(f"❌ Optimized Metal failed: {e}")
            optimized_sampler = None
    
    if not original_sampler and not optimized_sampler:
        print("❌ No Metal samplers available")
        return None
    
    # Generate test problem (same as CPU baseline)
    cpu_sampler = SimulatedAnnealingStructuredSampler()
    nodes = cpu_sampler.nodes
    edges = cpu_sampler.edges
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges)
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    
    # Test configurations
    test_configs = [
        (64, 128),   # Quick test
        (64, 256),   # Medium test  
        (64, 512),   # Standard test
    ]
    
    results = {
        'timeout_minutes': float(timeout_minutes),
        'target_energy': float(target_energy),
        'problem_info': {
            'num_variables': int(len(h)),
            'num_couplings': int(len(J)),
            'seed': int(seed)
        },
        'comparisons': []
    }
    
    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()
    
    print(f"\n🧪 Running comparison tests:")
    
    for num_reads, num_sweeps in test_configs:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break
        
        print(f"\n{'='*60}")
        print(f"📋 Test Configuration: {num_reads} reads, {num_sweeps} sweeps")
        print('='*60)
        
        comparison_result = {
            'num_reads': num_reads,
            'num_sweeps': num_sweeps,
            'original': None,
            'optimized': None,
            'speedup': None,
            'energy_comparison': None
        }
        
        # Test original sampler
        if original_sampler:
            print(f"\n🔵 Testing ORIGINAL Metal Sampler...")
            try:
                start_time = time.time()
                sampleset = original_sampler.sample_ising(
                    h=h, J=J,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps
                )
                runtime = time.time() - start_time
                
                energies = list(sampleset.record.energy)
                min_energy = float(min(energies))
                avg_energy = float(sum(energies) / len(energies))
                
                comparison_result['original'] = {
                    'runtime_s': runtime,
                    'avg_sweep_time_ms': (runtime / num_sweeps) * 1000,
                    'min_energy': min_energy,
                    'avg_energy': avg_energy,
                    'target_reached': min_energy <= target_energy
                }
                
                print(f"    ⏱️  Runtime: {runtime:.3f}s ({runtime/num_sweeps*1000:.2f}ms per sweep)")
                print(f"    🎯 Min energy: {min_energy:.1f}")
                print(f"    📊 Avg energy: {avg_energy:.1f}")
                
            except Exception as e:
                print(f"    ❌ Original test failed: {e}")
        
        # Test optimized sampler
        if optimized_sampler:
            print(f"\n🟢 Testing OPTIMIZED Metal Sampler...")
            try:
                start_time = time.time()
                sampleset = optimized_sampler.sample_ising(
                    h=h, J=J,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps
                )
                runtime = time.time() - start_time
                
                energies = list(sampleset.record.energy)
                min_energy = float(min(energies))
                avg_energy = float(sum(energies) / len(energies))
                
                comparison_result['optimized'] = {
                    'runtime_s': runtime,
                    'avg_sweep_time_ms': (runtime / num_sweeps) * 1000,
                    'min_energy': min_energy,
                    'avg_energy': avg_energy,
                    'target_reached': min_energy <= target_energy
                }
                
                print(f"    ⏱️  Runtime: {runtime:.3f}s ({runtime/num_sweeps*1000:.2f}ms per sweep)")
                print(f"    🎯 Min energy: {min_energy:.1f}")
                print(f"    📊 Avg energy: {avg_energy:.1f}")
                
            except Exception as e:
                print(f"    ❌ Optimized test failed: {e}")
        
        # Calculate comparison metrics
        if comparison_result['original'] and comparison_result['optimized']:
            original_time = comparison_result['original']['avg_sweep_time_ms']
            optimized_time = comparison_result['optimized']['avg_sweep_time_ms']
            speedup = original_time / optimized_time
            
            original_energy = comparison_result['original']['min_energy']
            optimized_energy = comparison_result['optimized']['min_energy']
            energy_diff = optimized_energy - original_energy
            
            comparison_result['speedup'] = speedup
            comparison_result['energy_comparison'] = {
                'energy_difference': energy_diff,
                'original_better': original_energy < optimized_energy,
                'optimized_better': optimized_energy < original_energy,
                'equivalent': abs(energy_diff) < 50  # Within 50 energy units
            }
            
            print(f"\n📊 COMPARISON RESULTS:")
            print(f"    🚀 Speedup: {speedup:.2f}x faster")
            print(f"    ⚡ Time improvement: {((original_time - optimized_time) / original_time) * 100:.1f}%")
            print(f"    🎯 Energy difference: {energy_diff:+.1f}")
            
            if speedup > 1.5:
                print(f"    ✅ SIGNIFICANT performance improvement!")
            
            quality_original = "EXCELLENT" if original_energy <= target_energy else "GOOD" if original_energy <= target_energy + 500 else "FAIR"
            quality_optimized = "EXCELLENT" if optimized_energy <= target_energy else "GOOD" if optimized_energy <= target_energy + 500 else "FAIR"
            print(f"    🏆 Energy quality: Original={quality_original}, Optimized={quality_optimized}")
        
        results['comparisons'].append(comparison_result)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"🏁 FINAL OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    if results['comparisons']:
        # Calculate average performance improvement
        valid_comparisons = [c for c in results['comparisons'] if c['speedup']]
        if valid_comparisons:
            avg_speedup = sum(c['speedup'] for c in valid_comparisons) / len(valid_comparisons)
            print(f"📈 Average speedup: {avg_speedup:.2f}x")
            print(f"🎯 Performance improvement: {((avg_speedup - 1) * 100):.1f}%")
            
            # Find best configuration
            best_config = max(valid_comparisons, key=lambda x: x['speedup'])
            print(f"🏆 Best configuration: {best_config['num_reads']} reads, {best_config['num_sweeps']} sweeps")
            print(f"    Speedup: {best_config['speedup']:.2f}x")
            print(f"    Original: {best_config['original']['avg_sweep_time_ms']:.2f}ms per sweep")
            print(f"    Optimized: {best_config['optimized']['avg_sweep_time_ms']:.2f}ms per sweep")
            
            if avg_speedup > 2.0:
                print(f"\n🎉 OUTSTANDING OPTIMIZATION SUCCESS!")
                print(f"   Achieved {avg_speedup:.2f}x average performance improvement")
            elif avg_speedup > 1.5:
                print(f"\n✅ EXCELLENT OPTIMIZATION SUCCESS!")
                print(f"   Achieved {avg_speedup:.2f}x average performance improvement")
            else:
                print(f"\n📊 Moderate performance improvement: {avg_speedup:.2f}x")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
    
    return results

def metal_baseline_test(timeout_minutes=20.0, output_file=None, target_energy=-15500.0):
    """Test Metal GPU performance with configurable timeout."""
    print("🔬 Metal GPU Baseline Parameter Test")
    print("=" * 40)
    print(f"⏰ Timeout: {timeout_minutes} minutes")
    print(f"🎯 Target energy: {target_energy}")
    
    if not METAL_AVAILABLE:
        print("❌ Metal GPU not available")
        return None
    
    # Initialize sampler
    try:
        sampler = MetalSampler("mps")
        print("✅ Metal sampler ready")
    except Exception as e:
        print(f"❌ Metal failed: {e}")
        return None
    
    # Generate test problem (same as CPU baseline)
    cpu_sampler = SimulatedAnnealingStructuredSampler()
    nodes = cpu_sampler.nodes
    edges = cpu_sampler.edges
    seed = 12345  # Fixed seed for reproducible results
    h, J = generate_ising_model_from_nonce(seed, nodes, edges)
    print(f"📊 Problem: {len(h)} variables, {len(J)} couplings")
    
    results = {
        'timeout_minutes': float(timeout_minutes),  # Ensure float
        'target_energy': float(target_energy),      # Ensure float
        'problem_info': {
            'num_variables': int(len(h)),           # Ensure int
            'num_couplings': int(len(J)),           # Ensure int
            'seed': int(seed)                       # Ensure int
        },
        'tests': []
    }
    
    # Metal test configurations - broader sweep range
    sweep_ranges = [
        32, 64, 96, 128, 192, 256, 320, 384, 448, 512,
        640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048
    ]
    
    # Base read count (can be adjusted)
    base_reads = 64
    
    print(f"\n🧪 Testing Metal sweep ranges (reads={base_reads}):")
    print(f"Sweep range: {sweep_ranges[0]} to {sweep_ranges[-1]}")
    
    timeout_seconds = timeout_minutes * 60
    total_start_time = time.time()
    
    for num_sweeps in sweep_ranges:
        elapsed_total = time.time() - total_start_time
        if elapsed_total > timeout_seconds:
            print(f"\n⏰ Total timeout ({timeout_minutes} min) reached, stopping")
            break
        
        print(f"\n  Testing {num_sweeps} sweeps...")
        
        try:
            start_time = time.time()
            sampleset = sampler.sample_ising(
                h=h, J=J,
                num_reads=base_reads,
                num_sweeps=num_sweeps
            )
            runtime = time.time() - start_time
            
            energies = list(sampleset.record.energy)
            min_energy = float(min(energies))  # Convert to Python float
            avg_energy = float(sum(energies) / len(energies))  # Convert to Python float
            std_energy = float((sum((e - avg_energy)**2 for e in energies) / len(energies)) ** 0.5)  # Convert to Python float
            
            print(f"    ⏱️  {runtime/60:.1f} min ({runtime:.1f}s)")
            print(f"    🎯 min_energy = {min_energy:.1f}")
            print(f"    📊 avg_energy = {avg_energy:.1f} (±{std_energy:.1f})")
            
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
            target_reached_bool = bool(min_energy <= target_energy)  # Convert to Python bool
            if target_reached_bool:
                print(f"    ✅ Target energy {target_energy} reached!")
            
            test_result = {
                'num_sweeps': int(num_sweeps),  # Ensure int
                'num_reads': int(base_reads),   # Ensure int
                'runtime_seconds': float(runtime),  # Ensure float
                'runtime_minutes': float(runtime / 60),  # Ensure float
                'min_energy': min_energy,       # Already converted to float
                'avg_energy': avg_energy,       # Already converted to float
                'std_energy': std_energy,       # Already converted to float
                'target_reached': target_reached,  # String, already JSON serializable
                'target_reached_bool': target_reached_bool  # Already converted to bool
            }
            results['tests'].append(test_result)
            
            # Early termination if we hit excellent quality
            if target_reached == "excellent":
                print(f"    🎉 Excellent quality achieved, can stop here if desired")
            
            # Individual test timeout check
            if runtime > timeout_seconds * 0.5:  # 50% of total timeout for single test
                print(f"    ⏰ Single test taking too long, stopping further tests")
                break
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            break
    
    # Summary and analysis
    total_runtime = time.time() - total_start_time
    print(f"\n📊 Metal Baseline Summary (total time: {total_runtime/60:.1f} min):")
    print("=" * 50)
    
    if results['tests']:
        # Best energy achieved
        best_result = min(results['tests'], key=lambda r: r['min_energy'])
        print(f"🏆 Best energy: {best_result['min_energy']:.1f}")
        print(f"   Required: {best_result['num_sweeps']} sweeps, {best_result['runtime_minutes']:.1f} min")
        
        # Target achievement analysis
        target_achievers = [r for r in results['tests'] if r['target_reached_bool']]
        if target_achievers:
            fastest_target = min(target_achievers, key=lambda r: r['runtime_seconds'])
            print(f"🎯 Fastest to reach {target_energy}: {fastest_target['num_sweeps']} sweeps ({fastest_target['runtime_minutes']:.1f} min)")
        
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
                print(f"   {tier.title()}: {fastest['num_sweeps']} sweeps ({fastest['runtime_minutes']:.1f} min)")
        
        # Scaling analysis
        if len(results['tests']) > 1:
            first = results['tests'][0]
            last = results['tests'][-1]
            sweep_scaling = (last['runtime_seconds'] / first['runtime_seconds']) / (last['num_sweeps'] / first['num_sweeps'])
            print(f"\n📈 Scaling: {sweep_scaling:.2f}x runtime per x sweeps")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Metal GPU baseline parameter testing tool')
    parser.add_argument(
        '--timeout', '-t', 
        type=float, 
        default=20.0,
        help='Timeout in minutes (default: 20.0)'
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
        help='Quick test mode (5 minute timeout, target -15300)'
    )
    parser.add_argument(
        '--extended',
        action='store_true', 
        help='Extended test mode (60 minute timeout, target -15600)'
    )
    parser.add_argument(
        '--cpu-match',
        action='store_true',
        help='CPU matching mode (30 min timeout, target -15500 to match CPU performance)'
    )
    parser.add_argument(
        '--compare-optimized',
        action='store_true',
        help='Compare original vs optimized Metal samplers (10 min timeout)'
    )
    
    args = parser.parse_args()
    
    # Handle preset modes
    if args.compare_optimized:
        timeout = 10.0
        target = -15500.0
        test_function = metal_optimized_comparison_test
        file_prefix = "metal_comparison"
    elif args.quick:
        timeout = 5.0
        target = -15300.0
        test_function = metal_baseline_test
        file_prefix = "metal_baseline"
    elif args.extended:
        timeout = 60.0
        target = -15600.0
        test_function = metal_baseline_test
        file_prefix = "metal_baseline"
    elif args.cpu_match:
        timeout = 30.0
        target = -15500.0
        test_function = metal_baseline_test
        file_prefix = "metal_baseline"
    else:
        timeout = args.timeout
        target = args.target
        test_function = metal_baseline_test
        file_prefix = "metal_baseline"
    
    # Generate default output filename if not specified
    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"{file_prefix}_results_{timestamp}.json"
    
    # Run test
    results = test_function(timeout_minutes=timeout, output_file=output_file, target_energy=target)
    
    if results:
        print(f"\n✅ Metal baseline test complete!")
    else:
        print(f"\n❌ Metal baseline test failed!")


if __name__ == "__main__":
    main()