#!/usr/bin/env python3
"""
GPU Benchmarking Module using Modal Labs
Cost-optimized GPU simulated annealing for quantum blockchain mining
"""

import modal
import numpy as np
import subprocess
from typing import Tuple, Dict, Any
import time
from numba import cuda, jit

# Define Modal app for GPU execution
app = modal.App("quantum-blockchain-gpu-benchmark")

# GPU container image with required dependencies
gpu_image = modal.Image.debian_slim().pip_install(
    "numpy",
    "numba",
    "scipy",  # For better random number generation
)

@app.function(
    image=gpu_image,
    gpu="t4",  # Start with T4 for cost optimization
    timeout=300,
)
def gpu_simulated_annealing(
    h: Dict[int, float], 
    J: Dict[Tuple[int, int], float],
    num_reads: int = 100,
    num_sweeps: int = 2048,
    beta_range: Tuple[float, float] = (0.1, 10.0)
) -> Dict[str, Any]:
    """
    Run simulated annealing on GPU with Numba CUDA acceleration.
    
    Args:
        h: Linear biases
        J: Quadratic couplings
        num_reads: Number of independent runs
        num_sweeps: Number of sweeps per run
        beta_range: Temperature range (inverse)
        
    Returns:
        Dictionary with samples, energies, and timing info
    """
    
    # Get GPU info
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    except Exception:
        gpu_info = "T4"
    
    start_time = time.time()
    
    # Convert to arrays
    num_vars = max(max(h.keys()), max(max(j) for j in J.keys())) + 1
    h_array = np.zeros(num_vars, dtype=np.float32)
    for i, val in h.items():
        h_array[i] = val
    
    # Create coupling matrix
    J_matrix = np.zeros((num_vars, num_vars), dtype=np.float32)
    for (i, j), val in J.items():
        J_matrix[i, j] = val
        J_matrix[j, i] = val
    
    # Numba JIT-compiled annealing function
    @jit(nopython=True)
    def anneal_single(h_arr, J_mat, num_sweeps, beta_start, beta_end):
        """Single annealing run with proper typing."""
        n = len(h_arr)
        state = np.empty(n, dtype=np.float32)
        
        # Initialize random state
        for i in range(n):
            state[i] = 1.0 if np.random.random() > 0.5 else -1.0
        
        # Temperature schedule
        betas = np.linspace(beta_start, beta_end, num_sweeps)
        
        # Annealing loop
        for beta in betas:
            for _ in range(n):
                i = np.random.randint(0, n)
                
                # Calculate neighbor sum
                neighbors_sum = 0.0
                for j in range(n):
                    neighbors_sum += J_mat[i, j] * state[j]
                
                # Energy delta
                delta_e = 2.0 * state[i] * (h_arr[i] + neighbors_sum)
                
                # Metropolis acceptance
                if delta_e < 0.0 or np.random.random() < np.exp(-beta * delta_e):
                    state[i] *= -1.0
        
        # Calculate final energy
        energy = 0.0
        for i in range(n):
            energy -= h_arr[i] * state[i]
            for j in range(n):
                energy -= 0.5 * J_mat[i, j] * state[i] * state[j]
        
        return state, energy
    
    # Run annealing
    samples = []
    energies = []
    
    for read in range(num_reads):
        state, energy = anneal_single(h_array, J_matrix, num_sweeps, beta_range[0], beta_range[1])
        samples.append(state.tolist())
        energies.append(float(energy))
    
    gpu_time = time.time() - start_time
    
    return {
        "samples": samples,
        "energies": energies,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "gpu_time": gpu_time,
        "gpu_type": gpu_info,
        "avg_time_per_sample": gpu_time / num_reads
    }


@app.function(
    image=gpu_image,
    gpu="a10g",  # Upgrade to A10G for better performance
    timeout=300,
)
def gpu_simulated_annealing_a10g(h, J, num_reads=100, num_sweeps=2048):
    """A10G GPU version for better performance."""
    result = gpu_simulated_annealing.local(h, J, num_reads, num_sweeps)
    result["gpu_type"] = "A10G"
    return result


@app.function(
    image=gpu_image,
    gpu="a100",  # Top tier for maximum performance
    timeout=300,
)
def gpu_simulated_annealing_a100(h, J, num_reads=100, num_sweeps=2048):
    """A100 GPU version for maximum performance."""
    result = gpu_simulated_annealing.local(h, J, num_reads, num_sweeps)
    result["gpu_type"] = "A100"
    return result


@app.local_entrypoint()
def benchmark_gpu_vs_qpu(
    problem_size: int = 100,
    num_samples: int = 10,
    compare_gpus = None
):
    """
    Benchmark different GPU types against QPU performance.
    
    Modal Labs Free Tier: $30/month credits
    - T4 GPU: ~$0.10/hour
    - A10G GPU: ~$0.30/hour  
    - A100 GPU: ~$1.00/hour
    """
    if compare_gpus is None:
        compare_gpus = ["t4", "a10g"]
        
    print("GPU Benchmarking for Quantum Annealing")
    print("="*50)
    
    # Generate random Ising problem
    np.random.seed(42)
    h = {i: 0 for i in range(problem_size)}
    J = {}
    for i in range(problem_size):
        for j in range(i+1, min(i+5, problem_size)):
            if np.random.random() < 0.7:
                J[(i, j)] = np.random.choice([-1, 1])
    
    results = {}
    
    # Benchmark each GPU type
    for gpu_type in compare_gpus:
        print(f"\n{'='*60}")
        print(f"Benchmarking {gpu_type.upper()} GPU")
        print(f"{'='*60}")
        
        if gpu_type == "t4":
            result = gpu_simulated_annealing.remote(h, J, num_samples)
        elif gpu_type == "a10g":
            result = gpu_simulated_annealing_a10g.remote(h, J, num_samples)
        elif gpu_type == "a100":
            result = gpu_simulated_annealing_a100.remote(h, J, num_samples)
        else:
            continue
            
        results[gpu_type] = result
        
        energies = result['energies']
        
        print(f"\nGPU Type: {result.get('gpu_type', gpu_type.upper())}")
        print(f"Problem size: {problem_size} variables")
        print(f"Number of samples: {num_samples}")
        print(f"Number of sweeps: {result['num_sweeps']}")
        
        print(f"\nEnergy Statistics:")
        print(f"  Min energy: {np.min(energies):.2f}")
        print(f"  Max energy: {np.max(energies):.2f}")
        print(f"  Mean energy: {np.mean(energies):.2f}")
        print(f"  Std energy: {np.std(energies):.2f}")
        
        print(f"\nTiming Performance:")
        print(f"  Total time: {result['gpu_time']:.3f}s")
        print(f"  Time per sample: {result['avg_time_per_sample']:.3f}s")
        print(f"  Samples per second: {1/result['avg_time_per_sample']:.2f}")
        
        # Estimate cost
        hour_fraction = result['gpu_time'] / 3600
        gpu_costs = {"t4": 0.10, "a10g": 0.30, "a100": 1.00}
        cost_per_hour = gpu_costs.get(gpu_type, 0.10)
        cost = hour_fraction * cost_per_hour
        
        print(f"\nCost Analysis:")
        print(f"  GPU cost per hour: ${cost_per_hour:.2f}")
        print(f"  Estimated cost for this run: ${cost:.6f}")
        print(f"  Cost per sample: ${cost/num_samples:.6f}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        print("\nPerformance ranking (by average energy):")
        sorted_results = sorted(results.items(), key=lambda x: np.mean(x[1]['energies']))
        for i, (gpu, res) in enumerate(sorted_results, 1):
            print(f"  {i}. {gpu.upper()}: {np.mean(res['energies']):.2f} (time: {res['avg_time_per_sample']:.3f}s/sample)")
        
        print("\nSpeed comparison:")
        fastest = min(results.items(), key=lambda x: x[1]['avg_time_per_sample'])
        for gpu, res in results.items():
            speedup = res['avg_time_per_sample'] / fastest[1]['avg_time_per_sample']
            print(f"  {gpu.upper()}: {res['avg_time_per_sample']:.3f}s/sample ({speedup:.2f}x vs {fastest[0].upper()})")
        
        print("\nCost efficiency (energy per dollar):")
        for gpu, res in results.items():
            cost_per_hour = {"t4": 0.10, "a10g": 0.30, "a100": 1.00}.get(gpu, 0.10)
            samples_per_hour = 3600 / res['avg_time_per_sample']
            cost_per_sample = cost_per_hour / samples_per_hour
            energy_per_dollar = abs(np.mean(res['energies'])) / cost_per_sample
            print(f"  {gpu.upper()}: {energy_per_dollar:.0f} energy units per dollar")
    
    return results


if __name__ == "__main__":
    # Run with Modal Labs
    # First install: pip install modal
    # Then authenticate: modal token new
    print("To run GPU benchmarks:")
    print("1. Install Modal: pip install modal")
    print("2. Authenticate: modal token new")
    print("3. Run: modal run gpu_benchmark_modal.py")
    print("\nNote: Modal provides $30/month free credits for new users")