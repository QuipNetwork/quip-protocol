import os
import time
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")


def test_reference_implementation():
    FAST = os.getenv("QUIP_TEST_FAST") == "1"
    REF_TIMEOUT = float(os.getenv("QUIP_TEST_REF_TIMEOUT", "5.0"))
    start = time.time()

    """Test the reference implementation from the meeting."""
    print("Testing Reference Implementation")
    print("=" * 50)

    # Initialize sampler
    sampler = SimulatedAnnealingSampler()

    # Create a simple test problem (smaller in FAST mode)
    num_vars = 64 if not FAST else 32
    h = {i: 0 for i in range(num_vars)}

    # Random couplers (simulating QPU architecture)
    J = {}
    np.random.seed(42)  # For reproducibility
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if np.random.random() < (0.3 if not FAST else 0.2):  # lower connectivity in FAST
                J[(i, j)] = 2*np.random.randint(2)-1

    print(f"Problem size: {num_vars} variables")
    print(f"Number of couplers: {len(J)}")

    # Test different num_sweeps values (reduced for FAST)
    sweep_values = [256, 512, 1024, 2048] if not FAST else [64, 128, 256]
    num_reads = 64 if not FAST else 16
    results = {}

    for num_sweeps in sweep_values:
        if time.time() - start > REF_TIMEOUT:
            print("Timeout reached, stopping sweeps early")
            break
        print(f"\nTesting num_sweeps={num_sweeps}")
        sampleset = sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=num_sweeps)
        energies = sampleset.record.energy

        results[num_sweeps] = {
            'energies': energies,
            'min': np.min(energies),
            'mean': np.mean(energies),
            'std': np.std(energies)
        }

        print(f"  Min energy: {results[num_sweeps]['min']:.2f}")
        print(f"  Mean energy: {results[num_sweeps]['mean']:.2f} ± {results[num_sweeps]['std']:.2f}")

    # Plot results (skip plotting in FAST mode)
    if not FAST:
        plt.figure(figsize=(12, 8))
        # Subplot 1: Box plots of energy distributions
        plt.subplot(2, 2, 1)
        data_for_box = [results[ns]['energies'] for ns in sweep_values if ns in results]
        plt.boxplot(data_for_box, labels=[str(ns) for ns in sweep_values if ns in results])
        plt.xlabel('Number of Sweeps')
        plt.ylabel('Energy')
        plt.title('Energy Distribution vs Number of Sweeps')
        plt.xticks(rotation=45)
        # Subplot 2: Min energy convergence
        plt.subplot(2, 2, 2)
        min_energies = [results[ns]['min'] for ns in sweep_values if ns in results]
        if min_energies:
            xs = [ns for ns in sweep_values if ns in results]
            plt.plot(xs, min_energies, 'bo-')
        plt.xlabel('Number of Sweeps')
        plt.ylabel('Minimum Energy')
        plt.title('Minimum Energy Convergence')
        plt.xscale('log')
        # Subplot 3: Mean energy convergence
        plt.subplot(2, 2, 3)
        mean_energies = [results[ns]['mean'] for ns in sweep_values if ns in results]
        std_energies = [results[ns]['std'] for ns in sweep_values if ns in results]
        if mean_energies:
            xs = [ns for ns in sweep_values if ns in results]
            plt.errorbar(xs, mean_energies, yerr=std_energies, fmt='ro-', capsize=5)
        plt.xlabel('Number of Sweeps')
        plt.ylabel('Mean Energy')
        plt.title('Mean Energy Convergence')
        plt.xscale('log')
        # Subplot 4: Energy histogram for a chosen sweep value present
        plt.subplot(2, 2, 4)
        chosen = [ns for ns in [1024, 2048, 4096] if ns in results]
        if chosen:
            key = chosen[-1]
            plt.hist(results[key]['energies'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Energy')
            plt.ylabel('Count')
            plt.title(f'Energy Distribution (num_sweeps={key})')
        plt.tight_layout()
        plt.savefig('reference/reference_test_results.png', dpi=300)
        print("\nSaved plot to reference/reference_test_results.png")

    print("\nDone.")


if __name__ == "__main__":
    test_reference_implementation()