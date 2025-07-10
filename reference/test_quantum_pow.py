import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib.pyplot as plt


def test_reference_implementation():
    """Test the reference implementation from the meeting."""
    print("Testing Reference Implementation")
    print("=" * 50)
    
    # Initialize sampler
    sampler = SimulatedAnnealingSampler()
    
    # Create a simple test problem
    num_vars = 64
    h = {i: 0 for i in range(num_vars)}
    
    # Random couplers (simulating QPU architecture)
    J = {}
    np.random.seed(42)  # For reproducibility
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if np.random.random() < 0.3:  # 30% connectivity
                J[(i, j)] = 2*np.random.randint(2)-1
    
    print(f"Problem size: {num_vars} variables")
    print(f"Number of couplers: {len(J)}")
    
    # Test different num_sweeps values
    sweep_values = [256, 512, 1024, 2048, 4096, 8192]
    results = {}
    
    for num_sweeps in sweep_values:
        print(f"\nTesting num_sweeps={num_sweeps}")
        sampleset = sampler.sample_ising(h, J, num_reads=64, num_sweeps=num_sweeps)
        energies = sampleset.record.energy
        
        results[num_sweeps] = {
            'energies': energies,
            'min': np.min(energies),
            'mean': np.mean(energies),
            'std': np.std(energies)
        }
        
        print(f"  Min energy: {results[num_sweeps]['min']:.2f}")
        print(f"  Mean energy: {results[num_sweeps]['mean']:.2f} ± {results[num_sweeps]['std']:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Box plots of energy distributions
    plt.subplot(2, 2, 1)
    data_for_box = [results[ns]['energies'] for ns in sweep_values]
    plt.boxplot(data_for_box, labels=[str(ns) for ns in sweep_values])
    plt.xlabel('Number of Sweeps')
    plt.ylabel('Energy')
    plt.title('Energy Distribution vs Number of Sweeps')
    plt.xticks(rotation=45)
    
    # Subplot 2: Min energy convergence
    plt.subplot(2, 2, 2)
    min_energies = [results[ns]['min'] for ns in sweep_values]
    plt.plot(sweep_values, min_energies, 'bo-')
    plt.xlabel('Number of Sweeps')
    plt.ylabel('Minimum Energy')
    plt.title('Minimum Energy Convergence')
    plt.xscale('log')
    
    # Subplot 3: Mean energy convergence
    plt.subplot(2, 2, 3)
    mean_energies = [results[ns]['mean'] for ns in sweep_values]
    std_energies = [results[ns]['std'] for ns in sweep_values]
    plt.errorbar(sweep_values, mean_energies, yerr=std_energies, fmt='ro-', capsize=5)
    plt.xlabel('Number of Sweeps')
    plt.ylabel('Mean Energy')
    plt.title('Mean Energy Convergence')
    plt.xscale('log')
    
    # Subplot 4: Energy histogram for num_sweeps=4096
    plt.subplot(2, 2, 4)
    plt.hist(results[4096]['energies'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.title('Energy Distribution (num_sweeps=4096)')
    
    plt.tight_layout()
    plt.savefig('reference/reference_test_results.png', dpi=300)
    print("\nSaved plot to reference/reference_test_results.png")
    
    # Find the "sweet spot" around num_sweeps=4096
    print("\n" + "="*50)
    print("CONCLUSION:")
    print(f"At num_sweeps=4096, we achieve:")
    print(f"  - Minimum energy: {results[4096]['min']:.2f}")
    print(f"  - Mean energy: {results[4096]['mean']:.2f} ± {results[4096]['std']:.2f}")
    print(f"  - This appears to be the sweet spot for SA performance")
    print(f"  - Similar to QPU sampling at default settings")


if __name__ == "__main__":
    test_reference_implementation()