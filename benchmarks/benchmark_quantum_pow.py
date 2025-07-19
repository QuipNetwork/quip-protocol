import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler

# Load environment variables
load_dotenv()

class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure

    """
    def __init__(
        self, qpu=None
    ):
        if qpu is None:
            qpu = DWaveSampler()
        
        substitute_sampler = SimulatedAnnealingSampler()
        super().__init__(
            nodelist=qpu.nodelist,
            edgelist=qpu.edgelist,
            properties=qpu.properties,
            substitute_sampler=substitute_sampler
        )
        self.sampler_type = "mock"
        self.parameters.update(substitute_sampler.parameters)  # Do not warn when SA parameters are seen.
        self.mocked_parameters.add('num_sweeps')  # Do not warn when this SA parameter is seen.
        
@dataclass
class BenchmarkResult:
    sampler_type: str
    seed: int
    num_reads: int
    num_sweeps: Optional[int]
    energies: List[float]
    min_energy: float
    mean_energy: float
    std_energy: float
    elapsed_time: float
    problem_size: int


class QuantumPowBenchmark:
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        
    def generate_ising_problem(self, seed: int, sampler) -> Tuple[Dict, Dict]:
        """
        Generate a random Ising problem based on seed.
        
        Args:
            seed: Random seed for reproducibility
            sampler: DWave sampler instance
            
        Returns:
            h: Linear terms
            J: Quadratic terms
        """
        np.random.seed(seed)
        
        # QPU sampler
        h = {i: 0 for i in sampler.nodelist}
        J = {edge: 2*np.random.randint(2)-1 for edge in sampler.edgelist}
        problem_size = len(sampler.nodelist)
            
        return h, J, problem_size
    
    def benchmark_sampler(self, sampler, sampler_name: str, seeds: List[int], 
                         num_reads: int = 100, num_sweeps: Optional[int] = None) -> None:
        """
        Benchmark a sampler across multiple seeds.
        
        Args:
            sampler: Sampler instance
            sampler_name: Name for the sampler
            seeds: List of random seeds to test
            num_reads: Number of samples per problem
            num_sweeps: Number of sweeps for SA (ignored for QPU)
        """
        print(f"\nBenchmarking {sampler_name}...")
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')
            
            # Generate problem
            h, J, problem_size = self.generate_ising_problem(seed, sampler)
            
            # Time the sampling
            start_time = time.time()
            
            if sampler_name == "QPU":
                sampleset = sampler.sample_ising(h, J, num_reads=100, answer_mode='raw')
            else:
                sampleset = sampler.sample_ising(h, J, num_reads=100, num_sweeps=num_sweeps)
            
            elapsed_time = time.time() - start_time
            
            # Extract energies
            energies = list(sampleset.record.energy)
            
            # Store results
            result = BenchmarkResult(
                sampler_type=sampler_name,
                seed=seed,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                energies=energies,
                min_energy=float(np.min(energies)),
                mean_energy=float(np.mean(energies)),
                std_energy=float(np.std(energies)),
                elapsed_time=elapsed_time,
                problem_size=problem_size
            )
            
            self.results.append(result)
            print(f"Done. Min energy: {result.min_energy:.2f}, Time: {elapsed_time:.2f}s")
    
    def compare_samplers(self, seeds: List[int], num_reads: int = 100, 
                        sweep_values: List[int] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]) -> None:
        """
        Compare QPU vs SA performance.
        
        Args:
            seeds: List of random seeds
            num_reads: Number of samples
            sweep_values: Different sweep counts to test for SA
        """
        # Benchmark QPU (if available)
        try:
            qpu_sampler = DWaveSampler()
            print(f"Connected to QPU: {qpu_sampler.properties['chip_id']}")
            print(f"QPU working qubits: {len(qpu_sampler.nodelist)}")
            self.benchmark_sampler(qpu_sampler, "QPU", seeds, num_reads)
        except Exception as e:
            print(f"QPU not available: {e}")
            print("Skipping QPU benchmarks")
        
        # Benchmark SA with different sweep values
        sa_sampler = SimulatedAnnealingStructuredSampler()
        for num_sweeps in sweep_values:
            self.benchmark_sampler(
                sa_sampler, 
                f"SA_{num_sweeps:04d}", 
                seeds, 
                num_reads, 
                num_sweeps
            )
    
    def plot_energy_distributions(self, output_file: str = "benchmarks/energy_distributions.png") -> None:
        """Plot energy distributions for different samplers."""
        if not self.results:
            print("No results to plot")
            return
            
        # Define consistent color mapping
        color_map = {'QPU': '#4285F4'}
        # Add orange for all SA variants
        for index, key in enumerate(['SA', 'SA_0008', 'SA_0016', 'SA_0032', 'SA_0064', 'SA_0128', 'SA_0256', 'SA_0512', 'SA_1024', 'SA_2048', 'SA_4096', 'SA_8192']):
            color_map[key] = (1.0, 0.45 + 0.05*index, 0.25 + 0.05*index)
        
        # Convert to DataFrame
        data = []
        for result in self.results:
            for energy in result.energies:
                data.append({
                    'Sampler': result.sampler_type,
                    'Seed': result.seed,
                    'Energy': energy
                })
        
        df = pd.DataFrame(data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy Distributions: QPU vs Simulated Annealing', fontsize=16)
        
        # Plot 1: Box plot by sampler
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='Sampler', y='Energy', ax=ax1, palette=color_map)
        ax1.set_title('Energy Distribution by Sampler')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Violin plot by sampler
        ax2 = axes[0, 1]
        sns.violinplot(data=df, x='Sampler', y='Energy', ax=ax2, palette=color_map)
        ax2.set_title('Energy Density by Sampler')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Energy vs Seed for each sampler
        ax3 = axes[1, 0]
        for sampler in sorted(df['Sampler'].unique()):
            sampler_df = df[df['Sampler'] == sampler]
            mean_energies = sampler_df.groupby('Seed')['Energy'].mean()
            color = color_map.get(sampler, 'gray')
            ax3.plot(mean_energies.index, mean_energies.values, marker='o', label=sampler, color=color)
        ax3.set_xlabel('Seed')
        ax3.set_ylabel('Mean Energy')
        ax3.set_title('Mean Energy by Seed')
        ax3.legend()
        
        # Plot 4: Histogram comparison
        ax4 = axes[1, 1]
        for sampler in sorted(df['Sampler'].unique()):
            sampler_energies = df[df['Sampler'] == sampler]['Energy']
            color = color_map.get(sampler, 'gray')
            ax4.hist(sampler_energies, alpha=0.5, label=sampler, bins=30, color=color)
        ax4.set_xlabel('Energy')
        ax4.set_ylabel('Count')
        ax4.set_title('Energy Histogram Comparison')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved energy distribution plot to {output_file}")
    
    def plot_performance_metrics(self, output_file: str = "benchmarks/performance_metrics.png") -> None:
        """Plot performance metrics comparison."""
        if not self.results:
            print("No results to plot")
            return
            
        # Define consistent color mapping
        color_map = {'QPU': '#4285F4'}
        # Add orange for all SA variants
        for index, key in enumerate(['SA', 'SA_0008', 'SA_0016', 'SA_0032', 'SA_0064', 'SA_0128', 'SA_0256', 'SA_0512', 'SA_1024', 'SA_2048', 'SA_4096', 'SA_8192']):
            color_map[key] = (1.0, 0.45 + 0.05*index, 0.25 + 0.05*index)
        
        # Aggregate results by sampler
        sampler_stats = {}
        for result in self.results:
            if result.sampler_type not in sampler_stats:
                sampler_stats[result.sampler_type] = {
                    'min_energies': [],
                    'mean_energies': [],
                    'times': []
                }
            sampler_stats[result.sampler_type]['min_energies'].append(result.min_energy)
            sampler_stats[result.sampler_type]['mean_energies'].append(result.mean_energy)
            sampler_stats[result.sampler_type]['times'].append(result.elapsed_time)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        
        # Plot 1: Average minimum energy
        ax1 = axes[0, 0]
        samplers = sorted(list(sampler_stats.keys()))
        avg_min_energies = [np.mean(sampler_stats[s]['min_energies']) for s in samplers]
        std_min_energies = [np.std(sampler_stats[s]['min_energies']) for s in samplers]
        colors = [color_map.get(s, 'gray') for s in samplers]
        ax1.bar(samplers, avg_min_energies, yerr=std_min_energies, capsize=5, color=colors)
        ax1.set_ylabel('Average Minimum Energy')
        ax1.set_title('Solution Quality (Lower is Better)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Average runtime
        ax2 = axes[0, 1]
        avg_times = [np.mean(sampler_stats[s]['times']) for s in samplers]
        std_times = [np.std(sampler_stats[s]['times']) for s in samplers]
        ax2.bar(samplers, avg_times, yerr=std_times, capsize=5, color=colors)
        ax2.set_ylabel('Average Runtime (seconds)')
        ax2.set_title('Computation Time')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Quality vs Time tradeoff
        ax3 = axes[1, 0]
        for i, sampler in enumerate(samplers):
            ax3.scatter(avg_times[i], avg_min_energies[i], s=100, color=colors[i], label=sampler)
            ax3.annotate(sampler, (avg_times[i], avg_min_energies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax3.set_xlabel('Average Runtime (seconds)')
        ax3.set_ylabel('Average Minimum Energy')
        ax3.set_title('Quality vs Time Tradeoff')
        
        # Plot 4: Success rate (finding energy below threshold)
        ax4 = axes[1, 1]
        threshold = -15600  # Example threshold
        success_rates = []
        for sampler in samplers:
            min_energies = sampler_stats[sampler]['min_energies']
            success_rate = sum(e < threshold for e in min_energies) / len(min_energies) * 100
            success_rates.append(success_rate)
        ax4.bar(samplers, success_rates, color=colors)
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title(f'Success Rate (Energy < {threshold})')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved performance metrics plot to {output_file}")
    
    def save_results(self, filename: str = "benchmarks/benchmark_results.json") -> None:
        """Save benchmark results to JSON file."""
        results_dict = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert numpy types to Python types
            result_dict['energies'] = [float(e) for e in result_dict['energies']]
            results_dict.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Saved results to {filename}")
    
    def generate_report(self) -> None:
        """Generate a summary report of the benchmarks."""
        print("\n" + "="*60)
        print("QUANTUM PROOF-OF-WORK BENCHMARK REPORT")
        print("="*60)
        
        # Group results by sampler
        sampler_results = {}
        for result in self.results:
            if result.sampler_type not in sampler_results:
                sampler_results[result.sampler_type] = []
            sampler_results[result.sampler_type].append(result)
        
        # Print statistics for each sampler
        for sampler, results in sampler_results.items():
            print(f"\n{sampler}:")
            print("-" * 40)
            
            min_energies = [r.min_energy for r in results]
            mean_energies = [r.mean_energy for r in results]
            times = [r.elapsed_time for r in results]
            
            print(f"  Number of runs: {len(results)}")
            print(f"  Problem size: {results[0].problem_size} variables")
            print(f"  Samples per run: {results[0].num_reads}")
            if results[0].num_sweeps:
                print(f"  Sweeps per sample: {results[0].num_sweeps}")
            print(f"\n  Energy Statistics:")
            print(f"    Best energy found: {min(min_energies):.2f}")
            print(f"    Average min energy: {np.mean(min_energies):.2f} ± {np.std(min_energies):.2f}")
            print(f"    Average mean energy: {np.mean(mean_energies):.2f} ± {np.std(mean_energies):.2f}")
            print(f"\n  Time Statistics:")
            print(f"    Average runtime: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
            print(f"    Total runtime: {sum(times):.2f}s")


def main():
    """Run the benchmark suite."""
    print("Quantum Proof-of-Work Benchmark Suite")
    print("=====================================")
    
    # Initialize benchmark
    benchmark = QuantumPowBenchmark()
    
    # Set up test parameters
    seeds = list(range(100))  # Test 10 different problems
    num_reads = 100
    sweep_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Run comparisons
    benchmark.compare_samplers(seeds, num_reads, sweep_values)
    
    # Generate visualizations
    benchmark.plot_energy_distributions()
    benchmark.plot_performance_metrics()
    
    # Save results
    benchmark.save_results()
    
    # Generate report
    benchmark.generate_report()


if __name__ == "__main__":
    main()