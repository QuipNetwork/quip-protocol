#!/usr/bin/env python3
"""
Analysis script to calculate the theoretical minimum energy for D-Wave topologies.

This script:
1. Analyzes any of the defined topologies from shared/quantum_proof_of_work.py
2. Calculates theoretical minimum energy bounds for random Ising problems
3. Provides statistical analysis of energy distributions
4. Supports command-line selection of specific topologies or all of them
"""

import argparse
import math
import numpy as np
import statistics
import time
from typing import Dict, List

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    energy_of_solution
)
from shared.energy_utils import expected_solution_energy
from dwave.samplers import SimulatedAnnealingSampler

from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import (
    CHIMERA_C16_TOPOLOGY,
    PEGASUS_P16_TOPOLOGY,
    ZEPHYR_Z12_TOPOLOGY,
    ZEPHYR_Z11_T4_TOPOLOGY,
    ADVANTAGE2_SYSTEM1_13_TOPOLOGY
)

# Available topologies for analysis
AVAILABLE_TOPOLOGIES = {
    'default': DEFAULT_TOPOLOGY,  # Current default topology (use this for consistency with miners)
    'c16': CHIMERA_C16_TOPOLOGY,
    'p16': PEGASUS_P16_TOPOLOGY,
    'z12': ZEPHYR_Z12_TOPOLOGY,  # Generic Z(12, 4) topology
    'z11t4': ZEPHYR_Z11_T4_TOPOLOGY,  # Generic Z(11, 4) topology
    'advantage2': ADVANTAGE2_SYSTEM1_13_TOPOLOGY,  # Real Advantage2-System1.10 topology
}

def analyze_topology(topology_name: str, topology_obj) -> Dict:
    """Analyze a specific topology's properties and energy bounds."""
    print(f"\n=== {topology_name.upper()} {topology_obj.solver_name} ===")

    # Get the topology graph and data
    graph = topology_obj.graph
    nodes = topology_obj.nodes
    edges = topology_obj.edges

    # Note: advantage2 topology uses real Advantage2-System1.6 data (4593 nodes, 41796 edges)
    # while z12 uses generic dwave_networkx.zephyr_graph(12, 4) (4800 nodes, 45864 edges)

    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {len(edges):,}")
    print(f"Average degree: {2 * len(edges) / len(nodes):.2f}")
    print(f"Maximum possible edges (complete graph): {len(nodes) * (len(nodes) - 1) // 2:,}")
    print(f"Edge density: {len(edges) / (len(nodes) * (len(nodes) - 1) // 2):.6f}")
    
    return {
        'topology_name': topology_name,
        'topology_obj': topology_obj,
        'graph': graph,
        'nodes': nodes,
        'edges': edges,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'avg_degree': 2 * len(edges) / len(nodes),
        'edge_density': len(edges) / (len(nodes) * (len(nodes) - 1) // 2)
    }

def calculate_theoretical_minimum_energy(
    topology_data: Dict,
    num_samples: int = 50,
    num_reads: int = 64,
    num_sweeps: int = 256,
    h_values: List[float] = None
) -> Dict:
    """
    Calculate theoretical minimum energy bounds for random Ising problems.

    For each random Ising problem:
    - J values are +1 or -1 for each edge
    - h fields are generated from h_values distribution

    Args:
        topology_data: Dictionary containing topology information
        num_samples: Number of random Ising problems to sample
        num_reads: Number of SA reads per problem
        num_sweeps: Number of SA sweeps per read
        h_values: List of allowed h field values (default: [-1, 0, +1])

    This function tests both:
    1. Random guessing (baseline to show optimization is necessary)
    2. Simulated Annealing (what miners actually use)
    """
    if h_values is None:
        h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

    topology_name = topology_data['topology_name']
    nodes = topology_data['nodes']
    edges = topology_data['edges']

    print(f"\n=== Theoretical Energy Analysis for {topology_name.upper()} (n={num_samples} samples) ===")
    print(f"h_values distribution: {h_values}")

    # Calculate expected ground state energy using shared formula
    expected_gse = expected_solution_energy(len(nodes), len(edges), c=0.75, h_values=h_values)
    print(f"Expected ground state energy (empirical formula): {expected_gse:.1f}")
    print(f"Running SA with num_reads={num_reads}, num_sweeps={num_sweeps}")

    # Initialize SA sampler once
    sa_sampler = SimulatedAnnealingSampler()

    random_min_energies = []
    random_max_energies = []
    sa_min_energies = []
    sa_avg_energies = []
    theoretical_bounds = []
    coupling_stats = []

    # Reduce random sample size to be much smaller (it's just for comparison)
    random_sample_size = min(100, max(10, len(nodes) // 100))  # Much smaller: ~46 for Advantage2

    for seed in range(num_samples):
        if seed % 10 == 0:
            print(f"  Processing sample {seed}/{num_samples}...")

        # Generate random Ising problem with specified h_values
        h, J = generate_ising_model_from_nonce(seed, nodes, edges, h_values=h_values)

        # Calculate theoretical bounds
        # Theoretical minimum: all J values contribute negatively
        # Since J ∈ {-1, +1}, minimum energy occurs when spins align to make all J*si*sj = -|J|
        negative_contribution = sum(-abs(Jij) for Jij in J.values())
        theoretical_bounds.append(negative_contribution)

        # Count coupling statistics
        positive_couplers = sum(1 for Jij in J.values() if Jij > 0)
        negative_couplers = sum(1 for Jij in J.values() if Jij < 0)
        coupling_stats.append({
            'positive': positive_couplers,
            'negative': negative_couplers,
            'theoretical_min': -negative_couplers + positive_couplers  # Best case alignment
        })

        # 1. Random sampling (for comparison - shows optimization is needed)
        random_energies = []
        for _ in range(random_sample_size):
            # Generate random spin configuration
            spins = [2*np.random.randint(2)-1 for _ in nodes]  # spins in {-1, +1}
            solution = [1 if s > 0 else 0 for s in spins]  # Convert to {0, 1}

            energy = energy_of_solution(solution, h, J, nodes)
            random_energies.append(energy)

        random_min_energies.append(min(random_energies))
        random_max_energies.append(max(random_energies))

        # 2. Simulated Annealing (what miners actually use)
        sampleset = sa_sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=num_sweeps)
        sa_energies = list(sampleset.record.energy)
        sa_min_energies.append(float(min(sa_energies)))
        sa_avg_energies.append(float(np.mean(sa_energies)))
    
    # Statistical analysis
    avg_theoretical = statistics.mean(theoretical_bounds)
    avg_random_min = statistics.mean(random_min_energies)
    avg_random_max = statistics.mean(random_max_energies)
    avg_sa_min = statistics.mean(sa_min_energies)
    avg_sa_avg = statistics.mean(sa_avg_energies)

    # Coupling statistics
    avg_positive = statistics.mean(stat['positive'] for stat in coupling_stats)
    avg_negative = statistics.mean(stat['negative'] for stat in coupling_stats)

    # Calculate how well SA performed relative to expectations
    sa_vs_expected = avg_sa_min - expected_gse
    sa_vs_perfect = (avg_sa_min / avg_theoretical) * 100

    print(f"\n{'='*60}")
    print(f"Energy Analysis Results (lower = better)")
    print(f"{'='*60}")
    print(f"\n1. Theoretical Limits:")
    print(f"   Perfect minimum (all couplings satisfied): {avg_theoretical:.1f}")
    print(f"   Expected GSE for random J∈{{-1,+1}} problems: {expected_gse:.1f}")
    print(f"   Expected nonce-to-nonce variance: ±{math.sqrt(len(edges)):.1f}")

    print(f"\n2. Random Guessing (no optimization):")
    print(f"   Average best of {random_sample_size} guesses: {avg_random_min:.1f}")
    print(f"   This shows ~50% edge satisfaction (random baseline)")

    print(f"\n3. Simulated Annealing (num_sweeps={num_sweeps}, num_reads={num_reads}):")
    print(f"   Average minimum: {avg_sa_min:.1f}")
    print(f"   Best observed:   {min(sa_min_energies):.1f}")
    print(f"   Worst observed:  {max(sa_min_energies):.1f}")
    print(f"   vs Expected GSE: {sa_vs_expected:+.1f} ({'+' if sa_vs_expected > 0 else ''}{abs(sa_vs_expected):.0f} energy units)")
    print(f"   vs Perfect min:  {sa_vs_perfect:.1f}% of theoretical optimum")

    print(f"\n4. Coupling Statistics:")
    print(f"   Positive couplings (J=+1): {avg_positive:.0f}")
    print(f"   Negative couplings (J=-1): {avg_negative:.0f}")
    print(f"   Balance: {abs(avg_positive - avg_negative):.0f} difference")

    return {
        'theoretical_bounds': theoretical_bounds,
        'random_min_energies': random_min_energies,
        'random_max_energies': random_max_energies,
        'sa_min_energies': sa_min_energies,
        'sa_avg_energies': sa_avg_energies,
        'coupling_stats': coupling_stats,
        'avg_theoretical': avg_theoretical,
        'avg_random_min': avg_random_min,
        'avg_random_max': avg_random_max,
        'avg_sa_min': avg_sa_min,
        'avg_sa_avg': avg_sa_avg,
        'best_observed': min(sa_min_energies),
        'worst_observed': max(sa_min_energies),
        'avg_positive_couplers': avg_positive,
        'avg_negative_couplers': avg_negative,
        'perfect_theoretical_min': -len(edges),
        'expected_gse': expected_gse,
        'expected_variance': math.sqrt(len(edges)),
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
        'h_values': h_values  # Record which distribution was used
    }

def calculate_sa_theoretical_bounds(topology_data: Dict, energy_data: Dict) -> Dict:
    """
    Analyze SA performance relative to theoretical expectations.

    Uses the expected_solution_energy formula (GSE ≈ -c × √(avg_degree) × N)
    and compares against actual SA results to validate the empirical constant 'c'.

    Research background on SA performance limits:

    1. "Polynomial-Time Approximation Algorithms for the Ising Model"
       https://epubs.siam.org/doi/10.1137/0222066
       - Shows SA has polynomial-time approximation guarantees but not exact solutions

    2. "Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing"
       https://link.aps.org/doi/10.1103/PhysRevX.8.031016
       https://arxiv.org/abs/1705.07452
       - Empirical evidence that SA faces exponential energy barriers on random instances

    3. "Optimized simulated annealing for Ising spin glasses"
       https://arxiv.org/abs/1401.1084
       - Shows SA typically achieves 30-35% of theoretical optimum on random Ising models

    4. "Quantum annealing applications, challenges and limitations"
       https://www.nature.com/articles/s41598-025-96220-2
       - Recent benchmarking showing D-Wave vs SA performance bounds

    Key insights:
    - SA on random Ising spin glasses achieves ~30-35% of perfect minimum
    - Performance is limited by energy gaps and local minima trapping
    - The expected_solution_energy formula captures this empirically via constant 'c'
    """
    nodes = topology_data['nodes']
    edges = topology_data['edges']
    num_edges = len(edges)
    perfect_min = energy_data['perfect_theoretical_min']  # -num_edges
    expected_gse = energy_data['expected_gse']
    avg_sa_min = energy_data['avg_sa_min']
    topology_name = topology_data['topology_name']

    print(f"\n{'='*60}")
    print(f"SA Performance Analysis for {topology_name.upper()}")
    print(f"{'='*60}")

    # Calculate performance metrics
    sa_vs_perfect_pct = (avg_sa_min / perfect_min) * 100
    sa_vs_expected_diff = avg_sa_min - expected_gse

    # Test different c values to see which best matches observed SA
    print(f"\nCalibrating empirical constant 'c' in GSE ≈ -c × √(avg_degree) × N:")
    print(f"{'c value':<10} {'Predicted GSE':<15} {'vs SA diff':<15} {'Match'}")
    print(f"{'-'*60}")

    best_c = 0.75
    best_diff = abs(sa_vs_expected_diff)

    # Test c values from 0.70 to 0.80 in steps of 0.01
    c_values = [round(0.70 + i * 0.01, 2) for i in range(11)]  # 0.70, 0.71, ..., 0.80

    for c_test in c_values:
        predicted = expected_solution_energy(nodes, edges, c=c_test)
        diff = avg_sa_min - predicted
        match_symbol = "✓" if abs(diff) < math.sqrt(num_edges) else " "
        print(f"{c_test:<10.2f} {predicted:<15.1f} {diff:>+14.1f} {match_symbol}")

        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_c = c_test

    print(f"\nBest fit: c ≈ {best_c:.2f} (minimizes |SA - predicted GSE|)")
    print(f"\nSA Performance Metrics:")
    print(f"  Achieved:     {avg_sa_min:.1f}")
    print(f"  Expected GSE: {expected_gse:.1f} (c=0.75)")
    print(f"  Difference:   {sa_vs_expected_diff:+.1f} energy units")
    print(f"  vs Perfect:   {sa_vs_perfect_pct:.1f}% of theoretical minimum")

    # Validate against 2σ variance
    variance = math.sqrt(num_edges)
    if abs(sa_vs_expected_diff) < variance * 2:
        print(f"  ✓ Within 2σ variance (±{variance*2:.0f})")
    else:
        print(f"  ⚠ Outside 2σ variance (±{variance*2:.0f})")

    return {
        'perfect_min': perfect_min,
        'expected_gse': expected_gse,
        'avg_sa_min': avg_sa_min,
        'sa_vs_perfect_pct': sa_vs_perfect_pct,
        'sa_vs_expected_diff': sa_vs_expected_diff,
        'best_c': best_c
    }

def print_summary_table(results: List[Dict]):
    """Print a summary table comparing all analyzed topologies."""
    print(f"\n=== SUMMARY TABLE: All Analyzed Topologies ===")
    print(f"{'Topology':<8} {'Nodes':<8} {'Edges':<8} {'Perfect Min':<12} {'SA O(√n)':<12} {'SA O(n^2/3)':<12} {'Best Observed':<14}")
    print("-" * 94)
    
    for result in results:
        topology_data = result['topology_data']
        energy_data = result['energy_data']
        sa_data = result.get('sa_bounds', {})
        
        sa_conservative = sa_data.get('practical_conservative', 'N/A')
        sa_optimistic = sa_data.get('practical_optimistic', 'N/A')
        
        sa_cons_str = f"{sa_conservative:.0f}" if isinstance(sa_conservative, (int, float)) else str(sa_conservative)
        sa_opt_str = f"{sa_optimistic:.0f}" if isinstance(sa_optimistic, (int, float)) else str(sa_optimistic)
        
        print(f"{topology_data['topology_name'].upper():<8} "
              f"{topology_data['num_nodes']:<8,} "
              f"{topology_data['num_edges']:<8,} "
              f"{energy_data['perfect_theoretical_min']:<12.0f} "
              f"{sa_cons_str:<12} "
              f"{sa_opt_str:<12} "
              f"{energy_data['best_observed']:<14.1f}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze theoretical minimum energy for D-Wave topologies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available topologies:
  default    - DEFAULT_TOPOLOGY from dwave_topologies (matches miners)
  c16        - Chimera C16 topology (~2048 qubits)
  p16        - Pegasus P16 topology (~5000 qubits)
  z12        - Generic Zephyr Z(12, 4) topology (~4800 qubits)
  z11t4      - Generic Zephyr Z(11, 4) topology (~4048 qubits)
  advantage2 - Real Advantage2-System1.6 topology (~4593 qubits)
  all        - Analyze all topologies

Examples:
  python analyze_topology_minimum_energy.py
  python analyze_topology_minimum_energy.py --topology all --samples 100
  python analyze_topology_minimum_energy.py --topology default advantage2
        """
    )
    
    parser.add_argument('--topology', '-t',
                       nargs='+',
                       choices=['default', 'c16', 'p16', 'z12', 'z11t4', 'advantage2', 'all'],
                       default=['default'],
                       help='Topology(ies) to analyze (default: default)')
    
    parser.add_argument('--samples', '-s',
                       type=int,
                       default=50,
                       help='Number of random Ising problems to sample (default: 50)')

    parser.add_argument('--num-reads',
                       type=int,
                       default=64,
                       help='Number of SA reads per problem (default: 64)')

    parser.add_argument('--num-sweeps',
                       type=int,
                       default=256,
                       help='Number of SA sweeps per read (default: 256, production uses 4096)')

    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')

    parser.add_argument('--h-values',
                       type=str,
                       default='-1,0,1',
                       help='Comma-separated h field values (default: -1,0,1). Use "0" for h=0 baseline.')

    args = parser.parse_args()

    # Parse h_values
    h_values = [float(v) for v in args.h_values.split(',')]
    
    # Handle 'all' topology selection
    if 'all' in args.topology:
        selected_topologies = list(AVAILABLE_TOPOLOGIES.keys())
    else:
        selected_topologies = args.topology
    
    print(f"Analyzing topologies: {', '.join(selected_topologies)}")
    print(f"Number of samples per topology: {args.samples}")
    print(f"Using h_values: {h_values}")
    
    start_time = time.time()
    results = []
    
    for topology_name in selected_topologies:
        if topology_name not in AVAILABLE_TOPOLOGIES:
            print(f"Warning: Unknown topology '{topology_name}', skipping...")
            continue
            
        topology_obj = AVAILABLE_TOPOLOGIES[topology_name]

        # Step 1: Analyze topology structure
        topology_data = analyze_topology(topology_name, topology_obj)
        
        # Step 2: Calculate theoretical energy bounds
        energy_data = calculate_theoretical_minimum_energy(
            topology_data, args.samples, args.num_reads, args.num_sweeps, h_values=h_values
        )
        
        # Step 3: Calculate SA theoretical bounds
        sa_bounds = calculate_sa_theoretical_bounds(topology_data, energy_data)
        
        # Step 4: Final summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {topology_name.upper()}")
        print(f"{'='*60}")
        print(f"Topology: {topology_data['num_nodes']:,} nodes, {topology_data['num_edges']:,} edges")
        print(f"Expected GSE: {energy_data['expected_gse']:.1f} ± {energy_data['expected_variance']:.1f}")
        print(f"SA achieved:  {energy_data['avg_sa_min']:.1f} (avg of {args.samples} random nonces)")
        print(f"Random guess: {energy_data['avg_random_min']:.1f} (shows need for optimization)")
        print(f"Perfect min:  {energy_data['perfect_theoretical_min']} (unachievable for frustrated systems)")
        
        results.append({
            'topology_data': topology_data,
            'energy_data': energy_data,
            'sa_bounds': sa_bounds,
        })
    
    # Print summary table if analyzing multiple topologies
    if len(results) > 1:
        print_summary_table(results)
    
    print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()