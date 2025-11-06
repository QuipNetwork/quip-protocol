#!/usr/bin/env python3
"""
Topology size and performance analysis tool.

Analyzes different Zephyr graph configurations to help select the optimal
generic topology for the QUIP protocol. Tests:
1. Graph structure (nodes, edges, connectivity)
2. Expected ground state energy (GSE)
3. Simulated Annealing performance
4. QPU embedding feasibility (if D-Wave credentials available)

Usage:
    python tools/analyze_topology_sizes.py
    python tools/analyze_topology_sizes.py --configs 11,4 12,3
    python tools/analyze_topology_sizes.py --samples 100 --num-sweeps 4096
"""

import argparse
import math
import time
from typing import List, Tuple, Dict, Any, Optional
import dwave_networkx as dnx
import numpy as np
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler

from shared.quantum_proof_of_work import generate_ising_model_from_nonce
from shared.energy_utils import expected_solution_energy


# Advantage2-System1.6 capacity (target QPU)
ADVANTAGE2_NODES = 4593
ADVANTAGE2_EDGES = 41796


def parse_timeout(timeout_str: str) -> int:
    """
    Parse timeout string to seconds.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 30
        "5m" -> 300
        "2h" -> 7200
        "1d" -> 86400
        "1w" -> 604800
    """
    timeout_str = timeout_str.strip().lower()

    if timeout_str.endswith('s'):
        return int(timeout_str[:-1])
    elif timeout_str.endswith('m'):
        return int(timeout_str[:-1]) * 60
    elif timeout_str.endswith('h'):
        return int(timeout_str[:-1]) * 3600
    elif timeout_str.endswith('d'):
        return int(timeout_str[:-1]) * 86400
    elif timeout_str.endswith('w'):
        return int(timeout_str[:-1]) * 604800
    else:
        # Try parsing as raw seconds
        return int(timeout_str)


def analyze_zephyr_config(m: int, t: int) -> Dict[str, Any]:
    """Analyze a Zephyr(m, t) graph configuration."""
    graph = dnx.zephyr_graph(m, t)
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    num_nodes = len(nodes)
    num_edges = len(edges)
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

    # Check if fits on Advantage2-System1.6
    fits = num_nodes <= ADVANTAGE2_NODES and num_edges <= ADVANTAGE2_EDGES
    node_util = 100 * num_nodes / ADVANTAGE2_NODES
    edge_util = 100 * num_edges / ADVANTAGE2_EDGES

    return {
        'm': m,
        't': t,
        'config': f'Z({m},{t})',
        'graph': graph,
        'nodes': nodes,
        'edges': edges,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'fits_advantage2': fits,
        'node_utilization_pct': node_util,
        'edge_utilization_pct': edge_util,
    }


def test_sa_performance(
    config: Dict[str, Any],
    num_samples: int = 50,
    num_reads: int = 64,
    num_sweeps: int = 256,
) -> Dict[str, Any]:
    """Test Simulated Annealing performance on this topology."""
    nodes = config['nodes']
    edges = config['edges']
    num_nodes = config['num_nodes']
    num_edges = config['num_edges']

    # Calculate expected GSE (need counts, not lists)
    expected_gse = expected_solution_energy(num_nodes=num_nodes, num_edges=num_edges, c=0.75)
    expected_variance = math.sqrt(num_edges)

    # Initialize SA sampler
    sa_sampler = SimulatedAnnealingSampler()

    sa_min_energies = []
    sa_times = []

    print(f"  Testing SA performance ({num_samples} samples, {num_reads} reads, {num_sweeps} sweeps)...")

    for seed in range(num_samples):
        if seed % 10 == 0 and seed > 0:
            print(f"    Sample {seed}/{num_samples}...")

        # Generate random Ising problem
        h, J = generate_ising_model_from_nonce(seed, nodes, edges)

        # Time SA solving
        start = time.perf_counter()
        sampleset = sa_sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=num_sweeps)
        elapsed = time.perf_counter() - start

        sa_min_energies.append(float(min(sampleset.record.energy)))
        sa_times.append(elapsed)

    avg_sa_min = np.mean(sa_min_energies)
    best_sa = min(sa_min_energies)
    worst_sa = max(sa_min_energies)
    avg_time = np.mean(sa_times)

    # Performance metrics
    sa_vs_expected = avg_sa_min - expected_gse
    perfect_min = -len(edges)
    sa_vs_perfect_pct = (avg_sa_min / perfect_min) * 100

    return {
        'expected_gse': expected_gse,
        'expected_variance': expected_variance,
        'avg_sa_min': avg_sa_min,
        'best_sa': best_sa,
        'worst_sa': worst_sa,
        'sa_vs_expected': sa_vs_expected,
        'sa_vs_perfect_pct': sa_vs_perfect_pct,
        'avg_solve_time_s': avg_time,
        'num_samples': num_samples,
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
    }


def test_embedding_feasibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test if this topology can be embedded on real Advantage2-System1.6 QPU.

    Measures actual embedding time using minorminer on the physical QPU topology.
    This tells us how long it would take to map the generic Zephyr graph onto
    the real hardware with its defect pattern.
    """
    result = {
        'embedding_tested': False,
        'embedding_found': None,
        'embedding_time_s': None,
        'num_omitted_vars': None,
        'num_omitted_edges': None,
        'qpu_chip_id': None,
        'qpu_num_qubits': None,
    }

    try:
        import minorminer
        from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_7_TOPOLOGY

        # Get target QPU topology from saved JSON (no live connection needed!)
        try:
            target_topology = ADVANTAGE2_SYSTEM1_7_TOPOLOGY
            target_graph = target_topology.graph

            qpu_chip_id = target_topology.solver_name
            qpu_num_qubits = target_topology.num_nodes

            source_graph = config['graph']
            source_nodes = config['num_nodes']
            source_edges = config['num_edges']

            print(f"  Testing embedding on Advantage2-System1.6 (from saved topology):")
            print(f"    Target: {qpu_chip_id} ({qpu_num_qubits:,} qubits, {target_topology.num_edges:,} couplers)")
            print(f"    Source: {config['config']} ({source_nodes:,} nodes, {source_edges:,} edges)")
            print(f"    Finding embedding with minorminer...")

            # Try to find embedding - this measures actual embedding computation time
            start = time.perf_counter()
            embedding = minorminer.find_embedding(
                source_graph.edges(),
                target_graph.edges(),
                verbose=0  # Suppress minorminer output
            )
            elapsed = time.perf_counter() - start

            if embedding:
                # Calculate what had to be omitted to fit on real hardware
                embedded_vars = len(embedding)
                total_vars = len(source_graph.nodes())
                omitted_vars = total_vars - embedded_vars

                # Count edges involving non-embedded nodes
                omitted_edges = 0
                for u, v in source_graph.edges():
                    if u not in embedding or v not in embedding:
                        omitted_edges += 1

                result.update({
                    'embedding_tested': True,
                    'embedding_found': True,
                    'embedding_time_s': elapsed,
                    'num_omitted_vars': omitted_vars,
                    'num_omitted_edges': omitted_edges,
                    'qpu_chip_id': qpu_chip_id,
                    'qpu_num_qubits': qpu_num_qubits,
                })

                omission_pct = 100 * omitted_vars / total_vars if total_vars > 0 else 0
                print(f"    ✓ Embedding found in {elapsed:.2f}s")
                print(f"      Omitted: {omitted_vars:,} vars ({omission_pct:.1f}%), {omitted_edges:,} edges")
            else:
                result.update({
                    'embedding_tested': True,
                    'embedding_found': False,
                    'embedding_time_s': elapsed,
                    'qpu_chip_id': qpu_chip_id,
                    'qpu_num_qubits': qpu_num_qubits,
                })
                print(f"    ✗ No embedding found after {elapsed:.2f}s")
                print(f"      Graph may be too large for available qubits")

        except Exception as e:
            print(f"    ! Embedding test failed: {e}")
            result['embedding_tested'] = False

    except ImportError as e:
        print(f"  ! minorminer not installed, skipping embedding test")
        print(f"    Install with: pip install minorminer")
        result['embedding_tested'] = False

    return result


def _find_embedding_worker(args):
    """
    Worker function for parallel embedding attempts with automatic restarts.

    Each worker keeps trying with new random seeds until either:
    1. Success - valid embedding found
    2. Worker timeout exceeded

    Args:
        args: Tuple of (worker_id, source_edges, target_edges, worker_timeout, try_timeout, verbose)

    Returns:
        Tuple of (worker_id, embedding, elapsed_time, num_tries)
    """
    import minorminer
    import random
    worker_id, source_edges, target_edges, worker_timeout, try_timeout, verbose = args

    worker_start = time.perf_counter()
    num_tries = 0
    best_embedding = None

    while True:
        # Check if we've exceeded worker timeout
        elapsed = time.perf_counter() - worker_start
        if elapsed >= worker_timeout:
            break

        # Calculate remaining time for this try
        remaining_time = worker_timeout - elapsed
        this_try_timeout = min(try_timeout, remaining_time)

        if this_try_timeout < 10:  # Not enough time for meaningful attempt
            break

        num_tries += 1

        # Generate random seed for this try
        random_seed = random.randint(0, 2**31 - 1)

        try_start = time.perf_counter()
        embedding = minorminer.find_embedding(
            source_edges,
            target_edges,
            verbose=verbose if worker_id == 0 else 0,  # Only first worker shows progress
            timeout=this_try_timeout,
            tries=1,  # Single try per call - we handle restarts ourselves
            threads=1,  # Single-threaded (threads parameter isn't effective anyway)
            max_no_improvement=10,  # Exit faster if stuck
            chainlength_patience=10,  # Exit faster if not improving chains
            random_seed=random_seed,  # Fresh random seed each attempt
        )
        try_elapsed = time.perf_counter() - try_start

        if embedding:
            # Success! Return immediately
            total_elapsed = time.perf_counter() - worker_start
            print(f"    ✓ Worker {worker_id}: Found embedding on try {num_tries} after {try_elapsed:.1f}s")
            return (worker_id, embedding, total_elapsed, num_tries)

        # No embedding found, log and continue
        print(f"    ✗ Worker {worker_id}: Try {num_tries} failed after {try_elapsed:.1f}s, restarting...")

    # All tries exhausted
    total_elapsed = time.perf_counter() - worker_start
    return (worker_id, None, total_elapsed, num_tries)


def _find_native_subgraph_seed(source_graph: nx.Graph, target_graph: nx.Graph, max_m: int, max_t: int) -> Optional[Dict]:
    """
    Find a native Zephyr subgraph that exists perfectly in target (no embedding needed).
    This can be used as a seed/initial_chains for faster embedding.

    Returns:
        Partial embedding dict (identity mapping for native nodes), or None
    """
    import dwave_networkx as dnx

    print(f"  Searching for native Zephyr subgraph seed (up to Z({max_m},{max_t}))...")

    # Try to find largest native subgraph
    best_seed = None
    best_size = 0

    for m in range(max_m, 1, -1):  # Start from largest
        for t in range(max_t, 0, -1):
            test_graph = dnx.zephyr_graph(m, t)

            # Check if all nodes and edges exist in target
            nodes_present = all(n in target_graph.nodes() for n in test_graph.nodes())
            edges_present = all(target_graph.has_edge(u, v) for u, v in test_graph.edges())

            if nodes_present and edges_present:
                # Found a perfect native subgraph
                num_nodes = len(test_graph.nodes())
                if num_nodes > best_size:
                    # Create identity embedding for these nodes
                    best_seed = {node: [node] for node in test_graph.nodes()}
                    best_size = num_nodes
                    print(f"    ✓ Found native Z({m},{t}) seed: {num_nodes:,} nodes")
                    return best_seed

    print(f"    ✗ No native subgraph found - starting from scratch")
    return None


def precompute_embedding(config: Dict[str, Any], target_solver_name: str = "Advantage2_system1_7", timeout: int = 3600, try_timeout: int = 600, num_processes: Optional[int] = None) -> Dict[str, Any]:
    """
    Precompute and save embedding for a topology using parallel multiprocessing.

    Uses multiple independent processes to search for embeddings in parallel.
    Each process runs minorminer with automatic restarts on new random seeds.

    Args:
        config: Topology configuration dict
        target_solver_name: Target solver name (default: Advantage2_system1_7)
        timeout: Total timeout in seconds for all workers
        try_timeout: Timeout per individual embedding attempt (default: 600s = 10min)
        num_processes: Number of parallel processes (default: CPU count)

    Saves to: dwave_topologies/embeddings/{target_solver_name}/zephyr_z{m}_t{t}.json.gz
    """
    import os
    import json
    import gzip
    import minorminer
    import multiprocessing as mp
    from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_7_TOPOLOGY

    # Get target topology
    target_topology = ADVANTAGE2_SYSTEM1_7_TOPOLOGY
    target_graph = target_topology.graph

    source_graph = config['graph']
    m = config['m']
    t = config['t']

    print(f"\n{'='*80}")
    print(f"PRECOMPUTING EMBEDDING: Z({m},{t}) → {target_solver_name}")
    print(f"{'='*80}")
    print(f"Source: Z({m},{t}) - {config['num_nodes']:,} nodes, {config['num_edges']:,} edges")
    print(f"Target: {target_solver_name} - {target_topology.num_nodes:,} qubits, {target_topology.num_edges:,} couplers")

    # Determine number of processes
    if num_processes is None:
        num_processes = os.cpu_count() or 4

    # Each worker gets the FULL timeout - they run in parallel!
    worker_timeout = timeout
    max_tries_per_worker = max(1, worker_timeout // try_timeout)
    total_attempts = num_processes * max_tries_per_worker

    print(f"\nStrategy: Parallel multiprocess search with automatic restarts")
    print(f"  Processes: {num_processes} (running in parallel)")
    print(f"  Total timeout: {timeout:,}s ({timeout/3600:.1f}h)")
    print(f"  Try timeout: {try_timeout:,}s ({try_timeout/60:.1f}m)")
    print(f"  Max tries per worker: ~{max_tries_per_worker}")
    print(f"  Total attempts (all workers): ~{total_attempts}")
    print()

    # Try to find a native subgraph seed for faster convergence
    # Look for smaller native Zephyrs that might exist
    seed_m = max(2, m - 3)  # Try 3 sizes smaller
    seed_t = max(1, t - 1)  # Try 1 size smaller
    initial_chains = _find_native_subgraph_seed(source_graph, target_graph, seed_m, seed_t)

    # Prepare worker arguments
    source_edges = list(source_graph.edges())
    target_edges = list(target_graph.edges())

    worker_args = [
        (i, source_edges, target_edges, worker_timeout, try_timeout, 1)  # verbose=1 for first worker
        for i in range(num_processes)
    ]

    print(f"Starting {num_processes} parallel workers (each will restart with new seeds automatically)...")
    print(f"(First worker will show progress, others run silently)")
    print(f"(Will terminate ALL workers as soon as ANY worker succeeds)\n")

    start = time.perf_counter()

    # Run parallel embedding search with early termination
    # Use imap_unordered to process results as they come in
    embedding = None
    best_worker = None
    best_time = None
    best_tries = None

    with mp.Pool(processes=num_processes) as pool:
        # Start all workers
        async_results = [pool.apply_async(_find_embedding_worker, (args,)) for args in worker_args]

        # Poll for results, terminate early on success
        while async_results:
            for i, result in enumerate(async_results):
                if result.ready():
                    worker_id, emb, worker_elapsed, num_tries = result.get()
                    if emb:
                        # Success! Terminate pool immediately
                        embedding = emb
                        best_worker = worker_id
                        best_time = worker_elapsed
                        best_tries = num_tries
                        print(f"\n✓ Embedding found! Terminating all other workers...")
                        pool.terminate()
                        pool.join()
                        break
                    else:
                        # Worker finished without success, remove from list
                        async_results.pop(i)
                        break
            else:
                # No result ready yet, sleep briefly
                time.sleep(0.1)

            # Break outer loop if we found embedding
            if embedding:
                break

        # If we get here without embedding, all workers failed
        if not embedding:
            pool.close()
            pool.join()

    elapsed = time.perf_counter() - start

    if not embedding:
        print(f"\n✗ No embedding found after {elapsed:.2f}s")
        print(f"  All {num_processes} workers exhausted their attempts")
        print(f"  Consider:")
        print(f"    - Increasing --embedding-timeout (total time)")
        print(f"    - Increasing --try-timeout (time per attempt)")
        print(f"    - Using a smaller topology configuration")
        print(f"    - Checking if topology fits on target hardware")
        return None

    print(f"\n✓ Embedding found by worker {best_worker} after {best_tries} tries in {best_time:.2f}s")
    print(f"  Wall clock time (includes early termination): {elapsed:.2f}s")

    # Convert embedding dict to JSON-serializable format
    embedding_list = {str(k): list(v) for k, v in embedding.items()}

    # Calculate statistics
    embedded_vars = len(embedding)
    total_vars = len(source_graph.nodes())
    omitted_vars = total_vars - embedded_vars

    # Count chain lengths
    chain_lengths = [len(chain) for chain in embedding.values()]
    avg_chain_length = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
    max_chain_length = max(chain_lengths) if chain_lengths else 0

    # Count omitted edges
    omitted_edges = 0
    for u, v in source_graph.edges():
        if u not in embedding or v not in embedding:
            omitted_edges += 1

    print(f"  Embedded: {embedded_vars:,} / {total_vars:,} vars ({100*embedded_vars/total_vars:.1f}%)")
    print(f"  Omitted: {omitted_vars:,} vars, {omitted_edges:,} edges")
    print(f"  Avg chain length: {avg_chain_length:.1f}")
    print(f"  Max chain length: {max_chain_length}")

    # Create embedding data structure
    embedding_data = {
        "metadata": {
            "description": f"Precomputed embedding: Z({m},{t}) → {target_solver_name}",
            "source_topology": f"Zephyr_Z{m}_T{t}_Generic",
            "target_solver": target_solver_name,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "computation_time_s": round(elapsed, 2),
            "source_nodes": config['num_nodes'],
            "source_edges": config['num_edges'],
            "target_qubits": target_topology.num_nodes,
            "target_couplers": target_topology.num_edges,
        },
        "statistics": {
            "embedded_vars": embedded_vars,
            "total_vars": total_vars,
            "omitted_vars": omitted_vars,
            "omitted_edges": omitted_edges,
            "embedding_coverage_pct": round(100 * embedded_vars / total_vars, 2),
            "avg_chain_length": round(avg_chain_length, 2),
            "max_chain_length": max_chain_length,
        },
        "embedding": embedding_list,
    }

    # Save to dwave_topologies/embeddings/{solver_name}/
    embeddings_dir = os.path.join("dwave_topologies", "embeddings", target_solver_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    filename = f"zephyr_z{m}_t{t}.embed.json.gz"
    filepath = os.path.join(embeddings_dir, filename)

    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(embedding_data, f, indent=2)

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"\n✓ Embedding saved to: {filepath}")
    print(f"  Compressed size: {file_size_kb:.1f} KB")

    return embedding_data


def generate_topology_file(m: int, t: int):
    """Generate a static gzipped JSON topology file for Zephyr(m, t)."""
    import os
    import json
    import gzip

    config = analyze_zephyr_config(m, t)

    filename = f"zephyr_z{m}_t{t}.json.gz"
    filepath = os.path.join("dwave_topologies", "topologies", filename)

    # Calculate utilization
    node_util = config['node_utilization_pct']
    edge_util = config['edge_utilization_pct']

    # Calculate expected GSE
    from shared.energy_utils import expected_solution_energy
    expected_gse = expected_solution_energy(
        num_nodes=config['num_nodes'],
        num_edges=config['num_edges'],
        c=0.75
    )

    # Create JSON topology data structure
    topology_data = {
        "metadata": {
            "description": f"D-Wave Zephyr Z({m}, {t}) topology definition",
            "generated_from": f"dwave_networkx.zephyr_graph({m}, {t})",
            "solver_name": f"Zephyr_Z{m}_T{t}_Generic",
            "topology_type": "zephyr",
            "topology_shape": [m, t],
            "num_nodes": config['num_nodes'],
            "num_edges": config['num_edges'],
            "avg_degree": round(config['avg_degree'], 2),
            "advantage2_node_utilization_pct": round(node_util, 1),
            "advantage2_edge_utilization_pct": round(edge_util, 1),
            "expected_gse": round(expected_gse, 1),
            "fits_advantage2": config['fits_advantage2'],
            "notes": [
                "Generic, QPU-agnostic graph structure (no solver-specific defect patterns)",
                f"{'Fits comfortably' if config['fits_advantage2'] else 'DOES NOT FIT'} on Advantage2-System1.6",
                f"Round parameters ({m}×{t}) for clean generation"
            ]
        },
        "properties": {
            "topology": {
                "type": "zephyr",
                "shape": [m, t]
            },
            "num_qubits": config['num_nodes'],
            "num_couplers": config['num_edges'],
            "chip_id": f"Generic_Z{m}_T{t}",
            "supported_problem_types": ["qubo", "ising"]
        },
        "nodes": config['nodes'],
        "edges": [[u, v] for u, v in config['edges']],  # Convert tuples to lists for JSON
        "docs": {
            "topology": "https://support.dwavesys.com/hc/en-us/articles/360003695354-What-Is-the-Zephyr-Topology",
            "solver": "https://docs.dwavesys.com/docs/latest/c_solver_properties.html",
            "overview": "https://docs.ocean.dwavesys.com/en/latest/concepts/topology.html"
        }
    }

    print(f"\n{'='*80}")
    print(f"Generating topology JSON file: {filepath}")
    print(f"{'='*80}")
    print(f"  Config: Z({m},{t})")
    print(f"  Nodes: {config['num_nodes']:,}")
    print(f"  Edges: {config['num_edges']:,}")
    print(f"  Avg degree: {config['avg_degree']:.2f}")
    print(f"  Fits Advantage2: {'✓' if config['fits_advantage2'] else '✗'}")
    print(f"  Expected GSE: {expected_gse:.1f}")

    # Write the gzipped JSON file
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(topology_data, f, indent=2)

    print(f"\n✓ Topology gzipped JSON file written to: {filepath}")
    print(f"  Compressed size: {os.path.getsize(filepath) / 1024:.1f} KB")
    print(f"\nTo use this topology:")
    print(f"  1. Load with the JSON topology loader (auto-detects .gz):")
    print(f"     from dwave_topologies.topologies.json_loader import load_json_topology")
    print(f"     topology = load_json_topology('zephyr_z{m}_t{t}.json')  # Works with .gz")
    print(f"  2. Or set as default in __init__.py")


def print_results_table(results: List[Dict[str, Any]]):
    """Print comparison table of all analyzed configurations."""
    print("\n" + "="*100)
    print("TOPOLOGY COMPARISON TABLE")
    print("="*100)
    print(f"{'Config':<10} {'Nodes':<8} {'Edges':<8} {'Avg Deg':<9} {'Node%':<8} {'Edge%':<8} {'Fits?':<6}")
    print("-"*100)

    for r in results:
        config = r['config_info']
        fits = '✓' if config['fits_advantage2'] else '✗'
        print(f"{config['config']:<10} "
              f"{config['num_nodes']:<8,} "
              f"{config['num_edges']:<8,} "
              f"{config['avg_degree']:<9.2f} "
              f"{config['node_utilization_pct']:<7.1f}% "
              f"{config['edge_utilization_pct']:<7.1f}% "
              f"{fits:<6}")

    print("\n" + "="*100)
    print("PERFORMANCE METRICS")
    print("="*100)
    print(f"{'Config':<10} {'Expected GSE':<14} {'SA Avg Min':<12} {'SA Best':<12} {'Avg Time (s)':<14}")
    print("-"*100)

    for r in results:
        perf = r['sa_performance']
        print(f"{r['config_info']['config']:<10} "
              f"{perf['expected_gse']:<14.1f} "
              f"{perf['avg_sa_min']:<12.1f} "
              f"{perf['best_sa']:<12.1f} "
              f"{perf['avg_solve_time_s']:<14.3f}")

    # Embedding results if available
    if any(r['embedding']['embedding_tested'] for r in results):
        print("\n" + "="*100)
        print("EMBEDDING FEASIBILITY (on Advantage2 QPU)")
        print("="*100)
        print(f"{'Config':<10} {'Found?':<8} {'Time (s)':<12} {'Omitted Vars':<14} {'Omitted Edges':<14}")
        print("-"*100)

        for r in results:
            emb = r['embedding']
            if emb['embedding_tested']:
                found = '✓' if emb['embedding_found'] else '✗'
                time_str = f"{emb['embedding_time_s']:.2f}" if emb['embedding_time_s'] else 'N/A'
                vars_str = str(emb['num_omitted_vars']) if emb['num_omitted_vars'] is not None else 'N/A'
                edges_str = str(emb['num_omitted_edges']) if emb['num_omitted_edges'] is not None else 'N/A'

                print(f"{r['config_info']['config']:<10} "
                      f"{found:<8} "
                      f"{time_str:<12} "
                      f"{vars_str:<14} "
                      f"{edges_str:<14}")

    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    # Find most aggressive config that fits
    fitting_configs = [r for r in results if r['config_info']['fits_advantage2']]
    if fitting_configs:
        best = max(fitting_configs, key=lambda r: r['config_info']['num_nodes'])
        config = best['config_info']
        perf = best['sa_performance']

        print(f"✓ Recommended: {config['config']}")
        print(f"  - {config['num_nodes']:,} nodes, {config['num_edges']:,} edges (avg degree: {config['avg_degree']:.2f})")
        print(f"  - Utilization: {config['node_utilization_pct']:.1f}% nodes, {config['edge_utilization_pct']:.1f}% edges")
        print(f"  - Expected GSE: {perf['expected_gse']:.1f} ± {perf['expected_variance']:.1f}")
        print(f"  - SA performance: {perf['avg_sa_min']:.1f} (avg min over {perf['num_samples']} samples)")

        if best['embedding']['embedding_tested'] and best['embedding']['embedding_found']:
            emb = best['embedding']
            print(f"  - QPU embedding: ✓ Found in {emb['embedding_time_s']:.2f}s "
                  f"({emb['num_omitted_vars']} vars, {emb['num_omitted_edges']} edges omitted)")
    else:
        print("⚠ No configurations fit within Advantage2-System1.6 capacity")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Zephyr topology configurations for QUIP protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--configs', '-c',
                       type=str,
                       default='10,2 10,3 10,4 11,2 11,3 11,4 12,2 12,3',
                       help='Space-separated list of m,t configs (default: 10,2 through 12,3)')

    parser.add_argument('--samples', '-s',
                       type=int,
                       default=50,
                       help='Number of random Ising problems to test (default: 50)')

    parser.add_argument('--num-reads',
                       type=int,
                       default=64,
                       help='Number of SA reads per problem (default: 64)')

    parser.add_argument('--num-sweeps',
                       type=int,
                       default=256,
                       help='Number of SA sweeps per read (default: 256)')

    parser.add_argument('--test-embedding',
                       action='store_true',
                       help='Test QPU embedding (requires D-Wave credentials)')

    parser.add_argument('--precompute-embedding',
                       action='store_true',
                       help='Precompute and save embedding')

    parser.add_argument('--embedding-timeout',
                       type=str,
                       default='1h',
                       help='Total timeout for all embedding workers (default: 1h). Examples: 30m, 2h, 1d, 1w')

    parser.add_argument('--try-timeout',
                       type=str,
                       default='10m',
                       help='Timeout per individual embedding attempt (default: 10m). Examples: 30s, 5m, 30m')

    parser.add_argument('--generate-topology',
                       type=str,
                       metavar='M,T',
                       help='Generate static topology file for Zephyr(m,t). Example: --generate-topology 11,4')

    args = parser.parse_args()

    # Handle topology file generation
    if args.generate_topology:
        m, t = map(int, args.generate_topology.split(','))
        generate_topology_file(m, t)
        return

    # Parse configurations
    configs: List[Tuple[int, int]] = []
    for cfg in args.configs.split():
        m, t = map(int, cfg.split(','))
        configs.append((m, t))

    print(f"Analyzing {len(configs)} Zephyr configurations...")
    print(f"Target QPU: Advantage2-System1.6 ({ADVANTAGE2_NODES:,} nodes, {ADVANTAGE2_EDGES:,} edges)")
    print(f"SA parameters: {args.num_reads} reads × {args.num_sweeps} sweeps")
    print(f"Test samples: {args.samples} random Ising problems per config\n")

    results = []

    for m, t in configs:
        print(f"\n{'='*80}")
        print(f"Analyzing Z({m},{t})")
        print(f"{'='*80}")

        # Analyze graph structure
        config_info = analyze_zephyr_config(m, t)
        print(f"  Nodes: {config_info['num_nodes']:,}")
        print(f"  Edges: {config_info['num_edges']:,}")
        print(f"  Avg degree: {config_info['avg_degree']:.2f}")
        print(f"  Fits Advantage2: {'✓' if config_info['fits_advantage2'] else '✗'}")
        print(f"  Utilization: {config_info['node_utilization_pct']:.1f}% nodes, "
              f"{config_info['edge_utilization_pct']:.1f}% edges")

        # Test SA performance
        sa_performance = test_sa_performance(
            config_info,
            num_samples=args.samples,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
        )
        print(f"  Expected GSE: {sa_performance['expected_gse']:.1f} ± {sa_performance['expected_variance']:.1f}")
        print(f"  SA avg min: {sa_performance['avg_sa_min']:.1f}")
        print(f"  SA best: {sa_performance['best_sa']:.1f}")
        print(f"  Avg solve time: {sa_performance['avg_solve_time_s']:.3f}s")

        # Test or precompute embedding if requested
        embedding_result = {'embedding_tested': False}
        if args.precompute_embedding:
            # Parse timeouts and precompute embedding
            timeout_seconds = parse_timeout(args.embedding_timeout)
            try_timeout_seconds = parse_timeout(args.try_timeout)
            precompute_embedding(
                config_info,
                target_solver_name="Advantage2_system1_7",
                timeout=timeout_seconds,
                try_timeout=try_timeout_seconds
            )
        elif args.test_embedding:
            embedding_result = test_embedding_feasibility(config_info)

        results.append({
            'config_info': config_info,
            'sa_performance': sa_performance,
            'embedding': embedding_result,
        })

    # Print comparison table
    print_results_table(results)


if __name__ == '__main__':
    main()
