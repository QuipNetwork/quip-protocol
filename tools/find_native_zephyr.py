#!/usr/bin/env python3
"""
Find which Zephyr(m,t) topologies are embeddable on Advantage2 using feasibility filter.

Uses minorminer's feasibility_filter to check if a Zephyr topology can be embedded
(even with chains, not just 1:1) onto the real QPU hardware.

This is more accurate than manual node/edge checking, as it uses minorminer's
heuristics to determine if an embedding is feasible.
"""

import argparse
import sys
from collections import defaultdict
from typing import List, Dict

# Add parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dwave_networkx as dnx
from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_12_TOPOLOGY

try:
    from minorminer.utils.feasibility import embedding_feasibility_filter
except ImportError:
    print("ERROR: minorminer not installed or feasibility module not available")
    print("Install with: pip install minorminer")
    sys.exit(1)


def check_zephyr_feasibility(m: int, t: int, target_graph) -> bool:
    """
    Check if Zephyr(m,t) can be embedded on target graph using feasibility filter.

    Args:
        m: Zephyr m parameter
        t: Zephyr t parameter
        target_graph: NetworkX graph of target hardware

    Returns:
        True if embedding is feasible, False otherwise
    """
    source_graph = dnx.zephyr_graph(m, t)
    return embedding_feasibility_filter(S=source_graph, T=target_graph)


def find_embeddable_zephyrs(target_graph, max_m: int = 12, max_t: int = 4) -> List[Dict]:
    """
    Find all Zephyr topologies that are embeddable on target graph.

    Returns:
        List of dicts with keys: m, t, nodes, edges, is_feasible
    """
    results = []

    target_nodes = len(target_graph.nodes())
    target_edges = len(target_graph.edges())

    print(f"Testing Zephyr topologies for embedding feasibility...")
    print(f"Target: {target_nodes:,} nodes, {target_edges:,} edges\n")

    for m in range(2, max_m + 1):
        for t in range(1, max_t + 1):
            print(f"Testing Z({m},{t})...", end=" ", flush=True)

            # Generate Zephyr topology
            zephyr = dnx.zephyr_graph(m, t)
            total_nodes = len(zephyr.nodes())
            total_edges = len(zephyr.edges())

            # Check feasibility
            is_feasible = check_zephyr_feasibility(m, t, target_graph)

            node_util_pct = 100 * total_nodes / target_nodes
            edge_util_pct = 100 * total_edges / target_edges

            result = {
                'm': m,
                't': t,
                'config': f'Z({m},{t})',
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'is_feasible': is_feasible,
                'node_utilization_pct': node_util_pct,
                'edge_utilization_pct': edge_util_pct,
            }

            results.append(result)

            if is_feasible:
                print(f"✓ FEASIBLE ({total_nodes:,} nodes, {total_edges:,} edges, "
                      f"{node_util_pct:.1f}% node util)")
            else:
                print(f"✗ INFEASIBLE ({total_nodes:,} nodes exceeds capacity or too dense)")

    return results


def print_results(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "="*100)
    print("EMBEDDABLE ZEPHYR TOPOLOGIES")
    print("="*100)

    feasible = [r for r in results if r['is_feasible']]
    infeasible = [r for r in results if not r['is_feasible']]

    if not feasible:
        print("✗ No feasible Zephyr topologies found")
        print("\nAll tested topologies are too large or too dense to embed.")
    else:
        print(f"{'Config':<10} {'Nodes':<8} {'Edges':<8} {'Node%':<8} {'Edge%':<8} {'Status':<10}")
        print("-"*100)

        # Sort by nodes descending to show largest first
        for r in sorted(feasible, key=lambda x: x['total_nodes'], reverse=True):
            print(f"{r['config']:<10} "
                  f"{r['total_nodes']:<8,} "
                  f"{r['total_edges']:<8,} "
                  f"{r['node_utilization_pct']:<7.1f}% "
                  f"{r['edge_utilization_pct']:<7.1f}% "
                  f"{'✓ Feasible':<10}")

    if infeasible:
        print("\nINFEASIBLE TOPOLOGIES (too large or dense):")
        print("-"*100)
        for r in infeasible:
            print(f"{r['config']:<10} "
                  f"{r['total_nodes']:<8,} "
                  f"{r['total_edges']:<8,} "
                  f"{r['node_utilization_pct']:<7.1f}% "
                  f"{r['edge_utilization_pct']:<7.1f}% "
                  f"{'✗ Infeasible':<10}")

    if feasible:
        print("\n" + "="*100)
        print("LARGEST FEASIBLE M FOR EACH T")
        print("="*100)

        # Group feasible results by t and find max m for each
        by_t = defaultdict(list)
        for r in feasible:
            by_t[r['t']].append(r)

        # Print largest m for each t value
        for t in sorted(by_t.keys()):
            largest_for_t = max(by_t[t], key=lambda r: r['m'])
            print(f"t={t}: m={largest_for_t['m']} → Z({largest_for_t['m']},{t}) "
                  f"({largest_for_t['total_nodes']:,} nodes, {largest_for_t['total_edges']:,} edges, "
                  f"{largest_for_t['node_utilization_pct']:.1f}% util)")

        print("\n" + "="*100)
        print("RECOMMENDATION")
        print("="*100)

        # Recommend largest feasible topology
        largest = max(feasible, key=lambda r: r['total_nodes'])
        print(f"✓ Largest feasible: {largest['config']}")
        print(f"  - {largest['total_nodes']:,} nodes, {largest['total_edges']:,} edges")
        print(f"  - {largest['node_utilization_pct']:.1f}% node utilization")
        print(f"  - Requires embedding (use minorminer or precompute)")
        print(f"\nTo precompute embedding:")
        print(f"  python tools/analyze_topology_sizes.py \\")
        print(f"    --configs \"{largest['m']},{largest['t']}\" \\")
        print(f"    --precompute-embedding \\")
        print(f"    --embedding-timeout 1w \\")
        print(f"    --try-timeout 15m")
        print(f"\nUsage:")
        print(f"  from dwave_topologies import zephyr")
        print(f"  topology = zephyr({largest['m']}, {largest['t']})")


def main():
    parser = argparse.ArgumentParser(
        description="Find embeddable Zephyr topologies using feasibility filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default range
  python tools/find_native_zephyr.py

  # Test larger topologies
  python tools/find_native_zephyr.py --max-m 15 --max-t 5

Notes:
  - Uses minorminer's feasibility_filter (more accurate than manual checking)
  - Tests embedding feasibility (with chains), not just 1:1 subgraph matching
  - Uses predownloaded Advantage2_system1.12 topology (no QPU access needed)
        """
    )
    parser.add_argument('--max-m', type=int, default=12,
                       help='Maximum m parameter to test (default: 12)')
    parser.add_argument('--max-t', type=int, default=4,
                       help='Maximum t parameter to test (default: 4)')

    args = parser.parse_args()

    # Load Advantage2 topology (no QPU access needed)
    topology = ADVANTAGE2_SYSTEM1_12_TOPOLOGY
    target_graph = topology.graph

    print(f"Target QPU: {topology.solver_name}")
    print(f"  Physical qubits: {topology.num_nodes:,}")
    print(f"  Physical couplers: {topology.num_edges:,}")
    print(f"  Topology type: {topology.topology_type} {topology.topology_shape}")
    print()

    # Find all embeddable topologies
    results = find_embeddable_zephyrs(
        target_graph,
        max_m=args.max_m,
        max_t=args.max_t
    )

    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
