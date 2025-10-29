#!/usr/bin/env python3
"""
Find the largest native Zephyr(m,t) subgraph within a defective QPU topology.

This identifies a "clean" Zephyr configuration with NO missing nodes or edges,
eliminating the need for embedding.
"""

import argparse
import networkx as nx
import dwave_networkx as dnx
from typing import Tuple, Set, List, Dict
from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_6_TOPOLOGY


def get_zephyr_coordinates(m: int, t: int) -> Dict[Tuple[int, int, int, int], int]:
    """
    Get mapping from Zephyr coordinates (u, w, k, z) to linear node indices.

    Zephyr(m, t) coordinates:
    - u ∈ [0, m): tile row/column
    - w ∈ {0, 1}: internal/external qubits
    - k ∈ [0, 2t): qubit position within tile
    - z ∈ {0, 1, 2, 3}: qubit orientation
    """
    G = dnx.zephyr_graph(m, t)
    coords = dnx.zephyr_coordinates(m, t)

    # coords.linear_to_zephyr gives us the mapping
    coord_to_node = {}
    for node in G.nodes():
        coord = coords.linear_to_zephyr(node)
        coord_to_node[coord] = node

    return coord_to_node


def check_zephyr_subgraph(m: int, t: int, target_graph: nx.Graph) -> Tuple[bool, int, int]:
    """
    Check if a perfect Zephyr(m,t) subgraph exists in target graph.

    Returns:
        (is_perfect, num_nodes_present, num_edges_present)
    """
    # Generate the Zephyr(m,t) graph
    zephyr = dnx.zephyr_graph(m, t)

    # Check which nodes are present
    missing_nodes = 0
    for node in zephyr.nodes():
        if node not in target_graph.nodes():
            missing_nodes += 1

    # Check which edges are present
    missing_edges = 0
    for u, v in zephyr.edges():
        # Only count edge if both nodes exist
        if u in target_graph.nodes() and v in target_graph.nodes():
            if not target_graph.has_edge(u, v):
                missing_edges += 1
        else:
            missing_edges += 1

    is_perfect = (missing_nodes == 0 and missing_edges == 0)
    nodes_present = len(zephyr.nodes()) - missing_nodes
    edges_present = len(zephyr.edges()) - missing_edges

    return is_perfect, nodes_present, edges_present


def find_largest_native_zephyr(target_graph: nx.Graph, max_m: int = 12, max_t: int = 4) -> List[Dict]:
    """
    Find all perfect Zephyr subgraphs within the target topology.

    Returns:
        List of dicts with keys: m, t, nodes, edges, is_perfect, utilization_pct
    """
    results = []

    target_nodes = len(target_graph.nodes())
    target_edges = len(target_graph.edges())

    print(f"Searching for perfect Zephyr subgraphs in target topology...")
    print(f"Target: {target_nodes:,} nodes, {target_edges:,} edges\n")

    for m in range(2, max_m + 1):
        for t in range(1, max_t + 1):
            print(f"Testing Z({m},{t})...", end=" ")

            is_perfect, nodes_present, edges_present = check_zephyr_subgraph(m, t, target_graph)

            zephyr = dnx.zephyr_graph(m, t)
            total_nodes = len(zephyr.nodes())
            total_edges = len(zephyr.edges())

            node_util_pct = 100 * total_nodes / target_nodes
            edge_util_pct = 100 * total_edges / target_edges

            result = {
                'm': m,
                't': t,
                'config': f'Z({m},{t})',
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'nodes_present': nodes_present,
                'edges_present': edges_present,
                'is_perfect': is_perfect,
                'node_utilization_pct': node_util_pct,
                'edge_utilization_pct': edge_util_pct,
            }

            results.append(result)

            if is_perfect:
                print(f"✓ PERFECT ({nodes_present:,} nodes, {edges_present:,} edges)")
            else:
                missing_nodes = total_nodes - nodes_present
                missing_edges = total_edges - edges_present
                print(f"✗ Missing {missing_nodes} nodes, {missing_edges} edges")

    return results


def print_results(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "="*100)
    print("PERFECT ZEPHYR SUBGRAPHS (No Embedding Needed)")
    print("="*100)

    perfect = [r for r in results if r['is_perfect']]

    if not perfect:
        print("✗ No perfect Zephyr subgraphs found")
        print("\nClosest matches:")
        # Sort by nodes_present descending
        closest = sorted(results, key=lambda r: r['nodes_present'], reverse=True)[:5]
        for r in closest:
            missing_nodes = r['total_nodes'] - r['nodes_present']
            missing_edges = r['total_edges'] - r['edges_present']
            print(f"  {r['config']}: {r['nodes_present']:,}/{r['total_nodes']:,} nodes, "
                  f"missing {missing_nodes} nodes + {missing_edges} edges")
    else:
        print(f"{'Config':<10} {'Nodes':<8} {'Edges':<8} {'Node%':<8} {'Edge%':<8}")
        print("-"*100)

        # Sort by nodes descending to show largest first
        for r in sorted(perfect, key=lambda x: x['total_nodes'], reverse=True):
            print(f"{r['config']:<10} "
                  f"{r['total_nodes']:<8,} "
                  f"{r['total_edges']:<8,} "
                  f"{r['node_utilization_pct']:<7.1f}% "
                  f"{r['edge_utilization_pct']:<7.1f}%")

        print("\n" + "="*100)
        print("RECOMMENDATION")
        print("="*100)

        # Recommend largest perfect subgraph
        largest = max(perfect, key=lambda r: r['total_nodes'])
        print(f"✓ Use {largest['config']} - largest perfect Zephyr subgraph")
        print(f"  - {largest['total_nodes']:,} nodes, {largest['total_edges']:,} edges")
        print(f"  - {largest['node_utilization_pct']:.1f}% node utilization")
        print(f"  - NO EMBEDDING REQUIRED - native hardware support")
        print(f"\nUsage:")
        print(f"  # In DEFAULT_TOPOLOGY:")
        print(f"  from dwave_topologies.topologies.zephyr_z{largest['m']}_t{largest['t']} import ZEPHYR_Z{largest['m']}_T{largest['t']}_TOPOLOGY")
        print(f"  DEFAULT_TOPOLOGY = ZEPHYR_Z{largest['m']}_T{largest['t']}_TOPOLOGY")


def main():
    parser = argparse.ArgumentParser(
        description="Find largest native Zephyr subgraph in QPU topology"
    )
    parser.add_argument('--max-m', type=int, default=12,
                       help='Maximum m parameter to test (default: 12)')
    parser.add_argument('--max-t', type=int, default=4,
                       help='Maximum t parameter to test (default: 4)')

    args = parser.parse_args()

    # Load Advantage2 topology
    topology = ADVANTAGE2_SYSTEM1_6_TOPOLOGY
    target_graph = topology.graph

    print(f"Target QPU: {topology.solver_name}")
    print(f"  Physical qubits: {topology.num_nodes:,}")
    print(f"  Physical couplers: {topology.num_edges:,}")
    print(f"  Topology type: {topology.properties.get('topology', {})}")
    print()

    # Find all perfect subgraphs
    results = find_largest_native_zephyr(
        target_graph,
        max_m=args.max_m,
        max_t=args.max_t
    )

    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
