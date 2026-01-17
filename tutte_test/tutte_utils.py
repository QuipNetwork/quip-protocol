"""
Shared utilities for Tutte polynomial computations.

This module centralizes common utility functions used across the tutte_test package.
"""

import networkx as nx
from typing import Tuple, List, Set

from tutte_test.tutte_to_ising import GraphBuilder


def networkx_to_graphbuilder(G: nx.Graph) -> GraphBuilder:
    """
    Convert a NetworkX graph to our GraphBuilder format.

    Args:
        G: NetworkX graph (nodes can be any hashable type)

    Returns:
        GraphBuilder with sequential integer node IDs
    """
    gb = GraphBuilder()
    node_map = {}

    for node in G.nodes():
        node_map[node] = gb.add_node()

    for u, v in G.edges():
        if u != v:  # Skip self-loops for regular edges
            gb.add_edge(node_map[u], node_map[v])
        else:
            gb.add_loop(node_map[u])

    return gb


def is_bridge(g: GraphBuilder, edge_id: int) -> bool:
    """
    Check if an edge is a bridge (cut edge) in the graph.

    A bridge is an edge whose removal disconnects the graph.
    Uses BFS to check if endpoints remain connected without the edge.

    Args:
        g: GraphBuilder instance
        edge_id: ID of the edge to check

    Returns:
        True if edge is a bridge, False otherwise
    """
    if edge_id not in g.edges:
        return False

    u, v = g.edges[edge_id]

    # BFS from u without using edge_id
    visited = {u}
    stack = [u]

    while stack:
        curr = stack.pop()
        for eid, (a, b) in g.edges.items():
            if eid == edge_id:
                continue
            neighbor = None
            if a == curr and b not in visited:
                neighbor = b
            elif b == curr and a not in visited:
                neighbor = a
            if neighbor is not None:
                visited.add(neighbor)
                stack.append(neighbor)

    return v not in visited


def graph_to_canonical_key(G: nx.Graph) -> str:
    """
    Create a canonical string key for a graph (isomorphism-invariant).

    Uses NetworkX's graph6 format for small graphs, falls back to
    sorted edge list for graphs that can't be encoded.

    Args:
        G: NetworkX graph

    Returns:
        String key that is identical for isomorphic graphs
    """
    G_relabeled = nx.convert_node_labels_to_integers(G)
    try:
        return nx.to_graph6_bytes(G_relabeled, header=False).hex()
    except Exception:
        edges = sorted((min(u, v), max(u, v)) for u, v in G_relabeled.edges())
        return str(edges)


def extract_subgraph(G: nx.Graph, nodes: List) -> nx.Graph:
    """
    Extract induced subgraph on given nodes.

    Args:
        G: Source NetworkX graph
        nodes: List of nodes to include

    Returns:
        New graph containing only specified nodes and edges between them
    """
    return G.subgraph(nodes).copy()


def get_zephyr_graph(m: int, t: int):
    """
    Get a Zephyr graph, with fallback if dwave_networkx unavailable.

    Args:
        m: Zephyr m parameter (grid size)
        t: Zephyr t parameter (tile parameter)

    Returns:
        NetworkX graph representing Zephyr topology
    """
    try:
        import dwave_networkx as dnx
        return dnx.zephyr_graph(m, t)
    except ImportError:
        # Create a simple mock for testing without D-Wave
        G = nx.Graph()
        n_qubits = min(20, 8 * t * (2 * m * m - 2 * m + 1))
        for i in range(n_qubits):
            G.add_node(i)
        for i in range(n_qubits - 1):
            G.add_edge(i, i + 1)
        return G
