"""Series-Parallel Graph Recognition.

A 2-terminal series-parallel graph can be reduced to a single edge using:
- Series reduction: Remove degree-2 vertex, connect its neighbors
- Parallel reduction: Merge multiple edges between same endpoints

Series-parallel graphs have treewidth ≤ 2.
"""

from __future__ import annotations
from typing import Dict
from .graph import Graph


def is_series_parallel(graph: Graph) -> bool:
    """Check if graph is series-parallel via reduction algorithm. O(n+m)."""
    if graph.edge_count() <= 1:
        return True

    # Adjacency with edge multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in graph.nodes}
    for u, v in graph.edges:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    n_edges = graph.edge_count()

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges between same pair
        for u in adj:
            for v, count in adj[u].items():
                if count > 1:
                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors
                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]
                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1
                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return False

    return True
