"""Disjoint Cover Algorithms for Graph Synthesis.

This module provides algorithms for covering a graph with disjoint
copies of known minors from the rainbow table. This is a key component
of the creation-expansion-join algorithm.

Key concepts:
- Tile: A mapping of a minor graph onto a subgraph
- Cover: A collection of non-overlapping tiles
- Fringe: Edges in the cover that don't exist in the input (over-coverage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Iterator

import networkx as nx
from networkx.algorithms import isomorphism

from .graph import Graph
from .rainbow_table import MinorEntry, RainbowTable


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Tile:
    """A mapping of a minor graph onto a portion of the target graph.

    The tile represents placing a copy of the minor at specific nodes
    and edges in the target graph.
    """
    minor: MinorEntry
    node_mapping: Dict[int, int]  # minor_node -> target_node
    edge_mapping: Dict[Tuple[int, int], Tuple[int, int]]  # minor_edge -> target_edge

    @property
    def covered_nodes(self) -> Set[int]:
        """Nodes in target graph covered by this tile."""
        return set(self.node_mapping.values())

    @property
    def covered_edges(self) -> Set[Tuple[int, int]]:
        """Edges in target graph covered by this tile."""
        return set(self.edge_mapping.values())

    @property
    def minor_nodes(self) -> Set[int]:
        """Nodes in the minor graph."""
        return set(self.node_mapping.keys())

    @property
    def minor_edges(self) -> Set[Tuple[int, int]]:
        """Edges in the minor graph."""
        return set(self.edge_mapping.keys())


@dataclass
class Cover:
    """A collection of non-overlapping tiles covering a graph."""
    tiles: List[Tile] = field(default_factory=list)
    covered_nodes: Set[int] = field(default_factory=set)
    covered_edges: Set[Tuple[int, int]] = field(default_factory=set)
    uncovered_edges: Set[Tuple[int, int]] = field(default_factory=set)

    def is_complete(self) -> bool:
        """Check if all edges are covered."""
        return len(self.uncovered_edges) == 0

    def add_tile(self, tile: Tile) -> bool:
        """Add a tile if it doesn't overlap with existing tiles.

        Returns True if tile was added, False if it overlaps.
        """
        # Check for node overlap
        if tile.covered_nodes & self.covered_nodes:
            return False

        # Check for edge overlap
        if tile.covered_edges & self.covered_edges:
            return False

        # Add tile
        self.tiles.append(tile)
        self.covered_nodes.update(tile.covered_nodes)
        self.covered_edges.update(tile.covered_edges)
        self.uncovered_edges -= tile.covered_edges

        return True

    def total_tiles(self) -> int:
        """Number of tiles in cover."""
        return len(self.tiles)


@dataclass
class Fringe:
    """Edges in the cover that don't exist in the input (over-coverage).

    When we tile a graph with minors, sometimes the tiling includes
    edges that aren't in the original graph. These are "fringe" edges
    that need to be handled in the synthesis.
    """
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    nodes: Set[int] = field(default_factory=set)

    def is_empty(self) -> bool:
        """Check if there are no fringe edges."""
        return len(self.edges) == 0

    def as_graph(self) -> Graph:
        """Convert fringe edges to a Graph."""
        if not self.edges:
            return Graph(nodes=frozenset(), edges=frozenset())
        return Graph(nodes=frozenset(self.nodes), edges=frozenset(self.edges))

    def edge_count(self) -> int:
        """Number of fringe edges."""
        return len(self.edges)


# =============================================================================
# SUBGRAPH ISOMORPHISM
# =============================================================================

def find_subgraph_isomorphisms(
    target: Graph,
    pattern: Graph,
    max_matches: int = 100
) -> List[Dict[int, int]]:
    """Find all subgraph isomorphisms of pattern in target.

    Uses NetworkX's VF2 algorithm for subgraph isomorphism.

    Args:
        target: Graph to search in
        pattern: Pattern graph to find
        max_matches: Maximum number of matches to return

    Returns:
        List of node mappings {pattern_node: target_node}
    """
    G_target = target.to_networkx()
    G_pattern = pattern.to_networkx()

    matcher = isomorphism.GraphMatcher(G_target, G_pattern)

    matches = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        # mapping is {target_node: pattern_node}, we want the inverse
        inverse_mapping = {v: k for k, v in mapping.items()}
        matches.append(inverse_mapping)
        if len(matches) >= max_matches:
            break

    return matches


def find_edge_mapping(
    target: Graph,
    pattern: Graph,
    node_mapping: Dict[int, int]
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Given a node mapping, compute the corresponding edge mapping.

    Args:
        target: Target graph
        pattern: Pattern graph
        node_mapping: {pattern_node: target_node}

    Returns:
        {pattern_edge: target_edge} mapping
    """
    edge_mapping = {}

    for p_u, p_v in pattern.edges:
        t_u = node_mapping[p_u]
        t_v = node_mapping[p_v]
        target_edge = (min(t_u, t_v), max(t_u, t_v))

        if target_edge in target.edges:
            pattern_edge = (min(p_u, p_v), max(p_u, p_v))
            edge_mapping[pattern_edge] = target_edge

    return edge_mapping


# =============================================================================
# DISJOINT COVER ALGORITHM
# =============================================================================

def find_disjoint_cover(
    graph: Graph,
    minor: MinorEntry,
    table: RainbowTable,
    max_depth: int = 5
) -> Cover:
    """Greedily tile graph with disjoint copies of minor.

    Algorithm:
    1. Find all occurrences of minor in graph (VF2 subgraph isomorphism)
    2. Greedily select non-overlapping occurrences (largest coverage first)
    3. For uncovered edges, recursively tile with smaller minors
    4. Base case: tile remaining edges with K_2

    Args:
        graph: Target graph to cover
        minor: Minor to use as primary tile
        table: Rainbow table for finding smaller minors
        max_depth: Maximum recursion depth

    Returns:
        Cover with all tiles and any uncovered edges
    """
    cover = Cover()
    cover.uncovered_edges = set(graph.edges)

    # Base case: no edges or no recursion budget
    if not graph.edges or max_depth <= 0:
        return cover

    # Build pattern graph from minor
    pattern = _minor_to_graph(minor)
    if pattern is None:
        return cover

    # Skip if pattern is larger than graph
    if pattern.edge_count() > graph.edge_count():
        return cover

    # Find all occurrences
    matches = find_subgraph_isomorphisms(graph, pattern, max_matches=50)

    # Sort by coverage (prefer matches that cover more uncovered edges)
    def coverage_score(mapping):
        edge_mapping = find_edge_mapping(graph, pattern, mapping)
        return len(set(edge_mapping.values()) & cover.uncovered_edges)

    matches.sort(key=coverage_score, reverse=True)

    # Greedily add non-overlapping tiles
    for node_mapping in matches:
        edge_mapping = find_edge_mapping(graph, pattern, node_mapping)

        tile = Tile(
            minor=minor,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping
        )

        # Only add if covers at least one uncovered edge
        if tile.covered_edges & cover.uncovered_edges:
            if cover.add_tile(tile):
                # Update uncovered edges
                cover.uncovered_edges -= tile.covered_edges

    # If still have uncovered edges, try smaller minors (with reduced depth)
    if cover.uncovered_edges and table is not None and max_depth > 1:
        remaining_graph = graph.edge_induced_subgraph(cover.uncovered_edges)

        # Only try if remaining graph is smaller
        if remaining_graph.edge_count() < graph.edge_count():
            smaller_minors = table.find_minors_of(remaining_graph)

            tried_keys = {minor.canonical_key}
            for smaller in smaller_minors:
                if smaller.canonical_key in tried_keys:
                    continue
                tried_keys.add(smaller.canonical_key)

                # Skip if smaller is same size or larger than remaining
                if smaller.edge_count >= remaining_graph.edge_count():
                    continue

                # Recursively cover remaining edges
                sub_cover = find_disjoint_cover(
                    remaining_graph, smaller, table, max_depth - 1
                )
                for tile in sub_cover.tiles:
                    if cover.add_tile(tile):
                        cover.uncovered_edges -= tile.covered_edges

                if cover.is_complete():
                    break

    return cover


def _minor_to_graph(minor: MinorEntry) -> Optional[Graph]:
    """Reconstruct a graph from a minor entry.

    This is a simple reconstruction for common graph types.
    For complex graphs, returns None (cannot reconstruct).
    """
    from .graph import complete_graph, cycle_graph, path_graph, star_graph, wheel_graph

    name = minor.name

    # Complete graphs
    if name.startswith('K_'):
        try:
            n = int(name[2:])
            return complete_graph(n)
        except ValueError:
            pass

    # Cycle graphs
    if name.startswith('C_'):
        try:
            n = int(name[2:])
            return cycle_graph(n)
        except ValueError:
            pass

    # Path graphs
    if name.startswith('P_'):
        try:
            n = int(name[2:])
            return path_graph(n)
        except ValueError:
            pass

    # Wheel graphs
    if name.startswith('W_'):
        try:
            n = int(name[2:])
            return wheel_graph(n)
        except ValueError:
            pass

    # Star graphs
    if name.startswith('S_'):
        try:
            n = int(name[2:])
            return star_graph(n)
        except ValueError:
            pass

    return None


# =============================================================================
# FRINGE COMPUTATION
# =============================================================================

def compute_fringe(cover: Cover, input_graph: Graph) -> Fringe:
    """Compute over-coverage: edges in tiled cover not in input.

    For each tile, check which of its minor's edges don't exist
    in the input graph at the mapped positions.

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Fringe containing edges in cover but not in input
    """
    fringe = Fringe()

    input_edges = set(input_graph.edges)

    for tile in cover.tiles:
        # Get the minor's graph structure
        pattern = _minor_to_graph(tile.minor)
        if pattern is None:
            continue

        # Check each edge in the minor
        for p_u, p_v in pattern.edges:
            # Map to target graph positions
            if p_u not in tile.node_mapping or p_v not in tile.node_mapping:
                continue

            t_u = tile.node_mapping[p_u]
            t_v = tile.node_mapping[p_v]
            target_edge = (min(t_u, t_v), max(t_u, t_v))

            # If this edge doesn't exist in input, it's fringe
            if target_edge not in input_edges:
                fringe.edges.add(target_edge)
                fringe.nodes.add(t_u)
                fringe.nodes.add(t_v)

    return fringe


def compute_inter_tile_edges(cover: Cover, input_graph: Graph) -> Set[Tuple[int, int]]:
    """Find edges in input graph that connect nodes from different tiles.

    These are the edges where we need to apply k-join formulas.

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Set of edges connecting different tiles
    """
    # Build node -> tile index mapping
    node_to_tile = {}
    for i, tile in enumerate(cover.tiles):
        for node in tile.covered_nodes:
            node_to_tile[node] = i

    inter_edges = set()

    for u, v in input_graph.edges:
        tile_u = node_to_tile.get(u, -1)
        tile_v = node_to_tile.get(v, -1)

        # Edge connects different tiles (or uncovered node)
        if tile_u != tile_v:
            inter_edges.add((u, v))

    return inter_edges


def analyze_tile_connections(
    cover: Cover,
    input_graph: Graph
) -> Dict[Tuple[int, int], Dict]:
    """Analyze how tiles are connected in the input graph.

    For each pair of tiles, determine:
    - Number of edges connecting them
    - Shared vertices (if any)
    - Type of connection (k-join type)

    Args:
        cover: The computed cover
        input_graph: The original input graph

    Returns:
        Dict mapping (tile_i, tile_j) to connection info
    """
    # Build node -> tile index mapping
    node_to_tile = {}
    for i, tile in enumerate(cover.tiles):
        for node in tile.covered_nodes:
            node_to_tile[node] = i

    connections = {}
    n_tiles = len(cover.tiles)

    for i in range(n_tiles):
        for j in range(i + 1, n_tiles):
            nodes_i = cover.tiles[i].covered_nodes
            nodes_j = cover.tiles[j].covered_nodes

            # Find shared vertices (shouldn't exist for disjoint cover)
            shared = nodes_i & nodes_j

            # Find edges connecting the tiles
            connecting_edges = []
            for u, v in input_graph.edges:
                u_tile = node_to_tile.get(u, -1)
                v_tile = node_to_tile.get(v, -1)
                if (u_tile == i and v_tile == j) or (u_tile == j and v_tile == i):
                    connecting_edges.append((u, v))

            if connecting_edges or shared:
                connections[(i, j)] = {
                    'shared_vertices': len(shared),
                    'connecting_edges': len(connecting_edges),
                    'edges': connecting_edges,
                    'k_join_type': _determine_k_join_type(len(shared), len(connecting_edges))
                }

    return connections


def _determine_k_join_type(shared_vertices: int, connecting_edges: int) -> str:
    """Determine the type of k-join based on connection structure."""
    if shared_vertices == 0:
        if connecting_edges == 0:
            return "disjoint"
        elif connecting_edges == 1:
            return "bridge"
        else:
            return "multi_edge"
    elif shared_vertices == 1:
        return "1_join"  # Cut vertex
    elif shared_vertices == 2:
        return "2_join"  # Edge identification
    else:
        return f"{shared_vertices}_join"  # k-clique join
