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
from typing import Dict, Set, Tuple, List, Optional, Iterator, FrozenSet

import networkx as nx
from networkx.algorithms import isomorphism

from ..graph import (
    Graph, CellSignature, NodeSignature,
    compute_signature, compute_node_signature, compute_all_node_signatures
)
from ..lookup.core import MinorEntry, RainbowTable


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

    First checks if the entry has a stored graph. Otherwise falls back
    to name-based reconstruction for common graph types.
    """
    # Use stored graph if available
    if minor.graph is not None:
        return minor.graph

    from ..graph import complete_graph, cycle_graph, path_graph, star_graph, wheel_graph

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

    # Zephyr graphs: Z(m,t)
    if name.startswith('Z(') and ',' in name:
        try:
            import dwave_networkx as dnx
            # Parse "Z(m,t)" format
            inner = name[2:-1]  # Remove "Z(" and ")"
            m, t = inner.split(',')
            m, t = int(m.strip()), int(t.strip())
            G = dnx.zephyr_graph(m, t)
            return Graph.from_networkx(G)
        except (ValueError, ImportError):
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


# =============================================================================
# HIERARCHICAL TILING (for graphs with repeating cell structure)
# =============================================================================

@dataclass
class InterCellInfo:
    """Information about edges connecting different cells."""
    edges: List[Tuple[int, int]]
    is_regular: bool  # True if same # edges between each cell pair
    edges_per_pair: int  # Number of edges between each adjacent cell pair
    cell_adjacencies: List[Tuple[int, int]]  # Which cells are adjacent


def find_cell_candidates(
    graph: Graph,
    table: RainbowTable,
    min_cells: int = 2
) -> List[MinorEntry]:
    """Find rainbow table entries that could tile the graph.

    Uses arithmetic filters to quickly eliminate candidates without
    running expensive VF2 subgraph isomorphism.

    Args:
        graph: Target graph to tile
        table: Rainbow table with potential tiles
        min_cells: Minimum number of cells required

    Returns:
        List of candidate entries sorted by edge count (descending)
    """
    target_nodes = graph.node_count()
    target_edges = graph.edge_count()
    target_sig = compute_signature(graph)

    candidates = []

    for entry in table.entries.values():
        # Skip entries without stored graphs (can't use as tiles)
        if entry.graph is None:
            # Try to reconstruct from name
            pattern = _minor_to_graph(entry)
            if pattern is None:
                continue
        else:
            pattern = entry.graph

        cell_nodes = entry.node_count
        cell_edges = entry.edge_count

        # Filter 1: node count must divide evenly
        if cell_nodes <= 0 or target_nodes % cell_nodes != 0:
            continue

        k = target_nodes // cell_nodes
        if k < min_cells:
            continue

        # Filter 2: edge count consistency
        # k disjoint cells have k * cell_edges edges
        # Inter-cell edges add to this
        cell_total_edges = k * cell_edges
        inter_cell_edges = target_edges - cell_total_edges

        if inter_cell_edges < 0:
            continue  # Would need negative inter-cell edges

        # Filter 3: degree sequence compatibility
        # Each cell contributes its degree sequence
        # Inter-cell edges add 2 to the degree sum
        cell_sig = compute_signature(pattern)
        cell_degree_sum = sum(cell_sig.degree_sequence)
        target_degree_sum = sum(target_sig.degree_sequence)

        # With k cells and inter_cell_edges between them:
        # target_degree_sum = k * cell_degree_sum + 2 * inter_cell_edges
        expected_degree_sum = k * cell_degree_sum + 2 * inter_cell_edges
        if target_degree_sum != expected_degree_sum:
            continue

        candidates.append(entry)

    # Sort by edge count descending (prefer larger tiles)
    return sorted(candidates, key=lambda e: e.edge_count, reverse=True)


def partition_into_cells(
    graph: Graph,
    cell: MinorEntry,
    k: int
) -> Optional[List[Set[int]]]:
    """Partition graph nodes into k groups that look like the cell.

    This is the main entry point for partitioning. It tries multiple
    strategies in order of sophistication:
    1. Disconnected components (trivial case)
    2. Node signature matching (works when cells are disjoint)
    3. Structural clustering (works when cells share edges)

    Args:
        graph: Target graph to partition
        cell: Cell pattern from rainbow table
        k: Expected number of cells

    Returns:
        List of node sets (one per cell) or None if partitioning fails
    """
    cell_graph = cell.graph if cell.graph is not None else _minor_to_graph(cell)
    if cell_graph is None:
        return None

    cell_size = cell.node_count

    if graph.node_count() != k * cell_size:
        return None

    # Strategy 1: Disconnected components
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            if all(comp.node_count() == cell_size for comp in components):
                return [set(comp.nodes) for comp in components]

    # Strategy 2: Node signature matching (for disjoint cells)
    target_sigs = compute_all_node_signatures(graph)
    cell_sigs = compute_all_node_signatures(cell_graph)

    partitions = _greedy_partition(graph, cell_graph, k, target_sigs, cell_sigs)
    if partitions is not None:
        # Verify the partition is actually valid before returning
        if _verify_partition_structure(graph, partitions, cell_graph):
            return partitions

    # Strategy 3: VF2-based structural matching for connected cells
    # This handles cases where inter-cell edges change node signatures
    partitions = _partition_by_structure(graph, cell_graph, k)

    return partitions


def _verify_partition_structure(
    graph: Graph,
    partition: List[Set[int]],
    cell_graph: Graph
) -> bool:
    """Quick verification that partition cells are isomorphic to pattern."""
    cell_edges = cell_graph.edge_count()

    for cell_nodes in partition:
        subgraph = graph.subgraph(cell_nodes)
        # Check edge count first (fast)
        if subgraph.edge_count() != cell_edges:
            return False

    # All cells have correct edge count - check one for isomorphism
    if partition:
        subgraph = graph.subgraph(partition[0])
        G1 = subgraph.to_networkx()
        G2 = cell_graph.to_networkx()
        if not nx.is_isomorphic(G1, G2):
            return False

    return True


def _partition_by_structure(
    graph: Graph,
    cell_graph: Graph,
    k: int
) -> Optional[List[Set[int]]]:
    """Partition using VF2 to find disjoint isomorphic copies.

    When cells share edges, node signatures change. We need to find
    groups where the *induced subgraph* is isomorphic to the cell.

    Strategy:
    1. Find all subgraph isomorphisms using VF2 (on small cells, this is fast)
    2. Find k disjoint copies that cover all nodes
    3. Return the partition
    """
    cell_size = cell_graph.node_count()
    total_nodes = graph.node_count()

    if total_nodes != k * cell_size:
        return None

    # Use VF2 to find all isomorphic copies of cell in graph
    G_target = graph.to_networkx()
    G_pattern = cell_graph.to_networkx()

    matcher = isomorphism.GraphMatcher(G_target, G_pattern)

    # Collect all matches as sets of nodes
    all_matches: List[Set[int]] = []
    seen_node_sets: Set[FrozenSet[int]] = set()

    for mapping in matcher.subgraph_isomorphisms_iter():
        nodes = frozenset(mapping.keys())
        if nodes not in seen_node_sets:
            seen_node_sets.add(nodes)
            all_matches.append(set(nodes))

        # Limit search for large graphs
        if len(all_matches) > 1000:
            break

    if len(all_matches) < k:
        return None

    # Find k disjoint matches that cover all nodes
    partition = _find_disjoint_partition(all_matches, k, total_nodes)

    return partition


def _find_disjoint_partition(
    matches: List[Set[int]],
    k: int,
    total_nodes: int
) -> Optional[List[Set[int]]]:
    """Find k disjoint matches that cover all nodes.

    Uses backtracking search to find a valid partition.
    """
    if k == 0:
        return []

    if not matches:
        return None

    # Sort matches by node indices for deterministic behavior
    matches = sorted(matches, key=lambda m: tuple(sorted(m)))

    def backtrack(
        index: int,
        used: Set[int],
        partition: List[Set[int]]
    ) -> Optional[List[Set[int]]]:
        if len(partition) == k:
            if len(used) == total_nodes:
                return partition.copy()
            return None

        for i in range(index, len(matches)):
            match = matches[i]

            # Check if this match is disjoint from already used nodes
            if not (match & used):
                partition.append(match)
                new_used = used | match

                result = backtrack(i + 1, new_used, partition)
                if result is not None:
                    return result

                partition.pop()

        return None

    return backtrack(0, set(), [])


def _grow_by_edge_density(
    graph: Graph,
    start: int,
    cell_size: int,
    cell_edges: int,
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a cell by maximizing internal edge density.

    Greedily add nodes that maximize edges within the growing cell.
    """
    cell_nodes = {start}
    all_edges = graph.edges

    while len(cell_nodes) < cell_size:
        best_node = None
        best_score = -1

        # Consider all neighbors of current cell
        frontier = set()
        for node in cell_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in cell_nodes and neighbor not in used_nodes:
                    frontier.add(neighbor)

        if not frontier:
            return None

        for candidate in frontier:
            # Count edges from candidate to current cell
            edges_to_cell = sum(1 for n in cell_nodes
                               if (min(candidate, n), max(candidate, n)) in all_edges)

            if edges_to_cell > best_score:
                best_score = edges_to_cell
                best_node = candidate

        if best_node is None:
            return None

        cell_nodes.add(best_node)

    return cell_nodes


def _greedy_partition(
    graph: Graph,
    cell_graph: Graph,
    k: int,
    target_sigs: Dict[int, NodeSignature],
    cell_sigs: Dict[int, NodeSignature]
) -> Optional[List[Set[int]]]:
    """Greedily partition graph into k cell-shaped groups.

    Strategy:
    1. Find anchor nodes (distinct signature in cell)
    2. Grow cells from anchors by signature matching
    3. Verify each cell is isomorphic to pattern

    For disconnected graphs, use connected components as natural partitions.
    """
    cell_size = len(cell_sigs)

    # Special case: disconnected graph
    # Use connected components as natural cell boundaries
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            # Check if each component has the right size
            valid = all(comp.node_count() == cell_size for comp in components)
            if valid:
                return [set(comp.nodes) for comp in components]

    # Find nodes with unique signatures in cell (good anchors)
    sig_counts: Dict[Tuple[int, int, int], int] = {}
    for sig in cell_sigs.values():
        key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
        sig_counts[key] = sig_counts.get(key, 0) + 1

    unique_sig_keys = [k for k, v in sig_counts.items() if v == 1]

    if not unique_sig_keys:
        # No unique signatures - all nodes look the same
        # For regular graphs, use connectivity-based partitioning
        # Try to find k disjoint subgraphs of the right size
        return _partition_by_connectivity(graph, cell_size, k)

    anchor_sig_key = unique_sig_keys[0]

    # Find all nodes in target with this signature
    anchor_candidates = [
        n for n, sig in target_sigs.items()
        if (sig.degree, sig.neighbor_degree_sum, sig.triangles) == anchor_sig_key
    ]

    if len(anchor_candidates) != k:
        return None  # Wrong number of anchors

    partitions: List[Set[int]] = []
    used_nodes: Set[int] = set()

    for anchor in anchor_candidates:
        if anchor in used_nodes:
            continue

        # Grow a cell from this anchor using BFS with signature matching
        cell_nodes = _grow_cell(graph, anchor, cell_size, target_sigs, cell_sigs, used_nodes)

        if cell_nodes is None or len(cell_nodes) != cell_size:
            # Try alternative approach: just grow connected component of size cell_size
            cell_nodes = _grow_connected_cell(graph, anchor, cell_size, used_nodes)

        if cell_nodes is None or len(cell_nodes) != cell_size:
            return None

        partitions.append(cell_nodes)
        used_nodes.update(cell_nodes)

    if len(partitions) != k:
        return None

    return partitions


def _partition_by_connectivity(
    graph: Graph,
    cell_size: int,
    k: int
) -> Optional[List[Set[int]]]:
    """Partition graph into k groups of cell_size using connectivity.

    For graphs where all nodes have identical signatures, use
    connected components or greedy growth.
    """
    # First check if graph has natural connected components
    if not graph.is_connected():
        components = graph.connected_components()
        if len(components) == k:
            valid = all(comp.node_count() == cell_size for comp in components)
            if valid:
                return [set(comp.nodes) for comp in components]

    # For connected graphs, use greedy growth from distributed starting points
    # Pick k starting nodes that are maximally spread out
    nodes_list = list(graph.nodes)
    if len(nodes_list) != k * cell_size:
        return None

    partitions: List[Set[int]] = []
    used_nodes: Set[int] = set()

    # Start from first unused node and grow cells
    for start_candidate in nodes_list:
        if start_candidate in used_nodes:
            continue

        cell_nodes = _grow_connected_cell(graph, start_candidate, cell_size, used_nodes)
        if cell_nodes is None or len(cell_nodes) != cell_size:
            return None

        partitions.append(cell_nodes)
        used_nodes.update(cell_nodes)

        if len(partitions) == k:
            break

    if len(partitions) != k or len(used_nodes) != k * cell_size:
        return None

    return partitions


def _grow_cell(
    graph: Graph,
    anchor: int,
    cell_size: int,
    target_sigs: Dict[int, NodeSignature],
    cell_sigs: Dict[int, NodeSignature],
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a cell from anchor by matching node signatures."""
    # Get required signature distribution
    sig_needed: Dict[Tuple[int, int, int], int] = {}
    for sig in cell_sigs.values():
        key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
        sig_needed[key] = sig_needed.get(key, 0) + 1

    cell_nodes = {anchor}
    anchor_sig = target_sigs[anchor]
    anchor_key = (anchor_sig.degree, anchor_sig.neighbor_degree_sum, anchor_sig.triangles)
    sig_needed[anchor_key] -= 1

    # BFS from anchor
    frontier = list(graph.neighbors(anchor) - used_nodes)

    while len(cell_nodes) < cell_size and frontier:
        best_node = None
        best_score = float('inf')

        for node in frontier:
            if node in cell_nodes or node in used_nodes:
                continue

            sig = target_sigs[node]
            sig_key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)

            if sig_needed.get(sig_key, 0) > 0:
                # This signature is still needed
                # Score by how many cell neighbors it has
                cell_neighbors = len(graph.neighbors(node) & cell_nodes)
                score = -cell_neighbors  # More neighbors = better
                if score < best_score:
                    best_score = score
                    best_node = node

        if best_node is None:
            # No matching node in frontier, expand frontier
            new_frontier = []
            for node in cell_nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor not in cell_nodes and neighbor not in used_nodes:
                        new_frontier.append(neighbor)
            if not new_frontier:
                break
            frontier = new_frontier
        else:
            cell_nodes.add(best_node)
            sig = target_sigs[best_node]
            sig_key = (sig.degree, sig.neighbor_degree_sum, sig.triangles)
            sig_needed[sig_key] -= 1

            # Expand frontier
            for neighbor in graph.neighbors(best_node):
                if neighbor not in cell_nodes and neighbor not in used_nodes:
                    if neighbor not in frontier:
                        frontier.append(neighbor)

    if len(cell_nodes) == cell_size:
        return cell_nodes
    return None


def _grow_connected_cell(
    graph: Graph,
    start: int,
    cell_size: int,
    used_nodes: Set[int]
) -> Optional[Set[int]]:
    """Grow a connected component of exactly cell_size nodes from start."""
    cell_nodes = {start}
    frontier = [n for n in graph.neighbors(start) if n not in used_nodes]

    while len(cell_nodes) < cell_size and frontier:
        # Pick node with most connections to current cell
        best = max(frontier, key=lambda n: len(graph.neighbors(n) & cell_nodes))
        cell_nodes.add(best)
        frontier.remove(best)

        for neighbor in graph.neighbors(best):
            if neighbor not in cell_nodes and neighbor not in used_nodes and neighbor not in frontier:
                frontier.append(neighbor)

    if len(cell_nodes) == cell_size:
        return cell_nodes
    return None


def verify_cell_partition(
    graph: Graph,
    partition: List[Set[int]],
    cell: MinorEntry
) -> bool:
    """Verify each partition element is isomorphic to the cell.

    Uses signature pre-checks before VF2 for speed.
    VF2 on small cell-sized graphs is fast!

    Args:
        graph: The full graph
        partition: List of node sets (one per cell)
        cell: Cell pattern from rainbow table

    Returns:
        True if all partitions are isomorphic to cell
    """
    cell_graph = cell.graph if cell.graph is not None else _minor_to_graph(cell)
    if cell_graph is None:
        return False

    cell_sig = compute_signature(cell_graph)

    for cell_nodes in partition:
        subgraph = graph.subgraph(cell_nodes)

        # Quick signature check first
        sub_sig = compute_signature(subgraph)
        if not cell_sig.could_match(sub_sig):
            return False

        # Edge count check (faster than VF2)
        if subgraph.edge_count() != cell.edge_count:
            return False

        # VF2 isomorphism check on small graph (fast!)
        G1 = subgraph.to_networkx()
        G2 = cell_graph.to_networkx()

        if not nx.is_isomorphic(G1, G2):
            return False

    return True


def analyze_inter_cell_edges(
    graph: Graph,
    partition: List[Set[int]]
) -> InterCellInfo:
    """Analyze edges between cells in a partition.

    Determines:
    - Which edges connect different cells
    - Whether the connection pattern is regular
    - How many edges exist between each cell pair

    Args:
        graph: The full graph
        partition: List of node sets (one per cell)

    Returns:
        InterCellInfo with edge analysis
    """
    k = len(partition)

    # Build node-to-cell mapping
    node_to_cell: Dict[int, int] = {}
    for i, cell_nodes in enumerate(partition):
        for node in cell_nodes:
            node_to_cell[node] = i

    # Find all inter-cell edges
    inter_edges: List[Tuple[int, int]] = []
    cell_pair_edges: Dict[Tuple[int, int], int] = {}

    for u, v in graph.edges:
        cell_u = node_to_cell.get(u, -1)
        cell_v = node_to_cell.get(v, -1)

        if cell_u != cell_v and cell_u >= 0 and cell_v >= 0:
            inter_edges.append((u, v))
            pair = (min(cell_u, cell_v), max(cell_u, cell_v))
            cell_pair_edges[pair] = cell_pair_edges.get(pair, 0) + 1

    # Check if pattern is regular
    edge_counts = list(cell_pair_edges.values())
    is_regular = len(set(edge_counts)) <= 1 if edge_counts else True
    edges_per_pair = edge_counts[0] if edge_counts else 0

    # Find adjacent cell pairs
    cell_adjacencies = list(cell_pair_edges.keys())

    return InterCellInfo(
        edges=inter_edges,
        is_regular=is_regular,
        edges_per_pair=edges_per_pair,
        cell_adjacencies=cell_adjacencies
    )


def try_hierarchical_partition(
    graph: Graph,
    table: RainbowTable
) -> Optional[Tuple[MinorEntry, List[Set[int]], InterCellInfo]]:
    """Try to partition graph into cells from the rainbow table.

    This is the main entry point for hierarchical tiling. It:
    1. Finds candidate cells that could tile the graph
    2. Tries to partition nodes into cell groups
    3. Verifies each group is isomorphic to the cell
    4. Analyzes inter-cell edges

    Args:
        graph: Graph to partition
        table: Rainbow table with potential tiles

    Returns:
        (cell_entry, partition, inter_cell_info) or None if no tiling found
    """
    # Find candidate tiles
    candidates = find_cell_candidates(graph, table)

    for cell in candidates:
        cell_size = cell.node_count
        k = graph.node_count() // cell_size

        # Try to partition
        partition = partition_into_cells(graph, cell, k)
        if partition is None:
            continue

        # Verify partition
        if not verify_cell_partition(graph, partition, cell):
            continue

        # Analyze inter-cell edges
        inter_info = analyze_inter_cell_edges(graph, partition)

        return (cell, partition, inter_info)

    return None
