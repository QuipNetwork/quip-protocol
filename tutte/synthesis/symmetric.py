"""Symmetric chord ordering for 2-cell hierarchical synthesis.

When a graph decomposes into 2 isomorphic cells with an automorphism σ that
preserves inter-cell edges, we can pair up chords by symmetry. Processing
symmetric pairs consecutively maximizes cache hits at deeper recursion levels,
since the merged graphs from partner chords are structurally similar.

For a symmetric pair (e₁, e₂) at symmetric state G:
  T(G + e₁ + e₂) = T(G) + T(G/{e₁}) + T((G+e₁)/{e₂})

Key properties:
- T(G/{e₁}) = T(G/{e₂}) when G is symmetric (same canonical key → automatic cache hit)
- T(G/{e₁}) and T((G+e₁)/{e₂}) are independent → can be parallelized
- After adding both chords, the graph is symmetric again for the next pair
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..graph import Graph, MultiGraph
from ..graphs.treewidth import compute_best_tree_decomposition, estimate_dp_cost


def find_cell_automorphism(
    graph: Graph,
    partition: List[Set[int]],
) -> Optional[Dict[int, int]]:
    """Find an automorphism σ: Cell₀ → Cell₁ that preserves inter-cell edges.

    Args:
        graph: Full graph
        partition: List of exactly 2 cell node sets

    Returns:
        Mapping dict (cell0_node → cell1_node) if found, None otherwise
    """
    if len(partition) != 2:
        return None

    cell0, cell1 = partition[0], partition[1]

    # Build NX subgraphs for isomorphism search
    nxg = nx.Graph()
    for u, v in graph.edges:
        nxg.add_edge(u, v)

    g0 = nxg.subgraph(cell0).copy()
    g1 = nxg.subgraph(cell1).copy()

    # Collect inter-cell edges (normalized: cell0 node first)
    inter_edges = set()
    for u, v in graph.edges:
        u_in_0 = u in cell0
        v_in_0 = v in cell0
        if u_in_0 != v_in_0:
            if u_in_0:
                inter_edges.add((u, v))
            else:
                inter_edges.add((v, u))

    # Search for isomorphism that preserves inter-cell edges
    matcher = nx.isomorphism.GraphMatcher(g0, g1)
    best_mapping = None
    best_preserved = 0

    for mapping in matcher.isomorphisms_iter():
        inv_mapping = {v: k for k, v in mapping.items()}

        preserved = 0
        for u, v in inter_edges:
            # u ∈ cell0, v ∈ cell1
            u_mapped = mapping.get(u)
            v_mapped = inv_mapping.get(v)
            if u_mapped is not None and v_mapped is not None:
                if (v_mapped, u_mapped) in inter_edges:
                    preserved += 1

        if preserved == len(inter_edges):
            return mapping  # Perfect match — return immediately
        if preserved > best_preserved:
            best_preserved = preserved
            best_mapping = mapping

    # Only return if ALL inter-cell edges are preserved
    if best_preserved == len(inter_edges):
        return best_mapping
    return None


def pair_chords_by_symmetry(
    chords: List[Tuple[int, int]],
    automorphism: Dict[int, int],
    partition: List[Set[int]],
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[int, int]]]:
    """Pair up chords by the cell automorphism.

    For each chord (u, v) with u ∈ cell0, v ∈ cell1, its partner is
    (σ⁻¹(v), σ(u)) where σ is the automorphism cell0 → cell1.

    Args:
        chords: List of inter-cell chord edges
        automorphism: Mapping cell0_node → cell1_node
        partition: List of exactly 2 cell node sets

    Returns:
        Tuple of (paired_chords, unpaired_chords) where paired_chords
        is a list of (chord_a, chord_b) tuples.
    """
    cell0, cell1 = partition[0], partition[1]
    inv_automorphism = {v: k for k, v in automorphism.items()}

    # Normalize chords: cell0 node first
    normalized = []
    for u, v in chords:
        if u in cell0:
            normalized.append((u, v))
        else:
            normalized.append((v, u))

    # Build partner mapping
    chord_set = set(chords)
    used = set()
    pairs = []
    unpaired = []

    for u, v in normalized:
        orig_edge = (min(u, v), max(u, v))
        if orig_edge in used:
            continue

        # Partner: (σ⁻¹(v), σ(u))
        partner_u = inv_automorphism.get(v)
        partner_v = automorphism.get(u)

        if partner_u is not None and partner_v is not None:
            partner_edge = (min(partner_u, partner_v), max(partner_u, partner_v))

            if partner_edge in chord_set and partner_edge != orig_edge and partner_edge not in used:
                pairs.append((orig_edge, partner_edge))
                used.add(orig_edge)
                used.add(partner_edge)
                continue

        if orig_edge not in used:
            unpaired.append(orig_edge)
            used.add(orig_edge)

    return pairs, unpaired


def build_symmetric_chord_order(
    chords: List[Tuple[int, int]],
    graph: Graph,
    partition: List[Set[int]],
) -> Tuple[List[Tuple[int, int]], Optional[Dict[int, int]]]:
    """Reorder chords for maximum cache benefit using cell automorphism.

    Returns chords ordered as: [pair1_a, pair1_b, pair2_a, pair2_b, ..., unpaired...]
    so that symmetric partners are consecutive.

    Args:
        chords: Original chord list
        graph: Full graph
        partition: List of cell node sets

    Returns:
        Tuple of (ordered_chords, automorphism_or_None)
    """
    if len(partition) != 2:
        return chords, None

    automorphism = find_cell_automorphism(graph, partition)
    if automorphism is None:
        return chords, None

    pairs, unpaired = pair_chords_by_symmetry(chords, automorphism, partition)

    # Interleave: pair_a, pair_b, pair_a, pair_b, ...
    ordered = []
    for a, b in pairs:
        ordered.append(a)
        ordered.append(b)
    ordered.extend(unpaired)

    return ordered, automorphism


def build_treewidth_minimizing_chord_order(
    chords: List[Tuple[int, int]],
    current_mg: MultiGraph,
    max_width: int = 9,
) -> List[Tuple[int, int]]:
    """Greedily select next chord to minimize merged graph treewidth.

    For each remaining chord, compute the treewidth and DP cost of the
    merged graph G/{u,v}. Pick the chord with lowest treewidth (then
    lowest cost as tiebreaker). This steers the chord ordering away
    from high-treewidth merged graphs that are expensive to synthesize.

    Overhead: O(|chords|^2) tree decomposition calls, ~20ms each.

    Args:
        chords: List of chord edges to reorder
        current_mg: Current running multigraph (with intra-cell + bridge edges)
        max_width: Maximum treewidth to attempt

    Returns:
        Reordered chord list minimizing merged graph treewidth
    """
    remaining = list(chords)
    ordered: List[Tuple[int, int]] = []
    mg = current_mg

    while remaining:
        best_chord = None
        best_tw = float('inf')
        best_cost = float('inf')

        for chord in remaining:
            u, v = chord
            merged = mg.merge_nodes(u, v)
            # Strip loops for treewidth check
            if merged.total_loop_count() > 0:
                merged = merged.remove_loops()
            td = compute_best_tree_decomposition(merged, max_width=max_width)
            if td is None:
                tw, cost = max_width + 1, float('inf')
            else:
                tw = td.width
                cost = estimate_dp_cost(td)
            if tw < best_tw or (tw == best_tw and cost < best_cost):
                best_tw = tw
                best_cost = cost
                best_chord = chord

        ordered.append(best_chord)
        remaining.remove(best_chord)

        # Update running multigraph with the selected chord
        u, v = best_chord
        edge = (min(u, v), max(u, v))
        ec = dict(mg.edge_counts)
        ec[edge] = ec.get(edge, 0) + 1
        mg = MultiGraph(nodes=mg.nodes, edge_counts=ec, loop_counts=mg.loop_counts)

    return ordered
