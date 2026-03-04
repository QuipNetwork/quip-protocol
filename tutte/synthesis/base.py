"""Base classes for Tutte polynomial synthesis.

Provides shared infrastructure used by all synthesis engines:
- UnionFind for bridge/chord classification
- BaseMultigraphSynthesizer with pattern recognition for multigraphs
- SynthesisResult dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..polynomial import TuttePolynomial
from ..graph import Graph, MultiGraph


# =============================================================================
# UNION-FIND (for bridge/chord classification)
# =============================================================================

class UnionFind:
    """O(alpha(n)) union-find for bridge/chord classification."""

    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank. Returns True if x,y were in different sets."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# =============================================================================
# BASE MULTIGRAPH SYNTHESIZER
# =============================================================================

class BaseMultigraphSynthesizer:
    """Base class for multigraph synthesis with pattern recognition.

    This class provides the shared multigraph synthesis logic used by both
    SynthesisEngine and HybridSynthesisEngine. Subclasses must implement:
    - synthesize(graph) -> result with polynomial attribute
    - _log(msg) -> logging function
    - _multigraph_cache -> dict for caching multigraph polynomials
    """

    def _synthesize_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int = 10,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Synthesize polynomial for a multigraph with pattern recognition.

        Cheap structural checks (O(n+m)) are done BEFORE canonical_key (O(n²))
        to avoid expensive hashing on graphs that can be factored directly.

        Pattern recognition order:
        1. Handle loops: T(G with loop) = y × T(G without loop)
        2. Parallel edges formula (for simple 2-node multi-edge graphs)
        3. Disconnected: T(G1 ∪ G2) = T(G1) × T(G2)
        4. Cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)
        5. Cache lookup (requires canonical_key)
        6. Simple graph -> use regular synthesis
        7. Reduce parallel edges: T(G) = T(G\\e) + T(G/e)

        Args:
            mg: MultiGraph to synthesize
            max_depth: Maximum recursion depth (for subclass methods)
            skip_minor_search: If True, skip expensive minor search for simple graphs

        Returns:
            TuttePolynomial for the multigraph
        """
        # 1. Handle loops first: T(G with loop) = y × T(G without loop)
        if mg.total_loop_count() > 0:
            loop_count = mg.total_loop_count()
            mg_no_loops = mg.remove_loops()
            return TuttePolynomial.y(loop_count) * self._synthesize_multigraph(mg_no_loops, max_depth, skip_minor_search)

        # 2. Parallel edges formula (very cheap check)
        if mg.is_just_parallel_edges():
            return self._parallel_edges_formula(mg.parallel_edge_count())

        # 3. Disconnected: T(G1 ∪ G2) = T(G1) × T(G2)
        if not mg.is_connected():
            return self._handle_disconnected_multigraph(mg, max_depth, skip_minor_search)

        # 4. Cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)
        #    O(n+m) check, avoids expensive canonical_key for factorable graphs
        cut = mg.has_cut_vertex()
        if cut is not None:
            components = mg.split_at_cut_vertex(cut)
            if len(components) > 1:
                self._log(f"Cut vertex {cut} splits into {len(components)} components")
                poly = TuttePolynomial.one()
                for comp in components:
                    poly = poly * self._synthesize_multigraph(comp, max_depth, skip_minor_search)
                return poly

        # 5. Cache lookup (requires canonical_key - expensive but needed for non-factorable graphs)
        cache_key = mg.canonical_key()
        if cache_key in self._multigraph_cache:
            return self._multigraph_cache[cache_key]

        self._log(f"Synthesizing multigraph: {mg.node_count()} nodes, {mg.edge_count()} edges")

        # 6. Simple graph -> use regular synthesis (or fast path)
        if mg.is_simple():
            simple = mg.to_simple_graph()
            if simple is not None:
                if skip_minor_search:
                    result = self._synthesize_fast(simple, max_depth)
                else:
                    result = self.synthesize(simple, max_depth)
                # Track minors: both from sub-synthesis AND table entry identity
                if hasattr(self, '_mg_minors_accum'):
                    self._mg_minors_accum |= result.minors_used
                    # If this graph is a known table entry, track it as a used minor
                    simple_key = simple.canonical_key()
                    if simple_key in self.table.entries:
                        self._mg_minors_accum.add(simple_key)
                self._multigraph_cache[cache_key] = result.polynomial
                return result.polynomial

        # 7. Reduce parallel edges one at a time: T(G) = T(G\\e) + T(G/e)
        max_mult_edge = max(mg.edge_counts.keys(), key=lambda e: mg.edge_counts[e])
        if mg.edge_counts[max_mult_edge] > 1:
            poly = self._reduce_parallel_edge(mg, max_mult_edge, max_depth, skip_minor_search)
            self._multigraph_cache[cache_key] = poly
            return poly

        # 8. Fall back (should not be reached)
        raise RuntimeError(
            f"D-C fallback reached for multigraph with "
            f"{mg.node_count()} nodes, {mg.edge_count()} edges"
        )

    def _parallel_edges_formula(self, k: int) -> TuttePolynomial:
        """Compute Tutte polynomial for k parallel edges between 2 nodes.

        The formula is derived from deletion-contraction:
        - T(1 edge) = x
        - T(2 parallel) = T(1 edge) + T(loop) = x + y
        - T(3 parallel) = T(2 parallel) + T(2 loops) = x + y + y^2
        - T(k parallel) = x + y + y^2 + ... + y^{k-1}

        General formula:
        T(k parallel) = x + sum_{i=1}^{k-1} y^i
        """
        if k <= 0:
            return TuttePolynomial.one()
        if k == 1:
            return TuttePolynomial.x()
        if k == 2:
            return TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})  # x + y

        # T(k parallel) = x + y + y^2 + ... + y^{k-1}
        coeffs = {(1, 0): 1}  # x
        for i in range(1, k):
            coeffs[(0, i)] = 1  # + y^i
        return TuttePolynomial.from_coefficients(coeffs)

    def _handle_disconnected_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Handle disconnected multigraph: T(G1 ∪ G2) = T(G1) × T(G2)."""
        start = next(iter(mg.nodes))
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in mg.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        comp1_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in visited}
        comp1_loops = {n: c for n, c in mg.loop_counts.items() if n in visited}
        comp1 = MultiGraph(nodes=frozenset(visited), edge_counts=comp1_edges, loop_counts=comp1_loops)

        rest_nodes = mg.nodes - visited
        rest_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in rest_nodes}
        rest_loops = {n: c for n, c in mg.loop_counts.items() if n in rest_nodes}
        rest = MultiGraph(nodes=frozenset(rest_nodes), edge_counts=rest_edges, loop_counts=rest_loops)

        return (self._synthesize_multigraph(comp1, max_depth, skip_minor_search) *
                self._synthesize_multigraph(rest, max_depth, skip_minor_search))

    def _reduce_parallel_edge(
        self,
        mg: MultiGraph,
        edge: Tuple[int, int],
        max_depth: int,
        skip_minor_search: bool = False
    ) -> TuttePolynomial:
        """Reduce a parallel edge using T(G) = T(G\\e) + T(G/e).

        For parallel edges, deletion removes one copy, contraction
        merges the endpoints (creating loops from other parallel copies).
        """
        u, v = edge
        multiplicity = mg.edge_counts[edge]

        # Delete one copy
        new_edge_counts = dict(mg.edge_counts)
        new_edge_counts[edge] = multiplicity - 1
        if new_edge_counts[edge] == 0:
            del new_edge_counts[edge]
        mg_delete = MultiGraph(
            nodes=mg.nodes,
            edge_counts=new_edge_counts,
            loop_counts=mg.loop_counts
        )

        # Contract (merge endpoints, creating loops from parallel copies)
        mg_merged = mg.merge_nodes(u, v)
        survivor = min(u, v)
        if survivor in mg_merged.loop_counts:
            new_loops = dict(mg_merged.loop_counts)
            new_loops[survivor] -= 1
            if new_loops[survivor] == 0:
                del new_loops[survivor]
            mg_contract = MultiGraph(
                nodes=mg_merged.nodes,
                edge_counts=mg_merged.edge_counts,
                loop_counts=new_loops
            )
        else:
            mg_contract = mg_merged

        t_delete = self._synthesize_multigraph(mg_delete, max_depth, skip_minor_search)
        t_contract = self._synthesize_multigraph(mg_contract, max_depth, skip_minor_search)

        return t_delete + t_contract

    def _build_spanning_tree_bfs(self, graph: Graph) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int]]]:
        """Build a spanning tree via BFS and return (tree_edges, chords).

        Args:
            graph: Connected graph to build spanning tree from

        Returns:
            Tuple of (tree_edges set, chords list)
        """
        tree_edges = set()
        visited = set()
        start = next(iter(graph.nodes))
        queue = [start]
        visited.add(start)

        while queue:
            node = queue.pop(0)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    edge = (min(node, neighbor), max(node, neighbor))
                    tree_edges.add(edge)

        chords = [e for e in graph.edges if e not in tree_edges]
        return tree_edges, chords


# =============================================================================
# SYNTHESIS RESULT
# =============================================================================

@dataclass
class SynthesisResult:
    """Result of a graph synthesis attempt."""
    polynomial: TuttePolynomial
    recipe: List[str] = field(default_factory=list)  # Human-readable steps
    verified: bool = False
    method: str = "unknown"
    tiles_used: int = 0
    fringe_edges: int = 0
    minors_used: Set[str] = field(default_factory=set)  # Canonical keys of table entries used

    def __repr__(self) -> str:
        status = "verified" if self.verified else "unverified"
        return f"SynthesisResult({self.polynomial}, method={self.method}, {status})"
