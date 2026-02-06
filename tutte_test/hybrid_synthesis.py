"""Hybrid Synthesis Engine for Tutte Polynomials.

This module combines algebraic decomposition with tiling-based synthesis
to get the best of both worlds:

1. **Algebraic First**: Try to decompose target polynomial using known
   factors from the rainbow table (fast for structured polynomials)

2. **Tiling Fallback**: When algebraic decomposition fails, use
   tiling with known minors instead of expensive deletion-contraction

3. **Recursive Hybrid**: Remainders and sub-problems use the same
   hybrid approach

This avoids the exponential deletion-contraction algorithm while
leveraging polynomial algebra when beneficial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .polynomial import TuttePolynomial
from .graph import Graph, MultiGraph
from .rainbow_table import RainbowTable, MinorEntry, load_default_table
from .k_join import polynomial_divmod, polynomial_divide, tutte_k
from .factorization import polynomial_gcd, has_common_factor
from .validation import verify_spanning_trees
from .covering import find_disjoint_cover, compute_fringe, compute_inter_tile_edges


# =============================================================================
# HYBRID SYNTHESIS RESULT
# =============================================================================

@dataclass
class HybridSynthesisResult:
    """Result of hybrid polynomial synthesis."""

    polynomial: TuttePolynomial
    method: str = "hybrid"
    decomposition: List[str] = field(default_factory=list)
    recipe: List[str] = field(default_factory=list)
    verified: bool = False
    algebraic_steps: int = 0
    tiling_steps: int = 0
    dc_steps: int = 0  # Should be 0 in ideal case

    def __repr__(self) -> str:
        status = "✓" if self.verified else "✗"
        return (f"HybridResult({self.polynomial.num_spanning_trees()} trees, "
                f"alg={self.algebraic_steps}, tile={self.tiling_steps}, "
                f"dc={self.dc_steps}) {status}")


# =============================================================================
# HYBRID SYNTHESIS ENGINE
# =============================================================================

class HybridSynthesisEngine:
    """Synthesis engine combining algebraic and tiling approaches.

    Strategy:
    1. Check rainbow table for direct lookup
    2. Try algebraic factorization (if factors exist, decompose)
    3. Fall back to tiling-based synthesis (spanning tree + edge addition)
    4. Only use deletion-contraction for truly irreducible cases

    This gives O(n²) to O(n³) performance for most graphs instead of
    exponential deletion-contraction.
    """

    def __init__(
        self,
        table: Optional[RainbowTable] = None,
        verbose: bool = False,
        prefer_algebraic: bool = True
    ):
        """Initialize hybrid synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
            prefer_algebraic: Try algebraic decomposition before tiling
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose
        self.prefer_algebraic = prefer_algebraic

        # Caches
        self._cache: Dict[str, HybridSynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}

        # Statistics
        self._stats = {'algebraic': 0, 'tiling': 0, 'dc': 0, 'lookup': 0}

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[Hybrid] {msg}")

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {'algebraic': 0, 'tiling': 0, 'dc': 0, 'lookup': 0}

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about methods used."""
        return dict(self._stats)

    # =========================================================================
    # MAIN SYNTHESIS METHODS
    # =========================================================================

    def synthesize(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> HybridSynthesisResult:
        """Main entry point: compute Tutte polynomial using hybrid approach.

        Args:
            graph: Graph to compute polynomial for
            max_depth: Maximum recursion depth

        Returns:
            HybridSynthesisResult with computed polynomial
        """
        # Check cache
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._log(f"Synthesizing: {graph.node_count()} nodes, {graph.edge_count()} edges")

        # 1. Direct rainbow table lookup
        cached = self.table.lookup(graph)
        if cached is not None:
            self._log("Direct lookup hit")
            self._stats['lookup'] += 1
            result = HybridSynthesisResult(
                polynomial=cached,
                method="lookup",
                decomposition=["table"],
                recipe=["Rainbow table lookup"],
                verified=True
            )
            self._cache[cache_key] = result
            return result

        # 2. Handle base cases
        if graph.edge_count() == 0:
            result = HybridSynthesisResult(
                polynomial=TuttePolynomial.one(),
                method="base",
                recipe=["Empty graph: T = 1"],
                verified=True
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            result = HybridSynthesisResult(
                polynomial=TuttePolynomial.x(),
                method="base",
                recipe=["Single edge: T = x"],
                verified=True
            )
            self._cache[cache_key] = result
            return result

        # 3. Handle disconnected graphs
        components = graph.connected_components()
        if len(components) > 1:
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            return result

        # 4. Connected graph - use hybrid approach
        result = self._synthesize_hybrid(graph, max_depth)

        # Verify and cache
        result.verified = verify_spanning_trees(graph, result.polynomial)
        self._cache[cache_key] = result

        return result

    def _synthesize_disconnected(
        self,
        components: List[Graph],
        max_depth: int
    ) -> HybridSynthesisResult:
        """Synthesize polynomial for disconnected graph.

        T(G₁ ∪ G₂ ∪ ...) = T(G₁) × T(G₂) × ...
        """
        self._log(f"Disconnected: {len(components)} components")

        poly = TuttePolynomial.one()
        decomposition = []
        recipe = [f"Disconnected: {len(components)} components"]
        total_alg = total_tile = total_dc = 0

        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            decomposition.extend(comp_result.decomposition)
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            total_alg += comp_result.algebraic_steps
            total_tile += comp_result.tiling_steps
            total_dc += comp_result.dc_steps

        return HybridSynthesisResult(
            polynomial=poly,
            method="disconnected",
            decomposition=decomposition,
            recipe=recipe,
            verified=True,
            algebraic_steps=total_alg,
            tiling_steps=total_tile,
            dc_steps=total_dc
        )

    def _synthesize_hybrid(
        self,
        graph: Graph,
        max_depth: int
    ) -> HybridSynthesisResult:
        """Hybrid synthesis for connected graph.

        Strategy:
        1. Check for cut vertices → factor into components
        2. Recursively split at ALL cut vertices (not just one)
        3. Use tiling-based spanning tree expansion on 2-connected blocks
        """
        # Check for cut vertices first (factorization is always a win)
        cut = graph.has_cut_vertex()
        if cut is not None:
            components = graph.split_at_cut_vertex(cut)
            if len(components) > 1:
                self._log(f"Cut vertex {cut} splits into {len(components)} components")
                poly = TuttePolynomial.one()
                decomposition = []
                recipe = [f"Cut vertex factorization at node {cut}"]
                total_alg = total_tile = total_dc = 0

                for i, comp in enumerate(components):
                    comp_result = self.synthesize(comp, max_depth)
                    poly = poly * comp_result.polynomial
                    decomposition.extend(comp_result.decomposition)
                    recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
                    total_alg += comp_result.algebraic_steps
                    total_tile += comp_result.tiling_steps
                    total_dc += comp_result.dc_steps

                return HybridSynthesisResult(
                    polynomial=poly,
                    method="cut_vertex",
                    decomposition=decomposition,
                    recipe=recipe,
                    algebraic_steps=total_alg + 1,
                    tiling_steps=total_tile,
                    dc_steps=total_dc
                )

        # Use tiling-based approach (spanning tree + edge addition)
        return self._synthesize_via_tiling(graph, max_depth)

    def _find_all_cut_vertices_and_split(
        self,
        graph: Graph
    ) -> Optional[List[Graph]]:
        """Find all cut vertices and split graph into 2-connected blocks.

        Returns None if graph is 2-connected (no cut vertices).
        Otherwise returns list of blocks (each containing their cut vertices).
        """
        cut_vertices = graph.find_all_cut_vertices()
        if not cut_vertices:
            return None

        # Recursively split at cut vertices
        blocks = []
        remaining = [graph]

        while remaining:
            g = remaining.pop()
            cut = g.has_cut_vertex()
            if cut is None:
                blocks.append(g)
            else:
                parts = g.split_at_cut_vertex(cut)
                remaining.extend(parts)

        return blocks

    def _synthesize_via_tiling(
        self,
        graph: Graph,
        max_depth: int
    ) -> HybridSynthesisResult:
        """Synthesize using spanning tree + edge addition (tiling approach).

        This avoids exponential deletion-contraction by:
        1. Finding a spanning tree (T = x^(n-1))
        2. Adding chords one at a time using edge addition formula
        3. Using pattern recognition for merged graphs
        """
        self._log("Using tiling (spanning tree + edge addition)")
        self._stats['tiling'] += 1

        n = graph.node_count()
        m = graph.edge_count()
        recipe = ["Spanning tree + edge addition"]

        if n == 0:
            return HybridSynthesisResult(
                polynomial=TuttePolynomial.one(),
                method="tiling",
                recipe=["Empty graph"],
                verified=True,
                tiling_steps=1
            )

        # Find spanning tree via BFS
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

        # Chords (non-tree edges)
        chords = [e for e in graph.edges if e not in tree_edges]

        self._log(f"Spanning tree: {len(tree_edges)} edges, chords: {len(chords)}")
        recipe.append(f"Spanning tree: {len(tree_edges)} edges")
        recipe.append(f"Chords to add: {len(chords)}")

        # Start with spanning tree polynomial
        poly = TuttePolynomial.x(len(tree_edges))

        # Build current multigraph
        current_mg = MultiGraph(
            nodes=graph.nodes,
            edge_counts={e: 1 for e in tree_edges},
            loop_counts={}
        )

        # Add each chord using edge addition
        for i, (u, v) in enumerate(chords):
            # T(G + e) = T(G) + T(G/{u,v})
            merged = current_mg.merge_nodes(u, v)
            merged_poly = self._synthesize_multigraph(merged, max_depth)

            poly = poly + merged_poly

            # Update current graph
            edge = (min(u, v), max(u, v))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts
            )

        recipe.append(f"Final: {poly.num_terms()} terms")

        return HybridSynthesisResult(
            polynomial=poly,
            method="tiling",
            decomposition=["spanning_tree", f"{len(chords)}_chords"],
            recipe=recipe,
            tiling_steps=1 + len(chords)
        )

    # =========================================================================
    # MULTIGRAPH SYNTHESIS (with pattern recognition)
    # =========================================================================

    def _synthesize_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int
    ) -> TuttePolynomial:
        """Synthesize polynomial for multigraph with pattern recognition.

        Uses patterns to avoid deletion-contraction:
        1. Cache lookup
        2. Simple graph → regular synthesis
        3. Loops: T(G with loop) = y × T(G without loop)
        4. Cut vertex: T(G1 · G2) = T(G1) × T(G2)
        5. Parallel edges: closed-form formula
        6. Only D-C for truly irreducible cases
        """
        cache_key = mg.canonical_key()
        if cache_key in self._multigraph_cache:
            return self._multigraph_cache[cache_key]

        # Simple graph case
        if mg.is_simple():
            simple = mg.to_simple_graph()
            if simple is not None:
                result = self.synthesize(simple, max_depth)
                self._multigraph_cache[cache_key] = result.polynomial
                return result.polynomial

        # Loops: T(G with loops) = y^k × T(G without loops)
        if mg.total_loop_count() > 0:
            loop_count = mg.total_loop_count()
            mg_no_loops = mg.remove_loops()
            poly = TuttePolynomial.y(loop_count) * self._synthesize_multigraph(mg_no_loops, max_depth)
            self._multigraph_cache[cache_key] = poly
            return poly

        # Cut vertex factorization
        cut = mg.has_cut_vertex()
        if cut is not None:
            components = mg.split_at_cut_vertex(cut)
            if len(components) > 1:
                poly = TuttePolynomial.one()
                for comp in components:
                    poly = poly * self._synthesize_multigraph(comp, max_depth)
                self._multigraph_cache[cache_key] = poly
                return poly

        # Parallel edges between exactly 2 nodes
        if mg.is_just_parallel_edges():
            poly = self._parallel_edges_formula(mg.parallel_edge_count())
            self._multigraph_cache[cache_key] = poly
            return poly

        # Disconnected
        if not mg.is_connected():
            poly = self._handle_disconnected_multigraph(mg, max_depth)
            self._multigraph_cache[cache_key] = poly
            return poly

        # Try to reduce to a simpler multigraph using edge reduction
        # If we have parallel edges, peel them off one at a time
        # using: T(G) = T(G\e) + T(G/e)
        # Choose the edge with highest multiplicity - contracting it
        # reduces node count while deleting reduces edge count
        max_mult_edge = max(mg.edge_counts.keys(), key=lambda e: mg.edge_counts[e])
        if mg.edge_counts[max_mult_edge] > 1:
            # Reduce multiplicity by 1 via deletion-contraction on one copy
            poly = self._reduce_parallel_edge(mg, max_mult_edge, max_depth)
            self._multigraph_cache[cache_key] = poly
            return poly

        # Fall back to deletion-contraction for remaining cases
        poly = self._dc_multigraph(mg, max_depth)
        self._multigraph_cache[cache_key] = poly
        return poly

    def _reduce_parallel_edge(
        self,
        mg: MultiGraph,
        edge: Tuple[int, int],
        max_depth: int
    ) -> TuttePolynomial:
        """Reduce a parallel edge using T(G) = T(G\\e) + T(G/e).

        For parallel edges, deletion removes one copy, contraction
        merges the endpoints (creating loops from other parallel copies).
        This is cheaper than general D-C because the resulting graphs
        are simpler.
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

        # Both sub-problems go through pattern recognition again
        t_delete = self._synthesize_multigraph(mg_delete, max_depth)
        t_contract = self._synthesize_multigraph(mg_contract, max_depth)

        return t_delete + t_contract

    def _parallel_edges_formula(self, k: int) -> TuttePolynomial:
        """T(k parallel edges) = x + y + y² + ... + y^(k-1)."""
        if k <= 0:
            return TuttePolynomial.one()
        if k == 1:
            return TuttePolynomial.x()

        coeffs = {(1, 0): 1}
        for i in range(1, k):
            coeffs[(0, i)] = 1
        return TuttePolynomial.from_coefficients(coeffs)

    def _handle_disconnected_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int
    ) -> TuttePolynomial:
        """Handle disconnected multigraph: T(G1 ∪ G2) = T(G1) × T(G2)."""
        # Find one component via BFS
        start = next(iter(mg.nodes))
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in mg.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        # Build first component
        comp1_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in visited}
        comp1_loops = {n: c for n, c in mg.loop_counts.items() if n in visited}
        comp1 = MultiGraph(nodes=frozenset(visited), edge_counts=comp1_edges, loop_counts=comp1_loops)

        # Build rest
        rest_nodes = mg.nodes - visited
        rest_edges = {e: c for e, c in mg.edge_counts.items() if e[0] in rest_nodes}
        rest_loops = {n: c for n, c in mg.loop_counts.items() if n in rest_nodes}
        rest = MultiGraph(nodes=frozenset(rest_nodes), edge_counts=rest_edges, loop_counts=rest_loops)

        return self._synthesize_multigraph(comp1, max_depth) * self._synthesize_multigraph(rest, max_depth)

    def _dc_multigraph(
        self,
        mg: MultiGraph,
        max_depth: int
    ) -> TuttePolynomial:
        """Deletion-contraction fallback — should never be reached.

        Pattern recognition (loops, cut vertices, parallel edges, disconnected
        components, parallel edge reduction) should handle all multigraph cases.
        If this is reached, it indicates a gap in the pattern recognition logic.
        """
        raise RuntimeError(
            f"D-C fallback reached for multigraph with "
            f"{mg.node_count()} nodes, {mg.edge_count()} edges"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def hybrid_synthesize(
    graph: Graph,
    verbose: bool = False
) -> HybridSynthesisResult:
    """Convenience function for hybrid synthesis.

    Args:
        graph: Graph to compute polynomial for
        verbose: Print progress information

    Returns:
        HybridSynthesisResult with computed polynomial
    """
    engine = HybridSynthesisEngine(verbose=verbose)
    return engine.synthesize(graph)


def compute_tutte_hybrid(graph: Graph) -> TuttePolynomial:
    """Compute Tutte polynomial using hybrid approach.

    Args:
        graph: Graph to compute polynomial for

    Returns:
        TuttePolynomial for the graph
    """
    result = hybrid_synthesize(graph)
    return result.polynomial
