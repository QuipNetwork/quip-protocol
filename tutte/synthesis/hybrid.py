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

from ..polynomial import TuttePolynomial
from ..graph import Graph, MultiGraph
from ..lookup.core import RainbowTable, MinorEntry, load_default_table
from ..graphs.k_join import polynomial_divmod, polynomial_divide, tutte_k
from ..factorization import polynomial_gcd, has_common_factor
from ..validation import verify_spanning_trees
from ..graphs.covering import find_disjoint_cover, compute_fringe, compute_inter_tile_edges
from ..graphs.series_parallel import compute_sp_tutte_if_applicable
from .base import BaseMultigraphSynthesizer
from ..logs import get_log, EventType, LogLevel


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
    minors_used: Set[str] = field(default_factory=set)  # Canonical keys of table entries used

    def __repr__(self) -> str:
        status = "✓" if self.verified else "✗"
        return (f"HybridResult({self.polynomial.num_spanning_trees()} trees, "
                f"alg={self.algebraic_steps}, tile={self.tiling_steps}, "
                f"dc={self.dc_steps}) {status}")


# =============================================================================
# HYBRID SYNTHESIS ENGINE
# =============================================================================

class HybridSynthesisEngine(BaseMultigraphSynthesizer):
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
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis

        # Structural engine for series-parallel, k-sum, and hierarchical decomposition
        from .engine import SynthesisEngine
        self._structural_engine = SynthesisEngine(table=self.table, verbose=verbose)
        # Share multigraph cache between engines
        self._structural_engine._multigraph_cache = self._multigraph_cache

        # Load precomputed multigraph lookup table if available
        loaded = self._structural_engine.load_multigraph_cache()
        if loaded > 0 and verbose:
            print(f"[Hybrid] Loaded {loaded} multigraph cache entries")

        # Load precomputed contraction cache if available
        cc_loaded = self._structural_engine.load_contraction_cache()
        if cc_loaded > 0 and verbose:
            print(f"[Hybrid] Loaded {cc_loaded} contraction cache entries")

        # Statistics
        self._stats = {'algebraic': 0, 'tiling': 0, 'dc': 0, 'lookup': 0}

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[Hybrid] {msg}", flush=True)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {'algebraic': 0, 'tiling': 0, 'dc': 0, 'lookup': 0}

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about methods used."""
        return dict(self._stats)

    def _synthesize_fast(self, graph: Graph, max_depth: int = 10) -> HybridSynthesisResult:
        """Fast synthesis path that skips minor search.

        HybridSynthesisEngine doesn't distinguish fast/slow paths since it
        uses treewidth/SP before falling back to tiling. Delegates to synthesize().
        """
        return self.synthesize(graph, max_depth)

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
        _log = get_log()
        # Check cache
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            _log.record(EventType.CACHE_HIT, "hybrid",
                        f"Cache hit: {graph.node_count()}n {graph.edge_count()}e",
                        LogLevel.DEBUG)
            return self._cache[cache_key]

        n = graph.node_count()
        m = graph.edge_count()
        _log.record(EventType.SYNTHESIS_START, "hybrid",
                    f"{n}n {m}e")
        self._log(f"Synthesizing: {n} nodes, {m} edges")

        # 1. Direct rainbow table lookup
        cached = self.table.lookup(graph)
        if cached is not None:
            _log.record(EventType.LOOKUP_HIT, "hybrid",
                        f"Table hit: {n}n {m}e")
            self._log("Direct lookup hit")
            self._stats['lookup'] += 1
            result = HybridSynthesisResult(
                polynomial=cached,
                method="lookup",
                decomposition=["table"],
                recipe=["Rainbow table lookup"],
                verified=True,
                minors_used={cache_key} if cache_key in self.table.entries else set(),
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
            _log.record(EventType.FACTORIZE, "hybrid",
                        f"Disconnected: {len(components)} components")
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            return result

        # 4. Try structural decompositions (series-parallel, k-sum, hierarchical)
        if graph.edge_count() >= 6:
            structural_result = self._try_structural(graph, max_depth)
            if structural_result is not None:
                self._cache[cache_key] = structural_result
                return structural_result

        # 5. Connected graph - use hybrid approach
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
        all_minors = set()

        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            decomposition.extend(comp_result.decomposition)
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            total_alg += comp_result.algebraic_steps
            total_tile += comp_result.tiling_steps
            total_dc += comp_result.dc_steps
            all_minors |= comp_result.minors_used

        return HybridSynthesisResult(
            polynomial=poly,
            method="disconnected",
            decomposition=decomposition,
            recipe=recipe,
            verified=True,
            algebraic_steps=total_alg,
            tiling_steps=total_tile,
            dc_steps=total_dc,
            minors_used=all_minors,
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
        _log = get_log()
        cut = graph.has_cut_vertex()
        if cut is not None:
            components = graph.split_at_cut_vertex(cut)
            if len(components) > 1:
                _log.record(EventType.FACTORIZE, "hybrid",
                            f"Cut vertex: {len(components)} components")
                self._log(f"Cut vertex {cut} splits into {len(components)} components")
                poly = TuttePolynomial.one()
                decomposition = []
                recipe = [f"Cut vertex factorization at node {cut}"]
                total_alg = total_tile = total_dc = 0
                all_minors = set()

                for i, comp in enumerate(components):
                    comp_result = self.synthesize(comp, max_depth)
                    poly = poly * comp_result.polynomial
                    decomposition.extend(comp_result.decomposition)
                    recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
                    total_alg += comp_result.algebraic_steps
                    total_tile += comp_result.tiling_steps
                    total_dc += comp_result.dc_steps
                    all_minors |= comp_result.minors_used

                return HybridSynthesisResult(
                    polynomial=poly,
                    method="cut_vertex",
                    decomposition=decomposition,
                    recipe=recipe,
                    algebraic_steps=total_alg + 1,
                    tiling_steps=total_tile,
                    dc_steps=total_dc,
                    minors_used=all_minors,
                )

        # Use tiling-based approach (spanning tree + edge addition)
        return self._synthesize_via_tiling(graph, max_depth)

    def _try_structural(
        self,
        graph: Graph,
        max_depth: int
    ) -> Optional[HybridSynthesisResult]:
        """Try SynthesisEngine's structural decompositions.

        Delegates to the structural engine for series-parallel, k-sum,
        and hierarchical tiling decompositions.
        """
        from .engine import SynthesisResult
        from ..polynomial import TuttePolynomial

        # Cycle graph: T(C_n) = x^{n-1} + ... + x + y
        if graph.node_count() >= 3 and graph.edge_count() == graph.node_count():
            if all(graph.degree(n) == 2 for n in graph.nodes):
                n = graph.node_count()
                coeffs = {(i, 0): 1 for i in range(1, n)}
                coeffs[(0, 1)] = 1
                self._log(f"Cycle C_{n}: direct formula")
                return HybridSynthesisResult(
                    polynomial=TuttePolynomial.from_coefficients(coeffs),
                    method="cycle_formula",
                    recipe=[f"Cycle C_{n}"],
                    verified=True,
                )

        _log = get_log()
        # Series-parallel O(n)
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            _log.record(EventType.SERIES_PARALLEL, "hybrid",
                        f"SP: {graph.node_count()}n {graph.edge_count()}e")
            self._log("Series-parallel: O(n) computation")
            return HybridSynthesisResult(
                polynomial=sp_poly,
                method="series_parallel",
                recipe=["Series-parallel decomposition"],
                verified=True,
            )

        engine = self._structural_engine

        # Treewidth DP (fast for tw <= 10, before expensive k-sum/hierarchical)
        if graph.edge_count() >= 10:
            from ..graphs.treewidth import compute_treewidth_tutte_if_applicable
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = compute_treewidth_tutte_if_applicable(full_mg, max_width=10)
            if tw_poly is not None:
                self._log(f"Treewidth DP: {graph.node_count()}n, {graph.edge_count()}e")
                return HybridSynthesisResult(
                    polynomial=tw_poly,
                    method="treewidth_dp",
                    recipe=["Treewidth-based DP (full graph)"],
                    verified=True,
                )

        # K-sum decomposition
        ksum_result = engine._try_ksum_decomposition(graph)
        if ksum_result is not None:
            _log.record(EventType.KSUM, "hybrid",
                        f"K-sum: {graph.node_count()}n {graph.edge_count()}e")
            self._log(f"K-sum decomposition: {ksum_result.method}")
            return HybridSynthesisResult(
                polynomial=ksum_result.polynomial,
                method=ksum_result.method,
                recipe=ksum_result.recipe,
                verified=ksum_result.verified,
            )

        # Hierarchical tiling
        if graph.edge_count() >= 20:
            hier_result = engine._try_hierarchical(graph, max_depth)
            if hier_result is not None:
                _log.record(EventType.HIERARCHICAL, "hybrid",
                            f"Hierarchical: {graph.node_count()}n {graph.edge_count()}e")
                self._log(f"Hierarchical tiling: {hier_result.method}")
                return HybridSynthesisResult(
                    polynomial=hier_result.polynomial,
                    method=hier_result.method,
                    recipe=hier_result.recipe,
                    verified=hier_result.verified,
                )

        return None

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
        _log = get_log()
        n = graph.node_count()
        m = graph.edge_count()
        _log.record(EventType.EDGE_ADD, "hybrid",
                    f"Tiling path: {n}n {m}e")
        self._log("Using tiling (spanning tree + edge addition)")
        self._stats['tiling'] += 1
        recipe = ["Spanning tree + edge addition"]

        if n == 0:
            return HybridSynthesisResult(
                polynomial=TuttePolynomial.one(),
                method="tiling",
                recipe=["Empty graph"],
                verified=True,
                tiling_steps=1
            )

        # Snapshot accumulator to diff later
        pre_minors = set(self._mg_minors_accum)

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

        # Harvest minors discovered during chord addition
        new_minors = self._mg_minors_accum - pre_minors

        return HybridSynthesisResult(
            polynomial=poly,
            method="tiling",
            decomposition=["spanning_tree", f"{len(chords)}_chords"],
            recipe=recipe,
            tiling_steps=1 + len(chords),
            minors_used=new_minors,
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
