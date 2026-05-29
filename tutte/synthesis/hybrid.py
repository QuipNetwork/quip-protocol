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
from typing import Any, Dict, List, Optional, Set, Tuple

from ..factorization import has_common_factor, polynomial_gcd
from ..family_recognition import recognize_family
from ..graph import Graph, MultiGraph
from ..graphs.covering import (compute_fringe, compute_inter_tile_edges,
                               find_disjoint_cover)
from ..graphs.k_sum import polynomial_divide, polynomial_divmod
from ..graphs.series_parallel import compute_sp_tutte_if_applicable
from ..logs import EventType, LogLevel, get_log
from ..lookup.core import MinorEntry, RainbowTable, load_default_table
from ..polynomial import TuttePolynomial
from ..validation import verify_spanning_trees
from .base import BaseMultigraphSynthesizer

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
    # Number of cell tiles used in the decomposition (surfaced by the
    # visualizer's Result card). 0 when the path didn't use a
    # hierarchical tiling (e.g. pure treewidth_dp on a non-cellular graph).
    tiles_used: int = 0
    minors_used: Set[str] = field(default_factory=set)  # Canonical keys of table entries used
    # Canonical keys of sub-problems the engine synthesized (not table hits).
    synthesized_minors: Set[str] = field(default_factory=set)
    # Snapshot of Graph|MultiGraph objects keyed by canonical_key.
    synthesized_graphs: Dict[str, Any] = field(default_factory=dict)

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
    ):
        """Initialize hybrid synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose

        # Caches
        self._cache: Dict[str, HybridSynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis
        # (canonical_key -> Graph|MultiGraph) for every sub-problem actually
        # synthesized this run; attached to the top-level result for the
        # visualizer's "synthesized" panel.
        self._synth_accum_graphs: Dict[str, Any] = {}
        self._synth_depth: int = 0
        # When True, the top-level synthesize() skips the rainbow-table
        # lookup for the input graph (sub-problems may still look up).
        self.skip_target_lookup: bool = False

        # Structural engine for series-parallel, k-sum, and hierarchical decomposition
        from .engine import SynthesisEngine
        self._structural_engine = SynthesisEngine(table=self.table, verbose=verbose)
        # Share multigraph cache between engines
        self._structural_engine._multigraph_cache = self._multigraph_cache

        # Load precomputed multigraph lookup table if available
        loaded = self._structural_engine.load_multigraph_cache()
        if loaded > 0 and verbose:
            print(f"[Hybrid] Loaded {loaded} multigraph cache entries")

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

        Thin wrapper that tracks recursion depth and stamps the top-level
        result with `synthesized_minors` and `synthesized_graphs` for the
        visualizer.
        """
        is_top = self._synth_depth == 0
        self._synth_depth += 1
        try:
            result = self._synthesize_inner(graph, max_depth)
        finally:
            self._synth_depth -= 1
        if is_top:
            result.synthesized_graphs = dict(self._synth_accum_graphs)
            result.synthesized_minors = set(self._synth_accum_graphs.keys())
            if getattr(self, 'promote_cache_on_finish', False):
                self._flush_cache_to_table()
        return result

    def _flush_cache_to_table(self) -> None:
        """Promote simple-graph cache entries to the rainbow table at
        end-of-synthesis. See SynthesisEngine._flush_cache_to_table."""
        from ..graph import compute_signature
        for cache_key, cached_result in list(self._cache.items()):
            if cache_key in self.table.entries:
                continue
            g = self._synth_accum_graphs.get(cache_key)
            if g is None or not hasattr(g, 'edges'):
                continue
            try:
                entry = MinorEntry(
                    name=(
                        f"auto_{g.node_count()}n{g.edge_count()}e_"
                        f"{cache_key[:8]}"
                    ),
                    polynomial=cached_result.polynomial,
                    node_count=g.node_count(),
                    edge_count=g.edge_count(),
                    canonical_key=cache_key,
                    spanning_trees=cached_result.polynomial.num_spanning_trees(),
                    num_terms=cached_result.polynomial.num_terms(),
                    graph=g,
                    signature=compute_signature(g),
                )
                self.table.add_entry(entry)
            except Exception:
                continue

    def _synthesize_inner(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> HybridSynthesisResult:
        """Core hybrid synthesis logic — see `synthesize()` for the public entrypoint."""
        _log = get_log()
        # Check cache
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            _log.record(EventType.CACHE_HIT, "hybrid",
                        f"Cache hit: {graph.node_count()}n {graph.edge_count()}e",
                        LogLevel.DEBUG, graph=graph)
            # Record so the visualizer surfaces cache-hit graphs too.
            self._record_synth(graph, cache_key)
            return self._cache[cache_key]

        n = graph.node_count()
        m = graph.edge_count()
        _log.record(EventType.SYNTHESIS_START, "hybrid",
                    f"{n}n {m}e", graph=graph)
        self._log(f"Synthesizing: {n} nodes, {m} edges")

        # 1. Family recognition fast path — O(n+m)
        family_poly = recognize_family(graph)
        if family_poly is not None:
            _log.record(EventType.FAMILY_RECOGNITION, "hybrid",
                        f"Family recognized: {n}n {m}e", LogLevel.INFO,
                        graph=graph)
            self._log(f"Family recognition: O(n+m) fast path")
            result = HybridSynthesisResult(
                polynomial=family_poly,
                method="family_recognition",
                recipe=["Family recognition"],
                verified=True,
            )
            self._cache[cache_key] = result
            self._record_synth(graph, cache_key)
            return result

        # 1.5 Transfer matrix for periodic lattice strips — mirrors engine
        # step 1.5. O(V+E) detection + C-accelerated sweep over non-crossing
        # partition states. Handles grid (m > 2), triangular, honeycomb,
        # square-octagon, elongated-triangular strips. Returns None on
        # non-lattice inputs. Known regression class: wider grids
        # (width ≥ 6) where Catalan(w)² > 2^tw — a width-aware gate is
        # a separate follow-up that should be applied to BOTH engines.
        try:
            from ..transfer_matrix import compute_tutte_via_transfer_matrix
            tm_poly = compute_tutte_via_transfer_matrix(graph)
            if tm_poly is not None:
                _log.record(EventType.SYNTHESIS_START, "hybrid",
                            f"Transfer matrix: {n}n {m}e", LogLevel.INFO,
                            graph=graph)
                self._log(f"Transfer matrix: O(V+E) detection + sweep")
                result = HybridSynthesisResult(
                    polynomial=tm_poly,
                    method="transfer_matrix",
                    recipe=["Transfer matrix"],
                    verified=True,
                )
                self._cache[cache_key] = result
                return result
        except Exception:
            pass

        # 2. Direct rainbow table lookup (optionally skipped at top-level).
        is_top_call = self._synth_depth == 1
        if not (is_top_call and self.skip_target_lookup):
            cached = self.table.lookup(graph)
            if cached is not None:
                _log.record(EventType.LOOKUP_HIT, "hybrid",
                            f"Table hit: {n}n {m}e", graph=graph)
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

        # Past the lookup gate: real synthesis work. Record the input graph.
        self._record_synth(graph, cache_key)

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
                        f"Disconnected: {len(components)} components",
                        graph=graph)
            # Per-component snapshot + provenance for the visualizer.
            self._structural_engine._emit_subgraph_provenance(
                graph, [c.nodes for c in components], EventType.FACTORIZE,
                lambda i, vs: f"Connected component {i + 1}: {len(vs)}v",
            )
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            return result

        # 3.5. Block-cut / cut-vertex factorization.
        # T(G) = ∏ T(block_i) for biconnected blocks. Structure gate:
        #   - 0 articulation points  → biconnected, skip (try 2-sum next)
        #   - 1 articulation point   → use cut_vertex (single split, cheapest)
        #   - 2+ articulation points → use block-cut (one pass for all
        #                              blocks; subsumes recursive cut_vertex)
        try:
            import networkx as _nx
            nxg = graph.to_networkx()
            arts = list(_nx.articulation_points(nxg))
        except Exception:
            arts = []
        if len(arts) >= 2:
            # Block-cut decomposition: one pass, all blocks at once.
            blocks = list(_nx.biconnected_components(nxg))
            _log.record(EventType.FACTORIZE, "hybrid",
                        f"Block-cut: {len(blocks)} blocks, "
                        f"{len(arts)} articulation points", graph=graph)
            # Per-block snapshot + provenance via shared engine helper.
            self._structural_engine._emit_subgraph_provenance(
                graph, blocks, EventType.FACTORIZE,
                lambda i, vs: f"Block {i + 1}: {len(vs)}v",
            )
            self._log(f"Block-cut decomposition: {len(blocks)} blocks")
            poly = TuttePolynomial.one()
            recipe = [
                f"Block-cut decomposition: {len(blocks)} biconnected blocks, "
                f"{len(arts)} articulation points"
            ]
            all_minors: Set[str] = set()
            for i, block_vs in enumerate(blocks):
                # Each block is a set of vertices; induced subgraph is biconnected.
                block_subgraph = Graph(
                    nodes=frozenset(block_vs),
                    edges=frozenset(
                        (min(u, v), max(u, v))
                        for (u, v) in graph.edges
                        if u in block_vs and v in block_vs
                    ),
                )
                comp_result = self.synthesize(block_subgraph, max_depth)
                poly = poly * comp_result.polynomial
                recipe.append(
                    f"  Block {i + 1} ({len(block_vs)} verts): "
                    f"{comp_result.polynomial}"
                )
                all_minors |= comp_result.minors_used
            result = HybridSynthesisResult(
                polynomial=poly,
                method="block_cut",
                recipe=recipe,
                verified=True,
                minors_used=all_minors,
            )
            self._cache[cache_key] = result
            return result
        if len(arts) == 1:
            # Single articulation point — recursive cut_vertex is the simple path.
            cut = arts[0]
            _log.record(EventType.FACTORIZE, "hybrid",
                        f"Cut vertex at {cut}", graph=graph)
            self._log(f"Cut vertex factorization at node {cut}")
            cut_components = graph.split_at_cut_vertex(cut)
            # Per-component snapshot + provenance for the visualizer.
            self._structural_engine._emit_subgraph_provenance(
                graph, [c.nodes for c in cut_components], EventType.FACTORIZE,
                lambda i, vs: f"Cut-vertex component {i + 1}: {len(vs)}v",
            )
            poly = TuttePolynomial.one()
            recipe = [
                f"Cut vertex factorization at node {cut}: "
                f"{len(cut_components)} components"
            ]
            all_minors: Set[str] = set()
            for i, comp in enumerate(cut_components):
                comp_result = self.synthesize(comp, max_depth)
                poly = poly * comp_result.polynomial
                recipe.append(f"  Component {i + 1}: {comp_result.polynomial}")
                all_minors |= comp_result.minors_used
            result = HybridSynthesisResult(
                polynomial=poly,
                method="cut_vertex",
                recipe=recipe,
                verified=True,
                minors_used=all_minors,
            )
            self._cache[cache_key] = result
            return result

        # 3.7. Early 2-sum / SPQR-style decomposition for biconnected graphs.
        # Strict superset of cut_vertex: catches 2-vertex separators (cycles
        # glued at an edge, K_4 with hanging edge, etc.) that have no single
        # articulation point.
        #
        # Gate (in cheap-first order to avoid per-graph overhead):
        #   1. m ≥ 80 (cheap) — small graphs are handled cheaply by the
        #      downstream cascade (treewidth_dp finishes in <100ms for
        #      tw≤10). Without this gate, the per-graph node_connectivity
        #      + treewidth_min_degree probes (each ~10-50ms on n=20..60
        #      graphs) regressed Z(1,2)_inter (25→65ms) and Grid_6x6
        #      (118→299ms). Pm2 (m=164) and other m≥80 graphs are the
        #      only realistic 2-cut candidates anyway.
        #   2. graph is biconnected (already known: arts == [])
        #   3. node_connectivity == 2 (cheap O(V·E) max-flow probe)
        #   4. treewidth upper bound > 8 — defer to tw_dp on low-tw graphs.
        if (graph.edge_count() >= 80
                and graph.node_count() >= 6
                and not arts):
            try:
                import networkx as _nx
                _nxg = graph.to_networkx()
                kappa = _nx.node_connectivity(_nxg)
            except Exception:
                kappa = 0
            invoke_ksum = False
            if kappa == 2:
                try:
                    from networkx.algorithms.approximation import (
                        treewidth_min_degree,
                    )
                    tw_upper, _ = treewidth_min_degree(_nxg)
                    invoke_ksum = tw_upper > 8
                except Exception:
                    invoke_ksum = False
            if invoke_ksum:
                try:
                    ksum_result = self._structural_engine._try_ksum_decomposition(
                        graph,
                    )
                except Exception:
                    ksum_result = None
                if ksum_result is not None:
                    _log.record(EventType.FACTORIZE, "hybrid",
                                f"Early 2-sum: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    # Per-side snapshot + provenance via shared engine helper.
                    try:
                        sep = set(_nx.minimum_node_cut(_nxg))
                        residual = _nxg.copy()
                        residual.remove_nodes_from(sep)
                        sides = [
                            set(c) | sep
                            for c in _nx.connected_components(residual)
                        ]
                        self._structural_engine._emit_subgraph_provenance(
                            graph, sides, EventType.FACTORIZE,
                            lambda i, vs: (
                                f"2-sum side {i + 1}: {len(vs)}v "
                                f"(incl. 2-separator)"
                            ),
                        )
                    except Exception:
                        pass
                    self._log(f"Early 2-sum: {ksum_result.method}")
                    result = HybridSynthesisResult(
                        polynomial=ksum_result.polynomial,
                        method=ksum_result.method,
                        recipe=list(ksum_result.recipe),
                        verified=ksum_result.verified,
                        minors_used=ksum_result.minors_used,
                    )
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
                            f"Cut vertex: {len(components)} components",
                            graph=graph)
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

        _log = get_log()
        # Series-parallel O(n)
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            _log.record(EventType.SERIES_PARALLEL, "hybrid",
                        f"SP: {graph.node_count()}n {graph.edge_count()}e",
                        graph=graph)
            self._log("Series-parallel: O(n) computation")
            return HybridSynthesisResult(
                polynomial=sp_poly,
                method="series_parallel",
                recipe=["Series-parallel decomposition"],
                verified=True,
            )

        # Cotree DP — subexponential exp(O(n^{2/3})) for cographs (P_4-free).
        # Mirrors engine step 7.5. Wins on K_n for n ≥ 5 (K_12 alone is
        # the difference between Hybrid timing out and completing in ~1.8s)
        # and on D-Wave K_{4,4} cells. Fast no-op on non-cographs.
        try:
            from ..cotree_dp import compute_tutte_cotree_dp
            cotree_poly = compute_tutte_cotree_dp(graph)
            if cotree_poly is not None:
                _log.record(EventType.COTREE_DP, "hybrid",
                            f"Cotree DP: {graph.node_count()}n {graph.edge_count()}e",
                            graph=graph)
                self._log(f"Cotree DP: {graph.node_count()}n, "
                          f"{graph.edge_count()}e")
                return HybridSynthesisResult(
                    polynomial=cotree_poly,
                    method="cotree_dp",
                    recipe=["Cotree-based DP (subexponential cograph)"],
                    verified=True,
                )
        except (ValueError, TypeError):
            pass

        # Almost-cograph DP — for graphs that become cographs after
        # removing ≤ 16 anomaly edges (e.g. D-Wave cells joined by
        # sparse inter-cell edges). Mirrors engine step 7.6.
        #
        # Gate: n ≤ 20. The probe is ~50ms even when it returns None
        # (greedy P_4 elimination on n=24 graphs). For n > 20, defer
        # to downstream paths (treewidth_dp will handle low-tw cases
        # in ms). Regressed Z(1,2)_inter (n=24) 25ms → 65ms before
        # this gate.
        if graph.node_count() <= 20:
            try:
                from ..cotree_dp import compute_tutte_almost_cograph
                almost_poly = compute_tutte_almost_cograph(
                    graph, self._structural_engine, max_anomalies=16,
                )
                if almost_poly is not None:
                    _log.record(EventType.COTREE_DP, "hybrid",
                                f"Almost-cograph DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._log(f"Almost-cograph DP: {graph.node_count()}n, "
                              f"{graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=almost_poly,
                        method="almost_cograph",
                        recipe=["Almost-cograph DP (greedy P_4 elim + chord rule)"],
                        verified=True,
                    )
            except Exception:
                pass

        engine = self._structural_engine

        # Chain recurrence — for linear cell-quotient chains with at least
        # 4 cells. Beats apply_kmatching_formula on Chimera Cm(1, n)
        # (which has n-1 junctions producing 5^(n-1) leaves in kmatching)
        # and beats try_heterogeneous_partition VF2 search inside
        # formula_shortcircuit. Mirrors engine step 7.4.
        #
        # Gate edge_count >= 80 so we skip Cm(1, 3) (56 edges, 3 cells)
        # where treewidth_dp at tw=4 is faster (~0.1s vs ~5s setup).
        # Cm(1, 4) has 76 edges → also skipped; Cm(1, 5)+ catches it.
        if graph.edge_count() >= 80:
            try:
                from ..roots.cell_quotient_bipartite_junction import (
                    build_bipartite_junction_spec,
                )
                from ..roots.chain_recurrence import (
                    compute_chain_full_poly_from_spec,
                    is_chain_topology,
                )
                _spec_built = build_bipartite_junction_spec(graph, engine.table)
                if (_spec_built is not None
                        and is_chain_topology(_spec_built[0].cell_tree)
                        and _spec_built[0].cell_tree.number_of_nodes() >= 3):
                    chain_poly = compute_chain_full_poly_from_spec(_spec_built[0])
                    if (chain_poly is not None
                            and verify_spanning_trees(graph, chain_poly)):
                        n_cells = _spec_built[0].cell_tree.number_of_nodes()
                        _log.record(EventType.HIERARCHICAL, "hybrid",
                                    f"Chain recurrence: "
                                    f"{graph.node_count()}n {graph.edge_count()}e, "
                                    f"{n_cells} cells",
                                    graph=graph)
                        self._log(f"Chain recurrence: {n_cells} cells")
                        return HybridSynthesisResult(
                            polynomial=chain_poly,
                            method="chain_recurrence",
                            recipe=[f"Chain recurrence: {n_cells} cells"],
                            verified=True,
                            tiles_used=n_cells,
                        )
            except Exception:
                pass  # any failure — fall through to formula_shortcircuit

        # Formula shortcut (unified topology + k-matching closed forms),
        # BEFORE treewidth_dp — for targets like Cm2 where the formulas give a
        # meaningful speedup over direct treewidth_dp (~4× on Cm2).
        # Gate at edge_count ≥ 60 to skip small graphs where detection
        # overhead outweighs tw_dp savings.
        if graph.edge_count() >= 60:
            formula_result = engine._try_formula_shortcircuit(graph, max_depth)
            if formula_result is not None:
                _method_event = {
                    "unified_formula": EventType.UNIFIED_FORMULA,
                    "kmatching_formula": EventType.KMATCHING_FORMULA,
                }.get(formula_result.method, EventType.HIERARCHICAL)
                _log.record(_method_event, "hybrid",
                            f"Formula shortcut via {formula_result.method}: "
                            f"{graph.node_count()}n {graph.edge_count()}e",
                            graph=graph)
                self._log(f"Formula shortcut: {formula_result.method}")
                return HybridSynthesisResult(
                    polynomial=formula_result.polynomial,
                    method=formula_result.method,
                    recipe=formula_result.recipe,
                    verified=formula_result.verified,
                    tiles_used=formula_result.tiles_used,
                )

        # Cell-quotient grid DP (streamed) — mirrors engine step 7.45.
        # For cell-decomposable graphs whose cell-quotient is a 2D grid of
        # K_{a,b} cells with M_k matching connectors and DISJOINT per-
        # direction anchors (Cm2 fits; Cm3 has shared anchors so this
        # rejects and falls through). Cm2: ~27s vs the older
        # cell_quotient_cycle_dp's 45s on the same target.
        if graph.edge_count() >= 60:
            try:
                from ..roots import compute_cell_quotient_grid_dp_streamed
                grid_poly = compute_cell_quotient_grid_dp_streamed(
                    graph, engine.table,
                )
                if grid_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Grid DP (streamed): "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient grid DP (streamed): "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=grid_poly,
                        method="cell_quotient_grid_dp_streamed",
                        recipe=["Cell-quotient grid DP (streamed)"],
                        verified=True,
                    )
            except Exception:
                pass  # precondition miss or import error — fall through

        # Cell-quotient cycle DP (mirrors engine step 7.7) BEFORE treewidth_dp
        # so cycle-topology graphs win without paying the tw_dp cost.
        if graph.edge_count() >= 60:
            from ..roots import compute_cell_quotient_cycle_dp
            try:
                cycle_poly = compute_cell_quotient_cycle_dp(
                    graph, engine.table,
                )
                if cycle_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Cell-quotient cycle DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient cycle DP: "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=cycle_poly,
                        method="cell_quotient_dp",
                        recipe=["Cell-quotient cycle DP"],
                        verified=True,
                    )
            except Exception:
                pass

        # Cell-quotient TREE DP (mirrors engine step 7.8) BEFORE treewidth_dp.
        if graph.edge_count() >= 60:
            from ..roots import compute_cell_quotient_tree_dp
            try:
                tree_poly = compute_cell_quotient_tree_dp(graph, engine.table)
                if tree_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Cell-quotient tree DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient tree DP: "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=tree_poly,
                        method="cell_quotient_tree_dp",
                        recipe=["Cell-quotient tree DP"],
                        verified=True,
                    )
            except Exception:
                pass

        # Cell-quotient BIPARTITE-JUNCTION DP (engine step 7.82) — handles
        # non-matching bipartite junctions (e.g., Z(m, t) families).
        if graph.edge_count() >= 60:
            from ..roots.cell_quotient_bipartite_junction import (
                compute_cell_quotient_bipartite_junction_dp,
            )
            try:
                bj_poly = compute_cell_quotient_bipartite_junction_dp(
                    graph, engine.table,
                )
                if bj_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Cell-quotient bipartite-junction DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient bipartite-junction DP: "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=bj_poly,
                        method="cell_quotient_bipartite_junction_dp",
                        recipe=["Cell-quotient bipartite-junction DP"],
                        verified=True,
                    )
            except Exception:
                pass

        # Per-component bipartite-junction DP (engine step 7.83) — splits
        # disconnected junctions into per-component sub-junctions, sidesteps
        # the Bell(joint_boundary) wall. Z(1, 2) lands here when the
        # persistent rooted cache holds T_rooted(cell, all-anchors).
        if graph.edge_count() >= 60:
            from ..roots.cell_quotient_bipartite_junction import (
                compute_bipartite_junction_per_component_dp,
            )
            try:
                # max_cell_boundary=8 — boundaries up to 8 proceed
                # directly; 9..12 are gated by cache hit (persistent
                # rooted-lookup); anything larger bails. Z(1, 2) cells
                # are 12-anchor and rely on the persistent cache.
                pcdp_poly = compute_bipartite_junction_per_component_dp(
                    graph, engine.table, max_cell_boundary=8,
                )
                if pcdp_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Cell-quotient bipartite-junction per-component DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient bipartite-junction per-component DP: "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=pcdp_poly,
                        method="cell_quotient_bipartite_junction_per_component_dp",
                        recipe=["Cell-quotient bipartite-junction per-component DP"],
                        verified=True,
                    )
            except Exception:
                pass

        # Cell-quotient HYBRID DP (mirrors engine step 7.85) BEFORE treewidth_dp.
        if graph.edge_count() >= 60:
            from ..roots.cell_quotient_hybrid import compute_cell_quotient_hybrid
            try:
                hybrid_poly = compute_cell_quotient_hybrid(
                    graph, engine.table,
                )
                if hybrid_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "hybrid",
                                f"Cell-quotient hybrid DP: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._structural_engine._maybe_emit_cell_partition(graph)
                    self._log(f"Cell-quotient hybrid DP: "
                              f"{graph.node_count()}n, {graph.edge_count()}e")
                    return HybridSynthesisResult(
                        polynomial=hybrid_poly,
                        method="cell_quotient_hybrid_dp",
                        recipe=["Cell-quotient hybrid (cycle-close + per-leaf synth)"],
                        verified=True,
                    )
            except Exception:
                pass

        # Unified decomposition + chord-peel (mirrors engine step 7.88) —
        # replaces legacy cross-cell + clique-atom + hierarchical paths
        # with a single dispatcher that discovers atom AND cell
        # decompositions, tries cell-only closed forms (unified_formula
        # / kmatching_formula / product_formula), then runs cost-gated
        # chord-rule on the cheapest. Same gate as engine 7.88.
        # Upper edge-count gate dropped — the dispatcher's cache-aware
        # probe (engine.py Phase C) now decides per-graph whether to
        # accept chord-peel based on whether the first trial contraction
        # hits the multigraph cache. Hybrid stays in parity with engine.
        if (graph.edge_count() >= 20 and self._synth_depth <= 2):
            try:
                # Bump engine depth to match hybrid depth so the engine's
                # own depth-gated recursion in Phase D residue peel
                # behaves consistently (the residue re-enters
                # engine.synthesize, not hybrid.synthesize).
                engine._synth_depth = self._synth_depth
                dp_result = engine._try_decomposition_chord_peel(
                    graph, max_depth,
                )
                if dp_result is not None:
                    _method_event = {
                        "unified_formula": EventType.UNIFIED_FORMULA,
                        "kmatching_formula": EventType.KMATCHING_FORMULA,
                        "product_formula": EventType.HIERARCHICAL,
                    }.get(dp_result.method, EventType.CHORD_RULE)
                    _log.record(_method_event, "hybrid",
                                f"Decomposition+peel via {dp_result.method}: "
                                f"{graph.node_count()}n {graph.edge_count()}e",
                                graph=graph)
                    self._log(
                        f"Decomposition+peel: "
                        f"{graph.node_count()}n, {graph.edge_count()}e "
                        f"({dp_result.method})"
                    )
                    return HybridSynthesisResult(
                        polynomial=dp_result.polynomial,
                        method=dp_result.method,
                        recipe=dp_result.recipe,
                        verified=dp_result.verified,
                        tiles_used=getattr(dp_result, 'tiles_used', 0),
                    )
            except Exception:
                pass

        # Treewidth DP (fast for tw <= 10, before expensive k-sum/hierarchical)
        # Gate matches engine.py:999 (max_width=10): Python tw_dp at tw=11
        # takes 3-10+ min on n=40 graphs (e.g. Z(2,1) measured stuck >10 min,
        # May 24). C-ext is gated 5 <= tw <= 10. Graphs at tw=11 fall through
        # to chord-rule / spanning-tree paths.
        if graph.edge_count() >= 10:
            from ..graphs.treewidth import \
                compute_treewidth_tutte_if_applicable
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = compute_treewidth_tutte_if_applicable(full_mg, max_width=10)
            if tw_poly is not None:
                _log.record(EventType.TREEWIDTH_DP, "hybrid",
                            f"Treewidth DP: {graph.node_count()}n {graph.edge_count()}e",
                            graph=graph)
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
                        f"K-sum: {graph.node_count()}n {graph.edge_count()}e",
                        graph=graph)
            self._log(f"K-sum decomposition: {ksum_result.method}")
            return HybridSynthesisResult(
                polynomial=ksum_result.polynomial,
                method=ksum_result.method,
                recipe=ksum_result.recipe,
                verified=ksum_result.verified,
                tiles_used=getattr(ksum_result, 'tiles_used', 0),
            )

        return None

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
