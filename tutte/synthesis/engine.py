"""Creation-Expansion-Join Synthesis Engine.

This module implements the main synthesis algorithm for computing
Tutte polynomials using algebraic composition of known minors.

The algorithm:
1. Find all minors of input_graph from rainbow table
2. Select largest minor M by polynomial complexity
3. Tile input_graph with disjoint copies of M
4. Compute base polynomial: T = T(M₁) × T(M₂) × ... (multiplication)
5. For each edge connecting different tiles, apply k-join formula
6. Compute fringe = edges_in_cover - edges_in_input (over-coverage)
7. If fringe is empty: return polynomial
8. If fringe is small: adjust polynomial directly
9. Else: recurse on fringe, combine results

Base case: K₂ with T(x) = x

Alternative approach (algebraic):
Use polynomial division and GCD to decompose target polynomials into
known factors, without requiring graph-level tiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..polynomial import TuttePolynomial
from ..graph import Graph, MultiGraph
from ..lookup.core import RainbowTable, MinorEntry, load_default_table
from ..graphs.covering import (
    Cover, Tile, Fringe, InterCellInfo,
    find_disjoint_cover,
    compute_fringe,
    compute_inter_tile_edges,
    analyze_tile_connections,
    try_hierarchical_partition,
)
from ..validation import verify_spanning_trees
from ..graph import compute_signature
from ..graphs.series_parallel import compute_sp_tutte_if_applicable
from ..matroids.core import GraphicMatroid, FlatLattice, enumerate_flats_with_hasse
from ..matroids.parallel_connection import (
    BivariateLaurentPoly,
    theorem6_parallel_connection,
    theorem6_eval_interp,
    theorem6_product_lattice,
    theorem6_product_lattice_factored,
    theorem10_k_sum,
    theorem10_k_sum_via_theorem6,
    precompute_contractions,
    precompute_contractions_product,
    build_extended_cell_graph,
    sp_guided_precompute_contractions,
    sp_guided_precompute_contractions_product,
    sp_guided_precompute_contractions_product_grouped,
    MAX_PRODUCT_FLATS,
    EVAL_INTERP_THRESHOLD,
)

from .base import UnionFind, BaseMultigraphSynthesizer, SynthesisResult


def _is_complete_graph(g: Graph) -> bool:
    """Check if g is a complete graph K_n."""
    n = len(g.nodes)
    return len(g.edges) == n * (n - 1) // 2


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class SynthesisEngine(BaseMultigraphSynthesizer):
    """Main synthesis engine using creation-expansion-join algorithm."""

    def __init__(
        self,
        table: Optional[RainbowTable] = None,
        verbose: bool = False,
        auto_promote: bool = False,
    ):
        """Initialize synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
            auto_promote: If True, auto-promote synthesized simple graphs to the rainbow table
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose
        self.auto_promote = auto_promote
        self._cache: Dict[str, SynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}  # For multigraph polynomials
        self._fast_hash_set: Set[str] = set()  # Fast hashes of all cached multigraphs
        self._fast_hash_set_complete: bool = True  # True when _fast_hash_set covers all cache entries
        self._fast_simple_hash_set: Set[str] = set()  # Fast hashes of all cached simple graphs
        self._table_nm_set: Set[Tuple[int, int]] = {
            (e.node_count, e.edge_count) for e in self.table.entries.values()
        }
        self._inter_cell_cache: Dict[str, TuttePolynomial] = {}  # For inter-cell graph polynomials
        self._contraction_cache: Dict[str, TuttePolynomial] = {}  # For SP-guided contraction polynomials
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[Synth] {msg}", flush=True)

    def _promote_to_table(self, graph: Graph, cache_key: str, result: 'SynthesisResult') -> None:
        """Auto-promote a synthesized simple graph to the rainbow table.

        Only promotes if auto_promote is enabled and the key is not already in the table.
        """
        if not self.auto_promote or cache_key in self.table.entries:
            return
        entry = MinorEntry(
            name=f"auto_{graph.node_count()}n{graph.edge_count()}e_{cache_key[:8]}",
            polynomial=result.polynomial,
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            canonical_key=cache_key,
            spanning_trees=result.polynomial.num_spanning_trees(),
            num_terms=result.polynomial.num_terms(),
            graph=graph,
            signature=compute_signature(graph),
        )
        self.table.add_entry(entry)

    def save_rainbow_table(self, json_path: str = None, bin_path: str = None) -> None:
        """Save the rainbow table (with any auto-promoted entries) to disk.

        Args:
            json_path: Path for JSON format (default: tutte/data/lookup_table.json)
            bin_path: Path for binary format (default: tutte/data/lookup_table.bin)
        """
        import os
        from ..lookup.binary import save_binary_rainbow_table

        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if json_path is None:
            json_path = os.path.join(base_dir, 'lookup_table.json')
        if bin_path is None:
            bin_path = os.path.join(base_dir, 'lookup_table.bin')

        self.table.resort()
        self.table.save(json_path)
        save_binary_rainbow_table(self.table, bin_path)

    def save_multigraph_cache(self) -> None:
        """Save the multigraph polynomial cache to default location (binary + JSON)."""
        from ..lookup.core import save_default_multigraph_table
        save_default_multigraph_table(self._multigraph_cache)

    def load_multigraph_cache(self) -> int:
        """Load multigraph polynomial cache from default location.

        Tries binary format first, falls back to JSON.

        Returns the number of entries loaded.
        """
        from ..lookup.core import load_default_multigraph_table
        loaded = load_default_multigraph_table()
        count = 0
        for key, poly in loaded.items():
            if key not in self._multigraph_cache:
                self._multigraph_cache[key] = poly
                count += 1
        if count > 0:
            self._fast_hash_set_complete = False
        return count

    def save_contraction_cache(self) -> None:
        """Save the contraction polynomial cache to default location (binary + JSON).

        The contraction cache stores pre-computed T(M_i/Z) contractions from
        the SP-guided bottom-up approach, keyed by contracted multigraph
        canonical key. This allows reloading across sessions.
        """
        from ..lookup.core import save_default_contraction_cache
        save_default_contraction_cache(self._contraction_cache)

    def load_contraction_cache(self) -> int:
        """Load contraction polynomial cache from default location.

        Tries binary format first, falls back to JSON.

        Returns the number of entries loaded.
        """
        from ..lookup.core import load_default_contraction_cache
        loaded = load_default_contraction_cache()
        count = 0
        for key, poly in loaded.items():
            if key not in self._contraction_cache:
                self._contraction_cache[key] = poly
                count += 1
        return count

    def _collect_simple_intermediates(
        self,
        mg: MultiGraph,
        out: Dict[str, Graph],
    ) -> None:
        """Recursively trace batch reduction to collect simple graph intermediates.

        Follows the same reduction path as _synthesize_multigraph but only
        collects the Graph objects that will eventually need synthesis,
        without actually computing polynomials.
        """
        # Skip loops
        if mg.total_loop_count() > 0:
            mg = mg.remove_loops()

        # Skip parallel-only
        if mg.is_just_parallel_edges():
            return

        # Skip disconnected — recurse into components
        if not mg.is_connected():
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
            self._collect_simple_intermediates(comp1, out)
            self._collect_simple_intermediates(rest, out)
            return

        # Cut vertex
        cut = mg.has_cut_vertex()
        if cut is not None:
            components = mg.split_at_cut_vertex(cut)
            if len(components) > 1:
                for comp in components:
                    self._collect_simple_intermediates(comp, out)
                return

        # Cache check
        cache_key = mg.canonical_key()
        if cache_key in self._multigraph_cache:
            return

        # Simple graph — this is what we want to collect
        if mg.is_simple():
            simple = mg.to_simple_graph()
            if simple is not None:
                sk = simple.canonical_key()
                if sk not in self._cache and sk not in self.table.entries:
                    out[sk] = simple
                return

        # Batch reduce parallel — recurse into G_0 and G_c
        max_mult_edge = max(mg.edge_counts.keys(), key=lambda e: mg.edge_counts[e])
        if mg.edge_counts[max_mult_edge] > 1:
            u, v = max_mult_edge
            new_edge_counts = dict(mg.edge_counts)
            del new_edge_counts[max_mult_edge]
            mg_0 = MultiGraph(nodes=mg.nodes, edge_counts=new_edge_counts, loop_counts=mg.loop_counts)
            mg_c = mg_0.merge_nodes(u, v)
            if mg_0.in_same_component(u, v):
                self._collect_simple_intermediates(mg_0, out)
            self._collect_simple_intermediates(mg_c, out)

    def precompute_intermediate_simple_graphs(
        self,
        extended_cell: Graph,
        lattice: 'FlatLattice',
        shared_edges: list,
    ) -> int:
        """Pre-compute simple graph intermediates from flat contractions.

        For each flat in the lattice, contracts the extended cell graph,
        traces the batch reduction to find simple graph intermediates,
        deduplicates, sorts by size, and synthesizes smallest-first.
        Auto-promotes each result to the rainbow table.

        Returns the number of new entries added.
        """
        from ..matroids.parallel_connection import _contract_edges_in_graph

        # Collect all simple graph intermediates
        all_intermediates: Dict[str, Graph] = {}
        for z_idx in range(lattice.num_flats):
            z_flat = lattice.flat_by_idx(z_idx)
            if not z_flat:
                mg = MultiGraph.from_graph(extended_cell)
            else:
                mg = _contract_edges_in_graph(extended_cell, z_flat)
            self._collect_simple_intermediates(mg, all_intermediates)

        if not all_intermediates:
            return 0

        # Sort by edge count (smallest first for bottom-up synthesis)
        sorted_graphs = sorted(
            all_intermediates.items(),
            key=lambda kv: (kv[1].edge_count(), kv[1].node_count()),
        )

        self._log(f"Pre-computing {len(sorted_graphs)} intermediate simple graphs")

        count = 0
        for sk, simple in sorted_graphs:
            if sk in self.table.entries:
                continue
            result = self.synthesize(simple)
            count += 1
            if count % 50 == 0:
                self._log(f"  Pre-computed {count}/{len(sorted_graphs)}")

        # Resort the table once after all promotions
        if count > 0:
            self.table.resort()

        self._log(f"Pre-computed {count} new intermediate graphs")
        return count

    def synthesize(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> SynthesisResult:
        """Main entry point: compute Tutte polynomial via creation-expansion-join.

        Args:
            graph: Graph to compute polynomial for
            max_depth: Maximum recursion depth

        Returns:
            SynthesisResult with computed polynomial
        """
        # Check cache
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            self._log(f"Cache hit: {cache_key[:16]}...")
            return self._cache[cache_key]

        self._log(f"Synthesizing graph with {graph.node_count()} nodes, {graph.edge_count()} edges")

        # 1. Check rainbow table first
        cached = self.table.lookup(graph)
        if cached is not None:
            self._log("Direct rainbow table lookup")
            result = SynthesisResult(
                polynomial=cached,
                recipe=["Rainbow table lookup"],
                verified=True,
                method="lookup",
                minors_used={cache_key} if cache_key in self.table.entries else set(),
            )
            self._cache[cache_key] = result
            return result

        # 2. Handle base cases
        if graph.edge_count() == 0:
            # Empty graph (just vertices) -> T = 1
            result = SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph: T = 1"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            # Single edge -> T = x
            result = SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        # 3. Check if graph is disconnected
        components = graph.connected_components()
        if len(components) > 1:
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 4. Check for cut vertices (fast factorization before expensive operations)
        cut = graph.has_cut_vertex()
        if cut is not None:
            result = self._synthesize_via_cut_vertex(graph, cut, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 5. Check for cycle graphs: T(C_n) = x^{n-1} + x^{n-2} + ... + x + y
        if graph.node_count() >= 3 and graph.edge_count() == graph.node_count():
            if all(graph.degree(n) == 2 for n in graph.nodes):
                n = graph.node_count()
                coeffs = {(i, 0): 1 for i in range(1, n)}
                coeffs[(0, 1)] = 1
                self._log(f"Cycle C_{n}: direct formula")
                result = SynthesisResult(
                    polynomial=TuttePolynomial.from_coefficients(coeffs),
                    recipe=[f"Cycle C_{n}"],
                    verified=True,
                    method="cycle_formula",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 6. Try series-parallel O(n) computation
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            self._log("Series-parallel: O(n) computation")
            result = SynthesisResult(
                polynomial=sp_poly,
                recipe=["Series-parallel decomposition"],
                verified=True,
                method="series_parallel",
            )
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 7. Try treewidth DP on full graph
        if graph.edge_count() >= 10:
            from ..graphs.treewidth import compute_treewidth_tutte_if_applicable
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = compute_treewidth_tutte_if_applicable(full_mg, max_width=10)
            if tw_poly is not None:
                self._log(f"Treewidth DP: {graph.node_count()}n, {graph.edge_count()}e")
                result = SynthesisResult(
                    polynomial=tw_poly,
                    recipe=["Treewidth-based DP (full graph)"],
                    verified=True,
                    method="treewidth_dp",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 8. Try k-sum decomposition (k=2..7, vertex separators)
        if graph.edge_count() >= 6:
            result = self._try_ksum_decomposition(graph)
            if result is not None:
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 9. Try hierarchical tiling for graphs with repeating structure
        if graph.edge_count() >= 20:  # Only try for larger graphs
            result = self._try_hierarchical(graph, max_depth)
            if result is not None:
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 10. Try creation-expansion-join
        result = self._synthesize_connected(graph, max_depth)
        self._cache[cache_key] = result
        self._promote_to_table(graph, cache_key, result)
        return result

    def _synthesize_disconnected(
        self,
        components: List[Graph],
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize polynomial for disconnected graph.

        For disconnected graphs: T(G₁ ∪ G₂ ∪ ...) = T(G₁) × T(G₂) × ...
        """
        self._log(f"Disconnected graph with {len(components)} components")

        poly = TuttePolynomial.one()
        recipe = [f"Disconnected: {len(components)} components"]
        all_minors = set()

        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            all_minors |= comp_result.minors_used

        recipe.append(f"Product: {poly}")

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,  # Product formula is exact
            method="disjoint_union",
            minors_used=all_minors,
        )

    def _try_ksum_decomposition(
        self,
        graph: Graph,
    ) -> Optional[SynthesisResult]:
        """Try to decompose graph via k-vertex separator (k=2..7).

        For each k, looks for a set S of k vertices whose removal disconnects
        the graph. Separator vertices may be adjacent — only the *missing*
        clique edges need to be deleted via inclusion-exclusion.

        Searches all k values and picks the separator with the lowest cost,
        preferring full k-sums (0 missing edges → flat-grouped Theorem 6)
        over partial separators that require expensive brute-force.

        Returns SynthesisResult if successful, None otherwise.
        """
        # Collect all separators across k values
        candidates = []
        for k in range(2, 8):
            min_edges = {2: 3, 3: 6, 4: 12, 5: 15, 6: 20, 7: 28}
            if graph.edge_count() < min_edges.get(k, 3 * k):
                continue

            separator = self._find_vertex_separator(graph, k)
            if separator is not None:
                sv = sorted(separator)
                total_clique = k * (k - 1) // 2
                missing = sum(1 for i in range(k) for j in range(i+1, k)
                             if (min(sv[i], sv[j]), max(sv[i], sv[j])) not in graph.edges)
                candidates.append((k, separator, missing, total_clique))

        # For graphs with 15-30 nodes, also search for full k-sums using
        # NetworkX minimum vertex cut expanded to nearby k values.
        if 15 <= graph.node_count() <= 30:
            self._search_full_ksum_separators(graph, candidates)

        if not candidates:
            return None

        # Score candidates: full k-sums (0 missing) are vastly cheaper than partial
        # Full k-sum cost ∝ #flats(K_k) (polynomial).
        # Partial cost ∝ 2^missing (exponential in missing edges).
        # Prefer: (1) full k-sums by k (higher k = more flats but still polynomial),
        #         (2) partial with fewest missing edges
        def score(candidate):
            k, sep, missing, total = candidate
            if missing == total:
                # Full k-sum: score by -k (prefer higher k = bigger decomposition)
                return (0, -k)
            elif missing <= 10:
                # Partial: score by 2^missing (exponential cost)
                return (1, 2 ** missing)
            else:
                # Too expensive
                return (2, missing)

        candidates.sort(key=score)

        for k, separator, missing, total in candidates:
            if missing > 10 and missing != total:
                continue  # skip expensive partial separators
            result = self._apply_ksum(graph, separator, k)
            if result is not None:
                return result

        return None

    def _find_vertex_separator(
        self,
        graph: Graph,
        k: int,
    ) -> Optional[Tuple[int, ...]]:
        """Find k vertices whose removal disconnects the graph.

        Unlike the old _find_independent_vertex_separator, this allows
        separator vertices to be adjacent. Prefers separators with fewer
        existing edges (more missing clique edges = more work), but accepts
        any disconnecting k-set.

        Returns tuple of k vertices if found, None otherwise. Only checks a bounded
        number of candidates to avoid combinatorial explosion on large graphs.
        """
        from itertools import combinations

        nodes = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)

        candidates = nodes[:min(len(nodes), 20)]

        if len(candidates) < k:
            return None

        best = None
        best_missing = None  # prefer fewer missing edges (cheaper inclusion-exclusion)
        checked = 0
        max_checks = 200_000  # limit search time (~2s for 24-node graphs)

        for combo in combinations(candidates, k):
            checked += 1
            if checked > max_checks:
                break
            # Check if removing these k vertices disconnects the graph
            sep_set = set(combo)
            remaining = graph.nodes - sep_set
            if not remaining:
                continue

            start = next(iter(remaining))
            reached = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in reached:
                    continue
                reached.add(node)
                for nb in graph.neighbors(node):
                    if nb not in reached and nb not in sep_set:
                        stack.append(nb)

            if len(reached) < len(remaining):
                # Count missing clique edges (fewer = cheaper)
                missing = 0
                for i in range(k):
                    for j in range(i + 1, k):
                        edge = (min(combo[i], combo[j]), max(combo[i], combo[j]))
                        if edge not in graph.edges:
                            missing += 1

                if missing == 0:
                    # All clique edges present → graph has a clique separator.
                    # This is a cut-vertex generalization; handle via
                    # split at clique separator (simpler than k-sum).
                    # Skip for now — cut vertex path already handles k=1.
                    continue

                if best is None or missing < best_missing:
                    best = combo
                    best_missing = missing

        return best

    def _search_full_ksum_separators(
        self,
        graph: Graph,
        candidates: list,
    ) -> None:
        """Search for full k-sum separators (all clique edges missing) using
        NetworkX minimum vertex cut as a starting point, then expanding.

        Much faster than exhaustive search for finding FULL k-sums where
        the flat-grouped Theorem 6 path applies.
        """
        import networkx as nx
        from itertools import combinations

        # Build NX graph
        nxg = nx.Graph()
        nxg.add_nodes_from(graph.nodes)
        nxg.add_edges_from(graph.edges)

        # Get all minimum vertex cuts (typically very fast)
        try:
            kappa = nx.node_connectivity(nxg)
        except Exception:
            return

        # Search for independent separators of size kappa, kappa+1, kappa+2
        # "Independent" = no edges between separator vertices = full k-sum
        all_nodes = sorted(graph.nodes)
        for k in range(max(kappa, 5), min(kappa + 3, 8)):
            total_clique = k * (k - 1) // 2
            # Skip if we already have a full k-sum at this k
            if any(c[0] == k and c[2] == c[3] for c in candidates):
                continue

            # Try minimum node cuts first (fast)
            try:
                all_cuts = list(nx.all_node_cuts(nxg, k=kappa))
            except Exception:
                all_cuts = []

            for cut in all_cuts:
                if len(cut) > k:
                    continue
                # Expand cut to size k by adding neighboring nodes
                if len(cut) == k:
                    sep = tuple(sorted(cut))
                    missing = sum(1 for i in range(k) for j in range(i+1, k)
                                 if (min(sep[i], sep[j]), max(sep[i], sep[j])) not in graph.edges)
                    if missing == total_clique:
                        candidates.append((k, sep, missing, total_clique))
                        self._log(f"Found full {k}-sum separator via min-cut: {sep}")
                        break

            # Also try: independent sets of high-degree nodes
            # For Zephyr-like graphs, the separator is often even-numbered nodes
            checked = 0
            for combo in combinations(all_nodes, k):
                checked += 1
                if checked > 50_000:
                    break
                # Quick check: all pairs must be non-adjacent (full k-sum)
                all_independent = True
                for i in range(k):
                    for j in range(i + 1, k):
                        edge = (min(combo[i], combo[j]), max(combo[i], combo[j]))
                        if edge in graph.edges:
                            all_independent = False
                            break
                    if not all_independent:
                        break
                if not all_independent:
                    continue

                # Check if removing these vertices disconnects
                sep_set = set(combo)
                remaining = graph.nodes - sep_set
                if not remaining:
                    continue
                start = next(iter(remaining))
                reached = set()
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node in reached:
                        continue
                    reached.add(node)
                    for nb in graph.neighbors(node):
                        if nb not in reached and nb not in sep_set:
                            stack.append(nb)

                if len(reached) < len(remaining):
                    candidates.append((k, combo, total_clique, total_clique))
                    self._log(f"Found full {k}-sum separator: {combo}")
                    return  # Found one, no need to continue

    def _apply_ksum(
        self,
        graph: Graph,
        separator: Tuple[int, ...],
        k: int,
    ) -> Optional[SynthesisResult]:
        """Apply Theorem 10 to compute Tutte polynomial via vertex separator.

        Reconstructs the parallel connection by adding back missing K_k clique
        edges, then uses inclusion-exclusion to delete them.

        Supports both fully-independent separators (classic k-sum, all C(k,2)
        clique edges missing) and partially-adjacent separators (only missing
        edges are deleted).

        Args:
            graph: The graph with k-vertex separator
            separator: Tuple of k separator vertices
            k: Number of shared vertices

        Returns:
            SynthesisResult if successful, None otherwise
        """
        try:
            sv = sorted(separator)
            all_clique_edges = [(sv[i], sv[j]) for i in range(k) for j in range(i + 1, k)]
            missing_edges = [e for e in all_clique_edges if e not in graph.edges]
            num_missing = len(missing_edges)

            self._log(f"Found {k}-vertex separator: {separator} "
                      f"({num_missing}/{len(all_clique_edges)} clique edges missing)")

            # Build PC graph by adding missing clique edges
            pc_edges = graph.edges | frozenset(missing_edges)
            pc_graph = Graph(nodes=graph.nodes, edges=pc_edges)

            if num_missing == len(all_clique_edges):
                # Classic k-sum: all clique edges missing. Use optimized paths.
                pc_edge_count = graph.edge_count() + num_missing
                if pc_edge_count > 20:
                    poly = theorem10_k_sum_via_theorem6(graph, separator, k, self)
                else:
                    poly = theorem10_k_sum(pc_graph, all_clique_edges, self)
            elif num_missing <= 10:
                # Partial separator: only delete missing edges via brute-force.
                # With ≤10 missing edges, 2^10 = 1024 terms is fast.
                poly = theorem10_k_sum(pc_graph, missing_edges, self)
            else:
                # Too many missing edges for brute-force, not a full k-sum.
                # Skip — this separator isn't efficient to decompose.
                self._log(f"  {num_missing} missing edges too many for partial separator")
                return None

            if poly is None:
                return None

            # Verify
            if verify_spanning_trees(graph, poly):
                self._log(f"{k}-vertex separator decomposition via Theorem 10 verified!")
                return SynthesisResult(
                    polynomial=poly,
                    recipe=[f"{k}-vertex separator at {separator} "
                            f"({num_missing} missing)", f"T = {poly}"],
                    verified=True,
                    method=f"{k}sum_theorem10",
                )
            else:
                self._log(f"{k}-vertex separator Theorem 10 failed Kirchhoff")
                return None

        except (KeyError, IndexError) as e:
            self._log(f"{k}-vertex separator decomposition failed: {e}")
            return None

    def _synthesize_via_cut_vertex(
        self,
        graph: Graph,
        cut: int,
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize using cut vertex factorization.

        For graphs with cut vertices:
        T(G1 · G2 at v) = T(G1) × T(G2)

        where G1 and G2 are the components obtained by splitting at the cut vertex.
        This is much faster than general synthesis.
        """
        self._log(f"Cut vertex factorization at node {cut}")

        components = graph.split_at_cut_vertex(cut)
        recipe = [f"Cut vertex factorization at node {cut}: {len(components)} components"]

        poly = TuttePolynomial.one()
        all_minors = set()
        for i, comp in enumerate(components):
            comp_result = self.synthesize(comp, max_depth)
            poly = poly * comp_result.polynomial
            recipe.append(f"  Component {i+1}: {comp_result.polynomial}")
            all_minors |= comp_result.minors_used

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,  # Cut vertex formula is exact
            method="cut_vertex",
            minors_used=all_minors,
        )

    def _try_hierarchical(
        self,
        graph: Graph,
        max_depth: int
    ) -> Optional[SynthesisResult]:
        """Try hierarchical tiling for graphs with repeating cell structure.

        This approach:
        1. Finds candidate cells from the rainbow table
        2. Partitions nodes into cell groups (without full-graph VF2)
        3. Verifies each group ≅ cell using small-graph VF2 (fast!)
        4. Computes polynomial as cell^k × inter-cell adjustments

        Only used when the cell is substantial (covers at least 1/3 of edges).
        Returns None if no suitable tiling is found.
        """
        self._log("Trying hierarchical tiling...")

        result = try_hierarchical_partition(graph, self.table)
        if result is None:
            self._log("No hierarchical partition found")
            return None

        cell, partition, inter_info = result
        k = len(partition)

        # Only use hierarchical tiling when:
        # 1. We have at least 2 cells (otherwise just use direct synthesis)
        # 2. The cell polynomial is already known (not a trivial graph)
        # 3. The cells account for a meaningful portion of edges
        if k < 2:
            self._log(f"Only {k} cell found, not worth hierarchical approach")
            return None

        # Check that the cell has non-trivial structure (not just a path/tree)
        cell_edges = cell.edge_count
        cell_nodes = cell.node_count
        if cell_edges < cell_nodes:  # Tree or forest - trivial polynomial
            self._log(f"Cell {cell.name} has tree structure, not worth hierarchical")
            return None

        self._log(f"Found {k}-cell partition using {cell.name}")

        return self._synthesize_hierarchical(graph, cell, partition, inter_info, max_depth)

    def _synthesize_hierarchical(
        self,
        graph: Graph,
        cell: MinorEntry,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        max_depth: int
    ) -> SynthesisResult:
        """Compute polynomial using hierarchical cell decomposition.

        Algorithm:
        1. Base: T(disjoint cells) = T(cell)^k
        2. Try product formula: T(full) = T(cell)^k × product(T(inter_components))
           - This only works for specific structures (like Zephyr graphs)
           - Verify result; if wrong, fall back to edge-by-edge addition
        3. Fallback: Add inter-cell edges one-by-one using chord/bridge formulas

        Args:
            graph: Full graph
            cell: Cell pattern from rainbow table
            partition: List of node sets (one per cell)
            inter_info: Information about inter-cell edges
            max_depth: Maximum recursion depth

        Returns:
            SynthesisResult with computed polynomial
        """
        k = len(partition)
        recipe = [f"Hierarchical: {k} × {cell.name} cells"]
        all_minors = {cell.canonical_key}

        # Step 1: Base polynomial = T(cell)^k (disjoint cells)
        base_poly = TuttePolynomial.one()
        for _ in range(k):
            base_poly = base_poly * cell.polynomial

        recipe.append(f"Base: T({cell.name})^{k}")
        self._log(f"Base polynomial has {base_poly.num_terms()} terms")

        # Step 2: Try product formula for inter-cell edges
        # This formula works for Zephyr-type graphs: T(full) = T(cell)^k × Π T(inter_components)
        # But doesn't work for arbitrary partitions, so we verify and fall back if needed.
        if inter_info.edges:
            inter_graph = self._build_inter_cell_graph(graph, partition, inter_info)
            inter_components = inter_graph.connected_components()

            self._log(f"Inter-cell: {len(inter_components)} components, {inter_graph.edge_count()} edges")

            # Try product formula first (fast path for Zephyr-like structures)
            poly = base_poly
            inter_minors = set()
            for i, comp in enumerate(inter_components):
                comp_result = self.synthesize(comp, max_depth)
                poly = poly * comp_result.polynomial
                inter_minors |= comp_result.minors_used

            # Verify - product formula only works for specific structures
            if verify_spanning_trees(graph, poly):
                self._log("Product formula verified")
                recipe.append(f"Inter-cell: {len(inter_components)} components")
                recipe.append("Product formula: T(cells)^k × Π T(inter_components)")

                return SynthesisResult(
                    polynomial=poly,
                    recipe=recipe,
                    verified=True,
                    method="hierarchical_tiling",
                    tiles_used=k,
                    fringe_edges=0,
                    minors_used=all_minors | inter_minors,
                )

            # Product formula failed - try treewidth DP on full graph
            self._log("Product formula failed, trying treewidth DP on full graph")
            from ..graphs.treewidth import compute_treewidth_tutte_if_applicable as _tw_compute
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = _tw_compute(full_mg, max_width=10)
            if tw_poly is not None:
                self._log(f"Treewidth DP solved full graph: {graph.node_count()}n, {graph.edge_count()}e")
                recipe.append("Treewidth-based DP (full graph, after product formula)")
                return SynthesisResult(
                    polynomial=tw_poly,
                    recipe=recipe,
                    verified=True,
                    method="treewidth_dp",
                    tiles_used=k,
                    fringe_edges=0,
                    minors_used=all_minors,
                )

            # Try Theorem 6 parallel connection for 2-cell case
            if len(partition) == 2:
                self._log("Trying Theorem 6 parallel connection")
                recipe.append("Product formula invalid, trying Theorem 6")

                # Try product-lattice Theorem 6 if inter-cell graph is disconnected
                if len(inter_components) > 1:
                    # Try staged per-component Theorem 6 FIRST — avoids
                    # expensive product-lattice computation by processing one
                    # component via Theorem 6 and adding remaining edges
                    # edge-by-edge. Especially effective when components are SP.
                    pc_poly = self._try_staged_theorem6(
                        graph, partition, inter_info, cell, inter_components,
                    )
                    if pc_poly is not None:
                        poly = pc_poly
                        recipe.append("Staged per-component Theorem 6 succeeded")
                        self._log(f"Final polynomial has {poly.num_terms()} terms")
                        verified = verify_spanning_trees(graph, poly)
                        return SynthesisResult(
                            polynomial=poly,
                            recipe=recipe,
                            verified=verified,
                            method="staged_theorem6",
                            tiles_used=k,
                            fringe_edges=0,
                            minors_used=all_minors,
                        )
                    self._log("Staged Theorem 6 failed")

                    # Fallback: product-lattice Theorem 6
                    pc_poly = self._try_product_lattice_theorem6(
                        graph, partition, inter_info, cell, inter_components,
                    )
                    if pc_poly is not None:
                        poly = pc_poly
                        recipe.append("Product-lattice Theorem 6 succeeded")
                        self._log(f"Final polynomial has {poly.num_terms()} terms")
                        verified = verify_spanning_trees(graph, poly)
                        return SynthesisResult(
                            polynomial=poly,
                            recipe=recipe,
                            verified=verified,
                            method="product_lattice_theorem6",
                            tiles_used=k,
                            fringe_edges=0,
                            minors_used=all_minors,
                        )
                    self._log("Product-lattice Theorem 6 failed")

                    # Fallback: SP-guided bottom-up product-lattice Theorem 6
                    pc_poly = self._try_sp_guided_theorem6(
                        graph, partition, inter_info, cell, inter_components,
                    )
                    if pc_poly is not None:
                        poly = pc_poly
                        recipe.append("SP-guided Theorem 6 succeeded")
                        self._log(f"Final polynomial has {poly.num_terms()} terms")
                        verified = verify_spanning_trees(graph, poly)
                        return SynthesisResult(
                            polynomial=poly,
                            recipe=recipe,
                            verified=verified,
                            method="sp_guided_theorem6",
                            tiles_used=k,
                            fringe_edges=0,
                            minors_used=all_minors,
                        )
                    self._log("SP-guided Theorem 6 failed")

                # Fall back to standard Theorem 6 (single matroid)
                pc_poly = self._try_parallel_connection(graph, partition, inter_info, cell, base_poly)
                if pc_poly is not None:
                    poly = pc_poly
                    recipe.append("Theorem 6 parallel connection succeeded")
                    self._log(f"Final polynomial has {poly.num_terms()} terms")
                    verified = verify_spanning_trees(graph, poly)
                    return SynthesisResult(
                        polynomial=poly,
                        recipe=recipe,
                        verified=verified,
                        method="parallel_connection",
                        tiles_used=k,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )
                self._log("Theorem 6 failed, falling back to edge-by-edge")
                recipe.append("Theorem 6 failed, using optimized edge-by-edge")

            # Fall back to component-optimized edge addition
            self._log("Using optimized edge-by-edge addition")
            poly = self._add_inter_cell_edges_optimized(base_poly, graph, partition, inter_info, cell)
        else:
            poly = base_poly

        self._log(f"Final polynomial has {poly.num_terms()} terms")

        # Verify result
        verified = verify_spanning_trees(graph, poly)

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=verified,
            method="hierarchical_tiling",
            tiles_used=k,
            fringe_edges=0,
            minors_used=all_minors,
        )

    def _build_inter_cell_graph(
        self,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo
    ) -> Graph:
        """Build the subgraph of just inter-cell edges.

        This creates a graph containing only:
        - Nodes that are endpoints of inter-cell edges
        - The inter-cell edges themselves

        Args:
            graph: Full graph
            partition: List of node sets (one per cell)
            inter_info: Information about inter-cell edges

        Returns:
            Graph containing only inter-cell structure
        """
        inter_nodes: Set[int] = set()
        inter_edges: Set[Tuple[int, int]] = set()

        for u, v in inter_info.edges:
            inter_nodes.add(u)
            inter_nodes.add(v)
            edge = (min(u, v), max(u, v))
            inter_edges.add(edge)

        return Graph(
            nodes=frozenset(inter_nodes),
            edges=frozenset(inter_edges)
        )

    def _try_parallel_connection(
        self,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        cell: MinorEntry,
        base_poly: TuttePolynomial,
    ) -> Optional[TuttePolynomial]:
        """Try Theorem 6 (Bonin-de Mier) for 2-cell decomposition.

        Steps:
        1. Build inter-cell graph -> GraphicMatroid N
        2. Enumerate flats with Hasse diagram
        3. Build FlatLattice with pre-built Hasse
        4. Build extended cell graphs (cell + inter-cell edges)
        5. Precompute T(M_i/Z) for all flats Z
        6. Apply Theorem 6
        7. Verify via Kirchhoff

        Args:
            graph: Full graph
            partition: List of 2 node sets
            inter_info: Information about inter-cell edges
            cell: Cell pattern from rainbow table
            base_poly: T(cell)^k polynomial

        Returns:
            TuttePolynomial if successful, None otherwise
        """
        if len(partition) != 2:
            return None

        try:
            # Step 1: Build inter-cell graph and matroid
            inter_graph = self._build_inter_cell_graph(graph, partition, inter_info)
            if inter_graph.edge_count() == 0:
                return None

            # Theorem 6 requires the shared matroid to be a modular flat,
            # which is only guaranteed for complete graphs K_n.
            if not _is_complete_graph(inter_graph):
                self._log(f"Inter-cell graph is not complete ({inter_graph.node_count()}n, "
                          f"{inter_graph.edge_count()}e), skipping Theorem 6")
                return None

            matroid_N = GraphicMatroid(inter_graph)
            r_N = matroid_N.rank()

            self._log(f"Inter-cell matroid: rank {r_N}, {inter_graph.edge_count()} edges")

            # Guard: skip if flat lattice would be too large
            MAX_PARALLEL_CONNECTION_FLATS = 5_000

            # Early bail: high-rank matroids with many edges have enormous flat counts.
            # Per-component flat counts of 17K each (rank 11) already make the combined
            # case hopeless. Skip flat enumeration entirely for large matroids.
            if r_N > 12 or inter_graph.edge_count() > 20:
                self._log(f"Inter-cell matroid too large (rank {r_N}, {inter_graph.edge_count()} edges), skipping Theorem 6")
                return None

            inter_key = inter_graph.canonical_key()

            # Step 2-3: Build FlatLattice (check cache first)
            entry = self.table.entries.get(inter_key)
            if entry is not None and entry.flat_data is not None:
                self._log("Using cached flat lattice data")
                lattice = FlatLattice.from_flat_lattice_data(matroid_N, entry.flat_data)
            else:
                flats, ranks, upper_covers = enumerate_flats_with_hasse(matroid_N)
                self._log(f"Enumerated {len(flats)} flats")
                lattice = FlatLattice(
                    matroid_N,
                    flats=flats,
                    ranks=ranks,
                    upper_covers=upper_covers,
                )

            if lattice.num_flats > MAX_PARALLEL_CONNECTION_FLATS:
                self._log(f"Flat count {lattice.num_flats} exceeds limit {MAX_PARALLEL_CONNECTION_FLATS}, skipping Theorem 6")
                return None

            # Step 4: Build extended cell graphs
            inter_edge_list = list(inter_info.edges)

            ext1, shared1 = build_extended_cell_graph(
                graph, partition[0], inter_edge_list,
            )
            ext2, shared2 = build_extended_cell_graph(
                graph, partition[1], inter_edge_list,
            )

            self._log(f"Extended cell 1: {ext1.node_count()} nodes, {ext1.edge_count()} edges")
            self._log(f"Extended cell 2: {ext2.node_count()} nodes, {ext2.edge_count()} edges")

            # Step 5: Precompute T(M_i/Z) for all flats Z
            t_m1 = precompute_contractions(ext1, shared1, lattice, self)
            t_m2 = precompute_contractions(ext2, shared2, lattice, self)

            # Step 6: Apply Theorem 6
            poly = theorem6_parallel_connection(lattice, t_m1, t_m2, r_N)

            # Step 7: Verify
            if verify_spanning_trees(graph, poly):
                self._log("Theorem 6 verified!")
                # Cache flat lattice data for future use
                if entry is not None and entry.flat_data is None:
                    entry.flat_data = lattice.to_flat_lattice_data()
                return poly
            else:
                self._log("Theorem 6 result failed Kirchhoff verification")
                return None

        except Exception as e:
            self._log(f"Theorem 6 failed with exception: {e}")
            return None

    def _try_product_lattice_theorem6(
        self,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        cell: MinorEntry,
        components: List[Graph],
    ) -> Optional[TuttePolynomial]:
        """Try Theorem 6 with product-lattice optimization for disconnected inter-cell matroid.

        When the inter-cell graph has multiple disconnected components, the shared
        matroid N = N1 + N2 + ... is a direct sum. The flat lattice decomposes as
        L(N) = L(N1) x L(N2) x ..., avoiding enumeration of the full (huge) lattice.

        Feasibility gate: product of component flat counts must be < MAX_PRODUCT_FLATS.

        Args:
            graph: Full graph
            partition: List of 2 node sets
            inter_info: Information about inter-cell edges
            cell: Cell pattern from rainbow table
            components: List of inter-cell graph connected components

        Returns:
            TuttePolynomial if successful, None otherwise
        """
        if len(partition) != 2 or len(components) < 2:
            return None

        try:
            # Theorem 6 requires each shared component to be a modular flat (complete graph)
            for i, comp in enumerate(components):
                if not _is_complete_graph(comp):
                    self._log(f"Component {i} is not complete ({comp.node_count()}n, "
                              f"{comp.edge_count()}e), skipping product-lattice Theorem 6")
                    return None

            # Build matroid and flat lattice for each component
            component_data = []  # (matroid, lattice, rank)
            total_product = 1

            for i, comp in enumerate(components):
                matroid = GraphicMatroid(comp)
                r = matroid.rank()

                # Check cache for flat data
                comp_key = comp.canonical_key()
                entry = self.table.entries.get(comp_key)

                if entry is not None and entry.flat_data is not None:
                    self._log(f"Component {i}: using cached flat lattice")
                    lattice = FlatLattice.from_flat_lattice_data(matroid, entry.flat_data)
                else:
                    flats, ranks, uc = enumerate_flats_with_hasse(matroid)
                    self._log(f"Component {i}: {len(flats)} flats (rank {r})")
                    lattice = FlatLattice(matroid, flats=flats, ranks=ranks, upper_covers=uc)

                    # Cache for future use
                    if entry is not None and entry.flat_data is None:
                        entry.flat_data = lattice.to_flat_lattice_data()

                component_data.append((matroid, lattice, r))
                total_product *= lattice.num_flats

            self._log(f"Product flat count: {total_product}")

            if total_product > MAX_PRODUCT_FLATS:
                self._log(f"Product flat count {total_product} exceeds limit {MAX_PRODUCT_FLATS}")
                return None

            # Currently only support 2 components
            if len(components) != 2:
                self._log(f"Product-lattice only supports 2 components, got {len(components)}")
                return None

            mat1, lat1, r1 = component_data[0]
            mat2, lat2, r2 = component_data[1]

            # Build extended cell graphs (cell + ALL inter-cell edges)
            inter_edge_list = list(inter_info.edges)
            ext1, shared1 = build_extended_cell_graph(graph, partition[0], inter_edge_list)
            ext2, shared2 = build_extended_cell_graph(graph, partition[1], inter_edge_list)

            self._log(f"Extended cell 1: {ext1.node_count()}n, {ext1.edge_count()}e")
            self._log(f"Extended cell 2: {ext2.node_count()}n, {ext2.edge_count()}e")

            # Precompute T(M_i/(Z1∪Z2)) for all flat pairs
            self._log("Precomputing contractions for cell 1...")
            t_m1 = precompute_contractions_product(ext1, shared1, lat1, lat2, self)
            self._log(f"Cell 1: {len(t_m1)} contraction results")

            self._log("Precomputing contractions for cell 2...")
            t_m2 = precompute_contractions_product(ext2, shared2, lat1, lat2, self)
            self._log(f"Cell 2: {len(t_m2)} contraction results")

            # Apply product-lattice Theorem 6
            self._log("Applying product-lattice Theorem 6...")
            poly = theorem6_product_lattice(lat1, lat2, t_m1, t_m2, r1, r2)

            # Verify
            if verify_spanning_trees(graph, poly):
                self._log("Product-lattice Theorem 6 verified!")
                return poly
            else:
                self._log("Product-lattice Theorem 6 failed Kirchhoff verification")
                return None

        except Exception as e:
            self._log(f"Product-lattice Theorem 6 failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _try_sp_guided_theorem6(
        self,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        cell: MinorEntry,
        components: List[Graph],
    ) -> Optional[TuttePolynomial]:
        """Try Theorem 6 with SP-guided bottom-up contraction cache.

        For series-parallel inter-cell components, builds the flat lattice and
        contraction polynomials bottom-up through the SP decomposition tree.
        This avoids synthesizing the full ~17K contractions from scratch by
        progressively composing from single-edge base cases.

        Args:
            graph: Full graph
            partition: List of 2 node sets
            inter_info: Information about inter-cell edges
            cell: Cell pattern from rainbow table
            components: List of inter-cell graph connected components

        Returns:
            TuttePolynomial if successful, None otherwise
        """
        from ..graphs.series_parallel import is_series_parallel

        if len(partition) != 2 or len(components) < 2:
            return None

        try:
            # Theorem 6 requires each shared component to be a modular flat (complete graph)
            for i, comp in enumerate(components):
                if not _is_complete_graph(comp):
                    self._log(f"Component {i} is not complete ({comp.node_count()}n, "
                              f"{comp.edge_count()}e), skipping SP-guided Theorem 6")
                    return None

            # Check all components are series-parallel
            for i, comp in enumerate(components):
                if not is_series_parallel(comp):
                    self._log(f"Component {i} is not series-parallel, skipping SP-guided")
                    return None

            # Build matroid and flat lattice for each component
            component_data = []
            total_product = 1

            for i, comp in enumerate(components):
                matroid = GraphicMatroid(comp)
                r = matroid.rank()
                flats, ranks, uc = enumerate_flats_with_hasse(matroid)
                lattice = FlatLattice(matroid, flats=flats, ranks=ranks, upper_covers=uc)
                component_data.append((matroid, lattice, r))
                total_product *= lattice.num_flats
                self._log(f"SP-guided component {i}: {lattice.num_flats} flats (rank {r})")

            self._log(f"SP-guided product flat count: {total_product}")

            # Currently only support 2 components
            if len(components) != 2:
                self._log(f"SP-guided only supports 2 components, got {len(components)}")
                return None

            mat1, lat1, r1 = component_data[0]
            mat2, lat2, r2 = component_data[1]

            # Build extended cell graphs
            inter_edge_list = list(inter_info.edges)
            ext1, shared1 = build_extended_cell_graph(graph, partition[0], inter_edge_list)
            ext2, shared2 = build_extended_cell_graph(graph, partition[1], inter_edge_list)

            self._log(f"Extended cell 1: {ext1.node_count()}n, {ext1.edge_count()}e")
            self._log(f"Extended cell 2: {ext2.node_count()}n, {ext2.edge_count()}e")

            # Use grouped SP-guided contraction precomputation
            self._log("SP-guided grouped precomputing contractions for cell 1...")
            grouped_1 = sp_guided_precompute_contractions_product_grouped(
                components, ext1, [lat1, lat2], self, verbose=self.verbose,
                persistent_cache=self._contraction_cache,
            )
            G1_1 = len(grouped_1.z1_groups)
            self._log(f"Cell 1: G1={G1_1} unique Z1 groups")

            self._log("SP-guided grouped precomputing contractions for cell 2...")
            grouped_2 = sp_guided_precompute_contractions_product_grouped(
                components, ext2, [lat1, lat2], self, verbose=self.verbose,
                persistent_cache=self._contraction_cache,
            )
            G1_2 = len(grouped_2.z1_groups)
            self._log(f"Cell 2: G1={G1_2} unique Z1 groups")

            # Apply factored product-lattice Theorem 6
            self._log(f"Applying factored Theorem 6 ({G1_1}×{G1_2} group pairs)...")
            poly = theorem6_product_lattice_factored(
                lat1, lat2, grouped_1, grouped_2, r1, r2,
            )

            # Verify
            if verify_spanning_trees(graph, poly):
                self._log("SP-guided Theorem 6 verified!")
                return poly
            else:
                self._log("SP-guided Theorem 6 failed Kirchhoff verification")
                return None

        except Exception as e:
            self._log(f"SP-guided Theorem 6 failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _try_staged_theorem6(
        self,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        cell: MinorEntry,
        components: List[Graph],
    ) -> Optional[TuttePolynomial]:
        """Apply Theorem 6 per-component, then add remaining edges via edge-by-edge.

        For disconnected inter-cell graphs with components too large for the full
        product-lattice approach, process one component at a time:

        1. Pick the component with fewest flats
        2. Apply Theorem 6 with that component as the shared matroid
           - Extended cells include only that component's inter-cell edges
        3. This gives T(G_partial) where G_partial = cells + component's edges
        4. Add remaining components' edges via edge-by-edge chord addition

        Args:
            graph: Full graph
            partition: List of 2 node sets
            inter_info: Information about inter-cell edges
            cell: Cell pattern from rainbow table
            components: Disconnected inter-cell graph components

        Returns:
            TuttePolynomial if successful, None otherwise
        """
        if len(partition) != 2:
            return None

        from ..graphs.series_parallel import is_series_parallel

        MAX_STAGED_FLATS = 2_000  # Max flats for non-SP components
        MAX_STAGED_FLATS_SP = 20_000  # Higher limit for SP components (SP-guided bottom-up)

        try:
            # Enumerate flats for each component and pick the smallest
            component_info = []
            for i, comp in enumerate(components):
                matroid = GraphicMatroid(comp)
                r = matroid.rank()
                flats, ranks, uc = enumerate_flats_with_hasse(matroid)
                lattice = FlatLattice(matroid, flats=flats, ranks=ranks, upper_covers=uc)
                sp = is_series_parallel(comp)
                component_info.append((i, comp, matroid, lattice, r, list(comp.edges), sp))
                self._log(f"Staged component {i}: {lattice.num_flats} flats"
                          f"{' (SP)' if sp else ''}")

            # Sort by flat count to pick the smallest
            component_info.sort(key=lambda x: x[3].num_flats)

            # Check if any component is feasible for Theorem 6
            best_idx, best_comp, best_matroid, best_lattice, best_rank, best_edges, best_sp = component_info[0]
            flat_limit = MAX_STAGED_FLATS_SP if best_sp else MAX_STAGED_FLATS
            if best_lattice.num_flats > flat_limit:
                self._log(f"Smallest component has {best_lattice.num_flats} flats, "
                          f"exceeds staged limit ({flat_limit})")
                return None

            self._log(f"Using component {best_idx} ({best_lattice.num_flats} flats"
                      f"{', SP-guided' if best_sp else ''}) for staged Theorem 6")

            # Try treewidth DP on partial graph first (avoids Theorem 6 modularity requirement)
            remaining_edges_for_tw = set()
            for idx, comp, _, _, _, edges, _sp in component_info[1:]:
                remaining_edges_for_tw.update(edges)
            partial_graph_edges_tw = graph.edges - frozenset(
                (min(u, v), max(u, v)) for u, v in remaining_edges_for_tw
            )
            partial_graph_tw = Graph(nodes=graph.nodes, edges=partial_graph_edges_tw)
            from ..graphs.treewidth import compute_treewidth_tutte_if_applicable
            partial_mg_tw = MultiGraph.from_graph(partial_graph_tw)
            partial_poly_tw = compute_treewidth_tutte_if_applicable(partial_mg_tw, max_width=10)
            if partial_poly_tw is not None:
                if verify_spanning_trees(partial_graph_tw, partial_poly_tw):
                    self._log(f"Treewidth DP on partial graph succeeded ({len(partial_graph_edges_tw)}e)")
                    # Add remaining edges via edge-by-edge
                    all_nodes = set(graph.nodes)
                    edge_counts_tw: Dict[Tuple[int, int], int] = {}
                    for u, v in partial_graph_edges_tw:
                        edge = (min(u, v), max(u, v))
                        edge_counts_tw[edge] = 1
                    current_mg = MultiGraph(
                        nodes=frozenset(all_nodes),
                        edge_counts=edge_counts_tw,
                        loop_counts={},
                    )
                    poly = partial_poly_tw
                    remaining_sorted = sorted(remaining_edges_for_tw)
                    for i, (u, v) in enumerate(remaining_sorted):
                        merged = current_mg.merge_nodes(u, v)
                        merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
                        poly = poly + merged_poly
                        edge = (min(u, v), max(u, v))
                        new_edge_counts = dict(current_mg.edge_counts)
                        new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
                        current_mg = MultiGraph(
                            nodes=current_mg.nodes,
                            edge_counts=new_edge_counts,
                            loop_counts=current_mg.loop_counts,
                        )
                        if (i + 1) % 5 == 0:
                            self._log(f"  Treewidth partial edge-by-edge: {i+1}/{len(remaining_sorted)}")
                    return poly
                else:
                    self._log("Treewidth DP partial result failed verification")

            # Theorem 6 requires the shared matroid to be a modular flat (complete graph)
            if not _is_complete_graph(best_comp):
                self._log(f"Best component is not complete ({best_comp.node_count()}n, "
                          f"{best_comp.edge_count()}e), skipping staged Theorem 6")
                return None

            # Build partial extended cell graphs (cell + ONLY this component's edges)
            partial_inter_edges = best_edges
            ext1, shared1 = build_extended_cell_graph(graph, partition[0], partial_inter_edges)
            ext2, shared2 = build_extended_cell_graph(graph, partition[1], partial_inter_edges)

            self._log(f"Partial extended cell 1: {ext1.node_count()}n, {ext1.edge_count()}e")
            self._log(f"Partial extended cell 2: {ext2.node_count()}n, {ext2.edge_count()}e")

            # Precompute T(M_i/Z) for all flats Z of this component's matroid
            if best_sp:
                self._log("Using SP-guided bottom-up precomputation...")
                t_m1 = sp_guided_precompute_contractions(
                    best_comp, ext1, best_lattice, self,
                    verbose=self.verbose,
                    persistent_cache=self._contraction_cache,
                )
                t_m2 = sp_guided_precompute_contractions(
                    best_comp, ext2, best_lattice, self,
                    verbose=self.verbose,
                    persistent_cache=self._contraction_cache,
                )
            else:
                t_m1 = precompute_contractions(ext1, shared1, best_lattice, self)
                t_m2 = precompute_contractions(ext2, shared2, best_lattice, self)

            # Apply Theorem 6 to get T(G_partial) = T(cells + this component's edges)
            if best_lattice.num_flats > EVAL_INTERP_THRESHOLD:
                # Use evaluation-interpolation to avoid fraction blow-up
                # Degree bounds: x-degree <= rank(G), y-degree <= nullity(G)
                # For partial graph: rank = |V|-1 (connected), nullity = |E|-|V|+1
                partial_n = len(graph.nodes)
                partial_e = sum(1 for u, v in graph.edges
                                if u in partition[0] and v in partition[0]
                                or u in partition[1] and v in partition[1]) + len(best_edges)
                max_x_deg = partial_n - 1
                max_y_deg = partial_e - partial_n + 1
                self._log(f"Applying Theorem 6 via eval-interp "
                          f"({best_lattice.num_flats} flats, "
                          f"degree bounds: x≤{max_x_deg}, y≤{max_y_deg})...")
                partial_poly = theorem6_eval_interp(
                    best_lattice, t_m1, t_m2, best_rank,
                    max_x_degree=max_x_deg, max_y_degree=max_y_deg,
                )
            else:
                self._log("Applying Theorem 6 for staged component...")
                partial_poly = theorem6_parallel_connection(
                    best_lattice, t_m1, t_m2, best_rank,
                )

            # Build G_partial = graph minus remaining components' edges
            remaining_edges = set()
            for idx, comp, _, _, _, edges, _sp in component_info[1:]:
                remaining_edges.update(edges)

            # Verify partial result: G_partial = graph - remaining_edges
            partial_graph_edges = graph.edges - frozenset(
                (min(u, v), max(u, v)) for u, v in remaining_edges
            )
            partial_graph = Graph(nodes=graph.nodes, edges=partial_graph_edges)
            if not verify_spanning_trees(partial_graph, partial_poly):
                self._log("Staged Theorem 6 partial result failed verification")
                return None

            self._log(f"Staged Theorem 6 partial verified! Adding {len(remaining_edges)} remaining edges")

            # Now add remaining edges via edge-by-edge
            # Build multigraph from G_partial
            all_nodes = set(graph.nodes)
            edge_counts: Dict[Tuple[int, int], int] = {}
            for u, v in partial_graph_edges:
                edge = (min(u, v), max(u, v))
                edge_counts[edge] = 1

            current_mg = MultiGraph(
                nodes=frozenset(all_nodes),
                edge_counts=edge_counts,
                loop_counts={},
            )

            # All remaining edges are chords (G_partial is connected)
            poly = partial_poly
            remaining_sorted = sorted(remaining_edges)
            for i, (u, v) in enumerate(remaining_sorted):
                merged = current_mg.merge_nodes(u, v)
                merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
                poly = poly + merged_poly

                edge = (min(u, v), max(u, v))
                new_edge_counts = dict(current_mg.edge_counts)
                new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
                current_mg = MultiGraph(
                    nodes=current_mg.nodes,
                    edge_counts=new_edge_counts,
                    loop_counts=current_mg.loop_counts,
                )
                if (i + 1) % 5 == 0:
                    self._log(f"  Staged edge-by-edge: {i+1}/{len(remaining_sorted)}")

            return poly

        except Exception as e:
            self._log(f"Staged Theorem 6 failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_inter_cell_edges_optimized(
        self,
        base_poly: TuttePolynomial,
        graph: Graph,
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        cell: MinorEntry
    ) -> TuttePolynomial:
        """Add inter-cell edges using single-pass with running multigraph.

        Uses UnionFind for O(alpha(n)) bridge/chord classification.
        Maintains a running multigraph across ALL inter-cell edges so that
        T(G/{u,v}) is computed on the correct graph state.

        Strategy:
        1. Build initial multigraph from ALL intra-cell edges
        2. Initialize union-find with cells pre-unioned
        3. Classify each inter-cell edge: bridge if uf.find(u) != uf.find(v)
        4. Process bridges first (cheap: poly *= x), then chords
        5. For 2-cell graphs with automorphism: use symmetric chord ordering
           - Pair chords by symmetry, process pairs consecutively
           - For each pair at symmetric state G: T(G/{e₁}) and T((G+e₁)/{e₂})
             are computed in parallel (independent subproblems)
        6. Running multigraph updated after each edge

        Args:
            base_poly: T(cell)^k polynomial
            graph: Full graph
            partition: List of node sets (one per cell)
            inter_info: Information about inter-cell edges
            cell: Cell pattern from rainbow table

        Returns:
            Combined polynomial
        """
        # Build initial multigraph from ALL intra-cell edges
        all_nodes: Set[int] = set()
        edge_counts: Dict[Tuple[int, int], int] = {}
        for cell_nodes in partition:
            all_nodes.update(cell_nodes)
            for u, v in graph.edges:
                if u in cell_nodes and v in cell_nodes:
                    edge = (min(u, v), max(u, v))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

        current_mg = MultiGraph(
            nodes=frozenset(all_nodes),
            edge_counts=edge_counts,
            loop_counts={}
        )

        # Initialize union-find: union all nodes within each cell
        uf = UnionFind(all_nodes)
        for cell_nodes in partition:
            nodes_list = list(cell_nodes)
            for i in range(1, len(nodes_list)):
                uf.union(nodes_list[0], nodes_list[i])

        # Classify inter-cell edges into bridges and chords
        bridges = []
        chords = []
        for u, v in inter_info.edges:
            if uf.find(u) != uf.find(v):
                bridges.append((u, v))
                uf.union(u, v)  # Now connected
            else:
                chords.append((u, v))

        self._log(f"Inter-cell edges: {len(bridges)} bridges, {len(chords)} chords")

        poly = base_poly

        # Process bridges first (cheap: T(G+e) = x * T(G))
        for u, v in bridges:
            poly = TuttePolynomial.x() * poly
            edge = (min(u, v), max(u, v))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts
            )

        # Compare symmetric vs treewidth-minimizing chord orderings
        symmetric_order = None
        automorphism = None
        tw_order = None

        if len(partition) == 2 and len(chords) > 2:
            from .symmetric import build_symmetric_chord_order
            symmetric_order, automorphism = build_symmetric_chord_order(
                chords, graph, partition
            )
            if automorphism is None:
                symmetric_order = None

        # Try treewidth-minimizing order for non-trivial chord counts
        if len(chords) > 2:
            from .symmetric import build_treewidth_minimizing_chord_order
            self._log(f"Computing treewidth-minimizing chord order for {len(chords)} chords...")
            tw_order = build_treewidth_minimizing_chord_order(
                chords, current_mg, max_width=10
            )

        # Estimate costs and pick the best ordering
        if tw_order is not None and symmetric_order is not None:
            tw_cost = self._estimate_chord_order_cost(current_mg, tw_order)
            sym_cost = self._estimate_chord_order_cost(current_mg, symmetric_order)
            self._log(f"Chord order costs: treewidth-min={tw_cost:.0f}, symmetric={sym_cost:.0f}")
            if tw_cost < sym_cost:
                self._log(f"Using treewidth-minimizing chord order")
                return self._process_chords_sequential(poly, current_mg, tw_order)
            else:
                self._log(f"Using symmetric chord ordering")
                return self._process_chords_symmetric(
                    poly, current_mg, symmetric_order, automorphism, partition
                )
        elif tw_order is not None:
            self._log(f"Using treewidth-minimizing chord order")
            return self._process_chords_sequential(poly, current_mg, tw_order)
        elif symmetric_order is not None:
            self._log(f"Using symmetric chord ordering")
            return self._process_chords_symmetric(
                poly, current_mg, symmetric_order, automorphism, partition
            )

        # Standard chord processing (T(G+e) = T(G) + T(G/{u,v}))
        return self._process_chords_sequential(poly, current_mg, chords)

    def _estimate_chord_order_cost(
        self,
        current_mg: MultiGraph,
        chords: List[Tuple[int, int]],
    ) -> float:
        """Estimate total DP cost for a chord ordering.

        Simulates the chord-by-chord process, computing tree decomposition
        cost for each merged graph G/{u,v}.
        """
        from ..graphs.treewidth import compute_best_tree_decomposition, estimate_dp_cost

        total_cost = 0.0
        mg = current_mg

        for u, v in chords:
            merged = mg.merge_nodes(u, v)
            if merged.total_loop_count() > 0:
                merged = merged.remove_loops()
            td = compute_best_tree_decomposition(merged, max_width=10)
            if td is None:
                total_cost += 1e15  # Penalty for infeasible treewidth
            else:
                total_cost += estimate_dp_cost(td)

            # Update running multigraph
            edge = (min(u, v), max(u, v))
            ec = dict(mg.edge_counts)
            ec[edge] = ec.get(edge, 0) + 1
            mg = MultiGraph(nodes=mg.nodes, edge_counts=ec, loop_counts=mg.loop_counts)

        return total_cost

    def _process_chords_sequential(
        self,
        poly: TuttePolynomial,
        current_mg: MultiGraph,
        chords: List[Tuple[int, int]],
    ) -> TuttePolynomial:
        """Process chords sequentially (standard path)."""
        import time as _time
        for i, (u, v) in enumerate(chords):
            t0 = _time.perf_counter()
            merged = current_mg.merge_nodes(u, v)
            merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
            dt = _time.perf_counter() - t0
            self._log(f"  Chord {i+1}/{len(chords)}: ({u},{v}) -> {merged.node_count()}n/{merged.edge_count()}e  [{dt:.1f}s]")

            poly = poly + merged_poly

            # Update running multigraph
            edge = (min(u, v), max(u, v))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts
            )

        return poly

    def _process_chords_symmetric(
        self,
        poly: TuttePolynomial,
        current_mg: MultiGraph,
        ordered_chords: List[Tuple[int, int]],
        automorphism: Dict[int, int],
        partition: List[Set[int]],
    ) -> TuttePolynomial:
        """Process chords using symmetric pair ordering with parallel merge computation.

        For each symmetric pair (e₁, e₂) at symmetric state G:
          T(G + e₁ + e₂) = T(G) + T(G/{e₁}) + T((G+e₁)/{e₂})

        T(G/{e₁}) and T((G+e₁)/{e₂}) are independent and can be parallelized.
        After adding both chords, the graph is symmetric again for the next pair.
        """
        total = len(ordered_chords)
        i = 0

        while i < total:
            u1, v1 = ordered_chords[i]

            # Check if next chord is the symmetric partner
            if i + 1 < total:
                u2, v2 = ordered_chords[i + 1]
                if self._is_symmetric_partner(u1, v1, u2, v2, automorphism, partition):
                    # Process as symmetric pair
                    self._log(f"  Symmetric pair {i+1}-{i+2}/{total}: "
                              f"({u1},{v1}) ↔ ({u2},{v2})")

                    # Build both merged graphs (independent of each other)
                    merge1 = current_mg.merge_nodes(u1, v1)

                    # G + e₁ for the second merge
                    edge1 = (min(u1, v1), max(u1, v1))
                    ec_with_e1 = dict(current_mg.edge_counts)
                    ec_with_e1[edge1] = ec_with_e1.get(edge1, 0) + 1
                    mg_plus_e1 = MultiGraph(
                        nodes=current_mg.nodes,
                        edge_counts=ec_with_e1,
                        loop_counts=current_mg.loop_counts,
                    )
                    merge2 = mg_plus_e1.merge_nodes(u2, v2)

                    # Synthesize both merges — in parallel if large enough
                    if self._should_parallelize(merge1, merge2):
                        from .parallel import parallel_synthesize_pair
                        poly1, poly2 = parallel_synthesize_pair(
                            self, merge1, merge2, 10, True
                        )
                    else:
                        poly1 = self._synthesize_multigraph(merge1, skip_minor_search=True)
                        poly2 = self._synthesize_multigraph(merge2, skip_minor_search=True)

                    # T(G + e₁ + e₂) = T(G) + T(G/{e₁}) + T((G+e₁)/{e₂})
                    poly = poly + poly1 + poly2

                    # Update running multigraph with both edges
                    edge2 = (min(u2, v2), max(u2, v2))
                    ec_with_both = dict(ec_with_e1)
                    ec_with_both[edge2] = ec_with_both.get(edge2, 0) + 1
                    current_mg = MultiGraph(
                        nodes=current_mg.nodes,
                        edge_counts=ec_with_both,
                        loop_counts=current_mg.loop_counts,
                    )

                    i += 2
                    continue

            # Single chord (unpaired or not a partner)
            merged = current_mg.merge_nodes(u1, v1)
            merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
            self._log(f"  Chord {i+1}/{total}")

            poly = poly + merged_poly

            edge = (min(u1, v1), max(u1, v1))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts,
            )

            i += 1

        return poly

    def _is_symmetric_partner(
        self,
        u1: int, v1: int,
        u2: int, v2: int,
        automorphism: Dict[int, int],
        partition: List[Set[int]],
    ) -> bool:
        """Check if (u2, v2) is the symmetric partner of (u1, v1) under σ."""
        cell0 = partition[0]
        inv = {v: k for k, v in automorphism.items()}

        # Normalize: cell0 node first
        if u1 not in cell0:
            u1, v1 = v1, u1
        if u2 not in cell0:
            u2, v2 = v2, u2

        # Partner of (u1∈cell0, v1∈cell1) = (σ⁻¹(v1), σ(u1))
        partner_u = inv.get(v1)
        partner_v = automorphism.get(u1)

        if partner_u is None or partner_v is None:
            return False

        return ({u2, v2} == {partner_u, partner_v})

    def _synthesize_connected(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize polynomial for connected graph using creation-expansion-join."""
        target_edges = graph.edge_count()

        # For small graphs, spanning tree expansion is faster than VF2 search.
        if target_edges <= 15:
            return self._synthesize_from_k2(graph, max_depth)

        # Only use tiles that cover a meaningful portion of the graph
        min_tile_edges = max(target_edges // 3, 4)

        # Only use explicitly-named entries as tile candidates
        candidates = [
            c for c in self.table.find_minors_of(graph)
            if not c.name.startswith("synth_") and not c.name.startswith("hybrid_")
               and c.edge_count >= min_tile_edges
               and c.canonical_key != graph.canonical_key()
        ]

        cover = None
        minor = None

        for candidate in candidates:
            self._log(f"Trying minor: {candidate.name} ({candidate.edge_count} edges)")
            trial_cover = find_disjoint_cover(graph, candidate, self.table)

            if not trial_cover.tiles:
                continue  # Not a real subgraph, try next

            if trial_cover.covered_nodes != graph.nodes:
                self._log(f"  Cover incomplete ({len(trial_cover.covered_nodes)}/{len(graph.nodes)} nodes)")
                continue  # Doesn't cover all nodes, try next

            # Found a usable cover
            cover = trial_cover
            minor = candidate
            break

        if cover is None:
            # No minor produces a useful cover, use spanning tree expansion
            return self._synthesize_from_k2(graph, max_depth)

        self._log(f"Cover: {len(cover.tiles)} tiles, {len(cover.uncovered_edges)} uncovered edges")

        # Compute base polynomial from disjoint tiles (product formula)
        poly = TuttePolynomial.one()
        recipe = [f"Tiling with {len(cover.tiles)} copies of {minor.name}"]
        all_minors = {minor.canonical_key}

        # Snapshot accumulator before edge addition
        pre_minors = set(self._mg_minors_accum)

        for tile in cover.tiles:
            poly = poly * tile.minor.polynomial
            recipe.append(f"  Tile {tile.minor.name}: T = {tile.minor.polynomial}")

        # Build the covered subgraph as a MultiGraph for edge addition
        covered_edge_counts = {}
        for tile in cover.tiles:
            for edge in tile.covered_edges:
                covered_edge_counts[edge] = covered_edge_counts.get(edge, 0) + 1

        # Handle uncovered edges using the correct formula:
        # - Bridge (connects different components): T(G+e) = x · T(G)
        # - Chord (within same component): T(G+e) = T(G) + T(G/{u,v})
        if cover.uncovered_edges:
            uncovered_list = sorted(cover.uncovered_edges)
            self._log(f"Adding {len(uncovered_list)} uncovered edges via edge addition")
            recipe.append(f"Edge addition for {len(uncovered_list)} uncovered edges")

            current_mg = MultiGraph(
                nodes=graph.nodes,
                edge_counts=covered_edge_counts,
                loop_counts={}
            )

            for u, v in uncovered_list:
                if current_mg.in_same_component(u, v):
                    # Chord: T(G+e) = T(G) + T(G/{u,v})
                    merged = current_mg.merge_nodes(u, v)
                    merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
                    poly = poly + merged_poly
                else:
                    # Bridge: T(G+e) = x · T(G)
                    poly = TuttePolynomial.x() * poly

                edge = (min(u, v), max(u, v))
                new_edge_counts = dict(current_mg.edge_counts)
                new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
                current_mg = MultiGraph(
                    nodes=current_mg.nodes,
                    edge_counts=new_edge_counts,
                    loop_counts=current_mg.loop_counts
                )

        # Harvest minors from edge addition
        all_minors |= (self._mg_minors_accum - pre_minors)

        # Verify
        verified = verify_spanning_trees(graph, poly)

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=verified,
            method="creation_expansion_join",
            tiles_used=len(cover.tiles),
            fringe_edges=0,
            minors_used=all_minors,
        )

    def _synthesize_from_k2(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Build polynomial from spanning tree + edge addition.

        Algorithm:
        1. Find a spanning tree of the graph
        2. Start with T(spanning tree) = x^(n-1)
        3. For each non-tree edge (chord), use edge addition:
           T(G + e) = T(G) + T(G/{u,v})

        This is the "create-expand" algorithm.
        """
        self._log("Building via spanning tree + edge addition")

        n = graph.node_count()
        m = graph.edge_count()

        if n == 0:
            return SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph"],
                verified=True,
                method="base_case"
            )

        # Snapshot accumulator to diff later
        pre_minors = set(self._mg_minors_accum)

        recipe = ["Spanning tree + edge addition"]

        # Find a spanning tree using BFS
        G_nx = graph.to_networkx()
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

        # Non-tree edges (chords)
        chords = [e for e in graph.edges if e not in tree_edges]

        self._log(f"Spanning tree: {len(tree_edges)} edges, chords: {len(chords)}")
        recipe.append(f"Spanning tree: {len(tree_edges)} edges, T = x^{len(tree_edges)}")
        recipe.append(f"Chords to add: {len(chords)}")

        # Start with spanning tree polynomial: x^(n-1)
        poly = TuttePolynomial.x(len(tree_edges))

        # Build the current graph (starting with spanning tree)
        current_mg = MultiGraph(
            nodes=graph.nodes,
            edge_counts={e: 1 for e in tree_edges},
            loop_counts={}
        )

        # Add each chord using edge addition formula
        for i, (u, v) in enumerate(chords):
            # T(G + e) = T(G) + T(G/{u,v})
            merged = current_mg.merge_nodes(u, v)
            merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)

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

            self._log(f"Added chord {i+1}/{len(chords)}: ({u},{v})")

        recipe.append(f"Final polynomial has {poly.num_terms()} terms")

        # Harvest minors discovered during chord addition
        new_minors = self._mg_minors_accum - pre_minors

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,
            method="spanning_tree_expansion",
            minors_used=new_minors,
        )

    def _synthesize_fast(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Fast synthesis path that skips minor search.

        This method handles basic cases and optimizations but skips the
        expensive minor search in _synthesize_connected. It's used for
        intermediate merged graphs that are unlikely to match known minors.

        The order of checks:
        1. Rainbow table lookup (still fast)
        2. Base cases (empty, single edge)
        3. Disconnected graphs
        4. Cut vertex factorization
        5. Direct spanning tree expansion (skip minor search)

        Args:
            graph: Graph to compute polynomial for
            max_depth: Maximum recursion depth

        Returns:
            SynthesisResult with computed polynomial
        """
        # Base cases (0-1 edges) — checked first to avoid hashing trivial graphs
        if graph.edge_count() <= 1:
            if graph.edge_count() == 0:
                return SynthesisResult(
                    polynomial=TuttePolynomial.one(),
                    recipe=["Empty graph: T = 1"],
                    verified=True,
                    method="base_case"
                )
            return SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )

        # Two-level cache: fast_hash filter before expensive canonical_key
        fh = graph.fast_hash()
        if not hasattr(self, '_fast_simple_hash_set'):
            self._fast_simple_hash_set = set()
        if not hasattr(self, '_table_nm_set'):
            self._table_nm_set = {
                (e.node_count, e.edge_count) for e in self.table.entries.values()
            }

        if fh in self._fast_simple_hash_set:
            # Potential cache/table hit — compute canonical_key
            cache_key = graph.canonical_key()
        elif (graph.node_count(), graph.edge_count()) in self._table_nm_set:
            # Could match a rainbow table entry — compute canonical_key
            cache_key = graph.canonical_key()
        else:
            # No cache entry and no table entry with this n,m — skip canonical_key
            cache_key = None

        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Rainbow table lookup by key (avoids recomputing canonical_key)
        if cache_key is not None:
            entry = self.table.get_entry_by_key(cache_key)
        else:
            entry = None
        if entry is not None:
            result = SynthesisResult(
                polynomial=entry.polynomial,
                recipe=["Rainbow table lookup"],
                verified=True,
                method="lookup",
                minors_used={cache_key},
            )
            self._cache[cache_key] = result
            return result

        # (Base cases already handled above — edge_count <= 1 returns early)

        # Helper to ensure cache_key is computed before caching
        def _ensure_cache_key():
            nonlocal cache_key
            if cache_key is None:
                cache_key = graph.canonical_key()
            return cache_key

        # 3. Disconnected graphs (recurse through _synthesize_fast, not full synthesize)
        components = graph.connected_components()
        if len(components) > 1:
            poly = TuttePolynomial.one()
            all_minors = set()
            for comp in components:
                comp_result = self._synthesize_fast(comp, max_depth)
                poly = poly * comp_result.polynomial
                all_minors |= comp_result.minors_used
            result = SynthesisResult(
                polynomial=poly,
                recipe=[f"Disconnected: {len(components)} components (fast)"],
                verified=True,
                method="disjoint_union",
                minors_used=all_minors,
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4. Cut vertex factorization (recurse through _synthesize_fast, not full synthesize)
        cut = graph.has_cut_vertex()
        if cut is not None:
            sub_components = graph.split_at_cut_vertex(cut)
            poly = TuttePolynomial.one()
            all_minors = set()
            for comp in sub_components:
                comp_result = self._synthesize_fast(comp, max_depth)
                poly = poly * comp_result.polynomial
                all_minors |= comp_result.minors_used
            result = SynthesisResult(
                polynomial=poly,
                recipe=[f"Cut vertex at {cut}: {len(sub_components)} components (fast)"],
                verified=True,
                method="cut_vertex",
                minors_used=all_minors,
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4.5 Try series-parallel O(n) computation
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            result = SynthesisResult(
                polynomial=sp_poly,
                recipe=["Series-parallel decomposition (fast)"],
                verified=True,
                method="series_parallel",
            )
            ck = _ensure_cache_key()
            self._cache[ck] = result
            self._fast_simple_hash_set.add(fh)
            self._promote_to_table(graph, ck, result)
            return result

        # 4.6 Try k-sum decomposition (useful for intermediate merged graphs)
        if graph.edge_count() >= 6:
            result = self._try_ksum_decomposition(graph)
            if result is not None:
                ck = _ensure_cache_key()
                self._cache[ck] = result
                self._fast_simple_hash_set.add(fh)
                self._promote_to_table(graph, ck, result)
                return result

        # 5. Direct spanning tree expansion (skip minor search)
        result = self._synthesize_from_k2_fast(graph, max_depth)
        ck = _ensure_cache_key()
        self._cache[ck] = result
        self._fast_simple_hash_set.add(fh)
        self._promote_to_table(graph, ck, result)
        return result

    def _synthesize_from_k2_fast(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Spanning tree expansion with fast path for merged graphs.

        Same as _synthesize_from_k2 but uses skip_minor_search=True for
        recursive multigraph synthesis.
        """
        self._log("Building via spanning tree + edge addition (fast path)")

        n = graph.node_count()

        if n == 0:
            return SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph"],
                verified=True,
                method="base_case"
            )

        # Snapshot accumulator to diff later
        pre_minors = set(self._mg_minors_accum)

        recipe = ["Spanning tree + edge addition (fast)"]

        # Find spanning tree using BFS
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

        # Chords — sorted by priority: prefer edges whose contraction is more
        # likely to create cut vertices (fewer shared neighbors between endpoints)
        chords = [e for e in graph.edges if e not in tree_edges]

        def chord_priority(e):
            u, v = e
            nu = graph.neighbors(u)
            nv = graph.neighbors(v)
            shared = len(nu & nv)
            min_deg = min(len(nu), len(nv))
            return (shared, min_deg)

        chords.sort(key=chord_priority)

        self._log(f"Spanning tree: {len(tree_edges)} edges, chords: {len(chords)}")
        recipe.append(f"Spanning tree: {len(tree_edges)} edges")
        recipe.append(f"Chords: {len(chords)}")

        # Start with spanning tree polynomial
        poly = TuttePolynomial.x(len(tree_edges))

        # Build current multigraph
        current_mg = MultiGraph(
            nodes=graph.nodes,
            edge_counts={e: 1 for e in tree_edges},
            loop_counts={}
        )

        # Add chords with skip_minor_search=True
        for i, (u, v) in enumerate(chords):
            merged = current_mg.merge_nodes(u, v)
            # Use skip_minor_search=True for recursive synthesis
            merged_poly = self._synthesize_multigraph(merged, max_depth, skip_minor_search=True)

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

        return SynthesisResult(
            polynomial=poly,
            recipe=recipe,
            verified=True,
            method="spanning_tree_expansion_fast",
            minors_used=new_minors,
        )

    # =========================================================================
    # EDGE ADDITION UTILITIES
    # =========================================================================

    def _add_edges_to_graph(
        self,
        base_graph: Graph,
        base_poly: TuttePolynomial,
        edges_to_add: List[Tuple[int, int]]
    ) -> TuttePolynomial:
        """Add edges using the edge addition formula.

        For each edge e=(u,v) added to graph G:
        T(G + e) = T(G) + T(G/{u,v})

        where G/{u,v} is G with nodes u,v merged.

        Args:
            base_graph: Starting graph
            base_poly: Polynomial for base_graph
            edges_to_add: List of edges to add

        Returns:
            Polynomial for graph with all edges added
        """
        if not edges_to_add:
            return base_poly

        self._log(f"Adding {len(edges_to_add)} edges via edge addition formula")

        current_poly = base_poly
        current_mg = MultiGraph.from_graph(base_graph)

        for u, v in edges_to_add:
            # Compute T(G/{u,v}) - the polynomial for merged graph
            merged = current_mg.merge_nodes(u, v)
            merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)

            # T(G + e) = T(G) + T(G/{u,v})
            current_poly = current_poly + merged_poly

            # Update current multigraph by adding the edge
            edge = (min(u, v), max(u, v))
            new_edge_counts = dict(current_mg.edge_counts)
            new_edge_counts[edge] = new_edge_counts.get(edge, 0) + 1
            current_mg = MultiGraph(
                nodes=current_mg.nodes,
                edge_counts=new_edge_counts,
                loop_counts=current_mg.loop_counts
            )

        return current_poly

    def _apply_k_joins(
        self,
        base_poly: TuttePolynomial,
        cover: Cover,
        graph: Graph,
        inter_edges: Set[Tuple[int, int]]
    ) -> Tuple[TuttePolynomial, List[str]]:
        """Apply k-join formulas for edges connecting different tiles.

        For each edge connecting nodes from different tiles,
        determine the k-join type and adjust polynomial.
        """
        recipe = [f"Inter-tile edges: {len(inter_edges)}"]

        # Analyze connections
        connections = analyze_tile_connections(cover, graph)

        # For now, use a simplified approach:
        # Each inter-tile edge contributes to the polynomial
        # The exact formula depends on the join type

        poly = base_poly

        for (i, j), conn_info in connections.items():
            k = conn_info['k_join_type']
            n_edges = conn_info['connecting_edges']

            if k == "disjoint":
                # Tiles are completely separate (covered by multiplication)
                continue
            elif k == "bridge":
                # Single edge connecting tiles - multiply by x
                poly = poly * TuttePolynomial.x()
                recipe.append(f"  Bridge {i}-{j}: × x")
            elif k == "1_join":
                # Cut vertex - polynomial already multiplied
                recipe.append(f"  1-join {i}-{j}: (already factored)")
            else:
                # More complex - for now just note it
                recipe.append(f"  {k} {i}-{j}: {n_edges} edges (approximated)")

        return poly, recipe

    def _adjust_for_fringe(
        self,
        poly: TuttePolynomial,
        fringe: Fringe,
        max_depth: int
    ) -> Tuple[TuttePolynomial, List[str]]:
        """Adjust polynomial for over-coverage (fringe edges).

        Fringe edges are edges in our tiling that don't exist in the input.
        We need to "subtract" their contribution.

        Uses deletion formula: T(G-e) = T(G) - T(G/e)
        Rearranged when we know T(G): if we have T(G) and want T(G-e),
        we need to compute T(G/e) and subtract.

        For simple cases (fringe is a forest), use direct formula.
        For complex cases, this is harder and may need approximation.
        """
        recipe = [f"Fringe: {fringe.edge_count()} edges"]

        if fringe.edge_count() == 0:
            return poly, recipe

        # For now, use a simplified approach for small fringes
        # Each fringe edge that would be a bridge subtracts x
        # Each fringe edge in a cycle is more complex

        # This is an approximation - full handling requires
        # computing the contracted graph polynomial

        fringe_graph = fringe.as_graph()
        if fringe_graph.is_connected() and fringe.edge_count() <= fringe_graph.node_count() - 1:
            # Fringe is a tree/forest - each edge is a bridge
            # Removing these edges divides by x^n
            n_fringe = fringe.edge_count()
            # This is an approximation
            recipe.append(f"  Fringe forest: {n_fringe} bridge(s)")
            # Don't adjust polynomial here - the tiling was incorrect
            # Fall back to deletion-contraction for accuracy
        else:
            recipe.append(f"  Complex fringe: needs deletion-contraction")

        return poly, recipe

    def _handle_uncovered(
        self,
        poly: TuttePolynomial,
        uncovered: Set[Tuple[int, int]],
        graph: Graph,
        cover: Cover,
        max_depth: int
    ) -> Tuple[TuttePolynomial, List[str]]:
        """Handle edges not covered by any tile.

        Uses edge addition formula: T(G + e) = T(G) + T(G/{u,v})

        If uncovered edges connect covered nodes, use edge addition.
        If uncovered edges introduce new nodes, those nodes are isolated
        in the base and we handle them separately.
        """
        recipe = [f"Uncovered edges: {len(uncovered)}"]

        if not uncovered:
            return poly, recipe

        # Check if all endpoints of uncovered edges are in covered nodes
        uncovered_endpoints = set()
        for u, v in uncovered:
            uncovered_endpoints.add(u)
            uncovered_endpoints.add(v)

        edges_with_covered_endpoints = []
        edges_with_new_nodes = []

        for u, v in uncovered:
            if u in cover.covered_nodes and v in cover.covered_nodes:
                edges_with_covered_endpoints.append((u, v))
            else:
                edges_with_new_nodes.append((u, v))

        # Handle edges that connect covered nodes using edge addition
        if edges_with_covered_endpoints:
            # Build the covered subgraph
            covered_graph = graph.subgraph(cover.covered_nodes)
            covered_subgraph = Graph(
                nodes=frozenset(cover.covered_nodes),
                edges=frozenset(cover.covered_edges)
            )

            # Use edge addition formula for these edges
            poly = self._add_edges_to_graph(
                covered_subgraph,
                poly,
                edges_with_covered_endpoints
            )
            recipe.append(f"  Edge addition for {len(edges_with_covered_endpoints)} edges")

        # Handle edges that introduce new nodes (rare case)
        if edges_with_new_nodes:
            # Build subgraph from these edges
            edge_nodes = set()
            for u, v in edges_with_new_nodes:
                edge_nodes.add(u)
                edge_nodes.add(v)

            # Nodes not in cover are isolated in our base polynomial
            # Need to handle this specially
            uncovered_graph = graph.edge_induced_subgraph(set(edges_with_new_nodes))
            if uncovered_graph.edge_count() > 0:
                if max_depth > 0:
                    sub_result = self.synthesize(uncovered_graph, max_depth - 1)
                    poly = poly * sub_result.polynomial
                    recipe.append(f"  Uncovered subgraph with new nodes: {sub_result.polynomial}")
                else:
                    recipe.append("  Max depth reached for uncovered edges with new nodes")

        return poly, recipe


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def synthesize(graph: Graph, verbose: bool = False, method: str = "auto") -> SynthesisResult:
    """Convenience function to synthesize polynomial for a graph.

    Args:
        graph: Graph to compute polynomial for
        verbose: Print progress information
        method: Synthesis method:
            - "auto": Tiling for small graphs (<=12 edges), hybrid for larger
            - "tiling": Tiling-based (spanning tree + edge addition)
            - "algebraic": Pure algebraic decomposition
            - "hybrid": Combined tiling + pattern recognition

    Returns:
        SynthesisResult with computed polynomial
    """
    if method == "algebraic":
        from .algebraic import AlgebraicSynthesisEngine
        engine = AlgebraicSynthesisEngine(verbose=verbose)
        alg_result = engine.synthesize(graph)
        return SynthesisResult(
            polynomial=alg_result.polynomial,
            recipe=alg_result.recipe,
            verified=alg_result.verified,
            method=alg_result.method
        )

    if method == "hybrid":
        from .hybrid import HybridSynthesisEngine
        engine = HybridSynthesisEngine(verbose=verbose)
        hybrid_result = engine.synthesize(graph)
        return SynthesisResult(
            polynomial=hybrid_result.polynomial,
            recipe=hybrid_result.recipe,
            verified=hybrid_result.verified,
            method=hybrid_result.method
        )

    if method == "tiling":
        engine = SynthesisEngine(verbose=verbose)
        return engine.synthesize(graph)

    # Auto mode: pick best engine based on graph size
    # Hybrid excels on larger graphs (>12 edges) due to better
    # pattern recognition for intermediate multigraphs.
    # Tiling has lower overhead for small graphs.
    if graph.edge_count() > 12:
        from .hybrid import HybridSynthesisEngine
        engine = HybridSynthesisEngine(verbose=verbose)
        hybrid_result = engine.synthesize(graph)
        return SynthesisResult(
            polynomial=hybrid_result.polynomial,
            recipe=hybrid_result.recipe,
            verified=hybrid_result.verified,
            method=hybrid_result.method
        )
    else:
        engine = SynthesisEngine(verbose=verbose)
        return engine.synthesize(graph)


def synthesize_algebraic(graph: Graph, verbose: bool = False) -> 'AlgebraicSynthesisResult':
    """Synthesize polynomial using algebraic decomposition.

    This method computes the polynomial using GCD-based factorization
    rather than graph tiling. It's particularly useful when:
    - The polynomial has clear algebraic structure
    - You want to understand the decomposition of a polynomial
    - Graph tiling is inefficient for the particular graph

    Args:
        graph: Graph to compute polynomial for
        verbose: Print progress information

    Returns:
        AlgebraicSynthesisResult with decomposition details
    """
    from .algebraic import AlgebraicSynthesisEngine
    engine = AlgebraicSynthesisEngine(verbose=verbose)
    return engine.synthesize(graph)


def decompose_polynomial(polynomial: TuttePolynomial, verbose: bool = False):
    """Decompose a known polynomial into algebraic factors.

    Given a Tutte polynomial, find its decomposition in terms of
    known graph polynomials from the rainbow table.

    Args:
        polynomial: Polynomial to decompose
        verbose: Print progress information

    Returns:
        AlgebraicSynthesisResult with decomposition
    """
    from .algebraic import AlgebraicSynthesisEngine
    engine = AlgebraicSynthesisEngine(verbose=verbose)
    return engine.synthesize_from_polynomial(polynomial)


def compute_tutte_polynomial(graph: Graph, method: str = "auto") -> TuttePolynomial:
    """Compute Tutte polynomial for a graph.

    This is the main entry point for polynomial computation.

    Args:
        graph: Graph to compute polynomial for
        method: Synthesis method - "auto", "tiling", or "algebraic"

    Returns:
        TuttePolynomial for the graph
    """
    result = synthesize(graph, method=method)
    return result.polynomial
