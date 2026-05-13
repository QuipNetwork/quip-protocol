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

from ..family_recognition import recognize_family
from ..graph import Graph, MultiGraph, compute_signature
from ..graphs.covering import (Cover, Fringe, InterCellInfo,
                               KMatchingJunction, Tile,
                               analyze_tile_connections,
                               apply_kmatching_formula,
                               compute_fringe,
                               compute_inter_tile_edges,
                               detect_kmatching_topology,
                               extract_cell_topology, find_disjoint_cover,
                               try_heterogeneous_partition,
                               try_hierarchical_partition)
from ..graphs.series_parallel import compute_sp_tutte_if_applicable
from ..cotree_dp import compute_tutte_cotree_dp, compute_tutte_almost_cograph
from ..roots import (
    compute_cell_quotient_cycle_dp,
    compute_cell_quotient_grid_dp_streamed,
    compute_cell_quotient_tree_dp,
)
from ..roots.cell_quotient_hybrid import compute_cell_quotient_hybrid
from ..logs import EventType, LogLevel, get_log
from ..lookup.core import MinorEntry, RainbowTable, load_default_table
from ..polynomial import TuttePolynomial
from ..validation import verify_spanning_trees
from .base import BaseMultigraphSynthesizer, SynthesisResult, UnionFind


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
        promote_cache_on_finish: bool = False,
        k_max: int = 12,
    ):
        """Initialize synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
            auto_promote: If True, auto-promote synthesized simple graphs to the rainbow table
            promote_cache_on_finish: If True, at the end of each top-level
                `synthesize()` call, promote every simple-graph entry in
                `self._cache` to the rainbow table. Lets cache_hits from
                the current run become `lookup_hit`s in the next run.
            k_max: Maximum k for k-sum vertex-separator search (default 12).
                Bounded by the engine's top-20-degree candidate filter; values
                above 20 are clamped. The chord rule is `1 + C(k,2)` syntheses
                per attempt, so larger k is uniformly more expensive.
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose
        self.auto_promote = auto_promote
        self.promote_cache_on_finish = promote_cache_on_finish
        self.k_max = max(2, min(k_max, 20))
        # (April 2026): default True after cold-cache A/B
        # showed no regressions (corpus median -2.8%, Z(1,1) regression
        # vanished). Smart ordering sorts chord edges by descending
        # |common_neighbors(u, v)| in the original graph, so high-impact
        # contractions happen early and the engine's parallel-edge / loop fast
        # paths fire sooner. Set to False to revert if a regression surfaces
        # for a chord-rule-heavy target (Pm3+, Cm3+, Z(2,t)+).
        self.chord_smart_order: bool = True
        self._cache: Dict[str, SynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}  # For multigraph polynomials
        self._fast_hash_set: Set[str] = set()  # Fast hashes of all cached multigraphs
        self._fast_hash_set_complete: bool = True  # True when _fast_hash_set covers all cache entries
        self._fast_simple_hash_set: Set[str] = set()  # Fast hashes of all cached simple graphs
        self._table_nm_set: Set[Tuple[int, int]] = {
            (e.node_count, e.edge_count) for e in self.table.entries.values()
        }
        self._inter_cell_cache: Dict[str, TuttePolynomial] = {}  # For inter-cell graph polynomials
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis
        # Accumulates (canonical_key -> Graph|MultiGraph) for every sub-problem
        # the engine actually synthesized (not a cache/lookup hit). Attached to
        # the top-level SynthesisResult so the visualizer can split "contributing
        # graphs from the lookup table" vs "graphs synthesized along the way".
        self._synth_accum_graphs: Dict[str, object] = {}
        self._synth_depth: int = 0
        # When persisting caches on finish, also auto-load the
        # multigraph lookup table at init so cache hits persist
        # across engine instances (visualizer reruns, successive
        # target syntheses, etc.).
        if self.promote_cache_on_finish:
            try:
                self.load_multigraph_cache()
            except Exception:
                pass
        # When True, the top-level synthesize() call skips the rainbow-table
        # lookup for the input graph (but sub-problems may still be looked up).
        # Useful for visualizer runs where we want to see what the engine
        # would do without a direct hit on the target.
        self.skip_target_lookup: bool = False

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

        Thin wrapper around `_synthesize_inner` that tracks recursion depth
        so the top-level call can stamp the final SynthesisResult with
        `synthesized_minors` and `synthesized_graphs` — the per-run snapshot
        of every sub-graph the engine actually synthesized (as opposed to
        retrieved from the rainbow table).
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
            # Promote cache entries to the rainbow table so that
            # cache_hits become lookup_hits on subsequent runs.
            if self.promote_cache_on_finish:
                self._flush_cache_to_table()
        return result

    def _flush_cache_to_table(self) -> None:
        """Promote cache entries to the persistent lookup tables.

        - Simple-graph `self._cache` entries with matching snapshots in
          `self._synth_accum_graphs` → added to `self.table` (rainbow
          table) in memory, then saved to disk.
        - Multigraph `self._multigraph_cache` entries → saved to the
          default multigraph lookup table on disk.

        Skips rainbow-table entries that already exist. Disk I/O is
        guarded inside try/except so a save failure doesn't abort the
        synthesis result.

        Called at the end of each top-level `synthesize()` when
        `self.promote_cache_on_finish` is enabled.
        """
        # In-memory promotion: simple graph cache → rainbow table.
        new_simple_entries = 0
        for cache_key, result in list(self._cache.items()):
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
                    polynomial=result.polynomial,
                    node_count=g.node_count(),
                    edge_count=g.edge_count(),
                    canonical_key=cache_key,
                    spanning_trees=result.polynomial.num_spanning_trees(),
                    num_terms=result.polynomial.num_terms(),
                    graph=g,
                    signature=compute_signature(g),
                )
                self.table.add_entry(entry)
                new_simple_entries += 1
            except Exception:
                continue

        # Persist both caches to disk so the NEXT run (new engine
        # instance) sees these entries as lookup_hit rather than
        # recomputing them. Merge with the on-disk tables to avoid
        # overwriting entries written by previous runs.
        try:
            if new_simple_entries > 0:
                self.save_rainbow_table()
        except Exception:
            pass
        try:
            if self._multigraph_cache:
                from ..lookup.core import (
                    load_default_multigraph_table,
                    save_default_multigraph_table,
                )
                existing = load_default_multigraph_table()
                # Merge in-memory cache into the on-disk table. Only
                # add entries not already present; do not overwrite.
                for k, p in self._multigraph_cache.items():
                    existing.setdefault(k, p)
                save_default_multigraph_table(existing)
        except Exception:
            pass

    def _synthesize_inner(
        self,
        graph: Graph,
        max_depth: int = 10
    ) -> SynthesisResult:
        """Core synthesis logic — see `synthesize()` for the public entrypoint."""
        _log = get_log()
        n, m = graph.node_count(), graph.edge_count()
        _log.record(EventType.SYNTHESIS_START, "engine",
                     f"{n}n {m}e", LogLevel.INFO, graph=graph)

        # 1. Family recognition fast path — O(n+m)
        # Runs before canonical_key() to avoid the O(n² log n) cost for
        # known families. canonical_key is only needed for cache/table ops.
        family_poly = recognize_family(graph)
        if family_poly is not None:
            _log.record(EventType.FAMILY_RECOGNITION, "engine",
                        f"Family recognized: {n}n {m}e", LogLevel.INFO,
                        graph=graph)
            self._log(f"Family recognition: O(n+m) fast path")
            result = SynthesisResult(
                polynomial=family_poly,
                recipe=["Family recognition"],
                verified=True,
                method="family_recognition",
            )
            # Family recognition is synthesis-from-formula; count the input.
            self._record_synth(graph)
            return result

        # 2. Compute canonical key (expensive — O(n² log n))
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            _log.record(EventType.CACHE_HIT, "engine",
                        f"Cache hit: {cache_key[:12]}", LogLevel.DEBUG,
                        graph=graph)
            self._log(f"Cache hit: {cache_key[:16]}...")
            # Record the cache-hit graph so the visualizer surfaces it
            # under Contributing Graphs alongside from-scratch syntheses.
            self._record_synth(graph, cache_key)
            return self._cache[cache_key]

        self._log(f"Synthesizing graph with {n} nodes, {m} edges")

        # 3. Check rainbow table (optionally skipped for the top-level call
        # when skip_target_lookup is enabled — lets the visualizer show the
        # engine's decomposition path for a graph that happens to be in the table).
        is_top_call = self._synth_depth == 1
        if not (is_top_call and self.skip_target_lookup):
            cached = self.table.lookup(graph)
            if cached is not None:
                _log.record(EventType.LOOKUP_HIT, "engine",
                            f"Rainbow table hit: {n}n {m}e", graph=graph)
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

        # Past the lookup gate: every remaining path is real synthesis work.
        # Record the input graph so the visualizer can show it in the
        # "synthesized" panel.
        self._record_synth(graph, cache_key)

        # 4. Handle base cases
        if graph.edge_count() == 0:
            _log.record(EventType.BASE_CASE, "engine", "Empty graph: T = 1",
                        graph=graph)
            result = SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph: T = 1"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            _log.record(EventType.BASE_CASE, "engine", "Single edge: T = x",
                        graph=graph)
            result = SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        # 5. Check if graph is disconnected
        components = graph.connected_components()
        if len(components) > 1:
            _log.record(EventType.FACTORIZE, "engine",
                        f"Disconnected: {len(components)} components",
                        graph=graph)
            result = self._synthesize_disconnected(components, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 6. Check for cut vertices (fast factorization before expensive operations)
        cut = graph.has_cut_vertex()
        if cut is not None:
            _log.record(EventType.FACTORIZE, "engine",
                        f"Cut vertex at {cut}", graph=graph)
            result = self._synthesize_via_cut_vertex(graph, cut, max_depth)
            self._cache[cache_key] = result
            self._promote_to_table(graph, cache_key, result)
            return result

        # 7. Try series-parallel O(n) computation
        sp_poly = compute_sp_tutte_if_applicable(graph)
        if sp_poly is not None:
            _log.record(EventType.SERIES_PARALLEL, "engine",
                        f"SP decomposition: {n}n {m}e", graph=graph)
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

        # 7.45 Cell-quotient grid DP (Phase B Round 6 streamed). For
        # cell-decomposable graphs whose cell-quotient is a 2D grid of
        # K_{a,b}-style cells connected by M_k matchings with disjoint
        # per-direction anchors (Cm2-style — Cm3 has shared anchors and
        # is rejected by precondition). Beats the §7.5 formula
        # short-circuit on Cm2 (~36 s vs ~55 s, 1.5×). Returns None
        # on any precondition mismatch so the engine falls through.
        # Same edge_count >= 60 gate as the other hierarchical paths.
        if graph.edge_count() >= 60:
            try:
                grid_streamed_poly = compute_cell_quotient_grid_dp_streamed(
                    graph, self.table,
                )
                if grid_streamed_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "engine",
                                f"Grid DP (streamed): {n}n {m}e", graph=graph)
                    self._log(f"Cell-quotient grid DP (streamed): {n}n, {m}e")
                    result = SynthesisResult(
                        polynomial=grid_streamed_poly,
                        recipe=["Cell-quotient grid DP (streamed)"],
                        verified=True,
                        method="cell_quotient_grid_dp_streamed",
                    )
                    self._cache[cache_key] = result
                    self._promote_to_table(graph, cache_key, result)
                    return result
            except Exception:
                pass  # any failure — fall through

        # 7.5 Hierarchical-formula short-circuit (Phase 12 unified +
        # Phase 13 k-matching). When the graph has a detectable cell
        # decomposition AND its inter-cell structure satisfies the
        # preconditions of the unified or k-matching formula, these
        # closed-form paths can beat `treewidth_dp` (Cm2: ~4× speedup
        # vs tw_dp). The formula-only shortcut avoids the internal
        # tw_dp fall-through inside `_synthesize_hierarchical`, so we
        # only commit to the hierarchical path when the formula
        # actually applies.
        #
        # Gate: edge_count ≥ 60. This filters out small structured
        # graphs (Petersen, small cycles, atlas graphs) where the
        # partition/detection overhead outweighs the tw_dp cost. Cm2
        # (80e) and Cm3 (192e) clear the gate; Z(1,1) (22e), Cm1
        # (16e), Petersen (15e) skip the shortcut and go straight to
        # tw_dp.
        if graph.edge_count() >= 60:
            formula_result = self._try_formula_shortcircuit(graph, max_depth)
            if formula_result is not None:
                _method_event = {
                    "unified_formula": EventType.UNIFIED_FORMULA,
                    "kmatching_formula": EventType.KMATCHING_FORMULA,
                }.get(formula_result.method, EventType.HIERARCHICAL)
                _log.record(_method_event, "engine",
                            f"Formula shortcut via {formula_result.method}: "
                            f"{formula_result.tiles_used} tiles",
                            graph=graph)
                self._cache[cache_key] = formula_result
                self._promote_to_table(graph, cache_key, formula_result)
                return formula_result

        # 7.5. Cotree DP — subexponential exp(O(n^{2/3})) for cographs
        # (P_4-free graphs). Wins where treewidth_dp can't fit (K_12+) and
        # the graph is a cograph. Fast no-op (early reject) on non-cographs.
        try:
            cotree_poly = compute_tutte_cotree_dp(graph)
            if cotree_poly is not None:
                _log.record(EventType.COTREE_DP, "engine",
                            f"Cotree DP: {n}n {m}e", graph=graph)
                self._log(f"Cotree DP: {n}n, {m}e")
                result = SynthesisResult(
                    polynomial=cotree_poly,
                    recipe=["Cotree-based DP (subexponential cograph)"],
                    verified=True,
                    method="cotree_dp",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result
        except (ValueError, TypeError):
            pass  # not a cograph, or input rejected — fall through

        # 7.6. Almost-cograph DP — for graphs that become cographs after
        # removing a small set of anomaly edges (e.g., D-Wave cells joined by
        # sparse inter-cell edges). Greedy P_4 elimination + bridge-aware
        # iterated chord rule on anomalies; the cograph skeleton routes
        # through cotree_dp. Returns None if anomaly count > max_anomalies.
        # Gate: cap at 16 anomalies (covers Cm2's 16 inter-cell edges; Cm3's
        # 48 falls through to existing chord-rule paths).
        try:
            almost_poly = compute_tutte_almost_cograph(
                graph, self, max_anomalies=16,
            )
            if almost_poly is not None:
                _log.record(EventType.COTREE_DP, "engine",
                            f"Almost-cograph DP: {n}n {m}e", graph=graph)
                self._log(f"Almost-cograph DP: {n}n, {m}e")
                result = SynthesisResult(
                    polynomial=almost_poly,
                    recipe=["Almost-cograph DP (greedy P_4 elim + chord rule)"],
                    verified=True,
                    method="almost_cograph",
                )
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result
        except Exception:
            pass  # any failure — fall through

        # 7.7. Cell-quotient cycle DP — for cell-decomposable graphs whose
        # cell-quotient is a SIMPLE CYCLE (e.g., D-Wave Cm2's 4-cycle of
        # K_{4,4} cells). Combines T_rooted of cells with vertex-sum
        # convolution + identification close. Generic over junction
        # connectivity (handles M_k matchings, K_{a,b} bipartite, etc. via
        # auto-detected component count c_J).
        # Gate: edge_count >= 60 (matches formula shortcut). Only fires
        # for graphs the formula shortcut and almost-cograph paths haven't
        # already handled — so a fallback for cycle-topology cases like
        # Cm2 when those paths return None.
        if graph.edge_count() >= 60:
            try:
                cq_poly = compute_cell_quotient_cycle_dp(graph, self.table)
                if cq_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "engine",
                                f"Cell-quotient cycle DP: {n}n {m}e",
                                graph=graph)
                    self._log(f"Cell-quotient cycle DP: {n}n, {m}e")
                    result = SynthesisResult(
                        polynomial=cq_poly,
                        recipe=["Cell-quotient cycle DP"],
                        verified=True,
                        method="cell_quotient_dp",
                    )
                    self._cache[cache_key] = result
                    self._promote_to_table(graph, cache_key, result)
                    return result
            except Exception:
                pass  # any failure — fall through

        # 7.8. Cell-quotient TREE DP — for cell-decomposable graphs whose
        # cell-quotient is a TREE (n cells, n-1 junctions, no cycles).
        # Generalizes the path/cycle DPs to arbitrary tree topologies.
        # Combined-aut path inside compute_tree_dp_recursive handles
        # keep_shared / fully-consumed cases (see
        # tutte/research/data/combined_aut_findings.md).
        if graph.edge_count() >= 60:
            try:
                tree_poly = compute_cell_quotient_tree_dp(graph, self.table)
                if tree_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "engine",
                                f"Cell-quotient tree DP: {n}n {m}e",
                                graph=graph)
                    self._log(f"Cell-quotient tree DP: {n}n, {m}e")
                    result = SynthesisResult(
                        polynomial=tree_poly,
                        recipe=["Cell-quotient tree DP"],
                        verified=True,
                        method="cell_quotient_tree_dp",
                    )
                    self._cache[cache_key] = result
                    self._promote_to_table(graph, cache_key, result)
                    return result
            except Exception:
                pass  # any failure — fall through

        # 7.85. Cell-quotient HYBRID DP — chord-rule cycle-close + per-leaf
        # synth for cyclic cell-quotients (e.g., D-Wave Cm₃'s 3×3 grid).
        # Recursively peels closing junctions; each leaf is synthesized
        # via the engine's standard pipeline. Self-loop-aware contraction
        # ensures correct y-factor accounting for parallel-edge leaves.
        if graph.edge_count() >= 60:
            try:
                hybrid_poly = compute_cell_quotient_hybrid(
                    graph, self.table,
                )
                if hybrid_poly is not None:
                    _log.record(EventType.CELL_QUOTIENT_DP, "engine",
                                f"Cell-quotient hybrid DP: {n}n {m}e",
                                graph=graph)
                    self._log(f"Cell-quotient hybrid DP: {n}n, {m}e")
                    result = SynthesisResult(
                        polynomial=hybrid_poly,
                        recipe=["Cell-quotient hybrid (cycle-close + per-leaf synth)"],
                        verified=True,
                        method="cell_quotient_hybrid_dp",
                    )
                    self._cache[cache_key] = result
                    self._promote_to_table(graph, cache_key, result)
                    return result
            except Exception:
                pass  # any failure — fall through

        # 8. Treewidth DP — fast for graphs with treewidth ≤ 11. When this
        # succeeds it's usually the best path for graphs that fit. For graphs
        # whose treewidth exceeds the cap, returns None and we fall through to
        # the chord-rule paths below.
        if graph.edge_count() >= 10:
            from ..graphs.treewidth import \
                compute_treewidth_tutte_if_applicable
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = compute_treewidth_tutte_if_applicable(full_mg, max_width=11)
            if tw_poly is not None:
                _log.record(EventType.TREEWIDTH_DP, "engine",
                            f"Treewidth DP: {n}n {m}e", graph=graph)
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

        # 9. k-sum decomposition (k=2..7, vertex separators) via the chord
        # rule (clique_chord_k_sum). Triggers when the graph has a vertex
        # separator that disconnects it cleanly.
        if graph.edge_count() >= 6:
            result = self._try_ksum_decomposition(graph)
            if result is not None:
                _log.record(EventType.KSUM, "engine",
                            f"k-sum: {result.method}", graph=graph)
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 10. Hierarchical tiling via the chord rule (boundary_quotient_tutte).
        # Triggers when the graph has a repeating cell decomposition. Cost is
        # 1 + chord_count syntheses. For graphs where treewidth_dp fits, that
        # path is preferred (above); hierarchical handles the cases where
        # treewidth exceeds 11 and a cell decomposition exists.
        if graph.edge_count() >= 20:
            result = self._try_hierarchical(graph, max_depth)
            if result is not None:
                # Record the SPECIFIC formula/path that succeeded so
                # the visualizer surfaces the actual dispatch (Phase
                # 12 unified formula, Phase 13 k-matching formula, or
                # the generic hierarchical/treewidth fallthrough).
                _method_event = {
                    "unified_formula": EventType.UNIFIED_FORMULA,
                    "kmatching_formula": EventType.KMATCHING_FORMULA,
                    "treewidth_dp": EventType.TREEWIDTH_DP,
                }.get(result.method, EventType.HIERARCHICAL)
                _log.record(_method_event, "engine",
                            f"Hierarchical via {result.method}: "
                            f"{result.tiles_used} tiles",
                            graph=graph)
                self._cache[cache_key] = result
                self._promote_to_table(graph, cache_key, result)
                return result

        # 12. Try creation-expansion-join
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
        # Approximate min-edge gates: roughly C(k, 2) + a small constant for
        # internal cycles. Below this, no useful separator can exist.
        min_edges = {
            2: 3, 3: 6, 4: 12, 5: 15, 6: 20, 7: 28,
            8: 36, 9: 45, 10: 55, 11: 66, 12: 78,
            13: 91, 14: 105, 15: 120, 16: 136, 17: 153,
            18: 171, 19: 190, 20: 210,
        }
        # Bound the high-k search by the actual node connectivity. If the
        # graph's minimum vertex cut is `kappa`, no k < kappa can produce a
        # separator and there's no point searching for k > kappa + 4 (separators
        # of size kappa + 5+ have very different structural properties and
        # almost never beat smaller separators in practice).
        try:
            import networkx as _nx
            kappa = _nx.node_connectivity(graph.to_networkx())
        except Exception:
            kappa = 0
        # Search range: [max(2, kappa), min(k_max, kappa + 5)] when kappa is
        # known. When kappa is unknown (failure), fall back to the configured
        # range.
        k_lo = max(2, kappa) if kappa else 2
        k_hi = min(self.k_max, kappa + 5) if kappa else self.k_max
        for k in range(k_lo, k_hi + 1):
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

                # Accept any separator, including full-clique (missing == 0).
                # `_apply_ksum` handles missing == 0 by peeling the EXISTING
                # K_k clique edges via the chord rule (the symmetric case of
                # adding back missing edges) — same C(k, 2) cost.
                # NOTE: prior to May 2026, missing == 0 was skipped under the
                # comment "cut vertex path already handles k=1", but the cut-
                # vertex path only handles k=1; missing == 0 separators for
                # k ≥ 2 were silently dropped, blocking the chord rule on
                # graphs with full clique separators (e.g., router-style
                # synthetics). See research/audit_chord_rule_findings.md.
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
        from itertools import combinations

        import networkx as nx

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
        for k in range(max(kappa, 5), min(kappa + 3, self.k_max + 1)):
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

            if num_missing > 0:
                # Standard "true k-sum" path: graph has SOME clique edges
                # missing. Build PC by adding missing edges back, then peel
                # them off via chord rule.
                from ..graphs.k_sum import clique_chord_k_sum
                poly = clique_chord_k_sum(
                    graph, separator, k, self,
                    missing_edges=missing_edges,
                )
                method_label = f"{k}sum_chord_rule"
            else:
                # Full clique separator: graph has ALL K_k clique edges. Symmetric
                # case — use the chord rule to PEEL the existing clique edges off
                # `graph`, getting a chord-free leaf with no shared edges between
                # cells, then combine via _combine_chord_iteration. Cost: same
                # 1 + C(k, 2) syntheses as the chord rule for missing > 0.
                #
                # T(graph) = (∏ factors) · T(graph − all clique) + Σ prefix·adds
                #
                # The chord-free leaf (graph − all clique edges) IS the true
                # k-sum target where the cells share only the k separator
                # vertices (no shared edges). Synthesizing it terminates via the
                # standard pipeline (it has no full-clique separator anymore;
                # the separator's clique edges are gone).
                from ..graphs.k_sum import (
                    _combine_chord_iteration,
                    _iterative_chord_rule,
                )
                smart_order = getattr(self, "chord_smart_order", False)
                g_chord_free, factors, adds = _iterative_chord_rule(
                    graph, all_clique_edges, self, smart_order=smart_order,
                )
                t_chord_free = self.synthesize(g_chord_free).polynomial
                poly = _combine_chord_iteration(t_chord_free, factors, adds)
                method_label = f"{k}sum_full_clique_chord_peel"

            if poly is None:
                return None

            # Verify
            if verify_spanning_trees(graph, poly):
                self._log(f"{k}-vertex separator decomposition verified! ({method_label})")
                return SynthesisResult(
                    polynomial=poly,
                    recipe=[f"{k}-vertex separator at {separator} "
                            f"({num_missing} missing)", f"T = {poly}"],
                    verified=True,
                    method=method_label,
                )
            else:
                self._log(f"{k}-vertex separator decomposition failed Kirchhoff ({method_label})")
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

    def _emit_partition_provenance(
        self, graph: Graph, cells, partition, inter_info,
    ) -> None:
        """Record a snapshot+provenance event for each cell of a
        hierarchical decomposition + the inter-cell edge set, so the
        visualizer can highlight where each sub-graph lives in the
        input graph.
        """
        log = get_log()
        if not log.capture_graphs:
            return
        # Per-cell records.
        for cell_idx, (cell_entry, cell_nodes) in enumerate(zip(cells, partition)):
            try:
                cell_subgraph = graph.subgraph(cell_nodes)
            except Exception:
                continue
            cell_node_list = sorted(cell_nodes)
            cell_edge_list = [
                [u, v] for (u, v) in cell_subgraph.edges
            ]
            log.record(
                EventType.HIERARCHICAL, "engine",
                f"Cell {cell_idx} ({cell_entry.name}): "
                f"{len(cell_node_list)}n {len(cell_edge_list)}e",
                LogLevel.DEBUG, graph=cell_subgraph,
                provenance={
                    "target_nodes": cell_node_list,
                    "target_edges": cell_edge_list,
                },
            )
        # Inter-cell edge "graph" snapshot — just the inter-cell
        # endpoints + their edges; provenance points back to the
        # relevant target nodes/edges.
        if inter_info is not None and inter_info.edges:
            inter_nodes = set()
            inter_edge_list = []
            for u, v in inter_info.edges:
                inter_nodes.add(u)
                inter_nodes.add(v)
                a, b = (u, v) if u < v else (v, u)
                inter_edge_list.append([a, b])
            try:
                inter_subgraph = graph.subgraph(inter_nodes)
            except Exception:
                inter_subgraph = None
            if inter_subgraph is not None:
                log.record(
                    EventType.HIERARCHICAL, "engine",
                    f"Inter-cell edges: {len(inter_nodes)}n "
                    f"{len(inter_edge_list)}e",
                    LogLevel.DEBUG, graph=inter_subgraph,
                    provenance={
                        "target_nodes": sorted(inter_nodes),
                        "target_edges": inter_edge_list,
                    },
                )

    def _try_formula_shortcircuit(
        self,
        graph: Graph,
        max_depth: int
    ) -> Optional[SynthesisResult]:
        """Closed-form shortcut for hierarchical graphs BEFORE treewidth_dp.

        Attempts the Phase 12 unified formula and the Phase 13 k-matching
        cell-cycle formula *without* falling through to treewidth_dp or the
        chord rule. Returns None if neither formula applies — the caller
        should continue to the standard pipeline.

        Gated to graphs where a D-Wave-style cell candidate (Cm1 = K_{4,4},
        Z1_1, or similar structured tile) exists. This avoids the VF2
        slow path on dense graphs (like router/clique hybrids) where
        K_n is the only candidate and tiling search is expensive without
        any payoff.

        Used to prefer the formula paths for visualization and for the
        Cm2-like case where the k-matching formula beats direct
        treewidth_dp (4x speedup on Cm2).
        """
        from ..graphs.covering import (apply_kmatching_formula,
                                       detect_kmatching_topology,
                                       extract_cell_topology,
                                       find_cell_candidates,
                                       try_hierarchical_partition,
                                       try_heterogeneous_partition)
        from ..graphs.k_sum import _classify_bridges_chords

        # Fast gate: skip the shortcut if no D-Wave-style cell candidate
        # exists. This leaves dense clique-hybrid graphs (router+clique)
        # unaffected since K_n-only candidates are not in the gate set.
        candidates = find_cell_candidates(graph, self.table)
        dwave_cell_names = {"Cm1", "Cm2", "Z1_1", "Z1_2"}
        if not any(c.name in dwave_cell_names for c in candidates):
            return None

        homo = try_hierarchical_partition(graph, self.table)
        # Heterogeneous partition is VF2-heavy (~60-180s/call on 72n+ graphs
        # because it walks the full ~1000-entry rainbow table). Skip in
        # recursive sub-syntheses (depth > 1): they produce derived graphs
        # (e.g., k-sum's PC = target + extra clique edges) that disrupt
        # cell structure; heterogeneous almost always fails after burning
        # significant VF2 time. Top-level calls still run both partitioners
        # so we don't miss legitimate heterogeneous covers on user inputs.
        if self._synth_depth <= 1:
            het = try_heterogeneous_partition(graph, self.table)
        else:
            het = None

        candidates = []
        if homo is not None:
            cell, partition, inter_info = homo
            candidates.append(([cell] * len(partition), partition, inter_info))
        if het is not None:
            cells, partition, inter_info = het
            candidates.append((cells, partition, inter_info))

        for cells, partition, inter_info in candidates:
            if not inter_info.edges:
                continue
            k_cells = len(partition)
            base_poly = TuttePolynomial.one()
            for c in cells:
                base_poly = base_poly * c.polynomial
            all_minors = {c.canonical_key for c in cells}
            recipe = [f"Formula shortcut: {k_cells} cells"]

            # Emit a snapshot+provenance event for each cell so the
            # visualizer can highlight where each cell lives in the
            # target graph. Also one for the inter-cell edge set.
            self._emit_partition_provenance(graph, cells, partition, inter_info)

            # Try Phase 12 unified formula
            H = extract_cell_topology(partition, list(inter_info.edges))
            if H is not None:
                T_H = self._synthesize_multigraph(H)
                unified_poly = base_poly * T_H
                if verify_spanning_trees(graph, unified_poly):
                    recipe.append(
                        f"Unified formula: H has {len(H.nodes)} nodes, "
                        f"{sum(H.edge_counts.values())} edges"
                    )
                    return SynthesisResult(
                        polynomial=unified_poly,
                        recipe=recipe,
                        verified=True,
                        method="unified_formula",
                        tiles_used=k_cells,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )

            # Try Phase 13 k-matching formula
            junctions = detect_kmatching_topology(
                graph, partition, list(inter_info.edges)
            )
            if junctions is not None and any(j.k > 1 for j in junctions):
                try:
                    km_poly = apply_kmatching_formula(
                        graph, junctions, self._synthesize_multigraph
                    )
                except Exception:
                    km_poly = None
                if km_poly is not None and verify_spanning_trees(graph, km_poly):
                    k_values = sorted({j.k for j in junctions})
                    recipe.append(
                        f"k-matching formula: {len(junctions)} junctions, "
                        f"k={k_values}"
                    )
                    return SynthesisResult(
                        polynomial=km_poly,
                        recipe=recipe,
                        verified=True,
                        method="kmatching_formula",
                        tiles_used=k_cells,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )

        return None

    def _try_hierarchical(
        self,
        graph: Graph,
        max_depth: int
    ) -> Optional[SynthesisResult]:
        """Try hierarchical tiling for graphs with repeating cell structure.

        (April 2026): cost-aware dispatch. Computes BOTH the
        homogeneous and heterogeneous partitions when available, predicts the
        chord-rule cost (`len(_classify_bridges_chords(...)[1])`) for each,
        and picks the partition with the fewest chord edges. On tie, prefers
        homogeneous (simpler `T(cell)^k` base polynomial).

        Why: Z(1,3) decomposes as 3×Z(1,1) homogeneously with
        96 inter-cell chord edges (intractable) but as Z(1,2)+Z(1,1)
        heterogeneously with ~10-20 chord edges. The previous "homogeneous
        always wins, heterogeneous is fallback" dispatch picked the bad
        decomposition.
        """
        from ..graphs.k_sum import _classify_bridges_chords

        self._log("Trying hierarchical tiling...")

        homo = try_hierarchical_partition(graph, self.table)
        het = try_heterogeneous_partition(graph, self.table)

        # Filter homogeneous through the original "is it worth it" gates.
        homo_cells = None  # type: Optional[List[MinorEntry]]
        homo_partition = None
        homo_inter = None
        homo_chords = None  # +inf if no usable partition
        if homo is not None:
            cell, partition, inter_info = homo
            k = len(partition)
            if k >= 2 and cell.edge_count >= cell.node_count:
                homo_cells = [cell] * k
                homo_partition = partition
                homo_inter = inter_info
                _, chords = _classify_bridges_chords(partition, list(inter_info.edges))
                homo_chords = len(chords)
                self._log(
                    f"Homogeneous candidate: {k} × {cell.name}, "
                    f"{len(inter_info.edges)} inter-cell ({homo_chords} chords)"
                )

        het_cells = None
        het_partition = None
        het_inter = None
        het_chords = None
        if het is not None:
            cells, partition, inter_info = het
            het_cells = cells
            het_partition = partition
            het_inter = inter_info
            _, chords = _classify_bridges_chords(partition, list(inter_info.edges))
            het_chords = len(chords)
            names = ", ".join(c.name for c in cells)
            self._log(
                f"Heterogeneous candidate: {names}, "
                f"{len(inter_info.edges)} inter-cell ({het_chords} chords)"
            )

        if homo_chords is None and het_chords is None:
            self._log("No hierarchical partition found")
            return None

        # Pick the partition with fewer chord edges. On tie, prefer
        # homogeneous (simpler base polynomial T(cell)^k vs ∏ T(cell_i)).
        prefer_het = (
            homo_chords is None or
            (het_chords is not None and het_chords < homo_chords)
        )
        if prefer_het:
            self._log(
                f"Choosing heterogeneous partition "
                f"({het_chords} chords vs homogeneous {homo_chords})"
            )
            return self._synthesize_hierarchical(
                graph, het_cells, het_partition, het_inter, max_depth,
            )

        self._log(
            f"Choosing homogeneous partition "
            f"({homo_chords} chords vs heterogeneous {het_chords})"
        )
        return self._synthesize_hierarchical(
            graph, homo_cells, homo_partition, homo_inter, max_depth,
        )

    def _synthesize_hierarchical(
        self,
        graph: Graph,
        cells: List[MinorEntry],
        partition: List[Set[int]],
        inter_info: InterCellInfo,
        max_depth: int
    ) -> SynthesisResult:
        """Compute polynomial using hierarchical cell decomposition.

        Algorithm:
        1. Base: T(disjoint cells) = ∏_i T(cell_i)
        2. Try product formula: T(full) = (∏ T(cell_i)) × ∏ T(inter_components)
           - This only works for specific structures (like Zephyr graphs)
           - Verify result; if wrong, fall back to edge-by-edge addition
        3. Fallback: boundary quotient + chord recursion via boundary_quotient_tutte

        Args:
            graph: Full graph
            cells: Per-partition cell entries (cells[i] is the rainbow-table
                entry isomorphic to ``graph.subgraph(partition[i])``).
                Homogeneous tilings pass ``[cell] * k``.
            partition: List of node sets (one per cell)
            inter_info: Information about inter-cell edges
            max_depth: Maximum recursion depth

        Returns:
            SynthesisResult with computed polynomial
        """
        k = len(partition)
        assert len(cells) == k, "cells and partition must have the same length"
        cell_names = [c.name for c in cells]
        # Compact the recipe label: collapse runs of the same cell name.
        if len(set(cell_names)) == 1:
            recipe = [f"Hierarchical: {k} × {cell_names[0]} cells"]
        else:
            recipe = [f"Hierarchical (heterogeneous): {' + '.join(cell_names)}"]
        all_minors = {c.canonical_key for c in cells}

        # Step 1: Base polynomial = ∏_i T(cell_i) (disjoint cells)
        base_poly = TuttePolynomial.one()
        for c in cells:
            base_poly = base_poly * c.polynomial

        if len(set(cell_names)) == 1:
            recipe.append(f"Base: T({cell_names[0]})^{k}")
        else:
            recipe.append(f"Base: ∏ T(cell_i) over {' + '.join(cell_names)}")
        self._log(f"Base polynomial has {base_poly.num_terms()} terms")

        # Emit visualizer-only events: per-cell snapshots + inter-cell
        # subgraph, each with provenance pointing back at the input
        # graph's nodes/edges. The visualizer uses this to highlight
        # the source location when the user hovers the sub-graph card.
        self._emit_partition_provenance(graph, cells, partition, inter_info)

        # Step 2: Phase-12 unified-formula short-circuit.
        # When every cell-pair's inter-cell edges share a single
        # vertex-pair, T(G) = (∏ T(cells)) × T(H), where H is the
        # cell-topology multigraph. Cell-agnostic: works for both
        # homogeneous and heterogeneous partitions.
        if inter_info.edges:
            H = extract_cell_topology(partition, list(inter_info.edges))
            if H is not None:
                get_log().record(
                    EventType.UNIFIED_FORMULA, "engine",
                    f"Unified formula attempt: H = {len(H.nodes)}n "
                    f"{sum(H.edge_counts.values())}e",
                    LogLevel.INFO, graph=graph,
                )
                T_H = self._synthesize_multigraph(H)
                unified_poly = base_poly * T_H
                if verify_spanning_trees(graph, unified_poly):
                    self._log(
                        f"Unified formula verified: T(G) = (∏ T(cells)) × T(H), "
                        f"H has {len(H.nodes)} nodes, {sum(H.edge_counts.values())} edges"
                    )
                    get_log().record(
                        EventType.UNIFIED_FORMULA, "engine",
                        f"Unified formula verified: T(G) = (∏ T(cells)) × T(H)",
                        LogLevel.INFO, graph=graph,
                    )
                    recipe.append(
                        f"Unified formula: cell-topology H has "
                        f"{len(H.nodes)} nodes, {sum(H.edge_counts.values())} edges"
                    )
                    return SynthesisResult(
                        polynomial=unified_poly,
                        recipe=recipe,
                        verified=True,
                        method="unified_formula",
                        tiles_used=k,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )
                # Verification failed - this would mean the Phase 11
                # proof is wrong for this configuration. Fall through to
                # the existing pipeline so we still return a correct
                # polynomial; log loudly so the case can be investigated.
                self._log(
                    "Unified formula failed verification, falling through "
                    "to product formula / chord rule"
                )

        # Step 2.5: Phase-13 k-matching cell-cycle formula.
        # When each cell-pair's inter-cell edges form a k-matching
        # (distinct vertex pairs) AND anchors per side lie in a single
        # vertex-transitive class (e.g., same bipartition side of a
        # K_{4,4} cell), apply the recursive cell-cycle formula.
        # Validated on Cm2 with 3.2x speedup vs direct treewidth_dp.
        if inter_info.edges:
            junctions = detect_kmatching_topology(
                graph, partition, list(inter_info.edges)
            )
            if junctions is not None and any(j.k > 1 for j in junctions):
                # Non-trivial k-matching topology detected.
                k_values = sorted({j.k for j in junctions})
                self._log(
                    f"k-matching topology detected: {len(junctions)} junctions, "
                    f"k values = {k_values}"
                )
                get_log().record(
                    EventType.KMATCHING_FORMULA, "engine",
                    f"k-matching topology: {len(junctions)} junctions, "
                    f"k={k_values}",
                    LogLevel.INFO, graph=graph,
                )
                try:
                    km_poly = apply_kmatching_formula(
                        graph, junctions, self._synthesize_multigraph
                    )
                except Exception as exc:
                    self._log(f"k-matching formula raised {exc!r}, falling through")
                    get_log().record(
                        EventType.KMATCHING_FORMULA, "engine",
                        f"k-matching formula raised {exc!r}, falling through",
                        LogLevel.WARN, graph=graph,
                    )
                    km_poly = None

                if km_poly is not None and verify_spanning_trees(graph, km_poly):
                    self._log(
                        f"k-matching formula verified: {len(junctions)} junctions"
                    )
                    get_log().record(
                        EventType.KMATCHING_FORMULA, "engine",
                        f"k-matching formula verified: {len(junctions)} junctions",
                        LogLevel.INFO, graph=graph,
                    )
                    recipe.append(
                        f"k-matching formula: {len(junctions)} junctions"
                    )
                    return SynthesisResult(
                        polynomial=km_poly,
                        recipe=recipe,
                        verified=True,
                        method="kmatching_formula",
                        tiles_used=k,
                        fringe_edges=0,
                        minors_used=all_minors,
                    )
                if km_poly is not None:
                    self._log(
                        "k-matching formula failed verification, falling through"
                    )
                    get_log().record(
                        EventType.KMATCHING_FORMULA, "engine",
                        "k-matching formula failed verification, falling through",
                        LogLevel.WARN, graph=graph,
                    )

        # Step 3: Try product formula for inter-cell edges
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
            from ..graphs.treewidth import \
                compute_treewidth_tutte_if_applicable as _tw_compute
            full_mg = MultiGraph.from_graph(graph)
            tw_poly = _tw_compute(full_mg, max_width=11)
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

            # Use boundary quotient + chord recursion (chord-rule based, no matroid theory).
            # See `tutte/graphs/chord_rule.py` and the reports in
            # `tutte/docs/06_*.md` and `07_*.md` for the validation that this
            # subsumes the previous Theorem 6 paths and edge-by-edge fallback.
            from ..graphs.k_sum import boundary_quotient_tutte
            self._log("Using boundary-quotient + chord-recursion (boundary quotient + chord recursion)")
            recipe.append("Product formula invalid, applying boundary quotient + chord recursion")
            poly = boundary_quotient_tutte(graph, partition, list(inter_info.edges), self)
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

    def _synthesize_connected(
        self,
        graph: Graph,
        max_depth: int
    ) -> SynthesisResult:
        """Synthesize polynomial for connected graph using creation-expansion-join."""
        _log = get_log()
        target_edges = graph.edge_count()

        # For small graphs, spanning tree expansion is faster than VF2 search.
        if target_edges <= 15:
            _log.record(EventType.EDGE_ADD, "engine",
                        f"Small graph ({target_edges}e), spanning tree expansion")
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
            _log.record(EventType.COVER_RESULT, "engine",
                        f"No cover found, falling back to spanning tree expansion")
            return self._synthesize_from_k2(graph, max_depth)

        _log.record(EventType.COVER_RESULT, "engine",
                    f"{len(cover.tiles)} tiles of {minor.name}, "
                    f"{len(cover.uncovered_edges)} uncovered edges")
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
            _log.record(EventType.EDGE_ADD, "engine",
                        f"Adding {len(uncovered_list)} uncovered edges")
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
        _log.record(EventType.VERIFY, "engine",
                    f"CEJ {'passed' if verified else 'FAILED'} Kirchhoff check")

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
