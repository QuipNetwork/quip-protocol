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

from .polynomial import TuttePolynomial
from .graph import Graph, MultiGraph
from .rainbow_table import RainbowTable, MinorEntry, load_default_table
from .covering import (
    Cover, Tile, Fringe, InterCellInfo,
    find_disjoint_cover,
    compute_fringe,
    compute_inter_tile_edges,
    analyze_tile_connections,
    try_hierarchical_partition,
)
from .validation import verify_spanning_trees
from .series_parallel import compute_sp_tutte_if_applicable
from .matroid import GraphicMatroid, FlatLattice, enumerate_flats_with_hasse
from .parallel_connection import (
    BivariateLaurentPoly,
    theorem6_parallel_connection,
    precompute_contractions,
    build_extended_cell_graph,
)


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


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class SynthesisEngine(BaseMultigraphSynthesizer):
    """Main synthesis engine using creation-expansion-join algorithm."""

    def __init__(
        self,
        table: Optional[RainbowTable] = None,
        verbose: bool = False
    ):
        """Initialize synthesis engine.

        Args:
            table: Rainbow table for lookups (loads default if None)
            verbose: Print progress information
        """
        self.table = table if table is not None else load_default_table()
        self.verbose = verbose
        self._cache: Dict[str, SynthesisResult] = {}
        self._multigraph_cache: Dict[str, TuttePolynomial] = {}  # For multigraph polynomials
        self._inter_cell_cache: Dict[str, TuttePolynomial] = {}  # For inter-cell graph polynomials
        self._mg_minors_accum: Set[str] = set()  # Accumulates minors found during multigraph synthesis

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[Synth] {msg}")

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
            return result

        # 4. Check for cut vertices (fast factorization before expensive operations)
        cut = graph.has_cut_vertex()
        if cut is not None:
            result = self._synthesize_via_cut_vertex(graph, cut, max_depth)
            self._cache[cache_key] = result
            return result

        # 5. Try hierarchical tiling for graphs with repeating structure
        if graph.edge_count() >= 20:  # Only try for larger graphs
            result = self._try_hierarchical(graph, max_depth)
            if result is not None:
                self._cache[cache_key] = result
                return result

        # 7. Try creation-expansion-join
        result = self._synthesize_connected(graph, max_depth)
        self._cache[cache_key] = result
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

            # Product formula failed - try Theorem 6 parallel connection for 2-cell case
            if len(partition) == 2:
                self._log("Product formula failed, trying Theorem 6 parallel connection")
                recipe.append("Product formula invalid, trying Theorem 6")
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

            matroid_N = GraphicMatroid(inter_graph)
            r_N = matroid_N.rank()

            self._log(f"Inter-cell matroid: rank {r_N}, {inter_graph.edge_count()} edges")

            # Step 2: Enumerate flats with Hasse diagram
            flats, ranks, upper_covers = enumerate_flats_with_hasse(matroid_N)
            self._log(f"Enumerated {len(flats)} flats")

            # Step 3: Build FlatLattice
            lattice = FlatLattice(
                matroid_N,
                flats=flats,
                ranks=ranks,
                upper_covers=upper_covers,
            )

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
                return poly
            else:
                self._log("Theorem 6 result failed Kirchhoff verification")
                return None

        except Exception as e:
            self._log(f"Theorem 6 failed with exception: {e}")
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
        5. Running multigraph updated after each edge

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

        # Process chords (T(G+e) = T(G) + T(G/{u,v}))
        for i, (u, v) in enumerate(chords):
            merged = current_mg.merge_nodes(u, v)
            merged_poly = self._synthesize_multigraph(merged, skip_minor_search=True)
            self._log(f"  Chord {i+1}/{len(chords)}")

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
                    merged_poly = self._synthesize_multigraph(merged)
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
            merged_poly = self._synthesize_multigraph(merged)

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
        # Check cache first
        cache_key = graph.canonical_key()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Rainbow table lookup (still worthwhile - it's fast)
        cached = self.table.lookup(graph)
        if cached is not None:
            result = SynthesisResult(
                polynomial=cached,
                recipe=["Rainbow table lookup"],
                verified=True,
                method="lookup",
                minors_used={cache_key} if cache_key in self.table.entries else set(),
            )
            self._cache[cache_key] = result
            return result

        # 2. Base cases
        if graph.edge_count() == 0:
            result = SynthesisResult(
                polynomial=TuttePolynomial.one(),
                recipe=["Empty graph: T = 1"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

        if graph.edge_count() == 1:
            result = SynthesisResult(
                polynomial=TuttePolynomial.x(),
                recipe=["Single edge: T = x"],
                verified=True,
                method="base_case"
            )
            self._cache[cache_key] = result
            return result

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
            self._cache[cache_key] = result
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
            self._cache[cache_key] = result
            return result

        # 5. Direct spanning tree expansion (skip minor search)
        result = self._synthesize_from_k2_fast(graph, max_depth)
        self._cache[cache_key] = result
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

        # Chords
        chords = [e for e in graph.edges if e not in tree_edges]

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
            merged_poly = self._synthesize_multigraph(merged)

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
            - "auto": Tiling for small graphs (≤12 edges), hybrid for larger
            - "tiling": Tiling-based (spanning tree + edge addition)
            - "algebraic": Pure algebraic decomposition
            - "hybrid": Combined tiling + pattern recognition

    Returns:
        SynthesisResult with computed polynomial
    """
    if method == "algebraic":
        from .algebraic_synthesis import AlgebraicSynthesisEngine
        engine = AlgebraicSynthesisEngine(verbose=verbose)
        alg_result = engine.synthesize(graph)
        return SynthesisResult(
            polynomial=alg_result.polynomial,
            recipe=alg_result.recipe,
            verified=alg_result.verified,
            method=alg_result.method
        )

    if method == "hybrid":
        from .hybrid_synthesis import HybridSynthesisEngine
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
        from .hybrid_synthesis import HybridSynthesisEngine
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
    from .algebraic_synthesis import AlgebraicSynthesisEngine
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
    from .algebraic_synthesis import AlgebraicSynthesisEngine
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


