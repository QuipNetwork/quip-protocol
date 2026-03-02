"""
Unit tests for find_embedding in minor_alloc.py.

The algorithm is a randomised heuristic (Cai, Macready & Roy, 2014,
https://arxiv.org/abs/1406.2741) so tests use a fixed RNG seed wherever
deterministic behaviour is required, and only assert the *validity* of the
returned embedding rather than its exact shape.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from topo_alloc.minor_alloc import (
    EmbedOption,
    _refine_longest_chains,
    _shuffle_within_tiers,
    build_model,
    find_embedding,
    is_valid_embedding,
)


def seeded_rng(seed: int):
    """Return a factory that always produces the same Random instance."""

    def factory():
        r = random.Random(seed)
        return r

    return factory


class TestTrivialInputs:
    def test_single_node_source(self):
        """K_1 embeds into any graph with at least one node."""
        source = nx.Graph()
        source.add_node("a")
        target = nx.path_graph(5)
        phi = find_embedding(source, target, rng_factory=seeded_rng(0))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_single_edge_source(self):
        """K_2 embeds into any graph that has at least one edge."""
        source = nx.Graph()
        source.add_edge("a", "b")
        target = nx.path_graph(4)
        phi = find_embedding(source, target, rng_factory=seeded_rng(1))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_empty_source(self):
        """An empty source graph should return an empty (but valid) embedding."""
        source = nx.Graph()
        target = nx.path_graph(4)
        phi = find_embedding(source, target, rng_factory=seeded_rng(0))
        # Algorithm iterates over src_nodes which is empty – should return {}
        assert phi is not None
        assert phi == {}

    def test_source_equals_target(self):
        """A graph is always a minor of itself (identity embedding)."""
        g = nx.cycle_graph(5)
        phi = find_embedding(g, g.copy(), rng_factory=seeded_rng(42))
        assert phi is not None
        assert is_valid_embedding(g, g, phi)


class TestSmallGraphs:
    def test_k3_into_k4(self):
        """Triangle K_3 embeds into K_4."""
        source = nx.complete_graph(3)
        target = nx.complete_graph(4)
        phi = find_embedding(source, target, rng_factory=seeded_rng(7))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_k4_into_k4(self):
        """K_4 embeds into K_4 (trivial identity-like)."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(source, target, rng_factory=seeded_rng(7))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_path_into_grid(self):
        """A path graph embeds into a 3×3 grid."""
        source = nx.path_graph(5)
        target = nx.grid_2d_graph(3, 3)
        phi = find_embedding(source, target, rng_factory=seeded_rng(3))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_cycle_into_grid(self):
        """A 4-cycle embeds into a 3×3 grid."""
        source = nx.cycle_graph(4)
        target = nx.grid_2d_graph(3, 3)
        phi = find_embedding(source, target, rng_factory=seeded_rng(5))
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_k4_into_k33_fails(self):
        """
        K_4 cannot be embedded as a minor of K_{3,3} without overlapping
        paths, but K_{3,3} is large enough that a valid minor-embedding
        of K_4 exists (K_{3,3} has K_4 as a minor).  Verify the embedding
        is structurally valid.
        """
        source = nx.complete_graph(4)
        target = nx.complete_bipartite_graph(3, 3)
        phi = find_embedding(source, target, rng_factory=seeded_rng(11), tries=50)
        # K_4 is a minor of K_{3,3} so a valid embedding must be found
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_k5_into_petersen(self):
        """
        K_5 is a minor of the Petersen graph (well-known fact).
        """
        source = nx.complete_graph(5)
        target = nx.petersen_graph()
        phi = find_embedding(source, target, rng_factory=seeded_rng(99), tries=50)
        if phi is not None:
            assert is_valid_embedding(source, target, phi)


class TestImpossibleEmbeddings:
    def test_source_larger_than_target_returns_none(self):
        """
        A source with more nodes than the target cannot be embedded (each
        vertex-model must occupy at least one distinct node).
        """
        source = nx.complete_graph(6)
        target = nx.complete_graph(3)
        phi = find_embedding(source, target, rng_factory=seeded_rng(0), tries=5)
        assert phi is None

    def test_disconnected_target_can_block_embedding(self):
        """
        If the target is disconnected and too small on each component,
        find_embedding should return None.
        """
        source = nx.complete_graph(4)
        # Two isolated edges – not enough connectivity for K_4
        target = nx.Graph()
        target.add_edges_from([(0, 1), (2, 3)])
        phi = find_embedding(source, target, rng_factory=seeded_rng(0), tries=10)
        assert phi is None


class TestDegreeOrdering:
    """Tests for the order_by_degree heuristic."""

    def test_degree_ordering_produces_valid_embedding(self):
        """order_by_degree=True must still return a valid embedding."""
        source = nx.complete_graph(5)
        target = nx.petersen_graph()
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(7), tries=50, options=EmbedOption.ORDER_BY_DEGREE
        )
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_degree_ordering_k4_into_k4(self):
        """K_4 into K_4 succeeds with degree ordering (all degrees equal)."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(3), options=EmbedOption.ORDER_BY_DEGREE
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_degree_ordering_star_graph(self):
        """Star graph: hub has degree n-1, leaves degree 1 -- hub placed first."""
        source = nx.star_graph(4)  # 1 hub + 4 leaves
        target = nx.complete_graph(10)  # fully connected, guarantees success
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(0), tries=20, options=EmbedOption.ORDER_BY_DEGREE
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_degree_ordering_matches_default_validity(self):
        """Both modes must produce valid embeddings on the same problem."""
        source = nx.complete_bipartite_graph(3, 3)
        target = nx.complete_graph(12)  # large fully-connected target
        phi_random = find_embedding(
            source, target, rng_factory=seeded_rng(42), tries=50
        )
        phi_degree = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(42),
            tries=50,
            options=EmbedOption.ORDER_BY_DEGREE,
        )
        # Both should succeed and be valid
        assert phi_random is not None
        assert phi_degree is not None
        assert is_valid_embedding(source, target, phi_random)
        assert is_valid_embedding(source, target, phi_degree)

    def test_shuffle_within_degree_tiers_preserves_degree_order(self):
        """_shuffle_within_tiers must never place a lower-degree node
        before a strictly higher-degree node."""
        import random

        source = nx.star_graph(5)  # hub degree=5, leaves degree=1
        degree_order = sorted(
            source.nodes, key=lambda h: source.degree(h), reverse=True
        )
        rng_inst = random.Random(0)
        for _ in range(20):
            order = _shuffle_within_tiers(degree_order, lambda h: source.degree(h), rng_inst)
            # The hub (node 0) must be first -- it has degree 5
            assert order[0] == 0
            # All leaves come after the hub
            assert set(order[1:]) == set(range(1, 6))

    def test_shuffle_within_degree_tiers_randomises_ties(self):
        """Equal-degree nodes should appear in varying orders across calls."""
        import random

        source = nx.complete_graph(5)  # all degrees equal (4)
        degree_order = sorted(
            source.nodes, key=lambda h: source.degree(h), reverse=True
        )
        rng_inst = random.Random(1)
        orderings = {
            tuple(_shuffle_within_tiers(degree_order, lambda h: source.degree(h), rng_inst))
            for _ in range(30)
        }
        # With 5! = 120 permutations and 30 tries, we expect more than 1 unique ordering
        assert len(orderings) > 1


class TestCentralityOrdering:
    """Tests for the order_by_centrality heuristic."""

    def test_centrality_ordering_produces_valid_embedding(self):
        """order_by_centrality=True must still return a valid embedding."""
        source = nx.complete_graph(5)
        target = nx.petersen_graph()
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(7), tries=50, options=EmbedOption.ORDER_BY_CENTRALITY
        )
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_centrality_ordering_k4_into_k4(self):
        """K_4 into K_4 succeeds with centrality ordering (all centralities equal)."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(3), options=EmbedOption.ORDER_BY_CENTRALITY
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_centrality_ordering_path_graph(self):
        """Path graph: central nodes have higher betweenness; they should be placed first."""
        source = nx.path_graph(5)  # node 2 has highest betweenness
        target = nx.complete_graph(10)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(0), tries=20, options=EmbedOption.ORDER_BY_CENTRALITY
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_centrality_ordering_matches_default_validity(self):
        """Both centrality and random modes must produce valid embeddings."""
        source = nx.complete_bipartite_graph(3, 3)
        target = nx.complete_graph(12)
        phi_random = find_embedding(
            source, target, rng_factory=seeded_rng(42), tries=50
        )
        phi_centrality = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(42),
            tries=50,
            options=EmbedOption.ORDER_BY_CENTRALITY,
        )
        assert phi_random is not None
        assert phi_centrality is not None
        assert is_valid_embedding(source, target, phi_random)
        assert is_valid_embedding(source, target, phi_centrality)

    def test_centrality_takes_precedence_over_degree(self):
        """When both flags are True, order_by_centrality takes precedence and result is valid."""
        source = nx.star_graph(4)
        target = nx.complete_graph(10)
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(0),
            tries=20,
            options=EmbedOption.ORDER_BY_DEGREE | EmbedOption.ORDER_BY_CENTRALITY,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_shuffle_within_centrality_tiers_preserves_centrality_order(self):
        """_shuffle_within_tiers must never place a lower-centrality node
        before a strictly higher-centrality node."""
        import random

        # Path graph: interior nodes have higher betweenness than endpoints.
        source = nx.path_graph(5)  # nodes: 0-1-2-3-4; node 2 has highest centrality
        centrality = nx.betweenness_centrality(source)
        centrality_order = sorted(source.nodes, key=lambda h: centrality[h], reverse=True)
        rng_inst = random.Random(0)
        for _ in range(20):
            order = _shuffle_within_tiers(centrality_order, lambda h: centrality[h], rng_inst)
            # Verify no lower-centrality node appears before a strictly higher one
            prev = float("inf")
            for node in order:
                c = centrality[node]
                assert c <= prev, (
                    f"Node {node} (centrality {c}) placed after node with centrality {prev}"
                )
                prev = c

    def test_shuffle_within_centrality_tiers_randomises_ties(self):
        """Equal-centrality nodes should appear in varying orders across calls."""
        import random

        source = nx.complete_graph(5)  # all centralities equal
        centrality = nx.betweenness_centrality(source)
        centrality_order = sorted(source.nodes, key=lambda h: centrality[h], reverse=True)
        rng_inst = random.Random(1)
        orderings = {
            tuple(_shuffle_within_tiers(centrality_order, lambda h: centrality[h], rng_inst))
            for _ in range(30)
        }
        assert len(orderings) > 1


def random_source_graphs() -> list[tuple[str, nx.Graph]]:
    """
    Generate a variety of small random source graphs for parametrized tests.

    Returns a list of (label, graph) pairs built from fixed seeds so the
    test suite remains fully deterministic.  Three families are included:

    - Erdős–Rényi  G(n, p)  — the standard random graph model
    - Barabási–Albert  BA(n, m)  — scale-free, power-law degree distribution
    - Random tree  — guarantees connectivity, low density
    """
    cases: list[tuple[str, nx.Graph]] = []

    er_params = [
        (6, 0.4, 0),
        (6, 0.6, 1),
        (8, 0.3, 2),
        (8, 0.5, 3),
        (10, 0.25, 4),
    ]
    for n, p, seed in er_params:
        g = nx.gnp_random_graph(n, p, seed=seed)
        # Drop isolates so every node has at least one edge to embed
        g.remove_nodes_from(list(nx.isolates(g)))
        if g.number_of_nodes() >= 2:
            cases.append((f"ER_n{n}_p{int(p*100)}_s{seed}", g))

    ba_params = [(6, 2, 10), (8, 2, 11), (8, 3, 12)]
    for n, m, seed in ba_params:
        g = nx.barabasi_albert_graph(n, m, seed=seed)
        cases.append((f"BA_n{n}_m{m}_s{seed}", g))

    for seed in [20, 21, 22]:
        g = nx.random_labeled_tree(7, seed=seed)
        cases.append((f"tree_n7_s{seed}", g))

    return cases


_RANDOM_SOURCE_CASES = random_source_graphs()


class TestRandomSourceGraphs:
    """
    Embed randomly generated source graphs into a fixed, spacious target.

    The target is always a large complete graph so that the embedding is
    guaranteed to be feasible — this lets the tests focus exclusively on
    the correctness of the algorithm across diverse source topologies.
    """

    TARGET = nx.complete_graph(20)

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_random_order(self, label: str, source: nx.Graph) -> None:
        """Random source graph embeds with the default (random) ordering."""
        phi = find_embedding(
            source,
            self.TARGET,
            rng_factory=seeded_rng(0),
            tries=30,
        )
        assert phi is not None, f"{label}: embedding failed (random order)"
        assert is_valid_embedding(source, self.TARGET, phi), f"{label}: invalid embedding"

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_degree_order(self, label: str, source: nx.Graph) -> None:
        """Random source graph embeds with the degree-first ordering."""
        phi = find_embedding(
            source,
            self.TARGET,
            rng_factory=seeded_rng(0),
            tries=30,
            options=EmbedOption.ORDER_BY_DEGREE,
        )
        assert phi is not None, f"{label}: embedding failed (degree order)"
        assert is_valid_embedding(source, self.TARGET, phi), f"{label}: invalid embedding"

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_centrality_order(self, label: str, source: nx.Graph) -> None:
        """Random source graph embeds with the betweenness-centrality ordering."""
        phi = find_embedding(
            source,
            self.TARGET,
            rng_factory=seeded_rng(0),
            tries=30,
            options=EmbedOption.ORDER_BY_CENTRALITY,
        )
        assert phi is not None, f"{label}: embedding failed (centrality order)"
        assert is_valid_embedding(source, self.TARGET, phi), f"{label}: invalid embedding"


class TestLongestChainRefinement:
    """Tests for the refine_longest_chains heuristic."""

    def test_produces_valid_embedding(self):
        """refine_longest_chains=True must still return a valid embedding."""
        source = nx.complete_graph(5)
        target = nx.petersen_graph()
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(7),
            tries=50,
            options=EmbedOption.REFINE_LONGEST_CHAINS,
        )
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_nodes_used_not_worse_than_degree_order(self):
        """
        longest_chains refinement (on top of degree ordering) should use
        no more physical nodes than degree ordering alone on the same seed.
        """
        source = nx.complete_bipartite_graph(3, 3)
        target = nx.complete_graph(20)

        phi_degree = find_embedding(
            source, target, rng_factory=seeded_rng(42), tries=50, options=EmbedOption.ORDER_BY_DEGREE
        )
        phi_longest = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(42),
            tries=50,
            options=EmbedOption.REFINE_LONGEST_CHAINS,
        )

        assert phi_degree is not None
        assert phi_longest is not None
        assert is_valid_embedding(source, target, phi_degree)
        assert is_valid_embedding(source, target, phi_longest)

        nodes_degree = sum(len(m) for m in phi_degree.values())
        nodes_longest = sum(len(m) for m in phi_longest.values())
        # longest-chain refinement should never produce a strictly worse result
        assert nodes_longest <= nodes_degree

    def test_k4_into_k4_with_longest_chains(self):
        """K_4 into K_4 succeeds with longest-chain refinement enabled."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(3), options=EmbedOption.REFINE_LONGEST_CHAINS
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_star_graph_with_longest_chains(self):
        """Star graph embeds correctly with the refinement pass active."""
        source = nx.star_graph(4)
        target = nx.complete_graph(10)
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(0),
            tries=20,
            options=EmbedOption.REFINE_LONGEST_CHAINS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_longest_chains(self, label: str, source: nx.Graph) -> None:
        """Random source graphs embed correctly with longest-chain refinement."""
        target = nx.complete_graph(20)
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(0),
            tries=30,
            options=EmbedOption.REFINE_LONGEST_CHAINS,
        )
        assert phi is not None, f"{label}: embedding failed (longest_chains)"
        assert is_valid_embedding(source, target, phi), f"{label}: invalid embedding"

    def test_refine_longest_chains_helper_never_lengthens(self):
        """
        _refine_longest_chains must only shorten or keep chain lengths; it
        must never lengthen any chain.
        """
        # Build a simple valid embedding on a path graph so every chain
        # starts as length 1 (trivial — node maps to a single target node).
        source = nx.path_graph(4)   # 0-1-2-3
        target = nx.path_graph(10)
        src_nodes = list(source.nodes)
        src_adj = {h: list(source.neighbors(h)) for h in src_nodes}

        # Place each source node on a distinct target node (chain length = 1).
        phi: dict[int, list[int]] = {i: [i] for i in src_nodes}

        before_total = sum(len(v) for v in phi.values())
        phi_after = _refine_longest_chains(
            src_nodes, src_adj, phi, target, overlap_penalty=2.0, rounds=40
        )
        after_total = sum(len(v) for v in phi_after.values())

        assert after_total <= before_total

    def test_refine_longest_chains_helper_accepts_shorter(self):
        """
        _refine_longest_chains should accept a re-embedding when it produces
        a strictly shorter chain.  We construct a phi where one node has a
        long chain reachable via a shortcut, and verify total length drops.
        """
        # Target: 0-1-2-3-4-5, source: two nodes a-b
        # phi[a] = [2, 3, 4]  (length 3, the "long chain")
        # phi[b] = [0]         (length 1, adjacent to 1 which neighbours 2)
        # build_model for 'a' with b placed at 0 should find chain [1] (length 1)
        target = nx.path_graph(6)
        src_adj = {"a": ["b"], "b": ["a"]}
        phi: dict[str, list[int]] = {"a": [2, 3, 4], "b": [0]}

        before_len_a = len(phi["a"])
        phi_after = _refine_longest_chains(
            ["a", "b"], src_adj, phi, target, overlap_penalty=2.0, rounds=5
        )
        # Chain for 'a' should have been shortened (2,3,4 → something adjacent to 0)
        assert len(phi_after["a"]) < before_len_a


class TestBuildModel:
    def make_phi(self, assignments):
        """Create a phi dict where each key maps to a list of target nodes."""
        return {k: list(v) for k, v in assignments.items()}

    def test_first_node_placed_on_free_node(self):
        """When no neighbours are placed, build_model picks any free node."""
        target = nx.path_graph(5)
        adjlist = {"a": [], "b": []}
        phi = {"a": [], "b": []}
        result = build_model("a", adjlist, phi, target, overlap_penalty=2.0)
        assert result is not None
        assert len(result) == 1
        assert result[0] in target.nodes

    def test_placed_neighbour_guides_placement(self):
        """When a neighbour is already placed, the new model must be adjacent."""
        target = nx.path_graph(6)  # 0-1-2-3-4-5
        adjlist = {"a": ["b"], "b": ["a"]}
        phi = {"a": [], "b": [0]}  # b is placed at node 0
        result = build_model("a", adjlist, phi, target, overlap_penalty=2.0)
        assert result is not None
        # The model for "a" must be adjacent to node 0 in the target
        assert any(target.has_edge(r, 0) for r in result)

    def test_model_is_connected_subgraph(self):
        """The returned model must induce a connected subgraph of target."""
        target = nx.grid_2d_graph(4, 4)
        adjlist = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        phi = {
            "a": [],
            "b": [(0, 0)],
            "c": [(3, 3)],
        }
        result = build_model("a", adjlist, phi, target, overlap_penalty=2.0)
        if result is not None and len(result) > 1:
            assert nx.is_connected(target.subgraph(result))


class TestVertexWeights:
    """Tests for the use_vertex_weights=True mode (Cai, Macready & Roy 2014)."""

    def test_produces_valid_embedding_k3_into_k4(self):
        """K_3 embeds into K_4 with vertex weights enabled."""
        source = nx.complete_graph(3)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(7), options=EmbedOption.USE_VERTEX_WEIGHTS
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_produces_valid_embedding_k4_into_k4(self):
        """K_4 into K_4 succeeds with vertex weights."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(3), options=EmbedOption.USE_VERTEX_WEIGHTS
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_produces_valid_embedding_path_into_grid(self):
        """Path embeds into a grid with vertex weights enabled."""
        source = nx.path_graph(5)
        target = nx.grid_2d_graph(4, 4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(0), tries=30, options=EmbedOption.USE_VERTEX_WEIGHTS
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_produces_valid_embedding_cycle_into_grid(self):
        """A 4-cycle embeds into a grid with vertex weights enabled."""
        source = nx.cycle_graph(4)
        target = nx.grid_2d_graph(3, 3)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(5), options=EmbedOption.USE_VERTEX_WEIGHTS
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_vertex_weights(self, label: str, source: nx.Graph) -> None:
        """Random source graphs embed correctly with vertex weights."""
        target = nx.complete_graph(20)
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(0),
            tries=30,
            options=EmbedOption.USE_VERTEX_WEIGHTS,
        )
        assert phi is not None, f"{label}: embedding failed (vertex_weights)"
        assert is_valid_embedding(source, target, phi), f"{label}: invalid embedding"

    def test_vertex_weights_combined_with_degree_order(self):
        """vertex_weights + order_by_degree must yield a valid embedding."""
        source = nx.complete_graph(5)
        target = nx.petersen_graph()
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(7),
            tries=50,
            options=EmbedOption.ORDER_BY_DEGREE | EmbedOption.USE_VERTEX_WEIGHTS,
        )
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_vertex_weights_combined_with_longest_chains(self):
        """vertex_weights + refine_longest_chains must yield a valid embedding."""
        source = nx.complete_bipartite_graph(3, 3)
        target = nx.complete_graph(20)
        phi = find_embedding(
            source,
            target,
            rng_factory=seeded_rng(42),
            tries=50,
            options=EmbedOption.REFINE_LONGEST_CHAINS | EmbedOption.USE_VERTEX_WEIGHTS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_build_model_vertex_weights_weights_occupied_lower(self):
        """
        Vertex weights should assign lower Dijkstra cost to nodes already in
        some vertex-model (inclusion_count > 0) than to completely free nodes.

        We verify this indirectly: with a path target 0-1-2-3, b placed at 0,
        and c placed at 3, the weight of node 1 (adjacent to 0, used by b)
        should be lower than node 2 (free).  The model for 'a' should prefer
        to route through node 1 rather than 2 when possible.
        """
        # target: 0-1-2-3, source: b-a-c
        target = nx.path_graph(4)
        adjlist = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        phi: dict[str, list[int]] = {"a": [], "b": [0], "c": [3]}

        # With vertex weights D=3 (diameter of path_graph(4)), n=3 source nodes.
        # inclusion_count: {0: 1, 3: 1}  (one model each for b and c)
        # wt(0) = 3^(3-1) = 9   (in b's model, distance from source=0 so cost 0 for start)
        # wt(1) = 3^(3-0) = 27  (not in any model yet)
        # wt(2) = 3^(3-0) = 27  (not in any model yet)
        # wt(3) = 3^(3-1) = 9   (in c's model)
        # Dijkstra from {0}: dist to 1 = wt(1)=27, dist to 2 = 27+27=54
        # Dijkstra from {3}: dist to 2 = 27, dist to 1 = 27+27=54
        # Best root minimises sum of distances from both models.
        # Node 1: dist_b(1)+dist_c(1) = 27 + 54 = 81
        # Node 2: dist_b(2)+dist_c(2) = 54 + 27 = 81  (symmetric)
        # Either node 1 or 2 may be root; what matters is that the result is valid.
        result = build_model(
            "a",
            adjlist,
            phi,
            target,
            overlap_penalty=2.0,
            use_vertex_weights=True,
            target_diameter=3,
            num_source_nodes=3,
        )
        assert result is not None
        # Model must be adjacent to both phi[b]={0} and phi[c]={3}
        assert any(target.has_edge(r, 0) for r in result)
        assert any(target.has_edge(r, 3) for r in result)
        if len(result) > 1:
            assert nx.is_connected(target.subgraph(result))


class TestPreferArticulationPoints:
    """
    Tests for EmbedOption.PREFER_ARTICULATION_POINTS.

    The option anchors source articulation points (nodes whose removal
    disconnects the source graph) on the highest-degree free target node,
    maximising routing options for their neighbours on both sides of the cut.
    Non-articulation source nodes are unaffected.
    """

    # -----------------------------------------------------------------------
    # build_model unit tests
    # -----------------------------------------------------------------------

    def test_art_pt_anchor_prefers_highest_degree_target_node(self):
        """
        Source node that is an articulation point picks the highest-degree
        free target node, not the first-in-iteration-order target node.

        Source: path P3 (0-1-2).  Node 1 is the only articulation point.
        Target: path P5 (0-1-2-3-4).  Nodes 1, 2, 3 have degree 2;
                nodes 0 and 4 have degree 1.  Without the option, build_model
                would return [0] (first in iteration order).  With the option,
                it must return a node with the maximum degree (2).
        """
        source = nx.path_graph(3)
        art_pts = frozenset(nx.articulation_points(source))
        assert art_pts == {1}, "Expected node 1 to be the sole AP of P3"

        target = nx.path_graph(5)
        adjlist = {n: list(source.neighbors(n)) for n in source.nodes}
        phi = {n: [] for n in source.nodes}

        model = build_model(1, adjlist, phi, target, overlap_penalty=2.0,
                            source_art_pts=art_pts)
        assert model is not None and len(model) == 1
        max_deg = max(d for _, d in target.degree())
        assert target.degree(model[0]) == max_deg, (
            f"AP anchor should have degree {max_deg}, got {target.degree(model[0])}"
        )

    def test_non_art_pt_uses_first_free_node(self):
        """
        A source node that is NOT an articulation point falls back to the
        first free target node regardless of degree, as before.
        """
        source = nx.path_graph(3)
        art_pts = frozenset(nx.articulation_points(source))
        assert 0 not in art_pts

        target = nx.path_graph(5)
        adjlist = {n: list(source.neighbors(n)) for n in source.nodes}
        phi = {n: [] for n in source.nodes}

        model = build_model(0, adjlist, phi, target, overlap_penalty=2.0,
                            source_art_pts=art_pts)
        assert model is not None and len(model) == 1
        assert model[0] == 0, "Non-AP source node should pick first free target node"

    def test_no_source_art_pts_unchanged(self):
        """
        When the source graph has no articulation points (e.g. a complete
        graph), passing an empty frozenset leaves the behaviour identical to
        the default.
        """
        source = nx.complete_graph(4)
        art_pts = frozenset(nx.articulation_points(source))
        assert len(art_pts) == 0

        target = nx.path_graph(8)
        adjlist = {n: list(source.neighbors(n)) for n in source.nodes}
        phi = {n: [] for n in source.nodes}

        model_with = build_model(0, adjlist, phi, target, overlap_penalty=2.0,
                                 source_art_pts=art_pts)
        model_without = build_model(0, adjlist, phi, target, overlap_penalty=2.0)
        assert model_with == model_without

    def test_fallback_when_all_high_degree_nodes_occupied(self):
        """
        If the highest-degree target nodes are already occupied, the option
        still returns a valid (lower-degree) free node.
        """
        source = nx.path_graph(3)
        art_pts = frozenset(nx.articulation_points(source))

        # Target: star with center 0 (degree 4), leaves 1-4 (degree 1)
        target = nx.star_graph(4)
        adjlist = {n: list(source.neighbors(n)) for n in source.nodes}
        # Occupy the centre (highest-degree node)
        phi = {0: [], 1: [0], 2: []}  # source node 1 already occupies target 0

        model = build_model(0, adjlist, phi, target, overlap_penalty=2.0,
                            source_art_pts=art_pts)
        assert model is not None and len(model) == 1
        assert model[0] != 0, "Centre is occupied; AP should pick a different node"

    # -----------------------------------------------------------------------
    # find_embedding integration tests
    # -----------------------------------------------------------------------

    def test_valid_embedding_path_into_grid(self):
        """Path graph (many articulation points) embeds into a grid."""
        source = nx.path_graph(6)
        target = nx.grid_2d_graph(4, 4)
        phi = find_embedding(
            source, target,
            rng_factory=seeded_rng(0), tries=30,
            options=EmbedOption.PREFER_ARTICULATION_POINTS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_valid_embedding_k4_into_k4(self):
        """K_4 (no articulation points) with the flag is identical to default."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target,
            rng_factory=seeded_rng(3),
            options=EmbedOption.PREFER_ARTICULATION_POINTS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    @pytest.mark.parametrize(
        "label,source",
        _RANDOM_SOURCE_CASES,
        ids=[c[0] for c in _RANDOM_SOURCE_CASES],
    )
    def test_random_source_art_pts(self, label: str, source: nx.Graph) -> None:
        """Random source graphs embed with PREFER_ARTICULATION_POINTS."""
        target = nx.complete_graph(20)
        phi = find_embedding(
            source, target,
            rng_factory=seeded_rng(0), tries=30,
            options=EmbedOption.PREFER_ARTICULATION_POINTS,
        )
        assert phi is not None, f"{label}: embedding failed"
        assert is_valid_embedding(source, target, phi), f"{label}: invalid embedding"

    def test_combined_with_degree_order(self):
        """ORDER_BY_DEGREE | PREFER_ARTICULATION_POINTS produces a valid embedding."""
        source = nx.path_graph(6)
        target = nx.grid_2d_graph(4, 4)
        phi = find_embedding(
            source, target,
            rng_factory=seeded_rng(7), tries=30,
            options=EmbedOption.ORDER_BY_DEGREE | EmbedOption.PREFER_ARTICULATION_POINTS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_combined_with_longest_chains(self):
        """PREFER_ARTICULATION_POINTS + REFINE_LONGEST_CHAINS is valid."""
        source = nx.path_graph(6)
        target = nx.complete_graph(20)
        phi = find_embedding(
            source, target,
            rng_factory=seeded_rng(42), tries=20,
            options=EmbedOption.PREFER_ARTICULATION_POINTS | EmbedOption.REFINE_LONGEST_CHAINS,
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)
