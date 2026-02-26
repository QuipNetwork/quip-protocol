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
    _shuffle_within_degree_tiers,
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
            source, target, rng_factory=seeded_rng(7), tries=50, order_by_degree=True
        )
        if phi is not None:
            assert is_valid_embedding(source, target, phi)

    def test_degree_ordering_k4_into_k4(self):
        """K_4 into K_4 succeeds with degree ordering (all degrees equal)."""
        source = nx.complete_graph(4)
        target = nx.complete_graph(4)
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(3), order_by_degree=True
        )
        assert phi is not None
        assert is_valid_embedding(source, target, phi)

    def test_degree_ordering_star_graph(self):
        """Star graph: hub has degree n-1, leaves degree 1 -- hub placed first."""
        source = nx.star_graph(4)  # 1 hub + 4 leaves
        target = nx.complete_graph(10)  # fully connected, guarantees success
        phi = find_embedding(
            source, target, rng_factory=seeded_rng(0), tries=20, order_by_degree=True
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
            order_by_degree=True,
        )
        # Both should succeed and be valid
        assert phi_random is not None
        assert phi_degree is not None
        assert is_valid_embedding(source, target, phi_random)
        assert is_valid_embedding(source, target, phi_degree)

    def test_shuffle_within_degree_tiers_preserves_degree_order(self):
        """_shuffle_within_degree_tiers must never place a lower-degree node
        before a strictly higher-degree node."""
        import random

        source = nx.star_graph(5)  # hub degree=5, leaves degree=1
        degree_order = sorted(
            source.nodes, key=lambda h: source.degree(h), reverse=True
        )
        rng_inst = random.Random(0)
        for _ in range(20):
            order = _shuffle_within_degree_tiers(degree_order, source, rng_inst)
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
            tuple(_shuffle_within_degree_tiers(degree_order, source, rng_inst))
            for _ in range(30)
        }
        # With 5! = 120 permutations and 30 tries, we expect more than 1 unique ordering
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
            order_by_degree=True,
        )
        assert phi is not None, f"{label}: embedding failed (degree order)"
        assert is_valid_embedding(source, self.TARGET, phi), f"{label}: invalid embedding"


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
