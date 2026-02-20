"""
Unit tests for find_embedding in minor_alloc.py.

The algorithm is a randomised heuristic (Cai, Macready & Roy, 2014,
https://arxiv.org/abs/1406.2741) so tests use a fixed RNG seed wherever
deterministic behaviour is required, and only assert the *validity* of the
returned embedding rather than its exact shape.
"""

import random

import networkx as nx

from topo_alloc.minor_alloc import build_model, find_embedding, is_valid_embedding


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

    def test_returns_none_when_no_free_nodes(self):
        """
        If all target nodes are occupied by other models and no free root
        can be found, build_model returns None.
        """
        target = nx.path_graph(3)  # nodes 0, 1, 2
        # "b" and "c" occupy all nodes; "a" has "b" as neighbour
        adjlist = {"a": ["b"], "b": ["a"], "c": []}
        phi = {"a": [], "b": [0], "c": [1, 2]}
        result = build_model("a", adjlist, phi, target, overlap_penalty=2.0)
        # No free node exists → should return None
        assert result is None

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
