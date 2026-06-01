# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 1 tests: shared CSR helpers in sampler_utils.py.

Pure CPU tests — no GPU required. Verifies build_csr_structure_from_edges
and build_edge_position_index produce correct structure and enable O(1)
J-value updates.
"""

import numpy as np
import pytest

from GPU.sampler_utils import (
    build_csr_from_ising,
    build_csr_structure_from_edges,
    build_edge_position_index,
)


def _small_graph():
    """5-node path graph: 0-1-2-3-4."""
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return nodes, edges


def _small_triangle():
    """3-node complete graph: 0-1, 1-2, 0-2."""
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (0, 2)]
    return nodes, edges


class TestBuildCsrStructureSmallGraph:
    """Verify CSR structure on a small, hand-checkable graph."""

    def test_shapes(self):
        nodes, edges = _small_graph()
        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        assert N == 5
        assert nnz == 2 * len(edges)  # symmetric
        assert row_ptr.shape == (N + 1,)
        assert col_ind.shape == (nnz,)

    def test_row_ptr_counts(self):
        nodes, edges = _small_graph()
        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        # Degree of each node in a path: [1, 2, 2, 2, 1]
        for i in range(N):
            degree = row_ptr[i + 1] - row_ptr[i]
            expected = len(neighbors[i])
            assert degree == expected, (
                f"Node {i}: degree {degree} != {expected}"
            )

    def test_node_to_idx_is_dense(self):
        nodes, edges = _small_graph()
        _, _, node_to_idx, _, N, _ = (
            build_csr_structure_from_edges(edges, nodes)
        )
        assert set(node_to_idx.values()) == set(range(N))

    def test_neighbors_sorted(self):
        nodes, edges = _small_graph()
        _, _, _, neighbors, N, _ = (
            build_csr_structure_from_edges(edges, nodes)
        )
        for i in range(N):
            assert neighbors[i] == sorted(neighbors[i])


class TestBuildCsrStructureFromTopology:
    """Test with full Zephyr topology."""

    def test_topology_sizes(self):
        from dwave_topologies import DEFAULT_TOPOLOGY

        topo = DEFAULT_TOPOLOGY
        nodes = list(topo.graph.nodes())
        edges = list(topo.graph.edges())

        row_ptr, col_ind, node_to_idx, _, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )

        assert N == len(nodes)
        assert nnz == 2 * len(edges)
        assert row_ptr.dtype == np.int32
        assert col_ind.dtype == np.int32


class TestEdgePositionIndex:
    """Verify edge position index enables correct J-value filling."""

    def test_roundtrip_small(self):
        """Fill J via positions and verify matches build_csr_from_ising."""
        nodes, edges = _small_triangle()
        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        positions = build_edge_position_index(
            edges, node_to_idx, row_ptr, neighbors,
        )

        # Assign known J values
        J_dict = {(0, 1): -1, (1, 2): -2, (0, 2): -3}
        h_dict = {0: 0, 1: 0, 2: 0}

        # Fill via edge positions (the fast path)
        j_vals_fast = np.zeros(nnz, dtype=np.int8)
        for eidx, ((i, j), Jij) in enumerate(J_dict.items()):
            pos_ij, pos_ji = positions[eidx]
            j_vals_fast[pos_ij] = int(Jij)
            j_vals_fast[pos_ji] = int(Jij)

        # Fill via build_csr_from_ising (the reference path)
        (ref_rp, ref_ci, ref_jv, ref_hv,
         _, _, ref_nti, _) = build_csr_from_ising(
            [h_dict], [J_dict],
        )

        # Both should produce identical J_vals arrays
        np.testing.assert_array_equal(j_vals_fast, ref_jv)

    def test_symmetry(self):
        """For each edge (i,j), both CSR entries get the same value."""
        nodes, edges = _small_graph()
        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        positions = build_edge_position_index(
            edges, node_to_idx, row_ptr, neighbors,
        )

        J_dict = {e: -(idx + 1) for idx, e in enumerate(edges)}
        j_vals = np.zeros(nnz, dtype=np.int8)
        for eidx, ((i, j), Jij) in enumerate(J_dict.items()):
            pos_ij, pos_ji = positions[eidx]
            j_vals[pos_ij] = int(Jij)
            j_vals[pos_ji] = int(Jij)

        # Verify symmetry: j_vals[pos_ij] == j_vals[pos_ji]
        for eidx, (i, j) in enumerate(edges):
            pos_ij, pos_ji = positions[eidx]
            assert j_vals[pos_ij] == j_vals[pos_ji], (
                f"Edge ({i},{j}): J[{pos_ij}]={j_vals[pos_ij]} "
                f"!= J[{pos_ji}]={j_vals[pos_ji]}"
            )

    def test_position_count_matches_edges(self):
        nodes, edges = _small_graph()
        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        positions = build_edge_position_index(
            edges, node_to_idx, row_ptr, neighbors,
        )
        assert len(positions) == len(edges)

    def test_topology_roundtrip(self):
        """Full topology: positions-based fill matches reference CSR."""
        from dwave_topologies import DEFAULT_TOPOLOGY
        from shared.quantum_proof_of_work import (
            generate_ising_model_from_nonce,
        )

        topo = DEFAULT_TOPOLOGY
        nodes = list(topo.graph.nodes())
        edges = list(topo.graph.edges())

        row_ptr, col_ind, node_to_idx, neighbors, N, nnz = (
            build_csr_structure_from_edges(edges, nodes)
        )
        positions = build_edge_position_index(
            edges, node_to_idx, row_ptr, neighbors,
        )

        h, J = generate_ising_model_from_nonce(42, nodes, edges)

        # Fast path fill
        j_vals_fast = np.zeros(nnz, dtype=np.int8)
        for eidx, ((i, j), Jij) in enumerate(J.items()):
            pos_ij, pos_ji = positions[eidx]
            j_vals_fast[pos_ij] = int(Jij)
            j_vals_fast[pos_ji] = int(Jij)

        # Reference fill
        (ref_rp, ref_ci, ref_jv, _, _, _, _, _) = (
            build_csr_from_ising([h], [J])
        )

        np.testing.assert_array_equal(j_vals_fast, ref_jv)
