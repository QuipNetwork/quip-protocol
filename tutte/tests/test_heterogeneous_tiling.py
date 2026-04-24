"""Tests for heterogeneous hierarchical tiling (Phase 3.1).

The homogeneous partitioner (`try_hierarchical_partition`) only finds k
identical cells. The heterogeneous variant (`try_heterogeneous_partition`)
greedily picks the largest cell from the rainbow table that fits, then the
next-largest, etc., enabling mixed decompositions like K_4 + 2·K_3.

These tests pin both the partitioner and the engine's hierarchical handler
when given a heterogeneous partition.
"""

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.graphs.covering import (
    try_hierarchical_partition,
    try_heterogeneous_partition,
)
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.validation import verify_spanning_trees


@pytest.fixture(scope="module")
def table():
    return load_default_table()


@pytest.fixture(scope="module")
def engine(table):
    return SynthesisEngine(table=table, verbose=False)


def _disjoint_blocks_with_bridges(block_sizes, cross_pairs):
    """Build the disjoint union of complete graphs sized per ``block_sizes``,
    then add the listed (block_i_node, block_j_node) inter-block edges."""
    G = nx.Graph()
    offsets = []
    cursor = 0
    for size in block_sizes:
        offsets.append(cursor)
        block = nx.complete_graph(size)
        G = nx.disjoint_union(G, block)
        cursor += size
    for (bi, ni), (bj, nj) in cross_pairs:
        G.add_edge(offsets[bi] + ni, offsets[bj] + nj)
    return Graph.from_networkx(G)


def test_partitioner_finds_k4_plus_2k3(table):
    """K_4 + K_3 + K_3 disjoint union (10 nodes) — the homogeneous
    partitioner can't handle a non-uniform decomposition; the heterogeneous
    variant should pick K_4 first, then two K_3's."""
    g = _disjoint_blocks_with_bridges([4, 3, 3], cross_pairs=[])
    assert try_hierarchical_partition(g, table) is None
    het = try_heterogeneous_partition(g, table)
    assert het is not None
    cells, partition, inter_info = het
    sizes = sorted(len(p) for p in partition)
    assert sizes == [3, 3, 4]
    assert sum(len(p) for p in partition) == g.node_count()
    names = sorted(c.name for c in cells)
    assert names == ["K_3", "K_3", "K_4"]
    # No inter-cell edges in this disjoint construction.
    assert len(inter_info.edges) == 0


def test_partitioner_rejects_pure_homogeneous(table):
    """A graph that decomposes homogeneously (3 × K_3) should not be
    accepted by the heterogeneous partitioner — homogeneous cases are
    handled by the existing homogeneous path."""
    g = _disjoint_blocks_with_bridges([3, 3, 3], cross_pairs=[])
    het = try_heterogeneous_partition(g, table)
    assert het is None  # All cells identical → defer to homogeneous path.


def test_partitioner_returns_none_when_no_cover(table):
    """Graph with no decomposition into rainbow-table cells (>= 3 nodes)
    should return None."""
    # 5 isolated edges = 10 nodes, each "cell" is just K_2 (2 nodes) which
    # is filtered out by min_cell_nodes=3.
    g = Graph.from_networkx(nx.disjoint_union_all([nx.path_graph(2)] * 5))
    het = try_heterogeneous_partition(g, table)
    assert het is None


def test_engine_synthesizes_heterogeneous_with_inter_edges(engine):
    """K_4 + K_3 + K_3 with inter-block edges. Synthesis must produce a
    polynomial whose Kirchhoff spanning-tree count matches the matrix-tree
    theorem."""
    g = _disjoint_blocks_with_bridges(
        [4, 3, 3],
        cross_pairs=[
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((0, 2), (2, 0)),
            ((0, 3), (2, 1)),
            ((1, 2), (2, 2)),
        ],
    )
    result = engine.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial), (
        f"Polynomial fails Kirchhoff: {result.polynomial}"
    )


def test_engine_heterogeneous_matches_direct_synthesis_path(engine, table):
    """Force the heterogeneous path by directly calling the engine's
    hierarchical handler, then compare its result to the engine's normal
    pipeline (which may pick a faster method like cut_vertex or
    treewidth_dp). Both should agree."""
    g = _disjoint_blocks_with_bridges(
        [4, 3, 3],
        cross_pairs=[
            ((0, 0), (1, 0)),
            ((0, 1), (2, 0)),
            ((1, 1), (2, 1)),
        ],
    )
    het = try_heterogeneous_partition(g, table)
    assert het is not None
    cells, partition, inter_info = het

    pipeline_poly = engine.synthesize(g).polynomial
    forced_poly = engine._synthesize_hierarchical(
        g, cells, partition, inter_info, max_depth=10,
    ).polynomial
    assert pipeline_poly == forced_poly, (
        f"heterogeneous vs pipeline mismatch:\n"
        f"  pipeline: {pipeline_poly}\n"
        f"  heterogeneous: {forced_poly}"
    )


def test_petersen_homogeneous_still_wins(engine, table):
    """Petersen has a homogeneous K_3 + K_3 + ... tiling? Actually Petersen
    has a 5-cell C_4 cover. This test verifies that *adding* the
    heterogeneous fallback doesn't perturb existing graphs that route
    through other paths."""
    g = Graph.from_networkx(nx.petersen_graph())
    result = engine.synthesize(g)
    assert verify_spanning_trees(g, result.polynomial)
    assert result.polynomial.num_spanning_trees() == 2000
