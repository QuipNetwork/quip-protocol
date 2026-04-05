"""Compute Tutte polynomial of a cograph via cotree DP.

Orchestrates the two-stage algorithm from Gimenez et al. (2006):
  Stage 1 (forest.py): Count spanning forests by component-size signature
  Stage 2 (subgraph.py): Count spanning subgraphs by signature + edge count

The Tutte polynomial is extracted from the Stage 2 table via the
rank-nullity formulation in the (x-1, y-1) basis.

Complexity: exp(O(vertices^{2/3})) — determined by the number of integer
partitions of vertices, which is exp(Theta(pi * sqrt(2 * vertices / 3)))
by the Hardy-Ramanujan asymptotic formula.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Dict, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .recognition import CotreeNode, build_cotree
from .signatures import SubgraphTable
from .subgraph import (
    leaf_subgraph_table,
    union_subgraph_combine,
    join_subgraph_combine,
)
from .combinatorics import clear_cellsel_cache


# =============================================================================
# COTREE TRAVERSAL
# =============================================================================

def _compute_subgraph_table(node: CotreeNode) -> SubgraphTable:
    """Compute subgraph signature table bottom-up on the cotree."""
    if node.node_type == 'leaf':
        assert node.vertex is not None, "Leaf node must have a vertex"
        return leaf_subgraph_table(node.vertex)

    child_tables = [_compute_subgraph_table(child) for child in node.children]

    if node.node_type == 'union':
        table = child_tables[0]
        for child_idx in range(1, len(child_tables)):
            table = union_subgraph_combine(table, child_tables[child_idx])
        return table

    elif node.node_type == 'join':
        table = child_tables[0]
        vertices_so_far = node.children[0].size()
        for child_idx in range(1, len(child_tables)):
            table = join_subgraph_combine(
                table, child_tables[child_idx], vertices_so_far,
            )
            vertices_so_far += node.children[child_idx].size()
        return table

    raise ValueError(f"Unknown node type: {node.node_type}")


# =============================================================================
# TUTTE POLYNOMIAL EXTRACTION
# =============================================================================

def _extract_tutte_polynomial(
    table: SubgraphTable,
    num_vertices: int,
    num_components: int,
) -> TuttePolynomial:
    """Extract T(G; x, y) from the subgraph signature table.

    T(G; x, y) = sum_{alpha, edge_count} S[alpha, edge_count]
                 * (x-1)^{rank_graph - rank_subgraph}
                 * (y-1)^{nullity_subgraph}

    where:
      alpha = component-size signature
      edge_count = number of edges in the spanning subgraph
      |alpha| = number of parts (= number of components in subgraph)
      rank_subgraph = num_vertices - |alpha|
      nullity_subgraph = edge_count - rank_subgraph
      rank_graph = num_vertices - num_components (rank of the full graph)
    """
    rank_graph = num_vertices - num_components

    # Accumulate in (a, b) = (x-1, y-1) basis
    ab_coeffs: Dict[Tuple[int, int], int] = defaultdict(int)

    for (sig, edge_count), count in table.items():
        num_parts = len(sig)
        rank_subgraph = num_vertices - num_parts
        nullity_subgraph = edge_count - rank_subgraph

        x_minus_1_power = rank_graph - rank_subgraph
        y_minus_1_power = nullity_subgraph

        if x_minus_1_power < 0 or y_minus_1_power < 0:
            continue

        ab_coeffs[(x_minus_1_power, y_minus_1_power)] += count

    # Convert from (x-1, y-1) basis to (x, y) basis via binomial expansion:
    # (x-1)^i * (y-1)^j = sum_x_pow C(i, x_pow) * (-1)^{i - x_pow} * x^x_pow
    #                    * sum_y_pow C(j, y_pow) * (-1)^{j - y_pow} * y^y_pow
    xy_coeffs: Dict[Tuple[int, int], int] = defaultdict(int)

    for (a_power, b_power), coeff in ab_coeffs.items():
        for x_power in range(a_power + 1):
            binom_x = comb(a_power, x_power) * ((-1) ** (a_power - x_power))
            for y_power in range(b_power + 1):
                binom_y = comb(b_power, y_power) * ((-1) ** (b_power - y_power))
                xy_coeffs[(x_power, y_power)] += coeff * binom_x * binom_y

    nonzero_coeffs = {key: val for key, val in xy_coeffs.items() if val != 0}
    return TuttePolynomial.from_coefficients(nonzero_coeffs)


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_tutte_cotree_dp(graph: Graph) -> TuttePolynomial:
    """Compute the Tutte polynomial of a cograph via cotree DP.

    Args:
        graph: A cograph (P4-free graph). Raises ValueError if not a cograph.

    Returns:
        The Tutte polynomial T(G; x, y).

    Raises:
        TypeError: if graph is a MultiGraph (cographs are simple graphs only).
        ValueError: if graph is not a cograph or has too many vertices.

    Complexity: exp(O(vertices^{2/3})) where vertices = |V|.
    """
    from ..graph import MultiGraph
    if isinstance(graph, MultiGraph):
        raise TypeError(
            "Cotree DP requires a simple Graph, not MultiGraph. "
            "Cographs are defined for simple graphs only (no parallel edges or loops)."
        )

    if graph.node_count() == 0:
        return TuttePolynomial.one()

    if graph.edge_count() == 0:
        return TuttePolynomial.one()

    if graph.edge_count() == 1:
        return TuttePolynomial.x()

    # Guard: signature table size grows as exp(pi * sqrt(2 * vertices / 3))
    # by Hardy-Ramanujan.
    if graph.node_count() > 35:
        raise ValueError(
            f"Graph has {graph.node_count()} vertices; cotree DP signature "
            f"tables grow as exp(O(vertices^{{2/3}})). vertices > 35 is likely too slow."
        )

    cotree = build_cotree(graph)
    if cotree is None:
        raise ValueError(
            "Graph is not a cograph (contains induced P4). "
            "Cotree DP only works on P4-free graphs."
        )

    num_vertices = graph.node_count()

    components = graph.connected_components()
    num_components = len(components)

    table = _compute_subgraph_table(cotree)

    # Clear CellSel cache after each graph to bound memory
    clear_cellsel_cache()

    return _extract_tutte_polynomial(table, num_vertices, num_components)
