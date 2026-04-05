"""Stage 2: Subgraph signature table computation (Algorithms 3.1 and 3.2).

Extends Stage 1 (forest counting) with edge counting. Each entry in the
subgraph table maps (signature, edge_count) to the number of spanning
subgraphs with that component structure and exactly that many edges.

This is the table from which the Tutte polynomial is extracted.

Two combine operations:
- Union: disjoint union, edge counts add
- Join (Algorithm 3.1): complete union with CellSel edge counting
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

from .signatures import (
    Signature, SubgraphTable, DoubleSig,
    merge_sigs, sig_to_double, double_to_sig,
)
from .combinatorics import distinct_submultisets, multiset_diff, cellsel


# =============================================================================
# LEAF
# =============================================================================

def leaf_subgraph_table(vertex: int) -> SubgraphTable:
    """Subgraph signature table for a single vertex.

    A single vertex has one spanning subgraph: itself with 0 edges.
    """
    return {((1,), 0): 1}


# =============================================================================
# UNION COMBINE
# =============================================================================

def union_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
) -> SubgraphTable:
    """Combine subgraph tables for disjoint union F | G.

    Same as Algorithm 2.4 but with edge counts: (sig, edges_f + edges_g).
    """
    result: SubgraphTable = defaultdict(int)
    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            merged_sig = merge_sigs(sig_f, sig_g)
            result[(merged_sig, edges_f + edges_g)] += count_f * count_g
    return dict(result)


# =============================================================================
# JOIN COMBINE (ALGORITHM 3.1)
# =============================================================================

def join_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
    num_f_vertices: int,
) -> SubgraphTable:
    """Algorithm 3.1: Combine subgraph tables for complete union F * G.

    Extends Algorithm 2.6 with edge counting and CellSel procedure.

    Args:
        table_f, table_g: Subgraph tables of the two children.
        num_f_vertices: Number of vertices in F.
    """
    result: SubgraphTable = defaultdict(int)

    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            contributions = join_subgraph_pair(sig_f, sig_g, num_f_vertices)
            for (merged_sig, extra_edges), contrib_count in contributions.items():
                result[(merged_sig, edges_f + edges_g + extra_edges)] += (
                    count_f * count_g * contrib_count
                )

    return dict(result)


def join_subgraph_pair(
    sig_f: Signature,
    sig_g: Signature,
    num_f_vertices: int,  # noqa: ARG — kept for API consistency with Algorithm 3.1
) -> Dict[Tuple[Signature, int], int]:
    """Core of Algorithm 3.1: join combine with edge counting.

    Like join_forest_pair but tracks edges added during the join.
    Uses CellSel (Algorithm 3.2) to count edge possibilities.

    Args:
        sig_f: Signature of subgraph in F.
        sig_g: Signature of subgraph in G.
        num_f_vertices: Number of vertices in F (= sum of parts in sig_f).

    Returns:
        Dict mapping (signature, extra_edges) to counts.
    """
    # State table: maps (double_sig, edges_accumulated) -> count
    init_double_sig = sig_to_double(sig_f)
    state: Dict[Tuple[DoubleSig, int], int] = {(init_double_sig, 0): 1}

    for g_comp_size in sig_g:
        next_state: Dict[Tuple[DoubleSig, int], int] = defaultdict(int)

        for (beta, edges_accumulated), beta_count in state.items():
            for gamma, multi_coeff in distinct_submultisets(beta):
                f_total = sum(f_size for f_size, _g in gamma)
                g_total = sum(g_size for _f, g_size in gamma)

                beta_minus_gamma = multiset_diff(beta, gamma)
                merged_entry = (f_total, g_total + g_comp_size)
                beta_prime = tuple(sorted(
                    list(beta_minus_gamma) + [merged_entry], reverse=True
                ))

                # Cell sizes: for each selected component with f_size F-side
                # vertices, there are g_comp_size × f_size edges in
                # K_{g_comp_size, f_size}. CellSel counts ways to pick exactly
                # num_edges_to_add edges total with at least one from each cell.
                cell_sizes = [g_comp_size * f_size for f_size, _g in gamma]
                max_possible_edges = g_comp_size * f_total

                for num_edges_to_add in range(len(gamma), max_possible_edges + 1):
                    num_selections = cellsel(cell_sizes, num_edges_to_add)

                    if num_selections > 0:
                        next_state[(beta_prime, edges_accumulated + num_edges_to_add)] += (
                            beta_count * multi_coeff * num_selections
                        )

        state = dict(next_state)

    # Convert double-signatures to regular signatures
    output: Dict[Tuple[Signature, int], int] = defaultdict(int)
    for (double_sig, total_edges), count in state.items():
        merged_sig = double_to_sig(double_sig)
        output[(merged_sig, total_edges)] += count

    return dict(output)
