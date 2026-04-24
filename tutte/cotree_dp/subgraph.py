"""Subgraph signature tables for cotree DP.

Defines the signature types and provides the three core operations:
- Leaf: base case table for a single vertex
- Disjoint union combine: no edges between children
- Complete union combine (Algorithm 3.1): all edges between children

A signature is a sorted tuple of component sizes representing the
structure of a spanning subgraph. For example, a spanning subgraph
with components of sizes {3, 2, 1, 1} is represented as (3, 2, 1, 1)
— sorted in non-increasing order for canonical dict keys.

A double-signature tracks how components split across the two sides
of a complete union (⊗) operation. Each entry (f_size, g_size) records
how many F-side and G-side vertices have been absorbed into a merged
component.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class Signature(tuple):
    """Component-size multiset, always sorted in non-increasing order.

    Represents the sizes of connected components in a spanning subgraph.
    Example: components of sizes {3, 2, 1, 1} → Signature([3, 2, 1, 1])

    The constructor automatically sorts the input, so
    Signature([1, 3, 1, 2]) produces (3, 2, 1, 1).

    Immutable and hashable (inherits from tuple).
    """

    def __new__(cls, parts: Iterable[int] = ()) -> 'Signature':
        return super().__new__(cls, sorted(parts, reverse=True))

    def merge(self, other: 'Signature') -> 'Signature':
        """Multiset union of two signatures."""
        return Signature(list(self) + list(other))

    def to_double(self) -> 'DoubleSig':
        """Convert to initial double-signature: all parts on F-side."""
        return DoubleSig((part_size, 0) for part_size in self)


class DoubleSig(tuple):
    """Multiset of (f_size, g_size) pairs, sorted in non-increasing order.

    Tracks how components split across the F-side and G-side of a
    complete union (⊗) operation. Each pair records how many F-side
    and G-side vertices have been absorbed into a merged component.

    The constructor automatically sorts the input.
    Immutable and hashable (inherits from tuple).
    """

    def __new__(cls, pairs: Iterable[Tuple[int, int]] = ()) -> 'DoubleSig':
        return super().__new__(cls, sorted(pairs, reverse=True))

    def to_signature(self) -> Signature:
        """Convert back to a regular signature.

        Each (f_size, g_size) pair becomes a single component of
        size f_size + g_size.
        """
        return Signature(f_size + g_size for f_size, g_size in self)


# Maps (Signature, edge_count) → count of spanning subgraphs
# with that component structure and exactly that many edges.
SubgraphTable = Dict[Tuple[Signature, int], int]


# =============================================================================
# LEAF
# =============================================================================

def leaf_subgraph_table(vertex: int) -> SubgraphTable:
    """Subgraph signature table for a single vertex.

    A single vertex has one spanning subgraph: itself with 0 edges.
    """
    return {(Signature([1]), 0): 1}


# =============================================================================
# DISJOINT UNION COMBINE
# =============================================================================

def disjoint_union_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
) -> SubgraphTable:
    """Combine subgraph tables for disjoint union F | G.

    No edges between F and G — signatures concatenate, edge counts add.
    """
    result: SubgraphTable = defaultdict(int)
    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            merged_sig = sig_f.merge(sig_g)
            result[(merged_sig, edges_f + edges_g)] += count_f * count_g
    return dict(result)


# =============================================================================
# COMPLETE UNION COMBINE (ALGORITHM 3.1)
# =============================================================================

def complete_union_subgraph_combine(
    table_f: SubgraphTable,
    table_g: SubgraphTable,
) -> SubgraphTable:
    """Algorithm 3.1: Combine subgraph tables for complete union F * G.

    All possible edges exist between F and G. For each pair of F/G
    signatures, enumerate which cross-edges to include and how they
    merge components. Uses CellSel (Algorithm 3.2) for edge counting.

    Args:
        table_f, table_g: Subgraph tables of the two children.
        num_f_vertices: Number of vertices in F.
    """
    result: SubgraphTable = defaultdict(int)

    for (sig_f, edges_f), count_f in table_f.items():
        for (sig_g, edges_g), count_g in table_g.items():
            contributions = complete_union_subgraph_pair(sig_f, sig_g)
            for (merged_sig, extra_edges), contrib_count in contributions.items():
                result[(merged_sig, edges_f + edges_g + extra_edges)] += (
                    count_f * count_g * contrib_count
                )

    return dict(result)


def complete_union_subgraph_pair(
    sig_f: Signature,
    sig_g: Signature,
) -> Dict[Tuple[Signature, int], int]:
    """Core of Algorithm 3.1: complete union combine with edge counting.

    For each G-side component, enumerate which F-side components it merges
    with (via cross-edges), and use CellSel to count the number of ways
    to select the required edges.

    Args:
        sig_f: Signature of subgraph in F.
        sig_g: Signature of subgraph in G.

    Returns:
        Dict mapping (signature, extra_edges) to counts.
    """
    from .combinatorics import distinct_submultisets, multiset_diff, cellsel

    # State table: maps (double_sig, edges_accumulated) -> count
    init_double_sig = sig_f.to_double()
    state: Dict[Tuple[DoubleSig, int], int] = {(init_double_sig, 0): 1}

    for g_comp_size in sig_g:
        next_state: Dict[Tuple[DoubleSig, int], int] = defaultdict(int)

        for (beta, edges_accumulated), beta_count in state.items():
            for gamma, multi_coeff in distinct_submultisets(beta):
                f_total = sum(f_size for f_size, _g in gamma)
                g_total = sum(g_size for _f, g_size in gamma)

                beta_minus_gamma = multiset_diff(beta, gamma)
                merged_entry = (f_total, g_total + g_comp_size)
                beta_prime = DoubleSig(
                    list(beta_minus_gamma) + [merged_entry]
                )

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
        merged_sig = double_sig.to_signature()
        output[(merged_sig, total_edges)] += count

    return dict(output)
