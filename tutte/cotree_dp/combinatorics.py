"""Combinatorial utilities for cotree DP.

Provides:
- CellSel (Algorithm 3.2): count cellular selections
- Submultiset enumeration with multinomial coefficients
- Multiset difference
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Dict, List, Tuple

from .signatures import DoubleSig


# =============================================================================
# CELLSEL (ALGORITHM 3.2) — WITH MEMOIZATION
# =============================================================================

# Cache: maps (sorted_cell_sizes_tuple, total_to_select) → result.
# Cell sizes are sorted for canonical keys — cellsel is order-independent
# (the DP result doesn't depend on the order of cells).
#
# Call clear_cellsel_cache() between unrelated graphs to bound memory.
_cellsel_cache: Dict[Tuple[Tuple[int, ...], int], int] = {}


def clear_cellsel_cache() -> None:
    """Clear the CellSel memoization cache.

    Call between unrelated graph computations to bound memory usage.
    The cache is also cleared automatically by compute_tutte_cotree_dp.
    """
    _cellsel_cache.clear()


def cellsel(cell_sizes: List[int], total_to_select: int) -> int:
    """Algorithm 3.2: Count cellular selections (memoized).

    Given num_cells pairwise disjoint cells of sizes d_1, d_2, ..., d_k,
    count the number of ways to select exactly total_to_select elements
    such that at least one element is selected from every cell.

    Results are cached by (sorted cell sizes, total_to_select). On K_14,
    this eliminates ~98% of redundant calls (158K duplicates out of 161K).

    Args:
        cell_sizes: List of cell sizes [d_1, d_2, ..., d_k].
        total_to_select: Total number of elements to select.

    Returns:
        Number of cellular selections.

    Complexity: O(num_cells × total_to_select²) on cache miss; O(1) on hit.
    """
    # Fast path for trivial cases (most common call — 3600x on K_14)
    if not cell_sizes:
        return 1 if total_to_select == 0 else 0

    num_cells = len(cell_sizes)
    if total_to_select < num_cells:
        return 0

    # Cache lookup with sorted key for order-independence
    cache_key = (tuple(sorted(cell_sizes)), total_to_select)
    cached = _cellsel_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _cellsel_compute(cell_sizes, num_cells, total_to_select)
    _cellsel_cache[cache_key] = result
    return result


def _cellsel_compute(
    cell_sizes: List[int],
    num_cells: int,
    total_to_select: int,
) -> int:
    """Core CellSel DP (uncached).

    Separated from cellsel() so the cache logic doesn't clutter the algorithm.
    """
    # DP: ways_prev[selected] = number of ways to select `selected` elements
    # from the first `cell_idx` cells with at least one from each cell.
    ways_prev = [0] * (total_to_select + 1)
    for selected in range(1, min(cell_sizes[0], total_to_select) + 1):
        ways_prev[selected] = comb(cell_sizes[0], selected)

    cumulative_size = cell_sizes[0]

    for cell_idx in range(1, num_cells):
        current_cell_size = cell_sizes[cell_idx]
        cumulative_size += current_cell_size
        ways_curr = [0] * (total_to_select + 1)
        for selected in range(cell_idx + 1, min(total_to_select, cumulative_size) + 1):
            for from_this_cell in range(1, min(selected - cell_idx, current_cell_size) + 1):
                ways_curr[selected] += (
                    ways_prev[selected - from_this_cell]
                    * comb(current_cell_size, from_this_cell)
                )
        ways_prev = ways_curr

    return ways_prev[total_to_select]


# =============================================================================
# SUBMULTISET ENUMERATION
# =============================================================================

def distinct_submultisets(
    multiset: DoubleSig,
) -> List[Tuple[DoubleSig, int]]:
    """Enumerate distinct submultisets of multiset (including empty).

    Returns list of (submultiset, multinomial_coefficient) pairs.
    The multinomial coefficient accounts for repeated elements:
    if an element appears `multiplicity` times in the multiset and we
    choose it `chosen` times, the coefficient includes C(multiplicity, chosen).

    Includes the empty submultiset (coefficient 1). For forest counting
    (Stage 1), the empty submultiset represents a G-side component that
    stays disconnected from all F-side components — correct for forests
    (which allow disconnected components).
    """
    element_counts = Counter(multiset)
    unique_elements = list(element_counts.keys())

    result: List[Tuple[DoubleSig, int]] = []
    _enum_submultisets(unique_elements, element_counts, 0, [], 1, result)
    return result


def _enum_submultisets(
    unique_elements: list,
    element_counts: dict,
    element_idx: int,
    current_selection: list,
    accumulated_coeff: int,
    result: list,
) -> None:
    """Recursive enumeration of distinct submultisets with coefficients."""
    if element_idx == len(unique_elements):
        submultiset = tuple(sorted(current_selection, reverse=True))
        result.append((submultiset, accumulated_coeff))
        return

    element = unique_elements[element_idx]
    multiplicity = element_counts[element]

    # Choose 0, 1, ..., multiplicity copies of this element
    for num_chosen in range(multiplicity + 1):
        new_coeff = accumulated_coeff * comb(multiplicity, num_chosen)
        new_selection = current_selection + [element] * num_chosen
        _enum_submultisets(
            unique_elements, element_counts,
            element_idx + 1, new_selection, new_coeff, result,
        )


# =============================================================================
# MULTISET DIFFERENCE
# =============================================================================

def multiset_diff(multiset: DoubleSig, to_remove: DoubleSig) -> DoubleSig:
    """Multiset difference: multiset minus to_remove."""
    remaining = list(multiset)
    for element in to_remove:
        remaining.remove(element)
    return tuple(sorted(remaining, reverse=True))
