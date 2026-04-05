"""Signature types and utilities for cotree DP.

A signature is a canonical representation of a component-size multiset
in a spanning subgraph. For example, a spanning subgraph with components
of sizes {3, 2, 1, 1} is represented as the tuple (3, 2, 1, 1) — sorted
in non-increasing order for canonical dict keys.

A double-signature tracks how components split across the two sides of a
⊗ (complete union) operation. Each entry (f_size, g_size) records how many
F-side and G-side vertices have been absorbed into a merged component.
"""

from __future__ import annotations

from typing import Dict, Tuple


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Component-size multiset in sorted non-increasing order.
# Example: {3, 2, 1, 1} → (3, 2, 1, 1)
Signature = Tuple[int, ...]

# Maps signature → count of spanning forests with that component structure.
ForestTable = Dict[Signature, int]

# Maps (signature, edge_count) → count of spanning subgraphs.
SubgraphTable = Dict[Tuple[Signature, int], int]

# Multiset of (f_size, g_size) pairs tracking how components split
# across the F-side and G-side of a ⊗ operation.
# Sorted in non-increasing order for canonical representation.
DoubleSig = Tuple[Tuple[int, int], ...]


# =============================================================================
# SIGNATURE UTILITIES
# =============================================================================

def merge_sigs(sig_left: Signature, sig_right: Signature) -> Signature:
    """Multiset union of two signatures."""
    return tuple(sorted(list(sig_left) + list(sig_right), reverse=True))


def sig_to_double(sig: Signature) -> DoubleSig:
    """Convert signature to initial double-signature: all parts on F-side."""
    return tuple(sorted(((part_size, 0) for part_size in sig), reverse=True))


def double_to_sig(double_sig: DoubleSig) -> Signature:
    """Convert double-signature back to regular signature.

    Each (f_size, g_size) pair becomes a single component of size f_size + g_size.
    """
    return tuple(sorted((f_size + g_size for f_size, g_size in double_sig), reverse=True))
