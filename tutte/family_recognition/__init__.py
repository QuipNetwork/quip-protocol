"""Graph Family Recognition — O(n+m) heuristics for known graph families.

Recognizes trees, cycles, wheels, fans, ladders, pans, sunlets, books, helms,
and grids, returning their Tutte polynomials via closed-form formulas or linear
recurrences. Bypasses the expensive canonical_key computation (O(n² log n)).

Public API:
    recognize_family(graph) -> Optional[TuttePolynomial]
    compute_structural_fingerprint(graph) -> StructuralFingerprint
    StructuralFingerprint  (frozen dataclass)
"""

from __future__ import annotations

from typing import Optional

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .fingerprint import StructuralFingerprint, compute_structural_fingerprint
from .formulas import (
    book_recurrence,
    cycle_formula,
    fan_recurrence,
    gear_recurrence,
    grid_recurrence,
    helm_formula,
    ladder_recurrence,
    mobius_recurrence,
    pan_formula,
    prism_recurrence,
    sunlet_formula,
    tree_formula,
    wheel_recurrence,
)
from .verification import (
    detect_grid_dims,
    verify_book,
    verify_gear,
    verify_helm,
    verify_ladder,
    verify_mobius,
    verify_prism,
    verify_sunlet,
)

__all__ = [
    'recognize_family',
    'compute_structural_fingerprint',
    'StructuralFingerprint',
]


def recognize_family(graph: Graph) -> Optional[TuttePolynomial]:
    """Recognize known graph families and return Tutte polynomial.

    Runs a cascade of O(n+m) structural checks. Each check either identifies
    the family and returns its polynomial, or falls through to the next.

    Pipeline position: after base cases and cut vertex split, before
    series-parallel check and canonical_key computation.

    Complexity: O(n + m) — dominated by fingerprint computation (BFS for
    bipartiteness) and structural verification passes.

    Returns:
        TuttePolynomial if the family is recognized, None otherwise.
    """
    n = graph.node_count()
    m = graph.edge_count()

    # --- O(1) + O(n+m) checks based on (n, m) ---

    # Tree (covers paths, stars, all trees)
    # Must verify connectivity: a disconnected graph with m = n-1 is a forest,
    # not a tree. T(forest) ≠ x^{n-1}; the engine handles forests via the
    # disconnected-component split at step 5.
    if m == n - 1:
        if graph.is_connected():
            return tree_formula(n)

    # Compute fingerprint once — O(n+m)
    fp = compute_structural_fingerprint(graph)

    # --- O(1) checks against fingerprint ---

    # Cycle: connected, |E| = |V|, all degree 2
    if m == n and fp.degree_counts == {2: n}:
        return cycle_formula(n)

    # Wheel: one hub with degree n-1, all others degree 3, 2(n-1) edges
    # W_k has k+1 vertices. For n=4 (W_3=K_4), hub degree = 3 = rim degree.
    if n >= 4 and m == 2 * (n - 1):
        if n == 4:
            # W_3 = K_4: all vertices degree 3
            if fp.degree_counts == {3: 4}:
                return wheel_recurrence(3)
        elif (fp.degree_counts.get(n - 1, 0) == 1
                and fp.degree_counts.get(3, 0) == n - 1
                and len(fp.degree_counts) == 2):
            return wheel_recurrence(n - 1)

    # Fan: one apex with degree n-1, two degree-2 endpoints, rest degree 3
    # F_k has k+1 vertices, 2k-1 edges. k = n-1.
    if (n >= 4 and m == 2 * (n - 1) - 1
            and fp.degree_counts.get(n - 1, 0) == 1
            and fp.degree_counts.get(2, 0) == 2
            and fp.degree_counts.get(3, 0) == n - 3
            and len(fp.degree_counts) == 3):
        return fan_recurrence(n - 1)

    # Pan: one pendant (deg 1), one deg-3 vertex, rest deg-2, |E|=|V|
    if (n >= 4 and m == n
            and fp.degree_counts.get(1, 0) == 1
            and fp.degree_counts.get(3, 0) == 1
            and fp.degree_counts.get(2, 0) == n - 2):
        return pan_formula(n - 1)

    # --- O(1) + O(n+m) verification checks ---

    # Ladder: P_k × P_2 — 2k vertices, 3k-2 edges, 4 degree-2 corners
    if n >= 4 and n % 2 == 0:
        k = n // 2
        if (m == 3 * k - 2
                and fp.degree_counts.get(2, 0) == 4
                and fp.degree_counts.get(3, 0) == n - 4
                and fp.is_bipartite
                and len(fp.degree_counts) == 2):
            if verify_ladder(graph, k):
                return ladder_recurrence(k)

    # Helm: hub degree k, rim degree 4, pendants degree 1
    # 2k+1 vertices, 3k edges
    # Special case k=4: hub and rim both degree 4 → degree_counts = {4: k+1, 1: k}
    if n >= 7 and n % 2 == 1:
        k = (n - 1) // 2
        if m == 3 * k and fp.degree_counts.get(1, 0) == k:
            if k == 4:
                if fp.degree_counts.get(4, 0) == k + 1:
                    if verify_helm(graph, k):
                        return helm_formula(k)
            elif (fp.degree_counts.get(k, 0) == 1
                    and fp.degree_counts.get(4, 0) == k):
                if verify_helm(graph, k):
                    return helm_formula(k)

    # Gear: hub degree k, k rim vertices degree 3, k subdivision vertices degree 2
    # 2k+1 vertices, 3k edges (same as helm but different degree pattern)
    # Special case k=3: hub and rim both degree 3 → degree_counts = {3: k+1, 2: k}
    if n >= 7 and n % 2 == 1:
        k = (n - 1) // 2
        if m == 3 * k and fp.degree_counts.get(2, 0) == k:
            if k == 3:
                if fp.degree_counts.get(3, 0) == k + 1:
                    if verify_gear(graph, k):
                        return gear_recurrence(k)
            elif (fp.degree_counts.get(k, 0) == 1
                    and fp.degree_counts.get(3, 0) == k):
                if verify_gear(graph, k):
                    return gear_recurrence(k)

    # Book: k triangles sharing one edge — (k+2) vertices, (2k+1) edges
    if n >= 4:
        k = n - 2
        if k >= 1:
            hub_deg = k + 1
            if (m == 2 * k + 1
                    and fp.degree_counts.get(hub_deg, 0) == 2
                    and fp.degree_counts.get(2, 0) == k):
                if verify_book(graph, k):
                    return book_recurrence(k)

    # Sunlet: half degree 1, half degree 3, |E| = |V|
    if n >= 6 and n % 2 == 0 and m == n:
        k = n // 2
        if (fp.degree_counts.get(1, 0) == k
                and fp.degree_counts.get(3, 0) == k):
            if verify_sunlet(graph, k):
                return sunlet_formula(k)

    # Prism / Möbius: 3-regular, 2k vertices, 3k edges
    # Prism C_k×K_2 is bipartite iff k is even; Möbius M_k is bipartite iff k is odd.
    # Try both verifiers — bipartiteness alone cannot distinguish them.
    if (n >= 6 and n % 2 == 0
            and fp.is_regular and fp.regularity == 3
            and m == 3 * (n // 2)):
        k = n // 2
        if verify_prism(graph, k):
            return prism_recurrence(k)
        if verify_mobius(graph, k):
            return mobius_recurrence(k)

    # Grid: bipartite, specific degree pattern, m <= 5 rows
    if fp.is_bipartite and fp.degree_counts.get(2, 0) == 4:
        dims = detect_grid_dims(graph, fp)
        if dims is not None:
            m_dim, n_dim = dims
            result = grid_recurrence(m_dim, n_dim)
            if result is not None:
                return result

    return None  # Not recognized — fall through to expensive path