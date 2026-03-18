"""Closed-form formulas and recurrences for known graph families.

Tier 1: Closed-form formulas — O(1) computation after detection.
Tier 2: Linear recurrences — O(n) polynomial multiplications after detection.

All base cases verified against SynthesisEngine + Kirchhoff T(1,1),
or derived from closed-form eigenvalue decompositions in [5] and [15].

References:
    [1] Tutte (1947) — foundational deletion-contraction, bridge/cut-vertex factorization
    [3] Brennan, Mansour, Mphako-Banda (2013) — wheel and fan formulas
    [4] Biggs, Damerell, Sands (1972) — recursive families, transfer matrices
    [5] Chang and Shrock (2004) — strip graph Tutte polynomials via transfer matrices
    [8] Shrock (2000) — Potts model partition functions on ladder graphs
    [15] Weisstein, MathWorld — gear graph closed-form formula
"""

from __future__ import annotations

from typing import Optional

from ..polynomial import TuttePolynomial, cycle_polynomial
from .constants import (
    BOOK_A, BOOK_B, BOOK_BASES,
    FAN_A, FAN_B, FAN_BASES,
    GEAR_A, GEAR_B, GEAR_C, GEAR_BASES,
    LADDER_A, LADDER_B, LADDER_BASES,
    MOBIUS_BASES,
    PRISM_BASES,
    WHEEL_A, WHEEL_B, WHEEL_C, WHEEL_BASES,
    apply_order6_recurrence,
)


# =============================================================================
# TIER 1 — CLOSED-FORM FORMULAS
# =============================================================================

def tree_formula(n: int) -> TuttePolynomial:
    """T(tree on n vertices) = x^(n-1).

    Every edge is a bridge. Cut vertex factorization gives the product.
    Complexity: O(1).

    Source: Tutte [1].
    """
    if n <= 1:
        return TuttePolynomial.one()
    return TuttePolynomial.x(n - 1)


def cycle_formula(n: int) -> TuttePolynomial:
    """T(C_n) = x^(n-1) + x^(n-2) + ... + x + y.

    Complexity: O(n) to build coefficient dict.

    Source: Folklore, derivable from deletion-contraction [1].
    """
    return cycle_polynomial(n)


def pan_formula(cycle_size: int) -> TuttePolynomial:
    """T(Pan_n) = x · T(C_n).

    Pan = cycle C_n with one pendant edge. Cut vertex factorization:
    bridge contributes x, cycle contributes T(C_n).

    Complexity: O(n) to build cycle polynomial.

    Source: Cut-vertex factorization [1].
    """
    return TuttePolynomial.x(1) * cycle_formula(cycle_size)


def sunlet_formula(k: int) -> TuttePolynomial:
    """T(Sunlet_k) = x^k · T(C_k).

    Sunlet = cycle C_k with pendant at each vertex. Each pendant
    contributes a bridge factor of x.

    Complexity: O(k) to build cycle polynomial.

    Source: Iterated cut-vertex factorization [1].
    """
    return TuttePolynomial.x(k) * cycle_formula(k)


def helm_formula(k: int) -> TuttePolynomial:
    """T(Helm_k) = x^k · T(W_k).

    Helm = wheel W_k with pendant at each rim vertex. Each pendant
    contributes a bridge factor of x.

    Complexity: O(k) polynomial multiplications (for wheel recurrence).

    Source: Wheel formula from [3] + bridge factorization [1].
    """
    return TuttePolynomial.x(k) * wheel_recurrence(k)


# =============================================================================
# TIER 1 — RECURRENCE-BASED (still O(k) after detection)
# =============================================================================

def wheel_recurrence(k: int) -> TuttePolynomial:
    """T(W_k) for wheel with k rim vertices (k+1 total vertices, 2k edges).

    Uses order-3 recurrence:
        T(W_n) = (x+y+2)·T(W_{n-1}) - (x+1)(y+1)·T(W_{n-2}) + xy·T(W_{n-3})

    Base cases verified against SynthesisEngine + Kirchhoff T(1,1).

    Complexity: O(k) polynomial multiplications.

    Source: Brennan, Mansour, Mphako-Banda [3].
    """
    if k < 3:
        raise ValueError(f"Wheel requires k >= 3 rim vertices, got {k}")

    # Base cases: W_3 (k=3), W_4 (k=4), W_5 (k=5)
    if k <= 5:
        return WHEEL_BASES[k - 3]

    prev3, prev2, prev1 = WHEEL_BASES
    for _ in range(6, k + 1):
        curr = WHEEL_A * prev1 - WHEEL_B * prev2 + WHEEL_C * prev3
        prev3, prev2, prev1 = prev2, prev1, curr

    return prev1


def fan_recurrence(k: int) -> TuttePolynomial:
    """T(F_k) for fan with k path vertices (k+1 total, 2k-1 edges).

    Uses order-2 recurrence:
        T(F_n) = (x+y+1)·T(F_{n-1}) - xy·T(F_{n-2})

    Base cases verified against SynthesisEngine + Kirchhoff T(1,1).

    Complexity: O(k) polynomial multiplications.

    Source: Brennan, Mansour, Mphako-Banda [3].
    """
    if k < 1:
        raise ValueError(f"Fan requires k >= 1, got {k}")

    # Base cases: F_1 (k=1), F_2 (k=2)
    if k <= 2:
        return FAN_BASES[k - 1]

    prev2, prev1 = FAN_BASES
    for _ in range(3, k + 1):
        curr = FAN_A * prev1 - FAN_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def ladder_recurrence(k: int) -> TuttePolynomial:
    """T(L_k) for ladder P_k × P_2 (2k vertices, 3k-2 edges).

    Uses order-2 recurrence:
        T(L_n) = (x² + x + y + 1)·T(L_{n-1}) - x²y·T(L_{n-2})

    Base cases verified against SynthesisEngine + Kirchhoff T(1,1).

    Complexity: O(k) polynomial multiplications.

    Source: Biggs, Damerell, Sands [4]; Chang and Shrock [5].
    """
    if k < 2:
        raise ValueError(f"Ladder requires k >= 2, got {k}")

    # Base cases: L_2 (k=2), L_3 (k=3)
    if k <= 3:
        return LADDER_BASES[k - 2]

    prev2, prev1 = LADDER_BASES
    for _ in range(4, k + 1):
        curr = LADDER_A * prev1 - LADDER_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def book_recurrence(k: int) -> TuttePolynomial:
    """T(Book_k) for k triangles sharing a common edge (k+2 vertices, 2k+1 edges).

    Uses order-2 recurrence:
        T(B_k) = (2x + y + 1)·T(B_{k-1}) - (x+1)(x+y)·T(B_{k-2})

    Base cases verified against SynthesisEngine + Kirchhoff T(1,1).

    Complexity: O(k) polynomial multiplications.

    Source: Folklore, recurrence derived via deletion-contraction.
    """
    if k < 1:
        raise ValueError(f"Book requires k >= 1, got {k}")

    # Base cases: B_1 (k=1), B_2 (k=2)
    if k <= 2:
        return BOOK_BASES[k - 1]

    prev2, prev1 = BOOK_BASES
    for _ in range(3, k + 1):
        curr = BOOK_A * prev1 - BOOK_B * prev2
        prev2, prev1 = prev1, curr

    return prev1


def gear_recurrence(k: int) -> TuttePolynomial:
    """T(Gear_k) for gear graph: wheel W_k with each rim edge subdivided.

    2k+1 vertices, 3k edges. Hub has degree k, k rim vertices have degree 3,
    k subdivision vertices have degree 2.

    Uses order-3 recurrence:
        T(G_n) = (x²+x+y+2)·T(G_{n-1}) - (x²y+x²+x+y+1)·T(G_{n-2}) + x²y·T(G_{n-3})

    Base cases derived from MathWorld closed-form eigenvalue formula [15].

    Complexity: O(k) polynomial multiplications.

    Source: Weisstein, MathWorld [15].
    """
    if k < 3:
        raise ValueError(f"Gear requires k >= 3 rim vertices, got {k}")

    # Base cases: G_3 (k=3), G_4 (k=4), G_5 (k=5)
    if k <= 5:
        return GEAR_BASES[k - 3]

    prev3, prev2, prev1 = GEAR_BASES
    for _ in range(6, k + 1):
        curr = GEAR_A * prev1 - GEAR_B * prev2 + GEAR_C * prev3
        prev3, prev2, prev1 = prev2, prev1, curr

    return prev1


def prism_recurrence(k: int) -> TuttePolynomial:
    """T(CL_k) for prism (circular ladder) C_k × K_2 (2k vertices, 3k edges).

    3-regular bipartite graph. Two parallel k-cycles connected by k rungs.

    Uses order-6 recurrence derived from transfer matrix eigenvalues:
        Characteristic polynomial (in Tutte variables):
        (z-1)(z-x)(z²-(x+y+2)z+xy)(z²-(x²+x+y+1)z+x²y)

    Base cases derived from Shrock [8] eigenvalue decomposition.

    Complexity: O(k) polynomial multiplications.

    Source: Shrock [8]; Chang and Shrock [5].
    """
    if k < 3:
        raise ValueError(f"Prism requires k >= 3, got {k}")

    # Base cases: CL_3 (k=3) through CL_8 (k=8)
    if k <= 8:
        return PRISM_BASES[k - 3]

    cl3, cl4, cl5, cl6, cl7, cl8 = PRISM_BASES
    return apply_order6_recurrence(cl3, cl4, cl5, cl6, cl7, cl8, 9, k)


def mobius_recurrence(k: int) -> TuttePolynomial:
    """T(M_k) for Möbius ladder (Möbius-Kantor graph generalization).

    2k vertices, 3k edges. 3-regular, non-bipartite.
    Same as prism but with one twisted rung (cross-connection).

    Uses the same order-6 characteristic polynomial as prism:
        (z-1)(z-x)(z²-(x+y+2)z+xy)(z²-(x²+x+y+1)z+x²y)

    Different base cases due to different boundary conditions.

    Base cases derived from Shrock [8] eigenvalue decomposition.

    Complexity: O(k) polynomial multiplications.

    Source: Shrock [8]; Chang and Shrock [5].
    """
    if k < 3:
        raise ValueError(f"Möbius ladder requires k >= 3, got {k}")

    # Base cases: M_3 (k=3) through M_8 (k=8)
    if k <= 8:
        return MOBIUS_BASES[k - 3]

    m3, m4, m5, m6, m7, m8 = MOBIUS_BASES
    return apply_order6_recurrence(m3, m4, m5, m6, m7, m8, 9, k)


# =============================================================================
# TIER 2 — GRID RECURRENCE
# =============================================================================

def grid_recurrence(m: int, n: int) -> Optional[TuttePolynomial]:
    """T(P_m × P_n) for grid graph with m rows, n columns.

    For m=1, this is a path. For m=2, this is the ladder.

    Complexity: O(n) polynomial multiplications for fixed small m.

    Source: Transfer matrix framework [4]; Chang and Shrock [5].
    """
    if m > n:
        m, n = n, m

    if m == 1:
        return tree_formula(n)

    if m == 2:
        return ladder_recurrence(n)

    # m >= 3: transfer matrix approach not yet implemented
    return None