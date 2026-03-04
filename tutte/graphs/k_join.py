"""K-Join Operations for Tutte Polynomial Synthesis.

This module implements k-join operations, which are the inverse of k-sum (k-clique sum)
operations. These allow us to compute Tutte polynomials by building graphs from
known minors using algebraic composition.

K-Sum Formulas (combining graphs that share a k-clique):
- 0-sum (disjoint union): T(G1 ∪ G2) = T(G1) × T(G2)
- 1-sum (cut vertex):     T(G1 ·₁ G2) = T(G1) × T(G2)
- 2-sum (edge clique):    T(G1 ⊕₂ G2) = (T(G1) × T(G2)) / T(K₂)
- k-sum (k-clique):       T(G1 ⊕ₖ G2) = (T(G1) × T(G2)) / T(Kₖ)

K-Join Formulas (inverse - separating a graph at a k-clique):
- 0-join: T(G1) = T(G1 ∪ G2) / T(G2)
- 1-join: T(G1) = T(G1 ·₁ G2) / T(G2)
- 2-join: T(G1) = T(G1 ⊕₂ G2) × T(K₂) / T(G2)
- k-join: T(G1) = T(G1 ⊕ₖ G2) × T(Kₖ) / T(G2)

When building a cover from tiles:
- Tiles joined at k vertices contribute: T(tile1) × T(tile2) / T(Kₖ)
- The shared k-clique's polynomial is divided out to avoid double-counting
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass

from ..polynomial import TuttePolynomial


# =============================================================================
# KNOWN COMPLETE GRAPH POLYNOMIALS
# =============================================================================

# T(Kₙ) - Tutte polynomials of complete graphs
# These are used for k-join calculations
_K_POLYNOMIALS: Dict[int, TuttePolynomial] = {}


def tutte_k(n: int) -> TuttePolynomial:
    """Get Tutte polynomial of complete graph Kₙ.

    Known values:
    - T(K₀) = 1 (empty graph)
    - T(K₁) = 1 (single vertex, no edges)
    - T(K₂) = x (single edge)
    - T(K₃) = x² + x + y
    - T(K₄) = x³ + 3x² + 4xy + 2x + 2y + 3y² + y³
    """
    if n in _K_POLYNOMIALS:
        return _K_POLYNOMIALS[n]

    if n <= 0:
        poly = TuttePolynomial.one()
    elif n == 1:
        poly = TuttePolynomial.one()
    elif n == 2:
        poly = TuttePolynomial.x()
    elif n == 3:
        poly = TuttePolynomial.from_coefficients({
            (2, 0): 1, (1, 0): 1, (0, 1): 1
        })
    elif n == 4:
        poly = TuttePolynomial.from_coefficients({
            (3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2,
            (0, 1): 2, (0, 2): 3, (0, 3): 1
        })
    elif n == 5:
        poly = TuttePolynomial.from_coefficients({
            (4, 0): 1, (3, 0): 6, (2, 1): 10, (2, 0): 11,
            (1, 2): 15, (1, 1): 20, (1, 3): 5, (1, 0): 6,
            (0, 1): 6, (0, 2): 15, (0, 3): 15, (0, 4): 10,
            (0, 5): 4, (0, 6): 1
        })
    else:
        # For larger n, we would need to compute or look up
        # For now, raise an error
        raise ValueError(f"T(K_{n}) not pre-computed. Add to lookup table.")

    _K_POLYNOMIALS[n] = poly
    return poly


# =============================================================================
# K-SUM OPERATIONS (Combining graphs)
# =============================================================================

def k_sum(t1: TuttePolynomial, t2: TuttePolynomial, k: int) -> TuttePolynomial:
    """Compute Tutte polynomial of k-sum of two graphs.

    The k-sum G1 ⊕ₖ G2 is formed by identifying a k-clique in G1 with
    a k-clique in G2.

    Formula: T(G1 ⊕ₖ G2) = T(G1) × T(G2) / T(Kₖ)

    Special cases:
    - k=0: Disjoint union, T = T(G1) × T(G2)
    - k=1: Cut vertex join, T = T(G1) × T(G2)
    - k≥2: Divide by T(Kₖ)

    Args:
        t1: Tutte polynomial of first graph
        t2: Tutte polynomial of second graph
        k: Size of shared clique (number of vertices)

    Returns:
        Tutte polynomial of k-sum
    """
    product = t1 * t2

    if k <= 1:
        # 0-sum (disjoint) or 1-sum (cut vertex): just multiply
        return product

    # k-sum for k >= 2: divide by T(Kₖ)
    t_k = tutte_k(k)
    return polynomial_divide(product, t_k)


def k_sum_multiple(polynomials: List[TuttePolynomial],
                   join_sizes: List[int]) -> TuttePolynomial:
    """Compute k-sum of multiple graphs with specified join sizes.

    Args:
        polynomials: List of Tutte polynomials [T(G1), T(G2), ...]
        join_sizes: List of k values for joins between consecutive graphs
                   [k₁₂, k₂₃, ...] where kᵢⱼ is join size between Gᵢ and Gⱼ

    Returns:
        Tutte polynomial of the k-sum
    """
    if not polynomials:
        return TuttePolynomial.one()

    if len(polynomials) == 1:
        return polynomials[0]

    if len(join_sizes) != len(polynomials) - 1:
        raise ValueError("join_sizes must have length len(polynomials) - 1")

    result = polynomials[0]
    for i, (poly, k) in enumerate(zip(polynomials[1:], join_sizes)):
        result = k_sum(result, poly, k)

    return result


# =============================================================================
# K-JOIN OPERATIONS (Separating/dividing graphs)
# =============================================================================

def k_join_divide(t_combined: TuttePolynomial,
                  t_part: TuttePolynomial,
                  k: int) -> TuttePolynomial:
    """Compute Tutte polynomial after removing a k-joined part.

    If G = G1 ⊕ₖ G2 (k-sum), then:
    T(G1) = T(G) × T(Kₖ) / T(G2)

    This is the k-join division: removing G2 from the k-sum.

    Args:
        t_combined: Tutte polynomial of combined graph G1 ⊕ₖ G2
        t_part: Tutte polynomial of part to remove (G2)
        k: Size of shared clique

    Returns:
        Tutte polynomial of remaining part (G1)
    """
    if k <= 1:
        # 0-join or 1-join: just divide
        return polynomial_divide(t_combined, t_part)

    # k-join for k >= 2: multiply by T(Kₖ) then divide by T(G2)
    t_k = tutte_k(k)
    numerator = t_combined * t_k
    return polynomial_divide(numerator, t_part)


# =============================================================================
# POLYNOMIAL DIVISION
# =============================================================================

def polynomial_divmod(
    numerator: TuttePolynomial,
    denominator: TuttePolynomial
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Divide two Tutte polynomials with remainder.

    Performs polynomial long division and returns both quotient and remainder
    such that: numerator = quotient * denominator + remainder

    Args:
        numerator: Polynomial to divide
        denominator: Polynomial to divide by

    Returns:
        Tuple of (quotient, remainder) polynomials

    Raises:
        ValueError: If denominator is zero
    """
    # Get coefficient dictionaries
    num_coeffs = numerator.to_coefficients()
    den_coeffs = denominator.to_coefficients()

    if not den_coeffs:
        raise ValueError("Cannot divide by zero polynomial")

    if not num_coeffs:
        # 0 / anything = 0 remainder 0
        return TuttePolynomial.zero(), TuttePolynomial.zero()

    # Find leading term of denominator (highest total degree, then highest x)
    den_leading = max(den_coeffs.keys(), key=lambda t: (t[0] + t[1], t[0]))
    den_leading_coeff = den_coeffs[den_leading]

    # Perform polynomial long division
    result_coeffs: Dict[Tuple[int, int], int] = {}
    remainder = dict(num_coeffs)

    while remainder:
        # Find leading term of remainder
        rem_leading = max(remainder.keys(), key=lambda t: (t[0] + t[1], t[0]))
        rem_leading_coeff = remainder[rem_leading]

        # Check if we can divide (leading term of remainder must be divisible)
        quot_exp = (rem_leading[0] - den_leading[0], rem_leading[1] - den_leading[1])

        if quot_exp[0] < 0 or quot_exp[1] < 0:
            # Can't divide further - remainder is what's left
            break

        # Check if coefficients divide evenly (for integer polynomials)
        if rem_leading_coeff % den_leading_coeff != 0:
            # Can't divide evenly - stop here, remainder is what's left
            break

        quot_coeff = rem_leading_coeff // den_leading_coeff
        result_coeffs[quot_exp] = result_coeffs.get(quot_exp, 0) + quot_coeff

        # Subtract quotient_term * denominator from remainder
        for (dx, dy), dc in den_coeffs.items():
            rx, ry = quot_exp[0] + dx, quot_exp[1] + dy
            remainder[(rx, ry)] = remainder.get((rx, ry), 0) - quot_coeff * dc
            if remainder[(rx, ry)] == 0:
                del remainder[(rx, ry)]

    # Clean up zero coefficients
    result_coeffs = {k: v for k, v in result_coeffs.items() if v != 0}
    remainder = {k: v for k, v in remainder.items() if v != 0}

    # Build result polynomials
    if not result_coeffs:
        quotient = TuttePolynomial.zero()
    else:
        quotient = TuttePolynomial.from_coefficients(result_coeffs)

    if not remainder:
        remainder_poly = TuttePolynomial.zero()
    else:
        remainder_poly = TuttePolynomial.from_coefficients(remainder)

    return quotient, remainder_poly


def polynomial_divide(numerator: TuttePolynomial,
                      denominator: TuttePolynomial) -> TuttePolynomial:
    """Divide two Tutte polynomials (exact division).

    This performs exact polynomial division. If the division is not exact
    (has remainder), raises an error.

    For Tutte polynomial synthesis, division should always be exact when
    the algorithm is correct.

    Args:
        numerator: Polynomial to divide
        denominator: Polynomial to divide by

    Returns:
        Quotient polynomial

    Raises:
        ValueError: If division is not exact
    """
    quotient, remainder = polynomial_divmod(numerator, denominator)

    if not remainder.is_zero():
        raise ValueError(
            f"Polynomial division has non-zero remainder. "
            f"Numerator: {numerator}, Denominator: {denominator}, "
            f"Remainder: {remainder}"
        )

    # Handle edge case: 0 / x = 0, but we may want to return 1 for compatibility
    if quotient.is_zero() and not numerator.is_zero():
        # This shouldn't happen for exact division
        raise ValueError(
            f"Unexpected zero quotient for non-zero numerator. "
            f"Numerator: {numerator}, Denominator: {denominator}"
        )

    return quotient


# =============================================================================
# COVER POLYNOMIAL COMPUTATION
# =============================================================================

@dataclass
class TileJoin:
    """Represents a k-join between two tiles in a cover."""
    tile1_idx: int  # Index of first tile
    tile2_idx: int  # Index of second tile
    k: int          # Size of shared clique (number of shared vertices)
    shared_vertices: Set[int]  # The actual shared vertices


def compute_cover_polynomial(
    tile_polynomials: List[TuttePolynomial],
    joins: List[TileJoin]
) -> TuttePolynomial:
    """Compute Tutte polynomial of a cover from tile polynomials and joins.

    The cover is built by k-summing tiles at their shared cliques.

    Algorithm:
    1. Start with product of all tile polynomials
    2. For each k-join with k >= 2, divide by T(Kₖ) to account for
       the shared clique being counted twice

    Args:
        tile_polynomials: Tutte polynomials of each tile
        joins: List of k-joins between tiles

    Returns:
        Tutte polynomial of the cover
    """
    if not tile_polynomials:
        return TuttePolynomial.one()

    # Start with product of all tiles
    result = TuttePolynomial.one()
    for poly in tile_polynomials:
        result = result * poly

    # Divide by T(Kₖ) for each k-join with k >= 2
    for join in joins:
        if join.k >= 2:
            t_k = tutte_k(join.k)
            result = polynomial_divide(result, t_k)

    return result


# =============================================================================
# FRINGE OPERATIONS
# =============================================================================

def divide_out_fringe(
    t_cover: TuttePolynomial,
    t_fringe: TuttePolynomial,
    fringe_join_k: int = 0
) -> TuttePolynomial:
    """Divide out the fringe polynomial from a cover polynomial.

    If cover = input ∪ fringe (via k-join), then:
    T(input) = T(cover) × T(Kₖ) / T(fringe)  for k >= 2
    T(input) = T(cover) / T(fringe)          for k <= 1

    Args:
        t_cover: Tutte polynomial of the cover
        t_fringe: Tutte polynomial of the fringe (excess)
        fringe_join_k: How fringe is joined to input (usually 0 for disjoint
                      edges, or higher if fringe shares vertices with input)

    Returns:
        Tutte polynomial of input (cover minus fringe)
    """
    return k_join_divide(t_cover, t_fringe, fringe_join_k)
