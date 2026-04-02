"""Parallel Connection via Bonin-de Mier Theorem 6.

This module implements the Bonin-de Mier formula for computing Tutte polynomials
of generalized parallel connections P_N(M1, M2). Instead of edge-by-edge chord
processing, the formula evaluates over the lattice of flats of the shared matroid N.

Key components:
1. BivariateLaurentPoly - Polynomial in u=x-1, v=y-1 with negative v-exponents
2. theorem6_parallel_connection - Main formula implementation
3. precompute_contractions - Pre-compute T(M_i/Z) for all flats Z of N
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

from ..polynomial import TuttePolynomial
from ..graph import Graph, MultiGraph
from .core import (
    GraphicMatroid, FlatLattice, Edge,
    enumerate_flats_with_hasse,
)

from ..logs import get_log, EventType, LogLevel

if TYPE_CHECKING:
    from ..synthesis.engine import SynthesisEngine
    from ..graphs.series_parallel import SPNode


# =============================================================================
# BIVARIATE LAURENT POLYNOMIAL
# =============================================================================

class BivariateLaurentPoly:
    """Polynomial in u=x-1, v=y-1 with possibly negative v-exponents.

    Theorem 6 works in the rank-generating basis u = x-1, v = y-1.
    Division by (y-1)^r = v^r is a v-exponent shift, but intermediate
    terms can have negative v-powers.

    Coefficients stored as {(u_pow, v_pow): int} where v_pow can be negative.
    """

    __slots__ = ('coeffs',)

    def __init__(self, coeffs: Dict[Tuple[int, int], int]):
        # Remove zero entries
        self.coeffs = {k: v for k, v in coeffs.items() if v != 0}

    @classmethod
    def zero(cls) -> 'BivariateLaurentPoly':
        return cls({})

    @classmethod
    def one(cls) -> 'BivariateLaurentPoly':
        return cls({(0, 0): 1})

    @classmethod
    def from_tutte(cls, poly: TuttePolynomial) -> 'BivariateLaurentPoly':
        """Convert T(x,y) to R(u,v) where x=u+1, y=v+1.

        Expand each x^i * y^j = (u+1)^i * (v+1)^j using binomial theorem.
        """
        result: Dict[Tuple[int, int], int] = defaultdict(int)

        for (i, j), c in poly.to_coefficients().items():
            # (u+1)^i = sum_{a=0}^{i} C(i,a) u^a
            # (v+1)^j = sum_{b=0}^{j} C(j,b) v^b
            for a in range(i + 1):
                binom_a = comb(i, a)
                for b in range(j + 1):
                    binom_b = comb(j, b)
                    result[(a, b)] += c * binom_a * binom_b

        return cls(dict(result))

    def is_zero(self) -> bool:
        return not self.coeffs

    def __add__(self, other: 'BivariateLaurentPoly') -> 'BivariateLaurentPoly':
        result = defaultdict(int, self.coeffs)
        for k, v in other.coeffs.items():
            result[k] += v
        return BivariateLaurentPoly(dict(result))

    def __sub__(self, other: 'BivariateLaurentPoly') -> 'BivariateLaurentPoly':
        result = defaultdict(int, self.coeffs)
        for k, v in other.coeffs.items():
            result[k] -= v
        return BivariateLaurentPoly(dict(result))

    def __mul__(self, other: 'BivariateLaurentPoly') -> 'BivariateLaurentPoly':
        result: Dict[Tuple[int, int], int] = defaultdict(int)
        for (a1, b1), c1 in self.coeffs.items():
            for (a2, b2), c2 in other.coeffs.items():
                result[(a1 + a2, b1 + b2)] += c1 * c2
        return BivariateLaurentPoly(dict(result))

    def __rmul__(self, scalar: int) -> 'BivariateLaurentPoly':
        if not isinstance(scalar, int):
            return NotImplemented
        return BivariateLaurentPoly({k: scalar * v for k, v in self.coeffs.items()})

    def __neg__(self) -> 'BivariateLaurentPoly':
        return BivariateLaurentPoly({k: -v for k, v in self.coeffs.items()})

    def eval_at(self, u: int, v: int) -> 'Fraction':
        """Evaluate polynomial at integer (u, v). Returns exact Fraction.

        For negative v-powers, uses common-denominator integer arithmetic
        to minimize Fraction overhead.
        """
        from fractions import Fraction
        if not self.coeffs:
            return Fraction(0)

        # Find minimum v-power to determine common denominator
        min_b = min(b for (_, b) in self.coeffs)

        if min_b >= 0:
            # All non-negative v-powers: result is integer
            result = 0
            for (a, b), c in self.coeffs.items():
                result += c * (u ** a if a else 1) * (v ** b if b else 1)
            return Fraction(result)

        # Has negative v-powers: use v^(-min_b) as common denominator
        # Multiply everything by v^(-min_b), then divide at the end
        shift = -min_b
        numer = 0
        for (a, b), c in self.coeffs.items():
            # c * u^a * v^b * v^shift = c * u^a * v^(b + shift)
            exp = b + shift
            numer += c * (u ** a if a else 1) * (v ** exp if exp else 1)
        denom = v ** shift
        return Fraction(numer, denom)

    def shift_v(self, k: int) -> 'BivariateLaurentPoly':
        """Multiply by v^k (negative k = divide by v^|k|)."""
        if k == 0:
            return self
        return BivariateLaurentPoly({
            (a, b + k): c for (a, b), c in self.coeffs.items()
        })

    def min_v_power(self) -> int:
        """Minimum v-power in the polynomial."""
        if not self.coeffs:
            return 0
        return min(b for (_, b) in self.coeffs.keys())

    def max_v_power(self) -> int:
        """Maximum v-power in the polynomial."""
        if not self.coeffs:
            return 0
        return max(b for (_, b) in self.coeffs.keys())

    def to_tutte_poly(self) -> TuttePolynomial:
        """Convert R(u,v) back to T(x,y) where u=x-1, v=y-1.

        Substitutes u = x-1, v = y-1:
        u^a * v^b = (x-1)^a * (y-1)^b
                   = sum_{i} C(a,i)(-1)^{a-i} x^i * sum_{j} C(b,j)(-1)^{b-j} y^j

        Asserts no negative powers remain.
        """
        min_v = self.min_v_power()
        if min_v < 0:
            raise ValueError(
                f"Cannot convert to TuttePolynomial: "
                f"negative v-power {min_v} found. "
                f"The formula should produce non-negative powers after cancellation."
            )

        result: Dict[Tuple[int, int], int] = defaultdict(int)

        for (a, b), c in self.coeffs.items():
            # (x-1)^a = sum_{i=0}^{a} C(a,i) (-1)^{a-i} x^i
            # (y-1)^b = sum_{j=0}^{b} C(b,j) (-1)^{b-j} y^j
            for i in range(a + 1):
                coeff_x = comb(a, i) * ((-1) ** (a - i))
                for j in range(b + 1):
                    coeff_y = comb(b, j) * ((-1) ** (b - j))
                    result[(i, j)] += c * coeff_x * coeff_y

        # Filter zeros
        coeffs = {k: v for k, v in result.items() if v != 0}
        return TuttePolynomial.from_coefficients(coeffs)

    def divmod(self, other: 'BivariateLaurentPoly') -> Tuple['BivariateLaurentPoly', 'BivariateLaurentPoly']:
        """Polynomial long division in Z[u, v, v^{-1}].

        Uses graded lexicographic order: (u_pow + v_pow, u_pow) as the
        monomial ordering. Returns (quotient, remainder).
        """
        if not other.coeffs:
            raise ZeroDivisionError("Division by zero polynomial")
        if not self.coeffs:
            return BivariateLaurentPoly.zero(), BivariateLaurentPoly.zero()

        def mono_key(uv: Tuple[int, int]) -> Tuple[int, int]:
            return (uv[0] + uv[1], uv[0])

        # Leading term of divisor
        den_leading = max(other.coeffs.keys(), key=mono_key)
        den_lc = other.coeffs[den_leading]

        quotient: Dict[Tuple[int, int], int] = {}
        remainder = dict(self.coeffs)

        while remainder:
            # Leading term of remainder
            rem_leading = max(remainder.keys(), key=mono_key)
            rem_lc = remainder[rem_leading]

            # Quotient monomial exponent
            q_exp = (rem_leading[0] - den_leading[0], rem_leading[1] - den_leading[1])

            # For u-powers, we need non-negative exponents
            if q_exp[0] < 0:
                break

            # Check integer divisibility
            if rem_lc % den_lc != 0:
                break

            q_coeff = rem_lc // den_lc
            quotient[q_exp] = quotient.get(q_exp, 0) + q_coeff

            # Subtract q_coeff * u^q_exp * other from remainder
            for (du, dv), dc in other.coeffs.items():
                key = (du + q_exp[0], dv + q_exp[1])
                remainder[key] = remainder.get(key, 0) - q_coeff * dc
                if remainder[key] == 0:
                    del remainder[key]

        return BivariateLaurentPoly(quotient), BivariateLaurentPoly(remainder)

    def __floordiv__(self, other: 'BivariateLaurentPoly') -> 'BivariateLaurentPoly':
        """Exact division (asserts remainder is zero)."""
        quotient, remainder = self.divmod(other)
        if not remainder.is_zero():
            raise ValueError(
                f"Non-exact division: remainder = {remainder}"
            )
        return quotient

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BivariateLaurentPoly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __repr__(self) -> str:
        if not self.coeffs:
            return "0"
        terms = []
        for (a, b), c in sorted(self.coeffs.items(), reverse=True):
            term = str(c)
            if a > 0:
                term += f"*u^{a}" if a > 1 else "*u"
            if b > 0:
                term += f"*v^{b}" if b > 1 else "*v"
            elif b < 0:
                term += f"*v^({b})"
            terms.append(term)
        return " + ".join(terms).replace("+ -", "- ")


# =============================================================================
# HELPER FUNCTIONS FOR THEOREM 6
# =============================================================================

def _y_power_in_uv(k: int) -> BivariateLaurentPoly:
    """y^k = (v+1)^k = sum C(k,j) v^j."""
    coeffs: Dict[Tuple[int, int], int] = {}
    for j in range(k + 1):
        coeffs[(0, j)] = comb(k, j)
    return BivariateLaurentPoly(coeffs)


def _chi_in_uv(chi_coeffs: Dict[int, int]) -> BivariateLaurentPoly:
    """chi(q) evaluated at q = u*v: substitute q = u*v into polynomial.

    chi(q) = sum_k c_k * q^k -> sum_k c_k * (uv)^k = sum_k c_k * u^k * v^k
    """
    coeffs: Dict[Tuple[int, int], int] = {}
    for power, c in chi_coeffs.items():
        coeffs[(power, power)] = c
    return BivariateLaurentPoly(coeffs)


# =============================================================================
# THEOREM 6: PARALLEL CONNECTION
# =============================================================================

def theorem6_parallel_connection(
    lattice_N: FlatLattice,
    T_M1_contracted: Dict[int, BivariateLaurentPoly],
    T_M2_contracted: Dict[int, BivariateLaurentPoly],
    r_N: int,
) -> TuttePolynomial:
    """Compute T(P_N(M1, M2)) via Bonin-de Mier Theorem 6.

    The corrected formula in rank-generating basis R(u,v) where u=x-1, v=y-1:

    R(P_N(M1,M2); u,v) = v^{r(N)} * sum_{W flat of N}
        [1 / ((v+1)^{|W|} * chi(N/W; uv))]
        * g_1(W) * g_2(W)

    where:
        g_i(W) = sum_{Z >= W} mu(W,Z) * (v+1)^{|Z|} * v^{-r(Z)} * R(M_i/Z; u,v)

    Uses common-denominator accumulation to handle the 1/chi denominator:
    numerator and denominator are accumulated separately, then exact
    division is performed at the end (guaranteed by theory).

    Args:
        lattice_N: FlatLattice of the shared matroid N
        T_M1_contracted: {flat_idx: T(M1/Z)} in BivariateLaurentPoly form
        T_M2_contracted: {flat_idx: T(M2/Z)} in BivariateLaurentPoly form
        r_N: rank of matroid N

    Returns:
        TuttePolynomial for the parallel connection
    """
    _log = get_log()
    _log.record(EventType.THEOREM6, "parallel_conn",
                f"Theorem 6: {lattice_N.num_flats} flats, rank {r_N}")

    # Precompute Mobius from bottom for all flats
    lattice_N.precompute_all_mobius_from_bottom()

    # Common-denominator accumulation: result = result_n / result_d
    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

    contributing = 0
    import time as _time
    t_start = _time.time()

    for w_idx in range(lattice_N.num_flats):
        w_flat = lattice_N.flat_by_idx(w_idx)
        w_size = len(w_flat)

        # Compute g_1(W) and g_2(W)
        g1 = _compute_g_for_flat(lattice_N, w_idx, T_M1_contracted)
        g2 = _compute_g_for_flat(lattice_N, w_idx, T_M2_contracted)

        if g1.is_zero() or g2.is_zero():
            continue

        # Denominator for this flat: (v+1)^{|W|} * chi(N/W; uv)
        chi_coeffs = lattice_N.characteristic_poly_coeffs(contraction_flat=w_flat)
        chi_uv = _chi_in_uv(chi_coeffs)
        y_pow_w = _y_power_in_uv(w_size)
        denom_W = y_pow_w * chi_uv

        # Numerator for this flat: g_1(W) * g_2(W)
        numer_W = g1 * g2

        # Accumulate: result_n/result_d += numer_W/denom_W
        # = (result_n * denom_W + numer_W * result_d) / (result_d * denom_W)
        result_n = result_n * denom_W + numer_W * result_d
        result_d = result_d * denom_W

        contributing += 1

        # Periodic simplification to prevent polynomial blow-up
        if contributing % 10 == 0:
            q, r = result_n.divmod(result_d)
            if r.is_zero():
                result_n = q
                result_d = BivariateLaurentPoly.one()

            if contributing % 100 == 0:
                t_now = _time.time()
                print(f"[Theorem6] {contributing} contributing flats "
                      f"(of {w_idx+1}/{lattice_N.num_flats} processed), "
                      f"numer={len(result_n.coeffs)} terms, "
                      f"denom={len(result_d.coeffs)} terms, "
                      f"{t_now - t_start:.1f}s", flush=True)

    t_end = _time.time()
    print(f"[Theorem6] Done: {contributing} contributing flats, "
          f"{t_end - t_start:.1f}s", flush=True)

    # Final: v^{r(N)} * result_n / result_d
    v_rN = BivariateLaurentPoly({(0, r_N): 1})
    final_n = v_rN * result_n

    # Exact division (guaranteed by theory)
    result = final_n // result_d

    # Convert back to T(x,y)
    return result.to_tutte_poly()


def theorem6_parallel_connection_ncell(
    lattice_N: FlatLattice,
    T_cells_contracted: List[Dict[int, BivariateLaurentPoly]],
    r_N: int,
) -> TuttePolynomial:
    """Compute T(P_N(M1, M2, ..., Mk)) for k >= 2 cells via Bonin-de Mier Theorem 6.

    Generalization of theorem6_parallel_connection to N cells. The formula:

    R(P_N; u,v) = v^{r(N)} * Σ_W [∏_i g_i(W)] / [(v+1)^{|W|} * χ(N/W; uv)]

    where g_i(W) = Σ_{Z≥W} μ(W,Z) * (v+1)^{|Z|} * v^{-r(Z)} * R(M_i/Z)

    The product ∏_i g_i(W) is over all cells instead of just 2.

    Args:
        lattice_N: FlatLattice of the shared matroid N
        T_cells_contracted: List of {flat_idx: T(Mi/Z)} for each cell
        r_N: rank of matroid N

    Returns:
        TuttePolynomial for the N-cell parallel connection
    """
    n_cells = len(T_cells_contracted)

    lattice_N.precompute_all_mobius_from_bottom()

    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

    contributing = 0
    import time as _time
    t_start = _time.time()

    for w_idx in range(lattice_N.num_flats):
        w_flat = lattice_N.flat_by_idx(w_idx)
        w_size = len(w_flat)

        # Compute g_i(W) for all cells and take product
        g_product = None
        skip = False
        for cell_idx in range(n_cells):
            g_i = _compute_g_for_flat(lattice_N, w_idx, T_cells_contracted[cell_idx])
            if g_i.is_zero():
                skip = True
                break
            if g_product is None:
                g_product = g_i
            else:
                g_product = g_product * g_i

        if skip or g_product is None:
            continue

        # Denominator: (v+1)^{|W|} * chi(N/W; uv)
        chi_coeffs = lattice_N.characteristic_poly_coeffs(contraction_flat=w_flat)
        chi_uv = _chi_in_uv(chi_coeffs)
        y_pow_w = _y_power_in_uv(w_size)
        denom_W = y_pow_w * chi_uv

        # Accumulate fraction
        result_n = result_n * denom_W + g_product * result_d
        result_d = result_d * denom_W

        contributing += 1

        if contributing % 10 == 0:
            q, r = result_n.divmod(result_d)
            if r.is_zero():
                result_n = q
                result_d = BivariateLaurentPoly.one()

    t_end = _time.time()

    # Final: v^{r(N)} * result_n / result_d
    v_rN = BivariateLaurentPoly({(0, r_N): 1})
    final_n = v_rN * result_n

    result = final_n // result_d
    return result.to_tutte_poly()


# =============================================================================
# EVALUATION-INTERPOLATION FOR THEOREM 6
# =============================================================================

# Threshold: use eval-interp when flat count exceeds this
EVAL_INTERP_THRESHOLD = 500


def _eval_chi_at(chi_coeffs: Dict[int, int], q: int) -> int:
    """Evaluate characteristic polynomial chi(q) = Σ c_k * q^k at integer q."""
    result = 0
    for power, c in chi_coeffs.items():
        result += c * (q ** power)
    return result


def _eval_y_power(k: int, v: int) -> int:
    """Evaluate (v+1)^k at integer v."""
    return (v + 1) ** k


def _eval_g_for_flat(
    lattice: FlatLattice,
    w_idx: int,
    t_contracted: Dict[int, BivariateLaurentPoly],
    u0: int,
    v0: int,
) -> 'Fraction':
    """Evaluate g_i(W) at numeric (u0, v0).

    g_i(W) = Σ_{Z >= W} mu(W,Z) * (v+1)^|Z| * v^{-r(Z)} * R(M_i/Z; u,v)

    At integer (u0, v0), this becomes a sum of Fractions.
    """
    from fractions import Fraction

    result = Fraction(0)
    flats_above = lattice.flats_above_idx(w_idx)
    lattice.precompute_mobius_from(lattice.flat_by_idx(w_idx))

    for z_idx in flats_above:
        mu_WZ = lattice._compute_mobius(w_idx, z_idx)
        if mu_WZ == 0:
            continue

        t_z = t_contracted.get(z_idx)
        if t_z is None:
            continue

        z_flat = lattice.flat_by_idx(z_idx)
        z_size = len(z_flat)
        z_rank = lattice.flat_rank_by_idx(z_idx)

        # (v0+1)^|Z| * v0^{-r(Z)} * R(M_i/Z; u0, v0)
        y_pow = (v0 + 1) ** z_size
        r_val = t_z.eval_at(u0, v0)  # Fraction

        if z_rank > 0:
            term = Fraction(mu_WZ * y_pow) * r_val / Fraction(v0 ** z_rank)
        else:
            term = Fraction(mu_WZ * y_pow) * r_val

        result += term

    return result


def _theorem6_eval_at_point(
    lattice_N: FlatLattice,
    T_M1_contracted: Dict[int, BivariateLaurentPoly],
    T_M2_contracted: Dict[int, BivariateLaurentPoly],
    r_N: int,
    u0: int,
    v0: int,
) -> 'Fraction':
    """Evaluate T(P_N(M1,M2)) at (u0, v0) in the rank-generating basis.

    Returns v0^{r_N} * Σ_W g1(W)*g2(W) / [(v0+1)^|W| * chi(N/W; u0*v0)]
    as an exact Fraction.
    """
    from fractions import Fraction

    q0 = u0 * v0  # chi argument

    result = Fraction(0)

    for w_idx in range(lattice_N.num_flats):
        w_flat = lattice_N.flat_by_idx(w_idx)
        w_size = len(w_flat)

        g1 = _eval_g_for_flat(lattice_N, w_idx, T_M1_contracted, u0, v0)
        if g1 == 0:
            continue

        g2 = _eval_g_for_flat(lattice_N, w_idx, T_M2_contracted, u0, v0)
        if g2 == 0:
            continue

        # Denominator: (v0+1)^|W| * chi(N/W; u0*v0)
        chi_coeffs = lattice_N.characteristic_poly_coeffs(contraction_flat=w_flat)
        chi_val = _eval_chi_at(chi_coeffs, q0)
        y_pow_w = (v0 + 1) ** w_size
        denom = y_pow_w * chi_val

        if denom == 0:
            raise ValueError(f"Zero denominator at flat {w_idx}, (u0,v0)=({u0},{v0})")

        result += g1 * g2 / Fraction(denom)

    # Multiply by v0^{r_N}
    result *= Fraction(v0 ** r_N)

    return result


def _lagrange_interpolate(xs: List[int], ys: List['Fraction']) -> List['Fraction']:
    """Univariate Lagrange interpolation: given (x_i, y_i), return coefficients [c_0, c_1, ..., c_n].

    Result: p(x) = c_0 + c_1*x + ... + c_n*x^n
    All arithmetic in Fraction for exact results.
    """
    from fractions import Fraction
    n = len(xs)
    # Build polynomial using Newton's divided differences for numerical stability
    # Then convert to monomial basis

    # Divided differences
    dd = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            dd[i] = (dd[i] - dd[i - 1]) / Fraction(xs[i] - xs[i - j])

    # Convert Newton form to monomial form
    # p(x) = dd[0] + dd[1]*(x-x0) + dd[2]*(x-x0)*(x-x1) + ...
    # Expand iteratively
    coeffs = [Fraction(0)] * n
    coeffs[0] = dd[n - 1]

    for k in range(n - 2, -1, -1):
        # Multiply current poly by (x - xs[k])
        new_coeffs = [Fraction(0)] * n
        for i in range(n - 1, 0, -1):
            new_coeffs[i] = coeffs[i - 1] - Fraction(xs[k]) * coeffs[i]
        new_coeffs[0] = -Fraction(xs[k]) * coeffs[0]
        # Add dd[k]
        new_coeffs[0] += dd[k]
        coeffs = new_coeffs

    return coeffs


def theorem6_eval_interp(
    lattice_N: FlatLattice,
    T_M1_contracted: Dict[int, BivariateLaurentPoly],
    T_M2_contracted: Dict[int, BivariateLaurentPoly],
    r_N: int,
    max_x_degree: int,
    max_y_degree: int,
) -> TuttePolynomial:
    """Compute T(P_N(M1,M2)) via evaluation-interpolation.

    Instead of symbolic fraction accumulation (which blows up for large lattices),
    evaluate the Theorem 6 formula at enough numeric points, then interpolate
    to recover the polynomial.

    Args:
        lattice_N: FlatLattice of the shared matroid N
        T_M1_contracted: {flat_idx: T(M1/Z)} in BivariateLaurentPoly form
        T_M2_contracted: {flat_idx: T(M2/Z)} in BivariateLaurentPoly form
        r_N: rank of matroid N
        max_x_degree: upper bound on x-degree of result
        max_y_degree: upper bound on y-degree of result

    Returns:
        TuttePolynomial for the parallel connection
    """
    from fractions import Fraction
    import time as _time

    t_start = _time.time()

    # Diagnostic: check coverage
    _nf_total = lattice_N.num_flats
    _m1_count = len(T_M1_contracted)
    _m2_count = len(T_M2_contracted)
    if _m1_count < _nf_total or _m2_count < _nf_total:
        _missing_m1 = set(range(_nf_total)) - set(T_M1_contracted.keys())
        _missing_m2 = set(range(_nf_total)) - set(T_M2_contracted.keys())
        print(f"[EvalInterp] WARNING: incomplete contractions! "
              f"M1: {_m1_count}/{_nf_total} (missing {len(_missing_m1)}), "
              f"M2: {_m2_count}/{_nf_total} (missing {len(_missing_m2)})", flush=True)
        if _missing_m1:
            _sample = sorted(_missing_m1)[:10]
            for _idx in _sample:
                _flat = lattice_N.flat_by_idx(_idx)
                _rank = lattice_N.flat_rank_by_idx(_idx)
                print(f"[EvalInterp]   Missing M1 z_idx={_idx}: "
                      f"rank={_rank}, size={len(_flat)}", flush=True)
    else:
        print(f"[EvalInterp] All {_nf_total} contractions present for both M1 and M2", flush=True)

    # Number of evaluation points needed
    nx = max_x_degree + 1  # coefficients 0..max_x_degree
    ny = max_y_degree + 1

    # Choose evaluation points avoiding chi(N/W; q) = 0.
    # chi(M; q) has integer roots at q = 0, 1, ..., r-1 for some matroids.
    # Since q = u*v = (x-1)*(y-1), we need u*v to avoid small integers.
    # Use x starting from r_N + 2 and y starting from r_N + 2 to ensure
    # q = u*v >= (r_N+1)^2 >> r_N, safely above any chi roots.
    x_start = r_N + 2
    y_start = r_N + 2
    x_points = list(range(x_start, x_start + nx))
    y_points = list(range(y_start, y_start + ny))
    u_points = [x - 1 for x in x_points]
    v_points = [y - 1 for y in y_points]

    nf = lattice_N.num_flats

    # Precompute flat metadata: sizes, ranks
    flat_size: List[int] = [0] * nf
    flat_rank: List[int] = [0] * nf
    for idx in range(nf):
        flat_size[idx] = len(lattice_N.flat_by_idx(idx))
        flat_rank[idx] = lattice_N.flat_rank_by_idx(idx)

    # Batch precompute Möbius values, flats_above, and chi coefficients
    flats_above_map, chi_coeffs_precomp = lattice_N.precompute_all_mobius_and_chi()

    # Determine active flats and build sparse Möbius lookups
    active_w: List[int] = []
    mobius_for_m1: Dict[int, List[Tuple[int, int]]] = {}
    mobius_for_m2: Dict[int, List[Tuple[int, int]]] = {}

    for w_idx in range(nf):
        flats_above = flats_above_map[w_idx]

        m1_entries = []
        m2_entries = []
        for z_idx in flats_above:
            mu_wz = lattice_N._mobius_cache.get((w_idx, z_idx), 0)
            if mu_wz == 0:
                continue
            if z_idx in T_M1_contracted:
                m1_entries.append((z_idx, mu_wz))
            if z_idx in T_M2_contracted:
                m2_entries.append((z_idx, mu_wz))

        if m1_entries and m2_entries:
            active_w.append(w_idx)
            mobius_for_m1[w_idx] = m1_entries
            mobius_for_m2[w_idx] = m2_entries

    t_prep = _time.time()
    print(f"[EvalInterp] Preparation: {t_prep - t_start:.1f}s "
          f"({nf} flats, {nx}×{ny}={nx*ny} points)", flush=True)

    # Evaluate T at all (x,y) points using batch Mobius inversion
    # For each (u0, v0) point:
    # 1. Compute h_i(Z) = (v0+1)^|Z| * v0^{-r(Z)} * R(M_i/Z; u0, v0) for all Z
    # 2. Compute g_i(W) via top-down Mobius inversion on the lattice
    # 3. Compute T = v0^r_N * Σ_W g1(W)*g2(W) / [(v0+1)^|W| * chi(N/W; u0*v0)]

    T_values: List[List['Fraction']] = []

    for j, v0 in enumerate(v_points):
        v0_plus_1 = v0 + 1
        # Precompute v0-dependent values
        v0_pow_r: Dict[int, int] = {0: 1}
        y_pow_s: Dict[int, int] = {0: 1}

        def _get_v0_pow(r: int) -> int:
            if r not in v0_pow_r:
                v0_pow_r[r] = v0 ** r
            return v0_pow_r[r]

        def _get_y_pow(s: int) -> int:
            if s not in y_pow_s:
                y_pow_s[s] = v0_plus_1 ** s
            return y_pow_s[s]

        row = []
        for i, u0 in enumerate(u_points):
            q0 = u0 * v0

            # Step 1: Compute h_i(Z) for all Z in active flats
            # h_i(Z) = (v0+1)^|Z| * v0^{-r(Z)} * R(M_i/Z; u0, v0)
            blp_cache: Dict[int, 'Fraction'] = {}  # id(blp) -> eval

            def _eval_blp(blp: BivariateLaurentPoly) -> 'Fraction':
                bid = id(blp)
                if bid not in blp_cache:
                    blp_cache[bid] = blp.eval_at(u0, v0)
                return blp_cache[bid]

            def _eval_h(z_idx: int, t_contracted: Dict[int, BivariateLaurentPoly]) -> 'Fraction':
                t_z = t_contracted.get(z_idx)
                if t_z is None:
                    return Fraction(0)
                z_size = flat_size[z_idx]
                z_rank = flat_rank[z_idx]
                y_pow = _get_y_pow(z_size)
                r_val = _eval_blp(t_z)
                if z_rank > 0:
                    return Fraction(y_pow) * r_val / Fraction(_get_v0_pow(z_rank))
                else:
                    return Fraction(y_pow) * r_val

            # Step 2: Correct Möbius inversion for g1, g2
            # g_i(W) = Σ_{Z >= W} μ(W,Z) * h_i(Z)
            point_result = Fraction(0)
            for w_idx in active_w:
                # g1(W)
                gv1 = Fraction(0)
                for z_idx, mu_wz in mobius_for_m1[w_idx]:
                    gv1 += mu_wz * _eval_h(z_idx, T_M1_contracted)
                if gv1 == 0:
                    continue

                # g2(W)
                gv2 = Fraction(0)
                for z_idx, mu_wz in mobius_for_m2[w_idx]:
                    gv2 += mu_wz * _eval_h(z_idx, T_M2_contracted)
                if gv2 == 0:
                    continue

                # chi(N/W; q0)
                w_size = flat_size[w_idx]
                chi_val = sum(
                    coeff * q0 ** power
                    for power, coeff in chi_coeffs_precomp[w_idx].items()
                )
                denom = _get_y_pow(w_size) * chi_val
                point_result += gv1 * gv2 / Fraction(denom)

            # Multiply by v0^r_N
            point_result *= Fraction(_get_v0_pow(r_N))
            if point_result.denominator == 1:
                row.append(Fraction(point_result.numerator))
            else:
                # Diagnostic: find which flat introduced the non-integer
                _diag_result = Fraction(0)
                for _dw in active_w:
                    _dg1 = Fraction(0)
                    for _dz, _dmu in mobius_for_m1[_dw]:
                        _dg1 += _dmu * _eval_h(_dz, T_M1_contracted)
                    if _dg1 == 0:
                        continue
                    _dg2 = Fraction(0)
                    for _dz, _dmu in mobius_for_m2[_dw]:
                        _dg2 += _dmu * _eval_h(_dz, T_M2_contracted)
                    if _dg2 == 0:
                        continue
                    _dchi = sum(c * q0**p for p, c in chi_coeffs_precomp[_dw].items())
                    _ddenom = _get_y_pow(flat_size[_dw]) * _dchi
                    _old = _diag_result
                    _diag_result += _dg1 * _dg2 / Fraction(_ddenom)
                    if _diag_result.denominator != 1 and _old.denominator == 1:
                        print(f"[DIAG] First non-integer at w_idx={_dw}, "
                              f"size={flat_size[_dw]}, rank={flat_rank[_dw]}", flush=True)
                        print(f"[DIAG]   chi_val={_dchi}, denom={_ddenom}", flush=True)
                        print(f"[DIAG]   gv1={_dg1}, gv2={_dg2}", flush=True)
                        print(f"[DIAG]   term denom={(_dg1*_dg2/Fraction(_ddenom)).denominator}", flush=True)
                        # Check contraction polynomials for this w
                        # Dump ALL h_i(Z) contributions to gv1
                        print(f"[DIAG]   h_i(Z) contributions to gv1:", flush=True)
                        for _dz, _dmu in mobius_for_m1[_dw]:
                            _h = _eval_h(_dz, T_M1_contracted)
                            _t = T_M1_contracted.get(_dz)
                            _tp = _t.to_tutte_poly() if _t else None
                            _neg = {}
                            if _tp:
                                _neg = {k: vv for k, vv in _tp.to_coefficients().items() if vv < 0}
                            print(f"[DIAG]     z={_dz} mu={_dmu} h={_h} "
                                  f"size={flat_size[_dz]} rank={flat_rank[_dz]}"
                                  f"{' NEG:'+str(_neg) if _neg else ''}", flush=True)
                            if _tp and flat_rank[_dz] <= 3:
                                print(f"[DIAG]       T(M/Z) coeffs: {_tp.to_coefficients()}", flush=True)
                        break
                raise ValueError(
                    f"T({x_points[i]},{y_points[j]}) not integer: "
                    f"denominator={point_result.denominator}"
                )


        T_values.append(row)

        if (j + 1) % 10 == 0 or j == 0:
            t_now = _time.time()
            print(f"[EvalInterp] Row {j+1}/{ny}: {t_now - t_prep:.1f}s", flush=True)

    t_eval = _time.time()
    print(f"[EvalInterp] All {nx*ny} evaluations: {t_eval - t_prep:.1f}s", flush=True)

    # Two-stage interpolation
    # Stage 1: For each y-point j, interpolate T(x, y_j) from x-values
    # Result: for each j, get coefficients c_a(y_j) for a=0..max_x_degree
    # where T(x, y_j) = Σ_a c_a(y_j) * x^a

    x_coeffs_at_y: List[List['Fraction']] = []  # [ny][nx] = c_a(y_j)

    for j in range(ny):
        coeffs = _lagrange_interpolate(x_points, T_values[j])
        x_coeffs_at_y.append(coeffs)

    # Stage 2: For each x-coefficient a, interpolate c_a(y) from y-values
    # c_a(y) = Σ_b t_{a,b} * y^b
    result_coeffs: Dict[Tuple[int, int], int] = {}

    for a in range(nx):
        # Collect c_a(y_j) for all y-points
        ca_values = [x_coeffs_at_y[j][a] for j in range(ny)]
        y_coeffs = _lagrange_interpolate(y_points, ca_values)

        for b in range(ny):
            val = y_coeffs[b]
            # Should be integer (or very close)
            int_val = int(round(float(val)))
            # Verify exactness
            if val != int_val:
                raise ValueError(
                    f"Non-integer coefficient at ({a},{b}): {val}. "
                    f"Degree bounds may be too low."
                )
            if int_val != 0:
                result_coeffs[(a, b)] = int_val

    t_interp = _time.time()
    print(f"[EvalInterp] Interpolation: {t_interp - t_eval:.1f}s", flush=True)
    print(f"[EvalInterp] Total: {t_interp - t_start:.1f}s, "
          f"{len(result_coeffs)} nonzero coefficients", flush=True)

    return TuttePolynomial.from_coefficients(result_coeffs)


def _compute_g_for_flat(
    lattice: FlatLattice,
    w_idx: int,
    t_contracted: Dict[int, BivariateLaurentPoly],
) -> BivariateLaurentPoly:
    """Compute g_i(W) = sum_{Z >= W} mu(W,Z) * (v+1)^{|Z|} * v^{-r(Z)} * R(M_i/Z).

    Args:
        lattice: FlatLattice of shared matroid
        w_idx: Index of flat W
        t_contracted: Pre-computed T(M_i/Z) in BivariateLaurentPoly form
    """
    result = BivariateLaurentPoly.zero()

    flats_above = lattice.flats_above_idx(w_idx)

    # Precompute Mobius from W for efficiency
    lattice.precompute_mobius_from(lattice.flat_by_idx(w_idx))

    for z_idx in flats_above:
        mu_WZ = lattice._compute_mobius(w_idx, z_idx)
        if mu_WZ == 0:
            continue

        t_z = t_contracted.get(z_idx)
        if t_z is None:
            continue

        z_flat = lattice.flat_by_idx(z_idx)
        z_size = len(z_flat)
        z_rank = lattice.flat_rank_by_idx(z_idx)

        # (v+1)^{|Z|} * v^{-r(Z)} * R(M_i/Z)
        y_pow_z = _y_power_in_uv(z_size)
        term = y_pow_z * t_z.shift_v(-z_rank)

        if mu_WZ != 1:
            term = mu_WZ * term

        result = result + term

    return result


def _theorem6_for_contraction(
    lattice_N: FlatLattice,
    f_idx: int,
    t_m1: Dict[int, BivariateLaurentPoly],
    t_m2: Dict[int, BivariateLaurentPoly],
    r_N: int,
) -> TuttePolynomial:
    """Compute T(PC/F) for a flat F using Theorem 6 on the interval [F, top].

    When we contract flat F in the parallel connection, the result is a
    parallel connection of contracted cells over the contracted shared matroid N/F.

    T(PC/F) is computed via Theorem 6 with:
    - Lattice restricted to flats >= F (the interval [F, top] in the lattice of N)
    - Cell contractions T(M_i/Z) for flats Z >= F (already precomputed)
    - Rank of contracted matroid = r(N) - r(F)

    The formula becomes:
    R(PC/F; u,v) = v^{r(N/F)} · Σ_{W >= F}
        [1 / ((v+1)^{|W|-|F|} · chi(N/W; uv))]
        · g_1^F(W) · g_2^F(W)

    where g_i^F(W) uses Mobius values mu(W, Z) in the interval [F, top]
    and T(M_i/Z) for Z >= F, with rank adjustments.

    Note: The characteristic polynomial chi(N/W) is the same regardless of F,
    since it only depends on the interval [W, top].

    Args:
        lattice_N: Full flat lattice of N
        f_idx: Index of flat F to contract
        t_m1: Precomputed T(M1/Z) for all flats Z
        t_m2: Precomputed T(M2/Z) for all flats Z
        r_N: Rank of full matroid N

    Returns:
        TuttePolynomial for T(PC/F)
    """
    f_flat = lattice_N.flat_by_idx(f_idx)
    f_rank = lattice_N.flat_rank_by_idx(f_idx)
    f_size = len(f_flat)
    contracted_rank = r_N - f_rank

    # Get flats above F (the interval [F, top])
    flats_above_f = lattice_N.flats_above_idx(f_idx)

    # Precompute Mobius from each W >= F within the interval
    for w_idx in flats_above_f:
        lattice_N.precompute_mobius_from(lattice_N.flat_by_idx(w_idx))

    # Common-denominator accumulation
    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

    for w_idx in flats_above_f:
        w_flat = lattice_N.flat_by_idx(w_idx)
        w_rank = lattice_N.flat_rank_by_idx(w_idx)

        # In the contracted matroid N/F:
        # - Size of flat W/F = |W| - |F|
        # - Rank of W/F = r(W) - r(F)
        w_contracted_size = len(w_flat) - f_size
        w_contracted_rank = w_rank - f_rank

        # Compute g_1^F(W) and g_2^F(W) in the interval [F, top]
        g1 = _compute_g_for_contracted_flat(lattice_N, w_idx, f_idx, t_m1, flats_above_f)
        g2 = _compute_g_for_contracted_flat(lattice_N, w_idx, f_idx, t_m2, flats_above_f)

        if g1.is_zero() or g2.is_zero():
            continue

        # Denominator: (v+1)^{|W|-|F|} · chi(N/W; uv)
        chi_coeffs = lattice_N.characteristic_poly_coeffs(contraction_flat=w_flat)
        chi_uv = _chi_in_uv(chi_coeffs)
        y_pow_w = _y_power_in_uv(w_contracted_size)
        denom_W = y_pow_w * chi_uv

        # Numerator: g1(W) · g2(W)
        numer_W = g1 * g2

        # Accumulate fractions
        result_n = result_n * denom_W + numer_W * result_d
        result_d = result_d * denom_W

    # Final: v^{contracted_rank} · result_n / result_d
    v_r = BivariateLaurentPoly({(0, contracted_rank): 1})
    final_n = v_r * result_n

    if result_d.is_zero():
        raise ValueError("_theorem6_for_contraction: zero denominator in Theorem 6 summation")
    if final_n.is_zero():
        return TuttePolynomial.zero()

    result = final_n // result_d
    return result.to_tutte_poly()


def _theorem6_for_contraction_ncell(
    lattice_N: FlatLattice,
    f_idx: int,
    t_cells: List[Dict[int, BivariateLaurentPoly]],
    r_N: int,
) -> TuttePolynomial:
    """N-cell version of _theorem6_for_contraction.

    Computes T(PC/F) where PC is the parallel connection of N cells.
    The formula is the same as the 2-cell version but with a product
    over all N cells instead of just 2.
    """
    f_flat = lattice_N.flat_by_idx(f_idx)
    f_rank = lattice_N.flat_rank_by_idx(f_idx)
    f_size = len(f_flat)
    contracted_rank = r_N - f_rank

    flats_above_f = lattice_N.flats_above_idx(f_idx)

    for w_idx in flats_above_f:
        lattice_N.precompute_mobius_from(lattice_N.flat_by_idx(w_idx))

    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

    for w_idx in flats_above_f:
        w_flat = lattice_N.flat_by_idx(w_idx)
        w_contracted_size = len(w_flat) - f_size

        # Compute g_i^F(W) for all cells and take product
        g_product = None
        skip = False
        for t_cell in t_cells:
            g_i = _compute_g_for_contracted_flat(
                lattice_N, w_idx, f_idx, t_cell, flats_above_f)
            if g_i.is_zero():
                skip = True
                break
            if g_product is None:
                g_product = g_i
            else:
                g_product = g_product * g_i

        if skip or g_product is None:
            continue

        # Denominator: (v+1)^{|W|-|F|} · chi(N/W; uv)
        chi_coeffs = lattice_N.characteristic_poly_coeffs(contraction_flat=w_flat)
        chi_uv = _chi_in_uv(chi_coeffs)
        y_pow_w = _y_power_in_uv(w_contracted_size)
        denom_W = y_pow_w * chi_uv

        # Accumulate fractions
        result_n = result_n * denom_W + g_product * result_d
        result_d = result_d * denom_W

    # Final: v^{contracted_rank} · result_n / result_d
    v_r = BivariateLaurentPoly({(0, contracted_rank): 1})
    final_n = v_r * result_n

    if result_d.is_zero():
        raise ValueError("_theorem6_for_contraction_ncell: zero denominator")
    if final_n.is_zero():
        return TuttePolynomial.zero()

    result = final_n // result_d
    return result.to_tutte_poly()


def _compute_g_for_contracted_flat(
    lattice: FlatLattice,
    w_idx: int,
    f_idx: int,
    t_contracted: Dict[int, BivariateLaurentPoly],
    interval_flats: List[int],
) -> BivariateLaurentPoly:
    """Compute g_i^F(W) for the contracted lattice interval [F, top].

    g_i^F(W) = Σ_{Z >= W, Z in interval} mu(W,Z) · (v+1)^{|Z|-|F|} · v^{-(r(Z)-r(F))} · R(M_i/Z)

    This is analogous to _compute_g_for_flat but with rank/size offsets for contraction.
    """
    f_size = len(lattice.flat_by_idx(f_idx))
    f_rank = lattice.flat_rank_by_idx(f_idx)

    result = BivariateLaurentPoly.zero()
    w_flat = lattice.flat_by_idx(w_idx)

    for z_idx in interval_flats:
        z_flat = lattice.flat_by_idx(z_idx)
        if not w_flat.issubset(z_flat):
            continue

        mu_WZ = lattice._compute_mobius(w_idx, z_idx)
        if mu_WZ == 0:
            continue

        t_z = t_contracted.get(z_idx)
        if t_z is None:
            continue

        # Contracted sizes/ranks relative to F
        z_contracted_size = len(z_flat) - f_size
        z_contracted_rank = lattice.flat_rank_by_idx(z_idx) - f_rank

        # (v+1)^{|Z|-|F|} · v^{-(r(Z)-r(F))} · R(M_i/Z)
        y_pow_z = _y_power_in_uv(z_contracted_size)
        term = y_pow_z * t_z.shift_v(-z_contracted_rank)

        if mu_WZ != 1:
            term = mu_WZ * term

        result = result + term

    return result


# =============================================================================
# THEOREM 10: K-SUM VIA DELETION OF SHARED EDGES
# =============================================================================

def theorem10_k_sum(
    pc_graph: Graph,
    shared_edges: List[Edge],
    engine: 'SynthesisEngine',
) -> TuttePolynomial:
    r"""Compute T(G1 ⊕_k G2) = T(P_N(M1,M2) \ T) via corrected inclusion-exclusion.

    The k-sum is the parallel connection with all shared edges deleted.
    Uses the identity:

        T(M\T) = sum_{S⊆T} (-1)^|S| (y-1)^{n(S)} T(M/S)

    where n(S) = |S| - r(S) is the nullity (number of dependent edges) of S
    in the graphic matroid. The (y-1)^{n(S)} correction accounts for edges
    that become loops after partial contraction — the naive formula without
    this factor is only correct when all subsets S are independent.

    Complexity: O(2^|T|) synthesis calls. Efficient for |T| <= ~10.

    Args:
        pc_graph: The parallel connection graph P_N(G1, G2)
        shared_edges: List of shared edges to delete (= E(N))
        engine: SynthesisEngine for computing Tutte polynomials

    Returns:
        TuttePolynomial for the k-sum
    """
    from itertools import combinations

    _log = get_log()
    n_subsets = 2 ** len(shared_edges)
    _log.record(EventType.KSUM, "parallel_conn",
                f"K-sum IE: {len(shared_edges)} shared edges, {n_subsets} subsets",
                LogLevel.DEBUG)

    result = TuttePolynomial.zero()

    for k in range(len(shared_edges) + 1):
        sign = (-1) ** k
        for S in combinations(shared_edges, k):
            # Compute nullity of S in the graphic matroid
            nullity = _edge_set_nullity(S)

            # Contract subset S in the PC graph
            mg = _contract_edges_in_graph(pc_graph, frozenset(S))
            poly = engine._synthesize_multigraph(mg)

            # Apply nullity correction: multiply by (y-1)^{n(S)}
            if nullity > 0:
                poly = _y_minus_1_power(nullity) * poly

            if sign == -1:
                poly = -poly
            result = result + poly

    return result


def _edge_set_nullity(edges: tuple) -> int:
    """Compute nullity n(S) = |S| - r(S) of an edge set in its graphic matroid.

    r(S) = number of edges in a spanning forest of the subgraph formed by S,
    computed via union-find.
    """
    if not edges:
        return 0

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    rank = 0
    for u, v in edges:
        parent.setdefault(u, u)
        parent.setdefault(v, v)
        if union(u, v):
            rank += 1

    return len(edges) - rank


def _y_minus_1_power(k: int) -> TuttePolynomial:
    """Compute (y-1)^k as a TuttePolynomial."""
    coeffs: Dict[Tuple[int, int], int] = {}
    for j in range(k + 1):
        coeff = comb(k, j) * ((-1) ** (k - j))
        if coeff != 0:
            coeffs[(0, j)] = coeff
    return TuttePolynomial.from_coefficients(coeffs)


# =============================================================================
# THEOREM 10 OPTIMIZED: K-SUM VIA FLAT-GROUPED THEOREM 6
# =============================================================================

def theorem10_k_sum_via_theorem6(
    ksum_graph: Graph,
    separator: Tuple[int, ...],
    k: int,
    engine: 'SynthesisEngine',
) -> TuttePolynomial:
    r"""Compute T(G1 ⊕_k G2) using flat-grouped Theorem 6, avoiding brute-force 2^|T|.

    Instead of iterating over all 2^|T| subsets of shared edges T (where |T| = C(k,2)),
    this groups subsets by their closure (flat) in the shared matroid N = M(K_k).
    Subsets with the same closure produce the same contracted matroid, collapsing
    2^|T| terms into |flats(N)| terms.

    For K_k on k vertices:
    - k=4: 15 flats vs 64 subsets
    - k=5: 52 flats vs 1024 subsets

    Each flat term uses polynomial arithmetic (BivariateLaurentPoly multiplication)
    instead of graph synthesis.

    Algorithm:
    1. Decompose ksum_graph into two cells by splitting at separator vertices
    2. Build shared matroid N = M(K_k) on separator vertices
    3. Build flat lattice of N
    4. For each cell, build extended graph (cell + separator clique edges)
    5. Precompute T(M_i/Z) for all flats Z
    6. For each flat F, compute:
       - coefficient(F) = Σ_{S: cl(S)=F} (-1)^|S| · (y-1)^{nullity(S)}
       - T(PC/F) via Theorem 6 restricted to flats >= F
    7. Sum: T(ksum) = Σ_F coefficient(F) · T(PC/F)

    Args:
        ksum_graph: The k-sum graph (with clique edges already deleted)
        separator: Tuple of k separator vertices
        k: Number of shared vertices
        engine: SynthesisEngine for computing Tutte polynomials

    Returns:
        TuttePolynomial for the k-sum
    """
    sv = sorted(separator)

    # Build the shared clique edges (K_k on separator vertices)
    clique_edges = [(sv[i], sv[j]) for i in range(k) for j in range(i + 1, k)]
    clique_edges_set = frozenset(clique_edges)

    # Reconstruct the parallel connection graph
    pc_edges = ksum_graph.edges | clique_edges_set
    pc_graph = Graph(nodes=ksum_graph.nodes, edges=pc_edges)

    # Split into two cells at separator vertices
    sep_set = set(sv)
    remaining_nodes = ksum_graph.nodes - sep_set

    # Find connected components of remaining nodes in ksum_graph
    visited = set()
    components = []
    for start in sorted(remaining_nodes):
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            for nb in ksum_graph.neighbors(node):
                if nb not in comp and nb not in sep_set:
                    stack.append(nb)
        visited |= comp
        components.append(comp)

    if len(components) < 2:
        return None

    # Build N cell node sets (each includes separator vertices)
    cell_node_sets = [comp | sep_set for comp in components]

    # Build the shared matroid N = graphic matroid of K_k
    inter_graph = Graph(
        nodes=frozenset(sv),
        edges=clique_edges_set,
    )
    matroid_N = GraphicMatroid(inter_graph)
    r_N = matroid_N.rank()

    # Build flat lattice of N
    flats, ranks, upper_covers = enumerate_flats_with_hasse(matroid_N)
    lattice_N = FlatLattice(matroid_N, flats=flats, ranks=ranks, upper_covers=upper_covers)

    # Build extended cell graphs and precompute contractions for ALL cells
    t_cells = []
    for cell_nodes in cell_node_sets:
        ext, shared = build_extended_cell_graph(pc_graph, cell_nodes, clique_edges)
        t_cell = precompute_contractions(ext, shared, lattice_N, engine)
        t_cells.append(t_cell)

    # Precompute all Mobius values
    lattice_N.precompute_all_mobius_from_bottom()

    # For each flat F, compute coefficient and T(PC/F) via N-cell Theorem 6
    result_uv = BivariateLaurentPoly.zero()

    for f_idx in range(lattice_N.num_flats):
        f_flat = lattice_N.flat_by_idx(f_idx)
        f_rank = lattice_N.flat_rank_by_idx(f_idx)

        # Compute flat coefficient
        coeff = _compute_flat_coefficient(matroid_N, lattice_N, f_idx)
        if coeff.is_zero():
            continue

        # Compute T(PC/F) via N-cell Theorem 6 on the interval [F, top]
        t_pc_f = _theorem6_for_contraction_ncell(lattice_N, f_idx, t_cells, r_N)

        # Convert T(PC/F) to uv-basis and multiply by coefficient
        t_pc_f_uv = BivariateLaurentPoly.from_tutte(t_pc_f)
        result_uv = result_uv + coeff * t_pc_f_uv

    # Convert back to Tutte polynomial
    return result_uv.to_tutte_poly()


def _compute_flat_coefficient(
    matroid_N: GraphicMatroid,
    lattice_N: FlatLattice,
    f_idx: int,
) -> BivariateLaurentPoly:
    r"""Compute the flat coefficient for flat F in the deletion formula.

    The brute-force Theorem 10 formula is:
        T(M\T) = Σ_{S⊆T} (-1)^|S| · (y-1)^{n(S)} · T(M/S)

    When grouping by closure F = cl(S), we use T_matroid(PC/F) from Theorem 6.
    But T(PC/S) ≠ T_matroid(PC/F) in general: contracting S leaves edges F\S
    in the graph, and since cl(S)=F, those edges become loops in PC/S.
    So T(PC/S) = y^{|F|-|S|} · T_matroid(PC/F).

    Absorbing the loop factor into the coefficient:

    coeff(F) = Σ_{S : cl(S)=F} (-1)^|S| · (y-1)^{|S|-r(F)} · y^{|F|-|S|}

    Via Möbius inversion (v = y-1, y = v+1):

    coeff(F) = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · (v+1)^{|F| - |G|}

    Derivation: group subsets by closure, apply Möbius inversion on
    f(S) = (-v)^|S| · (v+1)^{|F|-|S|}, simplify the binomial sum:

    Σ_{j=0}^{|G|} C(|G|,j) (-v)^j (v+1)^{|F|-j}
        = (v+1)^{|F|} · (1 - v/(v+1))^{|G|}
        = (v+1)^{|F|} · (1/(v+1))^{|G|}
        = (v+1)^{|F|-|G|}

    Args:
        matroid_N: The shared matroid
        lattice_N: Flat lattice of N
        f_idx: Index of flat F

    Returns:
        BivariateLaurentPoly representing the coefficient
    """
    f_flat = lattice_N.flat_by_idx(f_idx)
    f_rank = lattice_N.flat_rank_by_idx(f_idx)
    f_size = len(f_flat)

    # coeff(F) = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · (v+1)^{|F| - |G|}

    flats_below_f = []
    for g_idx in range(lattice_N.num_flats):
        g_flat = lattice_N.flat_by_idx(g_idx)
        if g_flat.issubset(f_flat):
            flats_below_f.append(g_idx)

    accumulator = BivariateLaurentPoly.zero()

    for g_idx in flats_below_f:
        mu_gf = lattice_N._compute_mobius(g_idx, f_idx)
        if mu_gf == 0:
            continue

        g_size = len(lattice_N.flat_by_idx(g_idx))
        exp = f_size - g_size  # |F| - |G|

        # (v+1)^{|F|-|G|} = Σ_{j=0}^{exp} C(exp, j) v^j
        v_plus_1_pow: Dict[Tuple[int, int], int] = {}
        for j in range(exp + 1):
            coeff_val = comb(exp, j)
            if coeff_val != 0:
                v_plus_1_pow[(0, j)] = coeff_val

        term = BivariateLaurentPoly(v_plus_1_pow)
        if mu_gf != 1:
            term = mu_gf * term

        accumulator = accumulator + term

    # Multiply by v^{-r(F)}
    return accumulator.shift_v(-f_rank)


# =============================================================================
# PRECOMPUTE CONTRACTIONS
# =============================================================================

def precompute_contractions(
    graph_i: Graph,
    inter_edges: FrozenSet[Edge],
    lattice_N: FlatLattice,
    engine: 'SynthesisEngine',
) -> Dict[int, BivariateLaurentPoly]:
    """For each flat Z of N, contract Z in graph_i and synthesize T(graph_i/Z).

    Contracting a flat Z means:
    1. For edges in Z that are also in graph_i, merge their endpoints
    2. The remaining graph is graph_i/Z
    3. Compute T(graph_i/Z) and convert to BivariateLaurentPoly

    Args:
        graph_i: Extended cell graph (cell + inter-cell edges)
        inter_edges: Shared edges (ground set of N) as frozenset
        lattice_N: FlatLattice of the shared matroid N
        engine: SynthesisEngine for computing Tutte polynomials

    Returns:
        Dict mapping flat index -> T(graph_i/Z) as BivariateLaurentPoly
    """
    _log = get_log()
    _log.record(EventType.THEOREM6, "parallel_conn",
                f"Precomputing {lattice_N.num_flats} contractions for "
                f"{graph_i.node_count()}n {graph_i.edge_count()}e",
                LogLevel.DEBUG)

    result: Dict[int, BivariateLaurentPoly] = {}
    # Cache by canonical key to avoid redundant synthesis
    canon_cache: Dict[str, BivariateLaurentPoly] = {}

    for z_idx in range(lattice_N.num_flats):
        z_flat = lattice_N.flat_by_idx(z_idx)

        if not z_flat:
            # Empty flat = no contraction. Use multigraph path to avoid
            # recursive k-sum on the cell graph (which creates dense PC graphs).
            mg = MultiGraph.from_graph(graph_i)
            synth_poly = engine._synthesize_multigraph(mg)
            result[z_idx] = BivariateLaurentPoly.from_tutte(synth_poly)
            continue

        # Contract edges in z_flat within graph_i
        # Returns a MultiGraph (contraction can create parallel edges)
        contracted_mg = _contract_edges_in_graph(graph_i, z_flat)

        if contracted_mg.edge_count() == 0:
            result[z_idx] = BivariateLaurentPoly.from_tutte(TuttePolynomial.one())
            continue

        # Check canonical key cache first
        canon_key = contracted_mg.canonical_key()
        if canon_key in canon_cache:
            result[z_idx] = canon_cache[canon_key]
            continue

        # Use multigraph synthesis with skip_minor_search for speed
        poly = engine._synthesize_multigraph(contracted_mg, skip_minor_search=True)
        poly_uv = BivariateLaurentPoly.from_tutte(poly)
        canon_cache[canon_key] = poly_uv
        result[z_idx] = poly_uv

    return result


def _contract_edges_in_graph(
    graph: Graph, edges_to_contract: FrozenSet[Edge]
) -> MultiGraph:
    """Contract specified edges in a graph by merging endpoints.

    For each edge in edges_to_contract that exists in graph,
    merge the endpoints. Returns a MultiGraph because contraction
    can create parallel edges (e.g., contracting one edge of a triangle
    produces two parallel edges between the remaining nodes).
    Loops (from contracting edges within a cycle) are tracked.

    Uses union-find to track node remapping across sequential contractions,
    so that later edges correctly reference nodes that were merged earlier.
    """
    # Start from a MultiGraph representation
    mg = MultiGraph.from_graph(graph)

    # Track node remapping: when merge_nodes(u, v) is called,
    # the removed node (max(u,v)) maps to the survivor (min(u,v)).
    node_map = {n: n for n in mg.nodes}

    def find(x):
        """Find current representative of node x."""
        while node_map.get(x, x) != x:
            # Path compression
            node_map[x] = node_map.get(node_map[x], node_map[x])
            x = node_map[x]
        return x

    # Contract each edge by merging its endpoints
    for u, v in edges_to_contract:
        # Map to current representatives
        ru, rv = find(u), find(v)
        if ru == rv:
            # Edge has become a loop due to prior merges.
            # Contracting a loop in a matroid removes it.
            if ru in mg.loop_counts and mg.loop_counts[ru] > 0:
                new_loops = dict(mg.loop_counts)
                new_loops[ru] -= 1
                if new_loops[ru] <= 0:
                    del new_loops[ru]
                mg = MultiGraph(
                    nodes=mg.nodes,
                    edge_counts=dict(mg.edge_counts),
                    loop_counts=new_loops,
                )
            continue
        if ru not in mg.nodes or rv not in mg.nodes:
            continue

        # Remove the contracted edge itself first, then merge
        edge = (min(ru, rv), max(ru, rv))
        new_edge_counts = dict(mg.edge_counts)
        if edge in new_edge_counts:
            new_edge_counts[edge] -= 1
            if new_edge_counts[edge] <= 0:
                del new_edge_counts[edge]

        mg = MultiGraph(
            nodes=mg.nodes,
            edge_counts=new_edge_counts,
            loop_counts=dict(mg.loop_counts),
        )
        mg = mg.merge_nodes(ru, rv)

        # Update node_map: the removed node maps to the survivor
        survivor = min(ru, rv)
        removed = max(ru, rv)
        node_map[removed] = survivor

    return mg


# =============================================================================
# BUILD EXTENDED CELL GRAPH
# =============================================================================

def build_extended_cell_graph(
    full_graph: Graph,
    cell_nodes: set,
    inter_edges: List[Tuple[int, int]],
) -> Tuple[Graph, FrozenSet[Edge]]:
    """Build extended cell graph: cell subgraph + inter-cell edges touching this cell.

    For Theorem 6, M_i = cell_i extended with inter-cell edges.
    The shared matroid N's ground set = the inter-cell edges.

    Args:
        full_graph: The full graph
        cell_nodes: Nodes belonging to this cell
        inter_edges: All inter-cell edges

    Returns:
        (extended_graph, shared_edges) where shared_edges are the
        inter-cell edges present in this extended graph
    """
    # Intra-cell edges
    intra_edges = set()
    for u, v in full_graph.edges:
        if u in cell_nodes and v in cell_nodes:
            intra_edges.add((u, v))

    # Inter-cell edges that touch this cell
    relevant_inter = set()
    all_nodes = set(cell_nodes)
    for u, v in inter_edges:
        if u in cell_nodes or v in cell_nodes:
            relevant_inter.add((min(u, v), max(u, v)))
            all_nodes.add(u)
            all_nodes.add(v)

    all_edges = intra_edges | relevant_inter

    extended = Graph(
        nodes=frozenset(all_nodes),
        edges=frozenset(all_edges),
    )

    return extended, frozenset(relevant_inter)


# =============================================================================
# PRODUCT-LATTICE THEOREM 6 (for disconnected inter-cell matroids)
# =============================================================================

# Maximum number of product flat pairs before we give up
MAX_PRODUCT_FLATS = 500_000


def precompute_contractions_product(
    graph_i: Graph,
    inter_edges: FrozenSet[Edge],
    lattice_1: FlatLattice,
    lattice_2: FlatLattice,
    engine: 'SynthesisEngine',
) -> Dict[Tuple[int, int], BivariateLaurentPoly]:
    """Precompute T(M_i / (Z1 union Z2)) for all flat pairs (Z1, Z2).

    Uses a two-stage approach:
    1. For each flat Z1 of N1: contract Z1 in graph_i -> cache intermediate multigraph
    2. For each (Z1, Z2): contract Z2 in the Z1-intermediate -> compute T

    With multigraph canonical key caching, many intermediates and finals are deduped.

    Args:
        graph_i: Extended cell graph (cell + ALL inter-cell edges)
        inter_edges: All shared edges (ground set of N1 + N2)
        lattice_1: FlatLattice of component 1 matroid
        lattice_2: FlatLattice of component 2 matroid
        engine: SynthesisEngine for computing Tutte polynomials

    Returns:
        Dict mapping (flat1_idx, flat2_idx) -> T(graph_i/(Z1∪Z2)) as BivariateLaurentPoly
    """
    _log = get_log()
    n_pairs = lattice_1.num_flats * lattice_2.num_flats
    _log.record(EventType.THEOREM6, "parallel_conn",
                f"Precomputing {n_pairs} product contractions for "
                f"{graph_i.node_count()}n {graph_i.edge_count()}e",
                LogLevel.DEBUG)

    result: Dict[Tuple[int, int], BivariateLaurentPoly] = {}

    # Stage 1: Contract each Z1 in graph_i, cache intermediate multigraphs
    intermediates: Dict[int, MultiGraph] = {}
    intermediate_keys: Dict[int, str] = {}

    for z1_idx in range(lattice_1.num_flats):
        z1_flat = lattice_1.flat_by_idx(z1_idx)
        if not z1_flat:
            intermediates[z1_idx] = MultiGraph.from_graph(graph_i)
        else:
            intermediates[z1_idx] = _contract_edges_in_graph(graph_i, z1_flat)
        intermediate_keys[z1_idx] = intermediates[z1_idx].canonical_key()

    # Stage 2: For each (Z1, Z2), contract Z2 in the Z1-intermediate
    # Use canonical key caching to avoid redundant synthesis
    final_cache: Dict[str, BivariateLaurentPoly] = {}

    for z1_idx in range(lattice_1.num_flats):
        mg_intermediate = intermediates[z1_idx]

        for z2_idx in range(lattice_2.num_flats):
            z2_flat = lattice_2.flat_by_idx(z2_idx)

            if not z2_flat:
                # No contraction needed for empty flat
                final_mg = mg_intermediate
            else:
                # Contract Z2 edges in the intermediate multigraph
                # Need to map Z2 edges through the Z1 contraction's node remapping
                # Since Z1 and Z2 are on disjoint edge sets (different components),
                # Z2's edges still exist in the intermediate (just with possibly remapped nodes)
                final_mg = _contract_edges_in_multigraph(mg_intermediate, z2_flat)

            canon_key = final_mg.canonical_key()
            if canon_key in final_cache:
                result[(z1_idx, z2_idx)] = final_cache[canon_key]
                continue

            if final_mg.edge_count() == 0:
                poly_uv = BivariateLaurentPoly.from_tutte(TuttePolynomial.one())
            else:
                poly = engine._synthesize_multigraph(final_mg)
                poly_uv = BivariateLaurentPoly.from_tutte(poly)

            final_cache[canon_key] = poly_uv
            result[(z1_idx, z2_idx)] = poly_uv

    return result


def _contract_edges_in_multigraph(
    mg: MultiGraph, edges_to_contract: FrozenSet[Edge]
) -> MultiGraph:
    """Contract edges in a multigraph by merging endpoints.

    Similar to _contract_edges_in_graph but operates on a MultiGraph.
    Handles the case where edge endpoints may have been remapped by prior contractions.
    """
    node_map = {n: n for n in mg.nodes}

    def find(x):
        while node_map.get(x, x) != x:
            node_map[x] = node_map.get(node_map[x], node_map[x])
            x = node_map[x]
        return x

    result_mg = mg

    for u, v in edges_to_contract:
        ru, rv = find(u), find(v)
        if ru == rv:
            # Already merged, remove loop if present
            if ru in result_mg.loop_counts and result_mg.loop_counts[ru] > 0:
                new_loops = dict(result_mg.loop_counts)
                new_loops[ru] -= 1
                if new_loops[ru] <= 0:
                    del new_loops[ru]
                result_mg = MultiGraph(
                    nodes=result_mg.nodes,
                    edge_counts=dict(result_mg.edge_counts),
                    loop_counts=new_loops,
                )
            continue
        if ru not in result_mg.nodes or rv not in result_mg.nodes:
            continue

        # Remove the edge, then merge nodes
        edge = (min(ru, rv), max(ru, rv))
        new_edge_counts = dict(result_mg.edge_counts)
        if edge in new_edge_counts:
            new_edge_counts[edge] -= 1
            if new_edge_counts[edge] <= 0:
                del new_edge_counts[edge]

        result_mg = MultiGraph(
            nodes=result_mg.nodes,
            edge_counts=new_edge_counts,
            loop_counts=dict(result_mg.loop_counts),
        )
        result_mg = result_mg.merge_nodes(ru, rv)

        survivor = min(ru, rv)
        removed = max(ru, rv)
        node_map[removed] = survivor

    return result_mg


def theorem6_product_lattice(
    lattice_1: FlatLattice,
    lattice_2: FlatLattice,
    T_M1_contracted: Dict[Tuple[int, int], BivariateLaurentPoly],
    T_M2_contracted: Dict[Tuple[int, int], BivariateLaurentPoly],
    r_N1: int,
    r_N2: int,
) -> TuttePolynomial:
    """Compute T(P_N(M1, M2)) via Theorem 6 with product lattice L(N1) x L(N2).

    For a disconnected shared matroid N = N1 + N2, the flat lattice decomposes as
    L(N) = L(N1) x L(N2). This exploits the product structure to factor:
    - Mobius: mu_N((W1,W2), (Z1,Z2)) = mu_N1(W1,Z1) * mu_N2(W2,Z2)
    - Chi: chi(N/(W1,W2); q) = chi(N1/W1; q) * chi(N2/W2; q)
    - Rank: r(W1,W2) = r(W1) + r(W2)
    - Size: |W1 union W2| = |W1| + |W2|

    Args:
        lattice_1: FlatLattice of component 1 matroid
        lattice_2: FlatLattice of component 2 matroid
        T_M1_contracted: {(flat1_idx, flat2_idx): T(M1/(Z1∪Z2))} as BivariateLaurentPoly
        T_M2_contracted: {(flat1_idx, flat2_idx): T(M2/(Z1∪Z2))} as BivariateLaurentPoly
        r_N1: rank of component 1 matroid
        r_N2: rank of component 2 matroid

    Returns:
        TuttePolynomial for the parallel connection
    """
    r_N = r_N1 + r_N2

    _log = get_log()
    n_product_flats = lattice_1.num_flats * lattice_2.num_flats
    _log.record(EventType.THEOREM6, "parallel_conn",
                f"Theorem 6 product: {lattice_1.num_flats}x{lattice_2.num_flats}="
                f"{n_product_flats} flat pairs, rank {r_N}")

    # Precompute Mobius values for both lattices
    lattice_1.precompute_all_mobius_from_bottom()
    lattice_2.precompute_all_mobius_from_bottom()

    # For each flat W in both lattices, precompute Mobius from W
    for w1_idx in range(lattice_1.num_flats):
        lattice_1.precompute_mobius_from(lattice_1.flat_by_idx(w1_idx))
    for w2_idx in range(lattice_2.num_flats):
        lattice_2.precompute_mobius_from(lattice_2.flat_by_idx(w2_idx))

    # Precompute chi polynomials for each component flat
    chi_cache_1: Dict[int, BivariateLaurentPoly] = {}
    for w1_idx in range(lattice_1.num_flats):
        w1_flat = lattice_1.flat_by_idx(w1_idx)
        chi_coeffs = lattice_1.characteristic_poly_coeffs(contraction_flat=w1_flat)
        chi_cache_1[w1_idx] = _chi_in_uv(chi_coeffs)

    chi_cache_2: Dict[int, BivariateLaurentPoly] = {}
    for w2_idx in range(lattice_2.num_flats):
        w2_flat = lattice_2.flat_by_idx(w2_idx)
        chi_coeffs = lattice_2.characteristic_poly_coeffs(contraction_flat=w2_flat)
        chi_cache_2[w2_idx] = _chi_in_uv(chi_coeffs)

    # Common-denominator accumulation
    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

    for w1_idx in range(lattice_1.num_flats):
        w1_flat = lattice_1.flat_by_idx(w1_idx)
        w1_size = len(w1_flat)

        flats_above_1 = lattice_1.flats_above_idx(w1_idx)

        for w2_idx in range(lattice_2.num_flats):
            w2_flat = lattice_2.flat_by_idx(w2_idx)
            w2_size = len(w2_flat)
            w_size = w1_size + w2_size

            flats_above_2 = lattice_2.flats_above_idx(w2_idx)

            # Compute g_1(W1,W2) and g_2(W1,W2) using factored Mobius
            g1 = _compute_g_product(
                lattice_1, lattice_2, w1_idx, w2_idx,
                flats_above_1, flats_above_2, T_M1_contracted,
            )
            g2 = _compute_g_product(
                lattice_1, lattice_2, w1_idx, w2_idx,
                flats_above_1, flats_above_2, T_M2_contracted,
            )

            if g1.is_zero() or g2.is_zero():
                continue

            # Factored denominator: (v+1)^{|W1|+|W2|} * chi(N1/W1) * chi(N2/W2)
            y_pow_w = _y_power_in_uv(w_size)
            denom_W = y_pow_w * chi_cache_1[w1_idx] * chi_cache_2[w2_idx]

            numer_W = g1 * g2

            # Accumulate fractions
            result_n = result_n * denom_W + numer_W * result_d
            result_d = result_d * denom_W

    # Final: v^{r(N)} * result_n / result_d
    v_rN = BivariateLaurentPoly({(0, r_N): 1})
    final_n = v_rN * result_n

    result = final_n // result_d
    return result.to_tutte_poly()


def _compute_g_product(
    lattice_1: FlatLattice,
    lattice_2: FlatLattice,
    w1_idx: int,
    w2_idx: int,
    flats_above_1: List[int],
    flats_above_2: List[int],
    t_contracted: Dict[Tuple[int, int], BivariateLaurentPoly],
) -> BivariateLaurentPoly:
    """Compute g_i(W1, W2) using factored Mobius over the product lattice.

    g_i(W1,W2) = sum_{Z1>=W1, Z2>=W2}
        mu_1(W1,Z1) * mu_2(W2,Z2) * (v+1)^{|Z1|+|Z2|} * v^{-r(Z1)-r(Z2)}
        * R(M_i / (Z1 union Z2))
    """
    result = BivariateLaurentPoly.zero()

    for z1_idx in flats_above_1:
        mu_1 = lattice_1._compute_mobius(w1_idx, z1_idx)
        if mu_1 == 0:
            continue

        z1_flat = lattice_1.flat_by_idx(z1_idx)
        z1_size = len(z1_flat)
        z1_rank = lattice_1.flat_rank_by_idx(z1_idx)

        for z2_idx in flats_above_2:
            mu_2 = lattice_2._compute_mobius(w2_idx, z2_idx)
            if mu_2 == 0:
                continue

            t_z = t_contracted.get((z1_idx, z2_idx))
            if t_z is None:
                continue

            z2_flat = lattice_2.flat_by_idx(z2_idx)
            z2_size = len(z2_flat)
            z2_rank = lattice_2.flat_rank_by_idx(z2_idx)

            z_size = z1_size + z2_size
            z_rank = z1_rank + z2_rank

            # (v+1)^{|Z|} * v^{-r(Z)} * R(M_i/Z)
            y_pow_z = _y_power_in_uv(z_size)
            term = y_pow_z * t_z.shift_v(-z_rank)

            mu_prod = mu_1 * mu_2
            if mu_prod != 1:
                term = mu_prod * term

            result = result + term

    return result


# =============================================================================
# SP-GUIDED BOTTOM-UP CONTRACTION CACHE
# =============================================================================

@dataclass
class SPNodeContractions:
    """Contraction data for one SP tree node, for one extended cell graph.

    At each SP node, we maintain:
    - flat_lattice: The FlatLattice of the sub-matroid at this node
    - contracted_mgs: For each flat index, the contracted multigraph
    - contracted_blps: For each flat index, T(M_i/Z) as BivariateLaurentPoly
    """
    flat_lattice: FlatLattice
    contracted_mgs: Dict[int, MultiGraph]
    contracted_blps: Dict[int, BivariateLaurentPoly]


def _collect_sp_edges(node: 'SPNode') -> List[Tuple[int, int]]:
    """Collect all edges from an SP tree node."""
    if node.type == "EDGE":
        return [node.edge]
    result = []
    for child in node.children:
        result.extend(_collect_sp_edges(child))
    return result


def _build_leaf_contractions(
    edge: Tuple[int, int],
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Build contraction data for a leaf (single edge) SP node.

    A single edge has 2 flats: {empty, {e}}.
    - flat 0 (empty): contracted graph = ext_cell_graph unchanged
    - flat 1 ({e}): contracted graph = ext_cell_graph with e's endpoints merged
    """
    # Build the sub-matroid for this single edge
    u, v = edge
    sub_graph = Graph(
        nodes=frozenset({u, v}),
        edges=frozenset({(min(u, v), max(u, v))}),
    )
    sub_matroid = GraphicMatroid(sub_graph)
    flats, ranks, uc = enumerate_flats_with_hasse(sub_matroid)
    lattice = FlatLattice(sub_matroid, flats=flats, ranks=ranks, upper_covers=uc)

    contracted_mgs: Dict[int, MultiGraph] = {}
    contracted_blps: Dict[int, BivariateLaurentPoly] = {}

    for z_idx in range(lattice.num_flats):
        z_flat = lattice.flat_by_idx(z_idx)
        if not z_flat:
            mg = MultiGraph.from_graph(ext_cell_graph)
        else:
            mg = _contract_edges_in_graph(ext_cell_graph, z_flat)

        contracted_mgs[z_idx] = mg

        canon_key = mg.canonical_key()
        if canon_key in canon_cache:
            contracted_blps[z_idx] = canon_cache[canon_key]
        else:
            if mg.edge_count() == 0:
                poly = TuttePolynomial.one()
            else:
                poly = engine._synthesize_multigraph(mg, skip_minor_search=True)
            blp = BivariateLaurentPoly.from_tutte(poly)
            canon_cache[canon_key] = blp
            contracted_blps[z_idx] = blp

    return SPNodeContractions(
        flat_lattice=lattice,
        contracted_mgs=contracted_mgs,
        contracted_blps=contracted_blps,
    )


def _compose_series_contractions(
    child_data_list: List[SPNodeContractions],
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Compose contraction data for a SERIES SP node.

    Series composition = direct sum of matroids.
    Flat lattice: L(N_A) x L(N_B) (product lattice).
    For flat (Z_A, Z_B): contract Z_A union Z_B in ext_cell_graph.

    For efficiency, we compose pairwise: fold children left-to-right.
    """
    if len(child_data_list) == 1:
        return child_data_list[0]

    # Fold left: compose first two, then compose result with third, etc.
    result = child_data_list[0]
    for i in range(1, len(child_data_list)):
        result = _compose_two_series(result, child_data_list[i], ext_cell_graph, engine, canon_cache)

    return result


def _compose_two_series(
    data_a: SPNodeContractions,
    data_b: SPNodeContractions,
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Compose two SPNodeContractions for series composition.

    Product lattice: flat (z_a, z_b) has edges Z_A union Z_B.
    Contract Z_A ∪ Z_B directly in ext_cell_graph; canonical key cache
    ensures we don't re-synthesize isomorphic contractions.
    """
    return _compose_two_generic(data_a, data_b, ext_cell_graph, engine, canon_cache)


def _compose_parallel_contractions(
    child_data_list: List[SPNodeContractions],
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Compose contraction data for a PARALLEL SP node.

    Parallel composition: the composed matroid's flat lattice is NOT a simple product.
    Shared terminals create cross-interactions. We build the flat lattice from scratch
    for the composed graph, but use children's cached graphs for efficiency.
    """
    if len(child_data_list) == 1:
        return child_data_list[0]

    # Fold left
    result = child_data_list[0]
    for i in range(1, len(child_data_list)):
        result = _compose_two_parallel(result, child_data_list[i], ext_cell_graph, engine, canon_cache)

    return result


def _compose_two_parallel(
    data_a: SPNodeContractions,
    data_b: SPNodeContractions,
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Compose two SPNodeContractions for parallel composition.

    Parallel composition shares terminals. The flat lattice of the parallel
    composition is built from scratch on the combined edge set.
    Contract each flat directly in ext_cell_graph; canonical key cache
    ensures we don't re-synthesize isomorphic contractions.
    """
    return _compose_two_generic(data_a, data_b, ext_cell_graph, engine, canon_cache)


def _compose_two_generic(
    data_a: SPNodeContractions,
    data_b: SPNodeContractions,
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    canon_cache: Dict[str, BivariateLaurentPoly],
) -> SPNodeContractions:
    """Generic composition: build combined flat lattice, contract each flat directly.

    Works for both series and parallel composition. The key insight is that
    contracting from ext_cell_graph is always correct (no node remapping issues),
    and the canonical key cache deduplicates across all levels so most contractions
    are cache hits.
    """
    lat_a = data_a.flat_lattice
    lat_b = data_b.flat_lattice
    edges_a = lat_a.matroid.ground_set
    edges_b = lat_b.matroid.ground_set
    all_edges = edges_a | edges_b

    all_nodes: Set[int] = set()
    for u, v in all_edges:
        all_nodes.add(u)
        all_nodes.add(v)

    combined_graph = Graph(nodes=frozenset(all_nodes), edges=all_edges)
    combined_matroid = GraphicMatroid(combined_graph)
    combined_flats, combined_ranks, combined_uc = enumerate_flats_with_hasse(combined_matroid)
    combined_lattice = FlatLattice(
        combined_matroid, flats=combined_flats, ranks=combined_ranks,
        upper_covers=combined_uc,
    )

    contracted_mgs: Dict[int, MultiGraph] = {}
    contracted_blps: Dict[int, BivariateLaurentPoly] = {}

    for z_idx in range(combined_lattice.num_flats):
        z_flat = combined_lattice.flat_by_idx(z_idx)

        if not z_flat:
            mg = MultiGraph.from_graph(ext_cell_graph)
        else:
            mg = _contract_edges_in_graph(ext_cell_graph, z_flat)

        contracted_mgs[z_idx] = mg

        canon_key = mg.canonical_key()
        if canon_key in canon_cache:
            contracted_blps[z_idx] = canon_cache[canon_key]
        else:
            if mg.edge_count() == 0:
                poly = TuttePolynomial.one()
            else:
                poly = engine._synthesize_multigraph(mg, skip_minor_search=True)
            blp = BivariateLaurentPoly.from_tutte(poly)
            canon_cache[canon_key] = blp
            contracted_blps[z_idx] = blp

    return SPNodeContractions(
        flat_lattice=combined_lattice,
        contracted_mgs=contracted_mgs,
        contracted_blps=contracted_blps,
    )


def build_contractions_bottom_up(
    sp_tree: 'SPNode',
    ext_cell_graph: Graph,
    engine: 'SynthesisEngine',
    verbose: bool = False,
    persistent_cache: Optional[Dict[str, 'TuttePolynomial']] = None,
) -> SPNodeContractions:
    """Build contraction cache bottom-up through SP decomposition tree.

    Traverses the SP tree bottom-up:
    - Leaves: Build 2-flat lattice, synthesize 2 contractions
    - Series nodes: Product lattice of children, progressive contraction
    - Parallel nodes: Build combined flat lattice, use children's cached graphs

    Returns SPNodeContractions for the root node, containing the full flat
    lattice and all T(M_i/Z) as BivariateLaurentPoly.

    Args:
        sp_tree: SP decomposition tree root
        ext_cell_graph: Extended cell graph (cell + inter-cell edges)
        engine: SynthesisEngine for polynomial computation
        verbose: Print progress information
        persistent_cache: Optional Dict[canonical_key, TuttePolynomial] to
            warm the canon_cache from and write new entries back to.

    Returns:
        SPNodeContractions with flat lattice and cached contractions
    """
    # Shared canonical key cache across all nodes (BLP form for computation)
    canon_cache: Dict[str, BivariateLaurentPoly] = {}

    # Warm from persistent cache if provided
    if persistent_cache:
        for key, poly in persistent_cache.items():
            canon_cache[key] = BivariateLaurentPoly.from_tutte(poly)

    initial_size = len(canon_cache)
    synthesis_count = [0]  # Use list for mutability in nested function

    def _log(msg: str):
        if verbose:
            print(f"[SP-BU] {msg}", flush=True)

    def _build_recursive(node: 'SPNode', depth: int = 0) -> SPNodeContractions:
        if node.type == "EDGE":
            data = _build_leaf_contractions(
                node.edge, ext_cell_graph, engine, canon_cache,
            )
            new_synth = len(canon_cache) - synthesis_count[0]
            if new_synth > 0:
                synthesis_count[0] = len(canon_cache)
            _log(f"{'  ' * depth}EDGE {node.edge}: "
                 f"{data.flat_lattice.num_flats} flats, "
                 f"cache={len(canon_cache)} unique")
            return data

        # Recurse into children
        child_data = [_build_recursive(c, depth + 1) for c in node.children]

        if node.type == "SERIES":
            data = _compose_series_contractions(
                child_data, ext_cell_graph, engine, canon_cache,
            )
        elif node.type == "PARALLEL":
            data = _compose_parallel_contractions(
                child_data, ext_cell_graph, engine, canon_cache,
            )
        else:
            raise ValueError(f"Unknown SP node type: {node.type}")

        _log(f"{'  ' * depth}{node.type}: "
             f"{data.flat_lattice.num_flats} flats, "
             f"cache={len(canon_cache)} unique")

        return data

    result = _build_recursive(sp_tree)

    # Write new entries back to persistent cache
    if persistent_cache is not None:
        new_entries = 0
        for key, blp in canon_cache.items():
            if key not in persistent_cache:
                persistent_cache[key] = blp.to_tutte_poly()
                new_entries += 1
        _log(f"Wrote {new_entries} new entries to persistent cache")

    _log(f"Bottom-up complete: {result.flat_lattice.num_flats} root flats, "
         f"{len(canon_cache)} unique contractions cached "
         f"({len(canon_cache) - initial_size} new)")

    return result


def sp_guided_precompute_contractions(
    component_graph: Graph,
    ext_cell_graph: Graph,
    component_lattice: FlatLattice,
    engine: 'SynthesisEngine',
    verbose: bool = False,
    persistent_cache: Optional[Dict[str, 'TuttePolynomial']] = None,
) -> Dict[int, BivariateLaurentPoly]:
    """Precompute T(M_i/Z) for all flats Z using SP-guided bottom-up approach.

    This is the main entry point that replaces precompute_contractions() for
    series-parallel inter-cell components.

    1. Decompose the component into an SP tree
    2. Build contractions bottom-up through the tree
    3. Map the root's flat lattice back to component_lattice indices

    Args:
        component_graph: The inter-cell component graph (must be SP)
        ext_cell_graph: Extended cell graph (cell + inter-cell edges)
        component_lattice: FlatLattice of the component matroid
        engine: SynthesisEngine for polynomial computation
        verbose: Print progress information
        persistent_cache: Optional Dict[canonical_key, TuttePolynomial] for
            warm-starting and persisting contraction results across sessions.

    Returns:
        Dict mapping flat index (in component_lattice) -> T(M_i/Z) as BLP
    """
    from ..graphs.series_parallel import decompose_series_parallel

    sp_tree = decompose_series_parallel(component_graph)
    if sp_tree is None:
        raise ValueError("Component graph is not series-parallel")

    # Build contractions bottom-up
    root_data = build_contractions_bottom_up(
        sp_tree, ext_cell_graph, engine, verbose=verbose,
        persistent_cache=persistent_cache,
    )

    # Map root's flat lattice to component_lattice indices
    root_lattice = root_data.flat_lattice
    root_flat_map: Dict[FrozenSet[Edge], int] = {}
    for idx in range(root_lattice.num_flats):
        root_flat_map[root_lattice.flat_by_idx(idx)] = idx

    result: Dict[int, BivariateLaurentPoly] = {}
    for z_idx in range(component_lattice.num_flats):
        z_flat = component_lattice.flat_by_idx(z_idx)
        root_idx = root_flat_map.get(z_flat)
        if root_idx is not None and root_idx in root_data.contracted_blps:
            result[z_idx] = root_data.contracted_blps[root_idx]

    return result


def sp_guided_precompute_contractions_product(
    component_graphs: List[Graph],
    ext_cell_graph: Graph,
    lattices: List[FlatLattice],
    engine: 'SynthesisEngine',
    verbose: bool = False,
    persistent_cache: Optional[Dict[str, 'TuttePolynomial']] = None,
) -> Dict[Tuple[int, int], BivariateLaurentPoly]:
    """Precompute T(M_i/(Z1∪Z2)) for product lattice using SP-guided approach.

    For each component, builds contractions bottom-up. Then for each flat pair
    (Z1, Z2), contracts Z2 in the Z1-contracted graph (or vice versa).

    Args:
        component_graphs: List of 2 inter-cell component graphs (must be SP)
        ext_cell_graph: Extended cell graph
        lattices: List of 2 FlatLattice objects
        engine: SynthesisEngine
        verbose: Print progress
        persistent_cache: Optional Dict[canonical_key, TuttePolynomial] for
            warm-starting and persisting contraction results across sessions.

    Returns:
        Dict mapping (flat1_idx, flat2_idx) -> T(M_i/(Z1∪Z2)) as BLP
    """
    from ..graphs.series_parallel import decompose_series_parallel, is_series_parallel

    assert len(component_graphs) == 2 and len(lattices) == 2

    def _log(msg: str):
        if verbose:
            print(f"[SP-Product] {msg}", flush=True)

    # Build SP-guided contractions for each component independently
    sp_data = []
    for i, (comp, lat) in enumerate(zip(component_graphs, lattices)):
        if not is_series_parallel(comp):
            raise ValueError(f"Component {i} is not series-parallel")

        sp_tree = decompose_series_parallel(comp)
        root_data = build_contractions_bottom_up(
            sp_tree, ext_cell_graph, engine, verbose=verbose,
            persistent_cache=persistent_cache,
        )
        _log(f"Component {i}: {root_data.flat_lattice.num_flats} root flats, "
             f"built bottom-up")

        # Map root flat lattice to component lattice
        root_lattice = root_data.flat_lattice
        root_flat_map: Dict[FrozenSet[Edge], int] = {}
        for idx in range(root_lattice.num_flats):
            root_flat_map[root_lattice.flat_by_idx(idx)] = idx

        sp_data.append((root_data, root_flat_map))

    lat1, lat2 = lattices
    root_data_1, root_map_1 = sp_data[0]
    root_data_2, root_map_2 = sp_data[1]

    # Now compute T(M_i/(Z1∪Z2)) for all flat pairs
    result: Dict[Tuple[int, int], BivariateLaurentPoly] = {}
    canon_cache: Dict[str, BivariateLaurentPoly] = {}

    # Pre-populate cache from persistent cache
    if persistent_cache:
        for key, poly in persistent_cache.items():
            canon_cache[key] = BivariateLaurentPoly.from_tutte(poly)

    # Also pre-populate from both roots' contractions
    for data in [root_data_1, root_data_2]:
        for idx, blp in data.contracted_blps.items():
            mg = data.contracted_mgs[idx]
            canon_cache[mg.canonical_key()] = blp

    initial_cache_size = len(canon_cache)
    total_pairs = lat1.num_flats * lat2.num_flats
    computed = 0

    for z1_idx in range(lat1.num_flats):
        z1_flat = lat1.flat_by_idx(z1_idx)

        # Get the Z1-contracted multigraph from component 1's cache
        root_idx_1 = root_map_1.get(z1_flat)
        if root_idx_1 is not None:
            mg_after_z1 = root_data_1.contracted_mgs[root_idx_1]
        else:
            # Fallback
            if not z1_flat:
                mg_after_z1 = MultiGraph.from_graph(ext_cell_graph)
            else:
                mg_after_z1 = _contract_edges_in_graph(ext_cell_graph, z1_flat)

        for z2_idx in range(lat2.num_flats):
            z2_flat = lat2.flat_by_idx(z2_idx)

            if not z2_flat:
                final_mg = mg_after_z1
            else:
                final_mg = _contract_edges_in_multigraph(mg_after_z1, z2_flat)

            canon_key = final_mg.canonical_key()
            if canon_key in canon_cache:
                result[(z1_idx, z2_idx)] = canon_cache[canon_key]
            else:
                if final_mg.edge_count() == 0:
                    blp = BivariateLaurentPoly.from_tutte(TuttePolynomial.one())
                else:
                    poly = engine._synthesize_multigraph(final_mg, skip_minor_search=True)
                    blp = BivariateLaurentPoly.from_tutte(poly)
                canon_cache[canon_key] = blp
                result[(z1_idx, z2_idx)] = blp

            computed += 1
            if verbose and computed % 1000 == 0:
                _log(f"  {computed}/{total_pairs} pairs computed, "
                     f"{len(canon_cache)} unique")

    # Write new entries back to persistent cache
    if persistent_cache is not None:
        new_entries = 0
        for key, blp in canon_cache.items():
            if key not in persistent_cache:
                persistent_cache[key] = blp.to_tutte_poly()
                new_entries += 1
        _log(f"Wrote {new_entries} new entries to persistent cache")

    _log(f"Product contractions complete: {computed} pairs, "
         f"{len(canon_cache)} unique contractions "
         f"({len(canon_cache) - initial_cache_size} new)")

    return result


# =============================================================================
# GROUPED / FACTORED PRODUCT LATTICE APPROACH
# =============================================================================

@dataclass
class GroupedContractionResult:
    """Result of grouped precomputation for one extended cell graph.

    Groups Z1 flats by canonical key of their contracted multigraph.
    For each group, stores one representative Z2 pass.

    Attributes:
        z1_groups: ckey -> (representative mg_after_z1, [z1_indices])
        group_z2_blps: ckey -> {z2_idx: BLP for T(ext_cell / (Z1∪Z2))}
        root_data: The SPNodeContractions from bottom-up build
        root_map: Mapping from flat frozenset -> root lattice index
    """
    z1_groups: Dict[str, Tuple[MultiGraph, List[int]]]
    group_z2_blps: Dict[str, Dict[int, BivariateLaurentPoly]]
    root_data: SPNodeContractions
    root_map: Dict[FrozenSet[Edge], int]


def sp_guided_precompute_contractions_product_grouped(
    component_graphs: List[Graph],
    ext_cell_graph: Graph,
    lattices: List[FlatLattice],
    engine: 'SynthesisEngine',
    verbose: bool = False,
    persistent_cache: Optional[Dict[str, 'TuttePolynomial']] = None,
) -> GroupedContractionResult:
    """Grouped precomputation: group Z1 by canonical key, run Z2 once per group.

    Instead of iterating all N1*N2 pairs, groups Z1 flats that produce
    isomorphic contracted multigraphs. For each unique Z1 contraction,
    runs the Z2 loop once. Cost: G1 * N2 instead of N1 * N2.

    Args:
        component_graphs: List of 2 inter-cell component graphs (must be SP)
        ext_cell_graph: Extended cell graph
        lattices: List of 2 FlatLattice objects
        engine: SynthesisEngine
        verbose: Print progress
        persistent_cache: Optional persistent cache

    Returns:
        GroupedContractionResult with grouped Z1 and per-group Z2 BLPs
    """
    from ..graphs.series_parallel import decompose_series_parallel, is_series_parallel

    assert len(component_graphs) == 2 and len(lattices) == 2

    def _log(msg: str):
        if verbose:
            print(f"[SP-Grouped] {msg}", flush=True)

    # Build SP-guided contractions for each component independently
    sp_data = []
    for i, (comp, lat) in enumerate(zip(component_graphs, lattices)):
        if not is_series_parallel(comp):
            raise ValueError(f"Component {i} is not series-parallel")

        sp_tree = decompose_series_parallel(comp)
        root_data = build_contractions_bottom_up(
            sp_tree, ext_cell_graph, engine, verbose=verbose,
            persistent_cache=persistent_cache,
        )
        _log(f"Component {i}: {root_data.flat_lattice.num_flats} root flats")

        root_lattice = root_data.flat_lattice
        root_flat_map: Dict[FrozenSet[Edge], int] = {}
        for idx in range(root_lattice.num_flats):
            root_flat_map[root_lattice.flat_by_idx(idx)] = idx

        sp_data.append((root_data, root_flat_map))

    lat1, lat2 = lattices
    root_data_1, root_map_1 = sp_data[0]
    root_data_2, root_map_2 = sp_data[1]

    # Step 1: Group Z1 flats by canonical key of their contracted multigraph
    z1_groups: Dict[str, Tuple[MultiGraph, List[int]]] = defaultdict(lambda: (None, []))
    z1_groups_build: Dict[str, List[int]] = defaultdict(list)
    z1_representatives: Dict[str, MultiGraph] = {}

    for z1_idx in range(lat1.num_flats):
        z1_flat = lat1.flat_by_idx(z1_idx)
        root_idx_1 = root_map_1.get(z1_flat)
        if root_idx_1 is not None:
            mg_after_z1 = root_data_1.contracted_mgs[root_idx_1]
        else:
            if not z1_flat:
                mg_after_z1 = MultiGraph.from_graph(ext_cell_graph)
            else:
                mg_after_z1 = _contract_edges_in_graph(ext_cell_graph, z1_flat)

        ckey = mg_after_z1.canonical_key()
        z1_groups_build[ckey].append(z1_idx)
        if ckey not in z1_representatives:
            z1_representatives[ckey] = mg_after_z1

    G1 = len(z1_groups_build)
    _log(f"G1={G1} unique Z1 contractions from {lat1.num_flats} flats")

    z1_groups_final: Dict[str, Tuple[MultiGraph, List[int]]] = {
        ckey: (z1_representatives[ckey], indices)
        for ckey, indices in z1_groups_build.items()
    }

    # Build canon cache from both roots
    canon_cache: Dict[str, BivariateLaurentPoly] = {}
    if persistent_cache:
        for key, poly in persistent_cache.items():
            canon_cache[key] = BivariateLaurentPoly.from_tutte(poly)
    for data in [root_data_1, root_data_2]:
        for idx, blp in data.contracted_blps.items():
            mg = data.contracted_mgs[idx]
            canon_cache[mg.canonical_key()] = blp

    initial_cache_size = len(canon_cache)

    # Step 2: For each unique Z1 group, run Z2 loop once
    group_z2_blps: Dict[str, Dict[int, BivariateLaurentPoly]] = {}
    total_ops = G1 * lat2.num_flats
    computed = 0

    for g_idx, (ckey, (mg_after_z1, z1_indices)) in enumerate(z1_groups_final.items()):
        z2_blps: Dict[int, BivariateLaurentPoly] = {}

        # Use structural fingerprint for dedup within this Z1 group's Z2 pass
        struct_cache: Dict[Tuple, str] = {}  # struct_key -> canonical_key

        for z2_idx in range(lat2.num_flats):
            z2_flat = lat2.flat_by_idx(z2_idx)

            if not z2_flat:
                final_mg = mg_after_z1
            else:
                final_mg = _contract_edges_in_multigraph(mg_after_z1, z2_flat)

            # Structural fingerprint for fast dedup
            struct_key = (
                frozenset(final_mg.nodes),
                tuple(sorted(final_mg.edge_counts.items())),
                tuple(sorted(final_mg.loop_counts.items())),
            )

            if struct_key in struct_cache:
                # Same structure -> same canonical key
                c_key = struct_cache[struct_key]
            else:
                c_key = final_mg.canonical_key()
                struct_cache[struct_key] = c_key

            if c_key in canon_cache:
                z2_blps[z2_idx] = canon_cache[c_key]
            else:
                if final_mg.edge_count() == 0:
                    blp = BivariateLaurentPoly.from_tutte(TuttePolynomial.one())
                else:
                    poly = engine._synthesize_multigraph(final_mg, skip_minor_search=True)
                    blp = BivariateLaurentPoly.from_tutte(poly)
                canon_cache[c_key] = blp
                z2_blps[z2_idx] = blp

            computed += 1

        group_z2_blps[ckey] = z2_blps

        if verbose:
            _log(f"  Group {g_idx+1}/{G1} (ckey {ckey[:16]}...): "
                 f"{len(z1_indices)} Z1 flats, {len(struct_cache)} unique Z2 structures, "
                 f"cache={len(canon_cache)}")

    # Write new entries back to persistent cache
    if persistent_cache is not None:
        new_entries = 0
        for key, blp in canon_cache.items():
            if key not in persistent_cache:
                persistent_cache[key] = blp.to_tutte_poly()
                new_entries += 1
        _log(f"Wrote {new_entries} new entries to persistent cache")

    _log(f"Grouped contractions complete: G1={G1}, {computed} group×Z2 ops "
         f"(vs {lat1.num_flats * lat2.num_flats} full product), "
         f"{len(canon_cache)} unique ({len(canon_cache) - initial_cache_size} new)")

    return GroupedContractionResult(
        z1_groups=z1_groups_final,
        group_z2_blps=group_z2_blps,
        root_data=root_data_1,
        root_map=root_map_1,
    )


def theorem6_product_lattice_factored(
    lattice_1: FlatLattice,
    lattice_2: FlatLattice,
    grouped_1: GroupedContractionResult,
    grouped_2: GroupedContractionResult,
    r_N1: int,
    r_N2: int,
) -> TuttePolynomial:
    """Theorem 6 with α/β-accelerated g computation via grouped contractions.

    Uses precomputed groups to decompose g_i(W1,W2) = Σ_k α_i_k(W1) · β_i_k(W2),
    replacing the O(|flats_above|²) inner Möbius sum with O(G1_active) lookups per
    (W1,W2) pair. Keeps the original fraction accumulation over (W1,W2) pairs.

    Row-by-row: for each W1, accumulates the W2 sum as a fraction, tries to
    simplify, then accumulates into the outer W1 sum.

    Args:
        lattice_1, lattice_2: FlatLattices of components
        grouped_1, grouped_2: GroupedContractionResult for cells 1 and 2
        r_N1, r_N2: Ranks of components

    Returns:
        TuttePolynomial for the parallel connection
    """
    import time
    r_N = r_N1 + r_N2

    # Precompute Möbius values
    lattice_1.precompute_all_mobius_from_bottom()
    lattice_2.precompute_all_mobius_from_bottom()
    for w1_idx in range(lattice_1.num_flats):
        lattice_1.precompute_mobius_from(lattice_1.flat_by_idx(w1_idx))
    for w2_idx in range(lattice_2.num_flats):
        lattice_2.precompute_mobius_from(lattice_2.flat_by_idx(w2_idx))

    # Precompute chi polynomials
    chi_cache_1: Dict[int, BivariateLaurentPoly] = {}
    for w1_idx in range(lattice_1.num_flats):
        w1_flat = lattice_1.flat_by_idx(w1_idx)
        chi_coeffs = lattice_1.characteristic_poly_coeffs(contraction_flat=w1_flat)
        chi_cache_1[w1_idx] = _chi_in_uv(chi_coeffs)

    chi_cache_2: Dict[int, BivariateLaurentPoly] = {}
    for w2_idx in range(lattice_2.num_flats):
        w2_flat = lattice_2.flat_by_idx(w2_idx)
        chi_coeffs = lattice_2.characteristic_poly_coeffs(contraction_flat=w2_flat)
        chi_cache_2[w2_idx] = _chi_in_uv(chi_coeffs)

    # Precompute (v+1)^|Z| * v^{-r(Z)} for all Z1 and Z2 flats
    z1_weight: Dict[int, BivariateLaurentPoly] = {}
    for z1_idx in range(lattice_1.num_flats):
        z1_flat = lattice_1.flat_by_idx(z1_idx)
        z1_rank = lattice_1.flat_rank_by_idx(z1_idx)
        z1_weight[z1_idx] = _y_power_in_uv(len(z1_flat)).shift_v(-z1_rank)

    z2_weight: Dict[int, BivariateLaurentPoly] = {}
    for z2_idx in range(lattice_2.num_flats):
        z2_flat = lattice_2.flat_by_idx(z2_idx)
        z2_rank = lattice_2.flat_rank_by_idx(z2_idx)
        z2_weight[z2_idx] = _y_power_in_uv(len(z2_flat)).shift_v(-z2_rank)

    n1 = lattice_1.num_flats
    n2 = lattice_2.num_flats

    t0 = time.time()

    # Precompute α_k(W1) and β_k(W2) for both cells
    # α_k(W1) = Σ_{Z1∈group_k, Z1≥W1} μ(W1,Z1) · weight(Z1)
    # β_k(W2) = Σ_{Z2≥W2} μ(W2,Z2) · weight(Z2) · R(G_k/Z2)

    def _precompute_alpha_beta(grouped: GroupedContractionResult):
        """Return (alpha, beta) where:
        alpha[ckey] = Dict[w1_idx, BLP]
        beta[ckey] = Dict[w2_idx, BLP]
        """
        alpha: Dict[str, Dict[int, BivariateLaurentPoly]] = {}
        beta: Dict[str, Dict[int, BivariateLaurentPoly]] = {}

        for ckey, (mg, z1_indices) in grouped.z1_groups.items():
            z1_set = set(z1_indices)

            alpha_k: Dict[int, BivariateLaurentPoly] = {}
            for w1_idx in range(n1):
                flats_above = lattice_1.flats_above_idx(w1_idx)
                acc = BivariateLaurentPoly.zero()
                for z1_idx in flats_above:
                    if z1_idx not in z1_set:
                        continue
                    mu = lattice_1._compute_mobius(w1_idx, z1_idx)
                    if mu == 0:
                        continue
                    term = z1_weight[z1_idx]
                    if mu != 1:
                        term = mu * term
                    acc = acc + term
                if not acc.is_zero():
                    alpha_k[w1_idx] = acc
            alpha[ckey] = alpha_k

            z2_blps = grouped.group_z2_blps[ckey]
            beta_k: Dict[int, BivariateLaurentPoly] = {}
            for w2_idx in range(n2):
                flats_above = lattice_2.flats_above_idx(w2_idx)
                acc = BivariateLaurentPoly.zero()
                for z2_idx in flats_above:
                    mu = lattice_2._compute_mobius(w2_idx, z2_idx)
                    if mu == 0:
                        continue
                    r_z2 = z2_blps.get(z2_idx)
                    if r_z2 is None:
                        continue
                    term = z2_weight[z2_idx] * r_z2
                    if mu != 1:
                        term = mu * term
                    acc = acc + term
                if not acc.is_zero():
                    beta_k[w2_idx] = acc
            beta[ckey] = beta_k

        return alpha, beta

    alpha_1, beta_1 = _precompute_alpha_beta(grouped_1)
    t1 = time.time()
    print(f"[T6-grouped] Cell 1 α/β: {len(alpha_1)} groups, {t1-t0:.1f}s", flush=True)

    alpha_2, beta_2 = _precompute_alpha_beta(grouped_2)
    t2 = time.time()
    print(f"[T6-grouped] Cell 2 α/β: {len(alpha_2)} groups, {t2-t1:.1f}s", flush=True)

    # Build list of active group keys for fast iteration
    keys_1 = list(grouped_1.z1_groups.keys())
    keys_2 = list(grouped_2.z1_groups.keys())

    # For each W1, precompute which groups have nonzero α
    w1_active_1: Dict[int, List[Tuple[str, BivariateLaurentPoly]]] = {}
    for ckey in keys_1:
        for w1_idx, blp in alpha_1[ckey].items():
            if w1_idx not in w1_active_1:
                w1_active_1[w1_idx] = []
            w1_active_1[w1_idx].append((ckey, blp))

    w1_active_2: Dict[int, List[Tuple[str, BivariateLaurentPoly]]] = {}
    for ckey in keys_2:
        for w1_idx, blp in alpha_2[ckey].items():
            if w1_idx not in w1_active_2:
                w1_active_2[w1_idx] = []
            w1_active_2[w1_idx].append((ckey, blp))

    # Similarly for W2
    w2_active_1: Dict[int, List[Tuple[str, BivariateLaurentPoly]]] = {}
    for ckey in keys_1:
        for w2_idx, blp in beta_1[ckey].items():
            if w2_idx not in w2_active_1:
                w2_active_1[w2_idx] = []
            w2_active_1[w2_idx].append((ckey, blp))

    w2_active_2: Dict[int, List[Tuple[str, BivariateLaurentPoly]]] = {}
    for ckey in keys_2:
        for w2_idx, blp in beta_2[ckey].items():
            if w2_idx not in w2_active_2:
                w2_active_2[w2_idx] = []
            w2_active_2[w2_idx].append((ckey, blp))

    t3 = time.time()
    print(f"[T6-grouped] Active group indices built in {t3-t2:.1f}s", flush=True)

    # Count contributing pairs for progress estimation
    contributing_w1 = 0
    for w1_idx in range(n1):
        if w1_idx in w1_active_1 and w1_idx in w1_active_2:
            contributing_w1 += 1
    print(f"[T6-grouped] Contributing W1: {contributing_w1}/{n1}", flush=True)

    # Row-by-row Theorem 6: for each W1, compute inner W2 sum, then accumulate
    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()
    processed_w1 = 0
    contributing_pairs = 0

    for w1_idx in range(n1):
        active_1 = w1_active_1.get(w1_idx)
        active_2 = w1_active_2.get(w1_idx)
        if not active_1 or not active_2:
            continue

        w1_flat = lattice_1.flat_by_idx(w1_idx)
        w1_size = len(w1_flat)

        # For this W1, precompute the α products indexed by (ckey_1, ckey_2)
        # g_i(W1,W2) = Σ_k α_i_k(W1) · β_i_k(W2)
        # We need: g1·g2 = [Σ_k a1_k · b1_k(W2)] · [Σ_j a2_j · b2_j(W2)]
        # where a1_k = α1_k(W1) (fixed for this W1), b1_k(W2) = β1_k(W2)

        # Inner W2 sum: Σ_{W2} g1(W1,W2)·g2(W1,W2) / d2(W2)
        inner_n = BivariateLaurentPoly.zero()
        inner_d = BivariateLaurentPoly.one()
        inner_count = 0

        for w2_idx in range(n2):
            b1_list = w2_active_1.get(w2_idx)
            b2_list = w2_active_2.get(w2_idx)
            if not b1_list or not b2_list:
                continue

            # g1 = Σ_k α1_k(W1) · β1_k(W2) for active groups
            g1 = BivariateLaurentPoly.zero()
            for ckey, beta_val in b1_list:
                # Look up α1_k(W1) — need to match by ckey
                alpha_dict = alpha_1.get(ckey)
                if alpha_dict is None:
                    continue
                alpha_val = alpha_dict.get(w1_idx)
                if alpha_val is None:
                    continue
                g1 = g1 + alpha_val * beta_val

            if g1.is_zero():
                continue

            g2 = BivariateLaurentPoly.zero()
            for ckey, beta_val in b2_list:
                alpha_dict = alpha_2.get(ckey)
                if alpha_dict is None:
                    continue
                alpha_val = alpha_dict.get(w1_idx)
                if alpha_val is None:
                    continue
                g2 = g2 + alpha_val * beta_val

            if g2.is_zero():
                continue

            w2_flat = lattice_2.flat_by_idx(w2_idx)
            w2_size = len(w2_flat)
            denom_w2 = _y_power_in_uv(w2_size) * chi_cache_2[w2_idx]

            numer_w2 = g1 * g2
            inner_n = inner_n * denom_w2 + numer_w2 * inner_d
            inner_d = inner_d * denom_w2

            inner_count += 1

            # Periodic simplification of inner fraction
            if inner_count % 20 == 0:
                q, r = inner_n.divmod(inner_d)
                if r.is_zero():
                    inner_n = q
                    inner_d = BivariateLaurentPoly.one()

        if inner_n.is_zero():
            processed_w1 += 1
            continue

        contributing_pairs += inner_count

        # Outer: accumulate inner_n/(inner_d · d1(W1)) into result
        denom_w1 = _y_power_in_uv(w1_size) * chi_cache_1[w1_idx]
        outer_d_term = inner_d * denom_w1
        result_n = result_n * outer_d_term + inner_n * result_d
        result_d = result_d * outer_d_term

        processed_w1 += 1

        # Periodic simplification of outer fraction
        if processed_w1 % 10 == 0:
            q, r = result_n.divmod(result_d)
            if r.is_zero():
                result_n = q
                result_d = BivariateLaurentPoly.one()

            if processed_w1 % 100 == 0:
                t_now = time.time()
                result_size = len(result_n.coeffs)
                denom_size = len(result_d.coeffs)
                print(f"[T6-grouped] W1 {processed_w1}/{contributing_w1}: "
                      f"{contributing_pairs} W2 pairs, "
                      f"numer={result_size} terms, denom={denom_size} terms, "
                      f"{t_now-t3:.1f}s", flush=True)

    t4 = time.time()
    print(f"[T6-grouped] Accumulation done: {contributing_pairs} total pairs, "
          f"{t4-t3:.1f}s", flush=True)

    # Final: v^r_N * result_n / result_d
    v_rN = BivariateLaurentPoly({(0, r_N): 1})
    final_n = v_rN * result_n
    result = final_n // result_d
    return result.to_tutte_poly()
