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
from itertools import combinations
from math import comb
from typing import Dict, FrozenSet, List, Optional, Tuple, TYPE_CHECKING

from ..polynomial import TuttePolynomial
from ..graph import Graph, MultiGraph
from .core import (
    GraphicMatroid, FlatLattice, Edge,
    enumerate_flats_with_hasse,
)

if TYPE_CHECKING:
    from ..synthesis.engine import SynthesisEngine


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
    # Precompute Mobius from bottom for all flats
    lattice_N.precompute_all_mobius_from_bottom()

    # Common-denominator accumulation: result = result_n / result_d
    result_n = BivariateLaurentPoly.zero()
    result_d = BivariateLaurentPoly.one()

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

    # Final: v^{r(N)} * result_n / result_d
    v_rN = BivariateLaurentPoly({(0, r_N): 1})
    final_n = v_rN * result_n

    # Exact division (guaranteed by theory)
    result = final_n // result_d

    # Convert back to T(x,y)
    return result.to_tutte_poly()


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

    if len(components) != 2:
        raise ValueError(f"Expected 2 components after removing separator, got {len(components)}")

    cell1_nodes = components[0] | sep_set
    cell2_nodes = components[1] | sep_set

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

    # Build extended cell graphs (cell + clique edges touching it)
    # Since all separator vertices are in both cells, all clique edges touch both cells
    ext1, shared1 = build_extended_cell_graph(pc_graph, cell1_nodes, clique_edges)
    ext2, shared2 = build_extended_cell_graph(pc_graph, cell2_nodes, clique_edges)

    # Precompute T(M_i/Z) for all flats Z of N
    t_m1 = precompute_contractions(ext1, shared1, lattice_N, engine)
    t_m2 = precompute_contractions(ext2, shared2, lattice_N, engine)

    # Precompute all Mobius values
    lattice_N.precompute_all_mobius_from_bottom()

    # For each flat F, compute coefficient and T(PC/F)
    result_uv = BivariateLaurentPoly.zero()

    for f_idx in range(lattice_N.num_flats):
        f_flat = lattice_N.flat_by_idx(f_idx)
        f_rank = lattice_N.flat_rank_by_idx(f_idx)

        # Compute flat coefficient
        coeff = _compute_flat_coefficient(matroid_N, lattice_N, f_idx)
        if coeff.is_zero():
            continue

        # Compute T(PC/F) via Theorem 6 on the interval [F, top]
        t_pc_f = _theorem6_for_contraction(lattice_N, f_idx, t_m1, t_m2, r_N)

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

    coeff(F) = Σ_{S ⊆ T : cl(S) = F} (-1)^|S| · (y-1)^{nullity(S)}

    Since cl(S) = F implies r(S) = r(F) for all such S (S spans the flat F),
    the nullity is |S| - r(F). So:

    coeff(F) = Σ_{S ⊆ F : cl(S) = F} (-1)^|S| · (y-1)^{|S| - r(F)}

    In the uv-basis (v = y-1):

    coeff(F) = Σ_{S ⊆ F : cl(S) = F} (-1)^|S| · v^{|S| - r(F)}

    The count of subsets S ⊆ F with cl(S) = F and |S| = j is computed via
    Mobius inversion over the lattice interval [bottom, F]:

    #{S ⊆ F : cl(S) = F, |S| = j} = C(|F|, j) - Σ_{G < F} #{S ⊆ G : cl(S) = G, |S| = j}

    But we can compute this more directly: the number of subsets of F with
    closure exactly F is obtained by inclusion-exclusion over proper subflats.

    #{S ⊆ F : cl(S) = F} = Σ_{G ≤ F} μ(G, F) · 2^{|G|}

    So coeff(F) = Σ_{G ≤ F} μ(G, F) · Σ_{j=0}^{|G|} C(|G|,j) · (-1)^j · v^{j - r(F)}

    = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · Σ_{j=0}^{|G|} C(|G|,j) · (-v)^j · (-1)^0

    Wait, let's be more careful. We need subsets S of F with cl(S) = F.

    Using Mobius inversion on the lattice:
    #{S ⊆ F : cl(S) = F} = Σ_{G ≤ F} μ(G, F) · 2^{|G|}

    For the weighted version with (-1)^|S| · v^{|S| - r(F)}:
    coeff(F) = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · Σ_{j} C(|G|,j)(-1)^j v^j
             = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · (1-v)^{|G|}
             = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · (2-y)^{|G|}

    In u,v basis: (1-v) is just the constant 2-y = 2-(v+1) = 1-v.
    So (1-v)^n = Σ_j C(n,j)(-v)^j = Σ_j C(n,j)(-1)^j v^j.

    Args:
        matroid_N: The shared matroid
        lattice_N: Flat lattice of N
        f_idx: Index of flat F

    Returns:
        BivariateLaurentPoly representing the coefficient
    """
    f_flat = lattice_N.flat_by_idx(f_idx)
    f_rank = lattice_N.flat_rank_by_idx(f_idx)

    # coeff(F) = v^{-r(F)} · Σ_{G ≤ F} μ(G, F) · (1-v)^{|G|}
    # We iterate over all flats G ≤ F

    # Precompute Mobius from all flats below F to F
    # We need mu(G, F) for G <= F
    # Use the interval: iterate over flats with rank <= r(F) that are subsets of F
    flats_below_f = []
    for g_idx in range(lattice_N.num_flats):
        g_flat = lattice_N.flat_by_idx(g_idx)
        if g_flat.issubset(f_flat):
            flats_below_f.append(g_idx)

    # Precompute Mobius values mu(G, F) for all G <= F
    # We need to compute these via the lattice
    accumulator = BivariateLaurentPoly.zero()

    for g_idx in flats_below_f:
        mu_gf = lattice_N._compute_mobius(g_idx, f_idx)
        if mu_gf == 0:
            continue

        g_flat = lattice_N.flat_by_idx(g_idx)
        g_size = len(g_flat)

        # (1-v)^{|G|} = Σ_{j=0}^{|G|} C(|G|,j) (-1)^j v^j
        one_minus_v_pow: Dict[Tuple[int, int], int] = {}
        for j in range(g_size + 1):
            coeff_val = comb(g_size, j) * ((-1) ** j)
            if coeff_val != 0:
                one_minus_v_pow[(0, j)] = coeff_val

        term = BivariateLaurentPoly(one_minus_v_pow)
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
    result: Dict[int, BivariateLaurentPoly] = {}
    # Cache by canonical key to avoid redundant synthesis
    canon_cache: Dict[str, BivariateLaurentPoly] = {}

    for z_idx in range(lattice_N.num_flats):
        z_flat = lattice_N.flat_by_idx(z_idx)

        if not z_flat:
            # Empty flat = no contraction
            synth_result = engine.synthesize(graph_i)
            result[z_idx] = BivariateLaurentPoly.from_tutte(synth_result.polynomial)
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
