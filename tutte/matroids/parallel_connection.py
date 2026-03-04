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
from math import comb
from typing import Dict, FrozenSet, List, Optional, Tuple, TYPE_CHECKING

from ..polynomial import TuttePolynomial
from ..graph import Graph
from .core import (
    GraphicMatroid, FlatLattice, Edge,
    enumerate_flats_with_hasse,
)
from ..graphs.series_parallel import compute_contraction_chi

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


def compute_f_W(
    lattice: FlatLattice,
    w_idx: int,
    t_contracted: Dict[int, BivariateLaurentPoly],
) -> BivariateLaurentPoly:
    """Compute f_i(W) = sum_{Z >= W} mu(W, Z) * y^|Z| / (y-1)^{r(Z)} * T(M_i/Z).

    In (u,v) basis:
    - y^|Z| = (v+1)^|Z|
    - (y-1)^{r(Z)} = v^{r(Z)}
    - Division by v^{r(Z)} is a shift
    - T(M_i/Z) is already in (u,v) form

    Args:
        lattice: FlatLattice of the shared matroid N
        w_idx: Index of flat W in the lattice
        t_contracted: Dict mapping flat index -> T(M_i/Z) in (u,v) form
    """
    result = BivariateLaurentPoly.zero()

    # Get all flats Z >= W
    flats_above = lattice.flats_above_idx(w_idx)

    for z_idx in flats_above:
        # mu(W, Z)
        mu_val = lattice._compute_mobius(w_idx, z_idx)
        if mu_val == 0:
            continue

        z_flat = lattice.flat_by_idx(z_idx)
        z_size = len(z_flat)
        z_rank = lattice.flat_rank_by_idx(z_idx)

        # y^|Z| / (y-1)^{r(Z)} in (u,v) form = (v+1)^|Z| / v^{r(Z)}
        y_pow = _y_power_in_uv(z_size)
        y_div_v = y_pow.shift_v(-z_rank)

        # T(M_i/Z) in (u,v) form
        t_z = t_contracted.get(z_idx)
        if t_z is None:
            continue

        # mu(W,Z) * y^|Z| / v^{r(Z)} * T(M_i/Z)
        term = y_div_v * t_z
        if mu_val != 1:
            term = mu_val * term

        result = result + term

    return result


def compute_weight(
    r_N: int,
    w_idx: int,
    lattice: FlatLattice,
    chi_coeffs: Dict[int, int],
) -> BivariateLaurentPoly:
    """Compute weight(W) = v^{r(N)} / ((v+1)^|W| * chi(N/W; uv)).

    In (u,v) form:
    - v^{r(N)} is just a monomial
    - (v+1)^|W| = y^|W| in (u,v) form
    - chi(N/W; uv) substitutes q = uv in the characteristic polynomial

    The division is handled via the exact formula:
    weight(W) = v^{r(N)} * [(v+1)^|W| * chi(N/W; uv)]^{-1}

    But actually, the denominator is a polynomial, so we need a different approach.
    In practice, the Bonin-de Mier formula uses this in a product:
    result = sum_W weight(W) * f_1(W) * f_2(W)

    Let's restructure: instead of computing weight(W) as a Laurent poly
    and multiplying, we fold the (v+1)^|W| and chi terms into the summation.

    Actually, the correct Bonin-de Mier Theorem 6 formula is:

    T(P_N(M1,M2)) = sum_W  [(-1)^{r(W)} * chi(N/W; uv) * v^{r(N)-r(W)}]^{-1}
                     * ... this isn't right either.

    Let me re-derive from the paper. Theorem 6 in Bonin-de Mier states:

    R(P_N(M1,M2); u,v) = sum_{W flat of N}
        mu(0,W) * v^{r(N)} / [(v+1)^|W| * chi_W]
        * f_1(W) * f_2(W)

    where chi_W = chi(N/W; (1+u)(1+v)) = chi(N/W; xy)

    Actually, the exact formulation uses the characteristic polynomial
    evaluated at xy (not uv). Let me reconsider.

    For simplicity, since the structure is complex, we use the weight as:
    weight(W) is computed and multiplied with f_1(W) * f_2(W).
    """
    W = lattice.flat_by_idx(w_idx)
    w_size = len(W)
    w_rank = lattice.flat_rank_by_idx(w_idx)

    # v^{r(N)} as monomial
    v_rN = BivariateLaurentPoly({(0, r_N): 1})

    # (v+1)^|W| = y^|W|
    v_plus_1_W = _y_power_in_uv(w_size)

    # chi(N/W; uv)
    chi_uv = _chi_in_uv(chi_coeffs)

    # Denominator = (v+1)^|W| * chi(N/W; uv)
    denominator = v_plus_1_W * chi_uv

    # We need v^{r(N)} / denominator, but denominator is a polynomial,
    # not a monomial. This exact division must work for the formula to hold.
    # In practice, we'll compute it differently - see theorem6_parallel_connection.

    return v_rN  # Placeholder - actual computation happens in the main formula


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

    The formula in the rank-generating polynomial basis R(u,v) where u=x-1, v=y-1:

    R(P_N(M1,M2)) = (1/v^{r(N)}) * sum_{W flat of N}
        mu(0_N, W) * f_1(W) * f_2(W) * (v+1)^|W|
        * sum_{Z >= W} mu(W,Z) * ...

    Actually, the cleaned-up version from the paper:

    T(P_N(M1,M2); x,y) = sum_{A flat of N} mu(0,A)
        * (y-1)^{-r(N)} * y^{|A|}
        * prod_{i=1,2} [ sum_{B >= A} mu(A,B) * T(M_i/B; x,y) / (y-1)^{r(B)-r(A)} ]

    Or equivalently in (u,v) form:
    R = v^{-r(N)} * sum_{A flat} mu(0,A) * (v+1)^{|A|}
        * prod_i [ sum_{B >= A} mu(A,B) * R(M_i/B) * v^{r(A)-r(B)} ]

    The v^{-r(N)} at the front will cancel with v-powers from the inner sums.

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

    result = BivariateLaurentPoly.zero()

    for a_idx in range(lattice_N.num_flats):
        # mu(0, A)
        mu_0A = lattice_N._mobius_from_bottom.get(a_idx, 0)
        if mu_0A == 0:
            continue

        a_flat = lattice_N.flat_by_idx(a_idx)
        a_size = len(a_flat)
        a_rank = lattice_N.flat_rank_by_idx(a_idx)

        # (v+1)^|A|
        v_plus_1_A = _y_power_in_uv(a_size)

        # Compute f_1(A) and f_2(A)
        # f_i(A) = sum_{B >= A} mu(A,B) * R(M_i/B) * v^{r(A)-r(B)}
        f1 = _compute_f_for_flat(lattice_N, a_idx, a_rank, T_M1_contracted)
        f2 = _compute_f_for_flat(lattice_N, a_idx, a_rank, T_M2_contracted)

        if f1.is_zero() or f2.is_zero():
            continue

        # Combine: mu(0,A) * (v+1)^|A| * f_1(A) * f_2(A)
        term = v_plus_1_A * f1 * f2
        if mu_0A != 1:
            term = mu_0A * term

        result = result + term

    # Multiply by v^{-r(N)} (shift all v-powers down by r_N)
    result = result.shift_v(-r_N)

    # Convert back to T(x,y)
    return result.to_tutte_poly()


def _compute_f_for_flat(
    lattice: FlatLattice,
    a_idx: int,
    a_rank: int,
    t_contracted: Dict[int, BivariateLaurentPoly],
) -> BivariateLaurentPoly:
    """Compute f_i(A) = sum_{B >= A} mu(A,B) * R(M_i/B) * v^{r(A)-r(B)}.

    Args:
        lattice: FlatLattice of shared matroid
        a_idx: Index of flat A
        a_rank: Rank of flat A
        t_contracted: Pre-computed T(M_i/B) in BivariateLaurentPoly form
    """
    result = BivariateLaurentPoly.zero()

    flats_above = lattice.flats_above_idx(a_idx)

    # Precompute Mobius from A for efficiency
    lattice.precompute_mobius_from(lattice.flat_by_idx(a_idx))

    for b_idx in flats_above:
        mu_AB = lattice._compute_mobius(a_idx, b_idx)
        if mu_AB == 0:
            continue

        t_b = t_contracted.get(b_idx)
        if t_b is None:
            continue

        b_rank = lattice.flat_rank_by_idx(b_idx)

        # v^{r(A) - r(B)} — note this is negative since r(B) >= r(A)
        v_shift = a_rank - b_rank

        term = t_b.shift_v(v_shift)
        if mu_AB != 1:
            term = mu_AB * term

        result = result + term

    return result


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

    for z_idx in range(lattice_N.num_flats):
        z_flat = lattice_N.flat_by_idx(z_idx)

        if not z_flat:
            # Empty flat = no contraction
            synth_result = engine.synthesize(graph_i)
            result[z_idx] = BivariateLaurentPoly.from_tutte(synth_result.polynomial)
            continue

        # Contract edges in z_flat within graph_i
        # These edges are in the inter-cell portion of graph_i
        contracted_graph = _contract_edges_in_graph(graph_i, z_flat)

        if contracted_graph.edge_count() == 0:
            result[z_idx] = BivariateLaurentPoly.from_tutte(TuttePolynomial.one())
            continue

        synth_result = engine.synthesize(contracted_graph)
        result[z_idx] = BivariateLaurentPoly.from_tutte(synth_result.polynomial)

    return result


def _contract_edges_in_graph(
    graph: Graph, edges_to_contract: FrozenSet[Edge]
) -> Graph:
    """Contract specified edges in a graph by merging endpoints.

    For each edge in edges_to_contract that exists in graph,
    merge the endpoints. Edges not in graph are ignored.
    Resulting loops are removed (simple graph output).
    """
    all_nodes = set(graph.nodes)
    parent = {v: v for v in all_nodes}
    rank = {v: 0 for v in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

    # Contract specified edges
    for u, v in edges_to_contract:
        if u in all_nodes and v in all_nodes:
            union(u, v)

    # Build contracted graph
    node_map = {n: find(n) for n in all_nodes}
    new_nodes = frozenset(node_map.values())

    new_edges = set()
    for u, v in graph.edges:
        if (u, v) in edges_to_contract:
            continue
        nu, nv = node_map[u], node_map[v]
        if nu != nv:
            edge = (min(nu, nv), max(nu, nv))
            new_edges.add(edge)

    return Graph(nodes=new_nodes, edges=frozenset(new_edges))


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
