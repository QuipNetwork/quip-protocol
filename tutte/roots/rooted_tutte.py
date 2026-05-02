"""Rooted Tutte polynomial primitives.

T_rooted(G, S)[P] = Σ over spanning subgraphs A of G of:
                    (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}
                   where A's component-partition restricted to S = P

Standard Tutte: T(G) = Σ_P T_rooted(G, S)[P].

These are the primitives extracted from
`tutte/scripts/algebraic_rooted_tutte.py` (Phase 17.E.9 research) needed
for cell-quotient cycle DP. Brute-force computation is exponential in
|E|, fine for small cells (≤ 16 edges) like K_{4,4}.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..graph import Graph
from ..polynomial import TuttePolynomial


def _power_poly(p: TuttePolynomial, k: int) -> TuttePolynomial:
    if k < 0:
        raise ValueError(f"Negative power: {k}")
    result = TuttePolynomial.one()
    for _ in range(k):
        result = result * p
    return result


def _find(parent: Dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _normalize_partition(part: Dict[int, Set[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Canonical key: sorted tuple of sorted-tuples."""
    return tuple(sorted(tuple(sorted(b)) for b in part.values()))


def t_rooted_bruteforce(
    graph: Graph,
    boundary: List[int],
) -> Dict[Tuple[Tuple[int, ...], ...], TuttePolynomial]:
    """Brute-force compute T_rooted(graph, boundary).

    Returns dict from canonical partition of boundary → polynomial.
    Uses rank-nullity: weight(A) = (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}.

    Cost: O(2^|E| × |V|). Fine for cells with |E| ≤ 16 (K_{4,4} = 16 edges).
    """
    edges = sorted(graph.edges)
    n_edges = len(edges)
    nodes = sorted(graph.nodes)
    n_nodes = len(nodes)

    full_dsu = {v: v for v in nodes}
    for u, v in edges:
        ru, rv = _find(full_dsu, u), _find(full_dsu, v)
        if ru != rv:
            full_dsu[max(ru, rv)] = min(ru, rv)
    k_G = len({_find(full_dsu, v) for v in nodes})
    r_E = n_nodes - k_G

    x_minus_1 = TuttePolynomial.x() + (-1) * TuttePolynomial.one()
    y_minus_1 = TuttePolynomial.y() + (-1) * TuttePolynomial.one()

    result: Dict[Tuple, TuttePolynomial] = defaultdict(lambda: TuttePolynomial.zero())

    for mask in range(2 ** n_edges):
        A_edges = [edges[i] for i in range(n_edges) if (mask >> i) & 1]
        dsu = {v: v for v in nodes}
        for u, v in A_edges:
            ru, rv = _find(dsu, u), _find(dsu, v)
            if ru != rv:
                dsu[max(ru, rv)] = min(ru, rv)
        k_A = len({_find(dsu, v) for v in nodes})
        r_A = n_nodes - k_A
        boundary_parts: Dict[int, Set[int]] = defaultdict(set)
        for v in boundary:
            boundary_parts[_find(dsu, v)].add(v)
        partition_key = _normalize_partition(boundary_parts)
        weight = _power_poly(x_minus_1, r_E - r_A) * _power_poly(y_minus_1, len(A_edges) - r_A)
        result[partition_key] = result[partition_key] + weight

    return dict(result)


def all_partitions(elements: List[int]) -> List[List[Set[int]]]:
    """All set-partitions of elements."""
    if not elements:
        return [[]]
    if len(elements) == 1:
        return [[{elements[0]}]]
    result = []
    first = elements[0]
    for sub in all_partitions(elements[1:]):
        result.append([{first}] + [set(b) for b in sub])
        for i in range(len(sub)):
            new_sub = [set(b) for b in sub]
            new_sub[i] = new_sub[i] | {first}
            result.append(new_sub)
    return result


def join_partitions(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    universe: List[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Compute join (transitive closure) of P1 and P2 over universe.

    Two elements are in the same block of the join iff there's a path
    of "same-block" relations through P1 ∪ P2.
    """
    parent = {v: v for v in universe}
    for blocks in [P1, P2]:
        for block in blocks:
            if len(block) <= 1:
                continue
            rep = block[0]
            for v in block[1:]:
                ru, rv = _find(parent, rep), _find(parent, v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
    out: Dict[int, Set[int]] = defaultdict(set)
    for v in universe:
        out[_find(parent, v)].add(v)
    return _normalize_partition(out)


def delta(
    P1: Tuple[Tuple[int, ...], ...],
    P2: Tuple[Tuple[int, ...], ...],
    shared_boundary: List[int],
) -> int:
    """DELTA(P_1, P_2) = nblocks(JOIN(P_1, P_2)) + |S| - nblocks(P_1) - nblocks(P_2).

    Rank deficit when components merge across the shared boundary in vertex-sum.
    """
    join = join_partitions(P1, P2, shared_boundary)
    return len(join) + len(shared_boundary) - len(P1) - len(P2)


def restrict_partition(
    P: Tuple[Tuple[int, ...], ...],
    subset: List[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Restrict partition P to subset; isolated vertices added as singletons."""
    subset_set = set(subset)
    blocks: List[Tuple[int, ...]] = []
    seen: set = set()
    for block in P:
        intersection = tuple(sorted(v for v in block if v in subset_set))
        if intersection:
            blocks.append(intersection)
            seen.update(intersection)
    for v in subset:
        if v not in seen:
            blocks.append((v,))
    return tuple(sorted(blocks))


def divide_by_x_minus_1_power(
    poly: TuttePolynomial, k: int,
) -> TuttePolynomial:
    """Divide polynomial by (x-1)^k via repeated synthetic division.

    Raises ValueError if poly is not divisible (remainder != 0).
    """
    if k == 0:
        return poly
    coeffs: Dict[Tuple[int, int], int] = {}
    for i, j, c in poly.terms():
        coeffs[(i, j)] = c
    if not coeffs:
        return TuttePolynomial.zero()
    for _ in range(k):
        new_coeffs: Dict[Tuple[int, int], int] = {}
        y_groups: Dict[int, Dict[int, int]] = defaultdict(dict)
        for (i, j), c in coeffs.items():
            y_groups[j][i] = c
        for j, x_coeffs in y_groups.items():
            max_i = max(x_coeffs.keys())
            running = 0
            q_coeffs: Dict[int, int] = {}
            for i in range(max_i, -1, -1):
                ci = x_coeffs.get(i, 0)
                running += ci
                if i > 0:
                    q_coeffs[i - 1] = running
            remainder = running
            if remainder != 0:
                raise ValueError(
                    f"Polynomial not divisible by (x-1) at y^{j}: "
                    f"remainder={remainder}"
                )
            for i_new, c_new in q_coeffs.items():
                if c_new != 0:
                    new_coeffs[(i_new, j)] = c_new
        coeffs = new_coeffs
    return TuttePolynomial.from_coefficients(coeffs)


_T_ROOTED_CACHE: Dict[Tuple[str, Tuple[int, ...]], Dict[Tuple, TuttePolynomial]] = {}


def t_rooted_cached(graph: Graph, boundary: List[int]) -> Dict[Tuple, TuttePolynomial]:
    """Like t_rooted_bruteforce, cached by (canonical_key, sorted_boundary)."""
    key = (graph.canonical_key(), tuple(sorted(boundary)))
    if key in _T_ROOTED_CACHE:
        return _T_ROOTED_CACHE[key]
    val = t_rooted_bruteforce(graph, boundary)
    _T_ROOTED_CACHE[key] = val
    return val


def relabel_partition_dict(
    T_dict: Dict[Tuple, TuttePolynomial],
    label_map: Dict[int, int],
) -> Dict[Tuple, TuttePolynomial]:
    """Apply label_map to partition keys; merge values for collisions."""
    new_T: Dict[Tuple, TuttePolynomial] = {}
    for P, val in T_dict.items():
        new_P = tuple(sorted(
            tuple(sorted(label_map.get(v, v) for v in block))
            for block in P
        ))
        new_T[new_P] = new_T.get(new_P, TuttePolynomial.zero()) + val
    return new_T
