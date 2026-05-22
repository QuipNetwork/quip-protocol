"""Named-family atom detection for chord-peel decompositions.

This module finds disjoint occurrences of named-family graph atoms
(K_n cliques, K_{a,b} complete bipartite, wheels, books, ...) inside
an input graph, for the cross-cell chord-peel dispatch in the
synthesis engine.

The cross-cell principle: when a graph G has two disjoint dense atoms
A_1, A_2 connected by a small bipartite junction J, peeling J's edges
via the chord rule is much faster than peeling internal-atom edges
because:
  - chord rule's per-step sub-synth cost is roughly constant for
    same-density graphs;
  - fewer chord edges = fewer expensive sub-syntheses;
  - g_chord_free still contains the dense atoms, so recursion can find
    more structure.

Empirical: Z(1,2) (2 K_4 atoms + 2 K_{2,2} junctions) went from 88s to
47s by peeling 4 junction edges instead of 12 internal K_4 edges.

This module generalizes from K_n atoms to other named families. The
K_{a,b} family is particularly important because D-Wave Chimera Cm cells
are K_{4,4}, and many structured data graphs (bipartite, recommendation,
knowledge) have K_{a,b}-rich substructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import networkx as nx

from ..graph import Graph


@dataclass(frozen=True)
class Atom:
    """A named-family subgraph occurrence in a host graph.

    family: short name like "K_4", "K_{3,3}", "W_5", "B_3"
    vertices: the host-graph vertex set induced by this atom
    """

    family: str
    vertices: FrozenSet[int]

    def __len__(self) -> int:
        return len(self.vertices)


# ---------------------------------------------------------------------------
# K_n (complete graph) atoms
# ---------------------------------------------------------------------------

def find_disjoint_kn_atoms(
    graph: Graph,
    nxg: Optional[nx.Graph] = None,
    min_k: int = 3,
    max_k: int = 6,
) -> List[Atom]:
    """Find ≥2 disjoint K_k atoms, preferring larger k.

    Returns the largest-k family with ≥2 disjoint occurrences. Empty list
    when no such family exists.

    Algorithm: enumerate maximal cliques (networkx find_cliques) then
    extract K_k-sized atoms (or sub-K_k of a larger maximal clique),
    greedily select disjoint set.
    """
    if nxg is None:
        nxg = _to_nx(graph)
    all_cliques = list(nx.find_cliques(nxg))
    for k in range(max_k, min_k - 1, -1):
        k_cliques: List[FrozenSet[int]] = []
        seen: set = set()
        for c in all_cliques:
            if len(c) >= k:
                if len(c) == k:
                    fc = frozenset(c)
                    if fc not in seen:
                        seen.add(fc)
                        k_cliques.append(fc)
                else:
                    # One canonical sub-K_k per parent clique
                    for sub in combinations(sorted(c), k):
                        fc = frozenset(sub)
                        if fc not in seen:
                            seen.add(fc)
                            k_cliques.append(fc)
                            break
        atoms = _greedy_disjoint(k_cliques)
        if len(atoms) >= 2:
            return [Atom(family=f"K_{k}", vertices=a) for a in atoms]
    return []


# ---------------------------------------------------------------------------
# K_{a,b} (complete bipartite) atoms
# ---------------------------------------------------------------------------

def find_disjoint_kab_atoms(
    graph: Graph,
    nxg: Optional[nx.Graph] = None,
    ab_pairs: Optional[List[Tuple[int, int]]] = None,
) -> List[Atom]:
    """Find ≥2 disjoint K_{a,b} atoms, preferring largest (a+b).

    ab_pairs defaults to common small bicliques:
      [(2,2), (2,3), (3,3), (2,4), (3,4), (4,4)]

    For each (a, b) with a ≤ b, finds candidate biclique vertex pairs
    via shared-neighborhood intersection (faster than full VF2):
      pick degree-≥-a vertex u; for each (a-1)-subset N_a-1 of N(u),
      the candidate A-side is N_a-1 ∪ {u}; the candidate B-side is
      ∩_{v ∈ A} N(v); if |B-side| ≥ b, we have a K_{a,b}.

    Greedily picks disjoint atoms.
    """
    if nxg is None:
        nxg = _to_nx(graph)
    if ab_pairs is None:
        ab_pairs = [(4, 4), (3, 4), (3, 3), (2, 4), (2, 3), (2, 2)]
    # Try pairs in given order (caller chooses preference)
    for (a, b) in ab_pairs:
        if a > b:
            a, b = b, a
        candidates = _enumerate_kab(nxg, a, b)
        atoms = _greedy_disjoint(candidates)
        if len(atoms) >= 2:
            return [Atom(family=f"K_{{{a},{b}}}", vertices=v) for v in atoms]
    return []


def _enumerate_kab(nxg: nx.Graph, a: int, b: int) -> List[FrozenSet[int]]:
    """Enumerate K_{a,b} occurrences via degree-pruned intersection.

    Returns one FrozenSet per K_{a,b} occurrence (deduplicated by
    vertex set; doesn't double-count A/B swaps for a=b case).
    """
    seen: set = set()
    out: List[FrozenSet[int]] = []
    nodes = sorted(nxg.nodes())
    # Build adjacency sets once
    adj = {n: frozenset(nxg.neighbors(n)) for n in nodes}
    # Iterate A-side vertex sets of size a
    # Prune by minimum degree b
    a_candidates = [n for n in nodes if len(adj[n]) >= b]
    if len(a_candidates) < a:
        return []
    # For each a-subset of a_candidates, compute common neighborhood.
    # This is C(|a_candidates|, a) which can be large; cap to keep it
    # tractable. For typical D-Wave-sized graphs (<100 vertices) and
    # small a (≤4), this is manageable.
    if len(a_candidates) > 80:
        # Too many — skip to avoid combinatorial blowup. Caller should
        # try larger a/b pairs only on graphs where such atoms exist
        # in moderate count.
        return []
    for a_set in combinations(a_candidates, a):
        common = adj[a_set[0]]
        for v in a_set[1:]:
            common = common & adj[v]
            if len(common) < b:
                break
        if len(common) < b:
            continue
        # Exclude A-side vertices from common (no self-loops in
        # the biclique — A and B sides are disjoint)
        common_b = common - frozenset(a_set)
        if len(common_b) < b:
            continue
        # Pick b vertices from common_b. To keep ONE canonical
        # K_{a,b} per (a_set, b_subset), iterate the lex first b.
        # (Multiple b-subsets give multiple atoms; for cross-cell
        # peel we mostly care that the atom EXISTS and is disjoint.)
        for b_set in combinations(sorted(common_b), b):
            atom = frozenset(a_set) | frozenset(b_set)
            if atom not in seen:
                seen.add(atom)
                out.append(atom)
            # For a == b case, also dedupe by full vertex set (above
            # frozenset already handles); for a < b case there's no
            # double-count risk because A and B sizes differ.
            break  # one canonical b-subset per a-set
    return out


# ---------------------------------------------------------------------------
# B_n (book) atoms — n triangles sharing a common edge
# ---------------------------------------------------------------------------

def find_disjoint_book_atoms(
    graph: Graph,
    nxg: Optional[nx.Graph] = None,
    min_pages: int = 2,
    max_pages: int = 5,
) -> List[Atom]:
    """Find ≥2 disjoint B_n book atoms, preferring larger n.

    B_n = n triangles sharing a common edge. Vertex set is {u, v, p_1,
    …, p_n} where (u, v) is the shared edge and each p_i is a triangle
    page (connected to both u and v).

    Detection: for each edge (u, v), enumerate common neighbors of u
    and v; if ≥n exist, pick the lex-smallest n to form the atom.
    """
    if nxg is None:
        nxg = _to_nx(graph)
    for n_pages in range(max_pages, min_pages - 1, -1):
        candidates: List[FrozenSet[int]] = []
        seen: set = set()
        for (u, v) in nxg.edges():
            common = frozenset(nxg.neighbors(u)) & frozenset(nxg.neighbors(v))
            common -= {u, v}
            if len(common) < n_pages:
                continue
            pages = tuple(sorted(common))[:n_pages]
            atom = frozenset({u, v}) | frozenset(pages)
            if atom not in seen:
                seen.add(atom)
                candidates.append(atom)
        atoms = _greedy_disjoint(candidates)
        if len(atoms) >= 2:
            return [Atom(family=f"B_{n_pages}", vertices=a) for a in atoms]
    return []


# ---------------------------------------------------------------------------
# W_n (wheel) atoms — hub + cycle C_n
# ---------------------------------------------------------------------------

def find_disjoint_wheel_atoms(
    graph: Graph,
    nxg: Optional[nx.Graph] = None,
    min_rim: int = 5,
    max_rim: int = 8,
) -> List[Atom]:
    """Find ≥2 disjoint W_n wheel atoms, preferring larger rim.

    W_n = hub vertex u + cycle C_n on n neighbors of u. Vertex set
    is {u} ∪ rim where rim is a cycle of length n in the induced
    subgraph G[N(u)].

    min_rim defaults to 5 to avoid overlap with K_4 (which is W_3)
    and K_5-minus-edge / book overlap (W_4). For most practical
    graphs, rim ≥ 5 wheels are distinct from cliques/books.

    Detection cost: O(deg(u) choose rim) per candidate hub. Capped
    by max_rim to keep enumeration tractable.
    """
    if nxg is None:
        nxg = _to_nx(graph)
    for rim_size in range(max_rim, min_rim - 1, -1):
        candidates: List[FrozenSet[int]] = []
        seen: set = set()
        for u in nxg.nodes():
            nbrs = list(nxg.neighbors(u))
            if len(nbrs) < rim_size:
                continue
            # Try the FULL neighborhood as rim if its size matches;
            # otherwise iterate combinations up to a small cap.
            # The induced subgraph on rim_size neighbors must be a cycle.
            if len(nbrs) == rim_size:
                ns_candidates = [tuple(sorted(nbrs))]
            elif len(nbrs) <= rim_size + 2:
                # Slight slack — enumerate a few subsets
                ns_candidates = [tuple(sorted(c))
                                 for c in combinations(nbrs, rim_size)]
            else:
                # Too many subsets — skip to avoid blowup
                continue
            for cand in ns_candidates:
                sub = nxg.subgraph(cand)
                if sub.number_of_edges() != rim_size:
                    continue
                if not all(sub.degree(v) == 2 for v in cand):
                    continue
                # 2-regular & |E|=|V|=rim_size → it's a single cycle
                atom = frozenset({u}) | frozenset(cand)
                if atom not in seen:
                    seen.add(atom)
                    candidates.append(atom)
                break  # one canonical wheel per hub
        atoms = _greedy_disjoint(candidates)
        if len(atoms) >= 2:
            return [Atom(family=f"W_{rim_size}", vertices=a) for a in atoms]
    return []


# ---------------------------------------------------------------------------
# Unified entry — combines families and picks the best
# ---------------------------------------------------------------------------

def find_disjoint_atoms(
    graph: Graph,
    *,
    try_kn: bool = True,
    try_kab: bool = True,
    try_books: bool = True,
    try_wheels: bool = True,
    kn_min: int = 3,
    kn_max: int = 6,
    kab_pairs: Optional[List[Tuple[int, int]]] = None,
    book_min: int = 2,
    book_max: int = 5,
    wheel_min: int = 5,
    wheel_max: int = 8,
) -> List[Atom]:
    """Find disjoint atoms from the requested named families.

    Returns the FIRST family-tier that yields ≥2 disjoint atoms,
    preferring (in order):
      1. K_n with largest k                — densest, most generic
      2. K_{a,b} with largest a+b          — bipartite cells
      3. B_n (books) with largest n        — triangulated fans
      4. W_n (wheels) with largest rim     — hub-spoke + cycle

    Empty list when no family yields ≥2 disjoint atoms.

    The caller's chord-peel dispatch evaluates the returned atoms
    independently of family; what matters is the inter-atom junction
    structure.
    """
    nxg = _to_nx(graph)
    if try_kn:
        atoms = find_disjoint_kn_atoms(graph, nxg=nxg, min_k=kn_min, max_k=kn_max)
        if len(atoms) >= 2:
            return atoms
    if try_kab:
        atoms = find_disjoint_kab_atoms(graph, nxg=nxg, ab_pairs=kab_pairs)
        if len(atoms) >= 2:
            return atoms
    if try_books:
        atoms = find_disjoint_book_atoms(
            graph, nxg=nxg, min_pages=book_min, max_pages=book_max,
        )
        if len(atoms) >= 2:
            return atoms
    if try_wheels:
        atoms = find_disjoint_wheel_atoms(
            graph, nxg=nxg, min_rim=wheel_min, max_rim=wheel_max,
        )
        if len(atoms) >= 2:
            return atoms
    return []


# ---------------------------------------------------------------------------
# Inter-atom junction analysis
# ---------------------------------------------------------------------------

def find_smallest_junction(
    graph: Graph,
    atoms: List[Atom],
    nxg: Optional[nx.Graph] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Return the smallest connected inter-atom junction edge set.

    For each pair (A_i, A_j) of atoms, gather inter-atom edges
    (one endpoint in A_i, other in A_j). Split by connected component
    of the bipartite-edge subgraph. Return the smallest component.

    Returns None when no inter-atom edges exist (atoms are disconnected
    from each other).
    """
    if nxg is None:
        nxg = _to_nx(graph)
    edges_sorted = sorted(graph.edges)
    smallest: Optional[List[Tuple[int, int]]] = None
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ai, aj = atoms[i].vertices, atoms[j].vertices
            inter: List[Tuple[int, int]] = [
                (u, v) for (u, v) in edges_sorted
                if (u in ai and v in aj) or (u in aj and v in ai)
            ]
            if not inter:
                continue
            comp_graph = nx.Graph()
            comp_graph.add_edges_from(inter)
            for comp_nodes in nx.connected_components(comp_graph):
                comp_edges = [
                    (u, v) for (u, v) in inter
                    if u in comp_nodes and v in comp_nodes
                ]
                if comp_edges and (
                    smallest is None or len(comp_edges) < len(smallest)
                ):
                    smallest = comp_edges
    return smallest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_nx(graph: Graph) -> nx.Graph:
    nxg = nx.Graph()
    nxg.add_nodes_from(graph.nodes)
    nxg.add_edges_from(graph.edges)
    return nxg


def _greedy_disjoint(
    candidates: List[FrozenSet[int]],
) -> List[FrozenSet[int]]:
    """Greedy disjoint selection (sorted lex for determinism)."""
    sorted_cands = sorted(candidates, key=lambda fc: tuple(sorted(fc)))
    out: List[FrozenSet[int]] = []
    used: set = set()
    for fc in sorted_cands:
        if not (fc & used):
            out.append(fc)
            used |= fc
    return out
