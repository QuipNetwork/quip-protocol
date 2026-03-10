"""Treewidth-Based Tutte Polynomial Computation.

For graphs with bounded treewidth w, computes the Tutte polynomial in
O(n * B(w+1)^2) time using DP on a tree decomposition, where B is the
Bell number. This is vastly faster than deletion-contraction for graphs
with treewidth 4-5 (B(5)=52, B(6)=203).

Algorithm: Fortuin-Kasteleyn / rank-nullity formulation.
We compute T(x,y) = sum_{A subset E} (x-1)^{r(E)-r(A)} * (y-1)^{|A|-r(A)}
via DP on set partitions of bag vertices tracking connectivity.

The DP state is a partition of the bag vertices (encoding which vertices
are connected by selected edges). We work in the (x-1, y-1) basis and
expand to standard (x, y) basis at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..graph import MultiGraph
from ..polynomial import TuttePolynomial


# =============================================================================
# SET PARTITION UTILITIES
# =============================================================================

def canonicalize(labels: Tuple[int, ...]) -> Tuple[int, ...]:
    """Renumber partition labels in first-appearance order.

    Example: (2, 0, 2, 1) -> (0, 1, 0, 2)
    """
    mapping: Dict[int, int] = {}
    next_id = 0
    result = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        result.append(mapping[lbl])
    return tuple(result)


def connect(labels: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    """Merge the blocks containing positions i and j."""
    li, lj = labels[i], labels[j]
    if li == lj:
        return labels
    # Replace all occurrences of lj with li
    target = min(li, lj)
    replace = max(li, lj)
    return canonicalize(tuple(target if l == replace else l for l in labels))


def forget(labels: Tuple[int, ...], i: int) -> Tuple[bool, Tuple[int, ...]]:
    """Remove element at position i from the partition.

    Returns (was_singleton, new_partition).
    A singleton is an element whose label appears exactly once.
    """
    lbl = labels[i]
    was_singleton = labels.count(lbl) == 1
    new_labels = labels[:i] + labels[i+1:]
    return was_singleton, canonicalize(new_labels)


# =============================================================================
# POLYNOMIAL ARITHMETIC IN (x-1, y-1) BASIS
# =============================================================================
# We represent polynomials as Dict[(a_pow, b_pow), coeff] where
# a = (x-1), b = (y-1). This avoids division when converting Z -> T.

Poly = Dict[Tuple[int, int], int]


def poly_zero() -> Poly:
    return {}


def poly_one() -> Poly:
    return {(0, 0): 1}


def poly_add(p: Poly, q: Poly) -> Poly:
    result = dict(p)
    for k, v in q.items():
        result[k] = result.get(k, 0) + v
        if result[k] == 0:
            del result[k]
    return result


def poly_mul_monomial(p: Poly, a_pow: int, b_pow: int) -> Poly:
    """Multiply polynomial by a^a_pow * b^b_pow."""
    return {(k[0] + a_pow, k[1] + b_pow): v for k, v in p.items()}


# =============================================================================
# TREE DECOMPOSITION
# =============================================================================

@dataclass
class TreeDecomposition:
    """Tree decomposition of a graph.

    bags: list of frozenset of vertices (each bag)
    tree_adj: adjacency list of the tree (bag index -> set of bag indices)
    root: index of root bag
    bag_edges: for each bag index, list of (u, v, multiplicity) edges assigned to it
    width: treewidth (max bag size - 1)
    """
    bags: List[frozenset]
    tree_adj: Dict[int, Set[int]]
    root: int
    bag_edges: Dict[int, List[Tuple[int, int, int]]]  # bag_idx -> [(u, v, mult)]
    width: int


def compute_tree_decomposition(mg: MultiGraph, max_width: int = 6) -> Optional[TreeDecomposition]:
    """Compute tree decomposition via greedy min-fill-in elimination.

    Returns None if treewidth exceeds max_width.
    """
    if not mg.nodes:
        return None

    nodes = sorted(mg.nodes)
    n = len(nodes)

    if n == 1:
        # Single node - trivial decomposition
        bag = frozenset(nodes)
        node = nodes[0]
        loop_edges = []
        if node in mg.loop_counts:
            loop_edges.append((node, node, mg.loop_counts[node]))
        return TreeDecomposition(
            bags=[bag],
            tree_adj={0: set()},
            root=0,
            bag_edges={0: loop_edges},
            width=0,
        )

    # Build simple adjacency (ignore multiplicities for decomposition)
    adj: Dict[int, Set[int]] = {v: set() for v in nodes}
    for (u, v) in mg.edge_counts:
        adj[u].add(v)
        adj[v].add(u)

    # Greedy elimination ordering by min-fill-in
    remaining = set(nodes)
    elim_order = []
    elim_bags: List[frozenset] = []  # bag for each eliminated vertex
    elim_adj = {v: set(adj[v]) for v in nodes}  # mutable copy

    for _ in range(n):
        # Pick vertex with minimum fill-in edges
        best_v = None
        best_fill = float('inf')
        for v in remaining:
            neighbors_in = elim_adj[v] & remaining
            # Count fill-in edges needed
            fill = 0
            nb_list = sorted(neighbors_in)
            for i in range(len(nb_list)):
                for j in range(i + 1, len(nb_list)):
                    if nb_list[j] not in elim_adj[nb_list[i]]:
                        fill += 1
            if fill < best_fill:
                best_fill = fill
                best_v = v

        # Create bag = {v} union neighbors still remaining
        neighbors_in = elim_adj[best_v] & remaining
        bag = frozenset({best_v} | neighbors_in)

        if len(bag) > max_width + 1:
            return None  # Treewidth too high

        elim_order.append(best_v)
        elim_bags.append(bag)

        # Make neighbors of v a clique (fill-in)
        nb_list = sorted(neighbors_in)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                u, w = nb_list[i], nb_list[j]
                elim_adj[u].add(w)
                elim_adj[w].add(u)

        # Remove v from remaining
        remaining.remove(best_v)
        for nb in list(elim_adj[best_v]):
            elim_adj[nb].discard(best_v)

    # Build tree: parent of bag(i) = first bag after i that shares a vertex
    elim_index = {v: i for i, v in enumerate(elim_order)}

    # Remove duplicate/subset bags
    # Keep track of bag indices and build tree
    bag_list = list(elim_bags)
    tree_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

    # Parent of bag i = the bag of the neighbor of elim_order[i] that was
    # eliminated earliest after i
    for i in range(n - 1):
        v = elim_order[i]
        bag = bag_list[i]
        # Find parent: bag with smallest index > i that overlaps with bag - {v}
        remaining_vertices = bag - {v}
        if not remaining_vertices:
            # Isolated vertex, connect to next bag
            tree_adj[i].add(i + 1)
            tree_adj[i + 1].add(i)
            continue

        # The parent bag is the bag where the first remaining vertex gets eliminated
        parent_idx = min(elim_index[u] for u in remaining_vertices)
        tree_adj[i].add(parent_idx)
        tree_adj[parent_idx].add(i)

    root = n - 1  # Last eliminated vertex's bag is root

    # Assign edges to bags
    bag_edges: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(n)}

    for (u, v), mult in mg.edge_counts.items():
        # Assign to the bag where the later-eliminated of u,v is eliminated
        idx_u, idx_v = elim_index[u], elim_index[v]
        bag_idx = max(idx_u, idx_v)
        # Verify both endpoints are in the bag
        if u in bag_list[bag_idx] and v in bag_list[bag_idx]:
            bag_edges[bag_idx].append((u, v, mult))
        else:
            # Shouldn't happen with correct tree decomposition, but find a valid bag
            for i in range(n):
                if u in bag_list[i] and v in bag_list[i]:
                    bag_edges[i].append((u, v, mult))
                    break

    # Assign loops to bags
    for node, count in mg.loop_counts.items():
        bag_idx = elim_index[node]
        bag_edges[bag_idx].append((node, node, count))

    width = max(len(b) - 1 for b in bag_list)

    return TreeDecomposition(
        bags=bag_list,
        tree_adj=tree_adj,
        root=root,
        bag_edges=bag_edges,
        width=width,
    )


# =============================================================================
# DP ON TREE DECOMPOSITION
# =============================================================================

# DP table: maps partition (tuple of ints) to polynomial in (a, b) basis
# where a = (x-1), b = (y-1).
DPTable = Dict[Tuple[int, ...], Poly]


def compute_treewidth_tutte(td: TreeDecomposition, mg: MultiGraph) -> TuttePolynomial:
    """Compute Tutte polynomial via DP on tree decomposition.

    Uses the rank-nullity formulation:
    T(x,y) = sum_{A subset E} (x-1)^{r(E)-r(A)} * (y-1)^{|A|-r(A)}

    where r(A) = |V_processed| - num_connected_components(A).
    """
    n_bags = len(td.bags)

    # Compute rank of entire graph: r(E) = |V| - k(G)
    # where k(G) = number of connected components
    visited: Set[int] = set()
    num_components = 0
    for start in mg.nodes:
        if start in visited:
            continue
        num_components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in mg.neighbors(node):
                if nb not in visited:
                    stack.append(nb)
    # Build children for rooted tree
    children: Dict[int, List[int]] = {i: [] for i in range(n_bags)}
    visited_bags: Set[int] = set()

    def build_children(bag_idx: int):
        visited_bags.add(bag_idx)
        for nb in td.tree_adj[bag_idx]:
            if nb not in visited_bags:
                children[bag_idx].append(nb)
                build_children(nb)

    build_children(td.root)

    # Process bottom-up
    def process_bag(bag_idx: int) -> Tuple[DPTable, List[int]]:
        """Process a bag and return (DP table, vertex ordering in partition).

        The vertex ordering maps partition indices to actual vertex IDs.
        """
        bag = td.bags[bag_idx]
        bag_verts = sorted(bag)

        # Start with trivial table: all singletons, coefficient 1
        # Each vertex starts in its own block
        init_partition = canonicalize(tuple(range(len(bag_verts))))
        table: DPTable = {init_partition: poly_one()}

        # Introduce edges assigned to this bag
        for (u, v, mult) in td.bag_edges[bag_idx]:
            if u not in bag or v not in bag:
                continue

            idx_u = bag_verts.index(u)
            idx_v = bag_verts.index(v) if u != v else idx_u

            for _ in range(mult):
                new_table: DPTable = {}
                for partition, poly in table.items():
                    # Edge absent: partition unchanged, poly unchanged
                    new_table[partition] = poly_add(
                        new_table.get(partition, poly_zero()), poly
                    )

                    # Edge present: connect u,v in partition, multiply by b=(y-1)
                    if u == v:
                        # Loop: connect(u,u) is no-op, just multiply by b
                        new_poly = poly_mul_monomial(poly, 0, 1)
                        new_table[partition] = poly_add(
                            new_table.get(partition, poly_zero()), new_poly
                        )
                    else:
                        new_part = connect(partition, idx_u, idx_v)
                        new_poly = poly_mul_monomial(poly, 0, 1)
                        new_table[new_part] = poly_add(
                            new_table.get(new_part, poly_zero()), new_poly
                        )
                table = new_table

        # Process children: merge child tables, then forget child-only vertices
        for child_idx in children[bag_idx]:
            child_table, child_verts = process_bag(child_idx)
            child_bag = td.bags[child_idx]

            # Forget vertices in child but not in current bag
            verts_to_forget = [v for v in child_verts if v not in bag]

            # Forget vertices from child table
            forgotten_table: DPTable = {}
            for partition, poly in child_table.items():
                current_verts = list(child_verts)
                current_part = partition
                current_poly = poly

                for v in verts_to_forget:
                    if v not in current_verts:
                        continue
                    idx = current_verts.index(v)
                    was_singleton, new_part = forget(current_part, idx)
                    if was_singleton:
                        # Completed component: multiply by a*b = (x-1)*(y-1)
                        current_poly = poly_mul_monomial(current_poly, 1, 1)
                    current_part = new_part
                    current_verts = current_verts[:idx] + current_verts[idx+1:]

                forgotten_table[current_part] = poly_add(
                    forgotten_table.get(current_part, poly_zero()),
                    current_poly,
                )

            # Now child table has same vertex set as shared vertices
            # Merge with current table via join
            # The shared vertices must be in the same order
            shared_verts = [v for v in child_verts if v in bag]

            # Remap child partition to match parent's vertex ordering
            table = _merge_tables(table, forgotten_table, bag_verts, shared_verts)

        return table, bag_verts

    root_table, root_verts = process_bag(td.root)

    # Forget all remaining vertices in root
    final_poly = poly_zero()
    for partition, poly in root_table.items():
        current_verts = list(root_verts)
        current_part = partition
        current_poly = poly

        for v in list(current_verts):
            idx = current_verts.index(v)
            was_singleton, new_part = forget(current_part, idx)
            if was_singleton:
                # Completed component: multiply by a*b = (x-1)*(y-1)
                current_poly = poly_mul_monomial(current_poly, 1, 1)
            current_part = new_part
            current_verts = current_verts[:idx] + current_verts[idx+1:]

        # current_part should now be empty
        final_poly = poly_add(final_poly, current_poly)

    # We computed sum_A a^{k(A)} * b^{|A|+k(A)} where a=(x-1), b=(y-1).
    # (Each edge contributes b; each completed component contributes a*b.)
    #
    # T(x,y) = sum_A a^{k(A)-k(E)} * b^{|A|+k(A)-|V|}
    #
    # So: final_poly = a^{k(E)} * b^{|V|} * T(x,y)
    # Therefore: T(x,y) = final_poly / (a^{k(E)} * b^{|V|})

    # Divide by (x-1)^{num_components} * (y-1)^{|V|}
    # In our (a, b) basis, this means shifting exponents down
    n_verts = len(mg.nodes)
    a_shift = num_components
    b_shift = n_verts

    tutte_poly: Poly = {}
    for (a_pow, b_pow), coeff in final_poly.items():
        new_a = a_pow - a_shift
        new_b = b_pow - b_shift
        if new_a < 0 or new_b < 0:
            # This shouldn't happen for valid Tutte polynomials
            # but check just in case
            if coeff != 0:
                raise RuntimeError(
                    f"Negative exponent in Tutte polynomial: "
                    f"a^{new_a} * b^{new_b} with coeff {coeff}"
                )
            continue
        if coeff != 0:
            tutte_poly[(new_a, new_b)] = coeff

    # Convert from (x-1, y-1) basis to (x, y) basis
    return _convert_ab_to_xy(tutte_poly)


def _merge_tables(
    parent_table: DPTable,
    child_table: DPTable,
    parent_verts: List[int],
    shared_verts: List[int],
) -> DPTable:
    """Merge child table into parent table.

    The child table has been reduced to only shared vertices.
    The child's partition encodes connectivity established through the child
    subtree. This connectivity is applied to the parent partition by merging
    blocks of shared vertices that the child says are connected.

    Every (parent_part, child_part) pair is combined: start with parent_part,
    then force-merge blocks as dictated by child_part.
    """
    if not shared_verts:
        # No shared vertices: just multiply polynomials
        child_poly = poly_zero()
        for poly in child_table.values():
            child_poly = poly_add(child_poly, poly)

        result: DPTable = {}
        for partition, poly in parent_table.items():
            new_poly = _poly_mul(poly, child_poly)
            if new_poly:
                result[partition] = poly_add(result.get(partition, poly_zero()), new_poly)
        return result

    shared_positions_in_parent = [parent_verts.index(v) for v in shared_verts]

    result: DPTable = {}

    for parent_part, parent_poly in parent_table.items():
        for child_part, child_poly in child_table.items():
            # Start with parent partition, apply child's connectivity
            merged = list(parent_part)
            for ci in range(len(shared_verts)):
                for cj in range(ci + 1, len(shared_verts)):
                    if child_part[ci] == child_part[cj]:
                        # Child says these shared verts are connected
                        pi = shared_positions_in_parent[ci]
                        pj = shared_positions_in_parent[cj]
                        old_label = merged[pj]
                        new_label = merged[pi]
                        if old_label != new_label:
                            target = min(old_label, new_label)
                            replace = max(old_label, new_label)
                            merged = [target if l == replace else l for l in merged]

            merged_part = canonicalize(tuple(merged))
            new_poly = _poly_mul(parent_poly, child_poly)
            if new_poly:
                result[merged_part] = poly_add(
                    result.get(merged_part, poly_zero()), new_poly
                )

    return result


def _poly_mul(p: Poly, q: Poly) -> Poly:
    """Multiply two polynomials."""
    if not p or not q:
        return {}
    result: Poly = {}
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            key = (a1 + a2, b1 + b2)
            result[key] = result.get(key, 0) + c1 * c2
            if result[key] == 0:
                del result[key]
    return result


def _convert_ab_to_xy(poly_ab: Poly) -> TuttePolynomial:
    """Convert polynomial from (a, b) = (x-1, y-1) basis to (x, y) basis.

    Expand each a^i * b^j = (x-1)^i * (y-1)^j using binomial theorem.
    """
    from math import comb

    coeffs_xy: Dict[Tuple[int, int], int] = {}

    for (a_pow, b_pow), coeff in poly_ab.items():
        # (x-1)^a_pow = sum_{p=0}^{a_pow} C(a_pow,p) * x^p * (-1)^{a_pow-p}
        # (y-1)^b_pow = sum_{q=0}^{b_pow} C(b_pow,q) * y^q * (-1)^{b_pow-q}
        for p in range(a_pow + 1):
            cx = comb(a_pow, p) * ((-1) ** (a_pow - p))
            for q in range(b_pow + 1):
                cy = comb(b_pow, q) * ((-1) ** (b_pow - q))
                contribution = coeff * cx * cy
                if contribution != 0:
                    key = (p, q)
                    coeffs_xy[key] = coeffs_xy.get(key, 0) + contribution

    # Remove zeros
    coeffs_xy = {k: v for k, v in coeffs_xy.items() if v != 0}
    return TuttePolynomial.from_coefficients(coeffs_xy)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def compute_treewidth_tutte_if_applicable(
    mg: MultiGraph, max_width: int = 6
) -> Optional[TuttePolynomial]:
    """Compute Tutte polynomial via tree decomposition if treewidth <= max_width.

    Returns None if treewidth exceeds max_width or graph is trivial.
    """
    if not mg.nodes:
        return TuttePolynomial.one()

    if not mg.edge_counts and not mg.loop_counts:
        return TuttePolynomial.one()

    td = compute_tree_decomposition(mg, max_width)
    if td is None:
        return None

    return compute_treewidth_tutte(td, mg)
