"""Tests for the tutte.roots/ module.

Sections:
    A. Anchor-group detection (cell_anchor_adapter.detect_cell_anchor_groups)
    B. Per-cell structural orbit canonicalization (aut_orbit)
    C. join_partitions C extension wrapper (_partition_c)
    D. compute_path_dp_grouped module-level cache (cell_quotient_path)
    E. compute_path_dp_grouped — generic path DP with shared-anchor support
    F. compute_grid_dp_grouped — generic grid DP (disjoint + shared anchors)
    G. compute_cell_quotient_cycle_dp — engine-level cycle dispatch
    H. Hamiltonian path / closing-edge helpers (interleaved DP)
    I. Synthetic K_3 grids (interleaved DP, fast)
    J. Cm2 (interleaved DP, slow)
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations, product
from typing import List, Set, Tuple

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.lookup.core import load_default_table
from tutte.polynomial import TuttePolynomial
from tutte.roots import (
    CellAnchorGroups,
    CellGridSpec,
    CellRowSpec,
    compute_cell_quotient_cycle_dp,
    compute_grid_dp_grouped,
    compute_path_dp_grouped,
    detect_cell_anchor_groups,
    extract_path_specs,
)
from tutte.roots.aut_orbit import (
    aut_compress_t_rooted_per_cell,
    per_cell_canonical_key,
    per_cell_orbit_rep,
    per_cell_orbit_size,
)
from tutte.roots.cell_quotient_interleaved import (
    compute_grid_dp_interleaved,
    grid_path_and_closing_edges,
    hamiltonian_path_grid,
)
from tutte.roots.cell_quotient_path import (
    clear_path_dp_cache,
    path_dp_cache_stats,
)
from tutte.roots.rooted_tutte import (
    divide_by_x_minus_1_power,
    join_partitions,
)
from tutte.roots._partition_c import join_partitions_c_wrapper
from tutte.synthesis.engine import SynthesisEngine


def _engine_T(g: Graph) -> TuttePolynomial:
    table = load_default_table()
    engine = SynthesisEngine(table=table, verbose=False)
    return engine.synthesize(g).polynomial


def _marginalize(T_dict: dict, td: int) -> TuttePolynomial:
    s = TuttePolynomial.zero()
    for v in T_dict.values():
        s = s + v
    return divide_by_x_minus_1_power(s, td)


# =============================================================================
# A. ANCHOR-GROUP DETECTION
# =============================================================================


def _build_k_n_grid(rows: int, cols: int, k_cell: int):
    """Synthetic (rows × cols) grid of K_{k_cell} cells with single-edge
    horizontal and vertical junctions.

    Returns (partition, inter_edges) for the detector.
    """
    nxg = nx.Graph()
    for cell in range(rows * cols):
        base = k_cell * cell
        nxg.add_nodes_from(range(base, base + k_cell))
        for u in range(k_cell):
            for v in range(u + 1, k_cell):
                nxg.add_edge(base + u, base + v)
    for r in range(rows):
        for c in range(cols - 1):
            cell_l = r * cols + c
            cell_r = r * cols + c + 1
            nxg.add_edge(k_cell * cell_l + 1, k_cell * cell_r + 0)
    for r in range(rows - 1):
        for c in range(cols):
            cell_u = r * cols + c
            cell_d = (r + 1) * cols + c
            nxg.add_edge(k_cell * cell_u + 3, k_cell * cell_d + 2)
    partition: List[Set[int]] = [
        set(range(k_cell * i, k_cell * (i + 1))) for i in range(rows * cols)
    ]
    cell_of = {n: i for i, cell in enumerate(partition) for n in cell}
    inter_edges = [
        (u, v) for u, v in nxg.edges() if cell_of[u] != cell_of[v]
    ]
    return partition, inter_edges


def test_detect_k4_grid_2x2_no_sharing():
    """2×2 K_4 grid: every cell vertex serves at most one junction → no sharing."""
    partition, inter = _build_k_n_grid(2, 2, 4)
    spec = detect_cell_anchor_groups(partition, inter)
    assert spec.has_shared_anchors() is False
    for cell_idx in range(4):
        n_junctions = sum(
            1 for (ca, cb, _, _) in spec.junction_groups
            if ca == cell_idx or cb == cell_idx
        )
        assert spec.groups_per_cell(cell_idx) == n_junctions


def test_detect_k4_grid_3x3_no_sharing():
    """3×3 K_4 grid: 4 corner cells (2 groups), 4 edge cells (3 groups),
    1 interior cell (4 groups). All single-anchor, no sharing."""
    partition, inter = _build_k_n_grid(3, 3, 4)
    spec = detect_cell_anchor_groups(partition, inter)
    assert spec.has_shared_anchors() is False
    sizes = sorted(spec.groups_per_cell(i) for i in range(9))
    assert sizes == [2, 2, 2, 2, 3, 3, 3, 3, 4]


def test_detect_cm3_shared_anchors():
    """D-Wave Cm3 interior cell shares anchor groups across multiple junctions."""
    dnx = pytest.importorskip("dwave_networkx")
    g = dnx.chimera_graph(3)
    cells = defaultdict(set)
    for n, data in g.nodes(data=True):
        coord = data.get("chimera_index")
        if coord is None:
            continue
        r, c, _, _ = coord
        cells[(r, c)].add(n)
    sorted_keys = sorted(cells.keys())
    partition = [cells[k] for k in sorted_keys]
    cell_of = {n: i for i, cell in enumerate(partition) for n in cell}
    inter = [(u, v) for u, v in g.edges() if cell_of[u] != cell_of[v]]
    spec = detect_cell_anchor_groups(partition, inter)
    assert spec.has_shared_anchors() is True
    interior_idx = sorted_keys.index((1, 1))
    assert spec.groups_per_cell(interior_idx) == 2
    n_junctions_interior = sum(
        1 for (ca, cb, _, _) in spec.junction_groups
        if ca == interior_idx or cb == interior_idx
    )
    assert n_junctions_interior == 4
    usage_counts: dict = defaultdict(int)
    for (ca, cb, ga, gb) in spec.junction_groups:
        usage_counts[(ca, ga)] += 1
        usage_counts[(cb, gb)] += 1
    assert usage_counts[(interior_idx, 0)] == 2
    assert usage_counts[(interior_idx, 1)] == 2


def test_extract_path_specs_k4_grid_3x3_row_1():
    """Middle row of 3×3 K_4 grid: cell 4 has L=group(to-cell-3),
    R=group(to-cell-5), and 2 vertical extras."""
    partition, inter = _build_k_n_grid(3, 3, 4)
    spec = detect_cell_anchor_groups(partition, inter)
    middle_row = [3, 4, 5]
    row_specs = extract_path_specs(spec, middle_row)
    assert row_specs[0].cell == 3
    assert row_specs[0].left_group is None
    assert row_specs[0].right_group is not None
    assert len(row_specs[0].extra_groups) == 2
    assert row_specs[1].cell == 4
    assert row_specs[1].left_group is not None
    assert row_specs[1].right_group is not None
    assert row_specs[1].left_group != row_specs[1].right_group
    assert row_specs[1].has_shared_horizontal is False
    assert row_specs[2].cell == 5
    assert row_specs[2].right_group is None


def test_extract_path_specs_cm3_middle_row_shared():
    """Cm3 middle row: interior cell (1,1) has L == R (shared horizontal)."""
    dnx = pytest.importorskip("dwave_networkx")
    g = dnx.chimera_graph(3)
    cells = defaultdict(set)
    for n, data in g.nodes(data=True):
        coord = data.get("chimera_index")
        if coord is None:
            continue
        r, c, _, _ = coord
        cells[(r, c)].add(n)
    sorted_keys = sorted(cells.keys())
    partition = [cells[k] for k in sorted_keys]
    cell_of = {n: i for i, cell in enumerate(partition) for n in cell}
    inter = [(u, v) for u, v in g.edges() if cell_of[u] != cell_of[v]]
    spec = detect_cell_anchor_groups(partition, inter)

    idx_10 = sorted_keys.index((1, 0))
    idx_11 = sorted_keys.index((1, 1))
    idx_12 = sorted_keys.index((1, 2))
    middle_row = [idx_10, idx_11, idx_12]
    row_specs = extract_path_specs(spec, middle_row)
    interior = row_specs[1]
    assert interior.cell == idx_11
    assert interior.left_group is not None
    assert interior.right_group is not None
    assert interior.has_shared_horizontal is True
    assert len(interior.extra_groups) == 1


def test_detect_cm2_no_sharing():
    """D-Wave Cm2 cells use disjoint anchor groups per junction."""
    dnx = pytest.importorskip("dwave_networkx")
    g = dnx.chimera_graph(2)
    cells = defaultdict(set)
    for n, data in g.nodes(data=True):
        coord = data.get("chimera_index")
        if coord is None:
            continue
        r, c, _, _ = coord
        cells[(r, c)].add(n)
    sorted_keys = sorted(cells.keys())
    partition = [cells[k] for k in sorted_keys]
    cell_of = {n: i for i, cell in enumerate(partition) for n in cell}
    inter = [(u, v) for u, v in g.edges() if cell_of[u] != cell_of[v]]
    spec = detect_cell_anchor_groups(partition, inter)
    assert spec.has_shared_anchors() is False
    for i in range(4):
        assert spec.groups_per_cell(i) == 2


# =============================================================================
# B. PER-CELL STRUCTURAL ORBIT CANONICALIZATION
# =============================================================================


def _normalize_partition(blocks):
    return tuple(sorted(tuple(sorted(b)) for b in blocks))


def _all_partitions(elements: List[int]) -> List[Tuple]:
    """Brute-force enumerate all set-partitions of `elements`."""
    if not elements:
        return [()]
    if len(elements) == 1:
        return [((elements[0],),)]
    out = []
    first = elements[0]
    for sub in _all_partitions(elements[1:]):
        out.append(_normalize_partition([(first,)] + [tuple(b) for b in sub]))
        for i in range(len(sub)):
            new_sub = [list(b) for b in sub]
            new_sub[i] = list(new_sub[i]) + [first]
            out.append(_normalize_partition(new_sub))
    seen, dedup = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _apply_per_cell_perm(P, cell_groups, perms):
    relabel = {}
    for c, group in enumerate(cell_groups):
        for i, v in enumerate(group):
            relabel[v] = group[perms[c][i]]
    new_blocks = []
    for block in P:
        new_blocks.append(tuple(sorted(relabel.get(v, v) for v in block)))
    return tuple(sorted(new_blocks))


def _brute_orbit(P, cell_groups):
    per_cell_perms = [list(permutations(range(len(g)))) for g in cell_groups]
    orbit = set()
    for combo in product(*per_cell_perms):
        orbit.add(_apply_per_cell_perm(P, cell_groups, combo))
    return orbit


def test_per_cell_key_invariant_under_within_cell_permutation():
    cell_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    P_a = ((0, 4), (1, 5), (2, 6), (3, 7))
    P_b = ((0, 5), (1, 4), (2, 6), (3, 7))
    assert per_cell_canonical_key(P_a, cell_groups) == per_cell_canonical_key(P_b, cell_groups)


def test_per_cell_key_distinguishes_different_shapes():
    cell_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    P1 = ((0, 4), (1, 5), (2, 6), (3, 7))
    P2 = ((0, 1), (4, 5), (2, 3), (6, 7))
    P3 = ((0, 1, 2, 3, 4, 5, 6, 7),)
    k1 = per_cell_canonical_key(P1, cell_groups)
    k2 = per_cell_canonical_key(P2, cell_groups)
    k3 = per_cell_canonical_key(P3, cell_groups)
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


def test_per_cell_key_matches_brute_orbit_2cells_3anchors():
    cell_groups = [[0, 1, 2], [3, 4, 5]]
    elems = [0, 1, 2, 3, 4, 5]
    by_key = {}
    for P in _all_partitions(elems):
        k = per_cell_canonical_key(P, cell_groups)
        by_key.setdefault(k, []).append(P)
    for k, members in by_key.items():
        rep = members[0]
        rep_orbit = _brute_orbit(rep, cell_groups)
        assert set(members) == rep_orbit


def test_per_cell_key_matches_brute_orbit_3cells_2anchors():
    cell_groups = [[0, 1], [2, 3], [4, 5]]
    elems = [0, 1, 2, 3, 4, 5]
    by_key = {}
    for P in _all_partitions(elems):
        k = per_cell_canonical_key(P, cell_groups)
        by_key.setdefault(k, []).append(P)
    for k, members in by_key.items():
        rep = members[0]
        rep_orbit = _brute_orbit(rep, cell_groups)
        assert set(members) == rep_orbit


def test_aut_compress_per_cell_basic():
    cell_groups = [[0, 1], [2, 3]]
    poly_a = TuttePolynomial.x()
    poly_b = TuttePolynomial.y()
    T = {
        ((0, 2), (1, 3)): poly_a,
        ((0, 3), (1, 2)): poly_a,
        ((0, 1), (2, 3)): poly_b,
    }
    orbit_T, orbit_parts = aut_compress_t_rooted_per_cell(T, cell_groups)
    assert len(orbit_T) == 2
    for canonical, parts in orbit_parts.items():
        vals = {T[p] for p in parts}
        assert len(vals) == 1


def test_per_cell_orbit_size_matches_brute_count():
    cell_groups = [[0, 1, 2], [3, 4, 5]]
    elems = [0, 1, 2, 3, 4, 5]
    by_key = {}
    for P in _all_partitions(elems):
        k = per_cell_canonical_key(P, cell_groups)
        by_key.setdefault(k, []).append(P)
    for canonical, members in by_key.items():
        analytical = per_cell_orbit_size(canonical, cell_groups)
        assert analytical == len(members)


def test_per_cell_orbit_size_3cells_2anchors():
    cell_groups = [[0, 1], [2, 3], [4, 5]]
    elems = [0, 1, 2, 3, 4, 5]
    by_key = {}
    for P in _all_partitions(elems):
        k = per_cell_canonical_key(P, cell_groups)
        by_key.setdefault(k, []).append(P)
    for canonical, members in by_key.items():
        assert per_cell_orbit_size(canonical, cell_groups) == len(members)


def test_per_cell_orbit_rep_roundtrip():
    cell_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    canonical = ((1, 1, 0), (1, 1, 0), (2, 2, 0))
    rep = per_cell_orbit_rep(canonical, cell_groups)
    assert per_cell_canonical_key(rep, cell_groups) == canonical


@pytest.mark.slow
def test_per_cell_path_dp_3cell_K44_shared():
    """Cm3 row scenario: 3 K_{4,4} cells with M_4 horizontal junctions and
    shared horizontal interior cell. Per-cell compression reduces Bell(12)
    = 4M partitions to ~6608 orbits. ~3 minutes wall-clock."""
    nxg = nx.Graph()
    for cell in range(3):
        base = 8 * cell
        for u in range(4):
            for v in range(4, 8):
                nxg.add_edge(base + u, base + v)
    for i in range(4):
        nxg.add_edge(4 + i, 12 + i)
    for i in range(4):
        nxg.add_edge(12 + i, 20 + i)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    M4 = Graph(list(range(8)), [(0, 4), (1, 5), (2, 6), (3, 7)])
    cell_anchor_groups = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=1, extra_groups=(0,)),
        CellRowSpec(cell=1, left_group=1, right_group=1, extra_groups=(0,)),
        CellRowSpec(cell=2, left_group=1, right_group=None, extra_groups=(0,)),
    ]
    T_compressed, _, td, state_cell_groups = compute_path_dp_grouped(
        K44, cell_anchor_groups, M4, [0, 1, 2, 3], [4, 5, 6, 7], specs,
        enable_per_cell_compression=True,
    )
    sum_T = TuttePolynomial.zero()
    for canonical, val in T_compressed.items():
        sum_T = sum_T + per_cell_orbit_size(canonical, state_cell_groups) * val
    poly = divide_by_x_minus_1_power(sum_T, td)
    assert poly == T_engine


def test_aut_compress_per_cell_raises_on_inconsistent_values():
    cell_groups = [[0, 1], [2, 3]]
    poly_a = TuttePolynomial.x()
    poly_b = TuttePolynomial.y()
    T = {
        ((0, 2), (1, 3)): poly_a,
        ((0, 3), (1, 2)): poly_b,
    }
    with pytest.raises(ValueError):
        aut_compress_t_rooted_per_cell(T, cell_groups)


# =============================================================================
# C. JOIN_PARTITIONS C EXTENSION WRAPPER
# =============================================================================


def test_partition_c_empty_partitions():
    universe = [0, 1, 2]
    assert join_partitions_c_wrapper((), (), universe) == ((0,), (1,), (2,))


def test_partition_c_single_block():
    universe = [0, 1, 2]
    assert join_partitions_c_wrapper(((0, 1),), (), universe) == ((0, 1), (2,))


def test_partition_c_transitive_closure():
    universe = [0, 1, 2]
    assert join_partitions_c_wrapper(((0, 1),), ((1, 2),), universe) == ((0, 1, 2),)


def test_partition_c_disjoint_blocks_each_partition():
    universe = [0, 1, 2, 3]
    assert join_partitions_c_wrapper(
        ((0, 1),), ((2, 3),), universe
    ) == ((0, 1), (2, 3))


def test_partition_c_multi_merge():
    universe = [0, 1, 2, 3]
    assert join_partitions_c_wrapper(
        ((0, 2), (1, 3)), ((0, 1),), universe
    ) == ((0, 1, 2, 3),)


def test_partition_c_non_contiguous_positions():
    universe = [10000, 10001, 10500, 11000]
    P1 = ((10000, 10001), (10500, 11000))
    P2 = ((10000, 11000),)
    assert join_partitions_c_wrapper(P1, P2, universe) == ((10000, 10001, 10500, 11000),)


def test_partition_c_matches_python_reference_random():
    universe = list(range(8))
    test_cases = [
        ((), ()),
        (((0, 1), (2, 3)), ((0, 2), (4, 5))),
        (((0, 1, 2),), ((3, 4, 5), (0, 6))),
        (((0, 4),), ((1, 5), (2, 6), (3, 7))),
        (((0, 1, 2, 3, 4, 5, 6, 7),), ()),
    ]
    for P1, P2 in test_cases:
        py = join_partitions(P1, P2, universe)
        c = join_partitions_c_wrapper(P1, P2, universe)
        assert py == c, f"mismatch for {P1}, {P2}: py={py} c={c}"


def test_partition_c_large_universe_returns_none_or_works():
    universe = list(range(64))
    P = (tuple(range(0, 64)),)
    result = join_partitions_c_wrapper(P, (), universe)
    if result is not None:
        assert result == ((tuple(range(64))),) or result == (tuple(range(64)),)


def test_partition_c_oversized_universe_returns_none():
    universe = list(range(300))
    P = (tuple(range(10)),)
    result = join_partitions_c_wrapper(P, (), universe)
    assert result is None


# =============================================================================
# D. compute_path_dp_grouped MODULE-LEVEL CACHE
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_path_dp_cache_fx():
    clear_path_dp_cache()
    yield
    clear_path_dp_cache()


def test_cache_hit_same_spec_different_label_offset():
    K3 = Graph.from_networkx(nx.complete_graph(3))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1], 1: [2]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=(1,)),
        CellRowSpec(cell=1, left_group=0, right_group=0, extra_groups=(1,)),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=(1,)),
    ]
    T1, _, td1, sg1 = compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=True,
        label_offset=0,
    )
    stats1 = path_dp_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    T2, _, td2, sg2 = compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=True,
        label_offset=1000000,
    )
    stats2 = path_dp_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1

    assert T1 == T2
    assert td1 == td2
    assert [len(g) for g in sg1] == [len(g) for g in sg2]
    assert len(sg1) >= 1
    assert sg1 != sg2


def test_cache_marginalized_polynomial_matches():
    nxg = nx.Graph()
    for cell in range(3):
        base = 3 * cell
        for u in range(3):
            for v in range(u+1, 3):
                nxg.add_edge(base+u, base+v)
    nxg.add_edge(0, 3); nxg.add_edge(1, 4)
    nxg.add_edge(3, 6); nxg.add_edge(4, 7)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K3 = Graph.from_networkx(nx.complete_graph(3))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=0, extra_groups=()),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=()),
    ]
    T1, _, td1, sg1 = compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=True,
    )
    sum1 = TuttePolynomial.zero()
    for c, v in T1.items():
        sum1 = sum1 + per_cell_orbit_size(c, sg1) * v
    poly1 = divide_by_x_minus_1_power(sum1, td1)
    assert poly1 == T_engine

    T2, _, td2, sg2 = compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=True,
        label_offset=500000,
    )
    sum2 = TuttePolynomial.zero()
    for c, v in T2.items():
        sum2 = sum2 + per_cell_orbit_size(c, sg2) * v
    poly2 = divide_by_x_minus_1_power(sum2, td2)
    assert poly2 == T_engine

    stats = path_dp_cache_stats()
    assert stats["hits"] >= 1


def test_cache_distinct_spec_separate_entries():
    K3 = Graph.from_networkx(nx.complete_graph(3))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1]}
    specs_3 = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=0, extra_groups=()),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=()),
    ]
    specs_2 = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=None, extra_groups=()),
    ]
    compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs_3,
        enable_per_cell_compression=True,
    )
    compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs_2,
        enable_per_cell_compression=True,
    )
    stats = path_dp_cache_stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0
    assert stats["size"] == 2


def test_cache_disabled_when_compression_off():
    K3 = Graph.from_networkx(nx.complete_graph(3))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=None, extra_groups=()),
    ]
    compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=False,
    )
    compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
        enable_per_cell_compression=False,
    )
    stats = path_dp_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


# =============================================================================
# E. compute_path_dp_grouped — generic path DP with shared-anchor support
# =============================================================================


def test_path_grouped_2cell_K4_M2_disjoint():
    """2 K_4 cells with M_2 horizontal junction. Disjoint left/right groups."""
    nxg = nx.Graph()
    for cell in range(2):
        base = 4 * cell
        for u in range(4):
            for v in range(u + 1, 4):
                nxg.add_edge(base + u, base + v)
    nxg.add_edge(2, 4)
    nxg.add_edge(3, 5)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K4 = Graph.from_networkx(nx.complete_graph(4))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1], 1: [2, 3]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=1, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=None, extra_groups=()),
    ]
    T_path, _, td = compute_path_dp_grouped(
        K4, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
    )
    T_marg = _marginalize(T_path, td)
    assert T_marg == T_engine


def test_path_grouped_3cell_K3_M2_shared_middle():
    """3 K_3 cells with M_2 junctions; middle cell SHARES anchors."""
    nxg = nx.Graph()
    for cell in range(3):
        base = 3 * cell
        for u in range(3):
            for v in range(u + 1, 3):
                nxg.add_edge(base + u, base + v)
    nxg.add_edge(0, 3)
    nxg.add_edge(1, 4)
    nxg.add_edge(3, 6)
    nxg.add_edge(4, 7)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K3 = Graph.from_networkx(nx.complete_graph(3))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=0, extra_groups=()),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=()),
    ]
    T_path, _, td = compute_path_dp_grouped(
        K3, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
    )
    T_marg = _marginalize(T_path, td)
    assert T_marg == T_engine
    assert T_marg.num_spanning_trees() == 288


def test_path_grouped_3cell_K4_M2_shared_middle():
    """3 K_4 cells with M_2 junctions; middle cell shares anchors."""
    nxg = nx.Graph()
    for cell in range(3):
        base = 4 * cell
        for u in range(4):
            for v in range(u + 1, 4):
                nxg.add_edge(base + u, base + v)
    nxg.add_edge(0, 4)
    nxg.add_edge(1, 5)
    nxg.add_edge(4, 8)
    nxg.add_edge(5, 9)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K4 = Graph.from_networkx(nx.complete_graph(4))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=0, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=0, extra_groups=()),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=()),
    ]
    T_path, _, td = compute_path_dp_grouped(
        K4, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
    )
    T_marg = _marginalize(T_path, td)
    assert T_marg == T_engine
    assert T_marg.num_spanning_trees() == 35840


def test_path_grouped_3cell_K4_M2_disjoint_middle():
    """3 K_4 cells with M_2 junctions; middle cell does NOT share anchors."""
    nxg = nx.Graph()
    for cell in range(3):
        base = 4 * cell
        for u in range(4):
            for v in range(u + 1, 4):
                nxg.add_edge(base + u, base + v)
    nxg.add_edge(0, 4)
    nxg.add_edge(1, 5)
    nxg.add_edge(6, 8)
    nxg.add_edge(7, 9)
    g = Graph.from_networkx(nxg)
    T_engine = _engine_T(g)

    K4 = Graph.from_networkx(nx.complete_graph(4))
    M2 = Graph(list(range(4)), [(0, 2), (1, 3)])
    cell_anchor_groups = {0: [0, 1], 1: [2, 3]}
    specs = [
        CellRowSpec(cell=0, left_group=None, right_group=1, extra_groups=()),
        CellRowSpec(cell=1, left_group=0, right_group=1, extra_groups=()),
        CellRowSpec(cell=2, left_group=0, right_group=None, extra_groups=()),
    ]
    T_path, _, td = compute_path_dp_grouped(
        K4, cell_anchor_groups, M2, [0, 1], [2, 3], specs,
    )
    T_marg = _marginalize(T_path, td)
    assert T_marg == T_engine
    assert T_marg.num_spanning_trees() == 36864


# =============================================================================
# F. compute_grid_dp_grouped — generic grid DP
# =============================================================================


def _build_disjoint_k_grid(rows: int, cols: int, k_cell: int) -> Graph:
    """Disjoint anchors: vertex 0=horiz-left, 1=horiz-right, 2=vert-up, 3=vert-down."""
    nxg = nx.Graph()
    for cell in range(rows * cols):
        base = k_cell * cell
        for u in range(k_cell):
            for v in range(u + 1, k_cell):
                nxg.add_edge(base + u, base + v)
    for r in range(rows):
        for c in range(cols - 1):
            nxg.add_edge(k_cell * (r * cols + c) + 1, k_cell * (r * cols + c + 1) + 0)
    for r in range(rows - 1):
        for c in range(cols):
            nxg.add_edge(k_cell * (r * cols + c) + 3, k_cell * ((r + 1) * cols + c) + 2)
    return Graph.from_networkx(nxg)


def _disjoint_specs(rows: int, cols: int) -> list:
    def mk(cell, r, c):
        L = 0 if c > 0 else None
        R = 1 if c < cols - 1 else None
        U = 2 if r > 0 else None
        D = 3 if r < rows - 1 else None
        return CellGridSpec(
            cell=cell, row=r, col=c,
            left_group=L, right_group=R, up_group=U, down_group=D,
            extra_groups=(),
        )
    return [[mk(r * cols + c, r, c) for c in range(cols)] for r in range(rows)]


def _build_shared_k_grid(rows: int, cols: int, k_cell: int) -> Graph:
    """Shared anchors: vertex 0 used for all horizontal, vertex 1 for all vertical."""
    nxg = nx.Graph()
    for cell in range(rows * cols):
        base = k_cell * cell
        for u in range(k_cell):
            for v in range(u + 1, k_cell):
                nxg.add_edge(base + u, base + v)
    for r in range(rows):
        for c in range(cols - 1):
            nxg.add_edge(k_cell * (r * cols + c) + 0, k_cell * (r * cols + c + 1) + 0)
    for r in range(rows - 1):
        for c in range(cols):
            nxg.add_edge(k_cell * (r * cols + c) + 1, k_cell * ((r + 1) * cols + c) + 1)
    return Graph.from_networkx(nxg)


def _shared_specs(rows: int, cols: int) -> list:
    def mk(cell, r, c):
        L = 0 if c > 0 else None
        R = 0 if c < cols - 1 else None
        U = 1 if r > 0 else None
        D = 1 if r < rows - 1 else None
        return CellGridSpec(
            cell=cell, row=r, col=c,
            left_group=L, right_group=R, up_group=U, down_group=D,
            extra_groups=(),
        )
    return [[mk(r * cols + c, r, c) for c in range(cols)] for r in range(rows)]


def test_grid_grouped_2x2_K4_disjoint():
    g = _build_disjoint_k_grid(2, 2, 4)
    T_engine = _engine_T(g)
    K4 = Graph.from_networkx(nx.complete_graph(4))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1], 2: [2], 3: [3]}
    T_grid = compute_grid_dp_grouped(
        K4, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        _disjoint_specs(2, 2),
    )
    assert T_grid == T_engine


def test_grid_grouped_3x3_K4_disjoint():
    g = _build_disjoint_k_grid(3, 3, 4)
    T_engine = _engine_T(g)
    K4 = Graph.from_networkx(nx.complete_graph(4))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1], 2: [2], 3: [3]}
    T_grid = compute_grid_dp_grouped(
        K4, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        _disjoint_specs(3, 3),
    )
    assert T_grid == T_engine
    assert T_grid.num_spanning_trees() == 66795331387392


def test_grid_grouped_2x2_K4_shared():
    """2x2 K_4 grid where each cell uses vertex 0 for horizontal, vertex 1
    for vertical. (Corner cells have only 1 horiz + 1 vert junction so the
    'shared' anchors aren't actually shared in this 2×2 case.)"""
    g = _build_shared_k_grid(2, 2, 4)
    T_engine = _engine_T(g)
    K4 = Graph.from_networkx(nx.complete_graph(4))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1]}
    T_grid = compute_grid_dp_grouped(
        K4, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        _shared_specs(2, 2),
    )
    assert T_grid == T_engine


def test_grid_grouped_3x3_K4_shared_interior():
    """3x3 K_4 grid where interior cell (1,1) genuinely SHARES anchors
    across left/right and up/down junctions (D-Wave Cm3 pattern)."""
    g = _build_shared_k_grid(3, 3, 4)
    T_engine = _engine_T(g)
    K4 = Graph.from_networkx(nx.complete_graph(4))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1]}
    specs = _shared_specs(3, 3)
    assert specs[1][1].has_shared_horizontal is True
    assert specs[1][1].has_shared_vertical is True
    T_grid = compute_grid_dp_grouped(
        K4, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        specs,
    )
    assert T_grid == T_engine
    assert T_grid.num_spanning_trees() == 54567559495680


def test_grid_grouped_3x3_K3_shared_interior():
    """Same shape but with K_3 cells (smaller, faster)."""
    g = _build_shared_k_grid(3, 3, 3)
    T_engine = _engine_T(g)
    K3 = Graph.from_networkx(nx.complete_graph(3))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1]}
    T_grid = compute_grid_dp_grouped(
        K3, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        _shared_specs(3, 3),
    )
    assert T_grid == T_engine


def test_grid_grouped_2x3_K4_shared():
    g = _build_shared_k_grid(2, 3, 4)
    T_engine = _engine_T(g)
    K4 = Graph.from_networkx(nx.complete_graph(4))
    K2 = Graph.from_networkx(nx.complete_graph(2))
    cell_anchor_groups = {0: [0], 1: [1]}
    T_grid = compute_grid_dp_grouped(
        K4, cell_anchor_groups, K2, [0], [1], K2, [0], [1],
        _shared_specs(2, 3),
    )
    assert T_grid == T_engine


# =============================================================================
# G. compute_cell_quotient_cycle_dp (engine-level dispatch)
# =============================================================================


@pytest.mark.slow
def test_cm2_cell_quotient_dp_matches_engine():
    """Cm2 polynomial via cell-quotient DP matches engine baseline."""
    dnx = pytest.importorskip("dwave_networkx")

    g = Graph.from_networkx(dnx.chimera_graph(2))
    assert g.node_count() == 32
    assert g.edge_count() == 80

    table = load_default_table()
    engine = SynthesisEngine(table=table, verbose=False)
    engine.skip_target_lookup = True
    engine_result = engine.synthesize(g)

    cq_poly = compute_cell_quotient_cycle_dp(g, table)
    assert cq_poly is not None, "Cell-quotient DP should fire on Cm2"
    assert cq_poly == engine_result.polynomial
    assert cq_poly.num_spanning_trees() == 11_686_511_179_538_104_320


def test_cell_quotient_dp_returns_none_on_non_cycle():
    g = Graph.from_networkx(nx.petersen_graph())
    table = load_default_table()
    assert compute_cell_quotient_cycle_dp(g, table) is None


def test_cell_quotient_dp_returns_none_on_simple_graphs():
    g = Graph.from_networkx(nx.complete_graph(5))
    table = load_default_table()
    assert compute_cell_quotient_cycle_dp(g, table) is None


# =============================================================================
# H. HAMILTONIAN PATH / CLOSING-EDGE HELPERS (interleaved DP)
# =============================================================================


def test_hamiltonian_path_grid_2x2():
    path = hamiltonian_path_grid(2, 2)
    assert path == [(0, 0), (0, 1), (1, 1), (1, 0)]


def test_hamiltonian_path_grid_3x3():
    path = hamiltonian_path_grid(3, 3)
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0),
                    (2, 0), (2, 1), (2, 2)]


def test_grid_path_and_closing_edges_2x2():
    path_edges, closing_edges, _ham = grid_path_and_closing_edges(2, 2)
    assert len(path_edges) == 3
    assert len(closing_edges) == 1
    closing = closing_edges[0]
    cells_involved = {closing[0], closing[1]}
    assert cells_involved == {(0, 0), (1, 0)}
    assert closing[2] == "vert"


def test_grid_path_and_closing_edges_3x3():
    path_edges, closing_edges, _ham = grid_path_and_closing_edges(3, 3)
    assert len(path_edges) == 8
    assert len(closing_edges) == 4
    for (a, b, _d) in closing_edges:
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# =============================================================================
# I. SYNTHETIC K_3 GRIDS — interleaved DP, fast
# =============================================================================


def _build_k3_grid_graph(rows: int, cols: int) -> Graph:
    nxg = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            base = 3 * (r * cols + c)
            nxg.add_edge(base + 0, base + 1)
            nxg.add_edge(base + 1, base + 2)
            nxg.add_edge(base + 0, base + 2)
    for r in range(rows):
        for c in range(cols - 1):
            base_l = 3 * (r * cols + c)
            base_r = 3 * (r * cols + c + 1)
            nxg.add_edge(base_l + 1, base_r + 1)
    for r in range(rows - 1):
        for c in range(cols):
            base_u = 3 * (r * cols + c)
            base_d = 3 * ((r + 1) * cols + c)
            nxg.add_edge(base_u + 0, base_d + 0)
    return Graph.from_networkx(nxg)


def _k3_template():
    K3 = Graph(list(range(3)), [(0, 1), (1, 2), (0, 2)])
    M1 = Graph(list(range(2)), [(0, 1)])
    return K3, M1


def test_interleaved_dp_2x2_k3_full():
    """2x2 K_3 grid: full poly == engine AND Kirchhoff cross-check."""
    from tutte.validation import count_spanning_trees_kirchhoff

    g = _build_k3_grid_graph(2, 2)
    T_engine = _engine_T(g)

    K3, M1 = _k3_template()
    T_ileaved, _stats = compute_grid_dp_interleaved(
        cell_template=K3,
        cell_anchor_groups_template={0: [0], 1: [1]},
        junction_template=M1,
        junction_anchors_A=[0],
        junction_anchors_B=[1],
        rows=2, cols=2,
        horiz_groups=(1, 1),
        vert_groups=(0, 0),
    )

    assert T_ileaved == T_engine
    assert T_ileaved.num_spanning_trees() == count_spanning_trees_kirchhoff(g)


def test_interleaved_dp_3x3_k3_full():
    """3x3 K_3 grid: full poly == engine AND Kirchhoff cross-check.
    Tests anchor-sharing via keep_shared + multi-closing-shared via keep_merged."""
    from tutte.validation import count_spanning_trees_kirchhoff

    g = _build_k3_grid_graph(3, 3)
    T_engine = _engine_T(g)

    K3, M1 = _k3_template()
    T_ileaved, _stats = compute_grid_dp_interleaved(
        cell_template=K3,
        cell_anchor_groups_template={0: [0], 1: [1]},
        junction_template=M1,
        junction_anchors_A=[0],
        junction_anchors_B=[1],
        rows=3, cols=3,
        horiz_groups=(1, 1),
        vert_groups=(0, 0),
    )

    assert T_ileaved == T_engine
    assert T_ileaved.num_spanning_trees() == count_spanning_trees_kirchhoff(g)


def test_interleaved_dp_2x2_k3_sigma_idr_path():
    """Force σ_idr enumeration path — cross-validates Rung 1 closing-step
    factorization vs the Bell(N) baseline."""
    from tutte.validation import count_spanning_trees_kirchhoff

    g = _build_k3_grid_graph(2, 2)
    T_engine = _engine_T(g)

    K3, M1 = _k3_template()
    T_sigma, _stats = compute_grid_dp_interleaved(
        cell_template=K3,
        cell_anchor_groups_template={0: [0], 1: [1]},
        junction_template=M1,
        junction_anchors_A=[0],
        junction_anchors_B=[1],
        rows=2, cols=2,
        horiz_groups=(1, 1),
        vert_groups=(0, 0),
        _force_sigma_idr=True,
    )

    assert T_sigma == T_engine
    assert T_sigma.num_spanning_trees() == count_spanning_trees_kirchhoff(g)


# =============================================================================
# J. CM2 INTERLEAVED DP (slow, ~1 minute)
# =============================================================================


def _build_cm2_graph():
    """4 K_{4,4} cells in 2x2 grid with M_4 horiz + vert junctions."""
    nxg = nx.Graph()
    for r in range(2):
        for c in range(2):
            base = 8 * (2 * r + c)
            for u in range(4):
                for v in range(4, 8):
                    nxg.add_edge(base + u, base + v)
    for r in range(2):
        cell_l = 8 * (2 * r + 0)
        cell_r = 8 * (2 * r + 1)
        for i in range(4):
            nxg.add_edge(cell_l + 4 + i, cell_r + 4 + i)
    for c in range(2):
        cell_u = 8 * c
        cell_d = 8 * (2 + c)
        for i in range(4):
            nxg.add_edge(cell_u + i, cell_d + i)
    return Graph.from_networkx(nxg)


@pytest.mark.slow
def test_interleaved_dp_cm2_full():
    """Cm2 (2x2 K_{4,4} grid): full polynomial == engine AND Kirchhoff
    cross-check. ~1 minute cold (engine ~50s, interleaved ~37s)."""
    from tutte.validation import count_spanning_trees_kirchhoff

    g = _build_cm2_graph()
    T_engine = _engine_T(g)

    K44 = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    M4 = Graph(list(range(8)), [(0, 4), (1, 5), (2, 6), (3, 7)])
    T_ileaved, _stats = compute_grid_dp_interleaved(
        cell_template=K44,
        cell_anchor_groups_template={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        junction_template=M4,
        junction_anchors_A=[0, 1, 2, 3],
        junction_anchors_B=[4, 5, 6, 7],
        rows=2, cols=2,
        horiz_groups=(1, 1),
        vert_groups=(0, 0),
    )

    assert T_ileaved == T_engine
    assert T_ileaved.num_spanning_trees() == count_spanning_trees_kirchhoff(g)
