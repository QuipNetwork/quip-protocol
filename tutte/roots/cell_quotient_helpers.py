"""Cell-quotient cycle DP helpers: M_precompute + orbit_convolve + caching.

Generic helpers for the cell-quotient cycle DP. Operate on the rooted-Tutte
partition framework with Aut-orbit compression. All generic — work for any
cell template + junction template combination, not D-Wave-specific.

Major optimizations (Phase 18.E.3.e Week 3):
1. Orbit-level M_precompute (143× speedup): pick rep_state ∈ O_state,
   iterate all P_junc, multiply by |O_state|. Validated mathematically.
2. Position-invariant `enumerate_partitions_cached` (7× speedup): cache
   canonical orbit partitions; relabel per call.
3. Raw-dict polynomial arithmetic in inner loops (3-5× speedup):
   skip TuttePolynomial encode/decode cycles inside hot loops.

Components-touching helper handles disconnected junctions (e.g., M_k matchings):
divisor for vertex-sum convolution is `(x-1)^{|S| - c_J(S)}` instead of
`(x-1)^{|S|-1}`, where c_J(S) is junction components touching shared boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..graph import Graph
from ..polynomial import TuttePolynomial
from .aut_orbit import (
    build_relabel_aut,
    canonical_partition,
    per_cell_canonical_key,
    per_cell_orbit_size,
)
from .rooted_tutte import (
    all_partitions,
    delta,
    join_partitions,
    restrict_partition,
)


def components_touching(template: Graph, boundary: List[int]) -> int:
    """Count components of template containing at least one boundary vertex.

    For vertex-sum convolution rank-shift: divisor (x-1)^{|S| - c_J(S)} where
    c_J(S) = number of junction components touching shared boundary S.

    Standard formula assumes connected junction (c_J=1, divisor=|S|-1).
    For disconnected junctions (M_k matchings have k components, each
    containing one shared vertex): divisor = (b - c_J).
    """
    nx_g = nx.Graph()
    nx_g.add_nodes_from(template.nodes)
    nx_g.add_edges_from(template.edges)
    boundary_set = set(boundary)
    count = 0
    for component in nx.connected_components(nx_g):
        if component & boundary_set:
            count += 1
    return count


def enumerate_partitions_per_orbit(
    boundary: List[int], aut_group: List[Dict[int, int]],
) -> Dict[Tuple, List[Tuple]]:
    """For each canonical orbit-rep, list all partitions of `boundary`."""
    out: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for p_list in all_partitions(list(boundary)):
        p_tuple = tuple(sorted(tuple(sorted(b)) for b in p_list))
        canonical = canonical_partition(p_tuple, aut_group)
        out[canonical].append(p_tuple)
    return dict(out)


# Cache for canonical orbit partitions, position-invariant.
_CANONICAL_ORBIT_CACHE: Dict[Tuple, Dict[Tuple, List[Tuple]]] = {}


def _relabel_partition(P: Tuple[Tuple[int, ...], ...],
                       pos_map: Dict[int, int]) -> Tuple[Tuple[int, ...], ...]:
    """Apply pos_map to all vertices in partition; return canonical sorted form."""
    return tuple(sorted(tuple(sorted(pos_map[v] for v in block)) for block in P))


def enumerate_partitions_cached(
    cell_template: Graph,
    cell_anchors_orig: List[int],
    boundary_pos: List[int],
    preserve_split_index: Optional[int] = None,
) -> Dict[Tuple, List[Tuple]]:
    """Position-invariant cached enumerate.

    Computes orbit partitions ONCE for canonical positions [0..n-1], then
    relabels to actual boundary_pos. Saves Bell(n)×|aut| work on repeat
    calls with same cell template / anchor structure but different positions.
    """
    boundary_size = len(boundary_pos)
    cache_key = (
        cell_template.canonical_key(),
        tuple(cell_anchors_orig),
        preserve_split_index,
        boundary_size,
    )
    if cache_key not in _CANONICAL_ORBIT_CACHE:
        canonical_pos = list(range(boundary_size))
        canonical_aut = build_relabel_aut(
            cell_template, cell_anchors_orig, canonical_pos,
            preserve_split_index=preserve_split_index,
        )
        _CANONICAL_ORBIT_CACHE[cache_key] = enumerate_partitions_per_orbit(
            canonical_pos, canonical_aut,
        )
    canonical_orbits = _CANONICAL_ORBIT_CACHE[cache_key]
    pos_map = {i: boundary_pos[i] for i in range(boundary_size)}
    return {
        _relabel_partition(ck, pos_map): [_relabel_partition(p, pos_map) for p in parts]
        for ck, parts in canonical_orbits.items()
    }


def precompute_M_table(
    state_orbit_partitions: Dict[Tuple, List[Tuple]],
    junction_orbit_partitions: Dict[Tuple, List[Tuple]],
    shared_boundary: List[int],
    extra_boundary: List[int],
    out_aut_group: List[Dict[int, int]],
    state_extra_boundary: Optional[List[int]] = None,
    keep_shared: bool = False,
    out_cell_anchor_groups: Optional[List[List[int]]] = None,
    state_cell_anchor_groups: Optional[List[List[int]]] = None,
) -> Dict[Tuple[Tuple, Tuple, Tuple], TuttePolynomial]:
    """Precompute M[O_state, O_junction, O_out] using Aut canonical form.

    ORBIT SHORTCUT: pick rep_state ∈ O_state, iterate all P_junc, multiply by
    |O_state|. Math: σ ∈ Aut_state preserving split induces σ' ∈ out_aut
    acting on state_left only; canonical(σ(P_out)) = canonical(P_out).
    Validated correct on K_4 and K_{4,4} cases.

    Optimization: precomputes (xy-1)^d as raw dicts, accumulates raw-dict
    arithmetic to avoid TuttePolynomial encode/decode cycles.

    Args:
        keep_shared: when True, shared_boundary positions are RETAINED in
            the output partition (the standard mode consumes them via the
            vertex-sum convolution, which marginalizes shared positions out).
            Used when the new piece's "right" anchors are physically the
            same vertices as state's existing shared boundary — i.e.,
            anchor-sharing cells (Cm3 interior). The shared positions
            persist for the next junction step.
        out_cell_anchor_groups: when provided, output canonical keys are
            computed via per-cell structural canonicalization (block-shape
            multiset over per-cell anchor groups). Coarser than aut-based
            grouping; orbits captured exactly under S_n^N (independent
            permutation within each cell's anchor group). Use this when
            state spans multiple cells and per-cell aut applies independently.
        state_cell_anchor_groups: when provided AND state_orbit_partitions
            holds only one rep per orbit, n_state is computed analytically
            via per_cell_orbit_size (avoiding the need to enumerate all
            partitions in the orbit explicitly).
    """
    if state_extra_boundary is None:
        state_extra_boundary = []
    full_universe = list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
    if keep_shared:
        out_boundary = (
            list(state_extra_boundary) + list(shared_boundary) + list(extra_boundary)
        )
    else:
        out_boundary = list(state_extra_boundary) + list(extra_boundary)

    max_d = len(shared_boundary)
    xy_minus_1_dict = {(1, 1): 1, (1, 0): -1, (0, 1): -1, (0, 0): 1}
    xy_powers_dict: List[Dict[Tuple[int, int], int]] = [{(0, 0): 1}]
    for k in range(1, max_d + 1):
        prev = xy_powers_dict[-1]
        new: Dict[Tuple[int, int], int] = defaultdict(int)
        for (i1, j1), c1 in prev.items():
            for (i2, j2), c2 in xy_minus_1_dict.items():
                new[(i1 + i2, j1 + j2)] += c1 * c2
        xy_powers_dict.append(dict(new))

    junc_data: Dict[Tuple, list] = {}
    for O_junc, pj_list in junction_orbit_partitions.items():
        per_junc = []
        for P_junc in pj_list:
            P_junc_S = restrict_partition(P_junc, shared_boundary)
            P_junc_ext_list = list(P_junc) + [(v,) for v in state_extra_boundary]
            P_junc_ext = tuple(sorted(P_junc_ext_list))
            per_junc.append((P_junc, P_junc_S, P_junc_ext))
        junc_data[O_junc] = per_junc

    # Try batched C extension when:
    # (a) per-cell canonicalization is requested (the most expensive case),
    # (b) state × junc total iterations exceed a threshold (otherwise marshal
    #     overhead exceeds savings).
    n_state_total = len(state_orbit_partitions)
    n_junc_total = sum(len(pj) for pj in junction_orbit_partitions.values())
    BATCH_C_THRESHOLD = 100  # low threshold to maximize C ext coverage
    if (out_cell_anchor_groups is not None
            and state_cell_anchor_groups is not None
            and n_state_total * n_junc_total >= BATCH_C_THRESHOLD):
        try:
            from ._partition_c import precompute_M_batched_inner_c
            n_state_per_orbit = {
                O_state: per_cell_orbit_size(O_state, state_cell_anchor_groups)
                for O_state in state_orbit_partitions
            }
            M_dict_c = precompute_M_batched_inner_c(
                state_orbit_partitions=state_orbit_partitions,
                junc_data_per_orbit=junc_data,
                state_extra_boundary=state_extra_boundary,
                extra_boundary=extra_boundary,
                shared_boundary=shared_boundary,
                out_boundary=out_boundary,
                out_cell_anchor_groups=out_cell_anchor_groups,
                n_state_per_orbit=n_state_per_orbit,
                xy_powers_dict=xy_powers_dict,
            )
            if M_dict_c is not None:
                # Convert per-cell canonical keys (tuples of shape tuples)
                # to TuttePolynomial values.
                M_out: Dict[Tuple, TuttePolynomial] = {}
                for key, val_dict in M_dict_c.items():
                    nonzero = {k: v for k, v in val_dict.items() if v != 0}
                    M_out[key] = TuttePolynomial.from_coefficients(nonzero)
                return M_out
        except Exception:
            pass  # fall back to Python

    M_dict: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    for O_state, ps_list in state_orbit_partitions.items():
        if state_cell_anchor_groups is not None:
            n_state = per_cell_orbit_size(O_state, state_cell_anchor_groups)
        else:
            n_state = len(ps_list)
        rep_state = ps_list[0]
        P_state_ext_list = list(rep_state) + [(v,) for v in extra_boundary]
        P_state_ext = tuple(sorted(P_state_ext_list))
        P_state_S = restrict_partition(rep_state, shared_boundary)

        for O_junc, _pj_list in junction_orbit_partitions.items():
            for (P_junc, P_junc_S, P_junc_ext) in junc_data[O_junc]:
                d = delta(P_state_S, P_junc_S, shared_boundary)
                if d < 0:
                    continue
                joint = join_partitions(P_state_ext, P_junc_ext, full_universe)
                P_out = restrict_partition(joint, out_boundary)
                if out_cell_anchor_groups is not None:
                    O_out = (
                        per_cell_canonical_key(P_out, out_cell_anchor_groups)
                        if out_boundary else ()
                    )
                else:
                    O_out = canonical_partition(P_out, out_aut_group) if out_boundary else ()
                target = M_dict[(O_state, O_junc, O_out)]
                for k, v in xy_powers_dict[d].items():
                    target[k] = target.get(k, 0) + v * n_state

    M: Dict[Tuple, TuttePolynomial] = {}
    for key, val_dict in M_dict.items():
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        M[key] = TuttePolynomial.from_coefficients(nonzero)
    return M


def _poly_to_dict(poly: TuttePolynomial) -> Dict[Tuple[int, int], int]:
    return {(i, j): c for i, j, c in poly.terms()}


# C extension for poly mul; falls back to Python on overflow / unavailable.
try:
    from .._polynomial_c import poly_mul as _poly_mul_dispatch
except Exception:
    _poly_mul_dispatch = None


def _dict_mul(d1: Dict[Tuple[int, int], int],
              d2: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    """Multiply two polynomial coefficient dicts.

    Uses C extension via `_polynomial_c.poly_mul` when available; falls
    back to pure Python on overflow or if extension unavailable.
    """
    if _poly_mul_dispatch is not None and d1 and d2:
        result = _poly_mul_dispatch(d1, d2)
        if result is not None:
            return result
    result_dict: Dict[Tuple[int, int], int] = defaultdict(int)
    for (i1, j1), c1 in d1.items():
        for (i2, j2), c2 in d2.items():
            result_dict[(i1 + i2, j1 + j2)] += c1 * c2
    return result_dict


def _dict_add(d1: Dict[Tuple[int, int], int],
              d2: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v
    return d1


def orbit_convolve(
    state_orbit_T: Dict[Tuple, TuttePolynomial],
    junction_orbit_T: Dict[Tuple, TuttePolynomial],
    M_table: Dict[Tuple, TuttePolynomial],
    out_orbit_sizes: Dict[Tuple, int],
) -> Dict[Tuple, TuttePolynomial]:
    """Convolve at orbit level.

    V_out[O_out] = (Σ V_state · T_junc · M) / |O_out|

    Optimization: raw coefficient dicts in hot loop to avoid encode/decode.
    """
    state_dict = {ok: _poly_to_dict(v) for ok, v in state_orbit_T.items()}
    junc_dict = {ok: _poly_to_dict(v) for ok, v in junction_orbit_T.items()}
    M_dict = {key: _poly_to_dict(v) for key, v in M_table.items()}

    # Group M entries by (O_state, O_junc). For each group, compute
    # `state * junc` exactly ONCE, then multiply by each M-table-entry per
    # O_out. This eliminates redundant `state * junc` poly muls when many
    # O_out share the same (O_state, O_junc) pair.
    out_dict: Dict[Tuple, Dict[Tuple[int, int], int]] = defaultdict(dict)
    grouped: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Dict[Tuple[int, int], int]]]] = (
        defaultdict(list)
    )
    for (O_state, O_junc, O_out), m_val in M_dict.items():
        grouped[(O_state, O_junc)].append((O_out, m_val))

    for (O_state, O_junc), out_entries in grouped.items():
        v1 = state_dict.get(O_state)
        v2 = junc_dict.get(O_junc)
        if v1 is None or v2 is None:
            continue
        sj = _dict_mul(v1, v2)
        if not sj:
            continue
        for (O_out, m_val) in out_entries:
            contrib = _dict_mul(sj, m_val)
            _dict_add(out_dict[O_out], contrib)

    final: Dict[Tuple, TuttePolynomial] = {}
    for O_out, val_dict in out_dict.items():
        size = out_orbit_sizes.get(O_out, 1)
        if size > 1:
            divided = {}
            for k, c in val_dict.items():
                if c % size != 0:
                    raise ValueError(
                        f"Non-divisible coeff {c} by orbit size {size}"
                    )
                if c != 0:
                    divided[k] = c // size
            val_dict = divided
        nonzero = {k: v for k, v in val_dict.items() if v != 0}
        final[O_out] = TuttePolynomial.from_coefficients(nonzero)
    return final
