"""C extension correctness tests for the partition-DP fast path.

Two C entry points are validated against their pure-Python references:

- `precompute_and_aggregate_c_mod` matches `precompute_and_convolve_c_mod`
  (Python dict accumulator) bit-for-bit.
- `h_canonicalize_c_wrapper` (single) and `h_canonicalize_c_batched`
  (batched) compute the lex-min canonical form across H-permutations of
  a partition, matching `min(apply_perm_to_partition(P, h) for h in H)`.
"""
from __future__ import annotations

import itertools

import pytest

from tutte.roots._partition_c import (
    h_canonicalize_c_batched,
    h_canonicalize_c_wrapper,
    precompute_and_aggregate_c_mod,
    precompute_and_convolve_c_mod,
)
from tutte.roots.aut_orbit import apply_perm_to_partition


def _make_synthetic_inputs(p=1009):
    """Construct a tiny but non-trivial state/junction setup."""
    state_extra_boundary = []
    extra_boundary = [100, 101]
    shared_boundary = [10, 11]
    out_boundary = list(extra_boundary)

    P_state_a = (tuple(shared_boundary),)
    P_state_b = ((shared_boundary[0],), (shared_boundary[1],))

    state_orbit_partitions = {
        P_state_a: [P_state_a],
        P_state_b: [P_state_b],
    }

    P_junc_1 = (tuple(shared_boundary) + tuple(extra_boundary),)
    P_junc_1_S = (tuple(shared_boundary),)

    P_junc_2 = ((shared_boundary[0], extra_boundary[0]),
                (shared_boundary[1], extra_boundary[1]))
    P_junc_2_S = ((shared_boundary[0],), (shared_boundary[1],))

    junc_data_per_orbit = {
        P_junc_1: [(P_junc_1, P_junc_1_S, P_junc_1)],
        P_junc_2: [(P_junc_2, P_junc_2_S, P_junc_2)],
    }

    n_state_per_orbit = {P_state_a: 1, P_state_b: 1}
    state_orbit_T_mod = {P_state_a: 3, P_state_b: 5}
    junction_orbit_T_mod = {P_junc_1: 7, P_junc_2: 11}

    xy_pow_mod = [pow(((2 - 1) * (3 - 1)) % p, d, p)
                  for d in range(len(shared_boundary) + 1)]
    out_cell_anchor_groups = [extra_boundary]

    return dict(
        state_orbit_partitions=state_orbit_partitions,
        junc_data_per_orbit=junc_data_per_orbit,
        state_extra_boundary=state_extra_boundary,
        extra_boundary=extra_boundary,
        shared_boundary=shared_boundary,
        out_boundary=out_boundary,
        out_cell_anchor_groups=out_cell_anchor_groups,
        n_state_per_orbit=n_state_per_orbit,
        state_orbit_T_mod=state_orbit_T_mod,
        junction_orbit_T_mod=junction_orbit_T_mod,
        xy_pow_mod=xy_pow_mod,
        p=p,
    )


def test_matches_r17_synthetic():
    """hash-map aggregation (C ext) bit-for-bit matches Python dict accumulator."""
    kwargs = _make_synthetic_inputs()
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 is not None and r18 is not None
    assert r17 == r18


def test_handles_empty_input():
    """Empty state/junction yields {} from both Python and C-ext paths."""
    kwargs = _make_synthetic_inputs()
    kwargs["state_orbit_partitions"] = {}
    kwargs["state_orbit_T_mod"] = {}
    kwargs["n_state_per_orbit"] = {}
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 == r18


def test_handles_zero_jv():
    """All zero junction values → both paths return {}."""
    kwargs = _make_synthetic_inputs()
    P_junc_1 = list(kwargs["junction_orbit_T_mod"].keys())[0]
    P_junc_2 = list(kwargs["junction_orbit_T_mod"].keys())[1]
    kwargs["junction_orbit_T_mod"] = {P_junc_1: 0, P_junc_2: 0}
    r17 = precompute_and_convolve_c_mod(**kwargs)
    r18 = precompute_and_aggregate_c_mod(**kwargs)
    assert r17 == r18
    assert len(r18) == 0


def test_cm2_modular_point_matches_r17():
    """End-to-end: Cm_2 T(2,3) mod 1009 = 806 via both Python and C-ext paths."""
    import os
    try:
        import dwave_networkx as dnx
    except ImportError:
        pytest.skip("dwave_networkx unavailable")
    from tutte.research.scripts.cm_modular_interp import (
        build_polynomial_state, evaluate_modular_at_point,
    )

    # Build state once; evaluate under Python then C-ext path.
    state = build_polynomial_state(2)

    import importlib
    import tutte.roots.cell_quotient_helpers as cqh

    os.environ["TUTTE_R18_AGGREGATE"] = "0"
    importlib.reload(cqh)
    v_r17 = evaluate_modular_at_point(state, 2, 3, 1009)

    os.environ["TUTTE_R18_AGGREGATE"] = "1"
    importlib.reload(cqh)
    v_r18 = evaluate_modular_at_point(state, 2, 3, 1009)

    assert v_r17 == 806
    assert v_r18 == 806
    assert v_r17 == v_r18


# =============================================================================
# H-canonicalize tests
# =============================================================================


def _s_n_perms(n):
    """All permutations in S_n as dicts on positions 0..n-1."""
    return [dict(zip(range(n), p)) for p in itertools.permutations(range(n))]


def test_h_canon_single_matches_python_s4():
    """Single-partition H-canonicalize matches Python reference under S_4."""
    H = _s_n_perms(4)
    universe = [0, 1, 2, 3]
    cases = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 1, 2),),
        ((0,), (1,), (2,), (3,)),
        ((0, 1, 2, 3),),
    ]
    for P in cases:
        py = min(apply_perm_to_partition(P, h) for h in H)
        c = h_canonicalize_c_wrapper(P, H, universe)
        assert c == py, f"Mismatch for P={P}: py={py}, c={c}"


def test_h_canon_non_zero_indexed_positions():
    """H-canonicalize handles non-zero-indexed universe (matches Python)."""
    universe = [10, 20, 30, 40, 50]
    H = _s_n_perms(5)
    H_pos = [{universe[i]: universe[h[i]] for i in range(5)} for h in H]
    P = ((10, 20), (30, 40), (50,))
    py = min(apply_perm_to_partition(P, h) for h in H_pos)
    c = h_canonicalize_c_wrapper(P, H_pos, universe)
    assert c == py


def test_h_canon_empty_H_returns_input():
    """Empty H acts as identity → returns canonical form of P."""
    P = ((1, 2), (3,))
    c = h_canonicalize_c_wrapper(P, [], [1, 2, 3])
    assert c == ((1, 2), (3,))


def test_h_canon_batched_matches_single_calls():
    """Batched H-canonicalize matches per-call h_canonicalize_c_wrapper."""
    universe = [0, 1, 2, 3, 4]
    H = _s_n_perms(5)
    P_list = [
        ((0, 1), (2, 3, 4)),
        ((0, 2), (1, 4), (3,)),
        ((0, 4), (1, 2, 3)),
        ((0,), (1,), (2,), (3,), (4,)),
    ]
    single = [h_canonicalize_c_wrapper(P, H, universe) for P in P_list]
    batched = h_canonicalize_c_batched(P_list, H, universe)
    assert batched == single


def test_h_canon_batched_matches_python_reference():
    """Batched C-ext matches Python `min(apply_perm…)` reference."""
    universe = [0, 1, 2, 3, 4, 5]
    # Use a subgroup of S_6: rotations by 2 positions (cycle 0,2,4 and 1,3,5)
    H = []
    for k in range(3):
        H.append({i: (i + 2 * k) % 6 for i in range(6)})
    P_list = [
        ((0, 1), (2, 3), (4, 5)),
        ((0, 3), (1, 4), (2, 5)),
        ((0, 1, 2), (3, 4, 5)),
    ]
    py = [min(apply_perm_to_partition(P, h) for h in H) for P in P_list]
    c = h_canonicalize_c_batched(P_list, H, universe)
    assert c == py
