"""Chain & Cycle Recurrence Algebra.

This module provides the **algebraic chain recurrence** evaluator for
cell-decomposable graphs whose cell-quotient is a linear path. The
Tutte polynomial of `T(chain_n)` satisfies an exact linear recurrence
in `n` with polynomial coefficients in `(x, y)` (re-derivation of
Noy & Ribò 2007 — see `tutte/docs/06_7_chain_recurrence_algebra.md`).

The primary practical use is **modular point evaluation**: given a
chain template (cell, connector) and integer point `(x_0, y_0, p)`,
compute `T(chain_n) mod p` in O(r³) integer ops setup + O(n) modular
muls, where `r` is the orbit count of the chain transfer matrix.

For full polynomial output, the recurrence on `TuttePolynomial`
objects grows quadratically in coefficient-size per step — NOT
faster than direct DP at moderate n. Use only for modular
evaluation or very large n.

## Public API

- `extract_chain_transfer_matrix(...)` — re-exported from research
- `compute_chain_recurrence_mod(...)` — modular point evaluation
- `is_chain_topology(spec)` — detect linear-path cell tree

## Validated templates

Five templates with bit-for-bit polynomial-level validation:
| Template | Cell | Connector | Order r |
|---|---|---|---|
| `k22_m2` | K_{2,2} | M_2 | 2 |
| `k33_m3` | K_{3,3} | M_3 | 3 |
| `k44_m4` | K_{4,4} | M_4 | 5 |
| `k4_m2` | K_4 | M_2 | 2 |
| `k5_m2` | K_5 | M_2 | 2 |

See `tutte/tests/test_chain_recurrence.py` for regression coverage.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

# Re-export the transfer-matrix extractor from research scripts.
# Long-term: lift this into roots/ itself; for now the research script
# is the source of truth and is well-tested.
from tutte.graph import Graph
from tutte.polynomial import TuttePolynomial
from tutte.research.scripts.extract_chain_transfer_matrix import (
    extract_chain_transfer_matrix,
)


__all__ = [
    "extract_chain_transfer_matrix",
    "faddeev_leverrier_charpoly_mod",
    "compute_chain_recurrence_mod",
    "compute_chain_full_poly_from_spec",
    "is_chain_topology",
]


# Module-level template cache for callers that don't thread an
# extract_cache. Keys: (cell_key, connector_key, frozen anchor groups,
# junction anchors, side groups). Without this, every modular call from
# `compute_tree_dp_simple_mod` re-extracts the polynomial transfer
# matrix (5-25 s/call); repeated calls on the same template share the
# extraction.
_MODULE_EXTRACT_CACHE: Dict[Tuple, Dict] = {}


def _template_cache_key(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    connector_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    left_anchor_group: int,
    right_anchor_group: int,
) -> Tuple:
    """Build a hashable cache key for a chain template.
    Uses canonical_key for the underlying graphs so isomorphic templates
    share entries.
    """
    cag = tuple(
        sorted((g, tuple(sorted(vs))) for g, vs in cell_anchor_groups.items())
    )
    return (
        cell_template.canonical_key(),
        connector_template.canonical_key(),
        cag,
        tuple(sorted(junction_anchors_A)),
        tuple(sorted(junction_anchors_B)),
        left_anchor_group,
        right_anchor_group,
    )


def compute_chain_full_poly_from_spec(spec) -> "Optional[TuttePolynomial]":
    """Compute T via the chain framework from a CellTreeSpec.

    Returns the full TuttePolynomial. Useful for engine dispatch when
    the cell tree is a linear path: extracts the chain transfer matrix
    from the existing path-DP infrastructure (via observer hook), then
    iterates the path-DP forward and divides by the accumulated divisor.

    This is **NOT** faster than the existing `compute_tree_dp_simple`
    for full-polynomial output (both grow quadratically in coefficient
    size per cell step). It provides:
      - A standalone entry point validating the chain framework path
      - Char poly cached across calls with the same (cell, connector)
        template — useful for repeated calls in benchmark sweeps

    Returns None if the spec is not a linear path or extraction fails.
    """
    from tutte.research.scripts.chain_recurrence_polynomial import chain_S_at_n
    from tutte.roots.rooted_tutte import divide_by_x_minus_1_power

    cell_tree = spec.cell_tree
    if not is_chain_topology(cell_tree):
        return None

    n_cells = cell_tree.number_of_nodes()
    if n_cells < 2:
        return None

    # Pick interior cell to extract template anchor groups (the chain
    # framework requires UNIFORM anchor groups across all cells).
    leaves = [n for n in cell_tree.nodes() if cell_tree.degree(n) == 1]
    path_order = list(nx.shortest_path(cell_tree, leaves[0], leaves[1]))
    if len(path_order) < 2:
        return None
    # Use first interior cell (or the leaf if only 2 cells) for template anchors.
    interior_idx = 1 if n_cells >= 3 else 0
    interior_cell = path_order[interior_idx]

    # Find left and right neighbors in the cell-tree path.
    interior_neighbors = list(cell_tree.neighbors(interior_cell))
    if len(interior_neighbors) < 1:
        return None
    # For chains, interior cells have exactly 2 neighbors; leaves have 1.
    cell_anchor_groups_raw = spec.cell_anchor_groups.get(interior_cell, {})
    if len(cell_anchor_groups_raw) < 1:
        return None

    # Translate the spec's per-neighbor anchor groups to the chain
    # framework's per-side groups. There are two distinct cases:
    #
    # 1. **Different-anchor cells** (e.g. the validated K_{4,4} chain test
    #    fixture where left = shore A, right = shore B): the interior
    #    cell has two DISTINCT anchor sets, one per neighbor. Use group 0
    #    for left, group 1 for right.
    #
    # 2. **Shared-anchor cells** (real Chimera Cm(1, n) interior cells —
    #    the same 4 K_{4,4} vertices serve as both the left and right
    #    junction anchors): the spec has the same anchor list for both
    #    neighbors. The chain framework's `left_anchor_group ==
    #    right_anchor_group` mode handles this (per the validated
    #    `extract_chain_transfer_matrix.py:k44_m4_chain` fixture which
    #    uses `right_anchor_group=0` for the a-a Chimera chain). Without
    #    this case, the divisor accounting (`divide_by_x_minus_1_power`)
    #    raises "Polynomial not divisible by (x-1) at y^j" because we
    #    over-count one (x-1) factor per step.
    nbr_ids = sorted(cell_anchor_groups_raw.keys())
    if len(nbr_ids) == 1:
        # Leaf cell — can't extract chain template from it.
        return None
    left_neighbor, right_neighbor = nbr_ids[0], nbr_ids[1]
    left_anchors = list(cell_anchor_groups_raw[left_neighbor])
    right_anchors = list(cell_anchor_groups_raw[right_neighbor])
    shared_anchors = sorted(left_anchors) == sorted(right_anchors)
    if shared_anchors:
        cell_anchor_groups = {0: left_anchors}
        right_group = 0
    else:
        cell_anchor_groups = {0: left_anchors, 1: right_anchors}
        right_group = 1

    # Junction anchors A/B come from the spec.
    junction_anchors_A = list(spec.junction_anchors_A)
    junction_anchors_B = list(spec.junction_anchors_B)

    try:
        extract = extract_chain_transfer_matrix(
            cell_template=spec.cell_template,
            cell_anchor_groups=cell_anchor_groups,
            connector_template=spec.junction_template,
            junction_anchors_A=junction_anchors_A,
            junction_anchors_B=junction_anchors_B,
            left_anchor_group=0,
            right_anchor_group=right_group,
            verify_position_invariance=False,
        )
    except Exception:
        return None

    div_per_step = extract["div_per_step"]
    div_terminal = (
        extract.get("div_terminal_step")
        or extract.get("div_per_step", div_per_step)
    )

    try:
        S_n, total_div = chain_S_at_n(
            extract["apply_step"],
            extract["apply_terminal_step"],
            extract["initial_state"],
            div_per_step,
            div_terminal,
            n_cells,
        )
    except Exception:
        return None

    if total_div > 0:
        return divide_by_x_minus_1_power(S_n, total_div)
    return S_n


def faddeev_leverrier_charpoly_mod(M: List[List[int]], p: int) -> List[int]:
    """Integer Faddeev-LeVerrier characteristic polynomial modulo prime `p`.

    Given an `n × n` integer matrix `M` and a prime `p`, returns
    `[1, c_1, c_2, ..., c_n]` where the char polynomial is
    `λ^n + c_1 λ^{n-1} + ... + c_n` (all mod p).

    The Cayley-Hamilton theorem then gives the recurrence:
        M^n + c_1 M^{n-1} + ... + c_n I ≡ 0 (mod p)

    Cost: O(n³) integer multiplications mod p.
    """
    n = len(M)
    M_k = [[m % p for m in row] for row in M]  # M_1 = M
    coeffs = [1]
    c_prev = (-sum(M_k[i][i] for i in range(n))) % p
    coeffs.append(c_prev)
    for k in range(2, n + 1):
        Mk_plus_c = [
            [(M_k[i][j] + (c_prev if i == j else 0)) % p for j in range(n)]
            for i in range(n)
        ]
        prod = [
            [sum(M[i][r] * Mk_plus_c[r][j] for r in range(n)) % p for j in range(n)]
            for i in range(n)
        ]
        trace_prod = sum(prod[i][i] for i in range(n)) % p
        k_inv = pow(k, p - 2, p)
        c_new = (-trace_prod * k_inv) % p
        coeffs.append(c_new)
        M_k = prod
        c_prev = c_new
    return coeffs


def is_chain_topology(cell_tree: nx.Graph) -> bool:
    """Detect linear-path cell tree (exactly 2 leaves, all other vertices degree 2)."""
    if cell_tree.number_of_nodes() < 2:
        return False
    if not nx.is_tree(cell_tree):
        return False
    leaves = [n for n in cell_tree.nodes() if cell_tree.degree(n) == 1]
    return len(leaves) == 2


def compute_chain_recurrence_mod(
    cell_template: Graph,
    cell_anchor_groups: Dict[int, List[int]],
    connector_template: Graph,
    junction_anchors_A: List[int],
    junction_anchors_B: List[int],
    left_anchor_group: int,
    right_anchor_group: int,
    n_cells: int,
    x_val: int,
    y_val: int,
    p: int,
    initial_S_mod: Optional[Dict[int, int]] = None,
    extract_cache: Optional[Dict] = None,
) -> Tuple[int, Dict]:
    """Modular evaluation of `T(chain_n) mod p` via chain recurrence.

    The chain recurrence has order `r = n_orbits` (computed by the
    underlying transfer-matrix extractor). The function:

    1. Extracts the polynomial transfer matrix `M(x, y)` of the chain
       (cached via `extract_cache` if provided)
    2. Evaluates `M` at `(x_val, y_val) mod p` to get an integer matrix
       `M_int`
    3. Computes the char poly of `M_int` mod p via Faddeev-LeVerrier
    4. Computes initial state-sum values `S_2, ..., S_{r+1}` if not
       provided in `initial_S_mod`
    5. Iterates the recurrence to `S_{n_cells}` mod p
    6. Divides by `(x-1)^total_div(n)` mod p to recover `T(chain_n) mod p`

    Args:
        cell_template, cell_anchor_groups, connector_template,
            junction_anchors_A, junction_anchors_B, left_anchor_group,
            right_anchor_group: chain template specification (same as
            `extract_chain_transfer_matrix`)
        n_cells: target chain length (>= 2)
        x_val, y_val, p: integer point + prime modulus
        initial_S_mod: optional dict {n: S_n mod p} for n in [2, r+1].
            If None, computed via direct DP (slow first time).
        extract_cache: optional dict caching transfer matrix extraction
            across multiple calls with same template.

    Returns:
        (T_value_mod_p, updated_extract_cache)
    """
    if n_cells < 2:
        raise ValueError(f"n_cells must be >= 2, got {n_cells}")
    if extract_cache is None:
        # Fall back to module-level cache so callers that don't thread an
        # extract_cache (e.g., `compute_tree_dp_simple_mod`) still
        # amortize the cold extraction across repeated point calls.
        tmpl_key = _template_cache_key(
            cell_template, cell_anchor_groups, connector_template,
            junction_anchors_A, junction_anchors_B,
            left_anchor_group, right_anchor_group,
        )
        if tmpl_key not in _MODULE_EXTRACT_CACHE:
            _MODULE_EXTRACT_CACHE[tmpl_key] = {}
        extract_cache = _MODULE_EXTRACT_CACHE[tmpl_key]

    # Stage 1: extract or fetch transfer matrix
    cache_key = "extract"  # caller can cache extracts across templates
    if cache_key not in extract_cache:
        extract = extract_chain_transfer_matrix(
            cell_template=cell_template,
            cell_anchor_groups=cell_anchor_groups,
            connector_template=connector_template,
            junction_anchors_A=junction_anchors_A,
            junction_anchors_B=junction_anchors_B,
            left_anchor_group=left_anchor_group,
            right_anchor_group=right_anchor_group,
            verify_position_invariance=False,
        )
        # Build M_poly (r × r polynomial transfer matrix)
        apply_step = extract["apply_step"]
        apply_terminal = extract["apply_terminal_step"]
        initial_state = extract["initial_state"]
        orbit_keys = sorted(initial_state.keys())
        r = len(orbit_keys)
        M_poly = [[TuttePolynomial.zero() for _ in range(r)] for _ in range(r)]
        # Build a terminal row-vector t_poly[j] =
        # sum_orbit (apply_terminal({unit at j})) — this reduces a state
        # vector to a scalar via dot product. Avoids per-point
        # polynomial DP for initial S values.
        t_poly = [TuttePolynomial.zero() for _ in range(r)]
        for j, k_in in enumerate(orbit_keys):
            unit_state = {k: TuttePolynomial.zero() for k in orbit_keys}
            unit_state[k_in] = TuttePolynomial.from_coefficients({(0, 0): 1})
            out_state = apply_step(unit_state)
            for i, k_out in enumerate(orbit_keys):
                M_poly[i][j] = out_state.get(k_out, TuttePolynomial.zero())
            term_state = apply_terminal(unit_state)
            sum_poly = TuttePolynomial.zero()
            for v in term_state.values():
                sum_poly = sum_poly + v
            t_poly[j] = sum_poly
        # Initial state polynomial vector — converted to mod p per call
        initial_state_poly = [initial_state.get(k, TuttePolynomial.zero()) for k in orbit_keys]
        extract_cache[cache_key] = {
            "extract": extract,
            "M_poly": M_poly,
            "t_poly": t_poly,
            "initial_state_poly": initial_state_poly,
            "r": r,
            "orbit_keys": orbit_keys,
        }

    extract = extract_cache[cache_key]["extract"]
    M_poly = extract_cache[cache_key]["M_poly"]
    t_poly = extract_cache[cache_key]["t_poly"]
    initial_state_poly = extract_cache[cache_key]["initial_state_poly"]
    r = extract_cache[cache_key]["r"]

    div_per_step = extract["div_per_step"]
    div_terminal = (
        extract.get("div_terminal_step")
        or extract.get("div_per_step", div_per_step)
    )

    # Stage 2: evaluate M, t, initial state mod p
    M_int_mod = [
        [int(M_poly[i][j].evaluate(x_val, y_val)) % p for j in range(r)]
        for i in range(r)
    ]
    t_int_mod = [int(t_poly[j].evaluate(x_val, y_val)) % p for j in range(r)]
    init_int_mod = [int(initial_state_poly[j].evaluate(x_val, y_val)) % p for j in range(r)]

    # Stage 3: char poly mod p
    char_mod = faddeev_leverrier_charpoly_mod(M_int_mod, p)

    # Stage 4: initial S values via DIRECT MODULAR matvec evolution.
    # S_n raw = t · M^(n-2) · init_state_vec  for n >= 2.
    # We need S_n for n in [2, r+1].
    if initial_S_mod is None:
        initial_S_mod = {}
        state_vec = list(init_int_mod)  # length r
        for nc in range(2, r + 2):
            # Apply (nc - 2) middle steps then a terminal step.
            # We've applied (nc - 3) middle steps already if nc > 2;
            # advance state_vec one step from previous iteration.
            if nc > 2:
                new_vec = [0] * r
                for i in range(r):
                    row = M_int_mod[i]
                    s = 0
                    for j in range(r):
                        s += row[j] * state_vec[j]
                    new_vec[i] = s % p
                state_vec = new_vec
            # Terminal step + sum: S_nc = t · state_vec
            S_nc = 0
            for j in range(r):
                S_nc += t_int_mod[j] * state_vec[j]
            initial_S_mod[nc] = S_nc % p

    # Stage 5: recurrence
    S_mod = dict(initial_S_mod)
    for n_idx in range(r + 2, n_cells + 1):
        v = 0
        for k in range(1, r + 1):
            v = (v - char_mod[k] * S_mod[n_idx - k]) % p
        S_mod[n_idx] = v

    # Stage 6: divide by (x-1)^total_div(n) mod p
    total_div = (n_cells - 2) * div_per_step + div_terminal
    xm1 = (x_val - 1) % p
    if xm1 == 0:
        raise ValueError(
            f"Cannot divide by (x-1) at x={x_val} mod p={p} — "
            f"the chain framework requires (x-1) ≠ 0 mod p."
        )
    xm1_inv = pow(xm1, -1, p)
    T_target = (S_mod[n_cells] * pow(xm1_inv, total_div, p)) % p

    return T_target, extract_cache
