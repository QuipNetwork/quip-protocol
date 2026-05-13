# `tutte/roots/` — rooted-Tutte composition for cell-decomposable graphs

This package houses dynamic-programming methods that decompose a graph
into **cells** plus inter-cell **junctions**, compute each cell's
**rooted Tutte polynomial** (boundary-partition-indexed), and compose
them via vertex-sum convolution.

> **Why "roots"?** Every file here computes or composes
> *rooted Tutte polynomials* `T_rooted(G, S)[P]` indexed by partitions
> `P` of a boundary set `S` ⊆ V(G). The boundary vertices are the
> "roots" that anchor the polynomial to the rest of the graph during
> composition. The standard Tutte polynomial recovers via
> `T(G) = Σ_P T_rooted(G, S)[P]`.

## When the engine uses this package

`tutte/synthesis/engine.py` step **7.7** dispatches to
`compute_cell_quotient_cycle_dp(graph, table)`. It fires when:

1. `try_hierarchical_partition` returns a usable cell decomposition
   (the graph divides into N cells of identical structure).
2. The cell-quotient graph is a **simple cycle** (every cell has
   degree 2 in the quotient).
3. Per-cell anchor sets can be aligned to the cell template (handled
   by `cell_anchor_adapter.normalize_cell_anchors_for_cycle`).

If any of these fail, the entry returns `None` and the engine falls
through to treewidth DP (step 8).

A grid-DP path is staged behind the same dispatch but the engine
wiring is pending an anchor-sharing-aware adapter (Phase 18.E.3.j —
the change that unlocks D-Wave Cm₃). See "Generalizing for shared
anchors" below.

## Module map

| File | Purpose |
|---|---|
| [`__init__.py`](__init__.py) | Public engine entry `compute_cell_quotient_cycle_dp(graph, table)` plus re-exports. |
| [`rooted_tutte.py`](rooted_tutte.py) | Brute-force `T_rooted` (cells of ≤ 16 edges) plus boundary primitives: `join_partitions`, `delta`, `restrict_partition`, `divide_by_x_minus_1_power`, `t_rooted_cached`. |
| [`aut_orbit.py`](aut_orbit.py) | Aut-based orbit canonicalizer — `compute_cell_aut` (VF2), `canonical_partition` (lex-min over aut), `aut_compress_t_rooted`, `build_relabel_aut`. Generic over any cell with non-trivial automorphism. |
| [`cell_quotient_helpers.py`](cell_quotient_helpers.py) | Hot-path helpers: `precompute_M_table` (orbit-level), `orbit_convolve` (raw-dict arithmetic + C extension via `_polynomial_c.poly_mul`), `enumerate_partitions_cached`, `components_touching` (auto-detects junction component count `c_J` for the disconnected-junction divisor fix). |
| [`cell_quotient_cycle.py`](cell_quotient_cycle.py) | `compute_cycle_dp(cell_template, ..., n_cells)` — cycle-topology DP. Validated on K₃, K₄, K_{4,4} synthetic cycles + real Cm2 (T(1,1) matches engine + Kirchhoff). |
| [`cell_quotient_path.py`](cell_quotient_path.py) | `compute_path_dp(...)` — same as cycle DP but without the closing step. Returns the boundary-partition-indexed dict for the full path; consumed by grid DP. |
| [`cell_quotient_grid.py`](cell_quotient_grid.py) | `compute_grid_dp_with_layout(...)` — composes a `(rows × cols)` grid via row path DPs convolved through vertical junctions. Validated on synthetic K_n grids with K_2 and M_k vertical junctions (8 passing tests). |
| [`cell_quotient_tree.py`](cell_quotient_tree.py) | Two entry points: (1) `compute_tree_dp_recursive(spec, enable_per_cell_compression=False)` — branching cell-tree topology DP via post-order recursion (orbit-compressed). (2) `compute_corrected_leaf_dp(spec)` — brute-force DP for chord-rule leaves with cross-cell vertex identifications, using the corrected `(y-1)^{k-m}/(x-1)^{m-c_comp}` convolution rule (RESOLVED 2026-05-07). See [docs/06_6_cell_quotient_tree_dp.md](../docs/06_6_cell_quotient_tree_dp.md) and `tutte/research/data/step3_milestone_b_design.md`. |
| [`cell_quotient_hybrid.py`](cell_quotient_hybrid.py) | `compute_cell_quotient_hybrid(graph, table)` — orbit-aware hybrid DP for cyclic cell-quotients. Peels closing junctions via Phase 13 §4 chord rule; Path A handles symmetric junctions with the standard `C(k, j)` formula, Path B uses canonical_key orbit enumeration for asymmetric (cross-junction shared anchors, e.g., 2x2 K_3 M_2 grid). Top-level spec dispatch to `compute_corrected_leaf_dp` bypasses per-leaf engine.synthesize when leaves admit cross-cell-ID structure. See `tutte/research/data/hybrid_cycle_close_findings.md`. |
| [`cell_anchor_adapter.py`](cell_anchor_adapter.py) | `normalize_cell_anchors_for_cycle` — graph-agnostic detection of cycle topology + per-cell anchor alignment. Bipartite-cell shortcut for K_{a,b} cells; generic VF2-Aut alignment is staged for Phase 18.E.3.j. |

## Algorithm overview

The cycle DP composes `n` cells `C_0, ..., C_{n-1}` (all isomorphic
to `cell_template`) connected in a cycle by `n` junctions `J_0, ...,
J_{n-1}` (all isomorphic to `junction_template`). The grid DP
generalizes to a `rows × cols` grid by computing per-row T_rooted via
path DP and convolving rows through vertical junctions.

### Cycle DP — three phases

1. **Path DP through `n − 1` junction-cell pairs.** Maintain
   `state_orbit_T` (orbit-compressed boundary-partition-indexed
   T_rooted) plus an accumulating `(x − 1)` divisor power. Each step
   convolves the current state with the next junction (`M_precompute`
   + `orbit_convolve`), then with the next cell.
2. **Cycle close, step 1.** Convolve state with the closing junction
   to a fresh boundary `pos_cB_FRESH`. Now state spans
   `state_left ∪ pos_cB_FRESH`.
3. **Cycle close, step 2 — identification.** Identify
   `state_left[i] ≡ pos_cB_FRESH[i]` for each `i`. The chain-aware
   union-find formula

       T(cycle) = (x − 1)^{−a} · Σ_P ((x − 1)(y − 1))^{actually_same(P)} · T_rooted_int[P]

   where `actually_same(P) = a − n_merges(P)` correctly accounts for
   identifications that chain through P's blocks (this was the a > 1
   bug fix from Phase 18.E.3.e Week 1).

### Disconnected-junction divisor

The standard vertex-sum convolution divisor is `(x − 1)^{|S| − 1}`
per junction step, which assumes the junction is a *connected*
graph. For matching junctions (M_k = k disjoint edges), the correct
divisor is `(x − 1)^{|S| − c_J(S)}` where `c_J(S) = ` number of
junction components touching the shared boundary. `components_touching`
detects this automatically; the cycle/grid DPs use the corrected
formula.

### Performance optimizations

- **Aut orbit compression** — cell template's automorphism group acts
  on partitions. Storing one polynomial per orbit (not per partition)
  cuts state space by `|Aut|`. K_{4,4} compresses 4140 partitions →
  43 orbits.
- **Orbit-level `M_precompute`** — pick one representative per state
  orbit, multiply contributions by orbit size. ~143× speedup on
  K_{4,4} cycles.
- **Position-invariant orbit caching** — `enumerate_partitions_cached`
  computes orbits once for canonical positions `[0..n−1]`, relabels
  per call.
- **Raw-dict polynomial arithmetic** — inner loops accumulate
  `dict[(xpow, ypow)] -> coeff` directly, skipping the
  `TuttePolynomial.encode/decode` cycle. Combined with the
  `_polynomial_c` C extension this gives 1.3-12.8× per-mul speedup.
- **Streaming junction enumeration** (`enumerate_junction_internally`,
  Phase B Round 6, May 2026) — for 2D K_{4,4} grid composition,
  `precompute_M_table` accepts compressed junction orbits (one rep
  each) and expands their per-cell orbit members **internally**, one
  orbit at a time. Avoids the OOM that bit external full enumeration
  on Cm₃, and combined with `out_cell_anchor_groups` on the 2a vertical
  junction step (which lifts state's per-cell aut through M_4 edges to
  the output boundary) keeps the 2a output orbit-compressed. Result:
  T(Cm₂) computed in ~36 s vs ~55 s for the engine's `kmatching_formula`
  baseline (1.5× win, first known method to beat that path on Cm₂).
  Reference recipe at
  [`tutte/research/scripts/cm2_via_v5_streamed.py`](../research/scripts/cm2_via_v5_streamed.py);
  correctness test at
  `test_precompute_M_table_internal_junction_enumeration_matches_external`.

- **Chunked row composition** (`precompute_M_and_convolve_streaming`,
  Phase B Round 9-10, May 2026) — wraps `precompute_M_table` in a
  state-orbit chunk loop with a raw-dict accumulator. Bounds Cm₃ peak
  memory (~1.9 GB) while exposing per-chunk `_dict_mul` (the polynomial
  state × junc × M coefficient multiply) as the irreducible cost.

- **Modular point-value DP** (Phase B Rounds 12-13, May 2026) — the
  precision-safe path for Cm₃-class graphs where Cm₂ polynomial
  coefficients fit in int64 but Cm₃'s don't (max coefficient ~10⁴⁰).
  `TuttePolynomial.evaluate_mod` + bivariate Lagrange + CRT recover
  the exact polynomial from grid evaluations.
  `precompute_M_table_mod` (Round 13) builds the M-table directly in
  modular arithmetic — one `int mod p` per `(O_state, O_junc, O_out)`
  triple, no polynomial allocation per chunk. Reference scripts:
  [`cm2_via_modular_interp.py`](../research/scripts/cm2_via_modular_interp.py),
  [`cm2_via_modular_dp.py`](../research/scripts/cm2_via_modular_dp.py),
  [`cm3_via_modular_dp.py`](../research/scripts/cm3_via_modular_dp.py).
  Cm₂ recovery is bit-exact in seconds; Cm₃ single-point in pure
  Python is ~2-3 hr (bottleneck moved to per-pair structural ops),
  with Round 14 C extension expected to give ~10× speedup. See
  [docs/06_5_cell_quotient_grid_dp.md](../docs/06_5_cell_quotient_grid_dp.md)
  Rounds 7-13 section for the full progression.

## Generalizing for shared anchors (Phase 18.E.3.j)

The current cycle/grid DPs assume each cell has **disjoint** anchor
sets per junction (e.g., Cm2's K_{4,4} cells split anchors as 4 ∪ 4
across two horizontal junctions). D-Wave Cm₃'s **interior cells**
break this assumption: cell 4's horizontal anchors `{32, 37, 38, 39}`
serve **both** junctions `(4, 3)` and `(4, 5)`, and its vertical
anchors `{33, 34, 35, 36}` serve both `(4, 1)` and `(4, 7)`.

The genericization principle: a cell has **K named anchor groups**;
each junction declares which group on each side it uses. Two junctions
that name the same group on the same cell **share** the underlying
vertex set. This generalizes cleanly beyond Cm₃: Pm₃ junctions,
random graphs whose cell-quotient has anchor sharing, and any future
cell-decomposable target with reused anchor groups all benefit from
the same change.

### Detection (shipped)

`cell_anchor_adapter.detect_cell_anchor_groups(partition, inter_edges)`
returns a `CellAnchorGroups` carrying:

* `cell_groups[i]` — list of `(group_id, sorted vertex tuple)` for
  cell `i`, with stable per-cell IDs.
* `junction_groups` — `[(cell_a, cell_b, group_a, group_b), ...]`
  per inter-cell junction.
* `has_shared_anchors()` — `True` iff any group serves more than one
  junction on the same cell.

`cell_anchor_adapter.extract_path_specs(spec, cells_in_path)` gives
one `CellRowSpec` per cell in path order (`left_group`, `right_group`,
`extra_groups`, `has_shared_horizontal`). Validated on Cm₂ (no
sharing), synthetic K_n grids (no sharing), and Cm₃ (interior cell
in middle row has `has_shared_horizontal == True`) — see
`tutte/tests/test_anchor_groups.py`.

### Consumption (next step)

The remaining work is rewriting `compute_path_dp` and
`compute_grid_dp_with_layout` to accept `List[CellRowSpec]` instead
of the disjoint-anchor `cell_left/right/up/down` lists. The
algorithmic change:

1. When a cell has `has_shared_horizontal`, the relabel map places
   its "right" anchors at the SAME canonical positions as its "left"
   anchors — no fresh boundary positions added.
2. `state_right` after that cell's convolution is identically
   `state_left`.
3. The next junction convolves at the same shared-boundary positions
   as the previous junction (one boundary, two junctions).
4. The disconnected-junction divisor `c_J` is unchanged.

## Testing

`tutte/tests/test_cell_quotient_dp.py` covers the engine integration
on Cm2 and the cycle DP on small synthetic cycles.
`tutte/tests/test_cell_quotient_grid.py` covers grid DP on
2×2 / 2×3 / 3×2 / 3×3 K_4 grids and 2×1 / 2×2 K_6 grids with K_2 and
M_2 verticals (8 tests, all passing).
