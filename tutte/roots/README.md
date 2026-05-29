# `tutte/roots/` — rooted-Tutte composition for cell-decomposable graphs

Dynamic-programming methods that decompose a graph into **cells** plus
inter-cell **junctions**, compute each cell's **rooted Tutte
polynomial** (boundary-partition-indexed), and compose them via
vertex-sum convolution.

> **Why "roots"?** Every file here computes or composes _rooted Tutte
> polynomials_ `T_rooted(G, S)[P]` indexed by partitions `P` of a
> boundary set `S ⊆ V(G)`. The boundary vertices are the "roots" that
> anchor the polynomial to the rest of the graph during composition.
> The standard Tutte polynomial recovers via
> `T(G) = Σ_P T_rooted(G, S)[P]`.

## When the engine uses this package

`tutte/synthesis/engine.py::_synthesize_inner` dispatches to several
entry points in this package, each gated to `edge_count ≥ 60`:

| Entry point                                          | Trigger                                                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `compute_cell_quotient_grid_dp_streamed`             | 2D-grid cell-quotient of `K_{a,b}` cells with disjoint per-direction anchors (Cm_2 fits)      |
| `compute_cell_quotient_cycle_dp`                     | Cell-quotient is a simple cycle (Cm_2 cycle topology)                                         |
| `compute_cell_quotient_tree_dp`                      | Cell-quotient is a tree (`n` cells, `n − 1` junctions, no cycles)                              |
| `compute_cell_quotient_bipartite_junction_dp`        | Non-matching bipartite junctions (Z(m, t) families with multi-degree anchors)                  |
| `compute_bipartite_junction_per_component_dp`        | Splits a disconnected junction into per-component sub-junctions to bound joint-boundary Bell counts |
| `compute_cell_quotient_hybrid`                       | Chord-rule cycle-close + per-leaf synthesis for cyclic cell-quotients (Cm_3's 3×3 grid)         |

All dispatchers share the precondition: `try_hierarchical_partition`
(in `tutte/graphs/covering.py`) must return a cell decomposition. If
no decomposition is found or the cell-quotient topology doesn't match,
the entry returns `None` and the engine falls through.

## Module map

| File                                                                       | Purpose                                                                                                                                                                                       |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`__init__.py`](__init__.py)                                               | Public engine entries (`compute_cell_quotient_cycle_dp`, `compute_cell_quotient_tree_dp`, `compute_cell_quotient_grid_dp_streamed`) plus re-exports.                                          |
| [`rooted_tutte.py`](rooted_tutte.py)                                       | Brute-force `T_rooted` (cells of ≤ 16 edges); boundary primitives (`join_partitions`, `delta`, `restrict_partition`, `divide_by_x_minus_1_power`); persistent rooted-lookup load/save.        |
| [`aut_orbit.py`](aut_orbit.py)                                             | Aut-based orbit canonicaliser: `compute_cell_aut` (VF2), `canonical_partition` (lex-min over aut), `aut_compress_t_rooted`, `build_relabel_aut`. Generic over any cell with non-trivial aut.    |
| [`cell_anchor_adapter.py`](cell_anchor_adapter.py)                         | Graph-agnostic cycle / grid / tree topology detection plus per-cell anchor alignment (`normalize_cell_anchors_for_cycle`, `detect_cell_anchor_groups`, `extract_grid_specs`, `extract_path_specs`). |
| [`cell_quotient_helpers.py`](cell_quotient_helpers.py)                     | Hot-path helpers: `precompute_M_table` (orbit-level), `orbit_convolve` (raw-dict arithmetic + `_polynomial_c.poly_mul`), `enumerate_partitions_cached`, junction component count `c_J`.        |
| [`cell_quotient_cycle.py`](cell_quotient_cycle.py)                         | `compute_cycle_dp(cell_template, ..., n_cells)` — cycle-topology DP with explicit identification close.                                                                                       |
| [`cell_quotient_path.py`](cell_quotient_path.py)                           | `compute_path_dp(...)` — path-of-cells DP (no closing step). Returns the boundary-partition-indexed dict; consumed by `cell_quotient_grid.py`.                                                |
| [`cell_quotient_grid.py`](cell_quotient_grid.py)                           | `compute_grid_dp_streamed_kab(...)` — composes a 2D grid via row path DPs convolved through vertical junctions; streaming version avoids OOM on Cm_3-scale targets.                            |
| [`cell_quotient_tree.py`](cell_quotient_tree.py)                           | `compute_tree_dp_recursive` — branching cell-tree topology DP via post-order recursion (orbit-compressed). `CellTreeSpec` carries the topology + per-cell anchor groups.                       |
| [`cell_quotient_hybrid.py`](cell_quotient_hybrid.py)                       | `compute_cell_quotient_hybrid` — chord-rule cycle-close + per-leaf synth for cyclic cell-quotients (Cm_3's grid topology).                                                                    |
| [`cell_quotient_bipartite_junction.py`](cell_quotient_bipartite_junction.py) | `compute_cell_quotient_bipartite_junction_dp` + per-component variant. Generalises the k-matching path to non-matching bipartite junctions; per-component variant bounds joint Bell counts.    |
| [`chain_recurrence.py`](chain_recurrence.py)                               | Constructive re-derivation of Noy & Ribò (2007) for chain-of-cells families. Faddeev-LeVerrier mod p extracts the order-`r` recurrence in ms per modular point.                              |
| [`chord_junction_closed_form.py`](chord_junction_closed_form.py)            | Unified bivariate chord-junction theorem (matching `E_J`). Symmetric + asymmetric closed forms over `2^|V_k|` merger terms. Backed by persistent merger lookup table.                          |
| [`sokal_z_chord_junction.py`](sokal_z_chord_junction.py)                    | Sokal-Z generalization for **arbitrary** `E_J` (non-matching / multi-edge). Brute-force + per-H_J-component + edge-by-edge tree-DP enumeration; multi-point eval + bivariate Lagrange interpolation. Handles Z(1,2)-class junctions where the matching-only theorem fails. |
| [`signed_quotient.py`](signed_quotient.py)                                  | Live σ-finder `find_best_sigma` for the σ-equivariant chord-ordering path (used by the engine + `graphs/k_sum.py`). The test-only signed-DP-via-interpolation pipeline moved to [`../deprecated/`](../deprecated/README.md). |
| [`multivariate.py`](multivariate.py)                                        | Reference Sokal multivariate `Z(G; q, vₑ)` ring (`UniformZ`, `MultivariateTutte`) + the Z↔T identity that underpins the chord-junction theory. Used as a cross-check oracle in tests. (Moved here from the package root in the 2026-05 cleanup.) |
| [`_partition_c.py`](_partition_c.py)                                       | cffi C extension wrappers for partition / canonicalisation hot paths (`apply_perm_canonical_c`, `h_canonicalize_c_batched`, `precompute_M_batched_inner_c_mod`).                              |

## Algorithm overview

### Cycle DP — three phases

1.  **Path DP through `n − 1` junction-cell pairs.** Maintain
    `state_orbit_T` (orbit-compressed boundary-partition-indexed
    `T_rooted`) plus an accumulating `(x − 1)` divisor power. Each step
    convolves the current state with the next junction
    (`precompute_M_table` → `orbit_convolve`), then with the next cell.
2.  **Cycle close, step 1.** Convolve the state with the closing
    junction to a fresh boundary. The state now spans
    `state_left ∪ pos_cB_FRESH`.
3.  **Cycle close, step 2 — identification.** Identify
    `state_left[i] ≡ pos_cB_FRESH[i]` for each `i`. The chain-aware
    union-find formula

        T(cycle) = (x − 1)^{−a} · Σ_P ((x − 1)(y − 1))^{actually_same(P)} · T_rooted_int[P]

    where `actually_same(P) = a − n_merges(P)`, correctly accounts for
    identifications that chain through `P`'s blocks.

### Tree DP — post-order recursion

Composes `T(graph)` by post-order recursion over the cell tree: each
cell absorbs each child subtree via a junction step (vertex-sum
convolution at the junction's shared boundary) followed by a
cell-merge step (vertex-sum at the child's parent-facing boundary).

`CellTreeSpec` carries the cell template, junction template,
per-cell-anchor groups, and root choice. Per-cell orbit compression is
enabled by default in the recursive entry point.

For chain-shaped cell trees, `compute_chain_full_poly_from_spec` (in
`chain_recurrence.py`) can replace the entire DP with a forward
recurrence in the number of cells.

### Grid DP — row composition

`compute_grid_dp_streamed_kab` composes a `rows × cols` grid by
computing per-row `T_rooted` via path DP and convolving rows through
vertical junctions. Streaming `precompute_M_table` chunks state orbits
to bound peak memory; the modular variant
(`precompute_M_table_mod`) builds the M-table directly in `int mod p`,
no polynomial allocation per chunk.

### Disconnected-junction divisor

The standard vertex-sum convolution divisor is `(x − 1)^{|S| − 1}` per
junction step, which assumes the junction is a connected graph. For
matching junctions (`M_k` = `k` disjoint edges), the correct divisor
is `(x − 1)^{|S| − c_J(S)}` where `c_J(S)` is the number of junction
components touching the shared boundary. `components_touching` in
`cell_quotient_helpers.py` detects this automatically; all DPs use the
corrected formula.

### Performance optimisations

- **Aut orbit compression** — the cell template's automorphism group
  acts on partitions; storing one polynomial per orbit (not per
  partition) cuts state space by `|Aut|`. K_{4,4} compresses 4140
  partitions to ~43 orbits.
- **Orbit-level `precompute_M_table`** — pick one representative per
  state orbit, multiply contributions by orbit size.
- **Position-invariant orbit caching** —
  `enumerate_partitions_cached` computes orbits once for canonical
  positions `[0..n−1]` and relabels per call.
- **Raw-dict polynomial arithmetic** — inner loops accumulate
  `dict[(xpow, ypow)] → coeff` directly, skipping the
  `TuttePolynomial.encode/decode` cycle. Combined with the
  `_polynomial_c` C extension this gives a meaningful per-mul
  speedup.
- **Streaming junction enumeration** — `precompute_M_table` accepts
  compressed junction orbits and expands their per-cell orbit members
  internally, one orbit at a time. Combined with `out_cell_anchor_groups`
  on the 2a vertical-junction step (which lifts the state's per-cell
  aut through `M_k` edges to the output boundary) keeps the 2a output
  orbit-compressed.
- **Chunked row composition** (`precompute_M_and_convolve_streaming`)
  wraps `precompute_M_table` in a state-orbit chunk loop with a
  raw-dict accumulator, bounding peak memory.
- **Modular point-value DP** — `precompute_M_table_mod` and
  `precompute_and_convolve_c_mod` evaluate the DP in modular
  arithmetic; the full polynomial is recovered by bivariate Lagrange
  interpolation per prime and CRT combine across primes. The
  precision-safe path for graphs whose coefficients overflow `int64`.

## Generalising for shared anchors

The current cycle / grid DPs assume each cell has **disjoint** anchor
sets per junction. D-Wave Cm_3's interior cells break this assumption:
the same anchor group serves both junctions on a cell's side.

`cell_anchor_adapter.detect_cell_anchor_groups` returns a
`CellAnchorGroups` that carries per-cell anchor IDs, per-junction
group assignments, and `has_shared_anchors()`. Consuming this in
`compute_path_dp` and `compute_grid_dp_with_layout` to allow shared
boundary positions is the remaining work to unlock Cm_3 via this path.

## Testing

`tutte/tests/test_roots.py` covers the cycle / grid / tree /
interleaved DPs against the engine + Kirchhoff oracle.
`tutte/tests/test_cell_quotient_bipartite_junction.py` covers the
bipartite-junction DP including the per-component decomposition.
`tutte/tests/test_chain_recurrence.py` covers chain + cycle
recurrences (the Noy-Ribò re-derivation).

## Related docs

- [`tutte/docs/06_3_rooted_tutte_framework.md`](../docs/06_3_rooted_tutte_framework.md)
  — algebraic foundation for rooted Tutte composition
- [`tutte/docs/06_4_cell_quotient_cycle_dp.md`](../docs/06_4_cell_quotient_cycle_dp.md)
  — cycle DP deep dive
- [`tutte/docs/06_5_cell_quotient_grid_dp.md`](../docs/06_5_cell_quotient_grid_dp.md)
  — grid DP and streaming variants
- [`tutte/docs/06_6_cell_quotient_tree_dp.md`](../docs/06_6_cell_quotient_tree_dp.md)
  — tree DP and per-cell compression
- [`tutte/docs/06_7_chain_recurrence_algebra.md`](../docs/06_7_chain_recurrence_algebra.md)
  — chain recurrence framework
- [`tutte/docs/06_8_modular_arithmetic_pathways.md`](../docs/06_8_modular_arithmetic_pathways.md)
  — modular point-value paths and interpolation
- [`tutte/research/engine_workflow_primer.md`](../research/engine_workflow_primer.md)
  — first-principles entry into where the cell-quotient DPs (including
  the chain recurrence) sit in the engine cascade
