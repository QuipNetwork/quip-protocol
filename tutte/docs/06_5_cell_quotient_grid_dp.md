# 6.5 — Cell-Quotient Grid Dynamic Programming

## Summary

Generalizes the [Cell-Quotient Cycle DP](06_4_cell_quotient_cycle_dp.md)
from cycle topology to **2D grid topology**. For graphs whose
hierarchical decomposition has cell-quotient `(rows × cols)` grid
structure, this DP composes T(graph) by:

1. Computing `T_rooted` of each row via path DP (cycle DP minus the
   close step), with vertical anchors as `state_extra_boundary`
   (persistent through the row's path DP).
2. Composing rows via vertex-sum convolution at vertical-junction
   shared boundaries.
3. Marginalizing remaining boundary positions; dividing by the
   accumulated `(x − 1)^total_div`.

**Status:** algorithm validated on synthetic K_n grids (8 passing
regression tests including 2×2 / 2×3 / 3×2 / 3×3 K_4 with K_2 verticals,
2×2 K_6 with K_2 and M_2 verticals); engine integration is **pending**
the anchor-sharing-aware adapter (Phase 18.E.3.j) needed for D-Wave
Cm₃.

## When it is used

Once integrated, the dispatch will fire from engine step 7.7 *before*
falling through to treewidth DP, when:

1. `try_hierarchical_partition(graph, table)` returns a valid cell
   decomposition.
2. The cell-quotient graph is a 2D grid (detected by
   `is_grid_topology`).
3. Per-cell anchor groups can be aligned to a canonical layout
   (top/edge/interior cell varieties detected by the adapter).

## Algorithm — row-by-row composition

```
def compute_grid_dp_with_layout(
    cell_template, cell_left, cell_right, cell_up, cell_down,
    horiz_junction, vert_junction, rows, cols
):
    # 1. Per-row T_rooted via path DP. Vertical anchors persist as state_extra_boundary.
    row_T = []
    for r in range(rows):
        if r == 0:                extras = cell_down                # top row
        elif r == rows - 1:       extras = cell_up                  # bottom row
        else:                     extras = cell_up + cell_down      # middle row
        row_T.append(compute_path_dp(cell_template, cell_left, cell_right, extras,
                                      horiz_junction, ..., n_cells=cols))

    # 2. Compose rows. State after row 0 = row_T[0]. For each row r >= 1:
    state_T = row_T[0]
    state_persistent_extras = row_left_pos(0) + row_right_pos(0)
    for r in range(1, rows):
        # 2a. Vertical junction (r-1) → r: cols disjoint copies of vert_junction.
        state_T = orbit_convolve(state_T, T_combined_vert_junction, M_vert)
        total_div += cols * v_a - cols * c_J(vert_junction, vert_A)

        # 2b. Convolve with row r's T_rooted at shared = row r's up positions.
        state_T = orbit_convolve(state_T, row_T[r], M_row)
        total_div += cols * v_b - 1
        state_persistent_extras += row_left_pos(r) + row_right_pos(r)

    # 3. Marginalize all boundary positions; divide by (x-1)^total_div.
    T_total = sum(state_T.values())
    return divide_by_x_minus_1_power(T_total, total_div)
```

## Path DP — cycle DP minus the close step

`compute_path_dp` (`tutte/roots/cell_quotient_path.py`) is structurally
the cycle DP from [technique 6.4](06_4_cell_quotient_cycle_dp.md)
*without* the closing identification phase. It returns the
boundary-partition-indexed dict for the full path along with the
accumulated `(x − 1)` divisor power.

The path DP is parameterized by **`extras`** — additional cell
anchors that are *not* consumed by the path's left/right horizontal
junctions but persist as `state_extra_boundary` through every step.
For a row in a grid, those extras are the cell's vertical anchors.

## Combined vertical junctions

Each adjacent row pair shares `cols` independent vertical junctions
(one per column). The grid DP packs them into a **combined junction
graph** = `cols` disjoint copies of `vert_junction_template`, then
treats the row-row composition as a single vertex-sum at the combined
boundary.

The disconnected-junction divisor `c_J` is multiplied accordingly:
`combined_c_J = cols * single_c_J`. For Cm₃-shape grids with M_4
vertical junctions, single junction `c_J = 4`, combined `c_J = 4 cols`.

## Generic anchor-group detection

`tutte/roots/cell_anchor_adapter.py:detect_cell_anchor_groups` provides
the generic detection of per-cell anchor groups + per-junction group
mapping. It works on any cell-decomposable graph regardless of
whether the cells share anchors or not.

```python
spec = detect_cell_anchor_groups(partition, inter_edges)
# spec.cell_groups[i] = [(group_id, sorted vertex tuple), ...]
# spec.junction_groups = [(cell_a, cell_b, group_a, group_b), ...]
spec.has_shared_anchors()  # True iff any group serves > 1 junction on the same cell
```

For path / row composition,
`extract_path_specs(spec, cells_in_path)` returns one `CellRowSpec`
per cell in the path order:

```python
@dataclass(frozen=True)
class CellRowSpec:
    cell: int
    left_group: Optional[int]   # group used by junction to PREV cell in path
    right_group: Optional[int]  # group used by junction to NEXT cell in path
    extra_groups: Tuple[int]    # groups used by junctions OUTSIDE the path
    has_shared_horizontal      # True iff left_group == right_group
```

For Cm₂ row 0 (cells 0, 1) with disjoint K_{4,4} anchors: every cell
has `left_group != right_group` → `has_shared_horizontal == False`.
For Cm₃ row 1 (cells 3, 4, 5): interior cell 4 has
`left_group == right_group` → `has_shared_horizontal == True`. The
test in `tutte/tests/test_anchor_groups.py:test_extract_path_specs_cm3_middle_row_shared`
locks this in.

## Limitation: anchor sharing in path / grid DP

`compute_grid_dp_with_layout` currently requires **disjoint** anchor
groups per cell:

- `cell_left ∩ cell_right = ∅` (horizontal anchors split between
  prev-cell and next-cell junctions)
- `cell_up ∩ cell_down = ∅` (vertical anchors split between prev-row
  and next-row junctions)
- `(cell_left ∪ cell_right) ∩ (cell_up ∪ cell_down) = ∅` (horizontal
  and vertical anchor sets are disjoint)

**D-Wave Cm₃ violates these.** Cm₃ interior cell 4 has 8 vertices
split into two 4-element sets:

- Set A `{32, 37, 38, 39}` serves *both* horizontal junctions (4, 3)
  and (4, 5) — `cell_left == cell_right` for interior cells.
- Set B `{33, 34, 35, 36}` serves *both* vertical junctions (4, 1)
  and (4, 7) — `cell_up == cell_down` for interior cells.

The genericization principle (Phase 18.E.3.j) is to allow each cell
to carry **K named anchor groups**; each junction declares which
group on each side it uses; two junctions naming the same group on
the same cell **share** the underlying vertex set.

**Status:** SHIPPED. `compute_path_dp_grouped` and
`compute_grid_dp_grouped` consume `List[CellRowSpec]` /
`List[List[CellGridSpec]]` and correctly handle shared anchors via
the `keep_shared` mode in `precompute_M_table`. Tests cover synthetic
disjoint and shared K_3 / K_4 grids, including 3×3 K_4 with shared
interior cell (`test_grid_grouped_3x3_K4_shared_interior` —
T(1,1) = 54,567,559,495,680).

**The change in detail:**

1. When `spec.has_shared_horizontal` for a cell, the path DP's relabel
   map places the cell's "right" anchors at the SAME canonical
   positions as its "left" anchors (no fresh boundary).
2. The state's `state_right` after that cell's convolution is
   identically `state_left` — no new positions accumulated.
3. The next junction's convolution runs at the same shared-boundary
   positions as the previous junction (one boundary, two junctions).
4. The vertex-sum divisor still uses `c_J` per junction; the only
   change is that the boundary positions are reused.

`compute_grid_dp_grouped` extends the same logic to vertical junctions
across rows: when row `r` cells have `has_shared_vertical`, the row's
"up" and "down" positions coincide and are kept across the row's
convolution (`keep_shared=True` in step 2b) so the next vertical
junction can reuse them.

Generalizes cleanly beyond D-Wave Cm₃ — any cell-decomposable graph
with reused anchor groups, including Pm₃ junctions and random graphs
with sparse cell sharing, can use the same path.

**Known scaling wall (Cm₃ and beyond):** the path DP through a row of
N K_{4,4} cells maintains state over up to 8 boundary positions
(2 anchor groups × 4 vertices). `Bell(8) = 4140` partitions per
junction step, and `M_precompute` iterates `Bell(8)²` partition pairs
per cell-step (~17M operations). Without multi-cell orbit
compression (cell template aut acting on per-cell anchor blocks
within the multi-cell state), Cm₃ row composition is hours of
wall-clock per row. The cycle DP at engine step 7.7 has per-cell
orbit compression but doesn't lift to multi-cell boundaries.
Productionizing Cm₃ requires either:

- Multi-cell orbit canonicalization (likely structural rather than
  via brute-force aut enumeration),
- A different decomposition (Hamiltonian-path with closing-edge
  identifications, treating the 3×3 grid as a 9-cell path with
  4 closing chords), or
- A hybrid where each path-DP step caches contracted-cell polynomials
  in the engine's lookup table for amortization across rows.

## Phase B Round 6: Streaming junction enumeration (2026-05-12)

A new parameter `enumerate_junction_internally: bool = False` on
[`precompute_M_table`](../roots/cell_quotient_helpers.py) — together with
a previously under-exercised `out_cell_anchor_groups` argument — unlocks
the 2D K_{4,4} grid composition for Cm₂ and gives the first known
production-quality win against the engine's `kmatching_formula` baseline
(36 s vs 55 s, 1.5×).

### The two ingredients

1. **`enumerate_junction_internally=True`** — when set together with
   `junction_cell_anchor_groups` and single-rep `junction_orbit_partitions`,
   `precompute_M_table` calls `_expand_per_cell_orbit_members` *internally*
   on one junction orbit at a time. The caller hands in compressed
   orbits (one rep each); the function iterates enumerated members in
   the inner loop and zeros out the `n_junc` scalar shortcut (which is
   only sound in trivial-orbit cases — see correctness note in the
   `precompute_M_table` docstring). This avoids materializing Bell(W)
   junction partitions in a single dict, which previously OOM'd Cm₃ at
   ~4.6 GB during external enumeration.

2. **`out_cell_anchor_groups = next_row_scg`** in the 2a (vertical
   junction) step. The M_4 vertical junction has per-edge cell groups
   `[[prev_down[i], next_up[i]] for i]`. State's per-cell S_4 on
   prev_down lifts through this edge structure to a corresponding S_4
   on next_up. By compressing the *output* of the 2a convolution
   with the next row's per-cell anchor groups, the orbit-shortcut on
   state side is preserved through to the out canonical key — so the
   2a output stays compressed (109 per-cell orbits on Cm₂) instead of
   expanding to all 1028 partitions on next_up.

The downstream 2b row-row composition then operates state-compressed
× junction-compressed-then-streamed, shrinking the `M_r` table from
112 052 entries (v4 external enumeration) to 11 881 entries (9.4×
smaller).

### Cm₂ timing comparison

| Approach | T(Cm₂) cold time | Notes |
|---|---|---|
| Engine `kmatching_formula` (step 7.5) | ~55 s | Current production baseline. |
| Engine `cell_quotient_cycle_dp` (step 7.7) | ~50 s | Cycle-DP path for the 4-cycle of K_{4,4} cells. |
| v4 external-enumeration grid DP (research) | ~4 min | First successful 2D K_{4,4} grid; OOM-bound on Cm₃. |
| **v5 streamed grid DP (research)** | **~36 s** | Compressed state + compressed-streamed junction + compressed out. |

The reference recipe is
[`tutte/research/scripts/cm2_via_v5_streamed.py`](../research/scripts/cm2_via_v5_streamed.py).

**Engine integration (shipped 2026-05-12)**: a new dispatch step
**7.45** fires before §7.5 formula short-circuit. It calls
`compute_cell_quotient_grid_dp_streamed(graph, table)` (in
[`tutte/roots/__init__.py`](../roots/__init__.py)) which performs
detection + invokes
`compute_grid_dp_streamed_kab` in
[`tutte/roots/cell_quotient_grid.py`](../roots/cell_quotient_grid.py).
Preconditions checked: 2D grid topology, K_{a,b}-style bipartite
cells, M_k vertical and horizontal junctions, **no shared-anchor
cells** (rejects Cm₃'s interior cells; falls through to other
hierarchical paths). Confirmed on Cm₂: engine dispatches via the new
step at ~36 s vs the previous `kmatching_formula` route at ~55 s.
The existing `compute_grid_dp_grouped` (uncompressed) is untouched
and remains the reference for the disjoint-anchor synthetic K_n grid
test corpus.

### Cm₃ status (Round 6 baseline)

The v5 recipe runs on Cm₃ row-by-row up to the 2b step but walls at
6 608 state orbits × Bell(12) ≈ 4 M junction members ≈ 26 B inner
iterations. An 11-hour CPU run did not produce 2b output. The wall
is genuine — even with C-extension speed (1 µs/iter) it would take
~7 h. A pair-orbit-aware compressed × compressed convolution (via
double-coset enumeration over `G = S_4^N × S_M`) is the documented
path forward; see the
[`project_phase_b_round_6_v5_streamed.md`](../../.claude/projects/-Users-colton-quip-stack-quip-protocol/memory/project_phase_b_round_6_v5_streamed.md)
memory file and task #237.

## Phase B Rounds 7-13: From memory wall to math foundation (2026-05-12)

Rounds 7-11 systematically eliminated the memory and bookkeeping walls
that bit Round 6's Cm₃ attempt; Round 12 pivoted to point-value
modular DP + Lagrange interpolation when polynomial allocation proved
the irreducible cost; Round 13 collapsed the modular M-table to a
single int per orbit triple.

| Round | Change | Cm₂ result | Cm₃ result |
|---|---|---|---|
| 7 | Pair-orbit-aware `precompute_M_table_pair_orbit` via `H`-orbit bucketing on the shared boundary | Matches v5; ~91 s (slower than v5 35 s due to per-pair `|H|` canonicalization) | ~616 B inner ops; still infeasible |
| 8 | C-ext gate relaxed (`enumerate_junction_internally=True` now allowed) + analytical `out_orbit_part` sizing | ~36 s (regression-free) | Bell(12) wall cleared; new wall = M-table marshaling (2.7 GB OOM) |
| 9 | `precompute_M_and_convolve_streaming` chunk wrapper accumulates per-chunk | ~36 s | Memory bounded ~1.5 GB; 2a alone 10+ min; total est. 30-60 min |
| 10 | Raw-dict accumulator (no `TuttePolynomial.__add__` overhead) | ~35 s | Memory ~1.9 GB; per-chunk `_dict_mul` confirmed irreducible cost |
| 11.A | `chunk_size` 200 → 1000 sensitivity test | regression-free | No timing change — confirmed Python framing isn't the wall |
| 11.B | `poly_mul_batched_int64` C ext + `_dict_mul_batched` dispatcher (gated at 100K pairs) | ~36 s | Still walled — C compute is the irreducible cost at Cm₃ scale |
| **12** | **Modular point-value DP**: `TuttePolynomial.evaluate_mod` + 1D/2D Lagrange + CRT combine (`tutte/roots/interpolation.py`); `precompute_M_and_convolve_streaming_mod` validates Cm₂ at 4 (x, y, p) points; oracle recovery exact | ~7-16 s per point | Math foundation ready |
| **13** | **Fast modular M-table**: `precompute_M_table_mod` — int-mod-p accumulation with no polynomial allocation per chunk | ~8-16 s per point (similar; Cm₂ poly M-table was already small + C-ext-fast) | Path DP 65 s (one-time); state ⊗ junc (2a) ~45 min pure Python → ~2-3 hr/point. Bottleneck moved to per-pair structural ops (`delta` / `join_partitions` / `restrict_partition` / `per_cell_canonical_key`) |

### Round 12: precision-safe point-value path

`TuttePolynomial.evaluate_mod(x, y, p)` (Horner-style, integer-only)
combined with bivariate Lagrange in
[`tutte/roots/interpolation.py`](../roots/interpolation.py) and CRT
combine (via existing `_crt_multi`) gives **bit-exact** polynomial
recovery from `(d_x + 1)(d_y + 1)` grid evaluations. Defense in depth
verifies `T(1, 1) = #ST` AND `T(2, 2) = 2^|E|` after recovery — the
Kirchhoff-alone failure mode (Round 11.A discovery) is closed by the
second invariant.

For Cm₃: `d_x ≤ 71`, `d_y ≤ 121` ⇒ 72 × 122 = **8 784 grid points**
× 3-5 50-bit primes for CRT.

Reference scripts:

- [`tutte/research/scripts/cm2_via_modular_interp.py`](../research/scripts/cm2_via_modular_interp.py)
  — sanity check: evaluates the engine's T(Cm₂) on a grid, interpolates,
  CRT-combines, asserts exact polynomial recovery + invariants.
- [`tutte/research/scripts/cm2_via_modular_dp.py`](../research/scripts/cm2_via_modular_dp.py)
  — runs the v5 streamed DP pipeline in modular arithmetic on Cm₂;
  4 test points all match `engine.T(Cm₂).evaluate_mod(...)`.
- [`tutte/research/scripts/cm3_via_modular_dp.py`](../research/scripts/cm3_via_modular_dp.py)
  — generalized N×N modular DP for Cm₃ benchmarking.

### Round 13: int-mod-p M-table built directly

`precompute_M_table_mod` in
[`tutte/roots/cell_quotient_helpers.py`](../roots/cell_quotient_helpers.py)
mirrors `precompute_M_table` exactly (orbits, partitions, anchors,
`enumerate_junction_internally`) but accumulates `int mod p` per
`(O_state, O_junc, O_out)` triple. The
`((x - 1)(y - 1))^d` polynomial dicts become a precomputed list of
single ints `xy_pow_mod[d] = pow((x - 1)(y - 1) % p, d, p)`.
Each inner iteration is one modular mul-add — no poly coefficient
dict to walk.

This removes the **polynomial allocation** wall (each Cm₃ M-entry
would otherwise grow to hundreds of bidegree coefficients) but
exposes the **structural** wall: `delta / join_partitions /
restrict_partition / per_cell_canonical_key` per (state, junc) pair
is now the dominant cost.

### Current Cm₃ wall and forward path

Cm₃ single-point benchmark (pure Python, Round 13):

| Stage | Time | Notes |
|---|---|---|
| Row 0 path DP | 65 s | 6 608 orbits, td = 6 (cached after first call) |
| Row 1, 2 path DP | 0 s | Cache hit |
| Vertical M₄ junction (cols-fold combined) | 0.6 s | 4 096 orbits |
| State ⊗ junction (2a) | ~45 min | **bottleneck** — 6 608 × 4 096 = 27 M pair-iterations |
| Total single point | ~2-3 hr est. (4 composition steps) | Pure Python |

Full Cm₃ polynomial via interpolation: 8 784 × 3 = ~26 K modular DP
runs. At 2-3 hr each that's 6-9 years single-threaded.

| Path | Per-point | 26 K runs |
|---|---|---|
| Pure Python (Round 13) | 2-3 hr | 6-9 yr single-threaded |
| + Round 14 C ext (~10×) | 12-18 min | 8-12 mo single |
| + Round 14 C ext + 8-core mp | 2-3 min | 1-2 mo |

Round 14 (C extension for `precompute_M_table_mod` inner loop) mirrors
the existing `_partition_c.precompute_M_batched_inner_c` polynomial-path
wrapper but accumulates ints. The C function
`batched_inner_iterations_c` already does delta / join / restrict /
canonical key in C — only the Python aggregation step changes.

See memory file
[`project_phase_b_round_13_modular_mtable.md`](../../.claude/projects/-Users-colton-quip-stack-quip-protocol/memory/project_phase_b_round_13_modular_mtable.md)
for the full Round 13 retrospective.

## Files

| File | Purpose |
|---|---|
| [`tutte/roots/cell_quotient_grid.py`](../roots/cell_quotient_grid.py) | `compute_grid_dp_with_layout`, `is_grid_topology`, `_grid_cell_layout` |
| [`tutte/roots/cell_quotient_path.py`](../roots/cell_quotient_path.py) | `compute_path_dp` — path DP consumed per row |
| [`tutte/roots/cell_anchor_adapter.py`](../roots/cell_anchor_adapter.py) | Cell anchor normalization (currently cycle-only; grid extension is the Phase 18.E.3.j work) |
| [`tutte/tests/test_cell_quotient_grid.py`](../tests/test_cell_quotient_grid.py) | 8 regression tests on synthetic K_n grids |

## References

- [6.3 — Rooted Tutte Polynomial: Algebraic Framework](06_3_rooted_tutte_framework.md)
  for the underlying math.
- [6.4 — Cell-Quotient Cycle DP](06_4_cell_quotient_cycle_dp.md) for
  the cycle-topology sibling.
- [`tutte/roots/README.md`](../roots/README.md) for the package
  overview.
