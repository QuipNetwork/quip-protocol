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
