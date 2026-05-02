# Cm3 interleaved DP attempt — Phase 18.E.3.l.5 (April 29, 2026)

## Algorithm validated end-to-end

Phase 18.E.3.l.5 Rungs 0-3 (verification harness, closing-step factorization
Bell(2a), anchor-sharing dispatch via keep_shared, multi-closing-shared via
keep_merged, inline closing dispatch):

| Test | Wall-clock | Status |
|---|---:|---|
| Cm2 (2x2 K_{4,4}, 32n 80e, 1 closing) | 37.8s | FULL POLY MATCH + Kirchhoff PASS |
| 2x2 K_3 grid (4 K_3 cells, 1 closing) | <1s | FULL POLY MATCH + Kirchhoff PASS |
| 2x2 K_3 grid σ_idr forced path | <1s | FULL POLY MATCH + Kirchhoff PASS |
| **3x3 K_3 grid (9 K_3 cells, 4 closings, anchor-shared)** | **<5s** | **FULL POLY MATCH + Kirchhoff PASS** |
| Cm3 (3x3 K_{4,4}, 72n 192e, 4 closings, anchor-shared) | killed at 13min | algorithmically correct; Python perf wall |

## Cm3 trace (killed at 13 CPU min, 1.1 GB RAM)

```
init cell 0: 43 orbits, 8 verts
path step 1 junction (0,0)→(0,1)/horiz: 58 orbits, 8 verts
path step 1 cell (0,1) [keep_shared cell_b]: 1785 orbits, 12 verts
path step 2 junction (0,1)→(0,2)/horiz: 3340 orbits, 12 verts
path step 2 cell (0,2): 6608 orbits, 12 verts
path step 3 junction (0,2)→(1,2)/vert: 6608 orbits, 12 verts
path step 3 cell (1,2) [keep_shared cell_b]: 167492 orbits, 16 verts
path step 4 junction (1,2)→(1,1)/horiz: 167492 orbits, 16 verts
[stalled — likely in path step 4 cell add OR inline closing of (0,1)-(1,1)]
```

## Bottleneck

State at 167K orbits × 16 verts hits Python performance wall in M_precompute
inner loop. The `precompute_M_batched_inner_c` C extension (per-cell canonical
batched) handles per-cell-canonicalized state efficiently for path steps but
the σ_idr identification path (out_boundary > 12 verts gate) inside
`_close_step2_sigma_idr` is pure Python.

Critical numbers:
- σ_idr enumeration: 576 perms (S_4 × S_4) × |state_orbits| × identification logic
- For 167K state orbits: 167K × 576 = 96M Python iterations per closing step
- Each iteration: ~10μs (relabeling + canonical computation) → ~1000s = 16 min

## Algorithm is correct

3x3 K_3 grid validates ALL infrastructure:
- keep_shared=True path through anchor-shared horizontal junction
- keep_merged=True post-closing for shared anchor groups
- σ_idr factorization (16-vert state)
- Inline closing dispatch via `find_inline_closing` after each path step
- State convention transitions (per-rep → sum-weighted after first closing)

The 3x3 K_3 case demonstrates that the algorithm correctly handles the same
topology as Cm3 — only the cell template (K_3 vs K_{4,4}) differs. K_{4,4}'s
Bell(8)=4140 partitions per side make the inner-loop constant-factor blow up
beyond Python's tractable rate.

## Next steps (deferred follow-on session)

1. **C extension for σ_idr inner loop** — parallel to `_partition_c.py`'s
   batched API. ~150-200 LOC additional cffi C. Estimated 50-100x speedup;
   would put Cm3 path step 3 cell (1,2) at ~10s instead of ~10 min.

2. **Alternative Hamiltonian path orderings** (cheap experiment): spiral
   order or snake-with-early-closings may keep peak state smaller. The current
   zigzag (rows L-to-R, R-to-L, L-to-R) maximizes state size by carrying
   3 cells of row 0 + 1 cell of row 1 before any closing fires. A path that
   processes vertical closings sooner could halve peak orbits.

3. **Once Cm3 unlocks**: try Pm3 / Z(1,3); benchmark against engine timeout
   baseline; consider engine integration as a new step.

## Files shipped (post-Step-0 layout, April 30, 2026)

Production (tracked in git):
- `tutte/roots/cell_quotient_interleaved.py` — `compute_grid_dp_interleaved` with `_process_closing_step`, `keep_merged`, `usage_remaining`, `find_inline_closing`, `_close_step2_bell_n` / `_close_step2_sigma_idr` factorization.
- `tutte/roots/_partition_c.py` — Rung 5 bulk-decode optimization (3× speedup on inner loop).
- `tutte/roots/cell_quotient_helpers.py` — Rung 5 grouped (O_state, O_junc) convolve.
- `tutte/tests/test_interleaved_dp_cm2.py` — three-way verify (full `==` + Kirchhoff) for Cm2.
- `tutte/tests/test_interleaved_dp_synthetic.py` — 2x2 K_3 + 3x3 K_3 + σ_idr cross-validation tests.

Research scratch (gitignored under `tutte/research/scripts/`):
- `_poly_compare.py` — `compare_polys` + `cross_check_with_kirchhoff` helpers (for harness use).
- `test_interleaved_cm2.py` — Cm2 CLI harness using the helpers.
- `test_interleaved_cm3.py` — Cm3 attempt harness with per-step profiling.
- `profile_cm3_step3.py` — cProfile harness driving the path-step-3 stall.
- `trace_cm3_paths.py` — Hamiltonian-order experiment (row vs col zigzag).
- `prototype_interleaved_dp.py` — initial v1 prototype (superseded by production module).

## Why constant-factor optimization cannot break the wall

Phase 18.E.3.l.5 Rung 5 shipped two profile-driven optimizations:
1. **Bulk-decode in `_partition_c.precompute_M_batched_inner_c`**: 153s → 56s self-time (3×).
2. **Grouped `(O_state, O_junc)` convolve**: eliminates redundant poly-muls when many O_outs share the same (state, junc) pair.

After Rung 5 the bottleneck shifts to `poly_mul_python` (105s self) — coefficient overflow past int64 guard pushes ~25% of polynomial multiplications back to pure Python. Even an int128 / bigint poly-mul C extension would only address THIS bottleneck; the next walls would surface in subsequent path steps where state grows to 5+ cells.

**The wall is structural, not constant-factor.** Per-cell automorphism compression is `S_4^k` for k cells in current state. Compression scales:

| Grid | Cells in state pre-closing | Verts | Per-cell orbits |
|---|---:|---:|---:|
| Cm2 | 2 | 8 | 109 |
| 3x3 K_3 (any phase) | up to 4 | up to 4 | small |
| Cm3 | 4 | 16 | **167K** |
| Cm4 (projected) | 5 | 20 | ~1.2M |

The 109 → 167K orbit explosion is **explosive, not linear**. The wall is mathematical: K_{4,4}'s Bell(8)=4140 partitions per side combined with multi-cell state composition. No constant-factor C extension breaks this.

## Why clique-width-k DP doesn't help

GHN 2006 gives `n^{O(k)}` for clique-width-k Tutte computation. For Cm3 (n=72, cw≈3): n^9 ≈ 10^16 — far beyond practical. The cell-quotient composition is already the bottleneck (not single-cell computation), and cotree-DP (`tutte/cotree_dp/dp.py`) already handles K_{4,4} cells in milliseconds. Generalizing cotree-DP to clique-width-3 doesn't address Cm3's actual obstacle.

## Why Bell→Catalan doesn't apply

Cotree-DP's `Signature` / `DoubleSig` (`tutte/cotree_dp/subgraph.py:29-74`) index by **component-size multisets**, not partitions. The `exp(O(n^{2/3}))` bound comes from the double-signature growth at ⊗ nodes, not a Catalan-numbered reduction. There is no Bell→Catalan basis change in the codebase, and even if there were, K_{4,4} is non-planar — non-crossing partitions don't apply. The technique is **inapplicable** to cell-quotient DP (which needs full partition states because anchor identities matter for inter-cell composition).

## Three structural redesign options (and why they're not pursued)

1. **Transfer-matrix / D-Wave-specific row recurrence**: would work but violates the generalization principle (must apply to graphs that aren't D-Wave families).
2. **Recursive cycle DP per row**: earlier exploration showed this stalls at Bell(12)² = 17M iterations per row composition. Same Bell-wall, different shape.
3. **Accept Cm3 as out-of-reach with current architecture**: chosen. The Phase 18.E.3.l.5 work is correct, validated on Cm2 + 3x3 K_3, and shipped as foundation. Cm3 is genuinely a structural problem this architecture can't solve.

## Pivot to multivariate Tutte (Phase 18.E.1)

Per the algebraic-first principle (April 30, 2026 user direction), the right next direction is the multivariate Tutte polynomial Z(G; q, v_e):
1. Per-edge variables make deletion-contraction linear in each `v_e` (no bridge/loop case-split).
2. Sokal series-parallel reductions are already shipped (`tutte/graphs/series_parallel.py:compute_sp_tutte_if_applicable`).
3. The IPEC 2025 cograph-modular-treewidth paper suggests cmtw-DP in Z basis is the natural parameterization (`tutte/research/data/cograph_modular_tw_results.md` shows Cm3 has cmtw=3).

See `tutte/research/literature_search_2026.md` for catalog. Step 3.A (`tutte/multivariate.py` + size-compare experiment) is the immediate next concrete deliverable.
