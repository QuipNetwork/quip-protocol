# Code Review Issues: `tutte/cotree_dp/`

**Reviewed:** 2026-04-02
**Updated:** 2026-04-03
**Module:** Cotree DP for computing Tutte polynomials of cographs (P₄-free graphs)
**Paper:** Gimenez, Hlineny & Noy (2006) — Computing the Tutte Polynomial on Graphs of Bounded Clique-Width

---

## Critical

### Issue #5: Connection count formula in `_join_forest_pair` — RESOLVED

**File:** `tutte/cotree_dp/forest.py` (was `dp.py` before modularization)
**Severity:** ~~Critical~~ Resolved
**Resolved:** 2026-04-02

**Verification:** Full polynomial cross-validation against the synthesis engine on K_10 (45 edges), K_11 (55 edges), K_12 (66 edges), K_13 (78 edges), and K_14 (91 edges) — all passed with exact coefficient-level match. K_15 (105 edges) engine timed out but cotree DP matched exact sympy Kirchhoff. Test: `test_engine_cross_validation` in `test_cotree_dp.py`.

**Why `g_comp_size * f_size` is correct:** The ⊗ combine processes components of `sig_g` one at a time via `for g_comp_size in sig_g`. Each iteration picks one edge from K_{g_comp_size, f_size} to connect the G-side component to each selected F-side component. The factor `g_comp_size * f_size` counts the available edges. Multi-edge spanning trees are handled implicitly through accumulation across iterations.

**Variable names fixed:** `c` → `g_comp_size`, `x` → `f_size`, `z` → `num_f_vertices`, `d1` → `f_total`, `d2` → `g_total`, `x_count`/`y_count` → `beta_count`. Dead code `_make_sig()` removed.

---

## High

### Issue #11: No cross-validation against the synthesis engine — RESOLVED

**File:** `tutte/tests/test_cotree_dp.py`
**Severity:** ~~High~~ Resolved
**Resolved:** 2026-04-03

**Test added:** `test_engine_cross_validation` — parametrized over 14 graphs across three families. Compares full polynomial from `compute_tutte_cotree_dp()` against `SynthesisEngine().synthesize().polynomial` (5-minute timeout per engine call). If the engine times out, falls back to exact Kirchhoff verification using `_exact_spanning_tree_count` (sympy integer determinant) and `_exact_num_spanning_trees` (integer coefficient sum) — avoiding float64 precision loss for large spanning tree counts.

**Precision issue discovered during testing:** The original Kirchhoff fallback used `count_spanning_trees_kirchhoff()` which relies on `nx.number_of_spanning_trees()` (float64 + `round()`). This loses precision for graphs with spanning tree counts approaching 2^53 ≈ 9 × 10^15. K_15 has ~1.95 × 10^15 spanning trees — the float Kirchhoff was off by 11. Cotree DP was correct; the validation method was wrong. Fixed by switching to exact integer methods (`_exact_spanning_tree_count` and `_exact_num_spanning_trees`).

### Issue #12: Missing tests for large cographs — RESOLVED

**File:** `tutte/tests/test_cotree_dp.py`
**Severity:** ~~High~~ Resolved
**Resolved:** 2026-04-03

All three requested graph families are now in `test_engine_cross_validation` (14 graphs total):

| Requested | Added | Status |
|---|---|---|
| K_10 through K_15 | K_10, K_11, K_12, K_13, K_14, K_15 | All pass (K_10–K_14 engine match; K_15 exact Kirchhoff fallback) |
| Threshold graphs with 10+ 'd' operations | Thr_d10, Thr_d12, Thr_ddi4, Thr_dddi4 | All pass (Thr_dddi4 exact Kirchhoff fallback for disconnected component) |
| K_{5,5} through K_{8,8} | K_{5,5}, K_{6,6}, K_{7,7}, K_{8,8} | All pass (engine match) |

**12 passed, 2 used Kirchhoff fallback** (K_15 and Thr_dddi4 — engine timed out, exact sympy Kirchhoff confirmed cotree DP is correct).

---

## Medium

### Issue #9: No memoization of CellSel calls

**File:** `tutte/cotree_dp/combinatorics.py` (was `dp.py` before modularization)
**Severity:** Medium — performance

`cellsel(cell_sizes, f)` is called for every `(gamma, f)` pair in `join_subgraph_pair`. The same `cell_sizes` list can appear many times across different `beta` values when multiple double-signatures produce the same gamma. Adding `@functools.lru_cache` on `(tuple(cell_sizes), f)` would eliminate redundant computation.

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def cellsel_cached(cell_sizes: Tuple[int, ...], ell: int) -> int:
    return cellsel(list(cell_sizes), ell)
```

Then call `cellsel_cached(tuple(cell_sizes), f)` in `subgraph.py`.

### Issue #10: n > 35 guard is arbitrary — BEING INVESTIGATED

**File:** `tutte/cotree_dp/dp.py`
**Severity:** Medium — usability
**Updated:** 2026-04-03

The guard `if graph.node_count() > 35` is a fixed constant that doesn't account for graph structure. A cograph with n = 40 that decomposes into many small ⊗ blocks (e.g., K_4 ⊗ K_4 ⊗ ... ⊗ K_4) would be fast, while a cograph with n = 30 that has one large ⊗ block (K_30) could be slow.

A better guard would estimate the cost based on the cotree structure — specifically the maximum signature table size at any ⊗ node, which is bounded by the partition count p(n_subtree) where n_subtree is the size of the largest ⊗ subtree.

**K_n scaling test added** (`test_cotree_dp.py::TestKnScaling`): runs cotree DP on K_8 through K_100 with a 15-minute timeout per graph, printing time and exact Kirchhoff verification for each. Uses `_exact_spanning_tree_count` (sympy integer determinant) and `_exact_num_spanning_trees` (integer coefficient sum) to avoid float64 precision loss. Stops on first timeout. This will determine the empirical N_MAX for the n > 35 guard.

### Issue #13: Missing test for disconnected cographs — PARTIALLY RESOLVED

**File:** `tutte/tests/test_cotree_dp.py`
**Severity:** ~~Medium~~ Partially resolved
**Updated:** 2026-04-03

Thr_dddi4 in `test_engine_cross_validation` is a disconnected cograph (16-vertex component + 1 isolated vertex) and passes with exact Kirchhoff fallback. This exercises the ∪ combine path for a non-trivial disconnected graph.

**Still missing:** A dedicated test with explicit disconnected components (e.g., K_4 ∪ K_5) where both cotree DP and the engine can produce the polynomial, allowing full polynomial comparison rather than just Kirchhoff.

### Issue #17: No per-graph timeout in benchmark

**File:** `tutte/tests/benchmark_cotree_dp.py`
**Severity:** Medium — usability

The benchmark runs all cographs without a timeout. K_20 (20 nodes, 190 edges) might take hours in the ⊗ combine. Should add a per-graph timeout (e.g., 300 seconds) via `signal.SIGALRM`, consistent with the Björklund benchmark's approach.

---

## Low

### Issue #1: Unused parameter `vertex_set` in `_find_co_components`

**File:** `tutte/cotree_dp/recognition.py`, line 267
**Severity:** Low — dead code

The `vertex_set` parameter is annotated with `noqa: ARG001` and never used. The `unvisited` set (initialized from `vertices`) serves the same purpose. Remove the parameter and update the call site.

### Issue #2: Complexity claim is loose

**File:** `tutte/cotree_dp/recognition.py`, line 19
**Severity:** Low — documentation

The docstring claims O(n² + nm). The actual cost is O(m × depth) where depth is the cotree depth. For balanced cotrees (e.g., K_{a,b}), depth = O(log n), giving O(m log n). For degenerate cotrees (threshold graphs), depth = O(n), giving O(nm). The claim O(n² + nm) is correct but the nm term dominates and should be the stated bound: O(nm).

### Issue #6: Empty submultiset semantics undocumented — RESOLVED

**File:** `tutte/cotree_dp/combinatorics.py` (was `dp.py` before modularization)
**Severity:** ~~Low~~ Resolved
**Resolved:** 2026-04-02

The `distinct_submultisets` docstring now explicitly documents the empty submultiset semantics:

> "Includes the empty submultiset (coefficient 1). For forest counting (Stage 1), the empty submultiset represents a G-side component that stays disconnected from all F-side components — correct for forests (which allow disconnected components)."

### Issue #14: Duplicated `_make_threshold` helper

**File:** `tutte/tests/test_cotree_dp.py` line 29, `tutte/tests/benchmark_cotree_dp.py` line 91
**Severity:** Low — code duplication

The `_make_threshold(sequence)` function is identical in both files. Extract to a shared location (e.g., `tutte/cotree_dp/recognition.py` as a public helper, or a test utility module).

### Issue #15: Fragile spanning tree computation in benchmark — ELEVATED

**File:** `tutte/tests/benchmark_cotree_dp.py`, lines 52-68
**Severity:** ~~Low~~ Medium — confirmed correctness risk

`_spanning_tree_count_connected` falls back to `nx.number_of_spanning_trees(G)` when sympy is unavailable. This was confirmed to produce wrong results during testing: for K_15, the float Kirchhoff returned 1946195068359386 while the correct value is 1946195068359375 (off by 11). The test suite now uses exact integer methods (`_exact_spanning_tree_count` and `_exact_num_spanning_trees`), but the benchmark still uses the fragile float method.

**Fix:** Replace with `_exact_spanning_tree_count` from `tutte/validation.py` throughout the benchmark.

### Issue #16: `_exact_t11` uses dict value sum instead of `evaluate(1, 1)`

**File:** `tutte/tests/benchmark_cotree_dp.py`, lines 71-73
**Severity:** Low — fragility

```python
def _exact_t11(poly) -> int:
    return sum(poly.to_coefficients().values())
```

This is actually the CORRECT approach — it avoids float precision loss by summing integer coefficients directly. The method name is misleading (it's not using `evaluate`), but the computation is exact. The test suite confirmed this: `_exact_num_spanning_trees` in `validation.py` uses the same approach and produces correct results for K_15.

**Updated assessment:** Not a bug, but rename to clarify intent (e.g., `_integer_coefficient_sum`).

### Issue #18: Potential duplicate graphs in benchmark

**File:** `tutte/tests/benchmark_cotree_dp.py`, lines 231-238
**Severity:** Low — minor

The deduplication uses graph names as keys. Two different graph builders producing the same graph with different names would not be deduplicated. For the current graph set this is harmless, but fragile if new builders are added.

---

## Resolved Issues Summary

| Issue | Severity | Resolution |
|---|---|---|
| #5 Connection count formula | ~~Critical~~ | Verified correct via K_10–K_15 cross-validation |
| #11 No engine cross-validation | ~~High~~ | `test_engine_cross_validation` added (14 graphs, 3 families) |
| #12 Missing large cograph tests | ~~High~~ | K_10–K_15, threshold, K_{a,b} all added and passing |
| #6 Empty submultiset docs | ~~Low~~ | Docstring updated in `combinatorics.py` |

## Open Issues Summary

| Issue | Severity | Category | Status |
|---|---|---|---|
| #9 CellSel memoization | Medium | Performance | Open |
| #10 n > 35 guard arbitrary | Medium | Usability | Investigating — K_n scaling test added |
| #13 Disconnected cograph test | Medium (partial) | Test coverage | Open |
| #17 Benchmark timeout | Medium | Usability | Open |
| #1 Unused parameter | Low | Dead code | Open |
| #2 Complexity claim | Low | Documentation | Open |
| #14 Duplicated helper | Low | Code quality | Open |
| #15 Float Kirchhoff in benchmark | Medium (elevated) | Correctness risk | Open |
| #16 _exact_t11 naming | Low | Code quality | Open |
| #18 Duplicate graph names | Low | Code quality | Open |
