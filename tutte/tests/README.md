# tutte/tests

Parametrized test suite for the Tutte synthesis library. Correctness
is validated via Kirchhoff's matrix-tree theorem, cross-validation
against `nx.tutte_polynomial`, and direct fixtures for each DP path.

## Running

```bash
# Full suite (excludes slow tests by default)
python -m pytest tutte/tests/ -v -m "not slow"

# Include graph atlas exhaustive tests
python -m pytest tutte/tests/ -v

# Update rainbow table with newly computed polynomials
python -m pytest tutte/tests/ -v --update-rainbow-table

# Collect benchmark timings into data/benchmark_results.json
python -m pytest tutte/tests/ -v --benchmark
```

Fixtures and CLI options live in [`conftest.py`](conftest.py).

## Files

| File                                       | Coverage                                                                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `test_tutte.py`                            | Core engine: Kirchhoff verification, NetworkX cross-check, k-sum / hierarchical paths, end-to-end on named graphs.       |
| `test_lattice_graphs.py`                   | Exhaustive graph-atlas coverage for connected graphs ≤ 7 vertices, plus transfer-matrix integration on lattice strips.   |
| `test_cotree_dp.py`                        | Cograph cotree DP and almost-cograph DP correctness/coverage.                                                            |
| `test_family_recognition.py`               | Closed-form family formulas (trees, cycles, wheels, ladders, prisms, …).                                                 |
| `test_family_verifier_fuzz.py`             | Random graphs to stress-test the family verifier preconditions.                                                          |
| `test_benchmark_family_recognition.py`     | Family-recognition timing regression.                                                                                    |
| `test_treewidth.py`                        | Treewidth DP (Python and C extension) on small structured inputs.                                                       |
| `test_partition_c.py`                      | cffi extensions for partition / M-table inner loops.                                                                     |
| `test_roots.py`                            | Cell-quotient cycle / grid / tree / interleaved DPs against engine + Kirchhoff oracle.                                  |
| `test_cell_quotient_bipartite_junction.py` | Bipartite-junction DP, including the per-component decomposition.                                                       |
| `test_rooted_cache.py`                     | Persistent `T_rooted` lookup table load/save and labelling-aware cache hits.                                            |
| `test_chain_recurrence.py`                 | Linear chain + cycle recurrences (Noy–Ribò re-derivation) for cell-decomposable families.                                |
| `test_modular.py`                          | Modular-DP vs engine bit-for-bit cross-validation across small graphs.                                                  |
| `test_signed_elim.py`                      | Signed-graph elimination DP (`signed_elim_dp.py`).                                                                      |
| `test_signed_quotient.py`                  | Signed/twisted Tutte on σ-quotient graphs.                                                                              |
| `test_sigma.py`                            | σ-equivariant unsigned Tutte DP on 2-fold covers.                                                                       |
| `test_sparse_interp.py`                    | Sparse Lagrange interpolation helpers used by modular point-value paths.                                                |
| `test_zephyr_engine.py`                    | End-to-end engine on D-Wave Zephyr targets (Z(1, 1), Z(1, 2)).                                                          |

## Markers

| Marker | Description                                                                          |
| ------ | ------------------------------------------------------------------------------------ |
| `slow` | Graph-atlas exhaustive tests and large D-Wave runs (deselect with `-m "not slow"`).  |
| `perf` | Performance regression tests (timing bounds).                                        |

## What "verified" means

Most synthesis tests check two things:

1. **Kirchhoff identity** — `T(1, 1)` equals the Laplacian determinant
   (matrix-tree theorem). See
   `tutte/validation.py::count_spanning_trees_kirchhoff`.
2. **Cross-validation** — when the graph is small enough,
   `nx.tutte_polynomial()` is used as the oracle for bit-for-bit
   polynomial equality.

The modular tests additionally check `evaluate_mod` against
`evaluate(...) mod p` for several primes and check the polynomial
recovered via Lagrange + CRT against the engine's direct synthesis.

## Related docs

- [`tutte/synthesis/README.md`](../synthesis/README.md) — engines
  exercised by `test_tutte.py` and `test_roots.py`
- [`tutte/docs/README.md`](../docs/README.md) — per-technique deep
  dives for the algorithms under test
