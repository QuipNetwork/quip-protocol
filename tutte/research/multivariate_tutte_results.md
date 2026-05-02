# Multivariate Tutte (Sokal Z) representation — empirical comparison

**Phase 18.E.1.a · April 30, 2026 · NEGATIVE result**

## Question

Does Sokal's multivariate Tutte polynomial Z(G; q, v) — `Σ_{A ⊆ E} q^{k(A)} v^{|A|}` — admit a more compact representation than the standard T(G; x, y) for graphs we care about? If so, it would be a reason to switch internal representation in `tutte/polynomial.py` and pursue Phase 18.E.1.b/c (k-sum closed-form hunt in Z basis, cmtw-DP).

## Method

For each of {K_3, K_4, K_5, K_{2,2}, K_{3,3}, K_{4,4}, C_5, P_4, Petersen}: compute T via `engine.synthesize` and Z via `tutte.multivariate.UniformZ.from_subgraph_sum` (brute force, 2^|E| subsets). Measure (a) monomial count, (b) total integer-coefficient bit-width as storage proxy, (c) maximum |coefficient|, (d) wall-clock for `T·T` vs `Z·Z` self-multiplication.

Pass criterion (decision gate): Z is 5× smaller than T in monomial count on at least 3/9 graphs **OR** Z·Z is 2× faster than T·T on at least 3/9 graphs.

## Result

| Graph | n | e | T terms | Z terms | Z/T | T bits | Z bits | bit-ratio | T max coef | Z max coef |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K_3 | 3 | 3 | 3 | 4 | 1.33 | 3 | 6 | 2.00 | 1 | 3 |
| K_4 | 4 | 6 | 7 | 8 | 1.14 | 13 | 24 | 1.85 | 4 | 16 |
| K_5 | 5 | 10 | 14 | 15 | 1.07 | 46 | 78 | 1.70 | 20 | 222 |
| K_{2,2} | 4 | 4 | 4 | 5 | 1.25 | 4 | 11 | 2.75 | 1 | 6 |
| K_{3,3} | 6 | 9 | 12 | 13 | 1.08 | 37 | 63 | 1.70 | 15 | 117 |
| K_{4,4} | 8 | 16 | 29 | 30 | 1.03 | 192 | 274 | 1.43 | 450 | **9 552** |
| C_5 | 5 | 5 | 5 | 6 | 1.20 | 5 | 16 | 3.20 | 1 | 10 |
| P_4 | 4 | 3 | 1 | 4 | 4.00 | 1 | 6 | 6.00 | 1 | 3 |
| Petersen | 10 | 15 | 26 | 27 | 1.04 | 152 | 222 | 1.46 | 240 | **5 805** |

Decision gate:
- **0 / 9** graphs where Z has fewer terms than T.
- **0 / 9** graphs where Z is 5× smaller than T.
- **1 / 9** graphs where Z·Z is 2× faster than T·T (only K_3, marginally — both <0.02ms).

Z is uniformly **larger** than T in all measured dimensions:
- Monomial count: +1 monomial per graph (consistent excess from `q^|V|` term).
- Coefficient bit-sum: 1.43–6.00× larger.
- Maximum coefficient: dramatically larger for nontrivial graphs (K_{4,4}: T_max = 450 vs Z_max = 9 552 ≈ 21×; Petersen: T_max = 240 vs Z_max = 5 805 ≈ 24×).

Z·Z multiplication is comparable to T·T when both have similar small term counts; no measurable Z advantage emerges with graph size.

## Interpretation

Sokal's identity Z = T · (x−1)^{k(G)} · (y−1)^{|V|} (with q = (x−1)(y−1), v = y−1) makes Z an **expansion** of T by the factor `(x−1)^{k(G)} (y−1)^{|V|}`. For our representative graphs that factor expands the coefficient magnitudes (each `(y-1)^{|V|}` binomial-expands to ≈ 2^{|V|} terms when applied; the resulting Z polynomial concentrates many T-coefficients into single Z-monomials but with much larger absolute coefficients). The net effect is strictly worse for compact storage.

This **mirrors the Phase 18.C.1 Whitney rank-nullity finding** (`tutte/research/data/whitney_vs_tutte_results.md`): both alternative representations expand the (x, y) basis without surfacing structure. The (x, y) Tutte basis is the right basis for compact representation across our graph corpus.

## Decision

**Phase 18.E.1.a closes as NEGATIVE.** Phase 18.E.1.b (closed-form k-sum identity hunt in Z basis) is no longer justified by representation efficiency. The argument for Z basis was "maybe (x, y) hides algebraic structure that per-edge variables surface"; the empirical evidence rules this out for our representative graphs.

Phase 18.E.1.c (cmtw-DP in Z basis) is also closed: the cmtw structural parameter does suggest a useful DP, but expressing it in Z basis offers no representation advantage over T basis. If a cmtw-DP is pursued in the future, it should be in T basis.

## Implications for the algebraic-first principle

The Z negative result joins Whitney as a strong empirical signal that **representation-level wins are unavailable** for the Quip target graphs. Future research must either:

1. **Find closed-form algebraic factors at the COMPOSITION level** (test B in `feedback_algebraic_first_principle.md`): identities that reduce exponential composition to polynomial arithmetic, like the shipped k-matching closed form (Phase 13) and unified formula (Phase 11). The (x, y) basis is the right home for these — and shipped wins live there already.
2. **Find cacheable canonical-key on-ramps** (test A): graph-based DPs that end in a `engine._multigraph_cache` / `RainbowTable` hit on a future call. The shipped cotree_dp + almost_cograph paths are good examples.

What's effectively closed:
- Z basis (Phase 18.E.1).
- Whitney (u, v) basis (Phase 18.C.1).
- Clique-width-k DP for n=72-class graphs (Phase 18.E.3.c).
- Cell-quotient DP scaling to Cm3+ (Phase 18.E.3.l).

What's open:
- Closed-form k-sum / k-vertex-cut identity hunt in **(x, y) basis** with rooted-Tutte boundary state (Phase 18.E.2 revisited from a different angle).
- Generic Aut-orbit cache amortization across runs (Phase 18.B): test-A win, modest but generic.
- Other test-B candidates: SP-reduction extensions, parallel-edge formula generalization, structure-specific shortcuts.

## Files

| Path | Purpose |
|---|---|
| `tutte/multivariate.py` | Production: `UniformZ` + `MultivariateTutte` classes. Validated by `tutte/tests/test_multivariate.py` (10 tests, including Sokal identity verification at 6 (x, y) points per graph). |
| `tutte/tests/test_multivariate.py` | Production regression: Sokal identity numerical verification on K_3 / K_4 / K_{2,2} / C_5 + UniformZ arithmetic + MultivariateTutte → UniformZ specialization. |
| `tutte/research/scripts/multivariate_tutte_experiment.py` | Empirical harness producing this comparison. |
| `tutte/research/data/multivariate_tutte_raw.md` | Raw output dump. |
| `tutte/research/multivariate_tutte_results.md` | This document. |
