# Phase 18.E.1.a / 18.E.5.a — Literature catalog (April 24, 2026)

Search-driven catalog of candidate algorithms surfaced during the
plan's "open to other ideas from search" phase. Focus: techniques
that could replace or accelerate the rooted-Tutte path-DP for
cell-decomposable graphs without baking in D-Wave-specific structure.

## Foundational papers

### Sokal (2005) — The multivariate Tutte polynomial (alias Potts model)

- **arXiv:** [math/0503607](https://arxiv.org/abs/math/0503607)
- **Survey paper** presented at the 2005 British Combinatorial Conference.
- Defines `Z(G; q, v) = Σ_{A ⊆ E} q^{k(A)} ∏_{e ∈ A} v_e` — one
  variable per edge, plus a global `q` parameter.
- Recovers ordinary Tutte: `T(G; x, y) = (x-1)^{-r(E)} (y-1)^{-|V|+1}
  Z(G; (x-1)(y-1), y-1)` (uniform `v_e = y-1`).
- **Series–parallel reductions** (the key algebraic tools):
  - **Parallel composition** (two parallel edges with weights w_1, w_2):
    `w* = (1 + w_1)(1 + w_2) - 1`.
  - **Series composition** (length-2 path, weights w_1, w_2):
    `w* = w_1 · w_2 / (q + w_1 + w_2)`.
  - These reductions are EXACT and reduce the graph by one edge per step.
  - Applicable to any series-parallel sub-structure within a larger graph.
- **Why this matters for us**: chord-rule contraction leaves often
  contain large series-parallel sub-components (per Phase 7's memory:
  Z(1,2) inter-cell components are SP). Sokal's reductions could
  collapse these in linear time before the DP gets to them.

### Giménez, Hliněný & Noy (2006) — Computing Tutte on Bounded Clique-Width

- **Paper:** [SIAM J. Discrete Math 20:4 932–946](https://web.mat.upc.edu/marc.noy/uploads/2013/05/Tutte-Clique-width.pdf)
- **Algorithm:** subexponential `exp(O(n^{1-ε}))` for clique-width-k graphs.
- **Status:** **already implemented** in the `tutte-2-cotree-dp` branch as
  cograph-only DP (clique-width-2 special case). Integrated into engine
  at step 10.
- **Open extension:** the GHN framework generalizes to any bounded clique-
  width — not just cographs. Random graphs with cell-like structure may
  have clique-width ≤ 4 even when their treewidth is high. Implementing
  the general clique-width-k DP would extend the engine's reach.

## Recent (2024-2025) developments

### Algorithm for Potts on SP-graphs (Yardim & Türker 2025)

- **arXiv:** [2507.22579](https://arxiv.org/abs/2507.22579) — July 2025.
- **Result:** **linear-time exact** computation of the Potts partition
  function on series-parallel graphs with arbitrary edge weights.
- Builds directly on Sokal's series+parallel reduction identities;
  recursively decomposes the SP-graph into a single edge of equivalent
  weight.
- **Why this matters for us**: this is a concrete, modern, linear-time
  algorithm that directly handles a subset of our chord-rule contraction
  leaves. We have an existing SP-recognition module in `tutte/graphs/
  series_parallel.py` (per Phase 7 memory: `is_series_parallel()` is
  already O(n + m)). **Pairing SP-recognition with this multivariate
  reduction would replace expensive treewidth_dp calls on SP leaves
  with linear-time reductions.**

### Cograph-Modular-Treewidth (Blažej, Jana, Ramanujan — IPEC 2025)

- **Paper:** [IPEC 2025 LIPIcs.IPEC.2025.18](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.IPEC.2025.18)
- **Concept:** a new structural graph parameter `C-modular-treewidth`
  where `C` is a fixed graph class (e.g., cographs, edgeless, etc.).
  Lies strictly between treewidth and clique-width.
- **Construction**: take the modular decomposition of G; restrict each
  module to be a graph from C; define cmtw as the treewidth of the
  decomposition tree, weighted by module structure.
- **Why this matters for us**: D-Wave Cm/Pm/Z graphs have low
  cograph-modular-treewidth because the cells (K_{4,4}, etc.) are
  themselves dense modules with simple structure (cographs!). Many
  random sparse structured graphs may also have low cmtw without being
  D-Wave-shaped. **This may be the right parameterization for the
  generic cell-decomposable family.**
- **Status quo on Tutte**: the IPEC 2025 paper does NOT extend to
  Tutte polynomial directly (it focuses on Graph Isomorphism, Chromatic
  Number, Hamiltonian Cycle). But the structural parameter alone
  unlocks GHN-style algorithms for any problem expressible in MSOL
  (Monadic Second-Order Logic) — and Tutte polynomial IS MSOL-
  expressible (per Courcelle 1990).

### Patel-Regts (2017) — Deterministic poly-time APPROXIMATION

- **arXiv:** [1607.01167](https://arxiv.org/abs/1607.01167)
- **Result:** deterministic polynomial-time `(1 ± ε)` approximation
  algorithms for partition functions including Tutte / Potts.
- **Status for us:** **OUT OF SCOPE** per user (April 2026):
  > "we need the whole polynomial"
- The approximation algorithms compute single-point evaluations, not
  the full polynomial. Documented for completeness; not pursued.

## Synthesis: actionable next steps

In rough order of expected payoff vs implementation cost:

### A — SP-leaf shortcut via Sokal series+parallel reductions (HIGH value, LOW cost)

**Hypothesis**: many chord-rule contraction leaves are series-parallel
or near-SP. Sokal's series+parallel reductions in the multivariate
representation collapse them in linear time. If we can detect SP
leaves cheaply (we already have `is_series_parallel()` in
`tutte/graphs/series_parallel.py`), we route them to the SP-Potts
algorithm instead of treewidth_dp.

**Concrete steps**:
1. Implement multivariate-Z arithmetic on multigraphs with edge weights
   (new module `tutte/graphs/multivariate_potts.py`).
2. Implement Sokal's series+parallel reductions.
3. Implement the conversion `Z → T` at the boundary (substitute `v_e = y-1`,
   normalize by `q^{...}`).
4. Wire into `_synthesize_multigraph` after SP-recognition: if SP, route
   to multivariate-Z + reductions; else fall through to existing path.
5. Benchmark on chord-rule leaves of Cm2, Z(1,2), Pm2 to measure win.

**Estimated wall-clock savings**: per Phase 7 memory, Z(1,2) inter-cell
components are SP and individual component synthesis was 0.2s. If 30%
of chord-rule leaves are SP (estimate), savings on Cm3/Pm3 attempts
could be 20-40% from this single optimization.

### B — Empirical multivariate-Z representation experiment (MEDIUM value, LOW cost)

Per Phase 18.C.1's negative result on Whitney rep, we should NOT assume
representation changes win. But the Phase 18.C.1 hypothesis only tested
the (x,y) ↔ (u,v) basis change; the **multivariate** Z representation
is structurally different (per-edge variables) and warrants its own
empirical test.

**Concrete script**: `tutte/scripts/multivariate_tutte_size.py`. Build
Z(G; q, v_e) for our representative corpus; report storage and
multiplication cost. If Z is structurally smaller for cell-symmetric
graphs (likely YES because edges in same orbit can share variables),
this opens a new optimization axis.

### C — Cograph-modular-treewidth structural analysis (MEDIUM value, MEDIUM cost)

For the corpus of D-Wave + random structured graphs, compute their
cograph-modular-treewidth (or estimate it). If cmtw ≪ tw on these graphs,
implementing GHN-style cmtw-DP could unlock graphs that are currently
infeasible via treewidth_dp.

**Concrete script**: `tutte/scripts/cograph_modular_tw_analysis.py` —
compute modular decomposition, identify cograph-shaped modules, estimate
cmtw. Compare against treewidth.

### D — General clique-width-k DP (HIGH value, HIGH cost)

Generalize the existing `tutte-2-cotree-dp` (cograph-only, clique-width-2)
to clique-width-k for k = 3, 4. This is a substantial implementation
(GHN's general algorithm has more state machinery than the cograph
special case) but would handle a much broader graph class than our
current pipeline.

**Estimated effort**: ~2-3 weeks, building on the existing cotree_dp
module structure.

### E — MSOL-based Tutte computation on cmtw-bounded graphs (RESEARCH, OPEN)

Combine cograph-modular-treewidth with Courcelle's MSOL theorem to get
a generic algorithm for Tutte polynomial on cmtw-bounded graphs. This
is theoretically clean but practically may not be faster than direct
clique-width-k DP. Worth exploring if cmtw turns out to be the right
parameter for our corpus.

## Recommended priority

1. **(A) SP-leaf shortcut** — highest leverage; small implementation;
   directly addresses the chord-rule per-leaf bottleneck.
2. **(C) Cograph-modular-treewidth analysis** — informs whether to invest
   in (D) general clique-width DP.
3. **(B) Multivariate-Z empirical test** — quick experiment to validate
   or rule out a representation refactor.
4. **(D) Clique-width-k DP** — major investment, only if (C) shows it'd
   pay off.
5. **(E) MSOL-based** — academic; defer.

## Sources

- [The multivariate Tutte polynomial (Sokal 2005)](https://arxiv.org/abs/math/0503607)
- [Algorithm for computing the partition function of the Potts model for SP-graphs (2025)](https://arxiv.org/abs/2507.22579)
- [Bridging Treewidth and Clique-Width via Cograph-Modular-Treewidth (IPEC 2025)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.IPEC.2025.18)
- [Computing Tutte on Bounded Clique-Width (Giménez-Hliněný-Noy 2006)](https://web.mat.upc.edu/marc.noy/uploads/2013/05/Tutte-Clique-width.pdf)
- [Deterministic Polynomial-Time Approximation Algorithms (Patel-Regts 2017)](https://arxiv.org/abs/1607.01167)
