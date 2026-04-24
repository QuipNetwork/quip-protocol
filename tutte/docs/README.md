# Tutte Synthesis Engine — Documentation

Detailed documentation for each technique used by the tutte synthesis engine to compute Tutte polynomials.

> **Pipeline note (April 2026)**: the matroid-theoretic Theorem 6 / Theorem 10 paths (Bonin & de Mier 2004) have been **retired** in favor of a chord-rule-based approach implemented in `tutte/graphs/k_sum.py`. The chord rule is mathematically simpler, computationally competitive (often faster), and uses only the standard deletion-contraction identity — no flat lattices, no Möbius function, no inclusion-exclusion bookkeeping. See [08_2_chord_rule_formalization.md](08_2_chord_rule_formalization.md) for the empirical validation, theoretical justification, and mathematical writeup.

## Motivation

| #   | Document                                                        | Description                                                     |
| --- | --------------------------------------------------------------- | --------------------------------------------------------------- |
| 0   | [Tutte Polynomials as a Difficulty Mechanism](00_motivation.md) | Why Tutte polynomials are used in Quip Protocol's proof of work |

## Technique Index

| #   | Technique                                                      | When Used                                                                                            | Complexity                                             |
| --- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 1   | [Family Recognition](01_family_recognition.md)                 | Trees, cycles, wheels, fans, ladders, prisms, books, gears, Möbius ladders, grids, etc.              | **O(n + m)** — fastest path, runs before canonical-key |
| 2   | [Rainbow Table Lookup](02_rainbow_table_lookup.md)             | Canonical-key match against pre-computed polynomials                                                 | O(n² × d) — dominated by canonical key computation     |
| 3   | [Base Cases](03_base_cases.md)                                 | Empty graph or single edge                                                                           | O(1)                                                   |
| 4   | [Disconnected Factorization](04_disconnected_factorization.md) | Graph has multiple connected components                                                              | O(n + m) + recursive synthesis per component           |
| 5   | [Cut Vertex Factorization](05_cut_vertex_factorization.md)     | Graph has an articulation point                                                                      | O(n + m) + recursive synthesis per block               |
| 6   | [Treewidth DP](06_treewidth_dp.md)                             | Graphs ≥ 10 edges with treewidth ≤ 11 (catches most ≤ ~50-edge graphs and the D-Wave cases that fit) | O(2^tw × n) C-extension                                |
| 6.5 | [Cotree DP](06_1_cotree_dp.md)                                 | Cographs (P*4-free) where treewidth_dp can't fit (K_12+, large K*{a,b}, threshold graphs)            | **exp(O(n^{2/3}))** — subexponential                   |
| 7   | [k-Sum Decomposition (chord rule)](07_k_sum_decomposition.md)  | Graphs with k-vertex separators (k=2..7) — treewidth_dp didn't apply or didn't run                   | **O(C(k,2)) full syntheses**                           |
| 8   | [Hierarchical Tiling (chord rule)](08_hierarchical_tiling.md)  | Graphs ≥ 20 edges with repeating cell structure — fallback when treewidth_dp doesn't fit             | **O(chord_count) full syntheses**                      |
| 9   | [Creation-Expansion-Join (CEJ)](09_creation_expansion_join.md) | Final fallback — spanning tree + chord addition                                                      | O(chords × synthesis_cost)                             |
| 10  | [Rooted Tutte Path DP](10_rooted_tutte_path_dp.md)             | (Research) Multi-cell path topologies via boundary-partition convolution — extends to D-Wave Pm3+    | O(n × Bell(W)² × poly²)                                |

## Pipeline Overview

```mermaid
flowchart TD
    A[Input Graph] --> Z{1. Family recognition?\nO(n+m) fast path}
    Z -- hit --> R[Return polynomial]
    Z -- miss --> B{2. Rainbow table?}
    B -- hit --> R
    B -- miss --> C{3. Base case?}
    C -- yes --> R
    C -- no --> D{4. Disconnected?}
    D -- yes --> E[Factor into components, recurse]
    E --> R
    D -- no --> F{5. Cut vertex?}
    F -- yes --> G[Split at cut vertex, recurse]
    G --> R
    F -- no --> H{6. Treewidth DP\n≥ 10 edges, tw ≤ 11?}
    H -- yes --> H1[C-extension treewidth DP]
    H1 --> R
    H -- no --> HC{6.5 Cotree DP\nP_4-free cograph?}
    HC -- yes --> HC1[compute_tutte_cotree_dp:\nsubexponential exp(O(n^{2/3}))]
    HC1 --> R
    HC -- no --> I{7. k-Sum decomposition\n≥ 6 edges + vertex separator?}
    I -- yes --> I1[clique_chord_k_sum:\niterative chord rule]
    I1 --> R
    I -- no --> J{8. Hierarchical tiling\n≥ 20 edges + cell decomposition?}
    J -- yes --> J1[boundary_quotient_tutte:\nboundary quotient + chord recursion]
    J1 --> R
    J -- no --> K[9. CEJ: spanning tree + chord addition]
    K --> R
```

> **Why family recognition runs first**: it costs O(n + m) and skips the expensive canonical-key computation (O(n² × d)) needed for rainbow-table lookup. For known parametric families — trees, cycles, wheels, fans, ladders, prisms, books, gears, Möbius ladders, grids — the polynomial is computed directly from a closed-form formula or constant-coefficient recurrence in `n`. When recognition fails, we fall through to the canonical-key + lookup path.

> **Why treewidth_dp runs first**: it's a cffi-accelerated C extension and wins outright when the graph fits (treewidth ≤ 11). The chord-rule paths take over for graphs whose treewidth exceeds the cap — this is exactly the regime needed for larger D-Wave topologies (Cm₃+, Pm₃+, Z(2,t)+). The chord_rule is _competitive_ with treewidth_dp on graphs both can handle (Cm2 chord_rule is ~2× faster; Petersen ~168× faster; Z(1,2) tied; Pm2 chord_rule slower due to 95+ chords) — but the simplest robust ordering is treewidth_dp first, chord_rule for the unreachable rest.

## Hierarchical Tiling — How It Works

For a graph with k disjoint cells `C_1, ..., C_k` (each isomorphic to a known minor) connected by some inter-cell edges:

**boundary quotient** (boundary-quotient formula, when no chords in the inter-cell graph):

```
T(target) = [∏_i T(C_i)] · T(B) / [∏_i T(B_i)]
```

where `B` is the induced subgraph on all boundary nodes (cell vertices touched by inter-cell edges) plus all inter-cell edges plus intra-cell edges among boundary nodes; `B_i` is each cell's boundary-induced subgraph.

**chord recursion** (iterative chord rule, peels off cycle-creating inter-cell edges):

```
T(target) = T(target − all chords) + Σ_i T((target − chord_1 − ... − chord_{i-1}) / chord_i)
```

Each contraction leaf is synthesized as a multigraph (parallel edges and loops are preserved).

**Cost**: 1 + (chord count) full syntheses. Linear in the cyclic complexity of the inter-cell graph.

**Implementation**: `tutte/graphs/k_sum.py:boundary_quotient_tutte`.

## k-Sum Decomposition — How It Works

For a graph with a `k`-vertex separator `S` (i.e., removing `S` disconnects the graph) and the K_k clique edges among `S` either present or deleted:

```
PC = target + missing K_k clique edges       (parallel connection)
T(target) = T(PC) − Σ_i T((PC − e_1 − ... − e_{i-1}) / e_i)
```

This is just the iterative chord rule applied to the K_k clique edges. **Cost**: 1 + C(k, 2) full syntheses. For k=2: 2; for k=3: 4; for k=4: 7.

**Implementation**: `tutte/graphs/k_sum.py:clique_chord_k_sum`.

## Treewidth DP Fallback

Used when neither hierarchical tiling nor k-sum applies. Builds a tree decomposition of width `≤ max_width` (default 11) and runs a bag-by-bag dynamic program in C (via cffi).

A historical bug fixed in April 2026: the int64 (a, b)-basis DP was used for graphs up to 76 edges, but coefficients can reach `2^E / sqrt(E)`, overflowing `2^63` for E ≥ 63. Symptom: `T(1,1)` aligned mod `2^64` so Kirchhoff verification missed the corruption, but other coefficients were off by multiples of `2^64`. Fix: lowered the int128-DP threshold to `> 62` so all 63–120 edge graphs use the `__int128` path.

## Supporting Theory

| #   | Document                                                     | Topic                                                                                                                                                                                  |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10  | [Chord-Rule Formalization](08_2_chord_rule_formalization.md) | Mathematical formalization (boundary quotient + chord recursion) generalizing Bonin-de Mier Theorem 6 to incomplete-graph boundaries; empirical validation; replacement for Theorem 10 |

## Engine Variants

- **SynthesisEngine** — Techniques 1–9 (primary engine, chord-rule based)
- **HybridSynthesisEngine** — Algebraic decomposition + tiling for polynomial-level shortcuts
- **AlgebraicSynthesisEngine** — Pure polynomial-level GCD decomposition (used by the visualizer)

## Benchmarks

The benchmark suite (`tutte/benchmarks/benchmark.py`) measures wall-clock synthesis time across three engines, starting from an empty rainbow table. After each successful synthesis, the computed polynomial is added to the engine's rainbow table, so subsequent graphs may use it as a tile or minor. Graphs are processed in ascending order of edge count.

### Engines Compared

| Engine                               | Description                                   | Default Timeout |
| ------------------------------------ | --------------------------------------------- | --------------- |
| **CEJ** (`SynthesisEngine`)          | Techniques 1–9 with growing rainbow table     | 60s             |
| **Hybrid** (`HybridSynthesisEngine`) | Algebraic + tiling with growing rainbow table | 60s             |
| **NetworkX** (`nx.tutte_polynomial`) | Reference deletion-contraction (no table)     | 30s             |

### Graph Set

Built from three sources, deduplicated by canonical key, sorted by edge count:

| Source            | Graphs   | Description                                                                   |
| ----------------- | -------- | ----------------------------------------------------------------------------- |
| Named graphs      | 13       | K₃–K₇, C₅/C₁₀/C₁₅, W₅/W₇, Petersen, Grid 3×3/4×4                              |
| Graph atlas       | ~1000    | All connected graphs from `nx.graph_atlas(1..1252)`                           |
| D-Wave topologies | up to 49 | Chimera Cm₁–Cm₁₆, Pegasus Pm₁–Pm₁₆, Zephyr Z(1,1) (requires `dwave-networkx`) |

### Verification

Every result is checked against Kirchhoff's matrix-tree theorem: `T(1,1)` must equal the number of spanning trees. When both a synthesis engine and NetworkX produce a polynomial for the same graph, they are cross-validated for exact equality.

### Usage

```bash
python -m tutte.benchmarks.benchmark                          # default
python -m tutte.benchmarks.benchmark --timeout 300            # extend timeout
python -m tutte.benchmarks.benchmark --compare a.json b.json  # diff two runs
```
