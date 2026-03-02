## `topo_alloc` — Topology Allocator

`topo_alloc` implements a minor-embedding heuristic for mapping an arbitrary problem graph *H* onto a D-Wave quantum annealer hardware topology graph *G*. A graph H is a **minor** of G when each node of H can be mapped to a disjoint, connected **chain** of physical qubits such that every logical edge is covered by at least one coupler. Finding a compact embedding is NP-hard in general; `topo_alloc` uses a practical randomised heuristic.

### Algorithm

The embedder follows Cai, Macready & Roy (2014) ([arXiv:1406.2741](https://arxiv.org/abs/1406.2741)).

**Stage 1 — Greedy initialisation.**
Source nodes are placed one by one. For each node, multi-source Dijkstra expands from the already-placed chains of its neighbours, finding the cheapest free root and a minimal connecting Steiner tree. Edge weights penalise passing through qubits already occupied by another chain. Overlapping chains are permitted at this stage.

**Stage 2 — Overlap-removal refinement.**
For up to `k × |V(H)|` rounds, the node with the most overlap is re-embedded from scratch using a flat-penalty scheme that steers away from occupied qubits. The loop exits early once all chains are disjoint.

**Stage 2b — Longest-chain refinement (optional).**
A further `k × |V(H)|` rounds re-embed the node with the longest chain, accepting only strictly shorter results. This targets `nodes_used` without risking embedding validity.

**Stage 3 — Validation.**
All three minor-embedding conditions are verified. If satisfied the embedding is returned immediately; otherwise the next attempt begins (up to `tries` total).

#### Placement strategies (`EmbedOption`)

| Flag | Strategy | Description |
|---|---|---|
| *(none)* | `random` | Uniform shuffle each attempt |
| `ORDER_BY_DEGREE` | `degree` | Descending source-degree first, ties shuffled randomly |
| `ORDER_BY_DEGREE_ASC` | `degree_asc` | Ascending source-degree first — hub placed last so Dijkstra bridges all already-placed neighbours |
| `ORDER_BY_CENTRALITY` | `centrality` | Descending betweenness centrality first |
| `REFINE_LONGEST_CHAINS` | `longest_chains` | Degree ordering + Stage 2b chain refinement |
| `USE_VERTEX_WEIGHTS` | `vertex_weights` | Cai et al. 2014 vertex-weight Dijkstra scheme |
| `PREFER_ARTICULATION_POINTS` | `art_pts` | Anchor source articulation points on the highest-degree free target node when they have no placed neighbours |

### Module Structure

```
topo_alloc/
├── minor_alloc.py      # Core embedding algorithm (find_embedding, build_model, EmbedOption)
├── topology.py         # Topology dataclasses (Cell, Coupling, Topology) - WORK IN PROGRESS
├── graphviz_render.py  # DOT rendering and chain statistics helpers
├── embed_cli.py        # CLI wrapper around find_embedding
├── demo_embedding.py   # Ising-model embedding demos (K₅, rings, K₃₃, K₄₄)
├── bench_random.py     # Benchmark CLI for random source graphs
└── test_minor_alloc.py # pytest test suite
```

### Running the Demo

Embeds five canonical Ising-model graphs (K₅, frustrated triangle, 8-spin antiferromagnetic ring, K₃₃, K₄₄) into Chimera, Zephyr, and Pegasus topologies and compares random vs degree-ordered placement:

```bash
python -m topo_alloc.demo_embedding
```

This writes `.dot` files into the `topo_alloc/` directory. Render any of them with Graphviz (run from the project root):

```bash
dot -Tsvg topo_alloc/demo_k5_chimera4_degree.dot -o demo_k5_chimera4_degree.svg
```

### Running the Benchmarks

`bench_random.py` generates random source graphs, embeds them into a chosen topology, and reports per-strategy chain-length statistics.

```bash
# ER graphs (n=8, p=0.5) → Chimera(4), all strategies, 30 samples, 10 warmup rounds
python -m topo_alloc.bench_random \
    --graph-model er --nodes 8 --er-p 0.5 \
    --topology chimera --topology-size 4 \
    --samples 30 --strategy all --warmup 10

# Barabási-Albert → Zephyr(3), auto strategy only
python -m topo_alloc.bench_random \
    --graph-model ba --nodes 8 --ba-m 2 \
    --topology zephyr --topology-size 3 \
    --samples 30 --strategy auto --warmup 10

# Trees → Pegasus(4), aggregate statistics only, CSV output
python -m topo_alloc.bench_random \
    --graph-model tree --nodes 10 \
    --topology pegasus --topology-size 4 \
    --samples 30 --strategy all --no-detail --csv results.csv
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--graph-model` | `er` | Source graph family: `er`, `ba`, `tree` |
| `--nodes` / `-n` | `8` | Source graph node count |
| `--topology` | `chimera` | Target topology: `chimera`, `zephyr`, `pegasus` |
| `--topology-size` | `4` | Size parameter (e.g. `4` → Chimera(4), 128 qubits) |
| `--strategy` | `both` | `random`, `degree`, `degree_asc`, `centrality`, `longest_chains`, `vertex_weights`, `art_pts`, `degree_art`, `auto`, `auto_quality`, `auto_speed`, `both`, `all` |
| `--samples` | `20` | Number of random source graphs to generate |
| `--tries` | `30` | Embedding attempts per sample |
| `--warmup` | `3` | Warmup rounds before recording (discards cold-start latency) |
| `--no-detail` | off | Suppress per-sample table, show only aggregates |
| `--csv PATH` | — | Write per-sample results to a CSV file |

### Running the Tests

```bash
pytest topo_alloc/
```

The suite covers trivial inputs, small graphs, impossible embeddings, all ordering strategies, chain refinement, and `build_model` internals. All tests use fixed seeds for full reproducibility.

### Benchmark Findings

30 samples per configuration, `--tries 100 --refinement-constant 10 --warmup 10`. All strategies benchmarked on each topology.

#### ER n=8, p=0.5 → Chimera(4) — 128 qubits (tight, ratio 16)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **28/30** | 2.06 | 5.25 | **691 ms** |
| degree | 18/30 | 1.91 | 4.39 | 2.62 s |
| degree\_asc | 22/30 | 2.02 | 5.32 | 1.38 s |
| centrality | 12/30 | 1.96 | 4.83 | 3.48 s |
| longest\_chains | 18/30 | 1.91 | 4.39 | 5.25 s |
| **vertex\_weights** | **29/30** | 2.08 | 4.90 | 882 ms |
| **auto** | **28/30** | 2.06 | 5.25 | 686 ms |

#### ER n=8, p=0.5 → Zephyr(3) — 336 qubits (spacious, ratio 42)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| random | 30/30 | 1.96 | 4.67 | 15 ms |
| **degree** | **30/30** | **1.71** | **3.37** | **13 ms** |
| degree\_asc | 30/30 | 2.06 | 4.97 | 29 ms |
| centrality | 29/30 | 1.77 | 3.55 | 1.04 s |
| **longest\_chains** | **30/30** | **1.70** | **3.33** | 267 ms |
| vertex\_weights | 30/30 | 1.98 | 4.77 | 217 ms |
| **auto** | **30/30** | 2.06 | 4.97 | 28 ms |

#### ER n=8, p=0.5 → Pegasus(4) — 264 qubits (spacious, ratio 33)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 2.01 | 4.40 | **14 ms** |
| degree | 29/30 | 1.95 | 3.79 | 700 ms |
| degree\_asc | 30/30 | 2.13 | 5.10 | 17 ms |
| centrality | 25/30 | 1.88 | 3.84 | 3.77 s |
| **longest\_chains** | 29/30 | **1.93** | **3.72** | 1.38 s |
| vertex\_weights | 30/30 | 2.21 | 4.53 | 200 ms |
| **auto** | **30/30** | 2.13 | 5.10 | 17 ms |

#### BA n=8, m=2 → Chimera(4) — 128 qubits (tight, ratio 16)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | **1.77** | 4.20 | **47 ms** |
| degree | 21/30 | 2.27 | 6.33 | 1.76 s |
| degree\_asc | 21/30 | 1.65 | 4.48 | 1.73 s |
| centrality | 17/30 | 2.16 | 6.41 | 2.40 s |
| longest\_chains | 21/30 | 2.27 | 6.33 | 3.77 s |
| **vertex\_weights** | **30/30** | **1.77** | **4.17** | 215 ms |
| **auto** | **30/30** | **1.77** | 4.20 | **47 ms** |

#### BA n=8, m=2 → Zephyr(3) — 336 qubits (spacious, ratio 42)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 1.77 | 4.53 | **11 ms** |
| degree | 29/30 | **1.64** | **3.24** | 983 ms |
| **degree\_asc** | **30/30** | 2.08 | 5.90 | 11 ms |
| centrality | 29/30 | 1.67 | 3.17 | 987 ms |
| longest\_chains | 29/30 | 1.63 | 3.21 | 2.09 s |
| vertex\_weights | 30/30 | 1.80 | 4.93 | 140 ms |
| **auto** | **30/30** | 2.08 | 5.90 | **11 ms** |

#### BA n=8, m=2 → Pegasus(4) — 264 qubits (spacious, ratio 33)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 1.80 | 4.47 | **10 ms** |
| degree | 27/30 | 1.83 | 3.52 | 2.07 s |
| **degree\_asc** | **30/30** | 1.98 | 5.10 | 7 ms |
| centrality | 27/30 | 1.84 | 3.44 | 2.03 s |
| longest\_chains | 27/30 | 1.82 | 3.48 | 4.18 s |
| vertex\_weights | 30/30 | 1.87 | 4.90 | 77 ms |
| **auto** | **30/30** | 1.98 | 5.10 | **7 ms** |

#### Tree n=10 → Chimera(4)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| random | 30/30 | 1.51 | 3.70 | 31 ms |
| degree | 30/30 | 1.10 | 1.63 | **15 ms** |
| degree\_asc | 30/30 | 1.51 | 3.93 | 13 ms |
| **centrality** | 29/30 | **1.02** | **1.14** | 109 ms |
| longest\_chains | 30/30 | 1.10 | 1.63 | 64 ms |
| vertex\_weights | 30/30 | 1.44 | 3.23 | 34 ms |
| **auto** | 29/30 | **1.02** | **1.14** | 109 ms |

#### Key observations

**Topology tightness determines the success-rate ranking.**
On tight topologies (Chimera(4), ratio 16), any degree-based ordering hurts success rates because front-loading constrained nodes leaves no slack for later placements. `random` (28/30) and `vertex_weights` (29/30) are the only strategies that stay competitive. On spacious targets (Zephyr, Pegasus, ratio 33–42) all strategies succeed near-perfectly on ER graphs.

**`degree` gives the best chain quality on spacious ER topologies.**
On Zephyr it cuts chain\_avg by ~13% (1.96 → 1.71) vs random at equal speed. `longest_chains` adds a further marginal improvement (1.71 → 1.70) at 20× the cost — worth it only for offline preprocessing.

**`degree` has a hub-graph failure mode on spacious topologies.**
On BA graphs (scale-free, m=2), `degree` places the dominant hub node first with no neighbours yet placed, anchoring it at an arbitrary target node that all 6–7 neighbours must then route back to. This causes 1–3 failures per 30 samples and mean latencies of 1–2 s on Zephyr and Pegasus. The same samples embed in ~10 ms with `random`.

**`degree_asc` is the reliable default on spacious topologies.**
Ascending-degree ordering places low-degree nodes first (cheap, minimal constraints) and the hub last, letting Dijkstra build the hub's chain as a natural bridge through all already-placed neighbours. This eliminates the hub failure mode (30/30 on BA Zephyr and Pegasus vs 27–29/30 for `degree`). The tradeoff: chain quality on ER graphs regresses slightly vs `degree` (chain\_avg 2.06 vs 1.71 on Zephyr) but remains close to `random`.

**`degree_asc` does not help on tight topologies.**
On Chimera (tight, ratio 16), ascending ordering still degrades success rate relative to random (22/30 ER, 21/30 BA vs 28–30/30 for random), for the same reason as descending: any deterministic ordering leaves the later-placed nodes with fewer free target resources. `random` and `vertex_weights` remain the only strong strategies on tight topologies.

**`centrality` shines on trees, hurts on dense graphs.**
On trees it produces near-identity embeddings (chain\_avg ≈ 1.02, chain\_max ≈ 1.14). On ER/BA graphs the betweenness computation is cheap (O(VE) for n=8) but success rates degrade similarly to `degree` on tight topologies and add no quality benefit on spacious ones.

**`vertex_weights` (Cai et al. 2014) is the safety net on tight topologies.**
It matches `random` on success rate (29/30 ER Chimera, 30/30 BA Chimera) while using a smarter path-reuse scheme, but produces longer chains than `degree` on spacious topologies and is 5–20× slower than random.

#### Strategy selection guide

| Situation | Recommended strategy |
|---|---|
| Spacious topology (Zephyr / Pegasus), ER-like source, quality matters | `degree` |
| Spacious topology, mixed or BA source, or when success rate is critical | `degree_asc` |
| Tight topology (Chimera) or unknown source structure | `random` |
| Tight topology, scale-free source | `random` or `vertex_weights` |
| Sparse / tree-structured source graph | `centrality` |
| Offline quality maximisation, time not critical | `longest_chains` |
| Automatic selection (balanced) | `auto` — uses `select_embed_options` |

`auto` implements these rules: `centrality` for tree sources, `random` on tight topologies (ratio < 25), `degree_asc` on spacious topologies (balanced priority), `degree_asc + longest_chains` (quality priority).

---

### `PREFER_ARTICULATION_POINTS` benchmark

`PREFER_ARTICULATION_POINTS` anchors source articulation points (nodes whose removal disconnects the source graph) on the highest-degree free target node when they have no already-placed neighbours. The intuition is that structurally critical source nodes benefit from the most-connected target anchors.

> **Note:** All D-Wave topologies (Chimera, Zephyr, Pegasus) are biconnected (zero articulation points of their own), so this flag operates on the **source** graph only.

The flag was benchmarked standalone (`art_pts`) and combined with degree ordering (`degree_art`), against the `random` and `degree` baselines. 30 samples, ER n=8 p=0.4 unless noted.

#### ER n=8, p=0.4 → Chimera(4) — 128 qubits (tight, ratio 16)

| Strategy | Success | chain\_avg | elapsed |
|---|---|---|---|
| random | **28/30** | 1.66 | 285 ms |
| degree | 24/30 | 1.58 | 712 ms |
| art\_pts | **28/30** | 1.68 | 286 ms |
| degree\_art | 23/30 | 1.76 | 809 ms |

#### BA n=8, m=2 → Chimera(4) — tight

| Strategy | Success | chain\_avg | elapsed |
|---|---|---|---|
| random | **30/30** | **1.77** | 82 ms |
| degree | 21/30 | 2.27 | 1.07 s |
| art\_pts | **30/30** | **1.77** | **68 ms** |
| degree\_art | 23/30 | 2.27 | 631 ms |

#### ER n=8, p=0.4 → Zephyr(3) — 336 qubits (spacious, ratio 42)

| Strategy | Success | chain\_avg | elapsed |
|---|---|---|---|
| random | 30/30 | 1.79 | 10.5 ms |
| degree | 30/30 | 1.54 | **10.1 ms** |
| art\_pts | 30/30 | 1.73 | 10.7 ms |
| degree\_art | 30/30 | **1.52** | 10.1 ms |

#### ER n=8, p=0.4 → Pegasus(4) — 264 qubits (spacious, ratio 33)

| Strategy | Success | chain\_avg | elapsed |
|---|---|---|---|
| random | **30/30** | 1.81 | **6.4 ms** |
| degree | 29/30 | 1.70 | 496 ms |
| art\_pts | **30/30** | 1.80 | 6.5 ms |
| degree\_art | 29/30 | **1.67** | 421 ms |

#### Tree n=8 → Chimera(4)

| Strategy | Success | chain\_avg | elapsed |
|---|---|---|---|
| random | 30/30 | 1.32 | 8.0 ms |
| degree | 30/30 | **1.04** | **1.2 ms** |
| art\_pts | 30/30 | 1.28 | 4.6 ms |
| degree\_art | 30/30 | **1.05** | 1.3 ms |

#### Observations

**`art_pts` alone behaves like `random`.**
The flag only fires when placing isolated source nodes (no placed neighbours) that happen to be articulation points — a rare event in dense source graphs. Success rates and chain quality track `random` within noise across all configurations.

**`degree_art` mirrors `degree`.**
On tight topologies it inherits `degree`'s success-rate penalty (24→23 on Chimera ER, 21→23 on BA). On spacious topologies it matches or barely improves on `degree` (chain\_avg 1.54→1.52 on Zephyr, 1.70→1.67 on Pegasus), well within sample variance. On trees, `degree_art` (1.05) is indistinguishable from `degree` alone (1.04).

**`PREFER_ARTICULATION_POINTS` is not included in `select_embed_options`.**
No configuration showed a consistent, statistically meaningful improvement over the existing strategy rules. The flag is exposed for manual experimentation but is not recommended automatically.
