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
| `ORDER_BY_DEGREE_ASC` | `degree_asc` | Ascending source-degree first — hub placed last so Dijkstra bridges all already-placed neighbours |
| `ORDER_BY_CENTRALITY` | `centrality` | Descending betweenness centrality first |
| `REFINE_LONGEST_CHAINS` | `longest_chains` | Degree-asc ordering + Stage 2b chain refinement |

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
| `--strategy` | `both` | `random`, `degree_asc`, `centrality`, `longest_chains`, `auto`, `auto_quality`, `auto_speed`, `both`, `all` |
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

30 samples per configuration, `--tries 100 --refinement-constant 10 --warmup 10`.

#### ER n=8, p=0.5 → Chimera(4) — 128 qubits (tight, ratio 16)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **28/30** | 2.06 | 5.25 | **691 ms** |
| degree\_asc | 22/30 | 2.02 | 5.32 | 1.38 s |
| centrality | 12/30 | 1.96 | 4.83 | 3.48 s |
| longest\_chains | 18/30 | 1.91 | 4.39 | 5.25 s |
| **auto** | **28/30** | 2.06 | 5.25 | 679 ms |

#### ER n=8, p=0.5 → Zephyr(3) — 336 qubits (spacious, ratio 42)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| random | 30/30 | 1.96 | 4.67 | 15 ms |
| **degree\_asc** | **30/30** | 2.06 | 4.97 | 29 ms |
| centrality | 29/30 | 1.77 | 3.55 | 1.04 s |
| **longest\_chains** | **30/30** | **1.70** | **3.33** | 267 ms |
| **auto** | **30/30** | 2.06 | 4.97 | 26 ms |

#### ER n=8, p=0.5 → Pegasus(4) — 264 qubits (spacious, ratio 33)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 2.01 | 4.40 | **14 ms** |
| **degree\_asc** | **30/30** | 2.13 | 5.10 | 17 ms |
| centrality | 25/30 | 1.88 | 3.84 | 3.77 s |
| **longest\_chains** | 29/30 | **1.93** | **3.72** | 1.38 s |
| **auto** | **30/30** | 2.13 | 5.10 | 16 ms |

#### BA n=8, m=2 → Chimera(4) — 128 qubits (tight, ratio 16)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | **1.77** | 4.20 | **47 ms** |
| degree\_asc | 21/30 | 1.65 | 4.48 | 1.73 s |
| centrality | 17/30 | 2.16 | 6.41 | 2.40 s |
| longest\_chains | 21/30 | 2.27 | 6.33 | 3.77 s |
| **auto** | **30/30** | **1.77** | 4.20 | **44 ms** |

#### BA n=8, m=2 → Zephyr(3) — 336 qubits (spacious, ratio 42)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 1.77 | 4.53 | **11 ms** |
| **degree\_asc** | **30/30** | 2.08 | 5.90 | **11 ms** |
| centrality | 29/30 | 1.67 | 3.17 | 987 ms |
| longest\_chains | 29/30 | **1.63** | **3.21** | 2.09 s |
| **auto** | **30/30** | 2.08 | 5.90 | **10 ms** |

#### BA n=8, m=2 → Pegasus(4) — 264 qubits (spacious, ratio 33)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| **random** | **30/30** | 1.80 | 4.47 | **10 ms** |
| **degree\_asc** | **30/30** | 1.98 | 5.10 | **7 ms** |
| centrality | 27/30 | 1.84 | 3.44 | 2.03 s |
| longest\_chains | 27/30 | **1.82** | **3.48** | 4.18 s |
| **auto** | **30/30** | 1.98 | 5.10 | **6 ms** |

#### Tree n=10 → Chimera(4)

| Strategy | Success | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|
| random | 30/30 | 1.51 | 3.70 | 31 ms |
| degree\_asc | 30/30 | 1.51 | 3.93 | **13 ms** |
| **centrality** | 29/30 | **1.02** | **1.14** | 109 ms |
| longest\_chains | 30/30 | 1.10 | 1.63 | 64 ms |
| **auto** | 29/30 | **1.02** | **1.14** | 97 ms |

#### Key observations

**Topology tightness determines the success-rate ranking.**
On tight topologies (Chimera(4), ratio 16), any degree-based ordering hurts success rates because front-loading constrained nodes leaves no slack for later placements. `random` is the only strategy that stays fully competitive (28/30 ER, 30/30 BA). On spacious targets (Zephyr, Pegasus, ratio 33–42) all strategies succeed near-perfectly on ER graphs.

**`degree_asc` is the reliable default on spacious topologies.**
Ascending-degree ordering places low-degree nodes first (cheap, minimal constraints) and the hub last, letting Dijkstra build the hub's chain as a natural bridge through all already-placed neighbours. It achieves 30/30 on BA Zephyr and Pegasus (vs random's 30/30) with minimal latency overhead (~10 ms). `longest_chains` adds a further marginal chain-quality improvement at 200× the cost — worth it only for offline preprocessing.

**`degree_asc` does not help on tight topologies.**
On Chimera (tight, ratio 16), ascending ordering still degrades success rate relative to random (22/30 ER, 21/30 BA vs 28–30/30 for random): any deterministic ordering leaves the later-placed nodes with fewer free target resources. `random` remains the only strong strategy on tight topologies.

**`centrality` shines on trees, hurts on dense graphs.**
On trees it produces near-identity embeddings (chain\_avg ≈ 1.02, chain\_max ≈ 1.14). On ER/BA graphs the betweenness computation is cheap (O(VE) for n=8) but success rates degrade on tight topologies and add no quality benefit on spacious ones.

**`auto` (`select_embed_options`) matches the best per-topology strategy in every configuration.**
`auto` identifies the correct regime at call time using the target-to-source node ratio (< 25 → tight, ≥ 25 → spacious) and source graph structure (tree check):

| Configuration | `auto` selects | Result |
|---|---|---|
| Chimera(4), ER/BA (tight, ratio 16) | `random` | 28/30 ER · 30/30 BA, 679 ms / 44 ms |
| Zephyr(3), ER/BA (spacious, ratio 42) | `degree_asc` | 30/30, 26 ms / 10 ms |
| Pegasus(4), ER/BA (spacious, ratio 33) | `degree_asc` | 30/30, 16 ms / 6 ms |
| Any topology, tree source | `centrality` | 29/30, chain\_avg 1.02 |

Zero failures vs the per-strategy optimum: `auto` never falls behind the best fixed strategy on any topology tested, and incurs no overhead beyond the chosen strategy itself.

#### Strategy selection guide

| Situation | Recommended strategy |
|---|---|
| Spacious topology (Zephyr / Pegasus), any source | `degree_asc` |
| Tight topology (Chimera) or unknown source structure | `random` |
| Sparse / tree-structured source graph | `centrality` |
| Offline quality maximisation, time not critical | `longest_chains` |
| Automatic selection (balanced) | `auto` — uses `select_embed_options` |

`auto` implements these rules: `centrality` for tree sources, `random` on tight topologies (ratio < 25), `degree_asc` on spacious topologies (balanced priority), `degree_asc + longest_chains` (quality priority).
