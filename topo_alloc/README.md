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
| `ORDER_BY_CENTRALITY` | `centrality` | Descending betweenness centrality first |
| `REFINE_LONGEST_CHAINS` | `longest_chains` | Degree ordering + Stage 2b chain refinement |
| `USE_VERTEX_WEIGHTS` | `vertex_weights` | Cai et al. 2014 vertex-weight Dijkstra scheme |

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
# ER graphs (n=8, p=0.5) → Chimera(4), all 5 strategies, 30 samples
python -m topo_alloc.bench_random \
    --graph-model er --nodes 8 --er-p 0.5 \
    --topology chimera --topology-size 4 \
    --samples 30 --strategy all

# Barabási-Albert → Zephyr(3), degree strategy only
python -m topo_alloc.bench_random \
    --graph-model ba --nodes 8 --ba-m 2 \
    --topology zephyr --topology-size 3 \
    --samples 30 --strategy degree

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
| `--strategy` | `both` | `random`, `degree`, `centrality`, `longest_chains`, `vertex_weights`, `both`, `all` |
| `--samples` | `20` | Number of random source graphs to generate |
| `--tries` | `30` | Embedding attempts per sample |
| `--no-detail` | off | Suppress per-sample table, show only aggregates |
| `--csv PATH` | — | Write per-sample results to a CSV file |

### Running the Tests

```bash
pytest topo_alloc/
```

The suite covers trivial inputs, small graphs, impossible embeddings, all ordering strategies, chain refinement, and `build_model` internals. All tests use fixed seeds for full reproducibility.

### Benchmark Findings

30 samples per configuration (ER n=8, p=0.5 unless noted). All five strategies benchmarked on the three D-Wave topologies.

#### ER n=8, p=0.5 → Chimera(4) — 128 qubits (tight)

| Strategy | Success | nodes (mean) | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|---|
| random | **27/30** | 16.19 | 2.04 | 5.19 | 671 ms |
| degree | 18/30 | 15.17 | 1.91 | 4.39 | 1.64 s |
| centrality | 12/30 | 15.50 | 1.96 | 4.83 | 2.07 s |
| longest\_chains | 18/30 | 15.17 | 1.91 | 4.39 | 3.37 s |
| vertex\_weights | **27/30** | 16.26 | 2.04 | 4.78 | 1.09 s |

#### ER n=8, p=0.5 → Zephyr(3) — 336 qubits

| Strategy | Success | nodes (mean) | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|---|
| random | 30/30 | 15.67 | 1.96 | 4.67 | **13.7 ms** |
| degree | 30/30 | **13.60** | **1.71** | **3.37** | 12.6 ms |
| centrality | 29/30 | 14.14 | 1.77 | 3.55 | 597 ms |
| longest\_chains | 30/30 | **13.53** | **1.70** | **3.33** | 498 ms |
| vertex\_weights | 30/30 | 15.77 | 1.98 | 4.77 | 340 ms |

#### ER n=8, p=0.5 → Pegasus(4) — 264 qubits

| Strategy | Success | nodes (mean) | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|---|
| random | **30/30** | 16.00 | 2.01 | 4.40 | **20 ms** |
| degree | 29/30 | 15.55 | 1.95 | 3.79 | 517 ms |
| centrality | 25/30 | **15.00** | **1.88** | 3.84 | 2.18 s |
| longest\_chains | 29/30 | 15.38 | 1.93 | **3.72** | 1.18 s |
| vertex\_weights | **30/30** | 17.63 | 2.21 | 4.53 | 341 ms |

#### BA n=8, m=2 → Chimera(4)

| Strategy | Success | nodes (mean) | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|---|
| random | **30/30** | **14.17** | **1.77** | 4.20 | **87 ms** |
| degree | 21/30 | 18.19 | 2.27 | 6.33 | 1.12 s |
| centrality | 17/30 | 17.29 | 2.16 | 6.41 | 1.47 s |
| longest\_chains | 21/30 | 18.19 | 2.27 | 6.33 | 2.44 s |
| vertex\_weights | **30/30** | **14.20** | **1.77** | **4.17** | 398 ms |

#### Tree n=10 → Chimera(4)

| Strategy | Success | nodes (mean) | chain\_avg | chain\_max | elapsed |
|---|---|---|---|---|---|
| random | 30/30 | 15.10 | 1.51 | 3.70 | 54 ms |
| degree | 30/30 | 11.03 | 1.10 | 1.63 | **24 ms** |
| centrality | 29/30 | **10.17** | **1.02** | **1.14** | 63 ms |
| longest\_chains | 30/30 | 11.00 | 1.10 | 1.63 | 116 ms |
| vertex\_weights | 30/30 | 14.37 | 1.44 | 3.23 | 55 ms |

#### Key observations

**Topology tightness flips the success-rate ranking.**
On tight topologies (Chimera(4), 128 qubits), `degree` and `centrality` ordering hurt success rates (12–18/30 vs 27/30 for random): front-loading the hardest nodes leaves no slack for later placements. On larger targets (Zephyr, Pegasus) all strategies succeed near-perfectly.

**`degree` is the best quality/speed tradeoff on spacious topologies.**
On Zephyr it cuts mean nodes used by ~13% (15.67 → 13.60) at virtually no extra cost (12.6 ms vs 13.7 ms for random).

**`longest_chains` squeezes a marginal further gain over `degree` at 40× the cost.**
Zephyr: 13.53 vs 13.60 nodes (+0.5% improvement), but 498 ms vs 13 ms. Worth it for offline preprocessing; not for interactive use.

**`centrality` shines on trees but has unpredictable latency on dense graphs.**
On trees it yields near-identity embeddings (chain\_max ≈ 1.14). On dense ER/BA graphs the betweenness centrality computation dominates (up to 17.5 s on one sample) with no consistent quality advantage.

**`vertex_weights` (Cai et al. 2014) maintains high success rates on tight topologies.**
It matches `random` on Chimera and ties for best on BA graphs, but produces longer chains than `degree` on spacious topologies.

#### Strategy selection guide

| Situation | Recommended strategy |
|---|---|
| Large topology (Zephyr / Pegasus), ER-like source | `degree` |
| Tight topology or unknown source structure | `random` |
| Sparse / tree-structured source graph | `centrality` |
| Scale-free (BA) source on tight topology | `random` or `vertex_weights` |
| Offline quality maximisation, time not critical | `longest_chains` |
