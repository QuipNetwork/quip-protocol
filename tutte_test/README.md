# Tutte Polynomial Analysis for Quantum Proof-of-Work

This directory contains tools for analyzing graphs via their Tutte polynomials, with applications to quantum annealing on D-Wave hardware.

## Overview

The **Tutte polynomial** `T(G; x, y)` is a graph invariant that encodes structural information including:

- Number of spanning trees: `T(1, 1)`
- Number of spanning forests: `T(2, 1)`
- Chromatic polynomial evaluations
- Reliability polynomial

For quantum proof-of-work, Tutte polynomials provide a principled way to characterize and compare graph structures used in Ising model problems.

## Key Results

### Z(1,1) Zephyr Topology

The smallest Zephyr topology Z(1,1) has been fully characterized:

| Property              | Value        |
| --------------------- | ------------ |
| Nodes                 | 12           |
| Edges                 | 22           |
| Spanning trees T(1,1) | **69,360**   |
| Polynomial terms      | 68           |
| x-degree              | 11           |
| y-degree              | 11           |
| Computation time      | ~1.4 seconds |

Full polynomial stored in `z11_tutte_polynomial.json`.

### Z(1,1) Synthesis Discovery

We discovered that **Z(1,1) can be synthesized from simpler components**:

```
Z(1,1) = K_4 core + C_8 periphery + 8 spoke edges
```

**Construction:**
```
        0---1
       /|   |\
      7 |   | 2
      | 8===9 |     (8-9-10-11 form K_4 core)
      6 |   | 3
       \|   |/
        5---4

Components:
  - K_4 core (nodes 8-11): 6 edges
  - C_8 cycle (nodes 0-7): 8 edges
  - Spokes: 8 edges connecting pairs of periphery nodes to each core node
    * Nodes 0,1 → core node 8
    * Nodes 2,3 → core node 9
    * Nodes 4,5 → core node 10
    * Nodes 6,7 → core node 11
```

**Properties:**
- Degree sequence: `[3,3,3,3,3,3,3,3,5,5,5,5]`
- Core nodes have degree 5 (3 from K_4 + 2 spokes)
- Periphery nodes have degree 3 (2 from C_8 + 1 spoke)
- Simple product K_4 × C_8 = 128 trees, but Z(1,1) has 69,360 trees (542× factor from spoke connectivity!)

This synthesis can be performed programmatically:
```python
from tutte_test.tutte_synthesis import synthesize_zephyr_z11

graph, polynomial, edge_info = synthesize_zephyr_z11()
assert polynomial.num_spanning_trees() == 69360
```

### Computational Complexity

| Edges            | Recursive Calls | Time  | Feasibility    |
| ---------------- | --------------- | ----- | -------------- |
| 12               | 714             | 2ms   | Easy           |
| 16               | 2,739           | 15ms  | Easy           |
| 20               | 54,300          | 0.37s | Feasible       |
| 22 (Z(1,1))      | 318,170         | 1.4s  | Feasible       |
| 13,716 (Z(12,2)) | ~10^4000        | ∞     | **Impossible** |

The deletion-contraction algorithm is exponential in edge count, but practical for graphs with ≤25 edges.

### Rainbow Table

Pre-computed Tutte polynomials for **161 graph minors** are stored in `tutte_rainbow_table.json`:

| Family              | Count | Examples                             |
| ------------------- | ----- | ------------------------------------ |
| Complete graphs K_n | 7     | K_2 through K_8                      |
| Cycles C_n          | 22    | C_3 through C_24                     |
| Paths P_n           | 23    | P_2 through P_24                     |
| Complete bipartite  | 10    | K_2,2 through K_5,5                  |
| Wheel graphs        | 9     | W_3 through W_11                     |
| Grid graphs         | 12    | 2×2 through 5×3                      |
| Hypercubes          | 3     | Q_1 through Q_3                      |
| Ladder graphs       | 7     | Ladder_2 through Ladder_8            |
| Circulant graphs    | 7     | Circ(5,[1,2]) through Circ(11,[1,2]) |
| Special graphs      | 2     | Petersen, Möbius-Kantor              |
| **Zephyr minors**   | 69    | 4-8 node induced subgraphs           |
| **Z(1,1)**          | 1     | Complete Zephyr topology             |

## Graph Composition & Synthesis

### Composition Operations

The module supports various graph composition operations:

| Operation         | Formula                          | Description                        |
| ----------------- | -------------------------------- | ---------------------------------- |
| Disjoint union    | `T(G₁ ∪ G₂) = T(G₁) × T(G₂)`     | No shared vertices                 |
| Cut vertex (1-sum)| `T(G₁ ·₁ G₂) = T(G₁) × T(G₂)`    | Share single vertex                |
| 2-sum             | Complex formula                  | Share edge, then delete it         |
| k-clique sum      | Depends on structure             | Share k-clique, optionally delete  |
| Parallel connection | Series-parallel formula        | Connect at two distinguished nodes |

### K-Join Operations

K-join is the inverse of k-cut - connecting components to increase connectivity:

```python
from tutte_test.tutte_synthesis import perform_1_join, perform_2_join, perform_3_join

# 1-join: connect at cut vertex (preserves polynomial product)
result = perform_1_join(g1, v1, g2, v2)

# 2-join: connect along edge, optionally delete shared edge
result = perform_2_join(g1, [u1, v1], g2, [u2, v2], delete_shared_edge=True)

# 3-join: connect on triangle, optionally delete triangle edges
result = perform_3_join(g1, [a1, b1, c1], g2, [a2, b2, c2], delete_triangle=True)
```

### Polynomial Difference Analysis

Analyze what operations transform one graph into another:

```python
from tutte_test.tutte_synthesis import analyze_polynomial_difference

# What's needed to go from K_3 to K_4?
diff = analyze_polynomial_difference(k4_poly, k3_poly)

print(f"Pure x terms (bridges): {diff.pure_x_terms}")
print(f"Pure y terms (cycles): {diff.pure_y_terms}")
print(f"Mixed terms (complex): {diff.mixed_terms}")
```

**Interpretation:**
- Pure `x^k` terms → k bridges/cut edges needed
- Pure `y^k` terms → k-cycles needed
- Mixed `x^i·y^j` terms → specific graph structures

### Synthesis Engine

Build graphs matching target Tutte polynomials:

```python
from tutte_test.tutte_synthesis import SynthesisEngine
from tutte_test.tutte_utils import TuttePolynomial

engine = SynthesisEngine()

# Synthesize a graph with target polynomial
target = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})  # K_3
result = engine.synthesize(target)

if result.success:
    print(f"Built graph with {result.graph.num_nodes()} nodes")
    print(f"Verified: {result.verified}")
```

**Synthesis Strategies:**
1. **Direct lookup**: Check rainbow table for exact match
2. **Factorization**: Find polynomials P₁, P₂ where P₁ × P₂ = target
3. **Incremental**: Build from smaller components using k-joins

## Files

### Core Modules

| File                    | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `tutte_utils.py`        | Core: TuttePolynomial, GraphBuilder, computation, Ising generation   |
| `graph_composition.py`  | Graph operations: unions, joins, clique sums, cut analysis           |
| `tutte_synthesis.py`    | Polynomial-guided synthesis, k-join primitives, SynthesisEngine      |

### Rainbow Table

| File                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `build_rainbow_table.py`    | Build and query pre-computed Tutte polynomials |
| `tutte_rainbow_table.json`  | Pre-computed polynomials (161 entries, ~122KB) |
| `z11_tutte_polynomial.json` | Complete Z(1,1) Tutte polynomial               |

### Testing

| File                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `test_graph_composition.py` | 38 tests: composition, networkx verification   |

### Integration

| File                   | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `tutte_pow_protocol.py`| Tutte-aware proof-of-work protocol design         |

## Usage

### Rebuilding the Rainbow Table

```bash
python -m tutte_test.build_rainbow_table
```

### Computing Tutte Polynomial for a Graph

```python
from tutte_test.tutte_utils import compute_tutte_polynomial, create_cycle_graph

# Create a graph
g = create_cycle_graph(5)

# Compute polynomial
t = compute_tutte_polynomial(g)
print(f"T(C_5) = {t}")
print(f"Spanning trees: {t.num_spanning_trees()}")
```

### Using the Rainbow Table

```python
from tutte_test.build_rainbow_table import RainbowTable

# Load pre-computed table
table = RainbowTable.load('tutte_test/tutte_rainbow_table.json')

# Lookup by name
poly = table.lookup_by_name('Petersen')
print(f"Petersen spanning trees: {poly.num_spanning_trees()}")

# Get full entry
entry = table.get_entry('K_4')
print(f"K_4: {entry['polynomial_str']}")
```

### Generating Ising Model from Tutte Polynomial

```python
from tutte_test.tutte_utils import generate_ising_from_tutte, KNOWN_TUTTE_POLYNOMIALS

tutte = KNOWN_TUTTE_POLYNOMIALS['K4']
h, J, nodes, edges = generate_ising_from_tutte(nonce=12345, tutte=tutte)
```

### Graph Composition

```python
from tutte_test.graph_composition import (
    disjoint_union,
    cut_vertex_join,
    clique_sum,
    find_cut_vertices,
    find_bridges,
    analyze_k_cuts,
)
from tutte_test.tutte_utils import create_complete_graph

# Create two K_4 graphs
g1 = create_complete_graph(4)
g2 = create_complete_graph(4)

# Join at cut vertex (1-join)
joined = cut_vertex_join(g1, 0, g2, 0)

# Analyze cut structure
cuts = analyze_k_cuts(joined, max_k=3)
print(f"1-cuts: {len(cuts[1])}")
print(f"2-cuts: {len(cuts[2])}")
```

### Synthesizing Z(1,1)

```python
from tutte_test.tutte_synthesis import synthesize_zephyr_z11
from tutte_test.build_rainbow_table import RainbowTable

# Build Z(1,1) from components
graph, polynomial, edge_info = synthesize_zephyr_z11()

print(f"Nodes: {graph.num_nodes()}")
print(f"Edges: {graph.num_edges()}")
print(f"  K_4 core: {len(edge_info['core'])} edges")
print(f"  C_8 cycle: {len(edge_info['cycle'])} edges")
print(f"  Spokes: {len(edge_info['spokes'])} edges")
print(f"Spanning trees: {polynomial.num_spanning_trees()}")

# Verify against rainbow table
rt = RainbowTable.load('tutte_test/tutte_rainbow_table.json')
z11_poly = rt._entry_to_polynomial(rt.get_entry('Z(1,1)'))
assert polynomial == z11_poly  # Exact match!
```

### Using Shared Utilities

```python
from tutte_test.tutte_utils import (
    networkx_to_graphbuilder,
    is_bridge,
    graph_to_canonical_key,
    get_zephyr_graph,
)
import networkx as nx

# Convert NetworkX graph
G = nx.petersen_graph()
gb = networkx_to_graphbuilder(G)

# Get canonical key for isomorphism-invariant comparison
key = graph_to_canonical_key(G)

# Get Zephyr topology (with fallback if D-Wave unavailable)
z11 = get_zephyr_graph(1, 1)
```

## Theoretical Background

### Composition Rules

Tutte polynomials satisfy useful composition rules:

**Disjoint Union:**
```
T(G₁ ∪ G₂) = T(G₁) × T(G₂)
```

**Cut Vertex (1-sum):**
```
T(G₁ ·₁ G₂) = T(G₁) × T(G₂)
```
When two graphs share only a single vertex (cut vertex), their Tutte polynomials multiply.

**Deletion-Contraction:**
```
T(G) = T(G-e) + T(G/e)     for regular edge e
T(G) = x · T(G-e)          for bridge e
T(G) = y · T(G-e)          for loop e
```

### K-Cut/K-Join Duality

- **k-cut**: Removing k edges/vertices that disconnect graph components
- **k-join**: The inverse - connecting components to increase edge connectivity

The polynomial difference `T(target) - T(source)` reveals what structure is missing:
- Positive coefficients indicate structure to add
- Term types indicate operation types (bridges, cycles, etc.)

### Synthesis Insights

From our Z(1,1) synthesis work:

1. **Spoke connectivity multiplies trees**: Adding 8 spoke edges between K_4 and C_8 creates a 542× increase in spanning trees (from 128 to 69,360)

2. **Degree sequence constrains structure**: Z(1,1)'s `[3,3,3,3,3,3,3,3,5,5,5,5]` degree sequence uniquely determines the K_4-core + C_8-periphery topology

3. **Rainbow table enables synthesis**: By finding graph components in the rainbow table, we can build target graphs through composition operations

## Running Tests

```bash
# Run the test suite (38 tests)
python -m unittest tutte_test.test_graph_composition -v

# Tests include:
# - Tutte polynomial basics (paths, cycles, complete graphs)
# - Composition operations (unions, joins, sums)
# - Cut analysis (bridges, cut vertices, k-cuts)
# - NetworkX verification (cross-check our implementation)
# - Rainbow table consistency
# - Synthesis engine correctness
```

## Limitations

1. **Computational complexity**: Direct computation infeasible for graphs with >30 edges
2. **Z(12,2) Zephyr**: 13,716 edges makes exact computation impossible
3. **Subgraph matching**: Finding minors in large graphs requires heuristics
4. **Synthesis completeness**: Not all polynomials have efficient synthesis paths

## References

- Tutte, W.T. (1954). "A contribution to the theory of chromatic polynomials"
- Bollobás, B. (1998). "Modern Graph Theory" - Chapter on Tutte polynomials
- D-Wave Ocean SDK documentation for Zephyr topology
