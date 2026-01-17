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

### Z(1,1) Zephyr Topology (Verified)

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

## Files

### Core Modules

| File                   | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `tutte_to_ising.py`    | Core Tutte polynomial computation, graph construction, Ising generation  |
| `tutte_integration.py` | Advanced construction algorithms, polynomial parsing, difficulty scaling |
| `tutte_utils.py`       | Shared utilities (graph conversion, bridge detection, canonical keys)    |

### Rainbow Table

| File                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `rainbow_table.py`          | Build and query pre-computed Tutte polynomials |
| `tutte_rainbow_table.json`  | Pre-computed polynomials (161 entries, ~122KB) |
| `z11_tutte_polynomial.json` | Complete Z(1,1) Tutte polynomial               |

### Integration

| File                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------------- |
| `quantum_proof_of_work_extended.py` | Tutte-aware proof-of-work validation, energy/diversity calculations |

## Usage

### Rebuilding the Rainbow Table

```bash
python tutte_test/rainbow_table.py
```

### Loading a Polynomial from Specification

```python
from tutte_test.tutte_integration import create_tutte_from_specification
from tutte_test.tutte_to_ising import TuttePolynomial

# Method 1: String notation
tutte = create_tutte_from_specification("x^3 + 2x^2y + xy^2 + y^3")

# Method 2: Named graph
tutte = create_tutte_from_specification("K4")

# Method 3: Direct coefficient dict {(x_power, y_power): coefficient}
tutte = TuttePolynomial({(3, 0): 1, (2, 1): 2, (1, 2): 1, (0, 3): 1})
```

### Computing Tutte Polynomial for a Graph

```python
from tutte_test.tutte_to_ising import compute_tutte_polynomial, create_cycle_graph

# Create a graph
g = create_cycle_graph(5)

# Compute polynomial
t = compute_tutte_polynomial(g)
print(f"T(C_5) = {t}")
print(f"Spanning trees: {t.num_spanning_trees()}")
```

### Using the Rainbow Table

```python
from tutte_test.rainbow_table import RainbowTable

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
from tutte_test.tutte_to_ising import generate_ising_from_tutte, KNOWN_TUTTE_POLYNOMIALS

tutte = KNOWN_TUTTE_POLYNOMIALS['K4']
h, J, nodes, edges = generate_ising_from_tutte(nonce=12345, tutte=tutte)
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

## Composition Rules

Tutte polynomials satisfy useful composition rules:

### Disjoint Union

```
T(G₁ ∪ G₂) = T(G₁) × T(G₂)
```

### Deletion-Contraction

```
T(G) = T(G-e) + T(G/e)     for regular edge e
T(G) = x · T(G-e)          for bridge e
T(G) = y · T(G-e)          for loop e
```

## Limitations

1. **Computational complexity**: Direct computation infeasible for graphs with >30 edges
2. **Z(12,2) Zephyr**: 13,716 edges makes exact computation impossible
3. **Subgraph matching**: Finding minors in large graphs requires heuristics

## References

- Tutte, W.T. (1954). "A contribution to the theory of chromatic polynomials"
- Bollobás, B. (1998). "Modern Graph Theory" - Chapter on Tutte polynomials
- D-Wave Ocean SDK documentation for Zephyr topology
