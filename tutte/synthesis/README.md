# tutte.synthesis

Synthesis engines for computing Tutte polynomials by decomposing graphs into known components.

## Modules

| Module | Description |
|--------|-------------|
| `base.py` | `UnionFind`, `BaseMultigraphSynthesizer`, `SynthesisResult` — shared infrastructure |
| `engine.py` | `SynthesisEngine` — main CEJ (Creation-Expansion-Join) algorithm |
| `algebraic.py` | `AlgebraicSynthesisEngine` — GCD/factorization-based decomposition |
| `hybrid.py` | `HybridSynthesisEngine` — combines algebraic + tiling for best coverage |

## Engine Hierarchy

```mermaid
graph TD
    B["BaseMultigraphSynthesizer<br/>(base.py)"] --> E["SynthesisEngine<br/>(engine.py)"]
    B --> A["AlgebraicSynthesisEngine<br/>(algebraic.py)"]
    E --> H["HybridSynthesisEngine<br/>(hybrid.py)"]
    A --> H
```

## Algorithm Selection (CEJ Engine)

```mermaid
graph TD
    I[Input Graph G] --> RT{Rainbow table hit?}
    RT -->|yes| Z[Return polynomial]
    RT -->|no| DC{Disconnected?}
    DC -->|yes| D["T(G₁ ∪ G₂) = T(G₁) × T(G₂)"]
    DC -->|no| CV{Cut vertex?}
    CV -->|yes| F["T(G₁ · G₂) = T(G₁) × T(G₂)"]
    CV -->|no| SP{Series-parallel?}
    SP -->|yes| SPT["O(n) SP decomposition"]
    SP -->|no| TILE{"≥20 edges + cell pattern?"}
    TILE -->|yes| T["Hierarchical tiling + chord addition"]
    TILE -->|no| EXP["Spanning tree expansion"]
    D --> Z
    F --> Z
    SPT --> Z
    T --> Z
    EXP --> Z
```

## Usage

```python
from tutte.lookup import load_default_table
from tutte.synthesis import SynthesisEngine, HybridSynthesisEngine

table = load_default_table()

# CEJ engine
result = SynthesisEngine(table).synthesize(graph)

# Hybrid engine (recommended — faster for structured graphs)
result = HybridSynthesisEngine(table).synthesize(graph)
```
