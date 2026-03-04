# tutte/tests

Parametrized test suite validating synthesis against Kirchhoff's theorem and NetworkX.

## Running

```bash
# Full suite (excludes slow tests by default)
python -m pytest tutte/tests/ -v -m "not slow"

# Include graph atlas exhaustive tests
python -m pytest tutte/tests/ -v

# Update rainbow table with newly computed polynomials
python -m pytest tutte/tests/ -v --update-rainbow-table

# Collect benchmark timings
python -m pytest tutte/tests/ -v --benchmark
```

## Test Sections

- **A. Spanning tree verification** — Kirchhoff matrix-tree theorem cross-check
- **B. Cross-validation** — Compare synthesis output against `nx.tutte_polynomial()`
- **C. Minor finding** — Rainbow table minor relationship correctness
- **D. Graph atlas coverage** — All connected graphs up to 7 nodes
- **E. D-Wave topologies** — Chimera, Zephyr, Pegasus hardware graphs
- **F. Composition formulas** — Bridge, chord, cut vertex, disjoint union
- **G. Performance regression** — Timing bounds on key graphs

## Markers

| Marker | Description |
|--------|-------------|
| `slow` | Graph atlas exhaustive tests (deselect with `-m "not slow"`) |
| `perf` | Performance regression tests |
