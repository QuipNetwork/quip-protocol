# tutte/benchmarks

Standalone benchmarking suite comparing synthesis engines against NetworkX.

## Usage

```bash
# Run benchmark (synthesizes all named graphs + atlas from empty table)
python -m tutte.benchmarks.benchmark --timeout 300 --nx-timeout 300

# Compare two benchmark runs (e.g., across branches)
python -m tutte.benchmarks.benchmark --compare run_a.json run_b.json
```

## What Gets Benchmarked

- **Named graphs**: Complete, cycle, path, wheel, grid, Petersen, D-Wave topologies
- **Graph atlas**: All connected graphs from `nx.graph_atlas_g()` up to 7 nodes
- **Engines**: CEJ (`SynthesisEngine`), Hybrid (`HybridSynthesisEngine`), NetworkX reference
- **Metrics**: Wall-clock time per graph, speedup ratios, minor relationship discovery

Results are written to `tutte/data/benchmark_results.json`.
