# tutte/data

Pre-computed lookup tables used by the library, plus the latest benchmark
timings. The lookup tables ship in a compact **binary** format; the loaders
(`load_default_table`, `load_default_multigraph_table`, `load_default_merger_table`,
…) read `.bin`, which is the authoritative on-disk form.

## Files

| File | Format | Description |
|------|--------|-------------|
| `lookup_table.bin` | Binary v2 | Rainbow table (named cells + minor relationships). Loaded by default. |
| `multigraph_lookup_table.bin` | Binary | Cached multigraph intermediate polynomials. |
| `merger_lookup_table.bin` | Binary | Cell-merger lookup. |
| `rooted_lookup_table.bin` | Binary | Rooted-Tutte (boundary-keyed) lookup. |
| `benchmark_results.json` | JSON | Timing data from the most recent benchmark run. |

## Reading the binary tables

The `.bin` files are not human-readable. The **preferred way to inspect them is
the visualizer in `tutte/scripts/`**, which loads the rainbow table and serves a
browser UI:

```bash
python tutte/scripts/visualize_tutte.py   # then open the printed localhost URL
```

JSON mirrors of these tables are no longer written — `.bin` is authoritative,
and the loaders only fall back to a `.json` if the binary is missing.

## Regenerating

```bash
# Update table with new polynomials discovered during testing
python -m pytest tutte/tests/ -v --update-rainbow-table

# Full rebuild via standalone benchmark
python -m tutte.benchmarks.benchmark --timeout 300
```
