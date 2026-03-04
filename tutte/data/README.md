# tutte/data

Pre-computed data files used by the library.

## Files

| File | Format | Description |
|------|--------|-------------|
| `lookup_table.bin` | Binary v2 | Compact rainbow table (~465 KB). Includes minor relationships. Loaded by default. |
| `lookup_table.json` | JSON | Human-readable rainbow table (~17 MB). Fallback if binary load fails. |
| `benchmark_results.json` | JSON | Timing data from the most recent benchmark run. |

## Regenerating

```bash
# Update table with new polynomials discovered during testing
python -m pytest tutte/tests/ -v --update-rainbow-table

# Full rebuild via standalone benchmark
python -m tutte.benchmarks.benchmark --timeout 300
```
