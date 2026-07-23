#!/usr/bin/env python3
"""Generate a drive-mode topology-spec fixture with NATIVE node ids (no remap).

The Rust miners now resolve `TopologyHash` jobs against the session-cached
topology, mapping native (possibly sparse) node ids to dense positions
themselves — so the spec keeps the D-Wave qubit ids as-is.

Usage:
    python benchmarks/drive/gen_topology_spec.py advantage2-system1
    python benchmarks/drive/gen_topology_spec.py smoke

Writes benchmarks/drive/<name>.spec.json.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

# Consensus allowed-value sets (milli): h ∈ {-1,0,+1}, J ∈ {-1,+1}.
ALLOWED_H_MILLI = [-1000, 0, 1000]
ALLOWED_J_MILLI = [-1000, 1000]


def from_dwave(name: str) -> dict:
    """Load a bundled D-Wave topology, keeping its native (sparse) ids."""
    gz = ROOT / "dwave_topologies" / "topologies" / f"{name.replace('-', '_')}.json.gz"
    with gzip.open(gz) as fh:
        d = json.load(fh)
    nodes = list(d["nodes"])
    edges = [[int(u), int(v)] for u, v in d["edges"]]
    return {"nodes": nodes, "edges": edges}


def smoke() -> dict:
    """A small 8-node ring — fast, native ids 0..7."""
    n = 8
    nodes = list(range(n))
    edges = [[i, (i + 1) % n] for i in range(n)]
    return {"nodes": nodes, "edges": edges}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    graph = smoke() if name == "smoke" else from_dwave(name)
    spec = {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "allowed_h_milli": ALLOWED_H_MILLI,
        "allowed_j_milli": ALLOWED_J_MILLI,
    }
    out = OUT_DIR / f"{name}.spec.json"
    out.write_text(json.dumps(spec))
    ids = graph["nodes"]
    print(
        f"wrote {out.relative_to(ROOT)}: {len(ids)} nodes "
        f"(id range {min(ids)}..{max(ids)}), {len(graph['edges'])} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
