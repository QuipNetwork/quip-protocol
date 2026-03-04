"""Benchmark Tutte polynomial computation from an empty rainbow table.

Each graph is synthesized from scratch. After synthesis, its polynomial is
added to the rainbow table so subsequent graphs can use it as a tile/minor.
Graphs are sorted by edge count so simpler ones seed the table first.

Engines benchmarked:
    - CEJ (SynthesisEngine): creation-expansion-join with growing rainbow table
    - Hybrid (HybridSynthesisEngine): algebraic + tiling with growing rainbow table
    - NetworkX (nx.tutte_polynomial): reference implementation via deletion-contraction

Standalone usage:
    python -m tutte.benchmarks.benchmark
    python -m tutte.benchmarks.benchmark --compare file1.json file2.json

Pytest integration: run with --benchmark flag to collect timings automatically.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import networkx as nx
from tutte.graph import (Graph, complete_graph, cycle_graph, grid_graph,
                              path_graph, petersen_graph, wheel_graph)
from tutte.synthesis import HybridSynthesisEngine
from tutte.polynomial import TuttePolynomial
from tutte.lookup import RainbowTable, save_binary_rainbow_table
from tutte.synthesis import SynthesisEngine
from tutte.validation import count_spanning_trees_kirchhoff

# ---------------------------------------------------------------------------
# Named graph set (merged with atlas below)
# ---------------------------------------------------------------------------

NAMED_GRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_7", lambda: complete_graph(7)),
    ("C_5", lambda: cycle_graph(5)),
    ("C_10", lambda: cycle_graph(10)),
    ("C_15", lambda: cycle_graph(15)),
    ("W_5", lambda: wheel_graph(5)),
    ("W_7", lambda: wheel_graph(7)),
    ("Petersen", lambda: petersen_graph()),
    ("Grid_3x3", lambda: grid_graph(3, 3)),
    ("Grid_4x4", lambda: grid_graph(4, 4)),
]


def _try_dwave_graphs():
    """Add D-Wave graphs if available: Chimera C1-C16, Pegasus P1-P16, Zephyr Z(1,1)."""
    extras = []
    try:
        import dwave_networkx as dnx
        for m in range(1, 17):
            _m = m  # capture for lambda
            extras.append((f"Cm{m}", lambda _m=_m: Graph.from_networkx(dnx.chimera_graph(_m))))
        for m in range(1, 17):
            _m = m
            G = dnx.pegasus_graph(_m)
            if G.number_of_nodes() > 0:
                extras.append((f"Pm{m}", lambda _m=_m: Graph.from_networkx(dnx.pegasus_graph(_m))))
        extras.append(("Z1_1", lambda: Graph.from_networkx(dnx.zephyr_graph(1, 1))))
    except ImportError:
        pass
    return extras


def _atlas_graphs():
    """All connected atlas graphs with >= 1 edge."""
    for i in range(1, 1253):
        try:
            G = nx.graph_atlas(i)
        except Exception:
            continue
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        if not nx.is_connected(G):
            continue
        yield f"atlas_{i}", Graph.from_networkx(G)


def _build_graph_list():
    """Build the full sorted graph list: named + atlas + dwave, sorted by edges.

    Deduplication uses canonical keys for small graphs (<=30 edges) only,
    since WL hashing is too expensive for large D-Wave topologies.
    """
    # Small graphs: deduplicate by canonical key
    small = []
    for name, builder in NAMED_GRAPHS:
        small.append((name, builder()))

    for name, g in _atlas_graphs():
        small.append((name, g))

    seen = {}
    deduped = []
    for name, g in small:
        key = g.canonical_key()
        if key in seen:
            if not name.startswith("atlas_") and seen[key].startswith("atlas_"):
                deduped = [(n, gr) if n != seen[key] else (name, g) for n, gr in deduped]
                seen[key] = name
            continue
        seen[key] = name
        deduped.append((name, g))

    # Large D-Wave graphs: no deduplication needed (unique topologies)
    for name, builder in _try_dwave_graphs():
        g = builder()
        deduped.append((name, g))

    deduped.sort(key=lambda x: (x[1].edge_count(), x[1].node_count(), x[0]))
    return deduped


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

_TIMEOUT = "TIMEOUT"
_ERROR = "ERROR"


def _time_fn(fn, timeout_s=60):
    """Time a function. Returns (elapsed_ms, result, None) on success,
    (None, None, _TIMEOUT) on timeout, or (None, None, _ERROR) on exception."""
    class _TimeoutExc(BaseException):
        """Inherits BaseException so it won't be caught by `except Exception`
        inside networkx/sympy internals."""
        pass

    def _handler(signum, frame):
        raise _TimeoutExc()

    old = None
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)

    try:
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return round(elapsed, 3), result, None
    except _TimeoutExc:
        return None, None, _TIMEOUT
    except Exception:
        return None, None, _ERROR
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)


def _tutte_networkx(G_nx):
    """Compute Tutte polynomial via NetworkX. Does NOT swallow exceptions,
    so SIGALRM timeouts propagate correctly."""
    from sympy import Poly, symbols
    x, y = symbols('x y')
    tutte_sympy = nx.tutte_polynomial(G_nx)
    poly = Poly(tutte_sympy, x, y)
    coeffs = {}
    for monom, coeff in poly.as_dict().items():
        coeffs[monom] = int(coeff)
    return TuttePolynomial.from_coefficients(coeffs)


def _fmt(ms):
    if ms is None:
        return "-"
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks(timeout_s=60, nx_timeout_s=30):
    """Run benchmarks from empty rainbow tables.

    Three engines are benchmarked independently:
      - CEJ (SynthesisEngine) with its own growing table
      - Hybrid (HybridSynthesisEngine) with its own growing table
      - NetworkX (nx.tutte_polynomial) as reference (no table)

    After each graph, if an engine produced a correct result, the polynomial
    is added to that engine's rainbow table for future graphs.
    """
    cej_table = RainbowTable()
    cej_engine = SynthesisEngine(cej_table)

    hybrid_table = RainbowTable()
    hybrid_engine = HybridSynthesisEngine(table=hybrid_table)

    graphs = _build_graph_list()
    results = []
    stats = {"cej_ok": 0, "hybrid_ok": 0, "nx_ok": 0,
             "cej_fail": 0, "hybrid_fail": 0, "nx_fail": 0,
             "poly_mismatch": 0}

    hdr = (f"{'#':>5} {'Graph':<20} {'N':>3} {'M':>3} {'Trees':>14} "
           f"{'CEJ':>10} {'Hybrid':>10} {'NetworkX':>10}")
    print(f"Benchmarking {len(graphs)} graphs (3 engines, empty tables)")
    print(hdr)
    print("-" * len(hdr))

    # Track which edge counts have been proven unsolvable per engine,
    # so we don't waste timeout_s on every C2-C16 graph.
    cej_max_solved = 0
    hybrid_max_solved = 0
    nx_max_solved = 0

    for idx, (name, graph) in enumerate(graphs, 1):
        n, m = graph.node_count(), graph.edge_count()
        G_nx = graph.to_networkx()

        # Ground truth via Kirchhoff
        kirchhoff = count_spanning_trees_kirchhoff(graph)

        # --- CEJ engine ---
        if m > cej_max_solved + 30:
            # Way beyond frontier — skip without wasting timeout
            cej_ms, cej_result, cej_err = None, None, "UNSOLVED"
        else:
            cej_ms, cej_result, cej_err = _time_fn(
                lambda: cej_engine.synthesize(graph), timeout_s
            )
        cej_ok = (cej_result is not None
                  and cej_result.polynomial.num_spanning_trees() == kirchhoff)
        if cej_ok:
            cej_table.add(graph, name, cej_result.polynomial, cej_result.minors_used)
            stats["cej_ok"] += 1
            cej_status = "OK"
            cej_max_solved = max(cej_max_solved, m)
        else:
            stats["cej_fail"] += 1
            cej_status = cej_err or "WRONG"

        # --- Hybrid engine ---
        if m > hybrid_max_solved + 30:
            hybrid_ms, hybrid_result, hybrid_err = None, None, "UNSOLVED"
        else:
            hybrid_ms, hybrid_result, hybrid_err = _time_fn(
                lambda: hybrid_engine.synthesize(graph), timeout_s
            )
        hybrid_ok = (hybrid_result is not None
                     and hybrid_result.polynomial.num_spanning_trees() == kirchhoff)
        if hybrid_ok:
            hybrid_table.add(graph, name, hybrid_result.polynomial, hybrid_result.minors_used)
            stats["hybrid_ok"] += 1
            hybrid_status = "OK"
            hybrid_max_solved = max(hybrid_max_solved, m)
        else:
            stats["hybrid_fail"] += 1
            hybrid_status = hybrid_err or "WRONG"

        # --- NetworkX ---
        if m > nx_max_solved + 10:
            nx_ms, nx_result, nx_err = None, None, "UNSOLVED"
        else:
            nx_ms, nx_result, nx_err = _time_fn(
                lambda: _tutte_networkx(G_nx), nx_timeout_s
            )
        nx_ok = (nx_result is not None
                 and nx_result.num_spanning_trees() == kirchhoff)
        if nx_ok:
            stats["nx_ok"] += 1
            nx_status = "OK"
            nx_max_solved = max(nx_max_solved, m)
        else:
            stats["nx_fail"] += 1
            nx_status = nx_err or "WRONG"

        # --- Polynomial cross-validation ---
        poly_match = {"cej_vs_nx": None, "hybrid_vs_nx": None}
        if nx_ok:
            if cej_ok:
                match = cej_result.polynomial == nx_result
                poly_match["cej_vs_nx"] = match
                if not match:
                    cej_status = "POLY_MISMATCH"
                    stats["poly_mismatch"] += 1
            if hybrid_ok:
                match = hybrid_result.polynomial == nx_result
                poly_match["hybrid_vs_nx"] = match
                if not match:
                    hybrid_status = "POLY_MISMATCH"
                    stats["poly_mismatch"] += 1

        trees_str = f"{kirchhoff:,}" if kirchhoff >= 0 else "?"

        # Show failure reason inline when not OK
        cej_col = _fmt(cej_ms) if cej_status == "OK" else cej_status
        hybrid_col = _fmt(hybrid_ms) if hybrid_status == "OK" else hybrid_status
        nx_col = _fmt(nx_ms) if nx_ok else nx_status

        print(f"{idx:>5} {name:<20} {n:>3} {m:>3} {trees_str:>14} "
              f"{cej_col:>10} {hybrid_col:>10} {nx_col:>10}",
              flush=True)

        results.append({
            "name": name,
            "nodes": n,
            "edges": m,
            "spanning_trees": kirchhoff,
            "timings_ms": {
                "synthesis_cej": cej_ms,
                "synthesis_hybrid": hybrid_ms,
                "networkx": nx_ms,
            },
            "status": {
                "cej": cej_status,
                "hybrid": hybrid_status,
                "networkx": nx_status,
            },
            "polynomial_match": poly_match,
        })

    # Summary
    print("-" * len(hdr))
    print(f"CEJ:     {stats['cej_ok']} ok, {stats['cej_fail']} failed "
          f"({len(cej_table)} table entries)")
    print(f"Hybrid:  {stats['hybrid_ok']} ok, {stats['hybrid_fail']} failed "
          f"({len(hybrid_table)} table entries)")
    print(f"NetworkX:{stats['nx_ok']} ok, {stats['nx_fail']} failed")
    if stats["poly_mismatch"]:
        print(f"WARNING: {stats['poly_mismatch']} polynomial mismatches vs NetworkX!")
    else:
        print(f"Polynomial cross-validation: all matches OK")

    # Per-edge-count summary
    by_edges = {}
    for r in results:
        m = r["edges"]
        if m not in by_edges:
            by_edges[m] = {"count": 0, "cej": [], "hybrid": [], "nx": []}
        by_edges[m]["count"] += 1
        for key in ("synthesis_cej", "synthesis_hybrid", "networkx"):
            short = key.replace("synthesis_", "").replace("networkx", "nx")
            t = r["timings_ms"][key]
            if t is not None:
                by_edges[m][short].append(t)

    print(f"\n{'Edges':>5} {'Count':>5}  {'CEJ avg':>10} {'Hybrid avg':>10} {'NX avg':>10}")
    print("-" * 50)
    for m in sorted(by_edges):
        b = by_edges[m]
        def avg(lst):
            return sum(lst) / len(lst) if lst else None
        print(f"{m:>5} {b['count']:>5}  "
              f"{_fmt(avg(b['cej'])):>10} {_fmt(avg(b['hybrid'])):>10} {_fmt(avg(b['nx'])):>10}")

    sys.stdout.flush()
    return results, cej_table


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(file1, file2):
    """Compare two benchmark result files."""
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    b1 = data1["metadata"].get("branch", "?")
    b2 = data2["metadata"].get("branch", "?")
    print(f"Comparing: {b1} vs {b2}\n")

    results1 = {r["name"]: r for r in data1["results"]}
    results2 = {r["name"]: r for r in data2["results"]}

    common = sorted(set(results1) & set(results2),
                    key=lambda n: (results1[n]["edges"], n))

    # Compare each engine
    for engine_key in ("synthesis_cej", "synthesis_hybrid", "networkx"):
        label = engine_key.replace("synthesis_", "").upper()
        print(f"\n--- {label} ---")
        print(f"{'Graph':<20} {'M':>3}  {b1:>10}  {b2:>10}  {'Speedup':>8}")
        print("-" * 60)

        speedups = []
        for name in common:
            r1, r2 = results1[name], results2[name]
            t1 = r1["timings_ms"].get(engine_key)
            t2 = r2["timings_ms"].get(engine_key)
            m = r1["edges"]

            if t1 and t2 and t2 > 0:
                s = t1 / t2
                speedups.append(s)
                print(f"{name:<20} {m:>3}  {_fmt(t1):>10}  {_fmt(t2):>10}  {s:>7.2f}x")

        if speedups:
            geo_mean = 1.0
            for s in speedups:
                geo_mean *= s
            geo_mean **= (1 / len(speedups))
            print(f"Geometric mean speedup: {geo_mean:.2f}x over {len(speedups)} graphs")


# ---------------------------------------------------------------------------
# Save / CLI
# ---------------------------------------------------------------------------

def save_results(results, cej_table=None):
    """Save benchmark results to JSON and optionally save rainbow table."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        branch = "unknown"

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "branch": branch,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "results": results,
    }

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    out_path = os.path.join(base_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    if cej_table and len(cej_table) > 0:
        # Compute comprehensive structural minor relationships
        print(f"\nComputing structural minor relationships for {len(cej_table)} entries...",
              flush=True)
        t0 = time.perf_counter()
        relationships = cej_table.compute_minor_relationships()
        elapsed = time.perf_counter() - t0
        total_minors = sum(len(v) for v in relationships.values())
        print(f"Found {total_minors} minor relationships across "
              f"{len(relationships)} entries ({elapsed:.1f}s)", flush=True)

        json_path = os.path.join(base_dir, "lookup_table.json")
        bin_path = os.path.join(base_dir, "lookup_table.bin")
        cej_table.save(json_path)
        save_binary_rainbow_table(cej_table, bin_path)
        print(f"Rainbow table saved: {len(cej_table)} entries ({json_path}, {bin_path})",
              flush=True)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Tutte polynomial synthesis from empty rainbow table"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE1", "FILE2"),
        help="Compare two benchmark result files",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-graph timeout in seconds for CEJ/hybrid (default: 60)",
    )
    parser.add_argument(
        "--nx-timeout", type=int, default=30,
        help="Per-graph timeout in seconds for NetworkX (default: 30)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results, cej_table = run_benchmarks(timeout_s=args.timeout, nx_timeout_s=args.nx_timeout)
        save_results(results, cej_table)


if __name__ == "__main__":
    main()
