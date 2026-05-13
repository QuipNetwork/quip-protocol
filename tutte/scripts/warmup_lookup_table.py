"""Warm up the rainbow table with atom polynomials + selected D-Wave headers.

Two value props:

1. **Cell lookup for D-Wave decompositions** (Phase 8.2 motivation): Cm2,
   Z(1,2) precomputed so Z(1,3), Cm3, Pm3 syntheses get instant cell hits
   via the cost-aware partitioner.

2. **Atom polynomials for algebraic factorization** (May 2026): the
   `AlgebraicSynthesisEngine` uses `RainbowTable.find_factors_of(target)`
   to find table entries whose polynomial divides the target — but this
   only works if candidate atoms are IN the table. Adding K_8..K_15,
   K_{a,b} for small (a, b), and other simple cogragh atoms is cheap
   (cotree_dp synthesizes them in <1s each) and broadens the algebraic
   engine's effective coverage.

Idempotent: skips targets already in the table.

Usage:
    python -m tutte.scripts.warmup_lookup_table
    python -m tutte.scripts.warmup_lookup_table --target K_10
    python -m tutte.scripts.warmup_lookup_table --timeout 1800
"""

import argparse
import os
import signal
import sys
import time

import dwave_networkx as dnx
import networkx as nx
from tutte.graph import Graph
from tutte.lookup.binary import save_binary_rainbow_table
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.validation import verify_spanning_trees

TARGETS = [
    # D-Wave headers (Phase 8.2) — cell lookups for larger D-Wave decompositions.
    ("Z1_2", lambda: dnx.zephyr_graph(1, 2)),
    ("Cm2", lambda: dnx.chimera_graph(2)),
    ("Pm2", lambda: dnx.pegasus_graph(2)),

    # Cograph atom polynomials (May 2026) — fast via cotree_dp (<1s each)
    # and ESSENTIAL for AlgebraicSynthesisEngine.find_factors_of(target),
    # which can only find factors that are atoms in the rainbow table.
    ("K_8",  lambda: nx.complete_graph(8)),
    ("K_9",  lambda: nx.complete_graph(9)),
    ("K_10", lambda: nx.complete_graph(10)),
    ("K_11", lambda: nx.complete_graph(11)),
    ("K_12", lambda: nx.complete_graph(12)),
    ("K_13", lambda: nx.complete_graph(13)),
    ("K_14", lambda: nx.complete_graph(14)),
    ("K_15", lambda: nx.complete_graph(15)),
    # Bipartite cograph atoms — common as D-Wave Chimera cell + as algebraic factors.
    ("K_2_3", lambda: nx.complete_bipartite_graph(2, 3)),
    ("K_2_4", lambda: nx.complete_bipartite_graph(2, 4)),
    ("K_3_3", lambda: nx.complete_bipartite_graph(3, 3)),
    ("K_3_4", lambda: nx.complete_bipartite_graph(3, 4)),
    ("K_4_4", lambda: nx.complete_bipartite_graph(4, 4)),  # = Cm1
    ("K_4_5", lambda: nx.complete_bipartite_graph(4, 5)),
    ("K_5_5", lambda: nx.complete_bipartite_graph(5, 5)),
    ("K_5_6", lambda: nx.complete_bipartite_graph(5, 6)),
    ("K_6_6", lambda: nx.complete_bipartite_graph(6, 6)),

    # Still NOT default (intractable / too slow):
    #   ("Z2_1", lambda: dnx.zephyr_graph(2, 1)), # 40n 114e — saw 3.5 GB / >5 min before kill
    #   ("Z1_3", lambda: dnx.zephyr_graph(1, 3)), # 36n 162e — known intractable per Phase 8.3
    #   ("Cm3", lambda: dnx.chimera_graph(3)),    # 72n 192e — known intractable per 18.E.3.l
]


def _data_paths() -> tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "data"))
    return (
        os.path.join(base, "lookup_table.json"),
        os.path.join(base, "lookup_table.bin"),
    )


def _existing_names(table) -> set[str]:
    return {e.name for e in table.entries.values()}


class _Timeout(BaseException):
    """Raised when a per-target synthesis exceeds its time budget.

    Inherits from BaseException so internal `except Exception` handlers in
    networkx / numpy / sympy don't swallow it.
    """


def _alarm_handler(signum, frame):
    raise _Timeout()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", default="all",
        choices=["all"] + [name for name, _ in TARGETS],
        help="Which target(s) to add to the table.",
    )
    p.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-target wall-clock budget in seconds (default 1800 = 30 min).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Synthesize but don't write the updated table to disk.",
    )
    args = p.parse_args()

    table = load_default_table()
    json_path, bin_path = _data_paths()
    print(f"Loaded {len(table.entries)} entries from {json_path}", file=sys.stderr)

    targets = TARGETS if args.target == "all" else [
        (n, b) for n, b in TARGETS if n == args.target
    ]

    existing = _existing_names(table)
    added = 0
    for name, builder in targets:
        if name in existing:
            print(f"  {name}: already in table — skip", file=sys.stderr)
            continue

        # Check canonical_key collision BEFORE synthesis (e.g., K_4_4 has
        # the same canonical_key as the existing Cm1 entry — adding it
        # would overwrite Cm1, breaking gates that key on cell name).
        try:
            g_check = Graph.from_networkx(builder())
        except Exception as e:
            print(f"  {name}: builder failed: {e}", file=sys.stderr)
            continue
        check_key = g_check.canonical_key()
        if check_key in table.entries:
            existing_entry = table.entries[check_key]
            print(
                f"  {name}: canonical_key collision with existing entry "
                f"'{existing_entry.name}' — skip (same polynomial, different name)",
                file=sys.stderr,
            )
            continue

        print(f"  {name}: synthesizing (budget {args.timeout}s)...", file=sys.stderr,
              flush=True)
        g = Graph.from_networkx(builder())
        engine = SynthesisEngine(table=table, verbose=False)
        engine.skip_target_lookup = True

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(args.timeout)
        t0 = time.perf_counter()
        try:
            result = engine.synthesize(g)
        except _Timeout:
            wall = time.perf_counter() - t0
            print(
                f"  {name}: TIMEOUT after {wall:.0f}s (budget {args.timeout}s) — skip",
                file=sys.stderr,
            )
            continue
        except Exception as e:
            print(f"  {name}: synthesis FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        wall = time.perf_counter() - t0

        if not verify_spanning_trees(g, result.polynomial):
            print(
                f"  {name}: synthesized in {wall:.0f}s ({result.method}) but "
                f"FAILED Kirchhoff verification — NOT adding to table",
                file=sys.stderr,
            )
            continue

        table.add(g, name, result.polynomial, result.minors_used)
        added += 1
        print(
            f"  {name}: added in {wall:.0f}s, method={result.method}, "
            f"ST={int(result.polynomial.num_spanning_trees())}",
            file=sys.stderr,
        )

    if added == 0:
        print("\nNo new entries added.", file=sys.stderr)
        return 0

    if args.dry_run:
        print(
            f"\n--dry-run: would write {len(table.entries)} entries to "
            f"{json_path} and {bin_path}",
            file=sys.stderr,
        )
        return 0

    table.resort()
    table.save(json_path)
    bin_size = save_binary_rainbow_table(table, bin_path)
    print(
        f"\nSaved {len(table.entries)} entries — {json_path} and "
        f"{bin_path} ({bin_size} bytes).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
