"""Phase 8.2 — Warm up the rainbow table with selected D-Wave headers.

Synthesizes targets that the default benchmark times out on (Cm2 takes
~245s, Z(1,2) ~108s; the benchmark's per-target 60s timeout excludes them).
Adds them to `tutte/data/lookup_table.{json,bin}` so future syntheses of
Z(1,3), Cm3, Pm3 (which decompose into these cells via the Phase 8.1
cost-aware partitioner) get instant cell lookup.

Idempotent: skips targets already in the table.

Usage:
    python -m tutte.scripts.warmup_lookup_table
    python -m tutte.scripts.warmup_lookup_table --target cm2  # single target
    python -m tutte.scripts.warmup_lookup_table --timeout 1800  # custom budget
"""

import argparse
import os
import signal
import sys
import time

import dwave_networkx as dnx

from tutte.graph import Graph
from tutte.lookup.binary import save_binary_rainbow_table
from tutte.lookup.core import load_default_table
from tutte.synthesis.engine import SynthesisEngine
from tutte.validation import verify_spanning_trees


# Phase 8.4 will extend this list with Z(1,3) / Cm3 once those synthesize.
TARGETS = [
    ("Cm2", lambda: dnx.chimera_graph(2)),
    ("Z1_2", lambda: dnx.zephyr_graph(1, 2)),
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
