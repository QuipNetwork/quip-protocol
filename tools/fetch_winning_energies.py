#!/usr/bin/env python3
"""Pull recent winning-solution energies from a quip chain node.

Walks back N blocks from the chain head, calls
``QuantumPowApi_winning_solution`` for each, and reports the energy
distribution. The output is meant to feed
``tools/qpu_throughput_canary.py --mode sweep --energy-threshold``
with a *realistic* threshold (mean / median of recent wins) instead of
the sweep-internal median — the latter under-rates higher-num_reads
configs because their advantage shows up in the deep tail of the
energy distribution, not in the median.

Usage::

    python tools/fetch_winning_energies.py \
        --ws-url wss://qpu-1.nodes.quip.network:9944 \
        --n-blocks 200

Reports count, mean, median, min, max, and several percentiles. The
energy values are emitted in the float-energy convention the canary
uses (``energy_milli / 1000``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from substrate.client import SubstrateClient


_ZERO_HASH = b"\x00" * 32


async def fetch_energies(
    ws_url: str,
    n_wanted: int,
) -> dict:
    """Walk the proof chain backward and collect ``n_wanted`` energies.

    Proofs land sparsely — only on blocks that included a successful
    ``submit_proof`` extrinsic, which is roughly once per difficulty
    period rather than once per block. Linear iteration over recent
    block heights returns ``None`` for the overwhelming majority. So we
    chain-walk via ``WinningSolution.last_proof_block_hash`` instead:

      1. Read ``QuantumPow.LastProofBlock`` → most recent winning block
         number.
      2. ``query_winning_solution(N)`` at that height returns the
         solution + the hash of the *previous* winning block.
      3. Resolve that hash to its block number; repeat.

    Returns when ``n_wanted`` proofs have been collected, the chain
    walks back to genesis, or a node-pruning gap aborts the walk.
    """
    client = SubstrateClient(url=ws_url)
    await client.connect()
    try:
        head = await client.get_head()
        head_number = await client.get_block_number(head)
        last_proof_block_number = await client.query_last_proof_block_number()
        if last_proof_block_number == 0:
            return {
                "ws_url": ws_url,
                "head_number": head_number,
                "last_proof_block": 0,
                "n_with_winner": 0,
                "errors_sample": [
                    "LastProofBlock is 0 — chain has no recorded proofs yet"
                ],
                "summary": {"count": 0},
                "energies": [],
            }

        energies_milli: List[int] = []
        block_numbers: List[int] = []
        errors: List[str] = []
        cur_number: Optional[int] = last_proof_block_number
        seen: set = set()

        while cur_number is not None and len(energies_milli) < n_wanted:
            if cur_number <= 0 or cur_number in seen:
                break
            seen.add(cur_number)
            try:
                ws = await client.query_winning_solution(cur_number)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{cur_number}: query_winning_solution: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            if ws is None:
                errors.append(
                    f"{cur_number}: None — proof chain broken or pruned"
                )
                break
            energies_milli.append(ws.solution.energy_milli)
            block_numbers.append(cur_number)
            prev_hash = ws.solution.last_proof_block_hash
            if prev_hash == _ZERO_HASH:
                break
            try:
                cur_number = await client.get_block_number(at=prev_hash)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"resolve hash for prior proof: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

        energies = [m / 1000.0 for m in energies_milli]
        if not energies:
            summary: dict = {"count": 0}
        else:
            quantiles = (
                statistics.quantiles(energies, n=10, method="inclusive")
                if len(energies) >= 2 else []
            )
            summary = {
                "count": len(energies),
                "mean": statistics.fmean(energies),
                "median": statistics.median(energies),
                "stdev": (
                    statistics.stdev(energies)
                    if len(energies) >= 2 else 0.0
                ),
                "min": min(energies),
                "max": max(energies),
                "p10": quantiles[0] if quantiles else energies[0],
                "p25": quantiles[2] if len(quantiles) >= 3 else energies[0],
                "p75": (
                    quantiles[6] if len(quantiles) >= 7 else energies[-1]
                ),
                "p90": (
                    quantiles[8] if len(quantiles) >= 9 else energies[-1]
                ),
            }

        return {
            "ws_url": ws_url,
            "head_number": head_number,
            "last_proof_block": last_proof_block_number,
            "earliest_proof_block": block_numbers[-1] if block_numbers else None,
            "n_requested": n_wanted,
            "n_with_winner": len(energies),
            "errors_sample": errors[:5],
            "summary": summary,
            "block_numbers": block_numbers,
            "energies": energies,
        }
    finally:
        await client.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--ws-url", required=True,
        help="substrate websocket URL "
             "(e.g. wss://qpu-1.nodes.quip.network:9944)",
    )
    p.add_argument(
        "--n", "--n-blocks", dest="n", type=int, default=200,
        help="how many proof winners to collect by walking the proof "
             "chain backward from LastProofBlock (default 200)",
    )
    p.add_argument(
        "--output", default=None,
        help="write full JSON (including energies list) to this path. "
             "Without it, stdout gets a summary line and JSON summary.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    result = asyncio.run(fetch_energies(args.ws_url, args.n))
    s = result["summary"]
    if s.get("count", 0) == 0:
        print(
            f"[fetch] no winners collected on {args.ws_url} "
            f"(head={result.get('head_number')}, "
            f"LastProofBlock={result.get('last_proof_block')})",
            file=sys.stderr,
        )
        for e in result.get("errors_sample") or []:
            print(f"  err: {e}", file=sys.stderr)
        return 1

    print(
        f"[fetch] {s['count']} winners from blocks "
        f"{result['last_proof_block']}..{result['earliest_proof_block']} "
        f"(head={result['head_number']})",
        file=sys.stderr,
    )
    if result.get("errors_sample"):
        for e in result["errors_sample"]:
            print(f"  walk-stop: {e}", file=sys.stderr)
    print(
        f"[fetch] energy: mean={s['mean']:.2f} median={s['median']:.2f} "
        f"stdev={s['stdev']:.2f} | min={s['min']:.2f} max={s['max']:.2f}",
        file=sys.stderr,
    )
    print(
        f"[fetch] p10={s['p10']:.2f} p25={s['p25']:.2f} "
        f"p75={s['p75']:.2f} p90={s['p90']:.2f}",
        file=sys.stderr,
    )
    print(
        f"[fetch] suggested --energy-threshold for canary sweep: "
        f"{s['mean']:.2f} (or {s['median']:.2f})",
        file=sys.stderr,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"[fetch] wrote {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
