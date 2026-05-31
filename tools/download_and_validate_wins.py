#!/usr/bin/env python3
"""Download every PoW winning solution from a quip node and re-validate it.

Walks the proof chain backward from ``QuantumPow.LastProofBlock`` via each
``WinningSolution.last_proof_block_hash`` (proofs are sparse, so block-by-block
scanning is wasteful), and for every winning block:

  1. Reads the persisted ``WinningSolution`` (miner, salt, claimed energy,
     active difficulty, last-proof hash) plus the runtime-derived nonce via
     ``QuantumPowApi_winning_solution``.
  2. Decodes the ``QuantumPow.submit_proof`` extrinsic in that block to recover
     the *actual submitted spins* (the on-chain win record does not store them).
  3. Reconstructs the topology + allowed-value specs the proof mined against via
     ``QuantumPowApi_mining_snapshot`` (cached per ``topology_hash``).
  4. Re-derives the Ising model from the nonce, recomputes every submitted
     solution's energy, and independently checks the proof clears its
     difficulty: nonce matches, recomputed best energy matches the claim,
     ``num_valid >= min_solutions``, and ``diversity >= min_diversity``.

Two artifacts are written:

  - ``<out>.wins.jsonl``       — the raw downloaded wins (archive, includes
                                  the packed solution hex).
  - ``<out>.validation.jsonl`` — one verdict record per win.

and a summary is printed to stderr.

Usage::

    python tools/download_and_validate_wins.py \
        --url wss://qpu-1.nodes.quip.network/rpc \
        --out qpu1_wins
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Imports below follow the sys.path bootstrap above (repo not installed).
from shared.packed_solution import unpack_solution  # noqa: E402
from shared.quantum_proof_of_work import (  # noqa: E402
    calculate_diversity,
    energy_of_solution,
    generate_ising_model_from_nonce,
)
from substrate.client import SubstrateClient  # noqa: E402
from substrate.types import (  # noqa: E402
    SubstrateMiningContext,
    WinningSolutionWithNonce,
)

_ZERO_HASH = b"\x00" * 32
# Energy is stored in milli-precision integers; the recomputed float energy is
# exact for the integer h/J spec, so allow one milli of float-rounding slack.
_ENERGY_MATCH_TOL_MILLI = 1


def _hx(s: str) -> bytes:
    """Decode a ``0x``-prefixed (or bare) hex string to bytes."""
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def _spin(milli_value: int) -> int:
    """Map a milli-precision spin value to a canonical Ising spin (-1/+1)."""
    return 1 if milli_value > 0 else -1


async def _decode_submit_proof(
    client: SubstrateClient, block_hash_hex: str, win_nonce: bytes
) -> Optional[Dict[str, Any]]:
    """Decode the *winning* ``QuantumPow.submit_proof`` call in a block.

    A winning block can carry several competing ``submit_proof`` extrinsics —
    the chain keeps the lowest-energy one. The winner is identified by the
    nonce the runtime recorded (``win_nonce``), so this returns the submission
    whose nonce matches, ignoring losing competitors in the same block.

    Returns a dict with ``topology_hash`` (bytes), ``nonce`` (32-byte
    big-endian), and ``solutions`` (list of packed ``bytes``), or ``None`` if
    no matching submission is present.
    """
    block = await client._run(
        lambda: client._iface.get_block(block_hash=block_hash_hex)
    )
    for extrinsic in block["extrinsics"]:
        call = extrinsic.value["call"]
        if call["call_module"] != "QuantumPow":
            continue
        if call["call_function"] != "submit_proof":
            continue
        proof = {a["name"]: a["value"] for a in call["call_args"]}["proof"]
        nonce = int(proof["nonce"]).to_bytes(32, "big")
        if nonce != win_nonce:
            continue
        return {
            "topology_hash": _hx(proof["topology_hash"]),
            "nonce": nonce,
            "solutions": [_hx(s) for s in proof["solutions"]],
        }
    return None


async def _topology_for(
    client: SubstrateClient,
    topology_hash: bytes,
    miner: bytes,
    at_block_hash: bytes,
    cache: Dict[bytes, SubstrateMiningContext],
) -> SubstrateMiningContext:
    """Fetch (and cache) the mining snapshot for ``topology_hash``."""
    cached = cache.get(topology_hash)
    if cached is not None:
        return cached
    snapshot = await client.get_mining_snapshot(
        miner_account_bytes=miner, at=at_block_hash, topology_hash=topology_hash
    )
    if snapshot is None:
        raise RuntimeError(
            f"no mining snapshot for topology 0x{topology_hash.hex()}"
        )
    cache[topology_hash] = snapshot
    return snapshot


def _validate(
    ws: WinningSolutionWithNonce,
    proof: Dict[str, Any],
    snapshot: SubstrateMiningContext,
) -> Dict[str, Any]:
    """Re-derive the Ising model and check the submitted spins clear the bar.

    Returns a verdict record with the individual check results and an overall
    ``valid`` boolean.
    """
    sol = ws.solution
    diff = sol.difficulty
    num_spins = len(snapshot.nodes)
    spins = [
        [_spin(v) for v in unpack_solution(packed, num_spins, snapshot.allowed_spin_values)]
        for packed in proof["solutions"]
    ]
    h, j = generate_ising_model_from_nonce(
        ws.nonce,
        snapshot.nodes,
        snapshot.edges,
        snapshot.allowed_h_values,
        snapshot.allowed_j_values,
    )
    energies = [energy_of_solution(s, h, j, snapshot.nodes) for s in spins]
    valid_spins = [s for s, e in zip(spins, energies) if e <= diff.max_energy]
    diversity = calculate_diversity(valid_spins) if len(valid_spins) >= 2 else 0.0
    best_milli = round(min(energies) * 1000) if energies else 0

    checks = {
        "nonce_match": proof["nonce"] == ws.nonce,
        "energy_match": abs(best_milli - sol.energy_milli) <= _ENERGY_MATCH_TOL_MILLI,
        "num_valid_ok": len(valid_spins) >= diff.min_solutions,
        "diversity_ok": diversity >= diff.min_diversity,
    }
    return {
        "miner": "0x" + sol.miner.hex(),
        "claimed_energy_milli": sol.energy_milli,
        "recomputed_best_milli": best_milli,
        "num_solutions": len(spins),
        "num_valid": len(valid_spins),
        "diversity": round(diversity, 6),
        "threshold": {
            "min_solutions": diff.min_solutions,
            "max_energy_milli": diff.max_energy_milli,
            "min_diversity_milli": diff.min_diversity_milli,
        },
        "checks": checks,
        "valid": all(checks.values()),
    }


async def _validate_one(
    client: SubstrateClient,
    block_number: int,
    topo_cache: Dict[bytes, SubstrateMiningContext],
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    """Download + validate the win at ``block_number``.

    Returns ``(archive_record, verdict_record, prev_proof_hash)``. The verdict
    carries an ``error`` field instead of checks when the proof can't be
    reconstructed (e.g. missing extrinsic on a pruned node).
    """
    ws = await client.query_winning_solution(block_number)
    if ws is None:
        raise RuntimeError(f"block {block_number}: no winning solution recorded")
    sol = ws.solution
    block_hash_hex = await client._run(
        lambda: client._iface.get_block_hash(block_number)
    )
    proof = await _decode_submit_proof(client, block_hash_hex, ws.nonce)
    archive = {
        "block_number": block_number,
        "miner": "0x" + sol.miner.hex(),
        "salt": "0x" + sol.salt.hex(),
        "nonce": "0x" + ws.nonce.hex(),
        "energy_milli": sol.energy_milli,
        "reward": sol.reward,
        "submitted_at": sol.submitted_at,
        "last_proof_block_hash": "0x" + sol.last_proof_block_hash.hex(),
        "difficulty": {
            "min_solutions": sol.difficulty.min_solutions,
            "max_energy_milli": sol.difficulty.max_energy_milli,
            "min_diversity_milli": sol.difficulty.min_diversity_milli,
        },
        "topology_hash": "0x" + proof["topology_hash"].hex() if proof else None,
        "solutions_hex": ["0x" + s.hex() for s in proof["solutions"]] if proof else [],
    }
    if proof is None:
        verdict = {"block_number": block_number, "valid": False,
                   "error": "no submit_proof extrinsic matching the winning nonce"}
        return archive, verdict, sol.last_proof_block_hash

    snapshot = await _topology_for(
        client, proof["topology_hash"], sol.miner, _hx(block_hash_hex), topo_cache
    )
    verdict = {"block_number": block_number, **_validate(ws, proof, snapshot)}
    return archive, verdict, sol.last_proof_block_hash


async def walk_and_validate(
    url: str, max_wins: Optional[int], out_prefix: str
) -> Dict[str, Any]:
    """Walk the proof chain backward, archiving and validating every win."""
    wins_path = Path(f"{out_prefix}.wins.jsonl")
    verdicts_path = Path(f"{out_prefix}.validation.jsonl")
    topo_cache: Dict[bytes, SubstrateMiningContext] = {}
    seen: set[int] = set()
    n_valid = n_invalid = 0
    errors: List[str] = []

    client = SubstrateClient(url=url)
    await client.connect()
    try:
        cur: Optional[int] = await client.query_last_proof_block_number()
        if not cur:
            return {"count": 0, "error": "LastProofBlock is 0 — no proofs yet"}

        with wins_path.open("w") as wf, verdicts_path.open("w") as vf:
            while cur is not None and cur > 0 and cur not in seen:
                if max_wins is not None and len(seen) >= max_wins:
                    break
                seen.add(cur)
                try:
                    archive, verdict, prev_hash = await _validate_one(
                        client, cur, topo_cache
                    )
                except Exception as exc:  # noqa: BLE001 — record & stop the walk
                    errors.append(f"{cur}: {type(exc).__name__}: {exc}")
                    break
                wf.write(json.dumps(archive) + "\n")
                vf.write(json.dumps(verdict) + "\n")
                if verdict.get("valid"):
                    n_valid += 1
                else:
                    n_invalid += 1
                    errors.append(
                        f"block {cur}: INVALID "
                        f"{verdict.get('checks') or verdict.get('error')}"
                    )
                mark = "ok " if verdict.get("valid") else "BAD"
                print(
                    f"[{mark}] block {cur} miner={archive['miner'][:12]} "
                    f"E={archive['energy_milli']/1000:.2f} "
                    f"valid={verdict.get('num_valid')}/{verdict.get('num_solutions')}",
                    file=sys.stderr,
                )
                if prev_hash == _ZERO_HASH:
                    break
                try:
                    cur = await client.get_block_number(at=prev_hash)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"resolve prev proof hash: {type(exc).__name__}: {exc}")
                    break
    finally:
        await client.close()

    return {
        "url": url,
        "count": n_valid + n_invalid,
        "valid": n_valid,
        "invalid": n_invalid,
        "errors": errors[:20],
        "wins_file": str(wins_path),
        "validation_file": str(verdicts_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--url", default="wss://qpu-1.nodes.quip.network/rpc",
        help="substrate RPC URL (ws/wss/https all work via /rpc)",
    )
    p.add_argument(
        "--max", type=int, default=None,
        help="cap the number of wins to download (default: all, back to genesis)",
    )
    p.add_argument(
        "--out", default="quip_wins",
        help="output path prefix; writes <out>.wins.jsonl and "
             "<out>.validation.jsonl (default: quip_wins)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    summary = asyncio.run(walk_and_validate(args.url, args.max, args.out))
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if summary.get("error"):
        return 1
    print(
        f"[done] {summary['count']} wins: {summary['valid']} valid, "
        f"{summary['invalid']} invalid → {summary['validation_file']}",
        file=sys.stderr,
    )
    return 0 if summary.get("invalid", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
