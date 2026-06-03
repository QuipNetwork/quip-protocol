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

With ``--dump-bqm`` a third artifact is written:

  - ``<out>.bqms.jsonl``       — the reconstructed Ising model (h, J) for each
                                  win, re-derived from its nonce + topology
                                  snapshot. One model per line.

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
import math
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
    SubstrateDifficulty,
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


def _difficulty_dict(d: SubstrateDifficulty) -> Dict[str, int]:
    """Serialize the three on-chain difficulty fields to a plain dict."""
    return {
        "min_solutions": d.min_solutions,
        "max_energy_milli": d.max_energy_milli,
        "min_diversity_milli": d.min_diversity_milli,
    }


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
        "threshold": _difficulty_dict(diff),
        "checks": checks,
        "valid": all(checks.values()),
    }


def _serialize_bqm(
    h: Dict[int, float], j: Dict[Tuple[int, int], float]
) -> Dict[str, Any]:
    """Serialize a reconstructed Ising model (h, J) to a JSON-safe dict.

    ``h`` is keyed by node id and ``J`` by ``(u, v)`` edge tuples. JSON object
    keys must be strings and tuples are not valid keys, so both fields are
    emitted as flat lists that preserve the integer node ids:

      - ``h``: ``[[node_id, bias], ...]``
      - ``j``: ``[[u, v, coupling], ...]``

    Reload with ``h = {n: b for n, b in rec["h"]}`` and
    ``J = {(u, v): c for u, v, c in rec["j"]}`` — exactly the dict shapes
    :func:`shared.quantum_proof_of_work.energy_of_solution` expects.

    Returns the ``h``/``j`` fields for one line of ``<out>.bqms.jsonl``.
    """
    return {
        "h": [[node, bias] for node, bias in h.items()],
        "j": [[u, v, coupling] for (u, v), coupling in j.items()],
    }


async def _dump_bqms(
    client: SubstrateClient,
    view: List[int],
    cache: Dict[int, Dict[str, Any]],
    topo_cache: Dict[bytes, SubstrateMiningContext],
    bqms_path: Path,
    errors: List[str],
) -> int:
    """Reconstruct + write the Ising model for every win in ``view``.

    Snapshots are stable per topology, so each unique ``topology_hash`` is
    fetched once (at chain head) and reused. Returns the number of models
    written; wins with no decoded proof (no ``topology_hash``) are skipped.
    """
    written = 0
    with bqms_path.open("w") as f:
        for bn in view:
            archive = cache[bn]["archive"]
            topo_hex = archive.get("topology_hash")
            if topo_hex is None:
                continue
            topo_hash = _hx(topo_hex)
            try:
                snapshot = topo_cache.get(topo_hash)
                if snapshot is None:
                    snapshot = await client.get_mining_snapshot(
                        miner_account_bytes=_hx(archive["miner"]),
                        topology_hash=topo_hash,
                    )
                    if snapshot is None:
                        raise RuntimeError(f"no snapshot for topology {topo_hex}")
                    topo_cache[topo_hash] = snapshot
                h, j = generate_ising_model_from_nonce(
                    _hx(archive["nonce"]),
                    snapshot.nodes,
                    snapshot.edges,
                    snapshot.allowed_h_values,
                    snapshot.allowed_j_values,
                )
            except Exception as exc:  # noqa: BLE001 — record & skip this win
                errors.append(f"bqm {bn}: {type(exc).__name__}: {exc}")
                continue
            record = {
                "block_number": bn,
                "nonce": archive["nonce"],
                "topology_hash": topo_hex,
                **_serialize_bqm(h, j),
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written


async def _validate_one(
    client: SubstrateClient,
    block_number: int,
    topo_cache: Dict[bytes, SubstrateMiningContext],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Download + validate the win at ``block_number``.

    Returns ``(archive_record, verdict_record)``. The verdict carries an
    ``error`` field instead of checks when the proof can't be reconstructed
    (e.g. missing extrinsic on a pruned node).
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
        "difficulty": _difficulty_dict(sol.difficulty),
        "topology_hash": "0x" + proof["topology_hash"].hex() if proof else None,
        "solutions_hex": ["0x" + s.hex() for s in proof["solutions"]] if proof else [],
    }
    if proof is None:
        verdict = {"block_number": block_number, "valid": False,
                   "error": "no submit_proof extrinsic matching the winning nonce"}
        return archive, verdict

    snapshot = await _topology_for(
        client, proof["topology_hash"], sol.miner, _hx(block_hash_hex), topo_cache
    )
    verdict = {"block_number": block_number, **_validate(ws, proof, snapshot)}
    return archive, verdict


def _load_cache(cache_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load the local win cache keyed by block number.

    Each line is ``{"archive": {...}, "verdict": {...}}``. The proof chain is
    append-only at the head and immutable below it, so cached records never go
    stale — re-runs only fetch wins newer than the cache.
    """
    cache: Dict[int, Dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["archive"]["block_number"]] = rec
    return cache


def _write_records(
    records: Dict[int, Dict[str, Any]], path: Path, view: Optional[List[int]] = None
) -> None:
    """Write cache records (full union, or the ``view`` subset) newest-first."""
    keys = view if view is not None else sorted(records, reverse=True)
    with path.open("w") as f:
        for bn in keys:
            f.write(json.dumps(records[bn]) + "\n")


def _write_view(
    view: List[int], cache: Dict[int, Dict[str, Any]],
    wins_path: Path, verdicts_path: Path,
) -> None:
    """Split this run's walked view into the wins/validation artifacts."""
    with wins_path.open("w") as wf, verdicts_path.open("w") as vf:
        for bn in view:
            rec = cache[bn]
            wf.write(json.dumps(rec["archive"]) + "\n")
            if rec["verdict"] is not None:
                vf.write(json.dumps(rec["verdict"]) + "\n")


async def _next_block(
    client: SubstrateClient, archive: Dict[str, Any], errors: List[str]
) -> Optional[int]:
    """Resolve the previous winning block number, caching it on the archive.

    Uses the stored ``prev_block_number`` when present (network-free re-walks),
    otherwise resolves ``last_proof_block_hash`` once and stamps it.
    """
    prev_hash = _hx(archive["last_proof_block_hash"])
    if prev_hash == _ZERO_HASH:
        return None
    prev_bn = archive.get("prev_block_number")
    if prev_bn is None:
        try:
            prev_bn = await client.get_block_number(at=prev_hash)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"resolve prev proof hash: {type(exc).__name__}: {exc}")
            return None
        archive["prev_block_number"] = prev_bn
    return prev_bn


async def _download_one(
    client: SubstrateClient, block_number: int,
    topo_cache: Dict[bytes, SubstrateMiningContext], errors: List[str],
) -> Optional[Dict[str, Any]]:
    """Download + validate one win, returning a cache record or ``None``."""
    try:
        archive, verdict = await _validate_one(client, block_number, topo_cache)
    except Exception as exc:  # noqa: BLE001 — record & stop the walk
        errors.append(f"{block_number}: {type(exc).__name__}: {exc}")
        return None
    if not verdict.get("valid"):
        errors.append(
            f"block {block_number}: INVALID "
            f"{verdict.get('checks') or verdict.get('error')}"
        )
    mark = "ok " if verdict.get("valid") else "BAD"
    print(
        f"[{mark}] block {block_number} miner={archive['miner'][:12]} "
        f"E={archive['energy_milli']/1000:.2f} "
        f"valid={verdict.get('num_valid')}/{verdict.get('num_solutions')}",
        file=sys.stderr,
    )
    return {"archive": archive, "verdict": verdict}


async def walk_and_validate(
    url: str, max_wins: Optional[int], out_prefix: str, *,
    use_cache: bool = True, dump_bqm: bool = False,
) -> Dict[str, Any]:
    """Walk the proof chain backward, validating every win, reusing the cache."""
    wins_path = Path(f"{out_prefix}.wins.jsonl")
    verdicts_path = Path(f"{out_prefix}.validation.jsonl")
    cache_path = Path(f"{out_prefix}.cache.jsonl")
    # Always load the union so a re-validation run can't shrink the cache;
    # `use_cache` only governs whether walked blocks are *reused* vs re-fetched.
    cache = _load_cache(cache_path)
    topo_cache: Dict[bytes, SubstrateMiningContext] = {}
    seen: set[int] = set()
    view: List[int] = []
    errors: List[str] = []
    n_new = 0
    n_bqms = 0
    bqms_path: Optional[Path] = None

    client = SubstrateClient(url=url)
    await client.connect()
    try:
        cur: Optional[int] = await client.query_last_proof_block_number()
        if not cur:
            return {"count": 0, "error": "LastProofBlock is 0 — no proofs yet"}
        while cur is not None and cur > 0 and cur not in seen:
            if max_wins is not None and len(view) >= max_wins:
                break
            seen.add(cur)
            rec = cache.get(cur) if use_cache else None
            if rec is None:
                rec = await _download_one(client, cur, topo_cache, errors)
                if rec is None:
                    break
                cache[cur] = rec
                n_new += 1
            view.append(cur)
            cur = await _next_block(client, rec["archive"], errors)
        if dump_bqm:
            bqms_path = Path(f"{out_prefix}.bqms.jsonl")
            n_bqms = await _dump_bqms(
                client, view, cache, topo_cache, bqms_path, errors
            )
    finally:
        await client.close()

    _write_records(cache, cache_path)
    _write_view(view, cache, wins_path, verdicts_path)
    recs = [cache[bn] for bn in view]
    valid = [r for r in recs if r["verdict"] and r["verdict"].get("valid")]
    n_invalid = sum(1 for r in recs if r["verdict"] and not r["verdict"].get("valid"))
    return {
        "url": url,
        "count": len(view),
        "valid": len(valid),
        "invalid": n_invalid,
        "new_downloaded": n_new,
        "reused_from_cache": len(view) - n_new,
        "cache_total": len(cache),
        "errors": errors[:20],
        "wins_file": str(wins_path),
        "validation_file": str(verdicts_path),
        "cache_file": str(cache_path),
        "bqms_file": str(bqms_path) if bqms_path else None,
        "bqms_written": n_bqms,
        "win_energies_milli": [r["archive"]["energy_milli"] for r in valid],
    }


# ----------------------------------------------------------------------
# Time-to-solution analysis (couples chain win energies to the QPU-TTS model)
# ----------------------------------------------------------------------

# QPU-TTS from the model is QPU-*access* time; wall-clock = access / fraction.
# 0.20 is the model's good-case effective_qpu_fraction; 0.04 is the contended
# qpu-1 observation (qpu_access ~61ms within a ~1.5s/attempt wall).
_QPU_FRACTION_GOOD = 0.20
_QPU_FRACTION_CONTENDED = 0.04
_DEFAULT_MODEL_ROOT = _REPO_ROOT / "test_results" / "qpu_tts_test"


def _format_duration(seconds: float) -> str:
    """Human-readable duration with adaptive units."""
    if seconds != seconds or seconds == float("inf"):  # NaN or inf
        return "inf"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _load_tts_predictor(model_root: Path) -> Tuple[Any, Dict[str, Any]]:
    """Import the gitignored QPU-TTS model and load its fit parameters.

    Returns ``(predict_fn, fit)``. Raises ``RuntimeError`` with an actionable
    message if the archive is absent (it is gitignored) or its scientific
    deps (pandas/scipy) are missing from the running interpreter.
    """
    import importlib.util

    module_path = model_root / "tools" / "qpu_tts_model.py"
    if not module_path.exists():
        raise RuntimeError(
            f"TTS model not found at {module_path} — the qpu_tts_test archive "
            "is gitignored; point --tts-model-root at a local copy"
        )
    spec = importlib.util.spec_from_file_location("qpu_tts_model", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ImportError as exc:
        raise RuntimeError(
            f"TTS model import failed ({exc}); run under an interpreter with "
            "pandas+scipy (e.g. .quip/bin/python)"
        ) from exc
    fit_path = model_root / "model" / "fit_params.json"
    fit = (
        json.loads(fit_path.read_text())
        if fit_path.exists()
        else module.build_fit(model_root)
    )
    return module.predict, fit


def analyze_tts(
    energies_milli: List[int],
    model_root: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate QPU mining time to reach each chain win-energy percentile.

    ``config`` carries ``reads``, ``anneal``, ``k``, ``diversity``,
    ``percentiles`` (best-first, e.g. ``[50, 25, 10, 5, 1]`` — p1 == top 1%),
    and ``confidence`` (e.g. 0.95). Returns a record with one row per
    percentile: target energy, model ``p_success``/``qpu_tts_s`` (mean access
    time), the confidence-adjusted time, and wall-clock bounds.
    """
    import numpy as np

    if not energies_milli:
        return {"error": "no valid wins to analyze"}
    predict, fit = _load_tts_predictor(model_root)
    energies = np.array(energies_milli, dtype=float) / 1000.0
    # Memoryless wins: P(>=1 by t) = 1 - e^{-t/TTS}; reach `confidence` at
    # t = -ln(1 - confidence) * TTS_mean.
    conf_mult = -math.log(1.0 - config["confidence"])

    rows: List[Dict[str, Any]] = []
    for pct in config["percentiles"]:
        energy = float(np.percentile(energies, pct))  # lower pct = better energy
        pred = predict(
            fit, energy, config["k"], config["diversity"],
            config["reads"], config["anneal"],
        )
        tts = pred["qpu_tts_s"]
        tts_conf = tts * conf_mult
        rows.append({
            "percentile": pct,
            "energy": round(energy, 1),
            "p_success": pred["p_success"],
            "qpu_tts_mean_s": tts,
            "qpu_tts_conf_s": tts_conf,
            "wall_good_s": tts_conf / _QPU_FRACTION_GOOD,
            "wall_contended_s": tts_conf / _QPU_FRACTION_CONTENDED,
            "binding_gate": pred["binding_gate"],
            "extrapolated": bool(pred["notes"]),
        })
    return {
        "n_wins": len(energies_milli),
        "config": config,
        "energy_percentiles": rows,
    }


def _print_tts_table(result: Dict[str, Any]) -> None:
    """Render the TTS analysis as a table on stderr."""
    if result.get("error"):
        print(f"[tts] {result['error']}", file=sys.stderr)
        return
    cfg = result["config"]
    print(
        f"\n[tts] QPU mining time to reach each win-energy percentile "
        f"(n={result['n_wins']} wins; {cfg['reads']}x{cfg['anneal']}us, "
        f"K={cfg['k']}, D={cfg['diversity']}, "
        f"{cfg['confidence']:.0%} confidence)",
        file=sys.stderr,
    )
    header = (
        f"  {'pct':>4} {'energy':>9} {'p_success':>11} "
        f"{'QPU-TTS':>9} {'conf':>9} {'wall(good)':>11} {'wall(busy)':>11}"
    )
    print(header, file=sys.stderr)
    for r in result["energy_percentiles"]:
        flag = " *extrap" if r["extrapolated"] else ""
        print(
            f"  {('p%g' % r['percentile']):>4} {r['energy']:>9.1f} "
            f"{r['p_success']:>11.2e} {_format_duration(r['qpu_tts_mean_s']):>9} "
            f"{_format_duration(r['qpu_tts_conf_s']):>9} "
            f"{_format_duration(r['wall_good_s']):>11} "
            f"{_format_duration(r['wall_contended_s']):>11}{flag}",
            file=sys.stderr,
        )
    print(
        "  (QPU-TTS = mean QPU-access time/win; conf = time for the given "
        "confidence; wall = access / effective_qpu_fraction "
        f"[{_QPU_FRACTION_GOOD} good .. {_QPU_FRACTION_CONTENDED} contended]; "
        "*extrap = beyond the model's calibrated envelope)",
        file=sys.stderr,
    )


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
        help="output path prefix; writes <out>.wins.jsonl, "
             "<out>.validation.jsonl, and <out>.cache.jsonl (default: quip_wins)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="re-download every walked win instead of reusing cached records "
             "(the cache is still updated, not discarded)",
    )
    p.add_argument(
        "--dump-bqm", action="store_true",
        help="also write <out>.bqms.jsonl: the reconstructed Ising model "
             "(h, J) for each walked win, re-derived from its nonce",
    )
    g = p.add_argument_group("time-to-solution analysis (--tts)")
    g.add_argument(
        "--tts", action="store_true",
        help="after the walk, map win-energy percentiles to QPU mining time "
             "via the qpu_tts_test model (needs pandas+scipy; run under "
             ".quip/bin/python)",
    )
    g.add_argument(
        "--tts-model-root", default=str(_DEFAULT_MODEL_ROOT),
        help=f"path to the qpu_tts_test archive (default: {_DEFAULT_MODEL_ROOT})",
    )
    g.add_argument(
        "--tts-percentiles", default="50,25,10,5,1",
        help="best-first energy percentiles to model; p1 == top 1%% "
             "(default: 50,25,10,5,1)",
    )
    g.add_argument("--tts-reads", type=int, default=112, help="num_reads (default 112)")
    g.add_argument("--tts-anneal", type=float, default=80.0,
                   help="annealing_time_us (default 80)")
    g.add_argument("--tts-k", type=int, default=5,
                   help="required below-threshold count (default 5)")
    g.add_argument("--tts-diversity", type=float, default=0.2,
                   help="required diversity gate (default 0.2)")
    g.add_argument("--tts-confidence", type=float, default=0.95,
                   help="confidence level for the 'ensure a win' time (default 0.95)")
    return p


def _tts_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble the TTS analysis config from parsed args."""
    return {
        "reads": args.tts_reads,
        "anneal": args.tts_anneal,
        "k": args.tts_k,
        "diversity": args.tts_diversity,
        "confidence": args.tts_confidence,
        "percentiles": [float(x) for x in args.tts_percentiles.split(",") if x.strip()],
    }


def main() -> int:
    args = _build_parser().parse_args()
    summary = asyncio.run(
        walk_and_validate(
            args.url, args.max, args.out,
            use_cache=not args.no_cache, dump_bqm=args.dump_bqm,
        )
    )
    energies = summary.pop("win_energies_milli", [])
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if summary.get("error"):
        return 1
    print(
        f"[done] {summary['count']} wins in view: {summary['valid']} valid, "
        f"{summary['invalid']} invalid "
        f"({summary['new_downloaded']} new, {summary['reused_from_cache']} cached; "
        f"cache holds {summary['cache_total']}) → {summary['validation_file']}",
        file=sys.stderr,
    )
    if summary.get("bqms_file"):
        print(
            f"[bqm] {summary['bqms_written']} models → {summary['bqms_file']}",
            file=sys.stderr,
        )
    if args.tts:
        try:
            tts = analyze_tts(energies, Path(args.tts_model_root), _tts_config(args))
        except RuntimeError as exc:
            print(f"[tts] skipped: {exc}", file=sys.stderr)
        else:
            _print_tts_table(tts)
            Path(f"{args.out}.tts.json").write_text(json.dumps(tts, indent=2))
            print(f"[tts] wrote {args.out}.tts.json", file=sys.stderr)
    return 0 if summary.get("invalid", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
