"""Smoke test: SA on GPU 0, Gibbs on GPU 1.

Validates the streaming API refactor end-to-end on real hardware.
Both miners run independently to completion, then results are
compared side by side.

Run:
  python tests/smoke_node_sa_gibbs.py

Requires: CuPy with CUDA available and at least 2 GPU devices.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from queue import Empty

from shared.block import (
    MinerInfo,
    create_genesis_block,
)
from shared.block_signer import BlockSigner
from shared.logging_config import setup_logging, setup_multiprocess_logging
from shared.miner_worker import MinerHandle

TARGET_ENERGY = -14880.0
TIMEOUT = 900  # 15 minutes max per miner


def run():
    setup_logging(log_level="INFO")
    log_queue, log_listener = setup_multiprocess_logging()

    genesis = create_genesis_block({
        "header": {
            "previous_hash": "00" * 32,
            "index": 0,
            "timestamp": 0,
            "data_hash": "00" * 32,
        },
        "data": "SA vs Gibbs smoke test",
        "next_block_requirements": {
            "difficulty_energy": TARGET_ENERGY,
            "min_diversity": 0.15,
            "min_solutions": 5,
            "timeout_to_difficulty_adjustment_decay": 600,
            "h_values": [-1.0, 0.0, 1.0],
        },
        "quantum_proof": None,
        "miner_info": None,
    })

    requirements = genesis.next_block_requirements
    signer = BlockSigner(seed=os.urandom(32))
    node_info = MinerInfo(
        miner_id="smoke-test",
        miner_type="smoke",
        reward_address=signer.ecdsa_public_key_bytes,
        ecdsa_public_key=signer.ecdsa_public_key_bytes,
        wots_public_key=signer.wots_plus_public_key,
        next_wots_public_key=signer.wots_plus_public_key,
    )

    sa_spec = {
        "id": "SA-GPU-0",
        "kind": "cuda",
        "cfg": {},
        "args": {"device": "0"},
    }
    gibbs_spec = {
        "id": "Gibbs-GPU-1",
        "kind": "cuda-gibbs",
        "cfg": {},
        "args": {"device": "1"},
    }

    handles = {
        "SA": MinerHandle(sa_spec, log_queue),
        "Gibbs": MinerHandle(gibbs_spec, log_queue),
    }

    print(
        f"Mining with SA (GPU 0) and Gibbs (GPU 1), "
        f"target energy: {TARGET_ENERGY}"
    )
    print("-" * 60)

    prev_timestamp = int(time.time())
    for h in handles.values():
        h.mine(genesis, node_info, requirements, prev_timestamp)

    # Poll both until each returns a result or times out
    results: dict[str, object] = {}
    deadline = time.monotonic() + TIMEOUT

    while len(results) < len(handles):
        if time.monotonic() > deadline:
            break
        for label, h in handles.items():
            if label in results:
                continue
            try:
                msg = h.resp.get_nowait()
            except Empty:
                continue
            if hasattr(msg, 'miner_id') and hasattr(msg, 'energy'):
                results[label] = msg
                print(
                    f"\n[{label}] DONE: energy={msg.energy:.1f}, "
                    f"solutions={msg.num_valid}, "
                    f"diversity={msg.diversity:.3f}, "
                    f"time={msg.mining_time:.1f}s"
                )
        time.sleep(0.2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for label in ["SA", "Gibbs"]:
        r = results.get(label)
        if r is None:
            print(f"  {label:8s}: TIMEOUT (no valid block in {TIMEOUT}s)")
        else:
            met = r.energy <= TARGET_ENERGY
            print(
                f"  {label:8s}: energy={r.energy:.1f}  "
                f"solutions={r.num_valid}  "
                f"diversity={r.diversity:.3f}  "
                f"time={r.mining_time:.1f}s  "
                f"{'PASS' if met else 'FAIL'}"
            )

    if len(results) == 2:
        sa_r, gb_r = results["SA"], results["Gibbs"]
        faster = "SA" if sa_r.mining_time < gb_r.mining_time else "Gibbs"
        delta = abs(sa_r.mining_time - gb_r.mining_time)
        print(f"\n  Winner: {faster} (by {delta:.1f}s)")
        print(
            f"  Energy delta: "
            f"{abs(sa_r.energy - gb_r.energy):.1f} "
            f"(SA={sa_r.energy:.1f}, Gibbs={gb_r.energy:.1f})"
        )

    print("=" * 60)

    # Cleanup
    for h in handles.values():
        h.close()
    log_listener.stop()

    ok = all(
        r.energy <= TARGET_ENERGY
        for r in results.values()
    )
    if not ok or len(results) < len(handles):
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
