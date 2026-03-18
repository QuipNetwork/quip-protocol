"""Smoke test: gpu_utilization, yielding, sms_per_nonce.

Parameterized end-to-end test of the GPU mining pipeline with
different utilization/yielding configurations. Uses a relaxed
energy target (-14700) for faster iteration.

Run:
  python tests/smoke_utilization_yielding.py

Requires: CuPy with CUDA, at least 1 GPU device.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from queue import Empty
from typing import Any, Dict, List

from shared.block import MinerInfo, create_genesis_block
from shared.block_signer import BlockSigner
from shared.logging_config import (
    setup_logging,
    setup_multiprocess_logging,
)
from shared.miner_worker import MinerHandle

TARGET_ENERGY = -14700.0
TIMEOUT = 300  # 5 minutes max per miner

# Each scenario: (label, kind, device, cfg overrides)
SCENARIOS: List[Dict[str, Any]] = [
    {
        "label": "SA default (100%)",
        "kind": "cuda",
        "device": "0",
        "cfg": {},
    },
    {
        "label": "SA util=50%",
        "kind": "cuda",
        "device": "0",
        "cfg": {"gpu_utilization": 50},
    },
    {
        "label": "SA yielding+util=80%",
        "kind": "cuda",
        "device": "0",
        "cfg": {"gpu_utilization": 80, "yielding": True},
    },
    {
        "label": "Gibbs default (100%)",
        "kind": "cuda-gibbs",
        "device": "0",
        "cfg": {},
    },
    {
        "label": "Gibbs util=50%",
        "kind": "cuda-gibbs",
        "device": "0",
        "cfg": {"gpu_utilization": 50},
    },
    {
        "label": "Gibbs yielding+util=80%",
        "kind": "cuda-gibbs",
        "device": "0",
        "cfg": {"gpu_utilization": 80, "yielding": True},
    },
    {
        "label": "Gibbs sms_per_nonce=8",
        "kind": "cuda-gibbs",
        "device": "0",
        "cfg": {"sms_per_nonce": 8},
    },
]


def make_genesis():
    return create_genesis_block({
        "header": {
            "previous_hash": "00" * 32,
            "index": 0,
            "timestamp": 0,
            "data_hash": "00" * 32,
        },
        "data": "utilization/yielding smoke test",
        "next_block_requirements": {
            "difficulty_energy": TARGET_ENERGY,
            "min_diversity": 0.10,
            "min_solutions": 3,
            "timeout_to_difficulty_adjustment_decay": 600,
            "h_values": [-1.0, 0.0, 1.0],
        },
        "quantum_proof": None,
        "miner_info": None,
    })


def run_scenario(
    scenario: Dict[str, Any],
    genesis,
    node_info: MinerInfo,
    log_queue,
) -> Dict[str, Any]:
    label = scenario["label"]
    spec = {
        "id": label.replace(" ", "-"),
        "kind": scenario["kind"],
        "cfg": dict(scenario["cfg"]),
        "args": {"device": scenario["device"]},
    }

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  kind={scenario['kind']}  cfg={scenario['cfg']}")
    print(f"{'=' * 60}")

    handle = MinerHandle(spec, log_queue)
    requirements = genesis.next_block_requirements
    prev_timestamp = int(time.time())
    handle.mine(genesis, node_info, requirements, prev_timestamp)

    result = None
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        try:
            msg = handle.resp.get(timeout=1.0)
        except Empty:
            continue
        if hasattr(msg, 'miner_id') and hasattr(msg, 'energy'):
            result = msg
            break

    handle.close()

    if result is None:
        print(f"  TIMEOUT ({TIMEOUT}s)")
        return {
            "label": label,
            "passed": False,
            "reason": "timeout",
        }

    passed = result.energy <= TARGET_ENERGY
    print(
        f"  energy={result.energy:.1f}  "
        f"solutions={result.num_valid}  "
        f"diversity={result.diversity:.3f}  "
        f"time={result.mining_time:.1f}s  "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return {
        "label": label,
        "passed": passed,
        "energy": result.energy,
        "solutions": result.num_valid,
        "diversity": result.diversity,
        "time": result.mining_time,
    }


def run():
    setup_logging(log_level="INFO")
    log_queue, log_listener = setup_multiprocess_logging()

    genesis = make_genesis()
    signer = BlockSigner(seed=os.urandom(32))
    node_info = MinerInfo(
        miner_id="smoke-util",
        miner_type="smoke",
        reward_address=signer.ecdsa_public_key_bytes,
        ecdsa_public_key=signer.ecdsa_public_key_bytes,
        wots_public_key=signer.wots_plus_public_key,
        next_wots_public_key=signer.wots_plus_public_key,
    )

    print(
        f"Running {len(SCENARIOS)} scenarios, "
        f"target energy: {TARGET_ENERGY}"
    )

    results = []
    for scenario in SCENARIOS:
        r = run_scenario(
            scenario, genesis, node_info, log_queue,
        )
        results.append(r)

    log_listener.stop()

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r.get("reason") == "timeout":
            detail = f"TIMEOUT ({TIMEOUT}s)"
        else:
            detail = (
                f"energy={r['energy']:.1f}  "
                f"time={r['time']:.1f}s"
            )
        print(f"  [{status}] {r['label']:30s}  {detail}")
        if not r["passed"]:
            all_ok = False

    print(f"{'=' * 60}")
    if not all_ok:
        print("SOME SCENARIOS FAILED")
        sys.exit(1)
    print("ALL SCENARIOS PASSED")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
