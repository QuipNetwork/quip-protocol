#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Mine one block with SA and Gibbs, with/without yielding.

Runs 4 trials:
  1. SA miner, yielding=False
  2. SA miner, yielding=True
  3. Gibbs miner, yielding=False
  4. Gibbs miner, yielding=True

Each trial mines until a block meeting -14900 energy is found.
Reports wall-clock time for each.
"""

import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

sys.path.append(str(Path(__file__).parent.parent))

from shared.block import BlockRequirements, create_genesis_block


@dataclass
class NodeInfo:
    """Minimal node info for mining."""
    miner_id: str


DIFFICULTY = -14700.0
REQUIREMENTS = BlockRequirements(
    difficulty_energy=DIFFICULTY,
    min_diversity=0.2,
    min_solutions=5,
    timeout_to_difficulty_adjustment_decay=0,
)

# Override adaptive params to keep kernel launches short.
# Default adapt gives ~1500 betas × 185 reads = 15 min/batch.
# These give ~200 betas × 64 reads = seconds/batch.
OVERRIDE_PARAMS = {
    'num_sweeps': 200,
    'num_reads': 64,
}


def run_trial(label, miner_cls, device, yielding):
    """Run a single mining trial, return elapsed seconds."""
    print(f"\n{'='*60}")
    print(f"  {label}  (yielding={yielding})")
    print(f"{'='*60}")

    miner = miner_cls(
        miner_id=f"test-{label}",
        device=device,
        yielding=yielding,
        gpu_utilization=50,
    )

    # Override adaptive params for fast kernel launches
    miner._adapt_mining_params = lambda *a, **kw: {
        **OVERRIDE_PARAMS, 'num_sweeps_per_beta': 1,
    }

    prev_block = create_genesis_block()
    prev_block.next_block_requirements = REQUIREMENTS
    node_info = NodeInfo(miner_id=miner.miner_id)
    stop_event = multiprocessing.Event()

    start = time.time()
    result = miner.mine_block(
        prev_block=prev_block,
        node_info=node_info,
        requirements=REQUIREMENTS,
        prev_timestamp=prev_block.header.timestamp,
        stop_event=stop_event,
    )
    elapsed = time.time() - start

    if result:
        print(
            f"\n  MINED in {elapsed:.1f}s | "
            f"energy={result.energy:.1f} | "
            f"solutions={result.num_valid} | "
            f"diversity={result.diversity:.3f}"
        )
    else:
        print(f"\n  FAILED after {elapsed:.1f}s")

    # Cleanup GPU memory (mine_block calls _post_mine_cleanup,
    # but free memory pools for clean inter-trial state)
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    return elapsed, result


def main():
    device = "0"  # CUDA_VISIBLE_DEVICES remaps

    from GPU.cuda_miner import CudaMiner

    def GibbsMiner(*a, **kw):
        return CudaMiner(*a, update_mode="gibbs", **kw)

    results = {}

    # SA trials
    for yielding in [False, True]:
        tag = f"SA-yield={yielding}"
        elapsed, res = run_trial(
            tag, CudaMiner, device, yielding,
        )
        results[tag] = elapsed

    # Gibbs trials
    for yielding in [False, True]:
        tag = f"Gibbs-yield={yielding}"
        elapsed, res = run_trial(
            tag, GibbsMiner, device, yielding,
        )
        results[tag] = elapsed

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (difficulty={DIFFICULTY})")
    print(f"{'='*60}")
    for tag, elapsed in results.items():
        print(f"  {tag:30s}  {elapsed:8.1f}s")


if __name__ == "__main__":
    main()
