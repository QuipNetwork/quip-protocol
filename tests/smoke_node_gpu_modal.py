"""Smoke test for Node with GPU Modal persistent miner worker.

Run:
  python tests/smoke_node_gpu_modal.py

Requires: modal installed and authenticated (modal token set up).
"""
from __future__ import annotations

import asyncio
import multiprocessing

from shared.node import Node
from shared.block import create_genesis_block


async def run():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8083},
        "gpu": {"backend": "modal", "types": ["t4"]},
    }
    genesis_block = create_genesis_block()
    node = Node(node_id="node-gpu-modal", miners_config=miners_config, genesis_block=genesis_block)

    print("Starting mining round on GPU (Modal t4)...")
    try:
        result = await node.mine_block(genesis_block)
    except Exception as e:
        print(f"GPU modal smoke test failed to start or run: {e}")
        node.close()
        return

    if result:
        print(f"Winner: {result.miner_id}, energy={result.energy:.2f}, num_valid={result.num_valid}")
    else:
        print("No winner (timeout or no valid solutions)")

    print("Node stats:")
    print(node.get_stats())

    node.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    asyncio.run(run())
