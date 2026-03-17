"""Smoke test for Node with GPU-local (CUDA) persistent miner worker.

Run:
  python tests/smoke_node_gpu_local.py

Requires: PyTorch with CUDA available and at least device 0.
"""
from __future__ import annotations

import asyncio
import multiprocessing

from shared.node import Node
from shared.block import create_genesis_block


async def run():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8082},
        "gpu": {"backend": "local", "devices": ["0"]},
    }
    genesis_block = create_genesis_block()
    node = Node(node_id="node-gpu-local", miners_config=miners_config, genesis_block=genesis_block)

    print("Starting mining round on GPU (local CUDA device 0)...")
    try:
        result = await node.mine_block(genesis_block)
    except Exception as e:
        print(f"GPU local smoke test failed to start or run: {e}")
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
