"""Smoke test for Node with QPU persistent miner worker.

Run:
  python tests/smoke_node_qpu.py

Requires: D-Wave Ocean SDK and DWAVE_API_KEY in .env for real QPU access.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from shared.node import Node
from shared.block import create_genesis_block


async def run():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8084},
        "qpu": {},
    }
    if not os.getenv("DWAVE_API_KEY"):
        print("DWAVE_API_KEY not set in .env; QPU connection will fail")
        return

    genesis_block = create_genesis_block()
    node = Node(node_id="node-qpu", miners_config=miners_config, genesis_block=genesis_block)

    print("Starting mining round on QPU...")
    try:
        result = await node.mine_block(genesis_block)
    except Exception as e:
        print(f"QPU smoke test failed to start or run: {e}")
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
