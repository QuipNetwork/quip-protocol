"""Smoke test for Node with QPU persistent miner worker (falls back to mock if no creds).

Run:
  python -m tests.smoke_node_qpu

Requires: D-Wave Ocean SDK and DWAVE_API_TOKEN for real QPU, otherwise mock sampler is used.
"""
from __future__ import annotations

import multiprocessing
import time
import os

from shared.node import Node


def main():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8084},
        "qpu": {},
    }
    if not os.getenv("DWAVE_API_TOKEN"):
        print("DWAVE_API_TOKEN not set; test will run with mock sampler")

    node = Node(node_id="node-qpu", miners_config=miners_config)

    stop_event = multiprocessing.Event()
    header = f"prevhash0|index1|{int(time.time())}|data"

    print("Starting mining round on QPU (or mock)...")
    try:
        result = node.mine_block(header, stop_event)
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
    main()

