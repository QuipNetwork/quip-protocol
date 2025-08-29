"""Smoke test for Node with CPU-only persistent miner worker.

Run:
  python -m tests.smoke_node_cpu_only
"""
from __future__ import annotations

import multiprocessing
import time

from shared.node import Node


def main():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8080},
        "cpu": {"num_cpus": 1},
    }
    node = Node(node_id="node-1", miners_config=miners_config)

    stop_event = multiprocessing.Event()
    header = f"prevhash0|index1|{int(time.time())}|data"

    print("Starting mining round...")
    result = node.mine_block(header, stop_event)
    if result:
        print(f"Winner: {result.miner_id}, energy={result.energy:.2f}, num_valid={result.num_valid}")
    else:
        print("No winner (timeout or no valid solutions)")

    print("Node stats:")
    print(node.get_stats())

    node.close()


if __name__ == "__main__":
    main()

