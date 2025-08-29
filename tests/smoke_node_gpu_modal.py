"""Smoke test for Node with GPU Modal persistent miner worker.

Run:
  python -m tests.smoke_node_gpu_modal

Requires: modal installed and authenticated (modal token set up).
"""
from __future__ import annotations

import multiprocessing
import time

from shared.node import Node


def main():
    miners_config = {
        "global": {"host": "0.0.0.0", "port": 8083},
        "gpu": {"backend": "modal", "types": ["t4"]},
    }
    node = Node(node_id="node-gpu-modal", miners_config=miners_config)

    stop_event = multiprocessing.Event()
    header = f"prevhash0|index1|{int(time.time())}|data"

    print("Starting mining round on GPU (Modal t4)...")
    try:
        result = node.mine_block(header, stop_event)
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
    main()

