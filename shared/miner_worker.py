"""Shared persistent miner worker process and factory.

This worker runs a loop handling commands from the parent process:
- mine_block {block, requirements}
- stop_mining
- get_stats
- shutdown

It constructs the correct concrete miner from a simple picklable spec dict:
  {"id": "CPU-1", "kind": "cpu", "args": {...},
   "cfg": {"difficulty_energy": -15500.0, "min_diversity": 0.38, "min_solutions": 70}}
"""
from __future__ import annotations

import os

# Set default DWave environment variables before any DWave libraries are imported
if "DWAVE_API_KEY" not in os.environ:
    os.environ["DWAVE_API_KEY"] = "MISSING IN CONFIG"
if "DWAVE_API_TOKEN" not in os.environ:
    os.environ["DWAVE_API_TOKEN"] = "MISSING IN CONFIG"

from shared.logging_config import QuipFormatter
import logging

# Global logger for this module
log = None

def _setup_child_process_logging():
    """Set up logging for child processes to use QuipFormatter."""
    global log

    # Get root logger
    root_logger = logging.getLogger()

    # Check if QuipFormatter is already configured
    has_quip_formatter = any(
        isinstance(handler.formatter, QuipFormatter)
        for handler in root_logger.handlers
    )

    if not has_quip_formatter:
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add QuipFormatter
        formatter = QuipFormatter()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    # Create module logger that will inherit from root
    module_logger = logging.getLogger(__name__)
    log = module_logger

# Initialize module logger
logger = logging.getLogger(__name__)

import multiprocessing as mp
import multiprocessing.synchronize as mpsync
from typing import Any, Dict

import CPU
import GPU
import QPU

def build_miner_from_spec(spec: Dict[str, Any]):
    kind = spec["kind"].lower()
    miner_id = spec["id"]
    cfg = dict(spec.get("cfg", {}))
    args = dict(spec.get("args", {}))

    if kind == "cpu":
        return CPU.SimulatedAnnealingMiner(miner_id, **cfg)
    elif kind == "metal":
        return GPU.MetalMiner(miner_id, **cfg)
    elif kind == "cuda":
        return GPU.CudaMiner(miner_id, **cfg, **args)
    elif kind == "modal":
        return GPU.ModalMiner(miner_id, **cfg, **args)
    elif kind == "qpu":
        return QPU.DWaveMiner(miner_id, **cfg)
    else:
        raise ValueError(f"Unknown miner kind '{kind}'")


def miner_worker_main(req_q: mp.Queue, resp_q: mp.Queue, spec: Dict[str, Any]):
    # Set up logging for child process
    _setup_child_process_logging()

    miner = build_miner_from_spec(spec)
    current_stop: mpsync.Event = mp.Event()

    while True:
        msg = req_q.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")

        if op == "shutdown":
            if log:
                log.info(f"Shutting down miner {miner.miner_id}")
            else:
                logger.info(f"Shutting down miner {miner.miner_id}")
            current_stop.set()
            return
        elif op == "get_stats":
            data = miner.get_stats()
            resp_q.put({"op": "stats", "data": data, "id": spec.get("id")})
        elif op == "stop_mining":
            current_stop.set()
        elif op == "mine_block":
            prev_block = msg.get("block")
            requirements = msg.get("requirements")
            node_info = msg.get("node_info")
            if prev_block is None or requirements is None or node_info is None:
                resp_q.put({"op": "error", "message": "Missing node_info, block or requirements", "id": spec.get("id")})
                continue
            current_stop = mp.Event()
            miner.mine_block(prev_block, node_info, requirements, resp_q, current_stop)
        else:
            resp_q.put({"op": "error", "message": f"Unknown op {op}", "id": spec.get("id")})
            if log:
                log.info(f"{miner.miner_id}: Unknown op {op}")
            else:
                logger.info(f"{miner.miner_id}: Unknown op {op}")
            continue

class MinerHandle:
    """Wrapper around a persistent miner worker process."""
    def __init__(self, ctx, spec: dict):
        self.spec = spec
        self.req: mp.Queue = ctx.Queue()
        self.resp: mp.Queue = ctx.Queue()
        self.proc: mp.Process = ctx.Process(
            target=miner_worker_main,
            args=(self.req, self.resp, spec),
        )

        self.proc.start()

    @property
    def miner_id(self) -> str:
        return self.spec.get("id", "")

    @property
    def miner_type(self) -> str:
        k = self.spec.get("kind", "")
        if k == "cpu":
            return "CPU"
        if k == "qpu":
            return "QPU"
        if k == "modal":
            t = (self.spec.get("args", {}) or {}).get("gpu_type", "t4")
            return f"GPU-{t.upper()}"
        if k == "cuda":
            d = (self.spec.get("args", {}) or {}).get("device", "0")
            return f"GPU-LOCAL:{d}"
        if k == "metal":
            return "GPU-MPS"
        return k.upper()

    def mine(self, block, node_info, requirements):
        self.req.put({"op": "mine_block", "block": block, "node_info": node_info, "requirements": requirements})

    def cancel(self):
        self.req.put({"op": "stop_mining"})

    def get_stats(self) -> dict:
        self.req.put({"op": "get_stats"})
        msg = self.resp.get(timeout=2.0)
        if isinstance(msg, dict) and msg.get("op") == "stats":
            return msg.get("data", {})
        else:
            raise ValueError(f"Miner {self.miner_id} did not respond to get_stats: {msg}")

    def close(self):
        self.req.put({"op": "shutdown"})
        try:
            if self.proc.is_alive():
                self.proc.join(timeout=2)
        except Exception:
            pass
