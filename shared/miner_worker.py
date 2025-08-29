"""Shared persistent miner worker process and factory.

This worker runs a loop handling commands from the parent process:
- mine_block {block_header}
- stop_mining
- get_stats
- shutdown

It constructs the correct concrete miner from a simple picklable spec dict:
  {"id": "CPU-1", "kind": "cpu", "args": {...},
   "cfg": {"difficulty_energy": -15500.0, "min_diversity": 0.38, "min_solutions": 70}}
"""
from __future__ import annotations

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
    miner = build_miner_from_spec(spec)
    current_stop: mpsync.Event | None = None

    while True:
        msg = req_q.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")

        if op == "shutdown":
            miner.shutdown()
            break
        elif op == "get_stats":
            try:
                data = miner.get_stats()
                resp_q.put({"op": "stats", "data": data, "id": spec.get("id")})
            except Exception as e:
                resp_q.put({"op": "error", "message": str(e), "id": spec.get("id")})
        elif op == "stop_mining":
            if current_stop is not None:
                current_stop.set()
        elif op == "mine_block":
            block_header = msg.get("block_header")
            if block_header is None:
                resp_q.put({"op": "error", "message": "No block header", "id": spec.get("id")})
                continue
            current_stop = mp.Event()
            try:
                # Miner.mine_block is expected to put a MiningResult on resp_q
                miner.mine_block(block_header, resp_q, current_stop)
            except Exception as e:
                resp_q.put({"op": "error", "message": str(e), "id": spec.get("id")})
        else:
            resp_q.put({"op": "error", "message": f"Unknown op {op}", "id": spec.get("id")})
            print(f"{miner.miner_id}: Unknown op {op}")
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

    def mine(self, block_header: str):
        self.req.put({"op": "mine_block", "block_header": block_header})

    def cancel(self):
        self.req.put({"op": "stop_mining"})

    def get_stats(self) -> dict:
        self.req.put({"op": "get_stats"})
        try:
            msg = self.resp.get(timeout=2.0)
            if isinstance(msg, dict) and msg.get("op") == "stats":
                return msg.get("data", {})
        except Exception:
            pass
        return {"miner_id": self.miner_id, "miner_type": self.miner_type}

    def close(self):
        try:
            self.req.put({"op": "shutdown"})
        except Exception:
            pass
        try:
            if self.proc.is_alive():
                self.proc.join(timeout=2)
        except Exception:
            pass
