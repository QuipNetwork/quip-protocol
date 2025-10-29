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

import time
from shared.logging_config import QuipFormatter
import logging
import signal

# Global logger for this module
log = None

def _setup_child_process_logging(log_queue=None):
    """Set up logging for child processes to use QuipFormatter and optionally queue logging."""
    global log

    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_queue is not None:
        # Use queue handler to send logs to parent process
        from logging.handlers import QueueHandler
        queue_handler = QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.DEBUG)  # Let parent process filter
    else:
        # Fallback to console logging with QuipFormatter
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
from typing import Any, Dict, Optional

import CPU
import GPU
import QPU

def _signal_aware_mining_worker(spec: Dict[str, Any], block, node_info, requirements, prev_timestamp: int, mining_queue: mp.Queue, result_queue: mp.Queue):
    """Dedicated mining worker process that handles mining with signal awareness."""
    # mining_queue is reserved for future use
    _ = mining_queue
    
    try:
        # Set up logging for child process
        _setup_child_process_logging()
        
        # Build the miner
        miner = build_miner_from_spec(spec)
        
        # Create a stop event that will never be set (child process doesn't monitor signals)
        # The parent process will terminate this process via SIGTERM when needed
        child_stop_event = mp.Event()
        
        # Perform the mining operation
        result = miner.mine_block(block, node_info, requirements, prev_timestamp, child_stop_event)
        
        # Send result back to parent
        if result is not None:
            result_queue.put(result)
            
    except Exception as e:
        # Log error and exit gracefully
        logger.error(f"Mining worker error: {e}")
    
    # Process exits naturally


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


def miner_worker_main(req_q: mp.Queue, resp_q: mp.Queue, spec: Dict[str, Any], log_queue: Optional[mp.Queue] = None):
    # Set up logging for child process
    _setup_child_process_logging(log_queue)
    miner = build_miner_from_spec(spec)
    current_stop: mpsync.Event = mp.Event()

    while True:
        msg = req_q.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")

        if op == "shutdown":
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
            prev_timestamp = msg.get("prev_timestamp")
            if prev_block is None or requirements is None or node_info is None or prev_timestamp is None:
                resp_q.put({"op": "error", "message": "Missing node_info, block or requirements", "id": spec.get("id")})
                continue
            current_stop = mp.Event()
            result = miner.mine_block(prev_block, node_info, requirements, prev_timestamp, current_stop)
            if result is not None:
                resp_q.put(result)
        else:
            resp_q.put({"op": "error", "message": f"Unknown op {op}", "id": spec.get("id")})
            logger.info(f"{miner.miner_id}: Unknown op {op}")
            continue

class MinerHandle:
    """Wrapper around a persistent miner worker process."""
    def __init__(self, spec: dict, log_queue: Optional[mp.Queue] = None):
        self.spec = spec
        self.req: mp.Queue = mp.Queue()
        self.resp: mp.Queue = mp.Queue()
        self.proc: mp.Process = mp.Process(
            target=miner_worker_main,
            args=(self.req, self.resp, spec, log_queue),
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

    def mine(self, block, node_info, requirements, prev_timestamp: int = 0):
        self.req.put({"op": "mine_block", "block": block, "node_info": node_info, "requirements": requirements, "prev_timestamp": prev_timestamp})

    def cancel(self):
        self.req.put({"op": "stop_mining"})

    def get_stats(self) -> dict:
        self.req.put({"op": "get_stats"})
        msg = self.resp.get(timeout=2.0)
        if isinstance(msg, dict) and msg.get("op") == "stats":
            return msg.get("data", {})
        else:
            raise ValueError(f"Miner {self.miner_id} did not respond to get_stats: {msg}")

    def mine_with_timeout(self, block, node_info, requirements, prev_timestamp: int, stop_event) -> Optional[Any]:
        """Mine a block with signal-responsive timeout using a dedicated child process."""
        # Create a dedicated mining worker process for this operation
        mining_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Create mining process
        mining_proc = mp.Process(
            target=_signal_aware_mining_worker,
            args=(self.spec, block, node_info, requirements, prev_timestamp, mining_queue, result_queue)
        )
        
        mining_proc.start()
        
        try:
            # Monitor stop_event while mining process runs
            while mining_proc.is_alive():
                if stop_event.is_set():
                    # Send SIGTERM for graceful cleanup
                    mining_proc.terminate()
                    
                    # Wait up to 2 seconds for graceful shutdown
                    mining_proc.join(timeout=2.0)
                    
                    # Force kill if still alive
                    if mining_proc.is_alive():
                        mining_proc.kill()
                        mining_proc.join(timeout=0.5)
                    
                    return None
                
                # Check every 100ms
                time.sleep(0.1)
            
            # Process completed, get result
            try:
                result = result_queue.get_nowait()
                return result
            except:
                return None
                
        finally:
            # Cleanup: ensure process is terminated
            if mining_proc.is_alive():
                mining_proc.terminate()
                mining_proc.join(timeout=1.0)
                if mining_proc.is_alive():
                    mining_proc.kill()

    def close(self):
        self.req.put({"op": "shutdown"})
        try:
            time.sleep(1)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=0.1)       
        except Exception:
            pass
