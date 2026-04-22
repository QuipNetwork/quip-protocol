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
        logger.info(f"Building miner in worker: kind={spec.get('kind')}, id={spec.get('id')}")
        miner = build_miner_from_spec(spec)
        logger.info(f"Miner built successfully in worker: {miner.miner_type} - {miner.miner_id}")

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
        import traceback
        logger.error(f"Mining worker error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

    # Process exits naturally


def build_miner_from_spec(spec: Dict[str, Any]):
    kind = spec["kind"].lower()
    miner_id = spec["id"]
    cfg = dict(spec.get("cfg", {}))
    args = dict(spec.get("args", {}))

    if kind == "cpu":
        return CPU.SimulatedAnnealingMiner(miner_id, **cfg)
    elif kind == "metal":
        if not GPU.METAL_AVAILABLE:
            raise RuntimeError("Metal miner requested but Metal is not available (requires macOS with Metal support)")
        return GPU.MetalMiner(miner_id, **cfg)
    elif kind == "cuda":
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError("CUDA miner requested but CUDA is not available (requires CuPy and CUDA toolkit)")
        return GPU.CudaMiner(miner_id, **cfg, **args)
    elif kind == "modal":
        if not GPU.MODAL_AVAILABLE:
            raise RuntimeError("Modal miner requested but Modal is not available (requires modal SDK: pip install modal)")
        return GPU.ModalMiner(miner_id, **cfg, **args)
    elif kind == "cuda-gibbs":
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA Gibbs miner requested but not available "
                "(requires CuPy and CUDA toolkit)")
        return GPU.CudaMiner(
            miner_id, update_mode="gibbs", **cfg, **args,
        )
    elif kind == "qpu":
        # Build QPU time config if daily budget is specified
        time_config = None
        if cfg.get("daily_budget"):
            from QPU.qpu_time_manager import QPUTimeConfig, parse_duration
            time_config = QPUTimeConfig(
                daily_budget_seconds=parse_duration(cfg["daily_budget"]),
                min_blocks_for_estimation=cfg.get("qpu_min_blocks_for_estimation", 5),
                ema_alpha=cfg.get("qpu_ema_alpha", 0.3),
            )
            # Remove time config keys from cfg to avoid passing them to miner
            cfg = {k: v for k, v in cfg.items()
                   if k not in ("daily_budget", "qpu_min_blocks_for_estimation",
                                "qpu_ema_alpha", "qpu_type")}
        return QPU.DWaveMiner(miner_id, time_config=time_config, **cfg)
    elif kind == "cpu-filtered":
        from CPU.sa_filtered_miner import SAFilteredMiner
        return SAFilteredMiner(miner_id, **cfg)
    else:
        raise ValueError(f"Unknown miner kind '{kind}'")


def miner_worker_main(
    req_q: mp.Queue,
    resp_q: mp.Queue,
    spec: Dict[str, Any],
    stop_event: mpsync.Event,
    log_queue: Optional[mp.Queue] = None,
):
    """Worker loop.

    ``stop_event`` is shared with the parent MinerHandle. The parent sets
    it from ``cancel()``; the miner polls it during its inner mining loop
    and returns None as soon as it fires. Sharing the event across the
    process boundary is what makes cancellation observable while
    ``mine_block`` is running — the command queue can't deliver a
    ``stop_mining`` op until ``mine_block`` returns, which defeats the
    whole point.
    """
    # Set up logging for child process
    _setup_child_process_logging(log_queue)
    logger.info(f"Building miner: kind={spec.get('kind')}, id={spec.get('id')}")
    try:
        miner = build_miner_from_spec(spec)
        logger.info(f"Miner built successfully: {miner.miner_type} - {miner.miner_id}")
    except Exception as e:
        logger.error(f"Failed to build miner {spec.get('id')}: {e}")
        raise

    while True:
        msg = req_q.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")

        if op == "shutdown":
            logger.info(f"Shutting down miner {miner.miner_id}")
            stop_event.set()
            return
        elif op == "get_stats":
            data = miner.get_stats()
            resp_q.put({"op": "stats", "data": data, "id": spec.get("id")})
        elif op == "stop_mining":
            # Redundant with the parent's direct set(), but keeps the op
            # available for callers that only have the request queue.
            stop_event.set()
        elif op == "mine_block":
            prev_block = msg.get("block")
            requirements = msg.get("requirements")
            node_info = msg.get("node_info")
            prev_timestamp = msg.get("prev_timestamp")
            if prev_block is None or requirements is None or node_info is None or prev_timestamp is None:
                resp_q.put({"op": "error", "message": "Missing node_info, block or requirements", "id": spec.get("id")})
                continue
            # NOTE: do not clear stop_event here. The parent clears it
            # in mine() before enqueueing this op; if cancel() lands in
            # the gap between that enqueue and our dequeue, the parent
            # has already set() it and a clear here would silently wipe
            # the cancel — exactly the original bug. Letting a set
            # event short-circuit mine_block() is the desired behaviour
            # (the cancellation reached us before mining started).
            result = miner.mine_block(prev_block, node_info, requirements, prev_timestamp, stop_event)
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
        # Shared with the worker so cancel() can signal the active
        # mine_block() directly, not via the command queue (which the
        # worker cannot drain while mining).
        self.stop_event: mpsync.Event = mp.Event()
        self.proc: mp.Process = mp.Process(
            target=miner_worker_main,
            args=(self.req, self.resp, spec, self.stop_event, log_queue),
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
        if k == "cpu-filtered":
            return "CPU-Filtered"
        if k == "cuda-gibbs":
            return "GPU-CUDA-Gibbs"
        return k.upper()

    def mine(self, block, node_info, requirements, prev_timestamp: int = 0):
        # Sole clear point for the shared cancel signal. The worker
        # never clears it — clearing on either side of the queue would
        # race with cancel() and silently wipe the signal (the original
        # bug this MR fixes). Clearing here, before enqueueing, means
        # any cancel() called between this clear and the worker
        # dequeueing the op stays observable: the worker enters
        # mine_block with stop_event already set and short-circuits.
        self.stop_event.clear()
        self.req.put({"op": "mine_block", "block": block, "node_info": node_info, "requirements": requirements, "prev_timestamp": prev_timestamp})

    def cancel(self):
        """Cancel the current mining operation.

        Signals the running ``mine_block()`` directly via the shared
        stop event so the inner loop observes the cancel within one
        iteration; also enqueues the ``stop_mining`` op so callers
        operating only on the request queue can still trigger a cancel.
        Idempotent — safe to call when the worker is idle (the set is a
        no-op cleared by the next ``mine()``).
        """
        self.stop_event.set()
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
            except Exception as e:
                # Queue.Empty is expected when no result, other exceptions should be logged
                if not str(type(e).__name__) == 'Empty':
                    logger.debug(f"No result from mining worker: {type(e).__name__}: {e}")
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
