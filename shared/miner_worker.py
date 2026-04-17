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


def _extract_subgraph(all_nodes, all_edges, size):
    """Extract a connected subgraph of ``size`` nodes via BFS.

    Starts from a random node and grows outward.  Returns
    (sub_nodes, sub_edges) where sub_edges only includes edges
    between nodes in the subgraph.
    """
    import random

    adj = {n: [] for n in all_nodes}
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)

    start = random.choice(all_nodes)
    visited = set()
    queue = [start]

    while queue and len(visited) < size:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    sub_nodes = sorted(visited)
    sub_node_set = set(sub_nodes)
    sub_edges = [(u, v) for u, v in all_edges
                 if u in sub_node_set and v in sub_node_set]
    return sub_nodes, sub_edges


def build_miner_from_spec(spec: Dict[str, Any]):
    kind = spec["kind"].lower()
    miner_id = spec["id"]
    cfg = dict(spec.get("cfg", {}))
    args = dict(spec.get("args", {}))

    if kind == "cpu":
        import CPU
        return CPU.SimulatedAnnealingMiner(miner_id, **cfg)
    elif kind == "metal":
        import GPU
        if not GPU.METAL_AVAILABLE:
            raise RuntimeError("Metal miner requested but Metal is not available (requires macOS with Metal support)")
        return GPU.MetalMiner(miner_id, **cfg)
    elif kind == "cuda":
        import GPU
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError("CUDA miner requested but CUDA is not available (requires CuPy and CUDA toolkit)")
        return GPU.CudaMiner(miner_id, **cfg, **args)
    elif kind == "modal":
        import GPU
        if not GPU.MODAL_AVAILABLE:
            raise RuntimeError("Modal miner requested but Modal is not available (requires modal SDK: pip install modal)")
        return GPU.ModalMiner(miner_id, **cfg, **args)
    elif kind == "cuda-gibbs":
        import GPU
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA Gibbs miner requested but not available "
                "(requires CuPy and CUDA toolkit)")
        return GPU.CudaMiner(
            miner_id, update_mode="gibbs", **cfg, **args,
        )
    elif kind == "qpu":
        import QPU
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
    elif kind == "ibm_qaoa":
        import QPU
        from dwave_topologies import DEFAULT_TOPOLOGY
        subgraph_size = cfg.pop("subgraph_size", 28)
        backend_type = cfg.pop("backend", "aer")
        ibm_api_token = cfg.pop("ibm_api_token", None)
        ibm_backend_name = cfg.pop("ibm_backend_name", None)

        # Build backend
        backend = None  # None → AerSimulator (default)
        if backend_type == "ibm":
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                if ibm_api_token:
                    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token)
                else:
                    service = QiskitRuntimeService()
                backend_name = ibm_backend_name or "ibm_brisbane"
                backend = service.backend(backend_name)
                logger.info(f"Using IBM backend: {backend_name}")
            except ImportError:
                raise RuntimeError("IBM backend requested but qiskit-ibm-runtime is not installed")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to IBM backend: {e}")
        else:
            logger.info("Using AerSimulator backend (local simulation)")

        # Extract subgraph
        nodes = list(DEFAULT_TOPOLOGY.nodes)
        edges = list(DEFAULT_TOPOLOGY.edges)
        if subgraph_size < len(nodes):
            nodes, edges = _extract_subgraph(nodes, edges, subgraph_size)
            logger.info(f"Extracted {len(nodes)}-node subgraph ({len(edges)} edges) from {len(DEFAULT_TOPOLOGY.nodes)}-node topology")
        return QPU.IBMQAOAMiner(miner_id, nodes=nodes, edges=edges, backend=backend, **cfg)
    elif kind == "ionq_qaoa":
        import QPU
        from dwave_topologies import DEFAULT_TOPOLOGY
        subgraph_size = cfg.pop("subgraph_size", 28)
        backend_type = cfg.pop("backend", "aer")
        ionq_api_token = cfg.pop("ionq_api_token", None)
        ionq_backend_name = cfg.pop("ionq_backend_name", "aer_simulator")
 
        # Build backend
        backend = None  # None → AerSimulator (default)
        if backend_type == "ionq":
            # IonQ cloud backends — solver handles provider setup internally
            backend = None  # solver will use _create_ionq_backend()
            logger.info(f"Using IonQ backend: {ionq_backend_name}")
        else:
            logger.info("Using AerSimulator backend (local simulation)")
 
        # Extract subgraph
        nodes = list(DEFAULT_TOPOLOGY.nodes)
        edges = list(DEFAULT_TOPOLOGY.edges)
        if subgraph_size < len(nodes):
            nodes, edges = _extract_subgraph(nodes, edges, subgraph_size)
            logger.info(f"Extracted {len(nodes)}-node subgraph ({len(edges)} edges) from {len(DEFAULT_TOPOLOGY.nodes)}-node topology")
 
        # Pass IonQ credentials for cloud backend setup
        ionq_token = ionq_api_token if backend_type == "ionq" else None
        ionq_backend = ionq_backend_name if backend_type == "ionq" else "aer_simulator"
 
        return QPU.IonQQAOAMiner(
            miner_id, nodes=nodes, edges=edges, backend=backend,
            ionq_token=ionq_token, ionq_backend=ionq_backend, **cfg,
        )
    
    elif kind == "cpu-filtered":
        from CPU.sa_filtered_miner import SAFilteredMiner
        return SAFilteredMiner(miner_id, **cfg)
    else:
        raise ValueError(f"Unknown miner kind '{kind}'")


def miner_worker_main(req_q: mp.Queue, resp_q: mp.Queue, spec: Dict[str, Any], log_queue: Optional[mp.Queue] = None):
    # Set up logging for child process
    _setup_child_process_logging(log_queue)
    logger.info(f"Building miner: kind={spec.get('kind')}, id={spec.get('id')}")
    try:
        miner = build_miner_from_spec(spec)
        logger.info(f"Miner built successfully: {miner.miner_type} - {miner.miner_id}")
    except Exception as e:
        logger.error(f"Failed to build miner {spec.get('id')}: {e}")
        raise
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
        if k == "ibm_qaoa":
            return "IBM_QAOA"
        if k == "ionq_qaoa":
            return "IONQ_QAOA"
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
