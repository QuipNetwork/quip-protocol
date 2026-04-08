"""Standalone long-lived miner service process.

Unlike the spawn-per-block ``MinerHandle``, a miner service starts
once, initializes hardware (CUDA context, Metal device, QPU connection),
and accepts mining work repeatedly via IPC. This eliminates process
spawn overhead and keeps hardware "warm" between blocks.

IPC protocol (parent <-> miner via multiprocessing.Queue)::

    Parent -> Miner:
        {"cmd": "mine", "block": Block, "node_info": MinerInfo,
         "requirements": BlockRequirements, "prev_timestamp": int}
        {"cmd": "stop"}
        {"cmd": "status"}
        {"cmd": "shutdown"}

    Miner -> Parent:
        MiningResult (on success)
        {"event": "stopped"}
        {"event": "status", "mining": bool, "miner_id": str, ...}
        {"event": "error", "message": str}

The service communicates via multiprocessing queues (same IPC mechanism
as the existing MinerHandle). When pyzmq is available, a ZMQ-based
variant can be layered on top for lower latency and better crash
detection.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize as mpsync
import queue
import signal
import time
from typing import Any, Dict, Optional

from shared.logging_config import QuipFormatter


def _setup_service_logging(miner_id: str) -> logging.Logger:
    """Configure logging for the miner service process."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(QuipFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger(f"miner-svc[{miner_id}]")


def miner_service_main(
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    spec: Dict[str, Any],
    stop_event: mpsync.Event,
) -> None:
    """Miner service process entry point.

    Builds the miner once, then enters a command loop accepting
    mining work until shutdown.

    Args:
        cmd_queue: Commands from parent (mine, stop, status, shutdown).
        result_queue: Results sent back to parent.
        spec: Miner spec dict (kind, id, cfg, args).
        stop_event: Shared event for cooperative shutdown.
    """
    miner_id = spec.get("id", "unknown")
    log = _setup_service_logging(miner_id)

    def _signal_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Build the miner once (hardware init happens here)
    log.info(f"Building miner: kind={spec.get('kind')}, id={miner_id}")
    try:
        from shared.miner_worker import build_miner_from_spec
        miner = build_miner_from_spec(spec)
        log.info(
            f"Miner ready: {miner.miner_type} - {miner.miner_id} "
            f"(hardware warm)"
        )
    except Exception as exc:
        log.error(f"Failed to build miner: {exc}")
        result_queue.put({
            "event": "error",
            "message": f"Build failed: {exc}",
        })
        return

    mining_stop = mp.Event()
    is_mining = False

    while not stop_event.is_set():
        try:
            msg = cmd_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not isinstance(msg, dict):
            continue

        op = msg.get("cmd")

        if op == "shutdown":
            log.info("Shutdown requested")
            mining_stop.set()
            break

        elif op == "stop":
            mining_stop.set()
            is_mining = False
            result_queue.put({"event": "stopped"})

        elif op == "status":
            result_queue.put({
                "event": "status",
                "mining": is_mining,
                "miner_id": miner_id,
                "miner_type": getattr(miner, 'miner_type', 'unknown'),
                "kind": spec.get("kind", "unknown"),
            })

        elif op == "mine":
            prev_block = msg.get("block")
            node_info = msg.get("node_info")
            requirements = msg.get("requirements")
            prev_timestamp = msg.get("prev_timestamp", 0)

            if prev_block is None or requirements is None or node_info is None:
                result_queue.put({
                    "event": "error",
                    "message": "Missing block, node_info, or requirements",
                })
                continue

            mining_stop = mp.Event()
            is_mining = True

            try:
                result = miner.mine_block(
                    prev_block, node_info, requirements,
                    prev_timestamp, mining_stop,
                )
                is_mining = False
                if result is not None:
                    result_queue.put(result)
            except Exception as exc:
                is_mining = False
                log.error(f"Mining error: {exc}")
                result_queue.put({
                    "event": "error",
                    "message": str(exc),
                })

        elif op == "get_stats":
            try:
                stats = miner.get_stats()
                result_queue.put({
                    "op": "stats",
                    "data": stats,
                    "id": miner_id,
                })
            except Exception as exc:
                result_queue.put({
                    "event": "error",
                    "message": f"get_stats failed: {exc}",
                })

        else:
            log.warning(f"Unknown command: {op}")

    log.info(f"Miner service {miner_id} stopped")


class MinerServiceHandle:
    """Parent-side handle for a long-lived miner service process.

    Drop-in replacement for ``MinerHandle`` that keeps the miner
    process alive between mining attempts. The miner initializes
    hardware once at startup and accepts repeated mining commands.

    Args:
        spec: Miner spec dict (kind, id, cfg, args).
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.cmd_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self._stop_event: mpsync.Event = mp.Event()
        self.proc: mp.Process = mp.Process(
            target=miner_service_main,
            args=(
                self.cmd_queue, self.result_queue,
                spec, self._stop_event,
            ),
            daemon=True,
        )
        self.proc.start()

    @property
    def miner_id(self) -> str:
        return self.spec.get("id", "")

    @property
    def miner_type(self) -> str:
        k = self.spec.get("kind", "")
        type_map = {
            "cpu": "CPU", "qpu": "QPU", "metal": "GPU-MPS",
            "cpu-filtered": "CPU-Filtered", "cuda-gibbs": "GPU-CUDA-Gibbs",
        }
        if k in type_map:
            return type_map[k]
        if k == "modal":
            t = (self.spec.get("args", {}) or {}).get("gpu_type", "t4")
            return f"GPU-{t.upper()}"
        if k == "cuda":
            d = (self.spec.get("args", {}) or {}).get("device", "0")
            return f"GPU-LOCAL:{d}"
        return k.upper()

    def mine(self, block, node_info, requirements, prev_timestamp: int = 0):
        """Submit a mining job. Non-blocking."""
        self.cmd_queue.put({
            "cmd": "mine",
            "block": block,
            "node_info": node_info,
            "requirements": requirements,
            "prev_timestamp": prev_timestamp,
        })

    def cancel(self):
        """Cancel current mining operation."""
        self.cmd_queue.put({"cmd": "stop"})

    def get_stats(self) -> dict:
        """Get miner statistics. Blocking with timeout."""
        self.cmd_queue.put({"cmd": "get_stats"})
        try:
            msg = self.result_queue.get(timeout=5.0)
            if isinstance(msg, dict) and msg.get("op") == "stats":
                return msg.get("data", {})
            return {}
        except queue.Empty:
            return {}

    def get_status(self) -> Optional[dict]:
        """Get service status. Blocking with timeout."""
        self.cmd_queue.put({"cmd": "status"})
        try:
            msg = self.result_queue.get(timeout=2.0)
            if isinstance(msg, dict) and msg.get("event") == "status":
                return msg
            return None
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Check if the service process is still running."""
        return self.proc.is_alive()

    def close(self):
        """Gracefully shut down the miner service."""
        self._stop_event.set()
        try:
            self.cmd_queue.put_nowait({"cmd": "shutdown"})
        except Exception:
            pass
        self.proc.join(timeout=5.0)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=1.0)

        # Drain queues to avoid macOS/Linux deadlock
        for q in (self.cmd_queue, self.result_queue):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

    # Compatibility with MinerHandle interface
    req = property(lambda self: self.cmd_queue)
    resp = property(lambda self: self.result_queue)
