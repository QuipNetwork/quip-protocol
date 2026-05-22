"""Shared persistent miner worker process and factory.

This worker runs a loop handling commands from the parent process:
- mine_work_item {context}
- stop_mining
- get_stats
- shutdown

It constructs the correct concrete miner from a simple picklable spec dict:
  {"id": "CPU-1", "kind": "cpu", "args": {...},
   "cfg": {"difficulty_energy": -15500.0, "min_diversity": 0.38, "min_solutions": 70}}
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import multiprocessing.synchronize as mpsync
import traceback
from typing import Any, Dict, Optional

import CPU  # noqa: E402
import GPU  # noqa: E402
import QPU  # noqa: E402

from shared.logging_config import QuipFormatter

# Global logger for this module
log = None
logger = logging.getLogger(__name__)


def _setup_child_process_logging(log_queue=None):
    """Set up logging for child processes to use QuipFormatter and optionally queue logging."""
    global log

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_queue is not None:
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)
        root_logger.setLevel(logging.DEBUG)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(QuipFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    log = logging.getLogger(__name__)

# NOTE: the legacy `_signal_aware_mining_worker` (a dedicated per-attempt
# child process used by `MinerHandle.mine_with_timeout`) was removed in
# v0.2 along with `BaseMiner.mine_block`. The substrate controller drives
# cancellation directly via the persistent worker's shared `stop_event`,
# so the second-process scaffolding is no longer needed.


def build_miner_from_spec(spec: Dict[str, Any]):
    kind = spec["kind"].lower()
    miner_id = spec["id"]
    cfg = dict(spec.get("cfg", {}))
    args = dict(spec.get("args", {}))

    if kind == "cpu":
        # Match the cuda/modal/cuda-gibbs paths: pass both cfg and args so a
        # `topology=` override in spec.args propagates to the sampler. Phase 4
        # controller relies on this to bind the miner to the chain's
        # registered topology.
        return CPU.SimulatedAnnealingMiner(miner_id, **cfg, **args)
    elif kind == "metal":
        if not GPU.METAL_AVAILABLE:
            raise RuntimeError(
                "Metal miner requested but Metal is not available (requires macOS with Metal support)"
            )
        return GPU.MetalMiner(miner_id, **cfg, **args)
    elif kind == "cuda":
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA miner requested but CUDA is not available (requires CuPy and CUDA toolkit)"
            )
        return GPU.CudaMiner(miner_id, **cfg, **args)
    elif kind == "modal":
        if not GPU.MODAL_AVAILABLE:
            raise RuntimeError(
                "Modal miner requested but Modal is not available (requires modal SDK: pip install modal)"
            )
        return GPU.ModalMiner(miner_id, **cfg, **args)
    elif kind == "cuda-gibbs":
        if not GPU.CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA Gibbs miner requested but not available "
                "(requires CuPy and CUDA toolkit)"
            )
        return GPU.CudaMiner(
            miner_id,
            update_mode="gibbs",
            **cfg,
            **args,
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
            cfg = {
                k: v
                for k, v in cfg.items()
                if k
                not in (
                    "daily_budget",
                    "qpu_min_blocks_for_estimation",
                    "qpu_ema_alpha",
                    "qpu_type",
                )
            }
        return QPU.DWaveMiner(miner_id, time_config=time_config, **cfg)
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
            # Wake any drainer blocked on `resp_q.get()`. Without this the
            # parent-side `loop.run_in_executor(None, handle.resp.get)`
            # blocks forever after worker exit (until Python 3.14's executor
            # join timeout — 300s by default — fires during loop shutdown).
            try:
                resp_q.put({"op": "shutdown_ack"})
            except Exception:  # noqa: BLE001 — best-effort
                pass
            return
        elif op == "get_stats":
            data = miner.get_stats()
            resp_q.put({"op": "stats", "data": data, "id": spec.get("id")})
        elif op == "mine_work_item":
            # Substrate-mode entry point. The controller pushes a
            # SubstrateMiningContext (or MempoolJobContext) through the
            # request queue; the worker hands it to
            # BaseMiner.mine_work_item which runs the protocol-neutral
            # search loop.
            #
            # We always emit *something* on resp_q after mine_work_item
            # returns — either a `mine_result` op wrapping the
            # MiningResult or a `work_item_done` sentinel. Both responses
            # carry the same `dispatch_id` the request came in with so
            # the controller can pair late results with the exact context
            # they were dispatched against, and reject results from a
            # dispatch that was already cancelled by the next one.
            dispatch_id = msg.get("dispatch_id", 0)
            context = msg.get("context")
            if context is None:
                resp_q.put(
                    {
                        "op": "error",
                        "message": "Missing context for mine_work_item",
                        "id": spec.get("id"),
                        "dispatch_id": dispatch_id,
                    }
                )
                continue
            try:
                result = miner.mine_work_item(context, stop_event)
            except Exception as exc:
                logger.error(
                    f"[{miner.miner_id}] mine_work_item raised: "
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                resp_q.put(
                    {
                        "op": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "id": spec.get("id"),
                        "dispatch_id": dispatch_id,
                    }
                )
            else:
                if result is not None:
                    resp_q.put(
                        {
                            "op": "mine_result",
                            "id": spec.get("id"),
                            "dispatch_id": dispatch_id,
                            "result": result,
                        }
                    )
                else:
                    resp_q.put(
                        {
                            "op": "work_item_done",
                            "id": spec.get("id"),
                            "dispatch_id": dispatch_id,
                        }
                    )
        else:
            resp_q.put(
                {"op": "error", "message": f"Unknown op {op}", "id": spec.get("id")}
            )
            logger.warning("%s: Unknown op %s", miner.miner_id, op)
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
        # Monotonic generation counter. Bumped on every mine_work_item
        # dispatch; the worker echoes the id on every response so the
        # controller can pair a late result with the exact context it
        # was produced against (rather than whatever's currently
        # dispatched).
        self._next_dispatch_id: int = 0
        self._active_dispatch_id: int = 0
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
        if k == "cuda-gibbs":
            return "GPU-CUDA-Gibbs"
        return k.upper()

    def mine_work_item(self, context) -> int:
        """Dispatch a substrate-mode mining attempt.

        ``context`` is a ``SubstrateMiningContext`` or ``MempoolJobContext``.
        Same stop_event semantics as ``mine``: the clear here brackets the
        enqueue so a cancel landing between clear and the worker dequeue
        still short-circuits the worker's loop.

        Returns the dispatch_id assigned to this attempt. Every worker
        response for this attempt (``mine_result`` / ``work_item_done`` /
        ``error``) will echo the same id so the caller can pair late
        results with the right context.
        """
        self._next_dispatch_id += 1
        self._active_dispatch_id = self._next_dispatch_id
        self.stop_event.clear()
        self.req.put(
            {
                "op": "mine_work_item",
                "context": context,
                "dispatch_id": self._active_dispatch_id,
            }
        )
        return self._active_dispatch_id

    def cancel(self):
        """Cancel the current mining operation.

        Signals the running mining loop directly via the shared
        ``stop_event`` so the worker observes the cancel within one
        iteration of its inner loop. We deliberately do NOT also enqueue
        a ``stop_mining`` op on the request queue: that op can sit in
        the queue while the worker is busy mining, and then get consumed
        by a *later* dispatch's clear → mine_work_item → req.get
        sequence — cancelling the new work with a stale cancel. The
        ``stop_event`` is the single source of truth for cancellation.

        Idempotent — safe to call when the worker is idle (the set is a
        no-op cleared by the next ``mine_work_item()``).
        """
        self.stop_event.set()

    def get_stats(self) -> dict:
        self.req.put({"op": "get_stats"})
        msg = self.resp.get(timeout=2.0)
        if isinstance(msg, dict) and msg.get("op") == "stats":
            return msg.get("data", {})
        else:
            raise ValueError(
                f"Miner {self.miner_id} did not respond to get_stats: {msg}"
            )

