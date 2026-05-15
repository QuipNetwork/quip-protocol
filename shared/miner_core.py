"""Miner orchestration state, extracted from `shared.node.Node` for Phase 5.

`MinerCore` owns the miner-side machinery that survives the v0.1 → v0.2
refactor: the persistent `MinerHandle` worker processes, the multiprocessing
log queue, the hardware descriptor cache, and the aggregate timing stats.
Everything Node was responsible for that pertains to chain state, P2P, or
SPHINCS+ block signing has been removed — the substrate controller owns
those concerns now.

Lifecycle:

    core = MinerCore(node_id="miner-1", miners_config={"cpu": {"num_cpus": 4}})
    controller = SubstrateMinerController(client, signer, core.miner_handles)
    try:
        await controller.run()
    finally:
        core.close()

`MinerCore` deliberately does NOT subclass `Node` and does NOT import legacy
chain types (`Block`, `BlockSigner`, `BlockRequirements`, etc.). The factory
helpers for cpu/gpu/qpu specs are forked from `Node._initialize_miners` —
Phase 5b's deletion sweep removes the original.
"""

from __future__ import annotations

import logging
import multiprocessing
from logging.handlers import QueueListener
from typing import Any, Callable, Dict, List, Optional

from shared.logging_config import init_component_logger
from shared.miner_worker import MinerHandle


# Config-section discovery, forked from `shared.node`. Phase 5b removes the
# duplicates when `node.py` is deleted.
_GPU_DEVICE_SECTIONS = ("cuda", "metal", "modal")
_QPU_DEVICE_SECTIONS = ("dwave", "ibm", "braket", "pasqal", "ionq", "origin")


class MinerCore:
    """Owns the persistent miner workers + telemetry, no chain or P2P state.

    The substrate controller drives the workers via
    `core.miner_handles[i].mine_work_item(context)` and `.cancel()` — same
    interface the legacy `Node.mine_block()` orchestration used. `MinerCore`
    itself doesn't initiate or coordinate mining; it just builds and tears
    down workers and tracks aggregate stats.
    """

    def __init__(
        self,
        node_id: str,
        miners_config: Dict[str, Any],
        descriptor_builder: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.node_id = node_id
        self.miners_config = miners_config
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self._descriptor_builder = descriptor_builder

        self._log_queue: Optional[multiprocessing.Queue] = None
        self._log_listener: Optional[QueueListener] = None
        self._setup_multiprocess_logging()

        self.logger = init_component_logger("miner-core", node_id)

        self.miner_handles: List[MinerHandle] = []
        self._initialize_miners(cfg=miners_config)

        # Aggregate stats across all handles. The controller fills these in
        # via `record_dispatch` / `record_result` callbacks — keeping the
        # core itself free of chain knowledge.
        self.timing_stats: Dict[str, Any] = {
            "total_attempts": 0,
            "total_wins": 0,
            "total_mining_time": 0.0,
            "wins_per_miner": {},
        }

        self._summary_miners = [(h.miner_id, h.miner_type) for h in self.miner_handles]

        self.logger.info(
            "MinerCore %s initialized with %d handles", node_id, len(self.miner_handles)
        )
        for h in self.miner_handles:
            self.logger.debug("  - %s (%s)", h.miner_id, h.miner_type)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down all workers and the log listener. Idempotent."""
        for h in self.miner_handles:
            try:
                h.cancel()
                h.req.put({"op": "shutdown"})
            except Exception as exc:  # noqa: BLE001 — best-effort tear-down
                self.logger.warning(
                    "close: shutdown signal failed for %s: %s", h.miner_id, exc
                )
        for h in self.miner_handles:
            try:
                h.proc.join(timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("close: join failed for %s: %s", h.miner_id, exc)
            if h.proc.is_alive():
                try:
                    h.proc.terminate()
                    h.proc.join(timeout=2.0)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "close: terminate failed for %s: %s", h.miner_id, exc
                    )
                if h.proc.is_alive():
                    try:
                        h.proc.kill()
                        h.proc.join(timeout=1.0)
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning(
                            "close: kill failed for %s: %s", h.miner_id, exc
                        )
        self.miner_handles = []

        if self._log_listener is not None:
            try:
                self._log_listener.stop()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("close: log listener stop failed: %s", exc)
            self._log_listener = None

    # ------------------------------------------------------------------
    # Telemetry surface (preserved across the v0.1 -> v0.2 refactor)
    # ------------------------------------------------------------------

    def descriptor(self) -> Dict[str, Any]:
        """Hardware descriptor, cached after the first build.

        Resilient to descriptor-builder failures: returns `{"error": ...}`
        so `/api/v1/system` keeps returning JSON even when the underlying
        platform probe (CUDA / Metal / DWave) misbehaves. Mirrors the
        resilience added to `Node.descriptor()` in commit 1859a35.
        """
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        if self._descriptor_builder is None:
            self._descriptor_cache = {
                "node_id": self.node_id,
                "miners": [
                    {"id": mid, "type": mtype} for mid, mtype in self._summary_miners
                ],
            }
            return self._descriptor_cache
        try:
            self._descriptor_cache = self._descriptor_builder()
        except Exception as exc:  # noqa: BLE001 — surface but don't fail
            self.logger.exception("descriptor builder failed: %s", exc)
            self._descriptor_cache = {
                "node_id": self.node_id,
                "error": f"{type(exc).__name__}: {exc}",
                "miners": [
                    {"id": mid, "type": mtype} for mid, mtype in self._summary_miners
                ],
            }
        return self._descriptor_cache

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate timing + win counters across all miners.

        Mirrors the shape used by `Node.get_stats()` so the existing
        telemetry REST API (`/api/v1/stats`) keeps returning the same JSON
        once Phase 5b rewires its backend.
        """
        attempts = self.timing_stats["total_attempts"]
        wins = self.timing_stats["total_wins"]
        win_rate = (wins / attempts) if attempts else 0.0
        avg_mining_time = (
            self.timing_stats["total_mining_time"] / attempts if attempts else 0.0
        )
        return {
            "node_id": self.node_id,
            "total_blocks_attempted": attempts,
            "total_blocks_won": wins,
            "win_rate": win_rate,
            "total_mining_time": self.timing_stats["total_mining_time"],
            "avg_mining_time": avg_mining_time,
            "wins_per_miner": dict(self.timing_stats["wins_per_miner"]),
            "miners": [
                {"id": mid, "type": mtype} for mid, mtype in self._summary_miners
            ],
        }

    def record_dispatch(self) -> None:
        """Called by the controller after dispatching a context to handles."""
        self.timing_stats["total_attempts"] += 1

    def record_result(
        self,
        *,
        winning_miner_id: str,
        mining_time: float,
    ) -> None:
        """Called by the controller after a result is accepted on chain."""
        self.timing_stats["total_wins"] += 1
        self.timing_stats["total_mining_time"] += mining_time
        wpm = self.timing_stats["wins_per_miner"]
        wpm[winning_miner_id] = wpm.get(winning_miner_id, 0) + 1

    # ------------------------------------------------------------------
    # Setup helpers (forked from Node._initialize_miners)
    # ------------------------------------------------------------------

    def _setup_multiprocess_logging(self) -> None:
        self._log_queue = multiprocessing.Queue()
        root_logger = logging.getLogger()
        handlers = list(root_logger.handlers)
        if handlers:
            self._log_listener = QueueListener(
                self._log_queue, *handlers, respect_handler_level=True
            )
            self._log_listener.start()
        else:
            logging.getLogger(__name__).warning(
                "MinerCore: root logger has no handlers; child-process logs will be "
                "discarded. Call setup_logging() before constructing MinerCore."
            )

    def _initialize_miners(self, cfg: Dict[str, Any]) -> None:
        # CPU section. `cfg["cpu"]["num_cpus"]` defaults to 1 if present.
        # `cfg["cpu"]["args"]` (optional) is forwarded to the miner — used
        # by the substrate CLI to pass `topology=zephyr(m, t)` so the
        # sampler binds to the chain's registered topology.
        cpu_cfg = cfg.get("cpu")
        if cpu_cfg is not None:
            num_cpus = int(cpu_cfg.get("num_cpus", 1))
            cpu_args = dict(cpu_cfg.get("args", {}))
            for i in range(num_cpus):
                spec = {
                    "id": f"{self.node_id}-CPU-{i + 1}",
                    "kind": "cpu",
                    "args": cpu_args,
                }
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # GPU section. Forked from Node's normalization path; we keep only
        # the device-section parsing the cuda/metal/modal kinds care about.
        has_gpu = cfg.get("gpu") is not None or any(
            cfg.get(k) is not None for k in _GPU_DEVICE_SECTIONS
        )
        if has_gpu:
            for spec in _build_gpu_specs(self.node_id, cfg):
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # QPU section. Same forked pattern.
        has_qpu = cfg.get("qpu") is not None or any(
            cfg.get(k) is not None for k in _QPU_DEVICE_SECTIONS
        )
        if has_qpu:
            for spec in _build_qpu_specs(self.node_id, cfg):
                self.miner_handles.append(MinerHandle(spec, self._log_queue))


# ----------------------------------------------------------------------
# Spec builders (extracted so the CLI can use them without instantiating
# the full MinerCore)
# ----------------------------------------------------------------------


def _build_gpu_specs(node_id: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Yield miner specs for each GPU device in the config.

    Defers the normalization to `shared.node` until Phase 5b deletes that
    module; once it's gone, the normalization moves here. Keeping the
    delegation in place now means cfg semantics stay identical with the
    legacy code path.
    """
    from shared.node import (  # local import: legacy module deleted in Phase 5b
        _build_gpu_miner_cfg,
        _normalize_gpu_config,
    )

    specs: List[Dict[str, Any]] = []
    gpu_cfg = _normalize_gpu_config(cfg)
    common_cfg = _build_gpu_miner_cfg(gpu_cfg)
    for dev in gpu_cfg.get("devices", []):
        if dev.get("enabled") is False:
            continue
        dev_type = dev["type"].lower()
        dev_cfg = _build_gpu_miner_cfg(dev, defaults=common_cfg)
        if dev_type == "cuda":
            device_id = dev.get("device", "0")
            specs.append(
                {
                    "id": f"{node_id}-GPU-CUDA-{device_id}",
                    "kind": "cuda",
                    "cfg": dev_cfg,
                    "args": {"device": str(device_id)},
                }
            )
        elif dev_type == "metal":
            specs.append(
                {
                    "id": f"{node_id}-GPU-MPS",
                    "kind": "metal",
                    "cfg": dev_cfg,
                    "args": {"device": "mps"},
                }
            )
        elif dev_type == "modal":
            gpu_type = dev.get("gpu_type", "t4")
            specs.append(
                {
                    "id": f"{node_id}-GPU-MODAL-{gpu_type}",
                    "kind": "modal",
                    "cfg": dev_cfg,
                    "args": {"gpu_type": str(gpu_type)},
                }
            )
        else:
            raise ValueError(f"Unknown GPU device type: {dev_type}")
    return specs


def _build_qpu_specs(node_id: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Yield miner specs for each QPU device in the config."""
    from shared.node import _normalize_qpu_config  # local import: see above

    specs: List[Dict[str, Any]] = []
    qpu_cfg = _normalize_qpu_config(cfg)
    for i, dev in enumerate(qpu_cfg.get("devices", []), start=1):
        dev_type = dev.get("type", "dwave").lower()
        tag = dev_type.upper()
        if dev_type == "dwave":
            specs.append(
                {
                    "id": f"{node_id}-QPU-{tag}-{i}",
                    "kind": "qpu",
                    "cfg": {
                        "qpu_type": "dwave",
                        "daily_budget": dev.get("daily_budget"),
                        "qpu_min_blocks_for_estimation": dev.get(
                            "qpu_min_blocks_for_estimation",
                            5,
                        ),
                        "qpu_ema_alpha": dev.get("qpu_ema_alpha", 0.3),
                    },
                }
            )
        elif dev_type in ("ibm", "braket", "pasqal", "ionq", "origin"):
            specs.append(
                {
                    "id": f"{node_id}-QPU-{tag}-{i}",
                    "kind": "qpu",
                    "cfg": {
                        "qpu_type": dev_type,
                        "token": dev.get("token"),
                        "daily_budget": dev.get("daily_budget"),
                    },
                }
            )
        else:
            raise ValueError(f"Unknown QPU device type: {dev_type}")
    return specs


__all__ = [
    "MinerCore",
]
