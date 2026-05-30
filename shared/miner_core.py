"""Miner orchestration state, extracted from `shared.node.Node` for Phase 5.

`MinerCore` owns the miner-side machinery that survives the v0.1 → v0.2
refactor: the persistent `MinerHandle` worker processes, the multiprocessing
log queue, the hardware descriptor cache, and the aggregate timing stats.
Everything Node was responsible for that pertains to chain state, P2P, or
SPHINCS+ block signing has been removed — the substrate controller owns
those concerns now.

Lifecycle:

    core = MinerCore(node_id="miner-1", miners_config={"cpu": {"num_cpus": 4}})
    controller = SubstrateMinerController(pool, signer, core.miner_handles)
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
import logging.handlers
import multiprocessing
from typing import Any, Callable, Dict, List, Optional

from shared.logging_config import init_component_logger
from shared.miner_worker import MinerHandle


# Config-section discovery + normalization helpers. Moved here in Phase 5b
# from `shared.node` (now deleted). The normalization rules cover the TOML
# shapes operators were using for the legacy CLI: `[gpu]` + `[cuda.0]` /
# `[cuda.1]` subtables for CUDA per-device config, `[metal]` / `[modal]`
# single-table sections, and `[[cuda]]` / `[[metal]]` array-of-tables for
# the substrate CLI path that wants to pass a single device dict directly.

_GPU_CFG_KEYS = (
    "utilization", "yielding", "enabled", "sms_per_nonce",
)

_GPU_DEVICE_SECTIONS = {
    "cuda": "cuda",
    "nvidia": "cuda",   # alias
    "metal": "metal",
    "modal": "modal",
}

_QPU_DEVICE_SECTIONS = (
    "dwave", "ibm", "braket", "pasqal", "ionq", "origin",
)


def _build_gpu_miner_cfg(
    section: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a miner cfg dict from a TOML section, inheriting `defaults`."""
    base = dict(defaults) if defaults else {}
    for key in _GPU_CFG_KEYS:
        if key in section:
            base[key] = section[key]
    return base


def _normalize_gpu_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the TOML GPU layout into `{"devices": [...]}` form."""
    gpu_cfg = dict(cfg.get("gpu") or {})
    devices: List[Dict[str, Any]] = []
    for section_key, dev_type in _GPU_DEVICE_SECTIONS.items():
        section = cfg.get(section_key)
        if section is None:
            continue
        if dev_type in ("cuda", "modal") and isinstance(section, dict):
            if any(isinstance(v, dict) for v in section.values()):
                for dev_id in sorted(section.keys()):
                    sub = section[dev_id]
                    if not isinstance(sub, dict):
                        continue
                    entry: Dict[str, Any] = {"type": dev_type}
                    if dev_type == "cuda":
                        entry["device"] = str(dev_id)
                    entry.update(sub)
                    devices.append(entry)
            else:
                entry = {"type": dev_type}
                entry.update(section)
                devices.append(entry)
        elif isinstance(section, list):
            for item in section:
                entry = {"type": dev_type}
                entry.update(item)
                devices.append(entry)
        elif isinstance(section, dict):
            entry = {"type": dev_type}
            entry.update(section)
            devices.append(entry)
    if devices:
        gpu_cfg["devices"] = devices
    return gpu_cfg


def _normalize_qpu_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the TOML QPU layout into `{"devices": [...]}` form."""
    qpu_cfg = dict(cfg.get("qpu") or {})
    devices: List[Dict[str, Any]] = []
    for section_key in _QPU_DEVICE_SECTIONS:
        section = cfg.get(section_key)
        if section is None:
            continue
        if isinstance(section, list):
            for item in section:
                entry: Dict[str, Any] = {"type": section_key}
                entry.update(item)
                devices.append(entry)
        elif isinstance(section, dict):
            if any(isinstance(v, dict) for v in section.values()):
                for sub_id in sorted(section.keys()):
                    sub = section[sub_id]
                    if not isinstance(sub, dict):
                        continue
                    entry = {"type": section_key}
                    entry.update(sub)
                    devices.append(entry)
            else:
                entry = {"type": section_key}
                entry.update(section)
                devices.append(entry)
    if devices:
        qpu_cfg["devices"] = devices
    return qpu_cfg


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
        topology: Any = None,
    ) -> None:
        self.node_id = node_id
        self.miners_config = miners_config
        self._topology = topology
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self._descriptor_builder = descriptor_builder

        self._log_queue: Optional[multiprocessing.Queue] = None
        self._log_proc: Optional[multiprocessing.Process] = None
        self._log_stop: Optional[multiprocessing.Event] = None  # type: ignore[type-arg]
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

        if getattr(self, "_log_proc", None) is not None:
            try:
                self._log_stop.set()
                self._log_queue.put(None)
                from shared.proc_util import terminate_join
                terminate_join(self._log_proc, 3.0)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("close: log writer stop failed: %s", exc)
            self._log_proc = None

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
        ctx = multiprocessing.get_context("spawn")
        self._log_queue = ctx.Queue()
        root_logger = logging.getLogger()
        handlers = list(root_logger.handlers)
        if not handlers:
            self._log_proc = None
            self._log_stop = None
            logging.getLogger(__name__).warning(
                "MinerCore: root logger has no handlers; child-process logs will be "
                "discarded. Call setup_logging() before constructing MinerCore."
            )
            return
        # Capture the file path + level so the writer process can recreate the
        # real handlers and become the SOLE owner of the file.
        level = root_logger.level or logging.INFO
        log_file_path = None
        for h in handlers:
            if isinstance(h, logging.FileHandler):  # RotatingFileHandler subclasses it
                log_file_path = h.baseFilename
                if h.level:
                    level = h.level
        from shared.logging_config import log_writer_main
        from shared.proc_util import spawn_worker
        self._log_stop = ctx.Event()
        self._log_proc = spawn_worker(
            log_writer_main,
            (self._log_queue, self._log_stop, log_file_path, level),
            name="log-writer",
        )
        # Route THIS process's logs through the queue too, so the writer owns
        # the file alone (no double-write / rotation race). Workers already
        # attach a QueueHandler to the same queue in miner_worker.
        for h in handlers:
            root_logger.removeHandler(h)
        root_logger.addHandler(logging.handlers.QueueHandler(self._log_queue))

    def _initialize_miners(self, cfg: Dict[str, Any]) -> None:
        # A topology is mandatory whenever we will build any handle — the chain
        # topology must reach every miner. None means it was not wired; fail
        # fast rather than silently mine the wrong (default) topology.
        cpu_cfg = cfg.get("cpu")
        has_gpu = cfg.get("gpu") is not None or any(
            cfg.get(k) is not None for k in _GPU_DEVICE_SECTIONS
        )
        has_qpu = cfg.get("qpu") is not None or any(
            cfg.get(k) is not None for k in _QPU_DEVICE_SECTIONS
        )
        if (cpu_cfg is not None or has_gpu or has_qpu) and self._topology is None:
            raise ValueError(
                "MinerCore requires a topology to build miners; none was "
                "provided (chain topology was not wired through)"
            )

        # CPU section. `cfg["cpu"]["num_cpus"]` defaults to 1 if present.
        # `cfg["cpu"]["args"]` (optional) is forwarded to the miner — used
        # by the substrate CLI to pass `topology=zephyr(m, t)` so the
        # sampler binds to the chain's registered topology.
        if cpu_cfg is not None:
            num_cpus = int(cpu_cfg.get("num_cpus", 1))
            cpu_args = dict(cpu_cfg.get("args", {}))
            for i in range(num_cpus):
                spec = {
                    "id": f"{self.node_id}-CPU-{i + 1}",
                    "kind": "cpu",
                    "args": cpu_args,
                }
                self._attach_topology(spec)
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # GPU section. Forked from Node's normalization path; we keep only
        # the device-section parsing the cuda/metal/modal kinds care about.
        if has_gpu:
            for spec in _build_gpu_specs(self.node_id, cfg):
                self._attach_topology(spec)
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

        # QPU section. Same forked pattern.
        if has_qpu:
            for spec in _build_qpu_specs(self.node_id, cfg):
                self._attach_topology(spec)
                self.miner_handles.append(MinerHandle(spec, self._log_queue))

    def _attach_topology(self, spec: Dict[str, Any]) -> None:
        """Stash the chain topology into a spec's ``args`` so every backend's
        ``build_miner_from_spec`` receives it as ``topology=`` (the qpu spec,
        which has no ``args`` by default, gains one)."""
        spec.setdefault("args", {})["topology"] = self._topology


# ----------------------------------------------------------------------
# Spec builders (extracted so the CLI can use them without instantiating
# the full MinerCore)
# ----------------------------------------------------------------------


def _build_gpu_specs(node_id: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Yield miner specs for each GPU device in the config."""
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
    specs: List[Dict[str, Any]] = []
    qpu_cfg = _normalize_qpu_config(cfg)
    for i, dev in enumerate(qpu_cfg.get("devices", []), start=1):
        dev_type = dev.get("type", "dwave").lower()
        tag = dev_type.upper()
        if dev_type == "dwave":
            cfg_block: Dict[str, Any] = {
                "qpu_type": "dwave",
                "daily_budget": dev.get("daily_budget"),
                "qpu_min_blocks_for_estimation": dev.get(
                    "qpu_min_blocks_for_estimation",
                    5,
                ),
                "qpu_ema_alpha": dev.get("qpu_ema_alpha", 0.3),
            }
            # TOML keys flow into the cfg dict 1:1 — operator-facing
            # names match what they put in the file. The
            # build_miner_from_spec boundary translates `solver` →
            # `solver_name` (DWaveMiner's kwarg) before instantiation.
            # The descriptor scrubber's whitelist
            # (`_QPU_HANDLE_FIELD_WHITELIST = {"solver", "daily_budget"}`)
            # surfaces `solver` to dashboards directly. `token` flows
            # through to DWaveSampler so a TOML-set token works without
            # also requiring DWAVE_API_KEY env.
            if dev.get("solver") is not None:
                cfg_block["solver"] = dev["solver"]
            if dev.get("region") is not None:
                cfg_block["region"] = dev["region"]
            if dev.get("token") is not None:
                cfg_block["token"] = dev["token"]
            # Throughput-tuning overrides: flow through to DWaveMiner
            # __init__ as kwargs of the same name. Absent => use the
            # hardcoded quality defaults in ``_adapt_mining_params``.
            if dev.get("num_reads") is not None:
                cfg_block["num_reads"] = dev["num_reads"]
            if dev.get("annealing_time_us") is not None:
                cfg_block["annealing_time_us"] = dev["annealing_time_us"]
            specs.append(
                {
                    "id": f"{node_id}-QPU-{tag}-{i}",
                    "kind": "qpu",
                    "cfg": cfg_block,
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
