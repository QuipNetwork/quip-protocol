# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Node hardware survey and whitelisted config summary.

Builds a ``NodeDescriptor`` — a single JSON-friendly object that
merges probed hardware facts (CPU/GPU/memory/OS/runtime) with an
explicit whitelist of configured miner parameters. This is the
payload reported to peers on JOIN/STATUS/STATS_RESPONSE and
persisted in ``telemetry/nodes.json``.

The whitelist is the single source of truth for privacy posture.
Anything not in the whitelist — secrets, tokens, peer addresses,
internal listen IPs, TLS paths, heartbeat timings — never leaves
the node. A second-pass substring block catches whitelist typos.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from shared.version import PROTOCOL_VERSION, get_version


logger = logging.getLogger(__name__)

DESCRIPTOR_VERSION = 1

# ── Whitelists ────────────────────────────────────────────────

_GLOBAL_WHITELIST = frozenset({
    "node_name", "public_host", "public_port", "auto_mine", "log_level",
})

_CPU_WHITELIST = frozenset({"num_cpus"})

_GPU_DEV_WHITELIST = frozenset({
    "device_index", "utilization", "yielding", "enabled",
})

_MODAL_DEV_WHITELIST = frozenset({
    "gpu_type", "utilization", "yielding", "enabled",
})

_QPU_DEV_WHITELIST = frozenset({
    "solver", "daily_budget",
    "qpu_min_blocks_for_estimation", "qpu_ema_alpha",
})

# Case-insensitive substrings that ban a key from any telemetry
# output — even if a future contributor mistakenly whitelists it.
_FORBIDDEN_SUBSTRINGS = (
    "secret", "token", "api_key", "password",
    "access_token", "private", "credential", "auth",
)

# TOML section keys that map to GPU/QPU device types. Kept in
# sync with shared/node.py._GPU_DEVICE_SECTIONS / _QPU_DEVICE_SECTIONS.
_GPU_DEVICE_SECTIONS = {
    "cuda": "cuda",
    "nvidia": "cuda",
    "metal": "metal",
    "modal": "modal",
}

_QPU_DEVICE_SECTIONS = (
    "dwave", "ibm", "braket", "pasqal", "ionq", "origin",
)


# ── Dataclasses ───────────────────────────────────────────────

@dataclass
class CPUInfo:
    logical_cores: Optional[int]
    physical_cores: Optional[int]
    brand: str
    arch: str


@dataclass
class GPUInfo:
    index: int
    vendor: str
    name: str
    memory_mb: Optional[int] = None
    observed_utilization_pct: Optional[int] = None


@dataclass
class SystemInfo:
    os: Dict[str, str]
    cpu: CPUInfo
    memory_mb: Optional[int]
    gpus: List[GPUInfo] = field(default_factory=list)


@dataclass
class Runtime:
    python: str
    quip_version: str
    protocol_version: int
    in_docker: bool
    docker_image: Optional[str]


@dataclass
class NodeDescriptor:
    descriptor_version: int
    node_name: str
    public_host: Optional[str]
    public_port: Optional[int]
    auto_mine: bool
    log_level: str
    runtime: Runtime
    miners: Dict[str, Dict[str, Any]]
    system_info: SystemInfo

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-friendly dict (no dataclass leftovers)."""
        return _scrub(asdict(self))


# ── Scrubbing (defense in depth) ──────────────────────────────

def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return any(sub in lowered for sub in _FORBIDDEN_SUBSTRINGS)


def _scrub(value: Any) -> Any:
    """Recursively drop any dict keys matching ``_FORBIDDEN_SUBSTRINGS``."""
    if isinstance(value, dict):
        return {
            k: _scrub(v) for k, v in value.items()
            if not _is_forbidden_key(k)
        }
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


# ── Hardware probes (stdlib + pynvml) ─────────────────────────

def _run_text(cmd: List[str], timeout: float = 2.0) -> Optional[str]:
    """Run *cmd*, return stripped stdout, or None on any failure.

    Output is decoded with ``errors='replace'`` so commands that emit
    non-UTF-8 bytes (e.g. macOS ``ioreg -l``) don't crash the probe.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True,
            timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    decoded = result.stdout.decode("utf-8", errors="replace").strip()
    return decoded or None


def _cpu_brand() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    if system == "Linux":
        brand = _linux_cpu_brand()
        if brand:
            return brand
    return platform.processor() or "unknown"


def _linux_cpu_brand() -> Optional[str]:
    """Extract a human CPU name from /proc/cpuinfo or device-tree.

    x86 uses ``model name``; ARM lacks it and exposes ``Hardware``,
    ``Model``, or the device-tree node ``/sys/firmware/devicetree/base/model``.
    Emulated containers may expose only ``CPU implementer`` + ``CPU part`` —
    we decode the implementer byte to a vendor label as a last resort.
    """
    name_keys = ("model name", "Model", "Hardware", "cpu model")
    implementer: Optional[str] = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                for key in name_keys:
                    if line.startswith(key):
                        _, _, rest = line.partition(":")
                        name = rest.strip().rstrip("\x00")
                        if name:
                            return name
                if line.startswith("CPU implementer") and implementer is None:
                    _, _, rhs = line.partition(":")
                    implementer = rhs.strip().lower()
    except OSError:
        pass
    try:
        with open("/sys/firmware/devicetree/base/model", "rb") as fh:
            raw = fh.read().decode("utf-8", errors="replace").rstrip("\x00").strip()
            if raw:
                return raw
    except OSError:
        pass
    if implementer:
        vendor = _ARM_CPU_IMPLEMENTERS.get(implementer)
        if vendor:
            return f"{vendor} {platform.machine()}"
    return None


# ARM `CPU implementer` byte → vendor label (from Linux arch/arm64 docs).
_ARM_CPU_IMPLEMENTERS = {
    "0x41": "ARM Ltd", "0x42": "Broadcom", "0x43": "Cavium",
    "0x44": "DEC", "0x46": "Fujitsu", "0x48": "HiSilicon",
    "0x49": "Infineon", "0x4d": "Motorola/Freescale",
    "0x4e": "NVIDIA", "0x50": "Ampere", "0x51": "Qualcomm",
    "0x53": "Samsung", "0x56": "Marvell", "0x61": "Apple",
    "0x66": "Faraday", "0x69": "Intel", "0xc0": "Ampere Computing",
}


def _cpu_physical_cores() -> Optional[int]:
    system = platform.system()
    if system == "Darwin":
        out = _run_text(["sysctl", "-n", "hw.physicalcpu"])
        try:
            return int(out) if out else None
        except ValueError:
            return None
    if system == "Linux":
        try:
            pairs = set()
            phys_id = core_id = None
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("physical id"):
                        phys_id = line.partition(":")[2].strip()
                    elif line.startswith("core id"):
                        core_id = line.partition(":")[2].strip()
                    elif not line.strip():
                        if phys_id is not None and core_id is not None:
                            pairs.add((phys_id, core_id))
                        phys_id = core_id = None
            return len(pairs) or None
        except OSError:
            return None
    if system == "Windows":
        return _windows_physical_cores()
    return None


def _windows_physical_cores() -> Optional[int]:
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    # GetLogicalProcessorInformation: call twice (size probe, then data).
    needed = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(needed))
    if needed.value == 0:
        return None
    buf = (ctypes.c_byte * needed.value)()
    if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(needed)):
        return None
    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION is complex; we can't parse the
    # buffer without defining the full struct. Accept best-effort None here
    # and let logical_cores stand in on Windows.
    return None


def _memory_mb() -> Optional[int]:
    system = platform.system()
    if system == "Darwin":
        out = _run_text(["sysctl", "-n", "hw.memsize"])
        try:
            return int(out) // (1024 * 1024) if out else None
        except ValueError:
            return None
    if system == "Linux":
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) // 1024
        except (OSError, ValueError):
            return None
    if system == "Windows":
        return _windows_memory_mb()
    return None


def _windows_memory_mb() -> Optional[int]:
    try:
        class MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemStatus()
        stat.dwLength = ctypes.sizeof(MemStatus)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return stat.ullTotalPhys // (1024 * 1024)
    except (AttributeError, OSError):
        return None


def _nvidia_gpus() -> List[GPUInfo]:
    """Enumerate NVIDIA GPUs via NVML (nvidia-ml-py). Empty on any failure."""
    try:
        import pynvml
    except ImportError:
        return []
    nvml_error = getattr(pynvml, "NVMLError", Exception)
    try:
        pynvml.nvmlInit()
    except nvml_error as exc:
        logger.debug("NVML init failed: %s", exc)
        return []
    gpus: List[GPUInfo] = []
    count = 0
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = _nvml_decode(pynvml.nvmlDeviceGetName(handle))
            except nvml_error as exc:
                logger.debug("NVML probe failed for GPU %d: %s", i, exc)
                continue
            mem_mb: Optional[int] = None
            util: Optional[int] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_mb = int(mem.total) // (1024 * 1024)
            except nvml_error as exc:
                logger.debug("NVML memory probe failed for GPU %d: %s", i, exc)
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util = int(rates.gpu)
            except nvml_error as exc:
                logger.debug("NVML util probe failed for GPU %d: %s", i, exc)
            gpus.append(GPUInfo(
                index=i, vendor="NVIDIA", name=name,
                memory_mb=mem_mb, observed_utilization_pct=util,
            ))
    except nvml_error as exc:
        logger.debug("NVML enumeration failed: %s", exc)
    finally:
        try:
            pynvml.nvmlShutdown()
        except nvml_error as exc:
            logger.debug("NVML shutdown failed: %s", exc)
    if count and len(gpus) < count:
        logger.warning(
            "NVML reported %d GPUs but only %d were probed successfully",
            count, len(gpus),
        )
    return gpus


def _nvml_decode(value: Any) -> str:
    """NVML sometimes returns bytes, sometimes str. Normalize to str."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else "unknown"


def _apple_gpu() -> Optional[GPUInfo]:
    """Probe Apple Silicon GPU name + core count via ioreg/system_profiler."""
    if platform.system() != "Darwin":
        return None
    name = _apple_gpu_name() or "Apple GPU"
    cores = _apple_gpu_core_count()
    if cores is not None:
        name = f"{name} ({cores}-core GPU)"
    return GPUInfo(index=0, vendor="Apple", name=name)


def _apple_gpu_name() -> Optional[str]:
    """Return the Apple GPU chipset model (e.g., 'Apple M2 Max') or None."""
    import json as _json
    out = _run_text(
        ["system_profiler", "-json", "SPDisplaysDataType"],
        timeout=4.0,
    )
    if not out:
        return None
    try:
        data = _json.loads(out)
    except ValueError:
        return None
    displays = data.get("SPDisplaysDataType") or []
    for entry in displays:
        name = entry.get("sppci_model") or entry.get("_name")
        if isinstance(name, str) and name:
            return name
    return None


def _apple_gpu_core_count() -> Optional[int]:
    """Parse GPU core count from ``ioreg -l`` output."""
    out = _run_text(["ioreg", "-l"], timeout=2.0)
    if not out:
        return None
    for line in out.splitlines():
        if "gpu-core-count" in line and "=" in line:
            _, _, rhs = line.partition("=")
            try:
                return int(rhs.strip())
            except ValueError:
                continue
    return None


def _is_docker() -> bool:
    """Return True when running inside a container (heuristic)."""
    return os.path.exists("/.dockerenv") or "container" in os.environ


def _docker_image() -> Optional[str]:
    """Container image identity from the ``QUIP_DOCKER_IMAGE`` env var."""
    return os.environ.get("QUIP_DOCKER_IMAGE") or None


# ── Aggregators ───────────────────────────────────────────────

def collect_system_info() -> SystemInfo:
    """Probe the host: OS, CPU, memory, and GPUs."""
    uname = platform.uname()
    os_info = {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
    }
    cpu = CPUInfo(
        logical_cores=os.cpu_count(),
        physical_cores=_cpu_physical_cores(),
        brand=_cpu_brand(),
        arch=uname.machine,
    )
    gpus = list(_nvidia_gpus())
    apple = _apple_gpu()
    if apple is not None:
        apple.index = len(gpus)
        gpus.append(apple)
    return SystemInfo(
        os=os_info, cpu=cpu, memory_mb=_memory_mb(), gpus=gpus,
    )


def _runtime() -> Runtime:
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return Runtime(
        python=py,
        quip_version=get_version(),
        protocol_version=PROTOCOL_VERSION,
        in_docker=_is_docker(),
        docker_image=_docker_image(),
    )


def _global_summary(miners_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract top-level [global] fields that are safe to announce."""
    global_cfg = miners_config.get("global")
    if isinstance(global_cfg, dict):
        source = global_cfg
    else:
        # When loaded via CLI instead of TOML, flat keys live at the
        # top level of miners_config directly.
        source = miners_config
    return {
        k: source[k] for k in _GLOBAL_WHITELIST
        if k in source and not _is_forbidden_key(k)
    }


def _cpu_miner_entry(node_id: str, cpu_section: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "kind": "CPU",
        "miner_id": f"{node_id}-CPU-1",
    }
    for k in _CPU_WHITELIST:
        if k in cpu_section and not _is_forbidden_key(k):
            entry[k] = cpu_section[k]
    return entry


def _gpu_miner_entry(
    node_id: str,
    dev: Dict[str, Any],
    metal_idx: int,
    modal_idx: int,
) -> Optional[tuple]:
    """Return ``(key, entry)`` for one normalized GPU device dict, or None."""
    dev_type = str(dev.get("type", "")).lower()
    if dev_type == "cuda":
        device_id = str(dev.get("device", "0"))
        entry: Dict[str, Any] = {
            "kind": "GPU",
            "backend": "cuda",
            "miner_id": f"{node_id}-GPU-CUDA-{device_id}",
        }
        try:
            entry["device_index"] = int(device_id)
        except ValueError:
            pass
        for k in _GPU_DEV_WHITELIST:
            if k in dev and not _is_forbidden_key(k):
                entry[k] = dev[k]
        return f"cuda.{device_id}", entry

    if dev_type == "metal":
        entry = {
            "kind": "GPU",
            "backend": "metal",
            "miner_id": f"{node_id}-GPU-MPS",
            "device_index": metal_idx,
        }
        for k in _GPU_DEV_WHITELIST:
            if k in dev and not _is_forbidden_key(k):
                entry[k] = dev[k]
        return f"metal.{metal_idx}", entry

    if dev_type == "modal":
        gpu_type = str(dev.get("gpu_type", "t4"))
        entry = {
            "kind": "GPU",
            "backend": "modal",
            "miner_id": f"{node_id}-GPU-MODAL-{gpu_type}",
        }
        for k in _MODAL_DEV_WHITELIST:
            if k in dev and not _is_forbidden_key(k):
                entry[k] = dev[k]
        return f"modal.{modal_idx}", entry

    return None


def _qpu_miner_entry(
    node_id: str, dev: Dict[str, Any], sequence: int,
) -> Optional[tuple]:
    provider = str(dev.get("type", "")).lower()
    if provider not in _QPU_DEVICE_SECTIONS:
        return None
    entry: Dict[str, Any] = {
        "kind": "QPU",
        "provider": provider,
        "miner_id": f"{node_id}-QPU-{provider.upper()}-{sequence}",
    }
    for k in _QPU_DEV_WHITELIST:
        if k in dev and not _is_forbidden_key(k):
            entry[k] = dev[k]
    return provider, entry


def summarize_miners(
    node_id: str, miners_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build the whitelisted ``miners`` dict for the descriptor.

    Delegates to ``shared.node._normalize_gpu_config`` and
    ``_normalize_qpu_config`` for parity with the real miner
    initialization path.
    """
    from shared.node import _normalize_gpu_config, _normalize_qpu_config

    result: Dict[str, Dict[str, Any]] = {}

    if miners_config.get("cpu") is not None:
        cpu_section = miners_config["cpu"] or {}
        result["cpu"] = _cpu_miner_entry(node_id, cpu_section)

    has_gpu = (
        miners_config.get("gpu") is not None
        or any(miners_config.get(k) is not None for k in _GPU_DEVICE_SECTIONS)
    )
    if has_gpu:
        gpu_cfg = _normalize_gpu_config(miners_config)
        metal_i = modal_i = 0
        for dev in gpu_cfg.get("devices", []):
            out = _gpu_miner_entry(node_id, dev, metal_i, modal_i)
            if out is None:
                continue
            key, entry = out
            result[key] = entry
            if entry["backend"] == "metal":
                metal_i += 1
            elif entry["backend"] == "modal":
                modal_i += 1

    has_qpu = (
        miners_config.get("qpu") is not None
        or any(miners_config.get(k) is not None for k in _QPU_DEVICE_SECTIONS)
    )
    if has_qpu:
        qpu_cfg = _normalize_qpu_config(miners_config)
        provider_seq: Dict[str, int] = {}
        for dev in qpu_cfg.get("devices", []):
            provider = str(dev.get("type", "")).lower()
            seq = provider_seq.get(provider, 0) + 1
            provider_seq[provider] = seq
            out = _qpu_miner_entry(node_id, dev, seq)
            if out is None:
                continue
            key, entry = out
            result[key] = entry

    return {k: _scrub(v) for k, v in result.items()}


def build_descriptor(
    node_id: str,
    miners_config: Dict[str, Any],
    system_info: Optional[SystemInfo] = None,
) -> NodeDescriptor:
    """Assemble a ``NodeDescriptor`` from node id + raw miners_config."""
    sysinfo = system_info if system_info is not None else collect_system_info()
    globals_ = _global_summary(miners_config)
    return NodeDescriptor(
        descriptor_version=DESCRIPTOR_VERSION,
        node_name=globals_.get("node_name") or node_id,
        public_host=globals_.get("public_host"),
        public_port=globals_.get("public_port"),
        auto_mine=bool(globals_.get("auto_mine", False)),
        log_level=str(globals_.get("log_level", "INFO")),
        runtime=_runtime(),
        miners=summarize_miners(node_id, miners_config),
        system_info=sysinfo,
    )


def override_public_address(
    descriptor: Optional[Dict[str, Any]],
    validated_address: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Replace ``descriptor.public_host/public_port`` with the receiver's view.

    The sender's self-reported ``public_host`` / ``public_port`` are never
    trusted by downstream consumers; the receiver validates the peer
    address (see ``_validate_peer_address`` in network_node) and injects
    the result here. A working DNS name survives validation (because it
    was reachable), while claimed IPs get replaced with the observed IP.

    Args:
        descriptor: A descriptor dict (may be ``None``).
        validated_address: ``host:port`` string produced by peer-address
            validation. Pass ``None`` to leave the descriptor unchanged.

    Returns:
        A new descriptor dict with patched connection fields, or the
        original ``descriptor`` if ``validated_address`` is falsy.
    """
    if not descriptor or not validated_address:
        return descriptor
    from shared.address_utils import parse_host_port
    try:
        host, port = parse_host_port(validated_address)
    except ValueError as exc:
        logger.warning(
            "override_public_address: rejecting descriptor, could not parse "
            "validated address %r: %s",
            validated_address, exc,
        )
        return None
    patched = dict(descriptor)
    patched["public_host"] = host
    patched["public_port"] = port
    return patched


def descriptor_summary_line(descriptor: NodeDescriptor) -> str:
    """One-line human-readable summary for the nodes table log."""
    sysinfo = descriptor.system_info
    os_part = f"{sysinfo.os.get('system', '?')}/{sysinfo.os.get('machine', '?')}"
    cpu_cores = sysinfo.cpu.logical_cores or "?"
    gpu_parts: List[str] = []
    if sysinfo.gpus:
        counts: Dict[str, int] = {}
        for gpu in sysinfo.gpus:
            counts[gpu.name] = counts.get(gpu.name, 0) + 1
        gpu_parts = [
            f"{n}×{name}" if n > 1 else name
            for name, n in counts.items()
        ]
    kinds = sorted({
        str(m.get("kind", "")).lower() for m in descriptor.miners.values()
    })
    miners_part = "+".join(k for k in kinds if k)
    pieces = [os_part, f"{cpu_cores}C"]
    if gpu_parts:
        pieces.append(" ".join(gpu_parts))
    if miners_part:
        pieces.append(miners_part)
    return " ".join(pieces)
