# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Node hardware survey, descriptor builder, and canonical serialization.

Builds a ``NodeDescriptor`` — a single JSON-friendly object that merges
probed hardware facts (CPU/GPU/memory/OS/runtime) with a per-miner
summary derived from ``MinerCore.miner_handles``. The output schema
matches the v0.1 ``telemetry/nodes.json`` entries so existing dashboards
continue to parse it without change.

The on-chain ``identify`` flow canonicalizes this descriptor to UTF-8
JSON (sorted keys, compact separators) and posts it as a signed
``System.remark_with_event`` so indexers can map AccountId → descriptor
without trusting any unsigned channel. ``schema`` is a discriminator
that lets remark-scanning indexers select the right parser.

The forbidden-substring scrubber is defense-in-depth: even if a future
contributor leaks a secret-bearing key through the per-miner whitelist,
the second pass drops anything whose name matches ``secret``, ``token``,
``api_key``, ``credential``, etc.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from shared.version import PROTOCOL_VERSION, get_version


if TYPE_CHECKING:
    from shared.miner_core import MinerCore
    from shared.miner_worker import MinerHandle


logger = logging.getLogger(__name__)

SCHEMA_NAME = "quip.node_descriptor.v1"
DESCRIPTOR_VERSION = 1

# Bounds for self-asserted fields. Enforced at validate_descriptor time
# so the CLI fails fast before submitting an oversized remark; the
# Phase 2 runtime storage will enforce the same limits via BoundedVec.
NODE_NAME_MAX_BYTES = 64
PUBLIC_HOST_MAX_BYTES = 253          # DNS hostname limit (RFC 1035).
RPC_ENDPOINT_MAX_BYTES = 256
RPC_ENDPOINTS_MAX_COUNT = 8
PUBLIC_PORT_MIN = 1
PUBLIC_PORT_MAX = 65535
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


# Per-handle field allow-lists. Kept minimal because the on-chain
# remark is space-constrained and dashboards only render the
# whitelisted fields.
_CPU_HANDLE_ARGS_WHITELIST = frozenset({"num_cpus"})

_GPU_HANDLE_ARGS_WHITELIST = frozenset({
    "device", "device_index", "utilization", "yielding", "enabled",
})

_MODAL_HANDLE_ARGS_WHITELIST = frozenset({
    "gpu_type", "utilization", "yielding", "enabled",
})

_QPU_HANDLE_FIELD_WHITELIST = frozenset({
    # Match the v0.1 ``nodes.json`` shape exactly: dashboards care about
    # the solver name and daily budget. Internal estimation knobs
    # (``qpu_min_blocks_for_estimation``, ``qpu_ema_alpha``) are operator
    # tuning, not identity — they don't leave the node. The throughput
    # overrides ``num_reads`` / ``annealing_time_us`` are deliberately
    # surfaced so dashboards can show which params each miner is running.
    "solver", "daily_budget", "num_reads", "annealing_time_us",
})

# Case-insensitive substrings that ban a key from any telemetry
# output — even if a future contributor whitelists it by mistake.
_FORBIDDEN_SUBSTRINGS = (
    "secret", "token", "api_key", "password",
    "access_token", "private", "credential", "auth",
)

# Value-level secret detection. The key-name scrub only blocks
# obviously-named fields; an operator who pastes a credential into a
# legitimately-named field (e.g. ``solver = "DWAVE_API_KEY=abc"``)
# would otherwise leak it. These patterns flag the common shapes so
# ``validate_descriptor`` can reject the payload before submission.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"DWAVE[_-]?API[_-]?(KEY|TOKEN)", re.IGNORECASE),
    re.compile(r"AWS[_-]?(ACCESS|SECRET)[_-]?KEY", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS access key id
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),          # OpenAI / Anthropic
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),  # JWT-ish
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S{8,}", re.IGNORECASE),
)

# Solver names follow stable D-Wave (and IBM/etc.) conventions: ASCII
# alphanumerics plus ``._-``. Anything outside that set is either a typo
# or someone smuggling a URL / credential through, so we drop it from
# the descriptor and warn rather than ship it on chain.
_SOLVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


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
    rpc_endpoints: List[str]
    auto_mine: bool
    log_level: str
    runtime: Runtime
    miners: Dict[str, Dict[str, Any]]
    system_info: Optional[SystemInfo]

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-friendly dict (no dataclass leftovers).

        The ``schema`` discriminator is prepended so remark-scanning
        indexers can route on it without first decoding the full body.
        """
        body: Dict[str, Any] = {"schema": SCHEMA_NAME}
        body.update(_scrub(asdict(self)))
        # asdict() on Optional[SystemInfo]=None emits ``null`` already;
        # drop the key entirely so the wire stays compact.
        if body.get("system_info") is None:
            body.pop("system_info", None)
        return body


# ── Scrubbing (defense in depth) ──────────────────────────────

def is_secret_key(key: str) -> bool:
    """True if ``key`` names a secret-bearing field (token, password, ...).

    Single source of truth for telemetry redaction across modules (the
    descriptor scrub here and the miner-survey config in
    ``shared.miner_survey``), so a credential can't leak through one path
    while being scrubbed on another.
    """
    lowered = key.lower()
    return any(sub in lowered for sub in _FORBIDDEN_SUBSTRINGS)


def _scrub(value: Any) -> Any:
    """Recursively drop any dict keys matching ``_FORBIDDEN_SUBSTRINGS``."""
    if isinstance(value, dict):
        return {
            k: _scrub(v) for k, v in value.items()
            if not is_secret_key(k)
        }
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


# ── Hardware probes (stdlib + pynvml) ─────────────────────────

def _run_text(cmd: List[str], timeout: float = 2.0) -> Optional[str]:
    """Run *cmd*, return stripped stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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


# ARM `CPU implementer` byte → vendor label (from Linux arch/arm64 docs).
_ARM_CPU_IMPLEMENTERS = {
    "0x41": "ARM Ltd", "0x42": "Broadcom", "0x43": "Cavium",
    "0x44": "DEC", "0x46": "Fujitsu", "0x48": "HiSilicon",
    "0x49": "Infineon", "0x4d": "Motorola/Freescale",
    "0x4e": "NVIDIA", "0x50": "Ampere", "0x51": "Qualcomm",
    "0x53": "Samsung", "0x56": "Marvell", "0x61": "Apple",
    "0x66": "Faraday", "0x69": "Intel", "0xc0": "Ampere Computing",
}


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
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: List[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _nvml_decode(pynvml.nvmlDeviceGetName(handle))
            mem_mb: Optional[int] = None
            util: Optional[int] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_mb = int(mem.total) // (1024 * 1024)
            except Exception:
                pass
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util = int(rates.gpu)
            except Exception:
                pass
            gpus.append(GPUInfo(
                index=i, vendor="NVIDIA", name=name,
                memory_mb=mem_mb, observed_utilization_pct=util,
            ))
    except Exception as exc:
        logger.debug("NVML probe failed: %s", exc)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
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
    out = _run_text(
        ["system_profiler", "-json", "SPDisplaysDataType"],
        timeout=4.0,
    )
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    displays = data.get("SPDisplaysDataType") or []
    for entry in displays:
        name = entry.get("sppci_model") or entry.get("_name")
        if isinstance(name, str) and name:
            return name
    return None


def _apple_gpu_core_count() -> Optional[int]:
    """Parse GPU core count from 'ioreg -l | grep gpu-core-count'."""
    try:
        result = subprocess.run(
            "ioreg -l | grep gpu-core-count",
            shell=True, capture_output=True, text=True, timeout=2,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
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


# ── Per-handle summarization (v0.2: drives off MinerCore handles) ──

def _spec_kind(spec: Dict[str, Any]) -> str:
    return str(spec.get("kind", "")).lower()


def filter_telemetry_fields(
    mapping: Dict[str, Any],
    whitelist=None,
) -> Dict[str, Any]:
    """Keep only safe, JSON-friendly entries from *mapping*.

    Args:
        mapping: Source dict to filter.
        whitelist: Optional collection of allowed keys.  When ``None`` every
            non-secret, JSON-safe key is kept; when provided only keys in
            the whitelist are kept.

    Returns:
        Filtered dict with no secret keys and no non-JSON-safe values.
    """
    out: Dict[str, Any] = {}
    for k, v in (mapping or {}).items():
        if not isinstance(k, str) or is_secret_key(k):
            continue
        if whitelist is not None and k not in whitelist:
            continue
        if not _is_jsonable(v):
            continue
        out[k] = v
    return out


def _filter_args(args: Dict[str, Any], whitelist) -> Dict[str, Any]:
    """Thin wrapper kept for internal call-sites; delegates to filter_telemetry_fields."""
    return filter_telemetry_fields(args, whitelist=whitelist)


def _is_jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_jsonable(v) for k, v in value.items()
        )
    return False


def _cpu_spec_entry(spec: Dict[str, Any], miner_id: str) -> Dict[str, Any]:
    args = spec.get("args") or {}
    entry: Dict[str, Any] = {
        "kind": "CPU",
        "miner_id": miner_id,
    }
    entry.update(_filter_args(args, _CPU_HANDLE_ARGS_WHITELIST))
    return entry


def _gpu_spec_entry(spec: Dict[str, Any], miner_id: str, kind: str) -> Dict[str, Any]:
    args = spec.get("args") or {}
    backend = "cuda-gibbs" if kind == "cuda-gibbs" else kind
    entry: Dict[str, Any] = {
        "kind": "GPU",
        "backend": backend,
        "miner_id": miner_id,
    }
    whitelist = (
        _MODAL_HANDLE_ARGS_WHITELIST if kind == "modal"
        else _GPU_HANDLE_ARGS_WHITELIST
    )
    # Apply generic whitelist fields first, then overwrite with kind-specific
    # coercions so the explicit values always win without a pop() dance.
    entry.update(_filter_args(args, whitelist))
    if kind in ("cuda", "cuda-gibbs"):
        device = args.get("device")
        if device is not None:
            entry["device_index"] = _coerce_int(device, default=device)
        entry.pop("device", None)
    elif kind == "metal":
        device = args.get("device")
        if device is not None:
            entry["device"] = str(device)
    elif kind == "modal":
        gpu_type = args.get("gpu_type")
        if gpu_type is not None:
            entry["gpu_type"] = str(gpu_type)
    return entry


def _qpu_spec_entry(spec: Dict[str, Any], miner_id: str) -> Dict[str, Any]:
    cfg = spec.get("cfg") or {}
    args = spec.get("args") or {}
    provider = str(cfg.get("qpu_type") or args.get("qpu_type") or "dwave").lower()
    entry: Dict[str, Any] = {
        "kind": "QPU",
        "provider": provider,
        "miner_id": miner_id,
    }
    for source in (cfg, args):
        for k in _QPU_HANDLE_FIELD_WHITELIST:
            if k in source and not is_secret_key(k):
                v = source[k]
                if _is_jsonable(v) and k not in entry:
                    entry[k] = v
    # ``solver`` is the most likely accidental credential-leak vector
    # (it's a free-form string the operator sets in TOML). Enforce a
    # strict ASCII identifier pattern; drop with a warning rather than
    # ship anything ambiguous on chain.
    solver = entry.get("solver")
    if solver is not None and not _is_safe_solver_name(solver):
        logger.warning(
            "dropping suspicious solver value from descriptor (miner_id=%s); "
            "expected ASCII identifier matching %s",
            miner_id, _SOLVER_NAME_PATTERN.pattern,
        )
        entry.pop("solver", None)
    return entry


def _is_safe_solver_name(value: Any) -> bool:
    """True iff ``value`` is a string matching the stable solver-name shape."""
    if not isinstance(value, str):
        return False
    if not _SOLVER_NAME_PATTERN.match(value):
        return False
    return not _looks_secretish(value)


def _looks_secretish(value: str) -> bool:
    """Heuristic match against well-known credential shapes."""
    return any(p.search(value) for p in _SECRET_VALUE_PATTERNS)


def _iter_string_values(node: Any):
    """Yield every string value reachable from ``node`` (depth-first)."""
    if isinstance(node, str):
        yield node
        return
    if isinstance(node, dict):
        for v in node.values():
            yield from _iter_string_values(v)
        return
    if isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_string_values(v)


def _coerce_int(value: Any, default: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize_miners_from_specs(
    specs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build the per-miner ``miners`` dict from a list of spec dicts.

    Each spec carries ``id``, ``kind``, optional ``args``/``cfg`` — same
    shape MinerHandle stores. Key layout mirrors v0.1 ``nodes.json``:

        - cpu                       (single CPU bucket)
        - cuda.<device_index>
        - cuda-gibbs.<index>        (counts within the kind)
        - metal.<index>
        - modal.<gpu_type>          (or modal.<gpu_type>.<n> for duplicates)
        - dwave                     (single bucket per QPU provider)
    """
    result: Dict[str, Dict[str, Any]] = {}
    metal_idx = 0
    gibbs_idx = 0
    modal_idx: Dict[str, int] = {}
    cpu_count = 0

    for spec in specs:
        kind = _spec_kind(spec)
        miner_id = str(spec.get("id", ""))
        if kind == "cpu":
            cpu_count += 1
            # First CPU handle defines the bucket; subsequent ones just
            # bump the count. Mirrors v0.1 ``nodes.json`` shape where the
            # CPU entry carries ``num_cpus`` instead of one entry per
            # worker — v0.2's MinerCore spawns one handle per CPU so we
            # re-aggregate here.
            result.setdefault("cpu", _cpu_spec_entry(spec, miner_id))
            continue
        if kind == "cuda":
            entry = _gpu_spec_entry(spec, miner_id, kind)
            device = entry.get("device_index", "0")
            result[f"cuda.{device}"] = entry
            continue
        if kind == "cuda-gibbs":
            entry = _gpu_spec_entry(spec, miner_id, kind)
            result[f"cuda-gibbs.{gibbs_idx}"] = entry
            gibbs_idx += 1
            continue
        if kind == "metal":
            entry = _gpu_spec_entry(spec, miner_id, kind)
            result[f"metal.{metal_idx}"] = entry
            metal_idx += 1
            continue
        if kind == "modal":
            entry = _gpu_spec_entry(spec, miner_id, kind)
            gpu_type = str(entry.get("gpu_type", "t4")).lower()
            seq = modal_idx.get(gpu_type, 0)
            modal_idx[gpu_type] = seq + 1
            result[f"modal.{gpu_type}" + (f".{seq}" if seq else "")] = entry
            continue
        if kind == "qpu":
            entry = _qpu_spec_entry(spec, miner_id)
            result.setdefault(entry["provider"], entry)
            continue
        # Unknown kind — preserve under its bare name so future backends
        # show up in dashboards before they ship.
        result[kind or "unknown"] = {
            "kind": kind.upper() or "UNKNOWN",
            "miner_id": miner_id,
        }

    if cpu_count and "cpu" in result:
        result["cpu"]["num_cpus"] = cpu_count

    return {k: _scrub(v) for k, v in result.items()}


def summarize_miners_from_handles(
    handles: List["MinerHandle"],
) -> Dict[str, Dict[str, Any]]:
    """Adapter for the in-process path: pass MinerHandle.spec through.

    Used by the live ``MinerCore``; the CLI ``identify`` flow uses
    ``summarize_miners_from_specs`` directly to avoid spawning workers.
    """
    return summarize_miners_from_specs([
        {**(h.spec or {}), "id": h.miner_id} for h in handles
    ])


# ── Descriptor builder ────────────────────────────────────────

def build_descriptor(
    *,
    node_id: str,
    node_name: Optional[str] = None,
    public_host: Optional[str] = None,
    public_port: Optional[int] = None,
    rpc_endpoints: Optional[List[str]] = None,
    auto_mine: bool = False,
    log_level: str = "INFO",
    core: Optional["MinerCore"] = None,
    miner_specs: Optional[List[Dict[str, Any]]] = None,
    system_info: Optional[SystemInfo] = None,
    include_system_info: bool = True,
) -> NodeDescriptor:
    """Assemble a NodeDescriptor from node id + miner inventory + flags.

    Miner inventory source precedence (most specific first):
      1. ``miner_specs``  — list of raw spec dicts (used by the CLI
         ``identify`` flow so it can render the miners block without
         spawning worker processes).
      2. ``core``         — live ``MinerCore`` whose handles are walked
         directly (used by the in-process telemetry endpoint).
      3. neither          — ``miners`` defaults to ``{}``.

    ``system_info`` is probed lazily; pass ``include_system_info=False``
    to skip the probe entirely (useful in CI / sandboxed environments
    where subprocess calls aren't permitted).
    """
    if include_system_info:
        sysinfo = system_info if system_info is not None else collect_system_info()
    else:
        sysinfo = None

    if miner_specs is not None:
        miners = summarize_miners_from_specs(miner_specs)
    elif core is not None:
        miners = summarize_miners_from_handles(
            list(getattr(core, "miner_handles", []) or [])
        )
    else:
        miners = {}

    return NodeDescriptor(
        descriptor_version=DESCRIPTOR_VERSION,
        node_name=node_name or node_id,
        public_host=public_host,
        public_port=public_port,
        rpc_endpoints=list(rpc_endpoints or []),
        auto_mine=bool(auto_mine),
        log_level=str(log_level or "INFO").upper(),
        runtime=_runtime(),
        miners=miners,
        system_info=sysinfo,
    )


# ── Validation + canonical serialization ──────────────────────

class DescriptorValidationError(ValueError):
    """Raised when a NodeDescriptor violates the wire-format bounds.

    The message is single-line and kebab-case so the CLI can surface it
    as ``descriptor-invalid field=<name> reason=<...>`` without further
    formatting.
    """


def validate_descriptor(descriptor: NodeDescriptor) -> None:
    """Enforce field-size and value bounds matching the Phase 2 storage limits."""
    name_bytes = descriptor.node_name.encode("utf-8")
    if not descriptor.node_name:
        raise DescriptorValidationError(
            "field=node_name reason=empty"
        )
    if len(name_bytes) > NODE_NAME_MAX_BYTES:
        raise DescriptorValidationError(
            f"field=node_name reason=too-long bytes={len(name_bytes)} "
            f"max={NODE_NAME_MAX_BYTES}"
        )
    if descriptor.public_host is not None:
        host_bytes = descriptor.public_host.encode("utf-8")
        if len(host_bytes) > PUBLIC_HOST_MAX_BYTES:
            raise DescriptorValidationError(
                f"field=public_host reason=too-long bytes={len(host_bytes)} "
                f"max={PUBLIC_HOST_MAX_BYTES}"
            )
    if descriptor.public_port is not None:
        if not (PUBLIC_PORT_MIN <= descriptor.public_port <= PUBLIC_PORT_MAX):
            raise DescriptorValidationError(
                f"field=public_port reason=out-of-range value={descriptor.public_port}"
            )
    if len(descriptor.rpc_endpoints) > RPC_ENDPOINTS_MAX_COUNT:
        raise DescriptorValidationError(
            f"field=rpc_endpoints reason=too-many count={len(descriptor.rpc_endpoints)} "
            f"max={RPC_ENDPOINTS_MAX_COUNT}"
        )
    for i, endpoint in enumerate(descriptor.rpc_endpoints):
        if not isinstance(endpoint, str) or not endpoint:
            raise DescriptorValidationError(
                f"field=rpc_endpoints[{i}] reason=empty"
            )
        if len(endpoint.encode("utf-8")) > RPC_ENDPOINT_MAX_BYTES:
            raise DescriptorValidationError(
                f"field=rpc_endpoints[{i}] reason=too-long max={RPC_ENDPOINT_MAX_BYTES}"
            )
    if descriptor.log_level not in ALLOWED_LOG_LEVELS:
        raise DescriptorValidationError(
            f"field=log_level reason=unknown value={descriptor.log_level!r} "
            f"allowed={sorted(ALLOWED_LOG_LEVELS)}"
        )
    # Walk every string value in the rendered descriptor for
    # credential-shaped content. The QPU spec builder already drops
    # bad solver names; this is the catch-all that protects rpc_endpoints,
    # daily_budget, public_host, etc.
    rendered = descriptor.to_dict()
    for s in _iter_string_values(rendered):
        if _looks_secretish(s):
            raise DescriptorValidationError(
                "field=<descriptor> reason=credential-shaped-value "
                "hint=value matches a known secret/token pattern; "
                "scrub it before resubmitting"
            )


def to_canonical_json(descriptor: NodeDescriptor) -> bytes:
    """Serialize the descriptor as deterministic UTF-8 JSON bytes.

    Sorted keys (recursively), compact separators, no trailing newline.
    The output is what gets posted as the ``System.remark`` body — and
    what the indexer hashes / compares against when deduplicating.
    """
    payload = descriptor.to_dict()
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


__all__ = [
    "ALLOWED_LOG_LEVELS",
    "CPUInfo",
    "DESCRIPTOR_VERSION",
    "DescriptorValidationError",
    "GPUInfo",
    "NODE_NAME_MAX_BYTES",
    "NodeDescriptor",
    "PUBLIC_HOST_MAX_BYTES",
    "PUBLIC_PORT_MAX",
    "PUBLIC_PORT_MIN",
    "RPC_ENDPOINTS_MAX_COUNT",
    "RPC_ENDPOINT_MAX_BYTES",
    "Runtime",
    "SCHEMA_NAME",
    "SystemInfo",
    "build_descriptor",
    "collect_system_info",
    "filter_telemetry_fields",
    "is_secret_key",
    "summarize_miners_from_handles",
    "summarize_miners_from_specs",
    "to_canonical_json",
    "validate_descriptor",
]
