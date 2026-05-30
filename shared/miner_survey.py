"""Build the `MinerSurveyV1` payload exposed at `/api/v1/miner/survey`.

The survey is a deterministic, JSON-safe snapshot of who the miner is
(node id, signer identity), what it can do (active miner handles +
capability flags), and the hardware backing each handle. The schema is
stable enough for the dashboard indexer to consume directly without any
chain-side lookup — the only required input from the chain is the signer
account, which is already loaded in the controller process.

The survey deliberately does not read `.env`, credential files, or any
secret-bearing settings. Only the spec dicts attached to each
`MinerHandle` and the topology metadata bundled with the sampler are
exposed. Unknown or unsafe values are normalized to `null`, `false`, or
empty lists rather than omitted, so dashboards never have to special-case
missing keys.

Forward compatibility:
  Future revisions bump `schema_version` and add fields; readers ignore
  fields they don't understand. The payload shape here is intended to
  later be reusable as the off-chain miner-survey serialization
  format — keep additions JSON-safe and free of Python-only objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from shared.system_info import is_secret_key
from shared.version import get_version


if TYPE_CHECKING:
    from shared.miner_core import MinerCore
    from shared.miner_worker import MinerHandle
    from shared.signer import Signer


SCHEMA_NAME = "quip.miner_survey.v1"
SCHEMA_VERSION = 1
CLIENT_NAME = "quip-miner"


def build_miner_survey(
    core: "MinerCore",
    signer: "Signer",
    *,
    controller: Optional[Any] = None,
) -> Dict[str, Any]:
    """Render `MinerSurveyV1` from the live miner singletons.

    `controller` is optional and only used to surface the chain-bound
    `topology_hash` when the PoW controller has one pinned. Passing
    `None` (or a controller with `topology_hash=None`) leaves the
    `topology.topology_hash` field as `null` — the indexer can still
    use the local topology metadata to identify the sampler shape.
    """
    handles = list(core.miner_handles)
    miner_entries = [_describe_handle(h, controller=controller) for h in handles]
    miner_types = sorted({h.miner_type for h in handles})

    cpu_handles = [h for h in handles if _kind(h) == "cpu"]
    gpu_handles = [h for h in handles if _kind(h) in _GPU_KINDS]
    qpu_handles = [h for h in handles if _kind(h) == "qpu"]

    account_bytes = signer.account_id_bytes()
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "node_id": core.node_id,
        "client": {
            "name": CLIENT_NAME,
            "version": get_version(),
        },
        "account": {
            "ss58_address": signer.ss58_address(),
            "account_id_hex": "0x" + account_bytes.hex(),
        },
        "capabilities": {
            "miner_types": miner_types,
            "pow": bool(handles),
            "mempool": False,
        },
        "miners": miner_entries,
        "hardware": {
            "cpu": {
                "worker_count": len(cpu_handles),
                "process_count": len(cpu_handles),
            },
            "gpu": {
                "available": bool(gpu_handles),
                "devices": [_gpu_device(h) for h in gpu_handles],
            },
            "qpu": {
                "available": bool(qpu_handles),
                "providers": [_qpu_provider(h) for h in qpu_handles],
            },
        },
    }


# ----------------------------------------------------------------------
# Per-handle description
# ----------------------------------------------------------------------


_GPU_KINDS = frozenset({"cuda", "cuda-gibbs", "metal", "modal"})

# Map MinerHandle.spec["kind"] -> the "backend" string the dashboard
# expects. `kind` is already a stable identifier (it's what
# `build_miner_from_spec` switches on), so this mapping is one-to-one.
_BACKEND_BY_KIND: Dict[str, str] = {
    "cpu": "simulated_annealing",
    "cuda": "cuda",
    "cuda-gibbs": "cuda-gibbs",
    "metal": "metal",
    "modal": "modal",
    "qpu": "dwave",
}


def _describe_handle(handle: "MinerHandle", *, controller: Optional[Any]) -> Dict[str, Any]:
    spec = handle.spec or {}
    kind = _kind(handle)
    return {
        "id": handle.miner_id,
        "type": handle.miner_type,
        "backend": _BACKEND_BY_KIND.get(kind),
        "device": _device(spec, kind),
        "topology": _topology_info(spec, controller=controller),
        "config": _config_dict(spec),
    }


def _kind(handle: "MinerHandle") -> str:
    return str((handle.spec or {}).get("kind", "")).lower()


def _device(spec: Dict[str, Any], kind: str) -> Optional[str]:
    args = spec.get("args") or {}
    if kind == "cuda" or kind == "cuda-gibbs":
        dev = args.get("device")
        return str(dev) if dev is not None else None
    if kind == "metal":
        return str(args.get("device") or "mps")
    if kind == "modal":
        gpu_type = args.get("gpu_type")
        return str(gpu_type) if gpu_type is not None else None
    if kind == "qpu":
        cfg = spec.get("cfg") or {}
        qtype = cfg.get("qpu_type")
        return str(qtype) if qtype is not None else None
    return None


# ----------------------------------------------------------------------
# Topology info — pulled from the sampler topology object stashed in
# spec["args"]["topology"] by `_inject_topology` in quip_cli. The object
# is a `DWaveTopology`-shaped class (Zephyr / Chimera / Pegasus / JSON
# hardware), so we duck-type the attributes we expose. The chain-bound
# `topology_hash` is only available when the controller has one pinned.
# ----------------------------------------------------------------------


def _topology_info(
    spec: Dict[str, Any], *, controller: Optional[Any]
) -> Optional[Dict[str, Any]]:
    args = spec.get("args") or {}
    topo = args.get("topology")
    if topo is None:
        return None
    return {
        "topology_hash": _controller_topology_hash(controller),
        "num_nodes": _safe_int(getattr(topo, "num_nodes", None)),
        "num_edges": _safe_int(getattr(topo, "num_edges", None)),
        "solver_name": _safe_str(getattr(topo, "solver_name", None)),
        "topology_type": _safe_str(getattr(topo, "topology_type", None)),
        "topology_shape": _topology_shape_list(topo),
    }


def _controller_topology_hash(controller: Optional[Any]) -> Optional[str]:
    if controller is None:
        return None
    raw = getattr(controller, "topology_hash", None)
    if not raw:
        return None
    try:
        return "0x" + bytes(raw).hex()
    except (TypeError, ValueError):
        return None


def _topology_shape_list(topo: Any) -> Optional[List[int]]:
    """Prefer the list form from `properties.topology.shape`; fall back to
    `m`/`n`/`t` attrs. The Protocol annotates `topology_shape` as a string
    (e.g. `"[9, 2]"`) which isn't useful to indexers — emit the integer
    list instead so the indexer can pivot on it directly."""
    props = getattr(topo, "properties", None)
    if isinstance(props, dict):
        inner = props.get("topology")
        if isinstance(inner, dict):
            shape = inner.get("shape")
            if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
                return list(shape)
    m = getattr(topo, "m", None)
    n = getattr(topo, "n", None)
    t = getattr(topo, "t", None)
    if isinstance(m, int) and isinstance(n, int) and isinstance(t, int):
        return [m, n, t]
    if isinstance(m, int) and isinstance(t, int):
        return [m, t]
    if isinstance(m, int):
        return [m]
    return None


# ----------------------------------------------------------------------
# Config — only JSON-safe primitives from spec args/cfg. The `topology`
# arg is excluded because it's a Python object (handled separately).
# ----------------------------------------------------------------------


def _config_dict(spec: Dict[str, Any]) -> Dict[str, Any]:
    cfg = spec.get("cfg") or {}
    args = spec.get("args") or {}
    merged: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(k, str) and not is_secret_key(k) and _is_jsonable(v):
            merged[k] = v
    for k, v in args.items():
        if k == "topology":
            continue
        if isinstance(k, str) and not is_secret_key(k) and _is_jsonable(v):
            merged[k] = v
    return merged


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


# ----------------------------------------------------------------------
# Hardware aggregates
# ----------------------------------------------------------------------


def _gpu_device(handle: "MinerHandle") -> Dict[str, Any]:
    spec = handle.spec or {}
    kind = _kind(handle)
    return {
        "id": handle.miner_id,
        "type": handle.miner_type,
        "backend": _BACKEND_BY_KIND.get(kind),
        "device": _device(spec, kind),
    }


def _qpu_provider(handle: "MinerHandle") -> Dict[str, Any]:
    spec = handle.spec or {}
    cfg = spec.get("cfg") or {}
    return {
        "id": handle.miner_id,
        "type": handle.miner_type,
        "provider": _safe_str(cfg.get("qpu_type")),
    }


# ----------------------------------------------------------------------
# Coercion helpers
# ----------------------------------------------------------------------


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


__all__ = ["build_miner_survey", "SCHEMA_NAME", "SCHEMA_VERSION"]
