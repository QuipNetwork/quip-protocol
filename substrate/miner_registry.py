"""Adapters for the runtime ``MinerRegistry`` pallet.

The runtime stores a compact, bounded descriptor schema rather than the richer
JSON document used by the miner's local REST API. This module is the boundary
that turns the already-validated Python ``NodeDescriptor`` into the exact
SCALE-friendly call params expected by ``MinerRegistry.set_descriptor`` and
``MinerRegistry.participate``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from shared.system_info import NodeDescriptor
from substrate.mempool_types import MinerType


_LOG_LEVELS = {
    "DEBUG": "Debug",
    "INFO": "Info",
    "WARNING": "Warning",
    "ERROR": "Error",
}

_QPU_PROVIDER_KIND = {
    "dwave": "qpu",
    "ibm": "qpu_ibm",
    "ionq": "qpu_ionq",
    "pasqal": "qpu_pasqal",
}


def descriptor_call_params(
    descriptor: NodeDescriptor,
    *,
    node_id: str,
) -> dict[str, Any]:
    """Return call params for ``MinerRegistry.set_descriptor``."""
    if not descriptor.miners:
        raise ValueError("descriptor must contain at least one miner")
    return {
        "descriptor": {
            "V1": {
                "node_id": node_id.encode("utf-8"),
                "node_name": descriptor.node_name.encode("utf-8"),
                "public_host": _optional_bytes(descriptor.public_host),
                "public_port": descriptor.public_port,
                "rpc_endpoints": [
                    endpoint.encode("utf-8")
                    for endpoint in descriptor.rpc_endpoints
                ],
                "auto_mine": descriptor.auto_mine,
                "log_level": _LOG_LEVELS[descriptor.log_level],
                "miners": [
                    _miner_spec(label, miner)
                    for label, miner in descriptor.miners.items()
                ],
            }
        }
    }


def participation_call_params(
    *,
    latest_qblock_id: Optional[int],
    kind: Any,
    budget_seconds: Optional[Any] = None,
) -> dict[str, Any]:
    """Return call params for ``MinerRegistry.participate``."""
    qblock_id = 1 if latest_qblock_id is None else int(latest_qblock_id) + 1
    return {
        "qblock_id": qblock_id,
        "kind": miner_kind_to_scale(kind),
        "budget_seconds": _optional_u32(budget_seconds),
    }


def miner_kind_to_scale(kind: Any) -> str:
    """Map local miner-kind strings to the runtime ``MinerKind`` variant."""
    if isinstance(kind, MinerType):
        return kind.to_scale_variant()
    return MinerType.from_kind(str(kind or "cpu")).to_scale_variant()


def _miner_spec(label: str, miner: dict[str, Any]) -> dict[str, Any]:
    kind = _descriptor_miner_kind(miner)
    backend = _descriptor_backend(miner)
    device_id = _descriptor_device_id(miner)
    return {
        "kind": miner_kind_to_scale(kind),
        "label": _optional_bytes(label),
        "backend": _optional_bytes(backend),
        "device_id": _optional_bytes(device_id),
    }


def _descriptor_miner_kind(miner: dict[str, Any]) -> str:
    raw_kind = str(miner.get("kind", "")).lower()
    if raw_kind == "cpu":
        return "cpu"
    if raw_kind == "gpu":
        return "gpu"
    if raw_kind == "qpu":
        provider = str(miner.get("provider", "dwave")).lower()
        return _QPU_PROVIDER_KIND.get(provider, "qpu")
    if raw_kind == "asic":
        return "asic"
    return raw_kind or "cpu"


def _descriptor_backend(miner: dict[str, Any]) -> Optional[str]:
    if miner.get("backend") is not None:
        return str(miner["backend"])
    if miner.get("provider") is not None:
        return str(miner["provider"])
    return None


def _descriptor_device_id(miner: dict[str, Any]) -> Optional[str]:
    for key in ("miner_id", "device_id", "solver", "device_index", "device"):
        value = miner.get(key)
        if value is not None:
            return str(value)
    return None


def _optional_bytes(value: Optional[str]) -> Optional[bytes]:
    if value is None:
        return None
    return value.encode("utf-8")


def _optional_u32(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


__all__ = [
    "descriptor_call_params",
    "miner_kind_to_scale",
    "participation_call_params",
]
