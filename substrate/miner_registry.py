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
    schema_version: int = 2,
) -> dict[str, Any]:
    """Return call params for ``MinerRegistry.set_descriptor``.

    ``schema_version`` selects the descriptor variant the runtime accepts
    (probe it via :meth:`SubstrateClient.descriptor_schema_version`):

    - **V2** (``>= 2``): also records the node's ``system_info`` (OS / CPU /
      memory / GPUs) and ``runtime`` (python / quip / protocol version, docker)
      — the hardware survey the chain was re-extended to hold (quip-protocol-rs
      v0.2 ``MinerRegistry`` descriptor V2).
    - **V1** (``< 2``): identity + miner inventory only, for runtimes that
      predate V2. V1 is byte-identical (enum index 0) so this stays compatible.

    Every ``BoundedVec<u8>`` field is wrapped via :func:`_bounded` (a 1-tuple)
    because the runtime metadata describes ``BoundedVec`` as a composite with a
    single inner ``Vec`` field, and scalecodec (1.2.x) does not flatten it —
    the same convention ``QuantumPow.register_topology`` uses for ``(nodes,)``.
    Nested structs (``os``/``cpu``/``gpus`` entries/``runtime``) are plain
    dicts; only their ``BoundedVec<u8>`` fields get wrapped.
    """
    if not descriptor.miners:
        raise ValueError("descriptor must contain at least one miner")
    body: dict[str, Any] = {
        "node_id": _bounded(node_id.encode("utf-8")),
        "node_name": _bounded(descriptor.node_name.encode("utf-8")),
        "public_host": _optional_bounded(descriptor.public_host),
        "public_port": descriptor.public_port,
        "rpc_endpoints": _bounded([
            endpoint.encode("utf-8")
            for endpoint in descriptor.rpc_endpoints
        ]),
        "auto_mine": descriptor.auto_mine,
        "log_level": _LOG_LEVELS[descriptor.log_level],
        "miners": _bounded([
            _miner_spec(label, miner)
            for label, miner in descriptor.miners.items()
        ]),
    }
    if schema_version >= 2:
        body["system_info"] = _system_info_params(descriptor.system_info)
        body["runtime"] = _runtime_params(descriptor.runtime)
        return {"descriptor": {"V2": body}}
    return {"descriptor": {"V1": body}}


def _system_info_params(system_info: Any) -> Optional[dict[str, Any]]:
    """Project a :class:`shared.system_info.SystemInfo` to the V2 SCALE shape,
    or ``None`` when the hardware survey was skipped (``--no-system-info``).

    The runtime rejects empty ``os.system`` / ``cpu.brand`` / ``cpu.arch`` /
    GPU ``vendor`` / GPU ``name`` and ``utilization_pct > 100``, so empties fall
    back to a non-empty sentinel and utilization is clamped — a best-effort
    survey must never make descriptor filing fail.
    """
    if system_info is None:
        return None
    os_info = system_info.os or {}
    cpu = system_info.cpu
    return {
        "os": {
            "system": _bounded(_nonempty(os_info.get("system"), "unknown")),
            # release / machine may be empty per the runtime; send as-is.
            "release": _bounded(str(os_info.get("release") or "").encode("utf-8")),
            "machine": _bounded(str(os_info.get("machine") or "").encode("utf-8")),
        },
        "cpu": {
            "logical_cores": cpu.logical_cores,
            "physical_cores": cpu.physical_cores,
            "brand": _bounded(_nonempty(cpu.brand, "unknown")),
            "arch": _bounded(_nonempty(cpu.arch, "unknown")),
        },
        "memory_mb": system_info.memory_mb,
        "gpus": _bounded([
            {
                "index": int(g.index) & 0xFF,
                "vendor": _bounded(_nonempty(g.vendor, "unknown")),
                "name": _bounded(_nonempty(g.name, "unknown")),
                "memory_mb": g.memory_mb,
                "utilization_pct": _clamp_pct(g.observed_utilization_pct),
            }
            for g in system_info.gpus
        ]),
    }


def _runtime_params(runtime: Any) -> Optional[dict[str, Any]]:
    """Project a :class:`shared.system_info.Runtime` to the V2 SCALE shape.

    The runtime rejects empty ``python`` / ``quip_version`` (and empty
    ``docker_image`` when present), so those fall back to a non-empty sentinel.
    """
    if runtime is None:
        return None
    return {
        "python": _bounded(_nonempty(runtime.python, "unknown")),
        "quip_version": _bounded(_nonempty(runtime.quip_version, "unknown")),
        "protocol_version": int(runtime.protocol_version),
        "in_docker": bool(runtime.in_docker),
        "docker_image": _optional_bounded(runtime.docker_image or None),
    }


def _nonempty(value: Any, fallback: str) -> bytes:
    """UTF-8 encode ``value``, substituting ``fallback`` when empty/None."""
    text = str(value) if value not in (None, "") else fallback
    return text.encode("utf-8")


def _clamp_pct(value: Optional[int]) -> Optional[int]:
    """Clamp a percentage to 0..=100 (the runtime rejects >100); pass None."""
    if value is None:
        return None
    return max(0, min(100, int(value)))


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
        "label": _optional_bounded(label),
        "backend": _optional_bounded(backend),
        "device_id": _optional_bounded(device_id),
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


def _bounded(value: Any) -> tuple[Any]:
    """Wrap a ``BoundedVec`` value as the single-field composite the runtime
    metadata expects.

    scalecodec (1.2.x) describes ``bounded_collections.bounded_vec.BoundedVec``
    as a composite with one inner ``Vec`` field and does not flatten it, so the
    inner value must be passed as a 1-tuple — the same convention
    ``QuantumPow.register_topology`` uses for ``(nodes,)`` / ``(edges,)``. The
    inner ``Vec<u8>`` still accepts raw ``bytes`` for its elements.
    """
    return (value,)


def _optional_bounded(value: Optional[str]) -> Optional[tuple[bytes]]:
    """Encode + :func:`_bounded`-wrap an ``Option<BoundedVec<u8>>`` field.

    ``None`` stays ``None`` (``Option::None``); a present string is UTF-8
    encoded and wrapped so the ``Some`` payload matches the BoundedVec
    composite shape.
    """
    if value is None:
        return None
    return _bounded(value.encode("utf-8"))


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
