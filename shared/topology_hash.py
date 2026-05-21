"""Canonical topology hash matching the on-chain ``hash_topology``.

Python mirror of ``pallets/quantum-pow/src/topology.rs::hash_topology`` in
``quip-protocol-rs``. The hash binds:

  - sorted node IDs
  - canonical (min, max) sorted edges
  - the three :class:`shared.allowed_value_spec.AllowedValueSpec` instances
    (h, j, spin) via :func:`shared.allowed_value_spec.canonical_bytes`

so two topologies that differ only by allowed-value sets receive distinct
hashes, and two topologies that differ only by element ordering (within a
Set spec or within the edge list) receive the same hash.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

from shared.allowed_value_spec import AllowedValueSpec, canonical_bytes


def _compact_len(n: int) -> bytes:
    """SCALE compact-int encoding (matches the Rust ``Encode`` derive)."""
    if n < 0:
        raise ValueError(f"compact len cannot be negative, got {n}")
    if n < 0x40:
        return bytes([n << 2])
    if n < 0x4000:
        return ((n << 2) | 0b01).to_bytes(2, "little")
    if n < 0x4000_0000:
        return ((n << 2) | 0b10).to_bytes(4, "little")
    raise ValueError(f"compact len {n} out of u32 range")


def topology_hash(
    nodes: Iterable[int],
    edges: Iterable[Tuple[int, int]],
    allowed_h: AllowedValueSpec,
    allowed_j: AllowedValueSpec,
    allowed_spin: AllowedValueSpec,
) -> bytes:
    """Compute the canonical 32-byte topology hash.

    The hash function is ``blake2_256`` of the SCALE encoding of
    ``(sorted_nodes, sorted_edges, canonical(h), canonical(j), canonical(spin))``.
    ``canonical_bytes`` from :mod:`shared.allowed_value_spec` is what the
    Rust side encodes as a ``Vec<u8>`` field of the tuple — so the SCALE
    encoding here writes each canonical buffer as a compact-length-prefixed
    byte sequence.
    """
    sorted_nodes = sorted(int(n) for n in nodes)
    sorted_edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
    )

    buf = bytearray()
    buf += _compact_len(len(sorted_nodes))
    for n in sorted_nodes:
        buf += n.to_bytes(4, "little")
    buf += _compact_len(len(sorted_edges))
    for u, v in sorted_edges:
        buf += u.to_bytes(4, "little") + v.to_bytes(4, "little")

    for spec in (allowed_h, allowed_j, allowed_spin):
        cb = canonical_bytes(spec)
        buf += _compact_len(len(cb))
        buf += cb

    return hashlib.blake2b(bytes(buf), digest_size=32).digest()


__all__ = ["topology_hash"]
