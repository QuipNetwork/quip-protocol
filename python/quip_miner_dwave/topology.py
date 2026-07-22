"""Native topology hashing for Hello.native_topology_hash advertisement."""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence, Tuple


def native_topology_hash(
    nodes: Sequence[int],
    edges: Iterable[Tuple[int, int]],
) -> bytes:
    """32-byte blake2b of sorted nodes + canonical sorted undirected edges.

    Used only for capability advertisement (``Hello.native_topology_hash``) so
    the coordinator can prefer native-sized jobs. Not the on-chain consensus
    topology hash (which also binds allowed-value specs).
    """
    sorted_nodes = sorted(int(n) for n in nodes)
    sorted_edges = sorted(
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
    )
    h = hashlib.blake2b(digest_size=32)
    h.update(len(sorted_nodes).to_bytes(4, "little"))
    for n in sorted_nodes:
        h.update(n.to_bytes(4, "little"))
    h.update(len(sorted_edges).to_bytes(4, "little"))
    for u, v in sorted_edges:
        h.update(u.to_bytes(4, "little"))
        h.update(v.to_bytes(4, "little"))
    return h.digest()
