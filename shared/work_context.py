"""Protocol-neutral seam between mining work-sources and `BaseMiner`.

`WorkContext` is a structural Protocol describing the shape both
PoW (`substrate.types.SubstrateMiningContext`) and mempool
(`shared.mempool_types.MempoolJobContext`) contexts satisfy. Each
context implements `resolve_ising(salt, nodes, edges)` and
`requirements()` itself — `shared/` no longer needs to import either
concrete type.
"""
from __future__ import annotations

import random
from typing import (
    Dict,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from shared.miner_types import BlockRequirements


@runtime_checkable
class WorkContext(Protocol):
    """Structural shape required by `base_miner.mine_work_item`."""

    nodes: Sequence[int]
    edges: Sequence[Tuple[int, int]]

    def resolve_ising(
        self,
        salt: bytes,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
        ...

    def requirements(self) -> BlockRequirements:
        ...


def resolve_ising(context, salt, nodes, edges):
    """Delegate to the context's own `resolve_ising`. Kept for call-site stability."""
    return context.resolve_ising(salt, nodes, edges)


def requirements_from_context(context) -> BlockRequirements:
    """Delegate to the context's own `requirements`. Kept for call-site stability."""
    return context.requirements()


def fresh_salt() -> bytes:
    """32-byte random salt for one mining iteration."""
    return random.randbytes(32)


__all__ = [
    "WorkContext",
    "fresh_salt",
    "requirements_from_context",
    "resolve_ising",
]
