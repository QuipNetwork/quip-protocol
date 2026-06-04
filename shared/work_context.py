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
    Any,
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

    def make_feeder(
        self,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
        *,
        buffer_size: int = 8,
    ) -> Any:
        """Construct the feeder ``BaseMiner.mine_work_item`` pops from.

        Each context flavor picks the right backing implementation: PoW
        contexts return a :class:`shared.ising_feeder.RandomIsingFeeder`
        that derives a fresh ``(salt -> nonce -> h, J)`` per iteration;
        mempool contexts return a :class:`FixedIsingFeeder` cycling the
        single ``(h, J)`` carried in the job order. Return type is
        duck-typed deliberately — both implementations expose the same
        ``pop_blocking`` / ``stop`` surface, so the loop doesn't need a
        nominal supertype.
        """
        ...

    def uses_decay_ratchet(self) -> bool:
        """Whether the loop ranks candidates with the decay ratchet.

        PoW (substrate) work takes the "stash the best, submit when the chain
        threshold catches up" ratchet loop — even with a flat decay schedule
        that path falls back to strict energy ranking. Mempool jobs carry fixed
        quality floors and use strict-energy evaluation instead. This single
        discriminator is what ``mine_work_item`` branches on; it replaces an
        ``isinstance`` check so the foundation need not import either concrete
        context type.
        """
        ...


def resolve_ising(
    context: WorkContext,
    salt: bytes,
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
    """Delegate to the context's own `resolve_ising`. Kept for call-site stability."""
    return context.resolve_ising(salt, nodes, edges)


def requirements_from_context(context: WorkContext) -> BlockRequirements:
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
