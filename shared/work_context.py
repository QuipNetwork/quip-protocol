"""Protocol-neutral seam between mining work-sources and `BaseMiner`.

`WorkContext` is a structural Protocol describing the shape both PoW
(`substrate.types.SubstrateMiningContext`) and mempool
(`shared.mempool_types.MempoolJobContext`) work-source contexts satisfy.
Defining it as a Protocol (rather than `Union[...]`) keeps `shared/` from
having to import its specializations — the substrate package can depend
on `shared/`, but not the other way around.

`resolve_ising(context, salt)` dispatches between PoW and mempool paths
via `isinstance` against the concrete types. `requirements_from_context`
does the same. Both will be replaced with method-based dispatch in a
follow-up so `shared/` no longer references the concrete types at all.
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
from shared.quantum_proof_of_work import derive_nonce, generate_ising_model_from_nonce


@runtime_checkable
class WorkContext(Protocol):
    """Structural shape required by `base_miner.mine_work_item`.

    Both `SubstrateMiningContext` and `MempoolJobContext` satisfy this
    Protocol without explicit inheritance. The Protocol enumerates only
    fields the mining loop and its dispatch helpers actually read.
    """

    nodes: Sequence[int]
    edges: Sequence[Tuple[int, int]]


# Sentinel timeout for legacy difficulty decay paths.
_DECAY_DISABLED = 2**31 - 1


def resolve_ising(
    context: WorkContext,
    salt: bytes,
    nodes,
    edges,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
    """Return `(h, J, telemetry_nonce)` for one mining iteration.

    Dispatches PoW vs mempool via `isinstance`. The PoW branch derives a
    fresh nonce from chain identity material; the mempool branch maps
    chain-carried `(h_values, j_values)` directly.
    """
    from shared.mempool_types import MempoolJobContext
    from substrate.types import SubstrateMiningContext

    if isinstance(context, MempoolJobContext):
        h = {
            int(node): float(hv) / 1000.0
            for node, hv in zip(context.nodes, context.h_values)
        }
        J: Dict[Tuple[int, int], float] = {
            (int(edge[0]), int(edge[1])): float(jv) / 1000.0
            for edge, jv in zip(context.edges, context.j_values)
        }
        return h, J, 0

    if not isinstance(context, SubstrateMiningContext):
        raise TypeError(f"resolve_ising: unknown context type {type(context)!r}")

    nonce = derive_nonce(
        context.last_proof_block_hash,
        context.miner_account_bytes,
        salt,
    )
    h, J = generate_ising_model_from_nonce(
        nonce,
        list(nodes),
        list(edges),
        allowed_h=context.allowed_h_values,
        allowed_j=context.allowed_j_values,
    )
    return h, J, nonce


def requirements_from_context(context: WorkContext) -> BlockRequirements:
    """Build `BlockRequirements` from a work context's quality floors."""
    from shared.mempool_types import MempoolJobContext
    from substrate.types import SubstrateMiningContext

    if isinstance(context, MempoolJobContext):
        difficulty_energy = (
            float(context.min_energy_milli) / 1000.0
            if context.min_energy_milli is not None
            else float("inf")
        )
        min_diversity = (
            float(context.min_diversity_milli) / 1000.0
            if context.min_diversity_milli is not None
            else 0.0
        )
        min_solutions = (
            int(context.min_solutions) if context.min_solutions is not None else 1
        )
        return BlockRequirements(
            difficulty_energy=difficulty_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=_DECAY_DISABLED,
        )

    if not isinstance(context, SubstrateMiningContext):
        raise TypeError(
            f"requirements_from_context: unknown context type {type(context)!r}"
        )

    d = context.difficulty
    return BlockRequirements(
        difficulty_energy=float(d.max_energy_milli) / 1000.0,
        min_diversity=float(d.min_diversity_milli) / 1000.0,
        min_solutions=int(d.min_solutions),
        timeout_to_difficulty_adjustment_decay=_DECAY_DISABLED,
        allowed_h_values=context.allowed_h_values,
        allowed_j_values=context.allowed_j_values,
    )


def fresh_salt() -> bytes:
    """32-byte random salt for one mining iteration."""
    return random.randbytes(32)


__all__ = [
    "WorkContext",
    "fresh_salt",
    "requirements_from_context",
    "resolve_ising",
]
