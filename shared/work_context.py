"""Protocol-neutral seam between mining work-sources and `BaseMiner`.

`BaseMiner.mine_work_item` is shared between the PoW path (chain head →
`SubstrateMiningContext`) and the mempool path (`JobProposed` event →
`MempoolJobContext`). The two contexts carry the same essentials —
`nodes`, `edges`, quality floors — but materialize the Ising problem
differently:

  - PoW derives a fresh `nonce` per attempt (`derive_nonce(
    last_proof_block_hash, miner, salt)`) and feeds it through
    `generate_ising_model_from_nonce` to get `(h, J)`. The chain checks
    this derivation in `submit_proof`.
  - Mempool just carries the `(h_values, j_values)` directly inside
    `IsingParams`; the chain takes whatever solver submitted spins solve
    against and validates energy + diversity, not the model.

`resolve_ising(context, salt)` dispatches between these two paths.
`requirements_from_context(context)` maps both kinds of quality floor
into a `BlockRequirements` so `evaluate_sampleset` sees one shape.

Keeping this in its own module keeps `base_miner.py` free of mempool
imports — the mempool types and the mining loop both depend on
`work_context` but not on each other.
"""
from __future__ import annotations

import random
from typing import Dict, Tuple, Union

from shared.mempool_types import MempoolJobContext
from shared.miner_types import BlockRequirements
from shared.quantum_proof_of_work import derive_nonce, generate_ising_model_from_nonce
from shared.substrate_types import SubstrateMiningContext


WorkContext = Union[SubstrateMiningContext, MempoolJobContext]


# Sentinel timeout for legacy difficulty decay paths. Phase 3 documents
# why substrate-mode mining never touches `compute_current_requirements`;
# we pass the same large value the v0.1 helper used (`2**31 - 1`) so any
# code that does sneak through treats decay as effectively-disabled.
_DECAY_DISABLED = 2**31 - 1


def resolve_ising(
    context: WorkContext,
    salt: bytes,
    nodes,
    edges,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
    """Return `(h, J, telemetry_nonce)` for one mining iteration.

    `nodes` and `edges` are the *sampler's* topology iteration order. The
    PoW chain validates proofs by re-deriving `(h, J)` against the
    iteration order of its registered topology — the controller already
    verified the sampler's `topology_hash` matches the chain's, so
    passing the sampler's order through here keeps on-chain validation
    byte-for-byte compatible. (Phase 4 caught this off-by-one; Phase 8b
    pins the contract via this parameter.)

    - PoW path: derives a fresh nonce from `salt` + chain identity
      material, regenerates `(h, J)` via ChaCha8 the same way the pallet
      does. `salt` + `nonce` become part of the on-chain commitment.
    - Mempool path: maps the chain-carried `h_values` / `j_values`
      millivalues (i32) into float dicts indexed by `context.nodes` /
      `context.edges` (the chain only checks that submitted spins solve
      the stored model — there's no re-derivation step, so `nodes` /
      `edges` are unused here). `salt` is unused. `telemetry_nonce` is
      `0`; preserved in `MiningResult` for stats compatibility.
    """
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
    """Build `BlockRequirements` from a work context's quality floors.

    The mining loop and `evaluate_sampleset` both consume `BlockRequirements`;
    this is the single point where chain-shaped difficulty (PoW) or
    optional quality floors (mempool) are translated into the same shape.

    Mempool semantics for unset floors:
      - `min_energy_milli=None`     → no upper bound on energy (use +inf)
      - `min_diversity_milli=None`  → no lower bound on diversity (0.0)
      - `min_solutions=None`        → at least 1 solution (the pallet
        rejects `NoSolutionsSubmitted` for an empty submission, so 1 is
        the natural floor)
    """
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

    # PoW path — mirrors `SubstrateDifficulty.max_energy` / `.min_diversity`
    # but reads the raw milli fields to avoid an extra property call.
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
    """32-byte random salt for one mining iteration.

    Used identically by both work-source paths — PoW feeds it through
    `derive_nonce`, mempool ignores it but keeps the call shape uniform.
    """
    return random.randbytes(32)


__all__ = [
    "WorkContext",
    "fresh_salt",
    "requirements_from_context",
    "resolve_ising",
]
