"""Encode a `MiningResult` + context into a `QuantumProof` and submit it.

Kept separate from `substrate_client.py` so the conversion logic — including
the float-to-milli boundary — stays reviewable independently of the RPC
plumbing. Phase 4's controller orchestrates: it calls into here, but neither
the client nor the controller knows about the proof shape.
"""
from __future__ import annotations

from typing import List, Tuple

from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.signer import Signer
from shared.substrate_client import SubstrateClient
from shared.substrate_types import ExtrinsicReceipt, SubstrateMiningContext


logger = get_logger("substrate_submitter")


# Same convention the Rust pallet uses everywhere: milli = ×1000.
MILLI_SCALE: int = 1000


def encode_quantum_proof(
    result: MiningResult,
    context: SubstrateMiningContext,
) -> dict:
    """Build the `QuantumProof` payload expected by `QuantumPow.submit_proof`.

    Floats from the miner are converted to milli-precision i32 at this
    boundary. Spin values are normalized to ±1 because the Rust pallet rejects
    anything else (`InvalidSpinValues`).

    The returned dict matches the SCALE struct field-for-field — `compose_call`
    serializes it for us.
    """
    if not result.solutions:
        raise ValueError("MiningResult has no solutions to submit")
    if len(result.salt) != 32:
        raise ValueError(
            f"MiningResult.salt must be 32 bytes, got {len(result.salt)}"
        )

    nodes = [int(n) for n in (result.node_list or context.nodes)]
    edges = _coerce_edges(result.edge_list) if result.edge_list else list(context.edges)
    solutions = [_normalize_spins(sol) for sol in result.solutions]
    h_values_milli = [int(round(v * MILLI_SCALE)) for v in context.h_values]

    return {
        "topology_hash": "0x" + context.topology_hash.hex(),
        "nonce": int(result.nonce),
        "salt": "0x" + result.salt.hex(),
        "nodes": nodes,
        "edges": [(int(u), int(v)) for u, v in edges],
        "solutions": solutions,
        "h_values": h_values_milli,
    }


async def submit_proof(
    client: SubstrateClient,
    signer: Signer,
    result: MiningResult,
    context: SubstrateMiningContext,
    wait_for: str = "inblock",
) -> ExtrinsicReceipt:
    """Encode and submit one proof. Returns the typed extrinsic receipt.

    Stale-submission errors (`InvalidNonce`, `TopologyNotRegistered`,
    `InvalidTopology`, `ProofLimitReached`) come back as `receipt.error`. The
    controller is responsible for classifying them — see Phase 4 plan.
    """
    proof = encode_quantum_proof(result, context)
    logger.info(
        "submitting QuantumPow.submit_proof: nonce=%d salt=%s solutions=%d block=%d",
        proof["nonce"],
        proof["salt"][:18] + "...",
        len(proof["solutions"]),
        context.block_number,
    )
    return await client.submit_extrinsic(
        call_module="QuantumPow",
        call_function="submit_proof",
        call_params={"proof": proof},
        signer=signer,
        wait_for=wait_for,
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _normalize_spins(solution) -> List[int]:
    """Map sampler-emitted spin values (0/1 or -1/+1) to ±1.

    Existing miners produce mixed conventions. Map both the Boolean
    {0, 1} and the spin {-1, +1} representations to {-1, +1} since the Rust
    pallet rejects anything else.
    """
    out: List[int] = []
    for s in solution:
        v = int(s)
        if v == 0:
            out.append(-1)
        elif v == 1 or v == -1:
            out.append(v)
        else:
            raise ValueError(
                f"spin value {v!r} is not 0, 1, or -1; cannot normalize"
            )
    return out


def _coerce_edges(edges) -> List[Tuple[int, int]]:
    """Accept edges as list[tuple] or list[list[int]]; emit list[tuple]."""
    out: List[Tuple[int, int]] = []
    for e in edges:
        if len(e) != 2:
            raise ValueError(f"edge must have 2 endpoints, got {e!r}")
        u, v = e
        out.append((int(u), int(v)))
    return out


__all__ = [
    "encode_quantum_proof",
    "submit_proof",
    "MILLI_SCALE",
]
