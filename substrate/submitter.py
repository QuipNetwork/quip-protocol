"""Encode a `MiningResult` + context into a `QuantumProof` and submit it.

Kept separate from `substrate_client.py` so the conversion logic — including
the float-to-milli boundary — stays reviewable independently of the RPC
plumbing. Phase 4's controller orchestrates: it calls into here, but neither
the client nor the controller knows about the proof shape.

Post-MR-!20: the on-chain ``QuantumProof`` carries only
``{topology_hash, nonce (U256), salt ([u8; 32]), solutions (BoundedVec<BoundedVec<u8>>)}``.
Nodes, edges, and h-values are looked up from ``RegisteredTopologies`` by
``topology_hash``. Each solution is bit-packed under the registered
``allowed_spin_values`` spec (1 byte per 8 binary spins).
"""
from __future__ import annotations

from typing import List

from shared.allowed_value_spec import MILLI_SCALE as _MILLI_SCALE
from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from shared.packed_solution import pack_solution
from shared.signer import Signer
from substrate.client import SubstrateClient
from substrate.pool_client import PoolClient
from substrate.types import ExtrinsicReceipt, SubstrateMiningContext


logger = get_logger("substrate_submitter")


# Same convention the Rust pallet uses everywhere: milli = ×1000.
MILLI_SCALE: int = _MILLI_SCALE


def encode_quantum_proof(
    result: MiningResult,
    context: SubstrateMiningContext,
) -> dict:
    """Build the `QuantumProof` payload expected by `QuantumPow.submit_proof`.

    The returned dict matches the post-MR-!20 SCALE struct field-for-field —
    ``compose_call`` serializes it for us.

    Each solution is bit-packed against ``context.allowed_spin_values`` via
    :func:`shared.packed_solution.pack_solution`. For the default binary
    spin spec this is 1 bit per spin; the resulting bytes are submitted as a
    ``BoundedVec<u8, MaxNodes>`` per solution.
    """
    if not result.solutions:
        raise ValueError("MiningResult has no solutions to submit")
    if len(result.salt) != 32:
        raise ValueError(
            f"MiningResult.salt must be 32 bytes, got {len(result.salt)}"
        )
    if not isinstance(result.nonce, (bytes, bytearray)) or len(result.nonce) != 32:
        raise ValueError(
            "MiningResult.nonce must be the 32-byte big-endian U256 digest"
        )

    num_spins = len(context.nodes)
    spin_spec = context.allowed_spin_values

    packed_solutions: List[bytes] = []
    for sol in result.solutions:
        # Convert ±1 spins back to milli-precision values matching the spec.
        # (For the default binary spin set {-1000, +1000}, this is just
        # multiplying by 1000.)
        milli_spins = [_milli_from_spin(s) for s in _normalize_spins(sol)]
        if len(milli_spins) != num_spins:
            raise ValueError(
                f"solution length {len(milli_spins)} does not match topology "
                f"node count {num_spins}"
            )
        packed_solutions.append(pack_solution(milli_spins, spin_spec))

    # NonceOf is `U256` in the runtime metadata — scalecodec's U256 encoder
    # rejects list-of-ints (gates on `0 <= int(value) <= 2**256 - 1`). The
    # MiningResult.nonce is the canonical 32-byte big-endian digest, so pass
    # the big-endian int directly.
    # SaltOf is `[u8; 32]` — accepts a list-of-ints.
    # SolutionsOf is `BoundedVec<BoundedVec<u8>>` — the outer BoundedVec is
    # exposed as a 1-tuple composite; each inner BoundedVec<u8> is also a
    # 1-tuple composite carrying the packed byte list.
    return {
        "topology_hash": "0x" + context.topology_hash.hex(),
        "nonce": int.from_bytes(result.nonce, "big"),
        "salt": [int(b) for b in result.salt],
        "solutions": ([([int(b) for b in packed],) for packed in packed_solutions],),
    }


def _milli_from_spin(spin: int) -> int:
    """Map a ±1 spin to its milli-precision representation (±MILLI_SCALE)."""
    if spin == 1:
        return MILLI_SCALE
    if spin == -1:
        return -MILLI_SCALE
    raise ValueError(f"normalized spin must be ±1, got {spin}")


async def submit_proof(
    build_client: SubstrateClient,
    pool_client: PoolClient,
    signer: Signer,
    result: MiningResult,
    context: SubstrateMiningContext,
    wait_for: str = "inblock",
) -> ExtrinsicReceipt:
    """Encode and submit one proof. Returns the typed extrinsic receipt.

    ``build_client`` composes + signs the extrinsic in the parent process
    (key material never crosses an IPC boundary). ``pool_client`` ships
    the hex through the swap-aware validator pool — a mid-flight swap
    raises ``ValidatorSwapped`` and the caller decides whether to
    re-sign with a fresh nonce.

    Stale-submission errors (`InvalidNonce`, `TopologyNotRegistered`,
    `InvalidTopology`, `ProofLimitReached`) come back as `receipt.error`. The
    controller is responsible for classifying them — see Phase 4 plan.
    """
    proof = encode_quantum_proof(result, context)
    # Salt + solutions are wrapped as composite tuples (see
    # encode_quantum_proof). The raw bytes/lists are at index 0 of each.
    logger.info(
        "submitting QuantumPow.submit_proof: nonce=0x%s... salt=0x%s... "
        "solutions=%d last_proof_block_hash=0x%s... topology_hash=%s",
        result.nonce.hex()[:16],
        result.salt.hex()[:8],
        len(proof["solutions"][0]),
        context.last_proof_block_hash.hex()[:16],
        proof["topology_hash"],
    )
    extrinsic_hex = await build_client.build_signed_extrinsic(
        call_module="QuantumPow",
        call_function="submit_proof",
        call_params={"proof": proof},
        signer=signer,
    )
    return await pool_client.submit_signed_extrinsic(
        extrinsic_hex, wait_for=wait_for,
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


__all__ = [
    "encode_quantum_proof",
    "submit_proof",
    "MILLI_SCALE",
]
