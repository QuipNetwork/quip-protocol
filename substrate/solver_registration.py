"""Guard D+ — query-first, race-tolerant mempool solver registration.

`ensure_solver_registered` is the startup guard for mempool participation:
verify (or create) the signer's `QuantumComputeMempool` solver registration
and report the result as an outcome enum. Chain-side failures NEVER raise —
in the miner a fatal child exit triggers the supervisor's
terminate-all-siblings rule and takes pow mining down node-wide, so callers
must be able to degrade to mempool-off instead. That contract shapes every
branch:

  - Query-first: `register_solver` is not idempotent on-chain (a repeat
    call fails with `SolverAlreadyRegistered` and still burns a fee), so a
    matching registration is confirmed via `query_solver` with no extrinsic.
  - Race-tolerant: sibling children share one signer account; if the
    extrinsic loses a registration race, re-query and accept a matching
    type as success.
  - Never deregisters: `deregister_solver` resets on-chain solver stats,
    and two children wanting different types would ping-pong registrations
    every boot. A type mismatch is reported, never "fixed".
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from shared.logging_config import get_logger
from shared.signer import Signer
from substrate.mempool_types import MinerType

if TYPE_CHECKING:
    from substrate.client import SubstrateClient


logger = get_logger("solver_registration")

# Matched by substring against the extrinsic receipt error, mirroring the
# classifier convention in `substrate.mempool_miner_controller`.
_RACE_ERROR = "SolverAlreadyRegistered"


class SolverGuardOutcome(Enum):
    """Result of the Guard D+ registration check.

    ALREADY_REGISTERED / REGISTERED are success; TYPE_MISMATCH means the
    account is registered under a different `MinerType` (operator must
    `deregister-solver` explicitly); FAILED covers RPC/chain errors and
    non-race extrinsic failures.
    """

    ALREADY_REGISTERED = "already_registered"
    REGISTERED = "registered"
    TYPE_MISMATCH = "type_mismatch"
    FAILED = "failed"


def _match_or_mismatch(existing: MinerType, target: MinerType) -> SolverGuardOutcome:
    if existing == target:
        logger.info("solver guard: already registered as %s", target.name)
        return SolverGuardOutcome.ALREADY_REGISTERED
    logger.error(
        "solver guard: account registered as %s but this process wants %s; "
        "run `quip-miner deregister-solver` and re-register to change types "
        "(the guard never auto-deregisters — that resets on-chain solver stats)",
        existing.name,
        target.name,
    )
    return SolverGuardOutcome.TYPE_MISMATCH


async def ensure_solver_registered(
    client: SubstrateClient, signer: Signer, miner_kind: str
) -> SolverGuardOutcome:
    """Verify or create ``signer``'s solver registration for ``miner_kind``.

    ``miner_kind`` must be the vendor-resolved kind (e.g. ``qpu_ibm``), not
    the backend-group name, so the registered ``MinerType`` matches what job
    proposers filter on. An unknown kind raises ``ValueError`` (caller bug);
    RPC/chain failures are logged and mapped to ``FAILED``, never raised.
    """
    target = MinerType.from_kind(miner_kind)
    account = signer.account_id_bytes()

    try:
        existing = await client.query_solver(account)
    except Exception:  # noqa: BLE001 — chain-side failure must not escape
        logger.exception("solver guard: query_solver failed")
        return SolverGuardOutcome.FAILED
    if existing is not None:
        return _match_or_mismatch(existing.solver_type, target)

    try:
        receipt = await client.register_solver(signer, target)
    except Exception:  # noqa: BLE001 — chain-side failure must not escape
        logger.exception("solver guard: register_solver submit failed")
        return SolverGuardOutcome.FAILED
    if receipt.error is None:
        logger.info(
            "solver guard: registered as %s (extrinsic=%s, block=%s)",
            target.name,
            receipt.extrinsic_hash,
            receipt.block_hash,
        )
        return SolverGuardOutcome.REGISTERED

    if _RACE_ERROR in receipt.error:
        # A sibling child on the same account won the race between our
        # query and our extrinsic — re-query and accept a matching type.
        try:
            raced = await client.query_solver(account)
        except Exception:  # noqa: BLE001 — chain-side failure must not escape
            logger.exception("solver guard: re-query after registration race failed")
            return SolverGuardOutcome.FAILED
        if raced is not None:
            return _match_or_mismatch(raced.solver_type, target)
        logger.error(
            "solver guard: %s but re-query found no solver — inconsistent "
            "chain view (concurrent deregister?)",
            _RACE_ERROR,
        )
        return SolverGuardOutcome.FAILED

    logger.error("solver guard: register_solver failed: %s", receipt.error)
    return SolverGuardOutcome.FAILED
