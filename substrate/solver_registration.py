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
  - Race-tolerant: sibling children share one signer account; if an
    extrinsic loses a registration race, re-query and accept a matching
    type as success. Siblings always want the SAME type — the config
    elects one mempool owner group per account.
  - Retypes on mismatch: the config is the source of truth for the solver
    type, so a stale registration (e.g. the owner group moved from CPU to
    GPU) is converged via deregister + register. The pallet has no
    update-in-place call, so this resets the on-chain
    `solutions_submitted`/`rewards_earned` counters — accepted, because
    the pallet only ever increments them (dashboard bookkeeping, never
    read for eligibility or payout).
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
# classifier convention in `substrate.mempool_submitter`.
_RACE_ERROR = "SolverAlreadyRegistered"
_DEREGISTER_RACE_ERROR = "SolverNotRegistered"


class SolverGuardOutcome(Enum):
    """Result of the Guard D+ registration check.

    ALREADY_REGISTERED / REGISTERED / RETYPED are success (RETYPED means a
    stale-type registration was converged to the configured kind); FAILED
    covers RPC/chain errors, non-race extrinsic failures, and an active
    conflicting writer detected mid-boot.
    """

    ALREADY_REGISTERED = "already_registered"
    REGISTERED = "registered"
    RETYPED = "retyped"
    FAILED = "failed"


async def ensure_solver_registered(
    client: SubstrateClient, signer: Signer, miner_kind: str
) -> SolverGuardOutcome:
    """Verify, create, or retype ``signer``'s solver registration.

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
        if existing.solver_type == target:
            logger.info("solver guard: already registered as %s", target.name)
            return SolverGuardOutcome.ALREADY_REGISTERED
        return await _retype(client, signer, account, existing.solver_type, target)

    return await _register(
        client,
        signer,
        account,
        target,
        on_success=SolverGuardOutcome.REGISTERED,
        on_race_match=SolverGuardOutcome.ALREADY_REGISTERED,
    )


async def _retype(
    client: SubstrateClient,
    signer: Signer,
    account: bytes,
    existing: MinerType,
    target: MinerType,
) -> SolverGuardOutcome:
    """Converge a stale-type registration to ``target`` (deregister + register)."""
    logger.warning(
        "solver guard: retyping registration %s -> %s (config is the source "
        "of truth; on-chain solutions_submitted/rewards_earned counters reset)",
        existing.name,
        target.name,
    )
    try:
        receipt = await client.deregister_solver(signer)
    except Exception:  # noqa: BLE001 — chain-side failure must not escape
        logger.exception("solver guard: deregister_solver submit failed")
        return SolverGuardOutcome.FAILED
    if receipt.error is not None:
        if _DEREGISTER_RACE_ERROR not in receipt.error:
            logger.error(
                "solver guard: deregister_solver failed: %s", receipt.error
            )
            return SolverGuardOutcome.FAILED
        # A sibling child already deregistered the stale type between our
        # query and our extrinsic — proceed straight to registration.
        logger.info(
            "solver guard: %s during retype — sibling already deregistered",
            _DEREGISTER_RACE_ERROR,
        )

    return await _register(
        client,
        signer,
        account,
        target,
        on_success=SolverGuardOutcome.RETYPED,
        on_race_match=SolverGuardOutcome.RETYPED,
    )


async def _register(
    client: SubstrateClient,
    signer: Signer,
    account: bytes,
    target: MinerType,
    *,
    on_success: SolverGuardOutcome,
    on_race_match: SolverGuardOutcome,
) -> SolverGuardOutcome:
    """Submit ``register_solver`` with race tolerance.

    ``on_success`` is returned when our extrinsic lands; ``on_race_match``
    when a sibling won the race but re-query shows the matching type. A
    race followed by a NON-matching type means an active conflicting
    writer on this account — FAILED, never fought in real time (the next
    boot's query-first path retypes calmly).
    """
    try:
        receipt = await client.register_solver(signer, target)
    except Exception:  # noqa: BLE001 — chain-side failure must not escape
        logger.exception("solver guard: register_solver submit failed")
        return SolverGuardOutcome.FAILED
    if receipt.error is None:
        logger.info(
            "solver guard: %s as %s (extrinsic=%s, block=%s)",
            on_success.value,
            target.name,
            receipt.extrinsic_hash,
            receipt.block_hash,
        )
        return on_success

    if _RACE_ERROR in receipt.error:
        # A sibling child on the same account won the race between our
        # query and our extrinsic — re-query and accept a matching type.
        try:
            raced = await client.query_solver(account)
        except Exception:  # noqa: BLE001 — chain-side failure must not escape
            logger.exception("solver guard: re-query after registration race failed")
            return SolverGuardOutcome.FAILED
        if raced is None:
            logger.error(
                "solver guard: %s but re-query found no solver — inconsistent "
                "chain view (concurrent deregister?)",
                _RACE_ERROR,
            )
            return SolverGuardOutcome.FAILED
        if raced.solver_type == target:
            logger.info(
                "solver guard: lost registration race to a sibling with "
                "matching type %s",
                target.name,
            )
            return on_race_match
        logger.error(
            "solver guard: lost registration race to a writer holding %s "
            "while this process wants %s — conflicting process on this "
            "account (one account = one solver type)",
            raced.solver_type.name,
            target.name,
        )
        return SolverGuardOutcome.FAILED

    logger.error("solver guard: register_solver failed: %s", receipt.error)
    return SolverGuardOutcome.FAILED
