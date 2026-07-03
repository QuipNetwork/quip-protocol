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

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, List, Optional

from substrateinterface.exceptions import SubstrateRequestException

from shared.allowed_value_spec import MILLI_SCALE
from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from substrate.validator_handle import ValidatorSwapped
from shared.packed_solution import pack_solution
from shared.signer import Signer
from substrate.client import ExtrinsicRejected, SubstrateClient
from substrate.pool_client import PoolClient
from substrate.types import ExtrinsicReceipt, SubstrateMiningContext


logger = get_logger("substrate_submitter")


# Pallet error names (matched as substrings in receipt.error) that a fire
# loop should RETRY rather than abandon — the proof is sound, only the
# *timing* was off, and a later attempt can still win the round:
#   - InsufficientEnergy: fired a hair early, before decay made the
#     threshold reachable. Validity only improves with elapsed blocks.
#   - ProofLimitReached: the block was already full of proofs this slot;
#     the next block has room.
RETRYABLE_SUBMISSION_ERRORS = (
    "InsufficientEnergy",
    "ProofLimitReached",
)

# Pallet error names that mean "this round is over for us" — the nonce is
# bound to a round that has advanced, so resubmitting the same proof can
# never succeed. Stop this round's fire loop and re-mine the next snapshot.
ROUND_STALE_SUBMISSION_ERRORS = (
    "InvalidNonce",
    "TopologyNotRegistered",
    "InvalidTopology",
)

# Pallet error names that mean the candidate/config is genuinely bad —
# retrying or waiting cannot help. Stop.
FATAL_SUBMISSION_ERRORS = (
    "InsufficientSolutions",
    "InsufficientDiversity",
    "MinerNotRegistered",
    "BadSignature",
    "BadProof",
)


class SubmitRetryAction(str, Enum):
    """Outcome taxonomy a fire loop branches on after one submit attempt.

    Inherits from ``str`` so ``==`` against the literal values keeps
    working while typo'd comparisons fail a type check. Task 6's
    anticipatory-submission loop dispatches on these:

      - ``SUCCESS``: the chain accepted the proof. Stop, we won (or at
        least it's in a block cleanly).
      - ``RETRY``: transient/timing failure (connection drop, RPC error,
        ``InsufficientEnergy``, ``ProofLimitReached``). Loop again after
        backoff; reconnect first.
      - ``STOP_ROUND_STALE``: the round advanced — the nonce is dead, so
        resubmitting can never succeed (``InvalidNonce`` /
        ``TopologyNotRegistered`` / ``InvalidTopology``). Stop this round,
        re-mine the next snapshot.
      - ``STOP_FATAL``: the candidate or config is genuinely bad
        (``InsufficientSolutions`` / ``InsufficientDiversity``), or the
        error string is unrecognized. Stop; retrying cannot help.
    """

    SUCCESS = "success"
    RETRY = "retry"
    STOP_ROUND_STALE = "stop_round_stale"
    STOP_FATAL = "stop_fatal"


@dataclass(frozen=True)
class SubmitResult:
    """Typed result of :func:`submit_with_retry`.

    ``action`` is the terminal taxonomy after the loop exhausted retries
    or hit a stop condition. ``receipt`` is the last receipt seen (``None``
    if every attempt raised before producing one). ``error`` carries the
    last exception text or receipt error for telemetry. ``attempts`` is the
    number of submit attempts actually made.
    """

    action: SubmitRetryAction
    receipt: Optional[ExtrinsicReceipt] = None
    error: Optional[str] = None
    attempts: int = 0


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
    *,
    tip: int = 0,
    ensure_live: bool = True,
) -> ExtrinsicReceipt:
    """Encode and submit one proof. Returns the typed extrinsic receipt.

    ``build_client`` composes + signs the extrinsic in the parent process
    (key material never crosses an IPC boundary). ``pool_client`` ships
    the hex through the swap-aware validator pool — a mid-flight swap
    raises ``ValidatorSwapped`` and the caller decides whether to
    re-sign with a fresh nonce.

    ``tip`` (plancks) is threaded into the signed extrinsic to raise its
    transaction-pool priority — the chain takes the lowest-energy proof
    *within a block*, so winning the inclusion race matters. ``tip=0``
    reproduces the exact pre-tip wire bytes.

    ``ensure_live`` runs a cheap liveness probe + reconnect on both clients
    immediately before submitting, so a websocket that went idle during a
    long mining gap is replaced *before* the proof submit (instead of the
    submit eating the failover round-trip). Set ``False`` only in tests or
    when the caller has already warmed the connection.

    Stale-submission errors (`InvalidNonce`, `TopologyNotRegistered`,
    `InvalidTopology`, `ProofLimitReached`) come back as `receipt.error`. The
    controller is responsible for classifying them — see Phase 4 plan.
    """
    if ensure_live:
        # Reconnect FIRST so the submit doesn't pay the failover cost on
        # the critical path. build_client signs locally; pool_client ships.
        await build_client.ensure_connected()
        await pool_client.ensure_connected()
    proof = encode_quantum_proof(result, context)
    # Salt + solutions are wrapped as composite tuples (see
    # encode_quantum_proof). The raw bytes/lists are at index 0 of each.
    logger.info(
        "submitting QuantumPow.submit_proof: nonce=0x%s... salt=0x%s... "
        "solutions=%d last_proof_block_hash=0x%s... topology_hash=%s tip=%d",
        result.nonce.hex()[:16],
        result.salt.hex()[:8],
        len(proof["solutions"][0]),
        context.last_proof_block_hash.hex()[:16],
        proof["topology_hash"],
        tip,
    )
    extrinsic_hex = await build_client.build_signed_extrinsic(
        call_module="QuantumPow",
        call_function="submit_proof",
        call_params={"proof": proof},
        signer=signer,
        tip=tip,
    )
    return await pool_client.submit_signed_extrinsic(
        extrinsic_hex, wait_for=wait_for,
    )


def _classify_receipt(receipt: ExtrinsicReceipt) -> SubmitRetryAction:
    """Map a receipt to a fire-loop action. See :class:`SubmitRetryAction`."""
    if receipt.is_success:
        return SubmitRetryAction.SUCCESS
    error = receipt.error or ""
    for name in RETRYABLE_SUBMISSION_ERRORS:
        if name in error:
            return SubmitRetryAction.RETRY
    for name in ROUND_STALE_SUBMISSION_ERRORS:
        if name in error:
            return SubmitRetryAction.STOP_ROUND_STALE
    for name in FATAL_SUBMISSION_ERRORS:
        if name in error:
            return SubmitRetryAction.STOP_FATAL
    # Unknown receipt error: stop fatally rather than hammer the chain with
    # a proof we can't classify. The raw text is logged so operators see it.
    logger.error(
        "submit_with_retry: unrecognized submission error; stopping fatally: %r",
        error,
    )
    return SubmitRetryAction.STOP_FATAL


async def submit_with_retry(
    build_client: SubstrateClient,
    pool_client: PoolClient,
    signer: Signer,
    result: MiningResult,
    context: SubstrateMiningContext,
    *,
    tip: int = 0,
    wait_for: str = "inblock",
    max_retries: int = 3,
    retry_backoff_ms: int = 250,
    sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
) -> SubmitResult:
    """Submit a proof, classifying each attempt; never raise on one failure.

    A resilient wrapper around :func:`submit_proof` for Task 6's
    anticipatory fire loop. It LOOPS on transient and timing failures and
    returns a typed :class:`SubmitResult` — the caller branches on
    ``result.action`` (see :class:`SubmitRetryAction`).

    Retry policy:
      - Transient exceptions (``BrokenPipeError`` / ``TimeoutError`` /
        ``ConnectionError`` / ``OSError`` / generic ``RuntimeError`` from an
        RPC, and ``SubstrateRequestException`` txpool rejections such as
        1010 stale-nonce races with a sibling extrinsic from the same
        account) → RETRY. ``submit_proof`` already reconnects first via its
        ``ensure_live`` probe.
      - ``ValidatorSwapped`` (the pool hot-swapped validators mid-submit) →
        RETRY. This is safe here *only* because each attempt re-composes
        and re-signs the extrinsic from scratch: ``submit_proof`` calls
        ``build_client.build_signed_extrinsic`` every iteration, which
        re-reads the account nonce against the (now swapped-in) validator
        and produces a fresh signature. We never re-broadcast a stale
        pre-signed hex — that would carry a dead nonce and be rejected.
      - Receipt errors → classified by :func:`_classify_receipt`:
        ``InsufficientEnergy`` / ``ProofLimitReached`` → RETRY;
        ``InvalidNonce`` / ``TopologyNotRegistered`` / ``InvalidTopology``
        → STOP_ROUND_STALE; ``InsufficientSolutions`` /
        ``InsufficientDiversity`` (and other bad-config) → STOP_FATAL.
      - SUCCESS → return immediately.

    ``max_retries`` bounds the number of *additional* attempts after the
    first, so total attempts ≤ ``max_retries + 1``. ``retry_backoff_ms`` is
    the per-attempt linear backoff; ``sleeper`` is injected so tests run
    with zero real delay — defaults to :func:`asyncio.sleep`.

    Returns the terminal :class:`SubmitResult`; does not raise for a single
    transient failure (transient + ``ValidatorSwapped`` failures are
    retried, then returned as ``RETRY``; classified stops are returned, not
    raised). The only exceptions that propagate are programming errors
    (e.g. a ``ValueError`` from proof encoding) — never a network/swap
    failure.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    sleep = sleeper if sleeper is not None else asyncio.sleep
    last_error: Optional[str] = None
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            # Watch-timeout cap: a submit-and-watch subscription can die
            # without ever delivering a terminal status (observed live —
            # it froze the anticipatory fire timer mid-await, silently
            # stopping all head/result processing while the run task
            # looked alive). asyncio.TimeoutError is transient, so a dead
            # watch becomes a fresh-compose retry instead of a hang.
            receipt = await asyncio.wait_for(
                submit_proof(
                    build_client,
                    pool_client,
                    signer,
                    result,
                    context,
                    wait_for=wait_for,
                    tip=tip,
                ),
                timeout=_SUBMIT_WATCH_TIMEOUT_S,
            )
        except _TRANSIENT_SUBMIT_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "submit_with_retry: transient failure on attempt %d/%d: %s",
                attempt, total_attempts, last_error,
            )
            terminal = await _retry_or_exhausted(
                attempt, total_attempts, None, last_error, retry_backoff_ms, sleep,
            )
            if terminal is None:
                continue
            return terminal

        action = _classify_receipt(receipt)
        if action is SubmitRetryAction.RETRY:
            last_error = receipt.error
            logger.info(
                "submit_with_retry: retryable receipt on attempt %d/%d: %s",
                attempt, total_attempts, receipt.error,
            )
            terminal = await _retry_or_exhausted(
                attempt, total_attempts, receipt, receipt.error, retry_backoff_ms, sleep,
            )
            if terminal is None:
                continue
            return terminal
        return SubmitResult(
            action=action,
            receipt=receipt,
            error=receipt.error,
            attempts=attempt,
        )


# Exception classes that mean "the submit failed in a way a later attempt
# might survive" — connection drops, timeouts, the RuntimeError the client
# raises for RPC-layer errors, and a mid-submit validator swap. Caught by
# submit_with_retry so a single network blip never tears down the fire loop.
# ValidatorSwapped is only safe to retry because submit_proof re-signs from
# scratch each attempt (fresh nonce against the swapped-in validator) — see
# the submit_with_retry docstring.
_TRANSIENT_SUBMIT_EXCEPTIONS = (
    BrokenPipeError,
    ConnectionError,
    TimeoutError,
    OSError,
    asyncio.TimeoutError,
    RuntimeError,
    ValidatorSwapped,
    # Txpool-level rejections raised by substrate-interface (e.g. 1010
    # "Transaction is outdated" when a sibling extrinsic from the same
    # account — a participation remark, or the result-path submission of
    # the same candidate — consumed the composed nonce first). Safe to
    # retry for the same reason ValidatorSwapped is: every attempt
    # re-composes and re-signs with a freshly-read nonce. Without this,
    # the exception escaped `_fire_preview` AFTER the mid-fire mark was
    # set, permanently wedging the round's submissions (found live by
    # tests/test_mempool_priority_integration.py).
    SubstrateRequestException,
    # Terminal watch statuses (usurped / dropped / retracted / ...). With
    # the mempool submitter tipping to outrank same-account pow traffic,
    # usurpation of a pooled pow extrinsic is a NORMAL pool outcome — the
    # retry recomposes against the advanced nonce and coexists.
    ExtrinsicRejected,
)

# Hard cap on one submit attempt (compose + submit + watch-to-inclusion).
# Inclusion is 1-2 blocks; a watch alive past this is a dead subscription.
_SUBMIT_WATCH_TIMEOUT_S = 90.0


def _backoff_seconds(attempt: int, retry_backoff_ms: int) -> float:
    """Linear backoff in seconds for the Nth attempt (1-indexed)."""
    return (retry_backoff_ms * attempt) / 1000.0


async def _retry_or_exhausted(
    attempt: int,
    total_attempts: int,
    receipt: Optional[ExtrinsicReceipt],
    error: Optional[str],
    retry_backoff_ms: int,
    sleep: Callable[[float], Awaitable[None]],
) -> Optional[SubmitResult]:
    """Sleep and return None (caller should continue) or return a terminal RETRY result.

    Shared by the transient-exception branch and the retryable-receipt branch of
    :func:`submit_with_retry` — both share the same sleep+continue / else-RETRY shape.
    Returns ``None`` when another attempt should be made (sleep has already fired);
    returns a :class:`SubmitResult` when retries are exhausted.
    """
    if attempt < total_attempts:
        await sleep(_backoff_seconds(attempt, retry_backoff_ms))
        return None
    return SubmitResult(
        action=SubmitRetryAction.RETRY,
        receipt=receipt,
        error=error,
        attempts=attempt,
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
    "submit_with_retry",
    "SubmitRetryAction",
    "SubmitResult",
    "RETRYABLE_SUBMISSION_ERRORS",
    "ROUND_STALE_SUBMISSION_ERRORS",
    "FATAL_SUBMISSION_ERRORS",
    "MILLI_SCALE",
]
