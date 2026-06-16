"""Async client shim that exposes the SubstrateClient method surface
but routes calls through a ValidatorPool.

Preserves familiar ``await client.X(...)`` ergonomics while gaining the
pool's hot-active swap + idempotent-retry semantics for free: every
method forwards to ``await self._pool.send(op_name, kwargs_dict)``.

Scope:
    * READ ops (idempotent — pool auto-retries across swaps).
    * ``submit_signed_extrinsic`` — non-idempotent write. The pool raises
      ``ValidatorSwapped`` on mid-flight swap and lets the caller decide
      whether to re-sign (with a fresh nonce) and resubmit.

Signing stays in the calling process: callers pair this submit path
with ``SubstrateClient.build_signed_extrinsic`` so key material never
crosses the mp.Queue IPC boundary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from substrate.pool import ValidatorPool


class PoolClient:
    """Async RPC client whose methods route through a ``ValidatorPool``.

    Each method forwards to ``await self._pool.send(op, kwargs)``. The
    pool handles hot-active swap on connection failure and auto-retries
    idempotent ops across swaps up to ``max_swap_retries``.

    Args:
        pool: The ``ValidatorPool`` to route through.
    """

    def __init__(self, pool: "ValidatorPool") -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Best-block / finalized-head primitives.
    # ------------------------------------------------------------------

    async def ensure_connected(self) -> bool:
        """Pre-submit liveness probe routed through the pool.

        Forwards to the active validator child's
        :meth:`SubstrateClient.ensure_connected`. As an idempotent op, a
        dead socket surfaces as a connection-class error that the pool
        turns into a hot-active swap — so a stale validator is replaced
        before the (non-idempotent) submit that follows. Returns the
        child's bool (``True`` = existing connection answered).
        """
        return await self._pool.send("ensure_connected", {})

    async def get_head(self) -> bytes:
        return await self._pool.send("get_head", {})

    async def get_finalized_head(self) -> bytes:
        return await self._pool.send("get_finalized_head", {})

    async def get_block_number(self, at: Optional[bytes] = None) -> int:
        return await self._pool.send("get_block_number", {"at": at})

    # ------------------------------------------------------------------
    # Runtime API: mining snapshot.
    # ------------------------------------------------------------------

    async def get_mining_snapshot(
        self,
        *,
        miner_account_bytes: bytes,
        at: Optional[bytes] = None,
        topology_hash: Optional[bytes] = None,
    ):
        return await self._pool.send(
            "get_mining_snapshot",
            {
                "miner_account_bytes": miner_account_bytes,
                "at": at,
                "topology_hash": topology_hash,
            },
        )

    # ------------------------------------------------------------------
    # Storage queries.
    # ------------------------------------------------------------------

    async def query_miner(self, account: bytes):
        return await self._pool.send("query_miner", {"account": account})

    async def query_proofs_submitted(self, account: bytes):
        """Return the miner's cumulative ``proofs_submitted`` counter, or None.

        Routes through the pool's idempotent-retry path; returns None when the
        miner is not registered. May raise if the pool exhausts its swap
        retries — submit-path callers should use the controller's
        ``_query_proofs_submitted_safe`` wrapper, which centralizes the
        no-raise guarantee.
        """
        return await self._pool.send("query_proofs_submitted", {"account": account})

    async def query_difficulty(self):
        return await self._pool.send("query_difficulty", {})

    async def query_current_difficulty(self, at_block: Optional[int] = None):
        return await self._pool.send(
            "query_current_difficulty", {"at_block": at_block}
        )

    async def query_last_proof_block_number(self):
        return await self._pool.send("query_last_proof_block_number", {})

    async def query_block_timestamp_ms(self, block_hash: bytes) -> "Optional[int]":
        return await self._pool.send(
            "query_block_timestamp_ms", {"block_hash": block_hash}
        )

    async def query_pow_constants(self):
        return await self._pool.send("query_pow_constants", {})

    async def query_winning_solution(self, block_number: int):
        return await self._pool.send(
            "query_winning_solution", {"block_number": block_number}
        )

    async def query_winning_solution_count(self) -> int:
        """Return the recorded ``QuantumPow.WinningSolutions`` count.

        The controller uses ``count + 1`` as the chain-global ordinal of the
        solution currently being mined (the on-disk archive key). Routes
        through the pool's idempotent-retry path.
        """
        return await self._pool.send("query_winning_solution_count", {})

    async def query_latest_qblock_id(self):
        return await self._pool.send("query_latest_qblock_id", {})

    async def query_balance(self, account: bytes) -> int:
        return await self._pool.send("query_balance", {"account": account})

    async def query_solver(self, account: bytes):
        return await self._pool.send("query_solver", {"account": account})

    async def query_job_order(self, order_id: int):
        return await self._pool.send("query_job_order", {"order_id": order_id})

    async def get_events_at(self, block_hash: bytes) -> list[dict]:
        return await self._pool.send("get_events_at", {"block_hash": block_hash})

    # ------------------------------------------------------------------
    # Submission: sign in parent, ship hex.
    # ------------------------------------------------------------------

    async def submit_signed_extrinsic(
        self,
        extrinsic_hex: str,
        wait_for: str = "inblock",
    ):
        """Submit pre-built signed extrinsic via the active validator child.

        The pool routes this through the swap-aware ``send()`` path. As a
        non-idempotent op, a mid-flight swap raises ``ValidatorSwapped``
        — the caller must re-sign (with a fresh nonce) and resubmit.
        """
        return await self._pool.send(
            "submit_signed_extrinsic",
            {"extrinsic_hex": extrinsic_hex, "wait_for": wait_for},
        )
