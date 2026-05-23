"""Async client shim that exposes the SubstrateClient method surface
but routes calls through a ValidatorPool.

The controller has ~16 ``await self.client.X(...)`` call sites. Migrating
each one to ``await self.pool.send("X", {...})`` would touch a lot of code
and lose the type-checked attribute access. PoolClient is a thin shim
that preserves the existing call-site shape while gaining the pool's
hot-active swap + idempotent-retry semantics for free: every method here
forwards to ``await self._pool.send(op_name, kwargs_dict)``.

Scope:
    * READ ops only (idempotent — pool auto-retries across swaps).
    * Mirrors the subset of ``SubstrateClient`` methods the controller
      actually uses.

NOT in scope yet:
    * ``submit_extrinsic`` — signing requires sending a ``Signer`` (which
      holds key material) across the mp.Queue IPC boundary, which is
      both a pickling problem and a security smell. Submitter migration
      is a separate task: refactor to sign in the parent and ship raw
      signed bytes to the child via a new ``submit_signed_extrinsic``
      RPC. Until then, callers needing submit_extrinsic must hold a
      direct ``SubstrateClient`` connection.
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

    async def query_difficulty(self):
        return await self._pool.send("query_difficulty", {})

    async def query_current_difficulty(self, at_block: Optional[int] = None):
        return await self._pool.send(
            "query_current_difficulty", {"at_block": at_block}
        )

    async def query_last_proof_block_number(self):
        return await self._pool.send("query_last_proof_block_number", {})

    async def query_pow_constants(self):
        return await self._pool.send("query_pow_constants", {})

    async def query_winning_solution(self, block_number: int):
        return await self._pool.send(
            "query_winning_solution", {"block_number": block_number}
        )

    async def query_balance(self, account: bytes) -> int:
        return await self._pool.send("query_balance", {"account": account})

    async def query_solver(self, account: bytes):
        return await self._pool.send("query_solver", {"account": account})

    async def query_job_order(self, order_id: int):
        return await self._pool.send("query_job_order", {"order_id": order_id})

    async def get_events_at(self, block_hash: bytes) -> list[dict]:
        return await self._pool.send("get_events_at", {"block_hash": block_hash})

    # ------------------------------------------------------------------
    # Submitter path — see module docstring "NOT in scope yet".
    # ------------------------------------------------------------------

    async def submit_extrinsic(self, *args, **kwargs):
        raise NotImplementedError(
            "PoolClient.submit_extrinsic is not implemented: signing "
            "requires shipping signer key material across IPC. Use a "
            "direct SubstrateClient connection for submission until the "
            "submitter is refactored to sign in the parent and send "
            "raw signed bytes through the pool."
        )
