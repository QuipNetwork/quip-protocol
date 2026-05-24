"""Tests for substrate.pool_client.PoolClient.

PoolClient is a shim that exposes the SubstrateClient method surface but
routes each call through a ValidatorPool. Its only job is to forward
``(method_name, kwargs)`` correctly; everything interesting happens in
the pool.
"""
from __future__ import annotations

from typing import Any

import pytest

from substrate.pool_client import PoolClient


class _RecordingPool:
    """Captures the (op, args) of every send() call; returns a scripted result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.scripted: dict[str, Any] = {}

    async def send(self, op: str, args: dict) -> Any:
        self.calls.append((op, args))
        if op in self.scripted:
            value = self.scripted[op]
            if isinstance(value, Exception):
                raise value
            return value
        return None


@pytest.mark.asyncio
async def test_get_head_forwards_empty_args():
    pool = _RecordingPool()
    pool.scripted["get_head"] = b"\xab" * 32
    client = PoolClient(pool)
    result = await client.get_head()
    assert result == b"\xab" * 32
    assert pool.calls == [("get_head", {})]


@pytest.mark.asyncio
async def test_get_finalized_head_forwards_empty_args():
    pool = _RecordingPool()
    pool.scripted["get_finalized_head"] = b"\xcd" * 32
    client = PoolClient(pool)
    assert await client.get_finalized_head() == b"\xcd" * 32
    assert pool.calls == [("get_finalized_head", {})]


@pytest.mark.asyncio
async def test_get_block_number_default_at_is_none():
    pool = _RecordingPool()
    pool.scripted["get_block_number"] = 99
    client = PoolClient(pool)
    assert await client.get_block_number() == 99
    assert pool.calls == [("get_block_number", {"at": None})]


@pytest.mark.asyncio
async def test_get_block_number_with_at():
    pool = _RecordingPool()
    pool.scripted["get_block_number"] = 100
    client = PoolClient(pool)
    block_hash = b"\x01" * 32
    assert await client.get_block_number(at=block_hash) == 100
    assert pool.calls == [("get_block_number", {"at": block_hash})]


@pytest.mark.asyncio
async def test_get_mining_snapshot_forwards_kwargs():
    pool = _RecordingPool()
    pool.scripted["get_mining_snapshot"] = "snapshot-sentinel"
    client = PoolClient(pool)
    result = await client.get_mining_snapshot(
        miner_account_bytes=b"\x00" * 32,
        at=None,
        topology_hash=b"\xab" * 32,
    )
    assert result == "snapshot-sentinel"
    assert pool.calls == [
        (
            "get_mining_snapshot",
            {
                "miner_account_bytes": b"\x00" * 32,
                "at": None,
                "topology_hash": b"\xab" * 32,
            },
        )
    ]


@pytest.mark.asyncio
async def test_query_methods_forward_single_kwarg():
    pool = _RecordingPool()
    pool.scripted["query_miner"] = "miner-info"
    pool.scripted["query_winning_solution"] = "solution"
    pool.scripted["query_balance"] = 1234
    pool.scripted["query_solver"] = "solver"
    pool.scripted["query_job_order"] = "order"
    pool.scripted["get_events_at"] = [{"event": "x"}]
    client = PoolClient(pool)

    assert await client.query_miner(b"\x01" * 32) == "miner-info"
    assert await client.query_winning_solution(42) == "solution"
    assert await client.query_balance(b"\x01" * 32) == 1234
    assert await client.query_solver(b"\x01" * 32) == "solver"
    assert await client.query_job_order(7) == "order"
    assert await client.get_events_at(b"\x02" * 32) == [{"event": "x"}]

    assert pool.calls == [
        ("query_miner", {"account": b"\x01" * 32}),
        ("query_winning_solution", {"block_number": 42}),
        ("query_balance", {"account": b"\x01" * 32}),
        ("query_solver", {"account": b"\x01" * 32}),
        ("query_job_order", {"order_id": 7}),
        ("get_events_at", {"block_hash": b"\x02" * 32}),
    ]


@pytest.mark.asyncio
async def test_query_difficulty_forwards_no_args():
    pool = _RecordingPool()
    pool.scripted["query_difficulty"] = "diff"
    client = PoolClient(pool)
    assert await client.query_difficulty() == "diff"
    assert pool.calls == [("query_difficulty", {})]


@pytest.mark.asyncio
async def test_query_current_difficulty_default_none():
    pool = _RecordingPool()
    pool.scripted["query_current_difficulty"] = "diff"
    client = PoolClient(pool)
    assert await client.query_current_difficulty() == "diff"
    assert pool.calls == [("query_current_difficulty", {"at_block": None})]


@pytest.mark.asyncio
async def test_query_current_difficulty_with_block():
    pool = _RecordingPool()
    pool.scripted["query_current_difficulty"] = "diff-100"
    client = PoolClient(pool)
    assert await client.query_current_difficulty(at_block=100) == "diff-100"
    assert pool.calls == [("query_current_difficulty", {"at_block": 100})]


@pytest.mark.asyncio
async def test_pool_exception_propagates():
    pool = _RecordingPool()
    pool.scripted["get_head"] = RuntimeError("pool failure")
    client = PoolClient(pool)
    with pytest.raises(RuntimeError, match="pool failure"):
        await client.get_head()


@pytest.mark.asyncio
async def test_submit_signed_extrinsic_forwards_hex_and_wait_for():
    pool = _RecordingPool()
    pool.scripted["submit_signed_extrinsic"] = "receipt-sentinel"
    client = PoolClient(pool)
    result = await client.submit_signed_extrinsic(
        "0xdeadbeef", wait_for="finalized",
    )
    assert result == "receipt-sentinel"
    assert pool.calls == [
        (
            "submit_signed_extrinsic",
            {"extrinsic_hex": "0xdeadbeef", "wait_for": "finalized"},
        )
    ]


@pytest.mark.asyncio
async def test_submit_signed_extrinsic_default_wait_is_inblock():
    pool = _RecordingPool()
    pool.scripted["submit_signed_extrinsic"] = "receipt"
    client = PoolClient(pool)
    await client.submit_signed_extrinsic("0xabc123")
    assert pool.calls == [
        (
            "submit_signed_extrinsic",
            {"extrinsic_hex": "0xabc123", "wait_for": "inblock"},
        )
    ]
