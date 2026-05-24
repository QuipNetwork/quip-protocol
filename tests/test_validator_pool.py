"""Tests for the rewritten substrate.pool.ValidatorPool.

The pool exposes an async RPC surface, routes calls to the currently-
active ValidatorHandle, and handles hot-active swap on connection
failure with idempotent-only auto-retry.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from substrate.pool import ValidatorPool, ValidatorSwapped
from substrate.url_failover import SubstrateUrlFailover


class _FakeHandle:
    """In-process stand-in for ValidatorHandle — no mp.Process spawned."""

    def __init__(self, url: str, *, behaviour: dict[str, Any] | None = None) -> None:
        self.url = url
        self.is_shutdown = False
        self._behaviour = behaviour or {}
        self.calls: list[tuple[str, dict]] = []

    def start(self) -> None:
        pass

    async def send(self, op: str, args: dict) -> Any:
        self.calls.append((op, args))
        if self.is_shutdown:
            raise ValidatorSwapped(f"{self.url} already shut down")
        behaviour = self._behaviour.get(op)
        if behaviour is None:
            return f"{self.url}:{op}"
        if isinstance(behaviour, Exception):
            raise behaviour
        if callable(behaviour):
            return behaviour()
        return behaviour

    async def shutdown(self) -> None:
        self.is_shutdown = True


def _make_pool(handle_specs: list[dict]) -> ValidatorPool:
    """Build a pool with one fake handle per URL."""
    urls = [spec["url"] for spec in handle_specs]
    failover = SubstrateUrlFailover(urls, initial_backoff_s=0.01, max_backoff_s=0.05)
    fakes_by_url = {
        spec["url"]: _FakeHandle(spec["url"], behaviour=spec.get("behaviour", {}))
        for spec in handle_specs
    }

    def handle_factory(url: str) -> _FakeHandle:
        return fakes_by_url[url]

    pool = ValidatorPool(
        urls=urls,
        failover=failover,
        handle_factory=handle_factory,
        max_swap_retries=3,
    )
    pool._fakes_by_url = fakes_by_url  # for assertions
    return pool


@pytest.mark.asyncio
async def test_pool_routes_to_active_handle():
    """An RPC call hits the currently-active handle."""
    pool = _make_pool([{"url": "http://a"}, {"url": "http://b"}])
    await pool.start()
    try:
        result = await pool.send("get_head", {})
        assert result == "http://a:get_head"
        assert pool._fakes_by_url["http://a"].calls == [("get_head", {})]
        assert pool._fakes_by_url["http://b"].calls == []
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_swaps_on_connection_error_and_retries_idempotent_op():
    """On ConnectionError, pool kills active, swaps, retries on next URL."""
    pool = _make_pool([
        {"url": "http://a", "behaviour": {"get_head": ConnectionError("dead")}},
        {"url": "http://b", "behaviour": {"get_head": b"\xab" * 32}},
    ])
    await pool.start()
    try:
        # get_head is idempotent → auto-retry across swap → succeeds on B
        result = await pool.send("get_head", {})
        assert result == b"\xab" * 32
        assert pool._fakes_by_url["http://a"].is_shutdown
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_does_NOT_auto_retry_submit_extrinsic():
    """submit_extrinsic raises ValidatorSwapped; pool does not retry on its own."""
    pool = _make_pool([
        {"url": "http://a", "behaviour": {"submit_extrinsic": ConnectionError("dead")}},
        {"url": "http://b"},
    ])
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await pool.send("submit_extrinsic", {"signed": b"\x00"})
        # B did NOT receive a submit_extrinsic call
        assert pool._fakes_by_url["http://b"].calls == []
        # A was swapped out, though
        assert pool._fakes_by_url["http://a"].is_shutdown
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_does_NOT_auto_retry_submit_signed_extrinsic():
    """submit_signed_extrinsic is non-idempotent; pool raises ValidatorSwapped."""
    pool = _make_pool([
        {
            "url": "http://a",
            "behaviour": {"submit_signed_extrinsic": ConnectionError("dead")},
        },
        {"url": "http://b"},
    ])
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await pool.send(
                "submit_signed_extrinsic",
                {"extrinsic_hex": "0xabcd", "wait_for": "inblock"},
            )
        # B did NOT receive a submit_signed_extrinsic call
        assert pool._fakes_by_url["http://b"].calls == []
        # A was swapped out, though
        assert pool._fakes_by_url["http://a"].is_shutdown
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_routes_submit_signed_extrinsic_to_active_handle():
    """Happy path: submit_signed_extrinsic reaches the active child with its hex."""
    pool = _make_pool([
        {
            "url": "http://a",
            "behaviour": {"submit_signed_extrinsic": "receipt-sentinel"},
        },
        {"url": "http://b"},
    ])
    await pool.start()
    try:
        result = await pool.send(
            "submit_signed_extrinsic",
            {"extrinsic_hex": "0xdeadbeef", "wait_for": "finalized"},
        )
        assert result == "receipt-sentinel"
        assert pool._fakes_by_url["http://a"].calls == [
            (
                "submit_signed_extrinsic",
                {"extrinsic_hex": "0xdeadbeef", "wait_for": "finalized"},
            ),
        ]
        assert pool._fakes_by_url["http://b"].calls == []
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_exhausts_retries_then_raises():
    """Idempotent op that fails on every URL eventually raises after max_swap_retries."""
    pool = _make_pool([
        {"url": "http://a", "behaviour": {"get_head": ConnectionError("dead")}},
        {"url": "http://b", "behaviour": {"get_head": ConnectionError("dead")}},
    ])
    await pool.start()
    try:
        with pytest.raises(ConnectionError):
            await pool.send("get_head", {})
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_force_swap_kills_active_and_spawns_next():
    """force_swap() rotates to the next URL regardless of in-flight state."""
    pool = _make_pool([{"url": "http://a"}, {"url": "http://b"}])
    await pool.start()
    try:
        # initially active = A
        assert pool.active_url() == "http://a"
        await pool.force_swap()
        assert pool.active_url() == "http://b"
        # an RPC now hits B
        await pool.send("get_head", {})
        assert pool._fakes_by_url["http://b"].calls == [("get_head", {})]
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_passes_non_connection_errors_through_immediately():
    """A non-connection error (e.g. RuntimeError from the chain) is NOT a reason to swap."""
    pool = _make_pool([
        {"url": "http://a", "behaviour": {"get_head": RuntimeError("chain returned None")}},
        {"url": "http://b"},
    ])
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="chain returned None"):
            await pool.send("get_head", {})
        # A is still active; no swap happened
        assert pool.active_url() == "http://a"
        assert not pool._fakes_by_url["http://a"].is_shutdown
    finally:
        await pool.shutdown()


# ---------------------------------------------------------------------------
# Constructor surface.
# ---------------------------------------------------------------------------


def test_pool_constructible_with_urls_only_defaults():
    """Callers `ValidatorPool(urls=[...])` work — failover + factory default."""
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944"])
    assert pool.urls == ("ws://a:9944", "ws://b:9944")
    assert pool.current_url == "ws://a:9944"


def test_pool_constructor_rejects_empty_urls():
    """Empty URL list is a usage error — fail fast."""
    with pytest.raises(ValueError, match="at least one validator URL"):
        ValidatorPool(urls=[])
