"""Tests for the rewritten substrate.pool.ValidatorPool.

The pool exposes an async RPC surface, routes calls to the currently-
active ValidatorHandle, and handles hot-active swap on connection
failure with idempotent-only auto-retry.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# ValidatorSwapped from a concurrently-shutdown handle: retry (idempotent) /
# propagate (non-idempotent), bounded by max_swap_retries.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_op_retries_when_captured_handle_reports_swapped():
    """A handle shut down under us raises ValidatorSwapped; the pool must
    re-read self._active and retry idempotent ops rather than propagate."""
    state = {"n": 0}

    def once_swapped():
        state["n"] += 1
        if state["n"] == 1:
            raise ValidatorSwapped("shut down by a concurrent swap")
        return b"head"

    pool = _make_pool([{"url": "http://a", "behaviour": {"get_head": once_swapped}}])
    await pool.start()
    try:
        assert await pool.send("get_head", {}) == b"head"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_non_idempotent_op_propagates_validator_swapped():
    """submit_signed_extrinsic must surface ValidatorSwapped (caller re-signs);
    the pool must not silently retry a non-idempotent op."""
    pool = _make_pool([
        {
            "url": "http://a",
            "behaviour": {"submit_signed_extrinsic": ValidatorSwapped("swapped")},
        },
    ])
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await pool.send(
                "submit_signed_extrinsic",
                {"extrinsic_hex": "0xabcd", "wait_for": "inblock"},
            )
        # Raised on the first call; no hidden retry.
        assert len(pool._fakes_by_url["http://a"].calls) == 1
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_validator_swapped_retry_is_bounded_by_max_swap_retries():
    """A handle that always reports swapped must not loop forever — the pool
    re-raises after exactly max_swap_retries attempts."""
    pool = _make_pool([
        {"url": "http://a", "behaviour": {"get_head": ValidatorSwapped("swapped")}},
    ])
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await pool.send("get_head", {})
        assert len(pool._fakes_by_url["http://a"].calls) == pool._max_swap_retries
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_single_url_racing_swap_does_not_teardown_fresh_handle():
    """Two callers race a swap on a single URL. The loser holds a stale handle
    whose URL equals the fresh handle's URL; the identity guard must make the
    loser's swap a no-op instead of tearing down the freshly-spawned handle."""
    spawned: list[_FakeHandle] = []

    def factory(url: str) -> _FakeHandle:
        handle = _FakeHandle(url)
        spawned.append(handle)
        return handle

    failover = SubstrateUrlFailover(
        ["http://a"], initial_backoff_s=0.01, max_backoff_s=0.05
    )
    pool = ValidatorPool(
        urls=["http://a"],
        failover=failover,
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    await pool.start()
    try:
        h0 = pool._active
        # Winner swaps h0 → h1 (fresh handle on the same URL).
        await pool._swap_after_failure(h0)
        h1 = pool._active
        assert h1 is not h0
        assert h0.is_shutdown
        assert not h1.is_shutdown
        assert len(spawned) == 2  # start + one swap

        # Loser still holds h0; its swap must be a no-op — h1 survives.
        assert await pool._swap_after_failure(h0) is False
        assert pool._active is h1
        assert not h1.is_shutdown
        assert len(spawned) == 2  # no extra spawn
    finally:
        await pool.shutdown()
