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


# ---------------------------------------------------------------------------
# Dedicated write handle (QUI-829 / gh-18): submits ride a WRITE handle that
# read-path (snapshot-poll) swaps can never tear down. Unlike _make_pool,
# these tests use a factory that returns a FRESH handle per call, mirroring
# the real ValidatorHandle (a new child process each spawn) so the read and
# write handles are distinct objects even on a single URL.
# ---------------------------------------------------------------------------


class _BlockableHandle:
    """Fresh-per-spawn fake whose ``send`` can block on a per-op event."""

    def __init__(self, url: str, *, gates: dict[str, asyncio.Event] | None = None,
                 fail_first_ensure: bool = False) -> None:
        self.url = url
        self.is_shutdown = False
        self.calls: list[tuple[str, dict]] = []
        self._gates = gates or {}
        self._fail_first_ensure = fail_first_ensure
        self._ensure_calls = 0

    def start(self) -> None:
        pass

    async def send(self, op: str, args: dict) -> Any:
        self.calls.append((op, args))
        if self.is_shutdown:
            raise ValidatorSwapped(f"{self.url} already shut down")
        if op == "ensure_connected":
            self._ensure_calls += 1
            if self._fail_first_ensure and self._ensure_calls == 1:
                raise ConnectionError("stale idle write socket")
            return True
        gate = self._gates.get(op)
        if gate is not None:
            await gate.wait()
        return f"{self.url}:{op}"

    async def shutdown(self) -> None:
        self.is_shutdown = True


def _make_pool_fresh(url: str = "http://a", **handle_kwargs) -> ValidatorPool:
    """Pool whose factory spawns a distinct handle per call (like production)."""
    spawned: list[_BlockableHandle] = []

    def factory(u: str) -> _BlockableHandle:
        h = _BlockableHandle(u, **handle_kwargs)
        spawned.append(h)
        return h

    pool = ValidatorPool(
        urls=[url],
        failover=SubstrateUrlFailover([url], initial_backoff_s=0.01, max_backoff_s=0.05),
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    pool._spawned = spawned  # for assertions
    return pool


@pytest.mark.asyncio
async def test_send_write_uses_dedicated_handle_not_read_handle():
    """A submit rides a separate handle from the read/snapshot path."""
    pool = _make_pool_fresh()
    await pool.start()
    try:
        await pool.send("get_head", {})  # read handle
        receipt = await pool.send_write("submit_signed_extrinsic", {"extrinsic_hex": "0x01"})
        assert receipt == "http://a:submit_signed_extrinsic"
        read_handle, write_handle = pool._active, pool._write_active
        assert read_handle is not write_handle
        # Read handle only saw the read; write handle saw ensure + submit.
        assert read_handle.calls == [("get_head", {})]
        assert write_handle.calls == [
            ("ensure_connected", {}),
            ("submit_signed_extrinsic", {"extrinsic_hex": "0x01"}),
        ]
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_read_swap_does_not_teardown_inflight_write():
    """The regression (QUI-829): a snapshot-poll swap while a submit is in
    flight must NOT cancel the submit. With the dedicated write handle the
    read swap only touches ``_active``, so the in-flight write completes."""
    gate = asyncio.Event()
    pool = _make_pool_fresh(gates={"submit_signed_extrinsic": gate})
    await pool.start()
    try:
        # Launch a submit that blocks inside the write handle's send().
        write_task = asyncio.create_task(
            pool.send_write("submit_signed_extrinsic", {"extrinsic_hex": "0x01"})
        )
        # Wait until the write handle is actually mid-submit.
        while pool._write_active is None or (
            "submit_signed_extrinsic",
            {"extrinsic_hex": "0x01"},
        ) not in pool._write_active.calls:
            await asyncio.sleep(0)
        write_handle = pool._write_active
        read_handle = pool._active

        # A concurrent read-path connection failure swaps the READ handle.
        await pool._swap_after_failure(read_handle)
        assert read_handle.is_shutdown
        assert pool._active is not read_handle
        # The write handle must be untouched.
        assert pool._write_active is write_handle
        assert not write_handle.is_shutdown

        # Release the submit; it completes successfully on the same handle.
        gate.set()
        assert await write_task == "http://a:submit_signed_extrinsic"
        assert not write_handle.is_shutdown
        assert pool.last_successful_submission is not None
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_send_write_reconnects_stale_idle_socket():
    """Submits are rare, so the write socket is usually stale. The write path
    health-reconnects on demand: a failed first ensure respawns the handle,
    and the submit lands on the fresh one."""
    # Handle 0 = read (start), handle 1 = stale first write handle, handle 2+
    # = healthy reconnect. Only handle 1's socket is stale.
    spawned: list[_BlockableHandle] = []

    def factory(u: str) -> _BlockableHandle:
        h = _BlockableHandle(u, fail_first_ensure=(len(spawned) == 1))
        spawned.append(h)
        return h

    pool = ValidatorPool(
        urls=["http://a"],
        failover=SubstrateUrlFailover(["http://a"], initial_backoff_s=0.01, max_backoff_s=0.05),
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    await pool.start()
    try:
        receipt = await pool.send_write("submit_signed_extrinsic", {"extrinsic_hex": "0x01"})
        assert receipt == "http://a:submit_signed_extrinsic"
        # First write handle (spawned[1]) failed ensure and was shut down; a
        # fresh one (spawned[2]) carried the submit.
        assert spawned[1].is_shutdown
        assert pool._write_active is spawned[2]
        assert not pool._write_active.is_shutdown
        assert ("submit_signed_extrinsic", {"extrinsic_hex": "0x01"}) in pool._write_active.calls
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_send_write_raises_validator_swapped_on_connection_error():
    """A connection-class failure during the submit itself surfaces as
    ValidatorSwapped (caller re-signs) and reconnects the write handle."""

    class _FailSubmitHandle(_BlockableHandle):
        async def send(self, op: str, args: dict) -> Any:
            if op == "submit_signed_extrinsic" and not self.is_shutdown:
                self.calls.append((op, args))
                raise ConnectionError("broken pipe mid-submit")
            return await super().send(op, args)

    spawned: list[_FailSubmitHandle] = []

    def factory(u: str) -> _FailSubmitHandle:
        h = _FailSubmitHandle(u)
        spawned.append(h)
        return h

    pool = ValidatorPool(
        urls=["http://a"],
        failover=SubstrateUrlFailover(["http://a"], initial_backoff_s=0.01, max_backoff_s=0.05),
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await pool.send_write("submit_signed_extrinsic", {"extrinsic_hex": "0x01"})
        # The failed write handle was reconnected (a fresh one is active).
        assert pool._write_active is not None
        assert not pool._write_active.is_shutdown
        assert pool.last_successful_submission is None
    finally:
        await pool.shutdown()


# ----------------------------------------------------------------------
# Cancellation on the write path (QUI-899)
#
# When an outer `wait_for` gives up on a submit, the child process is left
# mid-call. `CancelledError` derives from BaseException, so it matches
# neither `_CONNECTION_ERRORS` nor `ValidatorSwapped`, and the write
# handle's reconnect never runs — the next submit reuses a child that is
# still busy with the abandoned call.
#
# Note it must NOT simply be added to `_CONNECTION_ERRORS`: that tuple is
# caught and converted into `ValidatorSwapped`, which would swallow the
# cancellation and break shutdown.
# ----------------------------------------------------------------------


def _pool_with_fresh_handles(url: str, behaviour_by_generation: list[dict]):
    """Pool whose factory hands back a NEW fake handle on every call.

    The shared `_make_pool` returns one fake per URL, so a respawn is
    indistinguishable from a reuse — which is the exact thing under test.
    """
    made: list[_FakeHandle] = []

    def handle_factory(u: str) -> _FakeHandle:
        idx = min(len(made), len(behaviour_by_generation) - 1)
        h = _FakeHandle(u, behaviour=behaviour_by_generation[idx])
        made.append(h)
        return h

    pool = ValidatorPool(
        urls=[url],
        failover=SubstrateUrlFailover([url], initial_backoff_s=0.01,
                                      max_backoff_s=0.05),
        handle_factory=handle_factory,
        max_swap_retries=3,
    )
    pool._made_handles = made
    return pool


@pytest.mark.asyncio
async def test_cancelled_write_reraises_rather_than_swallowing():
    """A cancellation must propagate as CancelledError.

    Converting it to ValidatorSwapped would make a cancelled task look like
    a routine swap to the caller, and shutdown would stop working.
    """
    def _cancel():
        raise asyncio.CancelledError()

    pool = _pool_with_fresh_handles(
        "ws://a:9944", [{"submit_signed_extrinsic": _cancel}]
    )

    with pytest.raises(asyncio.CancelledError):
        await pool.send_write("submit_signed_extrinsic", {"x": 1})


@pytest.mark.asyncio
async def test_cancelled_write_does_not_reuse_the_busy_child():
    """The next submit after a cancellation must not land on the same child.

    That child may still be running the abandoned call; reusing it is the
    QUI-899 durable-bad-state path on the write side.
    """
    def _cancel():
        raise asyncio.CancelledError()

    pool = _pool_with_fresh_handles(
        "ws://a:9944",
        [{"submit_signed_extrinsic": _cancel}, {}],  # gen 0 cancels, gen 1 fine
    )

    with pytest.raises(asyncio.CancelledError):
        await pool.send_write("submit_signed_extrinsic", {"x": 1})

    first = pool._write_active
    await pool.send_write("submit_signed_extrinsic", {"x": 2})

    assert pool._write_active is not first, (
        "write path reused the child that was cancelled mid-call"
    )


@pytest.mark.asyncio
async def test_websocket_exception_is_treated_as_connection_class():
    """`WebSocketException` must trigger the swap like any dead socket.

    `_CONNECTION_ERRORS` carried an unfinished note saying substrate-
    interface's own websocket errors "will be added here when wiring
    against the real client." Until they were, a genuine websocket failure
    surfacing from the child propagated raw instead of swapping, so the
    pool's recovery never ran for that failure class.
    """
    from websocket import WebSocketException

    pool = _make_pool([
        {"url": "http://a", "behaviour": {"get_head": WebSocketException("dead")}},
        {"url": "http://b", "behaviour": {"get_head": b"\xab" * 32}},
    ])
    await pool.start()
    try:
        assert await pool.send("get_head", {}) == b"\xab" * 32
        assert pool._fakes_by_url["http://a"].is_shutdown
    finally:
        await pool.shutdown()
