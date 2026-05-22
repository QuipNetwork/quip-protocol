"""Unit tests for `SubstrateClient.subscribe_new_heads`'s pump model.

The pump moves coalescing to the notification source: the websocket reader
thread only signals an asyncio Event; a single async task wakes, fetches
``get_chain_head`` once, and invokes the callback. A burst of N headers
collapses into one or two callback invocations against *current* chain
state, never against historical block numbers.

These tests construct a `SubstrateClient` via `__new__`, plug in a
hand-rolled fake `_iface`, and patch `_run` so blocking RPC stand-ins run
inline on the event loop.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from shared.substrate_client import SubstrateClient


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


class _FakeIface:
    """Minimal substrate-interface stand-in for the pump.

    `subscribe_block_headers` returns when `stop_event` is set so the
    test can drive the dispatch callback and then unblock the await.
    """

    def __init__(
        self,
        head_hashes: list[str],
        stop_event: threading.Event,
    ):
        self._head_hashes = head_hashes
        self._head_calls = 0
        self._stop_event = stop_event
        self.captured_callback = None  # for tests that need to drive it

    def subscribe_block_headers(self, callback, **kwargs):  # noqa: ARG002
        self.captured_callback = callback
        # Block (on the executor thread, where the real subscribe runs)
        # until the test asks us to return.
        self._stop_event.wait()

    def get_chain_head(self) -> str:
        # Each call returns the next head in the list, or the last one
        # forever once exhausted. Pump tests want to see "current head"
        # vary across pump iterations.
        idx = min(self._head_calls, len(self._head_hashes) - 1)
        self._head_calls += 1
        return self._head_hashes[idx]

    def get_chain_finalised_head(self) -> str:
        return self.get_chain_head()

    def get_block_header(self, **kwargs):  # noqa: ARG002
        return {"header": {"number": self._head_calls * 10}}


def _build_client(iface: _FakeIface) -> SubstrateClient:
    client = SubstrateClient.__new__(SubstrateClient)
    client.url = "ws://test.invalid:0"
    client._iface = iface
    client._lock = None  # type: ignore[assignment]

    async def _direct_run(fn):
        # Call inline on the loop. The real `_run` would push to the
        # executor and acquire `_call_lock`; tests don't need either.
        return fn()

    client._run = _direct_run  # type: ignore[assignment]
    return client


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


async def test_pump_coalesces_burst_of_headers():
    """50 dispatch fires in quick succession must collapse into a small
    number of callback invocations — never one-per-header. The hashes
    delivered to the callback come from `get_chain_head`, NOT from the
    header payload's number field, so historical reordering is
    impossible by construction."""
    head_hashes = [f"0x{i:064x}" for i in range(1, 6)]
    stop = threading.Event()
    iface = _FakeIface(head_hashes, stop)
    client = _build_client(iface)

    received: list[tuple[bytes, int]] = []

    async def cb(block_hash: bytes, number: int) -> None:
        received.append((block_hash, number))

    async def driver():
        # Wait for the subscription to install its dispatch callback.
        for _ in range(50):
            if iface.captured_callback is not None:
                break
            await asyncio.sleep(0)
        assert iface.captured_callback is not None, "dispatch never registered"

        # Fire 50 header notifications back-to-back from a "reader thread"
        # context (the dispatch must be loop-safe via
        # call_soon_threadsafe).
        for _ in range(50):
            iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")

        # Yield repeatedly so the pump task gets time to run.
        for _ in range(20):
            await asyncio.sleep(0.01)

        # Stop the subscription.
        stop.set()

    await asyncio.gather(
        client.subscribe_new_heads(cb), driver()
    )

    # Prime-the-pump set means we expect at least 1 invocation. The
    # absolute upper bound depends on event-loop scheduling, but the
    # critical contract is "MUCH less than 50": the per-header model
    # would deliver exactly 50.
    assert 1 <= len(received) <= 5, f"pump should coalesce; got {len(received)}"
    # Every callback hash must come from `get_chain_head`'s sequence,
    # never from the header payload (which had number=0).
    for h, _n in received:
        assert "0x" + h.hex() in head_hashes


async def test_pump_survives_iface_none_during_reconnect():
    """If `_iface` goes None between signal arrival and pump wake (the
    close()/reconnect() window), the pump skips that iteration and stays
    alive rather than crashing the subscription task."""
    head_hashes = ["0x" + "ab" * 32]
    stop = threading.Event()
    iface = _FakeIface(head_hashes, stop)
    client = _build_client(iface)

    received: list[tuple[bytes, int]] = []

    async def cb(block_hash: bytes, number: int) -> None:
        received.append((block_hash, number))

    async def driver():
        for _ in range(50):
            if iface.captured_callback is not None:
                break
            await asyncio.sleep(0)
        # Drop the iface right as a notification arrives. The pump
        # should hit its None-guard and continue without raising.
        client._iface = None
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        await asyncio.sleep(0.05)
        # Restore the iface and fire another notification — pump must
        # still be alive to handle it.
        client._iface = iface
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        for _ in range(20):
            await asyncio.sleep(0.01)
        stop.set()

    await asyncio.gather(
        client.subscribe_new_heads(cb), driver()
    )

    # We must have received at least one callback (from the
    # post-restore notification or the priming wake). Crash-survival is
    # the actual contract being tested.
    assert len(received) >= 1


async def test_pump_finalized_only_uses_finalised_endpoint():
    """`finalized_only=True` routes to `get_chain_finalised_head` rather
    than `get_chain_head`. Verified by giving the two endpoints different
    return values."""
    head_hashes = ["0x" + "ff" * 32]
    stop = threading.Event()
    iface = _FakeIface(head_hashes, stop)
    finalised_head = "0x" + "11" * 32
    iface.get_chain_finalised_head = lambda: finalised_head  # type: ignore[assignment]
    client = _build_client(iface)

    received: list[tuple[bytes, int]] = []

    async def cb(block_hash: bytes, number: int) -> None:
        received.append((block_hash, number))

    async def driver():
        for _ in range(50):
            if iface.captured_callback is not None:
                break
            await asyncio.sleep(0)
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        for _ in range(20):
            await asyncio.sleep(0.01)
        stop.set()

    await asyncio.gather(
        client.subscribe_new_heads(cb, finalized_only=True), driver()
    )

    assert len(received) >= 1
    assert all(h == bytes.fromhex("11" * 32) for h, _ in received)


async def test_pump_propagates_no_validator_reachable():
    """When the pump's `_run` raises `NoValidatorReachable` (failover
    exhausted), `subscribe_new_heads` must propagate the exception out
    instead of swallowing it in the cleanup `finally`. The outer
    `_subscribe_heads` relies on seeing the exhausted-failover signal
    to shut the controller down — without this, the controller would
    wait on a dead subscription forever."""
    from shared.substrate_client import NoValidatorReachable, ValidatorAttempt

    head_hashes = ["0x" + ("aa" * 32)]
    stop = threading.Event()
    iface = _FakeIface(head_hashes, stop)
    client = _build_client(iface)

    # Override `_run` to raise NoValidatorReachable on the first call,
    # simulating exhausted failover.
    async def _failing_run(fn):  # noqa: ARG001
        raise NoValidatorReachable([
            ValidatorAttempt(url="ws://primary", exc_type="OSError", message="down"),
        ])

    client._run = _failing_run  # type: ignore[assignment]

    async def cb(block_hash: bytes, number: int) -> None:
        pass  # never reached — pump dies first

    async def driver():
        for _ in range(50):
            if iface.captured_callback is not None:
                break
            await asyncio.sleep(0)
        # Wake the pump so it tries to call `_run` (which raises).
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        await asyncio.sleep(0.05)
        # Unblock the outer subscribe so the finally runs.
        stop.set()

    with pytest.raises(NoValidatorReachable):
        await asyncio.gather(
            client.subscribe_new_heads(cb), driver()
        )


async def test_pump_survives_callback_exception():
    """A `callback` that raises (and any other non-connection exception
    inside the pump iteration) must not kill the pump task. The next
    chain head should still produce a callback invocation. Regression
    guard for the silent-controller-hang failure mode where a single
    runtime API shape drift takes the miner offline with no log."""
    head_hashes = ["0x" + ("aa" * 32), "0x" + ("bb" * 32)]
    stop = threading.Event()
    iface = _FakeIface(head_hashes, stop)
    client = _build_client(iface)

    received: list[tuple[bytes, int]] = []
    call_count = [0]

    async def cb(block_hash: bytes, number: int) -> None:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated runtime-API shape drift")
        received.append((block_hash, number))

    async def driver():
        for _ in range(50):
            if iface.captured_callback is not None:
                break
            await asyncio.sleep(0)
        # First wake (priming) — callback raises, pump must survive.
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        await asyncio.sleep(0.05)
        # Second wake — callback succeeds, proving pump is still alive.
        iface.captured_callback({"header": {"number": 0}}, 0, "sub-id")
        for _ in range(20):
            await asyncio.sleep(0.01)
        stop.set()

    await asyncio.gather(
        client.subscribe_new_heads(cb), driver()
    )

    # Callback was invoked at least twice (once raising, once
    # succeeding). The second invocation proves the pump survived.
    assert call_count[0] >= 2, f"pump died after exception; got {call_count[0]}"
    assert len(received) >= 1, "no successful callback after exception"
