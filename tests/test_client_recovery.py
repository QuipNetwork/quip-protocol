"""Recovery behaviour for `substrate.client.SubstrateClient` (QUI-899).

The miner could reach a state where every `submit_proof` timed out against a
measurably healthy validator and never recovered short of a process restart.
Two transport-level gaps produced it:

1. `SubstrateInterface` was constructed with no socket-level read timeout, so
   a black-holed TCP connection (no RST/FIN — routine behind NAT and idle
   load-balancer drops) blocks forever inside `recv()` and never raises.

2. When an outer `asyncio.wait_for` gave up on such a call, the coroutine took
   an `asyncio.CancelledError` at its `await loop.run_in_executor(...)` point.
   `CancelledError` derives from `BaseException`, so it matched none of the
   connection-error tuples in `_run`/`ensure_connected` and no reconnect ran.
   Worse, cancelling an already-running executor job is a documented no-op, so
   the worker thread kept running against `self._iface` — the exact concurrent
   access the `_call_lock` exists to prevent — while the lock was released by
   normal context-manager unwinding.

These tests pin both behaviours.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from substrate.client import SubstrateClient


@pytest.mark.asyncio
async def test_connect_sets_a_socket_read_timeout():
    """A connect must bound `recv()` so a black-holed socket eventually raises.

    Without this the transport has no guaranteed failure signal at all, and
    every recovery layer above it is waiting on an exception that never comes.
    """
    with patch("substrate.client.SubstrateInterface") as fake_cls:
        fake_cls.return_value = MagicMock()
        client = SubstrateClient("ws://test:9944")
        await client.connect()

    assert fake_cls.call_count == 1
    ws_options = fake_cls.call_args.kwargs.get("ws_options")
    assert ws_options is not None, (
        "SubstrateInterface built without ws_options; a socket with no read "
        "timeout can block in recv() forever"
    )
    timeout = ws_options.get("timeout")
    assert timeout is not None, "ws_options carries no socket timeout"
    # Must clear normal inter-block latency: substrate's submit-and-watch
    # idles on the same socket between blocks, so too small a value would
    # abort legitimate submits waiting for inclusion.
    assert timeout > 12.0, (
        f"socket timeout {timeout}s is under normal inter-block latency; "
        "this would break submit-and-watch on a healthy chain"
    )


@pytest.mark.asyncio
async def test_cancelled_call_does_not_reuse_the_same_iface():
    """After a cancellation, the next call must not reuse the old iface.

    The cancelled executor thread may still be running against it, and
    `SubstrateInterface` corrupts under concurrent access (see the
    `_call_lock` comment in client.py). Reusing it is the durable-bad-state
    bug: the object is never replaced, so every later submit rides a
    connection that may already be occupied by a zombie thread.
    """
    release = threading.Event()
    first_iface = MagicMock(name="first_iface")

    def blocking_call():
        # Stands in for a black-holed `recv()`: returns only when released.
        release.wait(timeout=5.0)
        return "too-late"

    client = SubstrateClient("ws://test:9944")
    with patch("substrate.client.SubstrateInterface") as fake_cls:
        fake_cls.return_value = first_iface
        await client.connect()

        assert client._iface is first_iface

        # An outer wait_for giving up looks exactly like this from inside.
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(client._raw_run(blocking_call), timeout=0.05)

        # A fresh iface must be built rather than the occupied one reused.
        second_iface = MagicMock(name="second_iface")
        fake_cls.return_value = second_iface
        await client.ensure_connected()

    release.set()
    assert client._iface is not first_iface, (
        "client reused the iface a cancelled executor thread may still hold; "
        "this is the QUI-899 durable-bad-state path"
    )
