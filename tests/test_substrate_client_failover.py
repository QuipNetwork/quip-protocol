"""Unit tests for `SubstrateClient`'s validator-list / failover surface.

Distinct from `test_substrate_client.py` which is gated on a live chain.
These tests monkeypatch `SubstrateInterface` so they run anywhere — they
exercise the URL-rotation policy, not the underlying RPC behavior.
"""
from __future__ import annotations

import pytest

from shared import substrate_client as sc_module
from shared.substrate_client import NoValidatorReachable, SubstrateClient


class _StubInterface:
    """Minimal stand-in for `substrateinterface.SubstrateInterface`.

    Records which URL it was constructed with so tests can assert the
    rotation order. Raises on construction when the URL is in `bad_urls`.
    """

    bad_urls: set[str] = set()
    construction_log: list[str] = []

    def __init__(self, url: str) -> None:
        type(self).construction_log.append(url)
        if url in type(self).bad_urls:
            raise ConnectionRefusedError(f"stub refused: {url}")
        self.url = url

    def close(self) -> None:  # pragma: no cover — close path not exercised here
        pass


@pytest.fixture(autouse=True)
def _patch_substrate_interface(monkeypatch):
    """Swap `SubstrateInterface` for the stub for every test in this module."""
    _StubInterface.bad_urls = set()
    _StubInterface.construction_log = []
    monkeypatch.setattr(sc_module, "SubstrateInterface", _StubInterface)
    yield


@pytest.mark.asyncio
async def test_single_url_back_compat():
    """`SubstrateClient(url=...)` keeps working with a single string."""
    client = SubstrateClient(url="ws://only:9944")
    await client.connect()
    assert client.current_url == "ws://only:9944"
    assert _StubInterface.construction_log == ["ws://only:9944"]


@pytest.mark.asyncio
async def test_urls_list_first_succeeds():
    """List form: first URL accepted, second never tried."""
    client = SubstrateClient(urls=["ws://primary:9944", "ws://standby:9944"])
    await client.connect()
    assert client.current_url == "ws://primary:9944"
    assert _StubInterface.construction_log == ["ws://primary:9944"]


@pytest.mark.asyncio
async def test_urls_list_falls_through_to_standby():
    """Primary refuses, secondary accepts. Client lands on secondary."""
    _StubInterface.bad_urls = {"ws://primary:9944"}
    client = SubstrateClient(urls=["ws://primary:9944", "ws://standby:9944"])
    await client.connect()
    assert client.current_url == "ws://standby:9944"
    assert _StubInterface.construction_log == [
        "ws://primary:9944",
        "ws://standby:9944",
    ]


@pytest.mark.asyncio
async def test_all_urls_fail_raises_no_validator_reachable():
    """When every validator refuses, raise NoValidatorReachable with attempts."""
    _StubInterface.bad_urls = {"ws://a:9944", "ws://b:9944"}
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    with pytest.raises(NoValidatorReachable) as exc_info:
        await client.connect()
    attempts = exc_info.value.attempts
    assert [a.url for a in attempts] == ["ws://a:9944", "ws://b:9944"]
    # Each attempt should carry both the exception type and the message
    assert all(a.exc_type == "ConnectionRefusedError" for a in attempts)
    assert all("stub refused" in a.message for a in attempts)


@pytest.mark.asyncio
async def test_empty_urls_rejected_at_construction():
    """Construction with no URLs is a programming error — not deferred."""
    with pytest.raises(ValueError, match="at least one validator URL"):
        SubstrateClient(urls=[])


@pytest.mark.asyncio
async def test_url_and_urls_both_set_rejected():
    """Don't allow ambiguous construction — caller must pick one form."""
    with pytest.raises(ValueError, match="exactly one of"):
        SubstrateClient(url="ws://a:9944", urls=["ws://b:9944"])


@pytest.mark.asyncio
async def test_reconnect_rotates_to_next_url():
    """After connecting to A, reconnect() should land on B."""
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    assert client.current_url == "ws://a:9944"
    await client.reconnect()
    assert client.current_url == "ws://b:9944"
    assert _StubInterface.construction_log == [
        "ws://a:9944",
        "ws://b:9944",
    ]


@pytest.mark.asyncio
async def test_reconnect_rotates_circularly():
    """[A, B, C] currently on B → reconnect should try C, then A."""
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    # Force first-connect to land on B by marking A bad.
    _StubInterface.bad_urls = {"ws://a:9944"}
    await client.connect()
    assert client.current_url == "ws://b:9944"
    # Now mark B + C bad, A becomes good — reconnect should walk C, then A.
    _StubInterface.bad_urls = {"ws://b:9944", "ws://c:9944"}
    _StubInterface.construction_log.clear()
    await client.reconnect()
    assert client.current_url == "ws://a:9944"
    assert _StubInterface.construction_log == ["ws://c:9944", "ws://a:9944"]


@pytest.mark.asyncio
async def test_reconnect_raises_when_all_dead():
    """If every URL refuses during reconnect, raise NoValidatorReachable."""
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    _StubInterface.bad_urls = {"ws://a:9944", "ws://b:9944"}
    with pytest.raises(NoValidatorReachable):
        await client.reconnect()


@pytest.mark.asyncio
async def test_run_catches_websocket_exception_and_failovers(monkeypatch):
    """A WebSocketException from a _run call triggers reconnect; the original
    exception still surfaces to the caller so caller-level retry kicks in."""
    import websocket as ws_module

    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    assert client.current_url == "ws://a:9944"

    def dying_call():
        raise ws_module.WebSocketConnectionClosedException("socket closed")

    with pytest.raises(ws_module.WebSocketConnectionClosedException):
        await client._run(dying_call)

    # _run should have triggered failover so the next call lands on B.
    assert client.current_url == "ws://b:9944"


@pytest.mark.asyncio
async def test_run_does_not_failover_on_non_connection_exception():
    """A logical (non-connection) error must NOT trigger failover."""
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()

    def app_error():
        raise ValueError("decoding failed")

    with pytest.raises(ValueError):
        await client._run(app_error)

    assert client.current_url == "ws://a:9944"


@pytest.mark.asyncio
async def test_run_does_not_recurse_when_reconnect_also_dies():
    """If _run's failover attempt itself can't find a healthy validator,
    surface NoValidatorReachable (not the original websocket exception)."""
    import websocket as ws_module

    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    _StubInterface.bad_urls = {"ws://a:9944", "ws://b:9944"}

    def dying_call():
        raise ws_module.WebSocketConnectionClosedException("socket closed")

    with pytest.raises(NoValidatorReachable):
        await client._run(dying_call)


@pytest.mark.asyncio
async def test_head_methods_call_new_iface_after_failover(monkeypatch):
    """`get_head()` and `get_finalized_head()` must hit the *current*
    `self._iface` after a failover swap, not a stale captured method.

    Each iface instance records its own call count. After reconnect()
    the second iface should serve the head/finalized queries.
    """

    class _IfaceWithHead:
        _next_id = 0

        def __init__(self, url: str) -> None:
            self.url = url
            type(self)._next_id += 1
            self.id = type(self)._next_id
            self.head_calls = 0
            self.finalized_calls = 0

        def get_chain_head(self) -> str:
            self.head_calls += 1
            return "0x" + f"{self.id:064x}"

        def get_chain_finalised_head(self) -> str:
            self.finalized_calls += 1
            return "0x" + f"{self.id:064x}"

        def close(self) -> None:
            pass

    monkeypatch.setattr(sc_module, "SubstrateInterface", _IfaceWithHead)

    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    first_iface = client._iface
    await client.get_head()
    assert first_iface.head_calls == 1

    await client.reconnect()
    second_iface = client._iface
    assert second_iface is not first_iface

    await client.get_head()
    await client.get_finalized_head()
    # New iface received both calls; old iface untouched after reconnect.
    assert second_iface.head_calls == 1
    assert second_iface.finalized_calls == 1
    assert first_iface.head_calls == 1  # unchanged
    assert first_iface.finalized_calls == 0


# ----------------------------------------------------------------------
# Pool integration — SubstrateClient delegates rotation to a ValidatorPool
# ----------------------------------------------------------------------


class _SpyPool:
    """Stub ValidatorPool that records advance_rotation calls and returns
    a scripted sequence of next URLs."""

    def __init__(self, urls, advance_returns):
        self.urls = tuple(urls)
        self._advance_returns = list(advance_returns)
        self.advance_calls: list[str] = []

    async def advance_rotation(self, from_url: str) -> str:
        self.advance_calls.append(from_url)
        return self._advance_returns.pop(0)


@pytest.mark.asyncio
async def test_pool_kwarg_defers_failover_to_pool():
    """With `pool=` set, `_run`'s failover handler calls
    `pool.advance_rotation(from_url=current)` to pick the next URL —
    not the client's own circular rotation."""
    import websocket as ws_module

    pool = _SpyPool(
        urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"],
        # Pool says "skip directly to C" even though B comes next in
        # local rotation — proves the pool is in control.
        advance_returns=["ws://c:9944"],
    )
    client = SubstrateClient(urls=pool.urls, pool=pool)
    await client.connect()
    assert client.current_url == "ws://a:9944"

    def dying_call():
        raise ws_module.WebSocketConnectionClosedException("dropped")

    with pytest.raises(ws_module.WebSocketConnectionClosedException):
        await client._run(dying_call)

    assert pool.advance_calls == ["ws://a:9944"]
    assert client.current_url == "ws://c:9944"  # pool's pick, not client's


@pytest.mark.asyncio
async def test_no_pool_uses_local_rotation_unchanged():
    """Without pool=, the client's own circular rotation runs (existing
    behavior preserved). Regression for the back-compat shim."""
    import websocket as ws_module

    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944"])
    await client.connect()
    assert client.current_url == "ws://a:9944"

    def dying_call():
        raise ws_module.WebSocketConnectionClosedException("dropped")

    with pytest.raises(ws_module.WebSocketConnectionClosedException):
        await client._run(dying_call)

    # Local rotation, not pool-driven: lands on B (the next URL in list).
    assert client.current_url == "ws://b:9944"


@pytest.mark.asyncio
async def test_reconnect_target_url_pins_first_attempt():
    """`reconnect(target_url='ws://c')` starts the rotation walk at C
    so the client lands on C (or falls forward if C is dead)."""
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    await client.connect()
    assert client.current_url == "ws://a:9944"
    await client.reconnect(target_url="ws://c:9944")
    assert client.current_url == "ws://c:9944"


@pytest.mark.asyncio
async def test_reconnect_target_url_falls_forward_when_target_dead():
    """If the pool's chosen URL turns out to be dead too, the walk
    continues from there. (Pool might be one step behind reality.)"""
    _StubInterface.bad_urls = {"ws://c:9944"}
    client = SubstrateClient(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    await client.connect()
    await client.reconnect(target_url="ws://c:9944")
    # C was dead, walk wraps to A and lands there.
    assert client.current_url == "ws://a:9944"


def test_no_validator_reachable_str_lists_each_attempt():
    """The exception's str() includes every URL and its failure reason."""
    err = NoValidatorReachable(
        attempts=[
            sc_module.ValidatorAttempt(
                url="ws://a:9944",
                exc_type="ConnectionRefusedError",
                message="Connection refused",
            ),
            sc_module.ValidatorAttempt(
                url="ws://b:9944",
                exc_type="TimeoutError",
                message="timed out after 3s",
            ),
        ]
    )
    rendered = str(err)
    assert "ws://a:9944" in rendered
    assert "ws://b:9944" in rendered
    assert "ConnectionRefusedError" in rendered
    assert "TimeoutError" in rendered
    assert "Connection refused" in rendered
    assert "timed out after 3s" in rendered
