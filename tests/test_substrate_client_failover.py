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
