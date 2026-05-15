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
