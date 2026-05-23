"""Unit tests for \`substrate.pool.ValidatorPool\`.

The pool sits between the controllers and the `SubstrateClient`
instances they would otherwise own directly. These tests cover the
parts that matter for that role:

- slot caching (one client per role, reused across get() calls)
- lazy connect (no network in __init__ / get() — only when the slot is
  asked to do work; matches today's controller behavior)
- forward-only rotation pointer that's idempotent under concurrent
  `advance_rotation` calls (the "two slots noticed the same death"
  race)
- close() closes every constructed slot, is idempotent, and doesn't
  touch un-instantiated slots
- test_slots injection lets unit tests skip the connect step entirely
"""
from __future__ import annotations

import asyncio

import pytest

from substrate import client as sc_module
from substrate.client import NoValidatorReachable, SubstrateClient
from substrate.pool import ValidatorPool


class _StubInterface:
    """Same shape as the stub used in test_substrate_client_failover."""

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
    _StubInterface.bad_urls = set()
    _StubInterface.construction_log = []
    monkeypatch.setattr(sc_module, "SubstrateInterface", _StubInterface)
    yield


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------


def test_empty_urls_rejected():
    with pytest.raises(ValueError, match="at least one validator URL"):
        ValidatorPool(urls=[])


def test_current_url_starts_at_first_entry():
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944"])
    assert pool.current_url == "ws://a:9944"
    assert pool.urls == ("ws://a:9944", "ws://b:9944")


# ----------------------------------------------------------------------
# get() — slot lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_lazy_connects_on_first_call():
    pool = ValidatorPool(urls=["ws://a:9944"])
    # Before any get(), no client constructed.
    assert _StubInterface.construction_log == []
    client = await pool.get("rpc")
    # After get(), the slot is built and connected.
    assert isinstance(client, SubstrateClient)
    assert client.current_url == "ws://a:9944"
    assert _StubInterface.construction_log == ["ws://a:9944"]


@pytest.mark.asyncio
async def test_get_caches_slot_per_role():
    """Same role → same client instance. Different roles → different."""
    pool = ValidatorPool(urls=["ws://a:9944"])
    rpc1 = await pool.get("rpc")
    rpc2 = await pool.get("rpc")
    sub = await pool.get("subscribe.pow")
    assert rpc1 is rpc2
    assert sub is not rpc1
    # Two clients constructed total (one per role).
    assert _StubInterface.construction_log == ["ws://a:9944", "ws://a:9944"]


@pytest.mark.asyncio
async def test_get_propagates_no_validator_reachable():
    """If the slot can't connect to any URL, get() raises."""
    _StubInterface.bad_urls = {"ws://a:9944", "ws://b:9944"}
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944"])
    with pytest.raises(NoValidatorReachable):
        await pool.get("rpc")


# ----------------------------------------------------------------------
# advance_rotation() — forward-only, idempotent under races
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advance_rotation_moves_pointer_forward():
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    assert pool.current_url == "ws://a:9944"
    next_url = await pool.advance_rotation(from_url="ws://a:9944")
    assert next_url == "ws://b:9944"
    assert pool.current_url == "ws://b:9944"


@pytest.mark.asyncio
async def test_advance_rotation_wraps_circularly():
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944"])
    await pool.advance_rotation(from_url="ws://a:9944")  # → b
    next_url = await pool.advance_rotation(from_url="ws://b:9944")  # → a
    assert next_url == "ws://a:9944"


@pytest.mark.asyncio
async def test_advance_rotation_is_idempotent_under_race():
    """Two slots both notice 'a' is dead and both call advance(from_url='a').
    Only one rotation must happen. Second caller learns the pool is already
    on 'b' and just gets that back — does NOT advance to 'c'."""
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    first = await pool.advance_rotation(from_url="ws://a:9944")
    second = await pool.advance_rotation(from_url="ws://a:9944")  # stale request
    assert first == "ws://b:9944"
    assert second == "ws://b:9944"
    assert pool.current_url == "ws://b:9944"  # NOT 'c'


@pytest.mark.asyncio
async def test_advance_rotation_handles_concurrent_callers():
    """Two coroutines call advance() simultaneously from the same stale URL.
    The pool's internal lock must serialize them so only one advance happens."""
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944", "ws://c:9944"])
    results = await asyncio.gather(
        pool.advance_rotation(from_url="ws://a:9944"),
        pool.advance_rotation(from_url="ws://a:9944"),
    )
    assert results == ["ws://b:9944", "ws://b:9944"]
    assert pool.current_url == "ws://b:9944"


@pytest.mark.asyncio
async def test_advance_rotation_from_unknown_url_returns_current():
    """If a slot reports a URL that isn't even in the rotation (shouldn't
    happen but defend), don't advance — just return current."""
    pool = ValidatorPool(urls=["ws://a:9944", "ws://b:9944"])
    next_url = await pool.advance_rotation(from_url="ws://stale:9944")
    assert next_url == "ws://a:9944"
    assert pool.current_url == "ws://a:9944"


# ----------------------------------------------------------------------
# test_slots injection
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_slots_inject_returns_provided_client_without_connect():
    fake_client = object()  # sentinel — pool must not call anything on it
    pool = ValidatorPool(
        urls=["ws://a:9944"],
        test_slots={"subscribe.pow": fake_client},
    )
    got = await pool.get("subscribe.pow")
    assert got is fake_client
    # No real SubstrateInterface was constructed.
    assert _StubInterface.construction_log == []


@pytest.mark.asyncio
async def test_test_slots_only_overrides_named_roles():
    """A test_slots map covers some roles; the rest fall through to real
    construction."""
    fake_subscribe = object()
    pool = ValidatorPool(
        urls=["ws://a:9944"],
        test_slots={"subscribe.pow": fake_subscribe},
    )
    sub = await pool.get("subscribe.pow")
    rpc = await pool.get("rpc")
    assert sub is fake_subscribe
    assert isinstance(rpc, SubstrateClient)


# ----------------------------------------------------------------------
# close()
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_closes_every_constructed_slot():
    pool = ValidatorPool(urls=["ws://a:9944"])
    rpc = await pool.get("rpc")
    sub = await pool.get("subscribe.pow")
    assert rpc._iface is not None
    assert sub._iface is not None
    await pool.close()
    assert rpc._iface is None
    assert sub._iface is None


@pytest.mark.asyncio
async def test_close_is_idempotent():
    pool = ValidatorPool(urls=["ws://a:9944"])
    await pool.get("rpc")
    await pool.close()
    # Second close shouldn't raise.
    await pool.close()


@pytest.mark.asyncio
async def test_close_with_no_slots_constructed():
    """A pool that was never asked for a slot can still be closed."""
    pool = ValidatorPool(urls=["ws://a:9944"])
    await pool.close()  # should not raise


@pytest.mark.asyncio
async def test_close_does_not_touch_test_slot_objects():
    """test_slots are caller-owned; pool shouldn't try to .close() them."""
    closed_calls = []

    class _FakeClient:
        async def close(self):
            closed_calls.append(True)

    fake = _FakeClient()
    pool = ValidatorPool(urls=["ws://a:9944"], test_slots={"rpc": fake})
    await pool.get("rpc")
    await pool.close()
    assert closed_calls == []  # caller closes their injected objects
