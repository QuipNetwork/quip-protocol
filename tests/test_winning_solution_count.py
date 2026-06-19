# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for `_count_map_keys` — the WinningSolutions key counter.

The chain has no stored solution counter; the ordinal solution number is
`count(WinningSolutions) + 1`, derived by enumerating the map's storage keys
via `state_getKeysPaged`. These tests pin the paging loop (single page,
multi-page continuation, empty map) against a fake substrate-interface.
"""
from __future__ import annotations

from types import SimpleNamespace

from substrate.client import SubstrateClient, _count_map_keys


class _FakeIface:
    """Minimal substrate-interface stand-in for key-paging tests.

    Serves ``keys`` in pages of ``page_size``, honoring the ``start``
    continuation cursor exactly as ``state_getKeysPaged`` does (start is
    *exclusive* — paging resumes after the last returned key).
    """

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self.calls = 0

    def generate_storage_hash(self, *, storage_module: str, storage_function: str) -> str:
        assert storage_module == "QuantumPow"
        assert storage_function == "WinningSolutions"
        return "0xPREFIX"

    def rpc_request(self, method: str, params: list) -> dict:
        assert method == "state_getKeysPaged"
        prefix, page_size, start = params
        assert prefix == "0xPREFIX"
        self.calls += 1
        begin = 0 if start is None else self._keys.index(start) + 1
        return {"result": self._keys[begin:begin + page_size]}


def test_count_single_page() -> None:
    iface = _FakeIface([f"0x{i:02x}" for i in range(5)])
    assert _count_map_keys(iface, "QuantumPow", "WinningSolutions", page_size=1000) == 5
    assert iface.calls == 1  # one short page → no extra RPC


def test_count_multi_page_continuation() -> None:
    iface = _FakeIface([f"0x{i:04x}" for i in range(2500)])
    # 2500 keys at page_size=1000 → pages of 1000, 1000, 500 (3 RPCs).
    assert _count_map_keys(iface, "QuantumPow", "WinningSolutions", page_size=1000) == 2500
    assert iface.calls == 3


def test_count_exact_multiple_stops_on_empty_page() -> None:
    iface = _FakeIface([f"0x{i:04x}" for i in range(2000)])
    # Exact 2×page → a full second page forces a third (empty) request.
    assert _count_map_keys(iface, "QuantumPow", "WinningSolutions", page_size=1000) == 2000
    assert iface.calls == 3


def test_count_empty_map() -> None:
    iface = _FakeIface([])
    assert _count_map_keys(iface, "QuantumPow", "WinningSolutions") == 0
    assert iface.calls == 1


# ----------------------------------------------------------------------
# query_winning_solution_count — runtime-version fallback (MR !35)
# ----------------------------------------------------------------------


class _RuntimeFakeIface:
    """Fake serving either the new ``QBlockCount`` value or the legacy
    ``WinningSolutions`` map, depending on ``has_qblock_count``."""

    def __init__(
        self, *, has_qblock_count: bool, qblock_count: int = 0,
        winning_keys: list[str] | None = None,
    ) -> None:
        self._has_qblock_count = has_qblock_count
        self._qblock_count = qblock_count
        self._keys = winning_keys or []
        self.queried: list[tuple[str, str]] = []

    def get_metadata_storage_function(self, module, name, block_hash=None):
        if name == "QBlockCount":
            return object() if self._has_qblock_count else None
        return object()  # legacy WinningSolutions always present

    def query(self, module, name, *args, **kwargs):
        self.queried.append((module, name))
        assert (module, name) == ("QuantumPow", "QBlockCount")
        return SimpleNamespace(value=self._qblock_count)

    def generate_storage_hash(self, *, storage_module, storage_function):
        assert (storage_module, storage_function) == ("QuantumPow", "WinningSolutions")
        return "0xPREFIX"

    def rpc_request(self, method, params):
        assert method == "state_getKeysPaged"
        _, page_size, start = params
        begin = 0 if start is None else self._keys.index(start) + 1
        return {"result": self._keys[begin:begin + page_size]}


def _client_with(iface: _RuntimeFakeIface) -> SubstrateClient:
    client = SubstrateClient(url="ws://unused")
    client._iface = iface  # type: ignore[attr-defined]

    async def _passthrough(fn, *, idempotent: bool = False):
        return fn()

    client._run = _passthrough  # type: ignore[assignment,method-assign]
    return client


async def test_count_new_runtime_reads_qblock_count() -> None:
    """New runtime: a single ``QBlockCount`` read, no key paging."""
    iface = _RuntimeFakeIface(has_qblock_count=True, qblock_count=42)
    count = await _client_with(iface).query_winning_solution_count()
    assert count == 42
    assert iface.queried == [("QuantumPow", "QBlockCount")]


async def test_count_new_runtime_zero_on_fresh_chain() -> None:
    iface = _RuntimeFakeIface(has_qblock_count=True, qblock_count=0)
    assert await _client_with(iface).query_winning_solution_count() == 0


async def test_count_old_runtime_falls_back_to_key_count() -> None:
    """Old runtime (no ``QBlockCount``): count legacy ``WinningSolutions`` keys."""
    iface = _RuntimeFakeIface(
        has_qblock_count=False, winning_keys=[f"0x{i:02x}" for i in range(7)],
    )
    count = await _client_with(iface).query_winning_solution_count()
    assert count == 7
    assert iface.queried == []  # never touched QBlockCount
