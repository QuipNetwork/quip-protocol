# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for `_count_map_keys` — the WinningSolutions key counter.

The chain has no stored solution counter; the ordinal solution number is
`count(WinningSolutions) + 1`, derived by enumerating the map's storage keys
via `state_getKeysPaged`. These tests pin the paging loop (single page,
multi-page continuation, empty map) against a fake substrate-interface.
"""
from __future__ import annotations

from substrate.client import _count_map_keys


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
