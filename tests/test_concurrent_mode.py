"""Tests for the Phase 8d `--mode pow|mempool|both` worker-split logic.

The split helper is a pure function (`_split_handles_for_mode` in
quip_cli) so this exercises it without any chain plumbing. The
integration test for "both" mode is deferred — it requires a registered
PoW miner AND mempool solver running concurrently against the docker
chain, which is more setup than the current Phase 6 / 8c live tests
cover. Phase 9 verification will add that path.
"""
from __future__ import annotations

import pytest

from quip_cli import _split_handles_for_mode


class _StubHandle:
    """Sentinel for split tests — we only need identity comparison."""

    def __init__(self, name: str) -> None:
        self.miner_id = name

    def __repr__(self) -> str:
        return f"_StubHandle({self.miner_id!r})"


def _handles(n: int) -> list:
    return [_StubHandle(f"h{i}") for i in range(n)]


def test_pow_mode_takes_all_handles():
    handles = _handles(4)
    pow_h, mempool_h = _split_handles_for_mode("pow", handles)
    assert pow_h == handles
    assert mempool_h == []


def test_mempool_mode_takes_all_handles():
    handles = _handles(4)
    pow_h, mempool_h = _split_handles_for_mode("mempool", handles)
    assert pow_h == []
    assert mempool_h == handles


def test_both_mode_splits_half_half_even():
    handles = _handles(4)
    pow_h, mempool_h = _split_handles_for_mode("both", handles)
    assert pow_h == handles[:2]
    assert mempool_h == handles[2:]


def test_both_mode_splits_floor_for_odd_count():
    """5 handles → 2 PoW, 3 mempool (mempool gets the remainder).

    This rule is documented in the helper's docstring; it favors mempool
    by 1 when the count is odd. Why mempool? In a 1-worker odd case
    (3, 5, 7) the user explicitly asked for "both" — defaulting the
    remainder to mempool ensures mempool is always represented even at
    minimal handle counts, while PoW (always running on chain heads)
    still has at least 1 worker.
    """
    handles = _handles(5)
    pow_h, mempool_h = _split_handles_for_mode("both", handles)
    assert len(pow_h) == 2
    assert len(mempool_h) == 3


def test_both_mode_with_single_handle_assigns_to_mempool():
    """1 handle in `both` mode produces (0, 1) — empty PoW side. The
    CLI catches this and fails fast with "needs ≥2 handles". The split
    helper itself just returns the floor-half so callers can inspect."""
    handles = _handles(1)
    pow_h, mempool_h = _split_handles_for_mode("both", handles)
    assert len(pow_h) == 0
    assert len(mempool_h) == 1


def test_both_mode_with_two_handles_splits_one_one():
    handles = _handles(2)
    pow_h, mempool_h = _split_handles_for_mode("both", handles)
    assert pow_h == handles[:1]
    assert mempool_h == handles[1:]


def test_split_returns_fresh_lists():
    """The helper must return lists (not slices of the input) so callers
    can mutate without affecting the source."""
    handles = _handles(4)
    pow_h, mempool_h = _split_handles_for_mode("pow", handles)
    pow_h.append(_StubHandle("extra"))
    assert len(handles) == 4
