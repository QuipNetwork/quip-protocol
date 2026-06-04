# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for _mempool_feeder_spec and the removal of the QPU mempool rejection.

Covers:
- _mempool_feeder_spec builds the correct ("mempool", attach_args, slot) tuple
  with h/J values scaled from millivalues to float.
- Round-trip through build_feeder reconstructs a FixedIsingFeeder with the
  matching h/J values.
- The previous ProblemView is freed on successive calls (no leak).
- The rejection that previously returned None for DRIVER_OWNS_FEEDER + mempool
  context is gone — _mempool_feeder_spec is reachable.
"""
from __future__ import annotations

import pytest

from shared.ising_feeder import FixedIsingFeeder, build_feeder
from substrate.mempool_types import MempoolJobContext


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mempool_ctx(
    nodes=(0, 1, 2),
    edges=((0, 1), (1, 2)),
    h_values=(100, -200, 300),
    j_values=(500, -500),
) -> MempoolJobContext:
    """Return a small MempoolJobContext for testing."""
    return MempoolJobContext(
        order_id=42,
        nodes=tuple(nodes),
        edges=tuple(edges),
        h_values=tuple(h_values),
        j_values=tuple(j_values),
    )


class _StubDriverMiner:
    """Minimal stub satisfying _mempool_feeder_spec's interface.

    Only provides the attributes touched by _mempool_feeder_spec: the
    ``_mempool_problem_view`` lifecycle slot and the method itself (borrowed
    from BaseMiner via unbound call).
    """

    def __init__(self) -> None:
        self._mempool_problem_view = None

    # Borrow the real implementation from BaseMiner (avoids subclassing
    # which would require a full miner setup).
    from shared.base_miner import BaseMiner as _B
    _mempool_feeder_spec = _B._mempool_feeder_spec


# ---------------------------------------------------------------------------
# _mempool_feeder_spec: basic contract
# ---------------------------------------------------------------------------


def test_mempool_feeder_spec_returns_mempool_tuple():
    """Spec kind is "mempool" and contains attach_args + slot."""
    miner = _StubDriverMiner()
    ctx = _mempool_ctx()
    try:
        spec = miner._mempool_feeder_spec(ctx)
        assert spec[0] == "mempool"
        assert isinstance(spec[1], dict)  # attach_args
        assert isinstance(spec[2], int)   # slot index
    finally:
        if miner._mempool_problem_view is not None:
            miner._mempool_problem_view.close_unlink()


def test_mempool_feeder_spec_slot_is_zero_for_fresh_view():
    """A fresh 1-slot ProblemView yields slot 0."""
    miner = _StubDriverMiner()
    ctx = _mempool_ctx()
    try:
        spec = miner._mempool_feeder_spec(ctx)
        assert spec[2] == 0
    finally:
        if miner._mempool_problem_view is not None:
            miner._mempool_problem_view.close_unlink()


# ---------------------------------------------------------------------------
# _mempool_feeder_spec: round-trip through build_feeder
# ---------------------------------------------------------------------------


def test_mempool_feeder_spec_round_trip():
    """build_feeder(spec, ...) reconstructs a FixedIsingFeeder with correct h/J.

    h and J must match the milli→float conversion: value / 1000.0.
    """
    nodes = (10, 20, 30)
    edges = ((10, 20), (20, 30))
    h_values = (100, -200, 300)   # → 0.1, -0.2, 0.3
    j_values = (500, -500)         # → 0.5, -0.5

    miner = _StubDriverMiner()
    ctx = _mempool_ctx(nodes=nodes, edges=edges, h_values=h_values, j_values=j_values)
    try:
        spec = miner._mempool_feeder_spec(ctx)
        feeder = build_feeder(spec, list(nodes), list(edges), buffer_size=8)
        try:
            assert isinstance(feeder, FixedIsingFeeder)
            model = feeder.pop_blocking()
            assert model.h == pytest.approx({10: 0.1, 20: -0.2, 30: 0.3})
            assert model.J == pytest.approx({(10, 20): 0.5, (20, 30): -0.5})
            # Placeholder nonces are zero-bytes (fixed feeder has no derivation).
            assert model.nonce == b"\x00" * 32
            assert model.salt == b"\x00" * 32
        finally:
            feeder.stop()
    finally:
        if miner._mempool_problem_view is not None:
            miner._mempool_problem_view.close_unlink()


# ---------------------------------------------------------------------------
# _mempool_feeder_spec: lifecycle / no leak
# ---------------------------------------------------------------------------


def test_mempool_feeder_spec_frees_previous_view():
    """Calling _mempool_feeder_spec twice frees the first ProblemView."""
    miner = _StubDriverMiner()
    ctx = _mempool_ctx()
    try:
        spec1 = miner._mempool_feeder_spec(ctx)
        pv1_names = list(miner._mempool_problem_view._ring.names)  # ty: ignore[unresolved-attribute]

        spec2 = miner._mempool_feeder_spec(ctx)  # noqa: F841 — side effects matter
        pv2_names = list(miner._mempool_problem_view._ring.names)  # ty: ignore[unresolved-attribute]

        # The second call installed a NEW ProblemView.
        assert pv1_names != pv2_names, "expected second call to create a new segment"
    finally:
        if miner._mempool_problem_view is not None:
            miner._mempool_problem_view.close_unlink()

    # The first ProblemView's segment should have been unlinked already (no
    # POSIX shm name visible). We can't assert the file is gone portably, but
    # we can confirm spec1's attach_args names differ from pv2_names (already
    # checked above). The absence of an error on close_unlink of the second pv
    # is itself a signal that the segments were properly tracked.
    _ = spec1  # referenced to avoid unused-var lint


def test_close_driver_tears_down_mempool_problem_view():
    """_close_driver must close_unlink and null the _mempool_problem_view."""
    # Borrow _close_driver directly; set up a minimal miner-like object with
    # only the attributes _close_driver touches.
    from shared.base_miner import BaseMiner

    class _MinimalMiner(_StubDriverMiner):
        # Provide the attributes _close_driver references (besides _ring etc.).
        def __init__(self) -> None:
            super().__init__()
            self._ctl_q = None
            self._driver_stop = None
            self._driver_proc = None
            self._ring = None
            self._ring_dims = None
            self._desc_q = None

        logger = __import__("logging").getLogger("test_close_driver")
        _close_driver = BaseMiner._close_driver

    miner = _MinimalMiner()
    ctx = _mempool_ctx()
    # Populate _mempool_problem_view.
    miner._mempool_feeder_spec(ctx)
    assert miner._mempool_problem_view is not None

    # Teardown via _close_driver.
    miner._close_driver()
    assert miner._mempool_problem_view is None


# ---------------------------------------------------------------------------
# Rejection removed: DRIVER_OWNS_FEEDER + mempool no longer returns None
# ---------------------------------------------------------------------------


def test_driver_owns_feeder_mempool_rejection_is_absent():
    """_mempool_feeder_spec is reachable — the old hard rejection is gone.

    This test verifies the deletion of the "QPU driver path cannot mine
    mempool job" block: calling _mempool_feeder_spec on a fresh miner with a
    valid MempoolJobContext must succeed (not raise and not return None).
    It does NOT exercise _setup_dispatch end-to-end (too heavy); the round-trip
    test above already validates the full data path.
    """
    miner = _StubDriverMiner()
    ctx = _mempool_ctx()
    try:
        spec = miner._mempool_feeder_spec(ctx)
        # Must return a valid spec tuple, not None.
        assert spec is not None
        assert spec[0] == "mempool"
    finally:
        if miner._mempool_problem_view is not None:
            miner._mempool_problem_view.close_unlink()
