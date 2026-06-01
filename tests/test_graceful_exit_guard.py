# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for BaseMiner._graceful_exit() and the SIGTERM handler guard.

The guard prevents ``sys.exit(0)`` from raising ``SystemExit`` during
interpreter finalization, which would produce:

    Exception ignored in: <module 'threading' ...>
    ...
    File "CPU/sa_miner.py", line 59, in _cleanup_handler
      sys.exit(0)
    SystemExit: 0

These tests verify:
  1. When ``sys.is_finalizing()`` returns True, ``_graceful_exit`` returns
     without raising SystemExit.
  2. When ``sys.is_finalizing()`` returns False (normal runtime),
     ``_graceful_exit`` raises SystemExit(0) as expected.
  3. The SA miner's ``_cleanup_handler`` respects the guard end-to-end.

All tests use a minimal stub sampler so no real hardware (QPU/GPU) is touched.
"""
from __future__ import annotations

import signal
import sys

import pytest

from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements


# ---------------------------------------------------------------------------
# Minimal stub sampler + concrete BaseMiner for testing
# ---------------------------------------------------------------------------

class _StubSampler:
    """Bare-minimum sampler stub — provides topology info, nothing else."""
    nodes = list(range(10))
    edges = [(i, i + 1) for i in range(9)]

    def sample_ising(self, h, J, **_):
        raise NotImplementedError("stub")


class _ConcreteMiner(BaseMiner):
    """Thinnest possible concrete subclass for testing BaseMiner methods."""

    def _adapt_mining_params(self, requirements, nodes, edges):
        return {"num_sweeps": 64, "num_reads": 64}


# ---------------------------------------------------------------------------
# Tests for BaseMiner._graceful_exit()
# ---------------------------------------------------------------------------

class TestGracefulExit:
    def _make_miner(self) -> _ConcreteMiner:
        return _ConcreteMiner(miner_id="test-miner", sampler=_StubSampler())

    def test_exits_with_system_exit_when_not_finalizing(self, monkeypatch):
        """During normal runtime _graceful_exit raises SystemExit(0)."""
        monkeypatch.setattr(sys, "is_finalizing", lambda: False)
        miner = self._make_miner()
        with pytest.raises(SystemExit) as exc_info:
            miner._graceful_exit()
        assert exc_info.value.code == 0

    def test_returns_silently_when_finalizing(self, monkeypatch):
        """During interpreter shutdown _graceful_exit returns without raising."""
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        miner = self._make_miner()
        # Must not raise anything
        miner._graceful_exit()

    def test_static_method_callable_on_class(self, monkeypatch):
        """_graceful_exit is a static method callable directly on the class."""
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        BaseMiner._graceful_exit()  # should not raise


# ---------------------------------------------------------------------------
# Tests for SimulatedAnnealingMiner._cleanup_handler
# ---------------------------------------------------------------------------

class TestSAMinerCleanupHandler:
    """End-to-end: SA miner's SIGTERM handler respects the is_finalizing guard."""

    def _make_sa_miner(self):
        from CPU.sa_miner import SimulatedAnnealingMiner
        from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
        sampler = SimulatedAnnealingStructuredSampler()
        return SimulatedAnnealingMiner(miner_id="sa-test", sampler=sampler)

    def test_cleanup_handler_raises_system_exit_normally(self, monkeypatch):
        """SIGTERM handler exits normally during regular runtime."""
        monkeypatch.setattr(sys, "is_finalizing", lambda: False)
        miner = self._make_sa_miner()
        with pytest.raises(SystemExit) as exc_info:
            miner._cleanup_handler(signal.SIGTERM, None)
        assert exc_info.value.code == 0

    def test_cleanup_handler_silent_during_finalization(self, monkeypatch):
        """SIGTERM handler returns silently when interpreter is shutting down."""
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        miner = self._make_sa_miner()
        # Must not raise SystemExit or any other exception
        miner._cleanup_handler(signal.SIGTERM, None)
