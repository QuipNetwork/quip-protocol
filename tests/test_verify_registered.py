# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for SubstrateMinerController._verify_registered retry behavior.

Startup self-registers the miner (CLI Guard D) before controllers spawn, but
the controller queries through the validator pool — possibly a node that hasn't
seen the registration block yet. `_verify_registered` retries briefly to absorb
that propagation lag before treating absence as a real failure.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from substrate.miner_controller import SubstrateMinerController

_INFO = SimpleNamespace(deposit=1, proofs_submitted=2, proofs_won=3)


def _verify_ctrl(query_results):
    ctrl = SubstrateMinerController.__new__(SubstrateMinerController)
    ctrl.pool_client = SimpleNamespace(query_miner=AsyncMock(side_effect=query_results))
    ctrl.signer = SimpleNamespace(ss58_address=lambda: "5Test")
    return ctrl


def test_verify_registered_passes_on_first_try():
    ctrl = _verify_ctrl([_INFO])
    asyncio.run(ctrl._verify_registered(b"\x00" * 32))  # no raise


def test_verify_registered_retries_then_passes(monkeypatch):
    monkeypatch.setattr("substrate.miner_controller.asyncio.sleep", AsyncMock())
    ctrl = _verify_ctrl([None, _INFO])
    asyncio.run(ctrl._verify_registered(b"\x00" * 32))  # absorbs one lag, no raise


def test_verify_registered_fails_after_retries(monkeypatch):
    monkeypatch.setattr("substrate.miner_controller.asyncio.sleep", AsyncMock())
    ctrl = _verify_ctrl([None, None, None])
    with pytest.raises(RuntimeError, match="is not in"):
        asyncio.run(ctrl._verify_registered(b"\x00" * 32))
