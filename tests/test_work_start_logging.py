# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""BaseMiner._log_work_start should announce a work item at INFO once and
drop repeats of the same item to DEBUG, so a worker that re-arms every head on
unchanged last_proof/topology stops spamming the log."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from shared.base_miner import BaseMiner


class _ConcreteMiner(BaseMiner):
    """Minimal concrete BaseMiner so __new__ isn't blocked by the ABC."""

    def _adapt_mining_params(self, *args, **kwargs):  # pragma: no cover - stub
        return None


def _pow_ctx(last_proof: bytes, topology: bytes = b"\xab" * 32):
    # uses_decay_ratchet() -> True marks this as a PoW work source, so
    # _log_work_start takes the PoW banner branch.
    return SimpleNamespace(
        last_proof_block_hash=last_proof,
        topology_hash=topology,
        nodes=[0, 1, 2],
        edges=[(0, 1)],
        uses_decay_ratchet=lambda: True,
    )


def _bare_miner():
    miner = _ConcreteMiner.__new__(_ConcreteMiner)
    miner.logger = MagicMock()
    miner._last_work_log_msg = None
    return miner


def test_log_work_start_repeat_drops_to_debug():
    miner = _bare_miner()
    miner._log_work_start(_pow_ctx(b"\x01" * 32))   # first → INFO
    miner._log_work_start(_pow_ctx(b"\x01" * 32))   # same → DEBUG
    miner._log_work_start(_pow_ctx(b"\x01" * 32))   # still same → DEBUG
    assert miner.logger.info.call_count == 1
    assert miner.logger.debug.call_count == 2


def test_log_work_start_change_logs_info_again():
    miner = _bare_miner()
    miner._log_work_start(_pow_ctx(b"\x01" * 32))   # INFO
    miner._log_work_start(_pow_ctx(b"\x02" * 32))   # changed last_proof → INFO
    assert miner.logger.info.call_count == 2
    assert miner.logger.debug.call_count == 0
