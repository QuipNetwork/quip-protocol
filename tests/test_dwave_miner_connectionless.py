# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the connection-less worker DWaveMiner (connect=False).

The worker miner owns the budget gate, parameter adaptation, and dispatch
loop but holds no D-Wave connection; the single connection lives in the
stream-driver process. These tests must NEVER reach the QPU.
"""
from __future__ import annotations


def test_dwave_miner_connect_false_builds_no_sampler():
    """connect=False constructs without a sampler but keeps the machinery."""
    from QPU.dwave_miner import DWaveMiner

    m = DWaveMiner(miner_id="worker-orchestrator", connect=False)
    assert m.sampler is None
    # budget/param machinery still present
    assert m.queue_depth >= 1
    # cleanup tolerates no sampler (it calls _graceful_exit -> sys.exit(0))
    try:
        m._cleanup_handler(15, None)
    except SystemExit:
        pass


def test_dwave_miner_sample_raises_not_implemented():
    """The legacy synchronous _sample fallback is gone; the stub raises."""
    import pytest

    from QPU.dwave_miner import DWaveMiner

    m = DWaveMiner(miner_id="worker-orchestrator", connect=False)
    with pytest.raises(NotImplementedError):
        m._sample({}, {}, num_reads=1, num_sweeps=1)
