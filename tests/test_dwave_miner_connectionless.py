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


def test_build_persistent_context_forwards_topology():
    """build_persistent_context passes topology to DWaveMiner and returns StreamContext."""
    from unittest.mock import patch
    from dwave_topologies import DEFAULT_TOPOLOGY
    from shared.stream_context import StreamContext
    import QPU.dwave_miner as dm

    with patch.object(dm, "DWaveMiner") as mk:
        result = dm.build_persistent_context(
            miner_id="m", queue_depth=2, nodes=[0, 1, 2], edges=[(0, 1)],
            feeder_buffer_size=4, num_reads=4, annealing_time=80.0,
            energy_threshold_milli=0,
            topology=DEFAULT_TOPOLOGY,
        )
    _a, kwargs = mk.call_args
    assert kwargs["topology"] is DEFAULT_TOPOLOGY
    assert isinstance(result, StreamContext)


def test_build_persistent_context_requires_topology():
    """build_persistent_context raises ValueError when topology is None."""
    import pytest
    import QPU.dwave_miner as dm

    with pytest.raises(ValueError, match="requires a topology"):
        dm.build_persistent_context(
            miner_id="m", queue_depth=2, nodes=[0, 1, 2], edges=[(0, 1)],
            feeder_buffer_size=4, num_reads=4, annealing_time=80.0,
            energy_threshold_milli=0,
            topology=None,
        )


def test_connectionless_finalize_sample_reconstructs_with_defect():
    """The worker miner reconstructs a clamped sample WITHOUT a live sampler.

    Regression for the production crash: D-Wave changed the QPU graph_id,
    introducing an offline qubit. The stream driver then stripped that qubit
    and shipped a reduced sampleset + DefectInfo to the connection-less worker.
    The worker's _finalize_sample must reconstruct the full-topology sample on
    its own — it has no D-Wave connection (self.sampler is None), so it must
    NOT depend on a live sampler instance to do this.
    """
    import numpy as np
    from QPU.dwave_miner import DWaveMiner
    from QPU.dwave_sampler import DefectInfo
    from shared.base_miner import _SharedSampleSet

    m = DWaveMiner(miner_id="worker-orchestrator", connect=False)
    assert m.sampler is None  # the precondition that triggered the crash

    # Topology has 3 qubits; qubit 2 went offline and was clamped to +1. The
    # stream driver ships a label-less _SharedSampleSet (raw int8 matrix), so
    # reconstruction is positional against ``nodes``. Live columns map to the
    # non-clamped nodes [0, 1] and carry [-1, 1].
    reduced = _SharedSampleSet(
        np.array([[-1, 1]], dtype=np.int8), np.array([-5.0], dtype=np.float64),
    )
    defect_info = DefectInfo(
        fixed_spins={2: 1}, energy_offset=2.0, removed_edges={},
    )

    full = m._finalize_sample(reduced, defect_info, [0, 1, 2])

    row = full.record.sample[0]
    assert list(row) == [-1, 1, 1], "clamped qubit must reappear with its spin"
    assert full.record.energy[0] == -3.0, "energy must be reduced energy + offset"
