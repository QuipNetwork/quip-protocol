# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Each streaming backend supplies its own stream-driver factory kwargs."""
from __future__ import annotations

from QPU.dwave_miner import DWaveMiner


def _sample_ctx(nodes):
    return {
        "nodes": nodes,
        "edges": [(0, 1)],
        "num_reads": 112,
        "num_sweeps": 0,
        "annealing_time": 80.0,
        "energy_threshold": -1.0,
        "feeder_buffer_size": 60,
    }


def test_dwave_factory_kwargs_carry_qpu_fields():
    miner = DWaveMiner(miner_id="QPU-1", connect=False)
    nodes = [0, 1]
    kw = miner._stream_factory_kwargs(_sample_ctx(nodes), nodes)
    assert kw["nodes"] is nodes
    assert kw["num_reads"] == 112
    assert kw["annealing_time"] == 80.0
    assert "energy_threshold_milli" in kw
    assert "solver_name" in kw and "region" in kw and "token" in kw
    assert "topology" in kw
    assert kw["queue_depth"] == miner.queue_depth
