# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""build_miner_from_spec wires the topology to every backend + fails on None."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dwave_topologies import DEFAULT_TOPOLOGY
from shared import miner_worker


def test_build_qpu_forwards_topology_connect_false():
    spec = {
        "id": "QPU-1",
        "kind": "qpu",
        "cfg": {"qpu_type": "dwave", "solver": "Advantage2_system1"},
        "args": {"topology": DEFAULT_TOPOLOGY},
    }
    with patch.object(miner_worker.QPU, "DWaveMiner") as mk:
        miner_worker.build_miner_from_spec(spec)
    _args, kwargs = mk.call_args
    assert kwargs["topology"] is DEFAULT_TOPOLOGY
    assert kwargs["connect"] is False
    assert kwargs.get("solver_name") == "Advantage2_system1"


def test_build_qpu_parses_reservoir_budget_knobs():
    spec = {
        "id": "QPU-1",
        "kind": "qpu",
        "cfg": {
            "qpu_type": "dwave",
            "daily_budget": "30m",
            "min_block_budget": "90s",
            "budget_cap": "5m",
        },
        "args": {"topology": DEFAULT_TOPOLOGY},
    }
    with patch.object(miner_worker.QPU, "DWaveMiner") as mk:
        miner_worker.build_miner_from_spec(spec)
    _args, kwargs = mk.call_args
    tc = kwargs["time_config"]
    assert tc.daily_budget_seconds == 1800.0
    assert tc.min_block_budget_seconds == 90.0
    assert tc.budget_cap_seconds == 300.0
    # Reservoir keys must not leak into the miner cfg kwargs.
    assert "min_block_budget" not in kwargs
    assert "budget_cap" not in kwargs


def test_build_qpu_defaults_min_block_budget_when_absent():
    spec = {
        "id": "QPU-1",
        "kind": "qpu",
        "cfg": {"qpu_type": "dwave", "daily_budget": "30m"},
        "args": {"topology": DEFAULT_TOPOLOGY},
    }
    with patch.object(miner_worker.QPU, "DWaveMiner") as mk:
        miner_worker.build_miner_from_spec(spec)
    tc = mk.call_args.kwargs["time_config"]
    assert tc.min_block_budget_seconds == 90.0
    assert tc.budget_cap_seconds is None


def test_build_qpu_without_topology_raises():
    spec = {"id": "QPU-1", "kind": "qpu", "cfg": {"qpu_type": "dwave"}, "args": {}}
    with pytest.raises(ValueError, match="requires a topology"):
        miner_worker.build_miner_from_spec(spec)


def test_build_cpu_without_topology_raises():
    spec = {"id": "CPU-1", "kind": "cpu", "args": {}}
    with pytest.raises(ValueError, match="requires a topology"):
        miner_worker.build_miner_from_spec(spec)
