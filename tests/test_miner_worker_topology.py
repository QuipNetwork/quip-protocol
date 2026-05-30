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


def test_build_qpu_without_topology_raises():
    spec = {"id": "QPU-1", "kind": "qpu", "cfg": {"qpu_type": "dwave"}, "args": {}}
    with pytest.raises(ValueError, match="requires a topology"):
        miner_worker.build_miner_from_spec(spec)


def test_build_cpu_without_topology_raises():
    spec = {"id": "CPU-1", "kind": "cpu", "args": {}}
    with pytest.raises(ValueError, match="requires a topology"):
        miner_worker.build_miner_from_spec(spec)
