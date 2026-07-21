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
    with patch("QPU.DWaveMiner") as mk:
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
    with patch("QPU.DWaveMiner") as mk:
        miner_worker.build_miner_from_spec(spec)
    _args, kwargs = mk.call_args
    tc = kwargs["time_config"]
    assert tc.daily_budget_seconds == 1800.0
    assert tc.min_block_budget_seconds == 90.0
    assert tc.budget_cap_seconds == 300.0
    # Reservoir keys must not leak into the miner cfg kwargs.
    assert "min_block_budget" not in kwargs
    assert "budget_cap" not in kwargs


def test_dwave_budget_config_threads_end_to_end(tmp_path):
    """`[dwave]` budget keys reach the QPUTimeConfig through the FULL
    producer→consumer seam (`_build_qpu_specs` → `build_miner_from_spec`).

    The producer (test_miner_core) and consumer (above) are each covered in
    isolation; this ties them so a key-name / string-passthrough drift between
    the two halves can't pass both green while production regresses. Without the
    `_build_qpu_specs` forwarding, `min_block_budget` never reaches the spec and
    this asserts 1200.0 against the 90.0 default — i.e. it reproduces the bug."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "qpu.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "15m"\n'
        'min_block_budget = "20m"\n'
        'budget_cap = "30m"\n'
    )
    spec = _build_qpu_specs("rig", load_backend_config(p))[0]
    spec["args"] = {"topology": DEFAULT_TOPOLOGY}
    with patch("QPU.DWaveMiner") as mk:
        miner_worker.build_miner_from_spec(spec)
    tc = mk.call_args.kwargs["time_config"]
    assert tc.daily_budget_seconds == 900.0
    assert tc.min_block_budget_seconds == 1200.0  # "20m" (bug defaulted to 90)
    assert tc.budget_cap_seconds == 1800.0  # "30m"
    assert "min_block_budget" not in mk.call_args.kwargs
    assert "budget_cap" not in mk.call_args.kwargs


def test_build_qpu_defaults_min_block_budget_when_absent():
    spec = {
        "id": "QPU-1",
        "kind": "qpu",
        "cfg": {"qpu_type": "dwave", "daily_budget": "30m"},
        "args": {"topology": DEFAULT_TOPOLOGY},
    }
    with patch("QPU.DWaveMiner") as mk:
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
