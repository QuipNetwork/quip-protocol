"""Unit tests for `shared.miner_survey.build_miner_survey`.

The survey builder is pure: it reads `MinerCore.miner_handles`, the
signer identity, and an optional controller's `topology_hash`. No chain
RPC, no disk I/O. These tests drive it with hand-built `MinerCore`
instances + `MagicMock` signers so the assertions stay focused on
schema shape and field normalization.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.miner_core import MinerCore
from shared.miner_survey import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_miner_survey,
)


def _signer() -> MagicMock:
    s = MagicMock()
    s.ss58_address.return_value = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    s.account_id_bytes.return_value = bytes(range(32))
    return s


def _fake_topology(**overrides):
    """Minimal duck-typed stand-in for a `DWaveTopology`.

    Keeps `properties.topology.shape` populated as a list so the survey
    builder picks it up via the preferred path, not the m/t fallback.
    """
    base = {
        "num_nodes": 1368,
        "num_edges": 7692,
        "solver_name": "Z(9,2)",
        "topology_type": "zephyr",
        "properties": {"topology": {"type": "zephyr", "shape": [9, 2]}},
        "m": 9,
        "t": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# Empty core
# ----------------------------------------------------------------------


def test_empty_core_returns_valid_survey():
    """An empty miner config still produces a complete, JSON-safe survey."""
    core = MinerCore(node_id="quip-miner-empty", miners_config={})
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    assert survey["schema"] == SCHEMA_NAME
    assert survey["schema_version"] == SCHEMA_VERSION
    assert survey["node_id"] == "quip-miner-empty"
    assert survey["miners"] == []
    assert survey["capabilities"]["miner_types"] == []
    assert survey["capabilities"]["pow"] is False
    assert survey["capabilities"]["mempool"] is False
    assert survey["hardware"]["cpu"]["worker_count"] == 0
    assert survey["hardware"]["gpu"]["available"] is False
    assert survey["hardware"]["qpu"]["available"] is False
    # Whole payload must round-trip through `json.dumps` cleanly — the
    # endpoint hands it to `web.json_response` without further coercion.
    json.dumps(survey)


# ----------------------------------------------------------------------
# CPU core
# ----------------------------------------------------------------------


def test_cpu_core_exposes_capabilities_and_hardware():
    core = MinerCore(
        node_id="quip-miner-cpu",
        miners_config={"cpu": {"num_cpus": 2}},
    )
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    assert survey["capabilities"]["miner_types"] == ["CPU"]
    assert survey["capabilities"]["pow"] is True
    assert survey["hardware"]["cpu"]["worker_count"] == 2
    assert survey["hardware"]["gpu"]["available"] is False
    assert len(survey["miners"]) == 2
    for entry in survey["miners"]:
        assert entry["type"] == "CPU"
        assert entry["backend"] == "simulated_annealing"
        assert entry["device"] is None
        # No topology injected — should be null, not absent.
        assert entry["topology"] is None
        # CPU spec has no extra cfg beyond `args`; merged should be a dict.
        assert isinstance(entry["config"], dict)


# ----------------------------------------------------------------------
# Identity rendering
# ----------------------------------------------------------------------


def test_account_id_rendered_as_0x_hex():
    core = MinerCore(node_id="quip-miner-id", miners_config={})
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    acc = survey["account"]
    assert acc["ss58_address"].startswith("5")
    assert acc["account_id_hex"].startswith("0x")
    assert len(acc["account_id_hex"]) == 2 + 64  # 0x + 32 bytes hex
    assert acc["account_id_hex"] == "0x" + bytes(range(32)).hex()


def test_client_block_is_static():
    core = MinerCore(node_id="quip-miner-client", miners_config={})
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()
    assert survey["client"]["name"] == "quip-miner"
    assert isinstance(survey["client"]["version"], str)
    assert survey["client"]["version"]  # non-empty


# ----------------------------------------------------------------------
# Topology injection
# ----------------------------------------------------------------------


def test_topology_metadata_pulled_from_spec_args():
    """When a topology object lives in `spec.args["topology"]`, the
    survey extracts num_nodes / num_edges / solver_name / shape."""
    core = MinerCore(node_id="quip-miner-topo", miners_config={"cpu": {"num_cpus": 1}})
    try:
        # Inject after the fact — mirrors what `_inject_topology` does
        # in `quip_cli._run_concurrent_miner` before MinerCore boots.
        handle = core.miner_handles[0]
        handle.spec.setdefault("args", {})["topology"] = _fake_topology()
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    topo = survey["miners"][0]["topology"]
    assert topo is not None
    assert topo["num_nodes"] == 1368
    assert topo["num_edges"] == 7692
    assert topo["solver_name"] == "Z(9,2)"
    assert topo["topology_type"] == "zephyr"
    assert topo["topology_shape"] == [9, 2]
    # No controller passed → topology_hash is null.
    assert topo["topology_hash"] is None


def test_topology_hash_pulled_from_controller():
    """When a controller with `topology_hash=<bytes>` is supplied, the
    survey renders it as 0x-hex on every handle that carries a topology."""
    core = MinerCore(node_id="quip-miner-hash", miners_config={"cpu": {"num_cpus": 1}})
    try:
        core.miner_handles[0].spec.setdefault("args", {})["topology"] = _fake_topology()
        controller = SimpleNamespace(topology_hash=bytes.fromhex("ab" * 32))
        survey = build_miner_survey(core, _signer(), controller=controller)
    finally:
        core.close()

    assert survey["miners"][0]["topology"]["topology_hash"] == "0x" + ("ab" * 32)


def test_topology_shape_falls_back_to_mt_attrs():
    """If `properties.topology.shape` is missing/non-list, the builder
    derives the shape list from the topology's `m`/`t` attrs."""
    core = MinerCore(node_id="quip-miner-shape", miners_config={"cpu": {"num_cpus": 1}})
    try:
        topo = _fake_topology(properties={})
        core.miner_handles[0].spec.setdefault("args", {})["topology"] = topo
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    assert survey["miners"][0]["topology"]["topology_shape"] == [9, 2]


# ----------------------------------------------------------------------
# Non-JSON-safe values are filtered out
# ----------------------------------------------------------------------


def test_non_jsonable_args_are_filtered_from_config():
    """A topology object (or other Python-only value) leaking into
    spec.args must not crash the survey or pollute `config` with
    non-serializable entries. Only the `topology` key is allowed to be
    a Python object — and even that one is filtered out of `config`."""
    core = MinerCore(node_id="quip-miner-junk", miners_config={"cpu": {"num_cpus": 1}})
    try:
        handle = core.miner_handles[0]
        args = handle.spec.setdefault("args", {})
        args["topology"] = _fake_topology()
        args["sampler_callback"] = lambda: None  # not JSON-safe
        args["extra_setting"] = 17  # JSON-safe
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()

    config = survey["miners"][0]["config"]
    assert "sampler_callback" not in config
    assert "topology" not in config
    assert config.get("extra_setting") == 17
    # And nothing in the survey blocks json.dumps.
    json.dumps(survey)


# ----------------------------------------------------------------------
# Capability flags reflect handle inventory
# ----------------------------------------------------------------------


def test_pow_capability_true_when_handles_present():
    core = MinerCore(
        node_id="quip-miner-pow", miners_config={"cpu": {"num_cpus": 1}}
    )
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()
    assert survey["capabilities"]["pow"] is True


def test_miner_types_are_sorted_and_deduplicated():
    """`capabilities.miner_types` must be deterministic — sort it so
    indexers can compare across snapshots without normalizing
    themselves."""
    core = MinerCore(node_id="quip-miner-sort", miners_config={"cpu": {"num_cpus": 3}})
    try:
        survey = build_miner_survey(core, _signer())
    finally:
        core.close()
    # 3 CPU handles collapse to a single ["CPU"] capability entry.
    assert survey["capabilities"]["miner_types"] == ["CPU"]


# ----------------------------------------------------------------------
# Defensive: missing controller / signer fields shouldn't raise
# ----------------------------------------------------------------------


def test_controller_without_topology_hash_attr_handled():
    """A controller object that doesn't expose `topology_hash` (or has
    it set to None) renders the field as null instead of crashing."""
    core = MinerCore(node_id="quip-miner-noattr", miners_config={"cpu": {"num_cpus": 1}})
    try:
        core.miner_handles[0].spec.setdefault("args", {})["topology"] = _fake_topology()
        controller = SimpleNamespace()  # no `topology_hash` at all
        survey = build_miner_survey(core, _signer(), controller=controller)
    finally:
        core.close()
    assert survey["miners"][0]["topology"]["topology_hash"] is None


@pytest.mark.parametrize("bad_hash", [b"", None])
def test_controller_with_empty_topology_hash_renders_null(bad_hash):
    core = MinerCore(node_id="quip-miner-empty-hash", miners_config={"cpu": {"num_cpus": 1}})
    try:
        core.miner_handles[0].spec.setdefault("args", {})["topology"] = _fake_topology()
        controller = SimpleNamespace(topology_hash=bad_hash)
        survey = build_miner_survey(core, _signer(), controller=controller)
    finally:
        core.close()
    assert survey["miners"][0]["topology"]["topology_hash"] is None
