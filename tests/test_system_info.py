"""Unit tests for `shared.system_info`.

Covers the v0.1 nodes.json-shaped NodeDescriptor builder, canonical JSON
serialization, field-bound validation, and the value-level secret scrub
(both the strict solver-name check and the catch-all credential pattern
match in `validate_descriptor`).
"""
from __future__ import annotations

import json

import pytest

from shared.system_info import (
    ALLOWED_LOG_LEVELS,
    DESCRIPTOR_VERSION,
    DescriptorValidationError,
    NODE_NAME_MAX_BYTES,
    RPC_ENDPOINTS_MAX_COUNT,
    RPC_ENDPOINT_MAX_BYTES,
    SCHEMA_NAME,
    _is_safe_solver_name,
    _looks_secretish,
    build_descriptor,
    summarize_miners_from_specs,
    to_canonical_json,
    validate_descriptor,
)


# ----------------------------------------------------------------------
# build_descriptor
# ----------------------------------------------------------------------


def test_empty_descriptor_has_schema_and_version():
    """An empty MinerCore still produces a valid, JSON-safe descriptor."""
    desc = build_descriptor(
        node_id="quip-miner-empty",
        node_name="quip-miner-empty",
        include_system_info=False,
    )
    rendered = desc.to_dict()
    assert rendered["schema"] == SCHEMA_NAME
    assert rendered["descriptor_version"] == DESCRIPTOR_VERSION
    assert rendered["node_name"] == "quip-miner-empty"
    assert rendered["miners"] == {}
    assert rendered["rpc_endpoints"] == []
    assert rendered["auto_mine"] is False
    assert rendered["log_level"] == "INFO"
    assert "system_info" not in rendered          # skipped via flag
    assert isinstance(rendered["runtime"], dict)  # always populated
    # round-trips through json.dumps cleanly
    json.dumps(rendered)


def test_descriptor_carries_self_asserted_fields():
    desc = build_descriptor(
        node_id="rig-01",
        node_name="rig-01",
        public_host="rig-01.example.com",
        public_port=20049,
        rpc_endpoints=["ws://rig-01.example.com:9944"],
        auto_mine=True,
        log_level="WARNING",
        include_system_info=False,
    )
    out = desc.to_dict()
    assert out["public_host"] == "rig-01.example.com"
    assert out["public_port"] == 20049
    assert out["rpc_endpoints"] == ["ws://rig-01.example.com:9944"]
    assert out["auto_mine"] is True
    assert out["log_level"] == "WARNING"


# ----------------------------------------------------------------------
# Per-miner shape — must match nodes.json exactly
# ----------------------------------------------------------------------


def test_cpu_entry_aggregates_num_cpus():
    """v0.2 spawns one handle per CPU; the descriptor re-aggregates into
    a single `cpu` bucket with `num_cpus` matching the spawned count."""
    specs = [
        {"id": "rig-CPU-1", "kind": "cpu", "args": {}},
        {"id": "rig-CPU-2", "kind": "cpu", "args": {}},
        {"id": "rig-CPU-3", "kind": "cpu", "args": {}},
    ]
    miners = summarize_miners_from_specs(specs)
    assert miners == {
        "cpu": {"kind": "CPU", "miner_id": "rig-CPU-1", "num_cpus": 3}
    }


def test_gpu_cuda_entry_shape_matches_nodes_json():
    """Per-device GPU entries with `cuda.<device_index>` keys, mirroring
    the v0.1 nodes.json layout."""
    specs = [
        {"id": "rig-GPU-CUDA-0", "kind": "cuda", "args": {"device": "0"}},
        {"id": "rig-GPU-CUDA-1", "kind": "cuda", "args": {"device": "1"}},
    ]
    miners = summarize_miners_from_specs(specs)
    assert set(miners.keys()) == {"cuda.0", "cuda.1"}
    assert miners["cuda.0"] == {
        "kind": "GPU",
        "backend": "cuda",
        "miner_id": "rig-GPU-CUDA-0",
        "device_index": 0,
    }


def test_qpu_dwave_entry_shape_matches_nodes_json():
    """QPU bucket carries provider/solver/daily_budget per nodes.json."""
    specs = [{
        "id": "rig-QPU-DWAVE-1",
        "kind": "qpu",
        "cfg": {
            "qpu_type": "dwave",
            "solver": "Advantage2_system1",
            "daily_budget": "5m",
        },
    }]
    miners = summarize_miners_from_specs(specs)
    assert miners["dwave"] == {
        "kind": "QPU",
        "provider": "dwave",
        "miner_id": "rig-QPU-DWAVE-1",
        "solver": "Advantage2_system1",
        "daily_budget": "5m",
    }


def test_modal_entry_keyed_by_gpu_type():
    specs = [{
        "id": "rig-GPU-MODAL-a100",
        "kind": "modal",
        "args": {"gpu_type": "a100"},
    }]
    miners = summarize_miners_from_specs(specs)
    assert "modal.a100" in miners
    assert miners["modal.a100"]["backend"] == "modal"
    assert miners["modal.a100"]["gpu_type"] == "a100"


# ----------------------------------------------------------------------
# Canonical JSON — sorted keys, compact separators, byte-stable
# ----------------------------------------------------------------------


def test_canonical_json_is_byte_stable():
    """Same descriptor → identical bytes across calls, sorted keys."""
    desc1 = build_descriptor(
        node_id="rig",
        rpc_endpoints=["ws://b", "ws://a"],
        public_port=9000,
        include_system_info=False,
    )
    desc2 = build_descriptor(
        node_id="rig",
        rpc_endpoints=["ws://b", "ws://a"],
        public_port=9000,
        include_system_info=False,
    )
    b1 = to_canonical_json(desc1)
    b2 = to_canonical_json(desc2)
    assert b1 == b2
    # Compact separators (no whitespace).
    assert b" " not in b1
    assert b"\n" not in b1
    # Top-level keys are sorted: `descriptor_version` precedes `node_name`.
    decoded = b1.decode("utf-8")
    assert decoded.startswith("{\"auto_mine\":")
    # `rpc_endpoints` list order is preserved (it carries semantic
    # priority — first entry is operator's primary advertisement).
    assert b"[\"ws://b\",\"ws://a\"]" in b1


def test_canonical_json_omits_system_info_key_when_skipped():
    desc = build_descriptor(
        node_id="rig", include_system_info=False,
    )
    body = json.loads(to_canonical_json(desc))
    assert "system_info" not in body


# ----------------------------------------------------------------------
# validate_descriptor — wire-format bounds
# ----------------------------------------------------------------------


def test_validate_rejects_empty_node_name():
    """`build_descriptor` falls back to `node_id` when `node_name=""`,
    so construct the descriptor directly to exercise the empty-name
    guard — it's a wire-format check, not a builder-side one."""
    desc = build_descriptor(node_id="x", include_system_info=False)
    desc.node_name = ""
    with pytest.raises(DescriptorValidationError, match="node_name.*empty"):
        validate_descriptor(desc)


def test_validate_rejects_oversized_node_name():
    desc = build_descriptor(
        node_id="x",
        node_name="x" * (NODE_NAME_MAX_BYTES + 1),
        include_system_info=False,
    )
    with pytest.raises(DescriptorValidationError, match="node_name.*too-long"):
        validate_descriptor(desc)


def test_validate_rejects_too_many_rpc_endpoints():
    desc = build_descriptor(
        node_id="x",
        rpc_endpoints=[f"ws://r{i}" for i in range(RPC_ENDPOINTS_MAX_COUNT + 1)],
        include_system_info=False,
    )
    with pytest.raises(DescriptorValidationError, match="rpc_endpoints.*too-many"):
        validate_descriptor(desc)


def test_validate_rejects_oversized_rpc_endpoint():
    bad = "ws://" + ("x" * RPC_ENDPOINT_MAX_BYTES)
    desc = build_descriptor(
        node_id="x", rpc_endpoints=[bad], include_system_info=False,
    )
    with pytest.raises(DescriptorValidationError, match=r"rpc_endpoints\[0\].*too-long"):
        validate_descriptor(desc)


def test_validate_rejects_bad_log_level():
    desc = build_descriptor(
        node_id="x", log_level="LOUD", include_system_info=False,
    )
    with pytest.raises(DescriptorValidationError, match="log_level.*unknown"):
        validate_descriptor(desc)


def test_validate_accepts_well_formed_descriptor():
    desc = build_descriptor(
        node_id="x",
        node_name="rig",
        public_host="rig.example.com",
        public_port=20049,
        rpc_endpoints=["ws://rig.example.com:9944"],
        log_level="INFO",
        include_system_info=False,
    )
    validate_descriptor(desc)  # no exception


# ----------------------------------------------------------------------
# Secret scrubbing — value-level defense
# ----------------------------------------------------------------------


def test_safe_solver_name_pattern():
    assert _is_safe_solver_name("Advantage2_system1")
    assert _is_safe_solver_name("DW_2000Q_6")
    assert _is_safe_solver_name("hybrid_binary_quadratic_model_v2")
    # Whitespace, slashes, query strings, equals signs all rejected.
    assert not _is_safe_solver_name("Advantage2 system1")
    assert not _is_safe_solver_name("https://cloud.dwave.com/solver?token=x")
    assert not _is_safe_solver_name("solver=Advantage2_system1")
    assert not _is_safe_solver_name("x" * 81)  # length cap


def test_looks_secretish_catches_known_credential_shapes():
    assert _looks_secretish("DWAVE_API_KEY=abc123def456")
    assert _looks_secretish("AKIAIOSFODNN7EXAMPLE")
    assert _looks_secretish("sk-1234567890abcdefghijklmnopqrstuv")
    assert _looks_secretish("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x")
    assert _looks_secretish("token=mysupersecretpw1234")
    # Innocuous values pass.
    assert not _looks_secretish("Advantage2_system1")
    assert not _looks_secretish("5m")
    assert not _looks_secretish("ws://node.example.com:9944")


def test_qpu_spec_drops_suspicious_solver(caplog):
    """A solver value that doesn't match the strict pattern is dropped
    (with a warning) rather than shipped on chain."""
    specs = [{
        "id": "n-QPU-DWAVE-1",
        "kind": "qpu",
        "cfg": {
            "qpu_type": "dwave",
            "solver": "DWAVE_API_KEY=abc_real_secret_value",
            "daily_budget": "5m",
        },
    }]
    with caplog.at_level("WARNING"):
        miners = summarize_miners_from_specs(specs)
    assert "solver" not in miners["dwave"]
    assert miners["dwave"]["daily_budget"] == "5m"
    assert any(
        "dropping suspicious solver" in r.message for r in caplog.records
    )


def test_validate_rejects_credential_in_rpc_endpoint():
    """Even if a credential survives the per-field scrub, the
    descriptor-wide value walk in validate_descriptor catches it."""
    desc = build_descriptor(
        node_id="x",
        rpc_endpoints=["ws://node:9944/?token=mysupersecretpw1234"],
        include_system_info=False,
    )
    with pytest.raises(
        DescriptorValidationError, match="credential-shaped-value"
    ):
        validate_descriptor(desc)


def test_validate_rejects_credential_in_public_host():
    desc = build_descriptor(
        node_id="x",
        public_host="node-DWAVE_API_KEY-abc.example.com",
        include_system_info=False,
    )
    with pytest.raises(
        DescriptorValidationError, match="credential-shaped-value"
    ):
        validate_descriptor(desc)


# ----------------------------------------------------------------------
# Key-name scrubber (defense in depth)
# ----------------------------------------------------------------------


def test_scrub_drops_secret_named_keys_recursively():
    """`_scrub` drops dict keys whose name contains any forbidden
    substring, even through nested structures."""
    from shared.system_info import _scrub
    leaky = {
        "ok": "value",
        "secret": "shh",
        "deeply": {
            "nested": {"api_key": "ABCDEFG"},
            "creds": [{"password": "x"}, {"username": "u"}],
        },
    }
    cleaned = _scrub(leaky)
    assert cleaned == {
        "ok": "value",
        "deeply": {"nested": {}, "creds": [{}, {"username": "u"}]},
    }


# ----------------------------------------------------------------------
# Allowed log levels are a closed set
# ----------------------------------------------------------------------


@pytest.mark.parametrize("level", sorted(ALLOWED_LOG_LEVELS))
def test_all_allowed_log_levels_pass(level):
    desc = build_descriptor(
        node_id="x", log_level=level, include_system_info=False,
    )
    validate_descriptor(desc)
