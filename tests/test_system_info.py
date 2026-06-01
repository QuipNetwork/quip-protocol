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


# ----------------------------------------------------------------------
# End-to-end: TOML backend tables → descriptor → System.remark payload
#
# Regression tests for the security boundary under the v0.2 regime where
# `[ibm] token = "..."` and similar credential-carrying TOML sections
# now flow through `load_backend_config` into the cpu/gpu/qpu CLI
# subcommands' miner_config dict (which `_auto_identify` then feeds
# into the on-startup System.remark via `_identify_specs_from_miner_config`
# → `build_descriptor` → `to_canonical_json`).
#
# These exercise the FULL chain — loader, spec builder, descriptor
# pipeline, canonical serializer — and assert on the bytes that would
# actually land on chain. The unit-level tests above check individual
# scrubbing layers; these check the system-wide property: "no value
# the operator wrote inside a `token =` key in TOML ever appears in a
# remark payload."
# ----------------------------------------------------------------------


import textwrap as _textwrap


def _build_canonical_payload_from_toml(tmp_path, toml_body: str, node_id: str = "rig"):
    """Helper: TOML → load_backend_config → _build_qpu_specs +
    _build_gpu_specs → build_descriptor → to_canonical_json bytes."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_gpu_specs, _build_qpu_specs

    p = tmp_path / "miner.toml"
    p.write_text(_textwrap.dedent(toml_body))
    backends = load_backend_config(p)

    specs = []
    if any(k in backends for k in ("gpu", "cuda", "nvidia", "metal", "modal")):
        specs.extend(_build_gpu_specs(node_id, backends))
    if any(
        k in backends
        for k in ("qpu", "dwave", "ibm", "braket", "pasqal", "ionq", "origin")
    ):
        specs.extend(_build_qpu_specs(node_id, backends))
    if "cpu" in backends:
        num_cpus = int(backends["cpu"].get("num_cpus", 1))
        for i in range(num_cpus):
            specs.append(
                {"id": f"{node_id}-CPU-{i + 1}", "kind": "cpu", "args": {}}
            )

    desc = build_descriptor(
        node_id=node_id,
        node_name=node_id,
        miner_specs=specs,
        include_system_info=False,
    )
    validate_descriptor(desc)
    return desc, to_canonical_json(desc)


@pytest.mark.parametrize(
    "vendor,sentinel",
    [
        ("ibm", "ibm-shouldnotleak-9f3e2a"),
        ("braket", "braket-shouldnotleak-b7c1d4"),
        ("pasqal", "pasqal-shouldnotleak-77a8b2"),
        ("ionq", "ionq-shouldnotleak-c0ffee"),
        ("origin", "origin-shouldnotleak-deadbeef"),
    ],
)
def test_qpu_token_does_not_leak_to_remark_payload(tmp_path, vendor, sentinel):
    """Each gate-model vendor's `token = "<value>"` in TOML must not
    appear anywhere in the canonical System.remark bytes.

    Regression guard: the spec builder DOES copy the token into the
    spec's cfg block (it's used in-process by the QPU sampler), so the
    scrubbing layer that protects the on-chain payload is the whitelist
    at `_qpu_spec_entry` (`_QPU_HANDLE_FIELD_WHITELIST`). If a future
    contributor adds `"token"` to that whitelist or routes the spec
    through a code path that bypasses `summarize_miners_from_specs`,
    this test catches it.
    """
    desc, payload = _build_canonical_payload_from_toml(
        tmp_path,
        f"""
        [miner]
        validators = ["ws://a:9944"]
        [{vendor}]
        token = "{sentinel}"
        daily_budget = "5m"
        """,
    )
    text = payload.decode("utf-8")
    assert sentinel not in text, (
        f"{vendor} token leaked to remark payload:\n{text}"
    )
    # Belt-and-braces: the vendor entry exists, just sans token.
    assert vendor in desc.miners
    assert desc.miners[vendor].get("daily_budget") == "5m"
    assert "token" not in desc.miners[vendor]


def test_dwave_region_url_does_not_leak_to_remark_payload(tmp_path):
    """`dwave_region_url` is not credential-shaped but also not on the
    QPU whitelist — should never reach the descriptor. Catches a regression
    where someone widens the whitelist to "all fields the operator set"."""
    _, payload = _build_canonical_payload_from_toml(
        tmp_path,
        """
        [miner]
        validators = ["ws://a:9944"]
        [dwave]
        daily_budget = "60s"
        solver = "Advantage2_system1"
        dwave_region_url = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"
        qpu_min_blocks_for_estimation = 7
        qpu_ema_alpha = 0.25
        """,
    )
    text = payload.decode("utf-8")
    assert "dwave_region_url" not in text
    assert "na-west-1" not in text
    assert "qpu_min_blocks_for_estimation" not in text
    assert "qpu_ema_alpha" not in text
    # The two whitelisted keys ARE present.
    assert "Advantage2_system1" in text
    assert "60s" in text


def test_credential_smuggled_through_solver_field_is_rejected_end_to_end(tmp_path):
    """`solver = "DWAVE_API_KEY=secret_value_12345"` in TOML: the strict
    solver-name regex drops the value at `_qpu_spec_entry`, AND the
    descriptor-wide value walk in `validate_descriptor` would catch it
    if anything slipped through. Both layers verified together here."""
    desc, payload = _build_canonical_payload_from_toml(
        tmp_path,
        """
        [miner]
        validators = ["ws://a:9944"]
        [dwave]
        daily_budget = "60s"
        solver = "Advantage2_system1"
        """,
    )
    assert desc.miners["dwave"].get("solver") == "Advantage2_system1"

    # Now the malicious variant.
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "leaky.toml"
    p.write_text(_textwrap.dedent("""
        [miner]
        validators = ["ws://a:9944"]
        [dwave]
        daily_budget = "60s"
        solver = "DWAVE_API_KEY=secret_value_12345"
    """))
    backends = load_backend_config(p)
    specs = _build_qpu_specs("rig", backends)
    desc = build_descriptor(
        node_id="rig",
        node_name="rig",
        miner_specs=specs,
        include_system_info=False,
    )
    # `_qpu_spec_entry` dropped the suspicious solver — descriptor passes
    # validation but with `solver` absent from the dwave entry.
    validate_descriptor(desc)
    assert "solver" not in desc.miners["dwave"]
    payload = to_canonical_json(desc)
    assert "DWAVE_API_KEY" not in payload.decode("utf-8")
    assert "secret_value_12345" not in payload.decode("utf-8")


def test_credential_smuggled_through_daily_budget_is_rejected(tmp_path):
    """`daily_budget` IS on the whitelist — so the value-level scan in
    `validate_descriptor` is what catches credentials smuggled through it.
    A JWT-shaped value should make `validate_descriptor` raise."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "leaky.toml"
    p.write_text(_textwrap.dedent("""
        [miner]
        validators = ["ws://a:9944"]
        [ibm]
        token = "ibm-real-token"
        daily_budget = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZWNyZXQiOiJzaG91bGRub3RsZWFrIn0"
    """))
    backends = load_backend_config(p)
    specs = _build_qpu_specs("rig", backends)
    desc = build_descriptor(
        node_id="rig",
        node_name="rig",
        miner_specs=specs,
        include_system_info=False,
    )
    with pytest.raises(
        DescriptorValidationError, match="credential-shaped-value"
    ):
        validate_descriptor(desc)


def test_multi_vendor_toml_no_secret_appears_in_payload(tmp_path):
    """Realistic mixed-rig TOML — every vendor section populated with a
    credential. Single pass asserts no sentinel survives serialization.
    The closest real-world equivalent to what an operator might write."""
    sentinels = {
        "ibm": "ibm-sentinel-aaaaa",
        "braket": "braket-sentinel-bbbbb",
        "pasqal": "pasqal-sentinel-ccccc",
        "ionq": "ionq-sentinel-ddddd",
        "origin": "origin-sentinel-eeeee",
    }
    toml = '[miner]\nvalidators = ["ws://a:9944"]\n'
    toml += '[dwave]\ndaily_budget = "60s"\nsolver = "Advantage2_system1"\n'
    for vendor, token in sentinels.items():
        toml += f'[{vendor}]\ntoken = "{token}"\ndaily_budget = "5m"\n'
    desc, payload = _build_canonical_payload_from_toml(tmp_path, toml)
    text = payload.decode("utf-8")
    for vendor, token in sentinels.items():
        assert token not in text, f"{vendor} token leaked"
        # And the vendor entry made it through (sans token).
        assert vendor in desc.miners


def test_cpu_args_passed_through_toml_args_table_does_not_leak(tmp_path):
    """`[cpu]` can carry an `args` subtable used to forward sampler
    config (e.g. topology). Verify a forbidden key smuggled into
    `[cpu.args]` doesn't leak. The CPU whitelist is `{"num_cpus"}` —
    nothing else from `[cpu]` makes it past `_cpu_spec_entry`."""
    desc, payload = _build_canonical_payload_from_toml(
        tmp_path,
        """
        [miner]
        validators = ["ws://a:9944"]
        [cpu]
        num_cpus = 2
        [cpu.args]
        api_key = "cpu-args-shouldnotleak-secret123"
        """,
    )
    text = payload.decode("utf-8")
    assert "cpu-args-shouldnotleak-secret123" not in text
    assert "api_key" not in text
    # The legitimate `num_cpus` made it through.
    assert desc.miners["cpu"]["num_cpus"] == 2


def test_modal_gpu_type_passes_but_secret_in_modal_table_does_not(tmp_path):
    """Modal cloud GPUs: `gpu_type` is whitelisted, but any other
    operator-set keys (including a stray `token`) must be scrubbed."""
    desc, payload = _build_canonical_payload_from_toml(
        tmp_path,
        """
        [miner]
        validators = ["ws://a:9944"]
        [modal]
        gpu_type = "a10g"
        token = "modal-shouldnotleak-xyz123"
        """,
    )
    text = payload.decode("utf-8")
    assert "modal-shouldnotleak-xyz123" not in text
    # Modal entry survives with gpu_type intact.
    modal_entries = [v for k, v in desc.miners.items() if k.startswith("modal")]
    assert modal_entries
    assert modal_entries[0].get("gpu_type") == "a10g"
