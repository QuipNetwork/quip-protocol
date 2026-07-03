"""Unit tests for `shared.miner_config` — TOML loader + CLI merge."""
from __future__ import annotations

import pytest

from shared.miner_config import (
    ALL_BACKEND_SECTIONS,
    CPU_BACKEND_SECTIONS,
    GPU_BACKEND_SECTIONS,
    MinerConfigError,
    QPU_BACKEND_SECTIONS,
    SubmissionConfig,
    load_backend_config,
    load_miner_config,
    load_submission_config,
    mempool_owner_group,
    merge_config,
    present_backend_groups,
    validate_merged,
)


# ----------------------------------------------------------------------
# load_miner_config
# ----------------------------------------------------------------------


def test_load_missing_file_raises_with_path(tmp_path):
    missing = tmp_path / "not-there.toml"
    with pytest.raises(MinerConfigError, match=str(missing)):
        load_miner_config(missing)


def test_load_no_miner_table_returns_empty(tmp_path):
    """A TOML without a [miner] section is allowed; loader returns {}."""
    p = tmp_path / "empty.toml"
    p.write_text("[other]\nstuff = 1\n")
    assert load_miner_config(p) == {}


def test_load_parses_validators_list(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944", "ws://b:9944"]\n'
        'signer_key = "~/.quip-miner/signing.json"\n'
        'topology = "zephyr:9,2"\n'
    )
    cfg = load_miner_config(p)
    assert cfg["validators"] == ["ws://a:9944", "ws://b:9944"]
    assert cfg["signer_key"] == "~/.quip-miner/signing.json"
    assert cfg["topology"] == "zephyr:9,2"


def test_load_parses_kind_subtables(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        '[miner.cpu]\n'
        'num_cpus = 4\n'
        '[miner.gpu]\n'
        'backend = "metal"\n'
    )
    cfg = load_miner_config(p)
    assert cfg["cpu"] == {"num_cpus": 4}
    assert cfg["gpu"] == {"backend": "metal"}


def test_load_rejects_malformed_toml(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("[miner\nvalidators = [")
    with pytest.raises(MinerConfigError, match="parse failed"):
        load_miner_config(p)


def test_load_rejects_non_string_validator(tmp_path):
    """Each entry in `validators` must be a string — catch typos early."""
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944", 9944]\n')
    with pytest.raises(MinerConfigError, match="validators"):
        load_miner_config(p)


# ----------------------------------------------------------------------
# merge_config (CLI > TOML)
# ----------------------------------------------------------------------


def test_merge_cli_overrides_toml():
    """CLI values win over TOML — covers list, scalar, and rest_host/rest_port."""
    toml = {
        "validators": ["ws://toml:9944"],
        "topology": "zephyr:9,2",
        "rest_host": "127.0.0.1",
        "rest_port": 8086,
    }
    cli = {
        "validators": ("ws://cli:9944",),
        "topology": "zephyr:10,2",
        "rest_host": "0.0.0.0",
        "rest_port": 9000,
    }
    merged = merge_config(toml, cli)
    # CLI list materialized to a list — order preserved.
    assert merged["validators"] == ["ws://cli:9944"]
    assert merged["topology"] == "zephyr:10,2"
    assert merged["rest_host"] == "0.0.0.0"
    assert merged["rest_port"] == 9000


def test_merge_falls_back_to_toml_when_cli_empty():
    """Empty tuples / None / unset CLI values must NOT clobber TOML —
    covers validators, signer_key, topology, and rest_host/rest_port."""
    toml = {
        "validators": ["ws://toml:9944"],
        "signer_key": "~/.quip-miner/signing.json",
        "topology": "zephyr:9,2",
        "rest_host": "0.0.0.0",
        "rest_port": 8086,
    }
    cli = {
        "validators": (),
        "signer_key": None,
        "topology": None,
        "rest_host": None,
        "rest_port": None,
    }
    merged = merge_config(toml, cli)
    assert merged["validators"] == ["ws://toml:9944"]
    assert merged["signer_key"] == "~/.quip-miner/signing.json"
    assert merged["topology"] == "zephyr:9,2"
    assert merged["rest_host"] == "0.0.0.0"
    assert merged["rest_port"] == 8086


def test_merge_unknown_cli_key_passes_through():
    """Extra CLI keys (e.g. num_cpus on the cpu command) get merged in."""
    merged = merge_config({}, {"validators": ("ws://a:9944",), "num_cpus": 4})
    assert merged["num_cpus"] == 4


# ----------------------------------------------------------------------
# validate_merged
# ----------------------------------------------------------------------


def test_validate_passes_with_validators_and_signer_key():
    merged = {
        "validators": ["ws://a:9944"],
        "signer_key": "~/.quip-miner/signing.json",
    }
    validate_merged(merged)  # no raise


def test_validate_empty_validators_raises_with_actionable_message():
    with pytest.raises(MinerConfigError, match="at least one validator"):
        validate_merged({"validators": [], "signer_key": "/tmp/k.json"})


def test_validate_missing_validators_key_raises():
    with pytest.raises(MinerConfigError, match="validator"):
        validate_merged({"signer_key": "/tmp/k.json"})


def test_validate_missing_signer_key_raises():
    with pytest.raises(MinerConfigError, match="signer_key"):
        validate_merged({"validators": ["ws://a:9944"]})


# ----------------------------------------------------------------------
# v0.1 alias resolution (listen/port -> rest_host/rest_port)
# ----------------------------------------------------------------------


def test_load_aliases_listen_and_port_to_rest_host_rest_port(tmp_path):
    """v0.1 [global].listen and .port (which configured the removed QUIC
    server) map onto the v0.2 telemetry rest_host/rest_port."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        'listen = "0.0.0.0"\n'
        'port = 8087\n'
    )
    cfg = load_miner_config(p)
    assert cfg["rest_host"] == "0.0.0.0"
    assert cfg["rest_port"] == 8087
    assert "listen" not in cfg
    assert "port" not in cfg


def test_load_canonical_wins_when_both_alias_and_canonical_present(tmp_path):
    """If both `listen` and `rest_host` are set, the canonical key wins
    silently — operators can stage v0.2 overrides in a v0.1-shaped file
    without breaking the file for v0.1 readers."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        'listen = "0.0.0.0"\n'
        'rest_host = "127.0.0.1"\n'
        'port = 8087\n'
        'rest_port = 9999\n'
    )
    cfg = load_miner_config(p)
    assert cfg["rest_host"] == "127.0.0.1"
    assert cfg["rest_port"] == 9999
    assert "listen" not in cfg
    assert "port" not in cfg


def test_load_only_alias_present_no_canonical(tmp_path):
    """Alias alone (no canonical) — should still get promoted."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        'port = 8086\n'
    )
    cfg = load_miner_config(p)
    assert cfg["rest_port"] == 8086
    assert "rest_host" not in cfg


# ----------------------------------------------------------------------
# [miner] mempool / mempool_min_reward validation
# ----------------------------------------------------------------------


def test_load_mempool_true_and_false_accepted(tmp_path):
    for literal, expected in (("true", True), ("false", False)):
        p = tmp_path / f"mempool-{literal}.toml"
        p.write_text(
            f'[miner]\nvalidators = ["ws://a:9944"]\nmempool = {literal}\n'
        )
        assert load_miner_config(p)["mempool"] is expected


def test_load_mempool_string_false_rejected(tmp_path):
    """`mempool = "false"` is a truthy TOML string — reject loudly instead
    of silently enabling mempool."""
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\nmempool = "false"\n')
    with pytest.raises(MinerConfigError, match=r"mempool.*boolean"):
        load_miner_config(p)


def test_load_mempool_int_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\nmempool = 1\n')
    with pytest.raises(MinerConfigError, match=r"mempool.*boolean"):
        load_miner_config(p)


def test_load_mempool_keys_absent_stay_absent(tmp_path):
    """No default is injected at load time — effective-default resolution
    is per backend group and happens downstream."""
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\n')
    cfg = load_miner_config(p)
    assert "mempool" not in cfg
    assert "mempool_min_reward" not in cfg


def test_load_mempool_min_reward_zero_and_positive_accepted(tmp_path):
    for value in (0, 12345):
        p = tmp_path / f"reward-{value}.toml"
        p.write_text(
            f'[miner]\nvalidators = ["ws://a:9944"]\nmempool_min_reward = {value}\n'
        )
        assert load_miner_config(p)["mempool_min_reward"] == value


def test_load_mempool_min_reward_negative_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\nmempool_min_reward = -1\n'
    )
    with pytest.raises(MinerConfigError, match="non-negative"):
        load_miner_config(p)


def test_load_mempool_min_reward_bool_rejected(tmp_path):
    # bool is an int subclass; reject so `mempool_min_reward = true` fails loud.
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\nmempool_min_reward = true\n'
    )
    with pytest.raises(MinerConfigError, match="integer"):
        load_miner_config(p)


def test_load_mempool_min_reward_string_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\nmempool_min_reward = "5"\n'
    )
    with pytest.raises(MinerConfigError, match="integer"):
        load_miner_config(p)


# ----------------------------------------------------------------------
# load_backend_config — v0.1-shape hardware inventory tables
# ----------------------------------------------------------------------


def test_load_backend_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(MinerConfigError, match=str(missing)):
        load_backend_config(missing)


def test_load_backend_no_sections_returns_empty(tmp_path):
    """A TOML with only `[miner]` (no backend tables) yields {}."""
    p = tmp_path / "miner-only.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\n')
    assert load_backend_config(p) == {}


def test_load_backend_cpu_section(tmp_path):
    p = tmp_path / "cpu.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        '[cpu]\n'
        'num_cpus = 4\n'
    )
    backends = load_backend_config(p)
    assert backends == {"cpu": {"num_cpus": 4}}


def test_load_backend_gpu_defaults_plus_cuda_devices(tmp_path):
    """`[gpu]` defaults + per-device `[cuda.N]` is the v0.1 layout
    `_normalize_gpu_config` expects."""
    p = tmp_path / "gpu.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[gpu]\n'
        'utilization = 80\n'
        'yielding = true\n'
        'sms_per_nonce = 4\n'
        '[cuda.0]\n'
        '[cuda.1]\n'
        'utilization = 50\n'
    )
    backends = load_backend_config(p)
    assert backends["gpu"] == {"utilization": 80, "yielding": True, "sms_per_nonce": 4}
    # tomllib gives `[cuda.0]` / `[cuda.1]` as nested tables under "cuda";
    # `_normalize_gpu_config` handles dict-of-dicts plus arrays-of-tables.
    assert "0" in backends["cuda"]
    assert backends["cuda"]["1"] == {"utilization": 50}


def test_load_backend_metal_and_modal(tmp_path):
    p = tmp_path / "mac-cloud.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[metal]\n'
        'utilization = 100\n'
        '[modal]\n'
        'gpu_type = "a10g"\n'
    )
    backends = load_backend_config(p)
    assert backends["metal"] == {"utilization": 100}
    assert backends["modal"] == {"gpu_type": "a10g"}


def test_load_backend_dwave_with_all_fields(tmp_path):
    """All v0.1 D-Wave keys round-trip — solver/daily_budget/region/EMA tuning."""
    p = tmp_path / "qpu.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'solver = "Advantage2_system1"\n'
        'dwave_region_url = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"\n'
        'qpu_min_blocks_for_estimation = 5\n'
        'qpu_ema_alpha = 0.3\n'
    )
    backends = load_backend_config(p)
    assert backends["dwave"] == {
        "daily_budget": "60s",
        "solver": "Advantage2_system1",
        "dwave_region_url": "https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        "qpu_min_blocks_for_estimation": 5,
        "qpu_ema_alpha": 0.3,
    }


def test_load_backend_all_qpu_vendors_recognised(tmp_path):
    """ibm/braket/pasqal/ionq/origin must all be parsed — they each carry
    a vendor token. Missing any one breaks `register-solver --miner-type`
    flows that need to see the inventory."""
    p = tmp_path / "every-vendor.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[ibm]\ntoken = "ibm-tok"\n'
        '[braket]\ntoken = "braket-tok"\n'
        '[pasqal]\ntoken = "pasqal-tok"\n'
        '[ionq]\ntoken = "ionq-tok"\n'
        '[origin]\ntoken = "origin-tok"\n'
    )
    backends = load_backend_config(p)
    for vendor in ("ibm", "braket", "pasqal", "ionq", "origin"):
        assert backends[vendor]["token"] == f"{vendor}-tok"


def test_load_backend_ignores_miner_section(tmp_path):
    """`[miner]` is consumed by load_miner_config; load_backend_config
    must not double-include it (otherwise CLI conflict detection would
    misfire on every config file)."""
    p = tmp_path / "mixed.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\nsigner_key = "/tmp/k.json"\n'
        '[cpu]\nnum_cpus = 2\n'
    )
    backends = load_backend_config(p)
    assert "miner" not in backends
    assert "cpu" in backends


def test_load_backend_ignores_unknown_top_level_tables(tmp_path):
    """Forward-compat: unknown top-level tables (e.g. `[experimental]`)
    are silently dropped rather than rejected — TOML is the operator's
    file, we just don't claim to understand keys we don't wire up."""
    p = tmp_path / "future.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 1\n'
        '[experimental]\nbeta = true\n'
    )
    backends = load_backend_config(p)
    assert set(backends.keys()) == {"cpu"}


def test_load_backend_rejects_non_table_section(tmp_path):
    """`[cpu]` must be a table — a scalar value at the section name is
    almost certainly a typo and we want to surface it early. Scalar
    needs to be at the top level (before any `[table]`) to land at the
    document root rather than inside whatever table preceded it."""
    p = tmp_path / "broken.toml"
    p.write_text(
        'cpu = 4\n'
        '[miner]\nvalidators = ["ws://a:9944"]\n'
    )
    with pytest.raises(MinerConfigError, match=r"\[cpu\]"):
        load_backend_config(p)


def test_load_backend_array_of_tables(tmp_path):
    """`[[cuda]]` array-of-tables form is the substrate-CLI legacy path
    `_normalize_gpu_config` accepts. Must round-trip."""
    p = tmp_path / "aot.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[[cuda]]\ndevice = "0"\n'
        '[[cuda]]\ndevice = "1"\nutilization = 70\n'
    )
    backends = load_backend_config(p)
    assert backends["cuda"] == [{"device": "0"}, {"device": "1", "utilization": 70}]


# ----------------------------------------------------------------------
# present_backend_groups — used by CLI conflict detection
# ----------------------------------------------------------------------


def test_present_backend_groups_empty():
    assert present_backend_groups({}) == {"cpu": [], "gpu": [], "qpu": []}


def test_present_backend_groups_mixed():
    backends = {"cpu": {}, "cuda": {}, "metal": {}, "dwave": {}, "ibm": {}}
    groups = present_backend_groups(backends)
    assert groups["cpu"] == ["cpu"]
    # Order matches the canonical group tuple, not insertion order.
    assert groups["gpu"] == ["cuda", "metal"]
    assert groups["qpu"] == ["dwave", "ibm"]


def test_backend_section_groupings_dont_overlap():
    """Each section name belongs to exactly one group — the conflict
    detector relies on this disjointness to attribute a TOML section
    to the right CLI subcommand."""
    cpu, gpu, qpu = set(CPU_BACKEND_SECTIONS), set(GPU_BACKEND_SECTIONS), set(QPU_BACKEND_SECTIONS)
    assert cpu & gpu == set()
    assert cpu & qpu == set()
    assert gpu & qpu == set()
    assert cpu | gpu | qpu == set(ALL_BACKEND_SECTIONS)


def test_load_passes_through_identification_keys(tmp_path):
    """node_name, public_host, public_port, log_level, node_log are
    loaded verbatim — the CLI threads them into auto-identify and
    setup_logging at startup."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944"]\n'
        'node_name = "rig-01"\n'
        'public_host = "miner.example.com"\n'
        'public_port = 8086\n'
        'log_level = "DEBUG"\n'
        'node_log = "/var/log/quip-miner.log"\n'
    )
    cfg = load_miner_config(p)
    assert cfg["node_name"] == "rig-01"
    assert cfg["public_host"] == "miner.example.com"
    assert cfg["public_port"] == 8086
    assert cfg["log_level"] == "DEBUG"
    assert cfg["node_log"] == "/var/log/quip-miner.log"


# ----------------------------------------------------------------------
# resolve_mode / resolve_modes — Docker entrypoint dispatch
# ----------------------------------------------------------------------

from shared.miner_config import (  # noqa: E402
    MODE_NAMES,
    ModeResolutionError,
    resolve_mode,
    resolve_modes,
)


def test_resolve_modes_cpu_only_section():
    assert resolve_modes({"cpu": {"num_cpus": 4}}) == ["cpu"]


def test_resolve_modes_dwave_section_resolves_to_qpu():
    assert resolve_modes({"dwave": {"daily_budget": "60s"}}) == ["qpu"]


@pytest.mark.parametrize(
    "backends",
    [
        {"cuda": {"0": {}}},
        {"metal": {}},
        {"modal": {"gpu_type": "a10g"}},
    ],
)
def test_resolve_modes_gpu_section_resolves_to_gpu(backends):
    """Each GPU-group section (cuda/metal/modal) resolves to the gpu mode."""
    assert resolve_modes(backends) == ["gpu"]


def test_resolve_modes_multi_group_returns_canonical_order():
    """`[cpu]` + `[dwave]` → two children (cpu before qpu by canonical
    MODE_NAMES order). The supervisor uses this order to allocate
    --rest-port slots deterministically across restarts."""
    backends = {"cpu": {}, "dwave": {}}
    assert resolve_modes(backends) == ["cpu", "qpu"]


def test_resolve_modes_all_three_groups_returns_all_three():
    backends = {"cpu": {}, "cuda": {"0": {}}, "dwave": {}}
    assert resolve_modes(backends) == ["cpu", "gpu", "qpu"]


def test_resolve_modes_empty_with_default_returns_default():
    assert resolve_modes({}, default="cpu") == ["cpu"]


def test_resolve_modes_empty_no_default_raises():
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes({})
    assert excinfo.value.code == "no-mode-resolvable"


def test_resolve_modes_unsupported_mode_raises():
    """`[cuda.0]` in a config with image_supports=cpu,qpu → error."""
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes({"cuda": {"0": {}}}, image_supports=["cpu", "qpu"])
    assert excinfo.value.code == "unsupported-mode"


def test_resolve_modes_partial_unsupported_in_multi_group_raises():
    """`[cpu]` + `[cuda.0]` in a cpu-only image: cpu is supported but
    cuda isn't — the whole resolution must fail rather than silently
    drop the unsupported group."""
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes(
            {"cpu": {}, "cuda": {"0": {}}},
            image_supports=["cpu", "qpu"],
        )
    assert excinfo.value.code == "unsupported-mode"
    # Error must name the offending section so the operator knows what to fix.
    assert "cuda" in str(excinfo.value)


def test_resolve_modes_default_unsupported_raises():
    """If config is empty and the operator-supplied default isn't in
    image-supports, that's an operator misconfiguration — error."""
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes({}, default="gpu", image_supports=["cpu", "qpu"])
    assert excinfo.value.code == "unsupported-mode"


def test_resolve_modes_bad_default_raises():
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes({}, default="tpu")
    assert excinfo.value.code == "bad-default"


def test_resolve_modes_bad_image_supports_raises():
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_modes({"cpu": {}}, image_supports=["cpu", "tpu"])
    assert excinfo.value.code == "bad-image-supports"


def test_resolve_mode_single_mode_passes_through():
    """resolve_mode is the convenience wrapper for callers that only
    handle one mode — single-group config returns the single mode."""
    assert resolve_mode({"cpu": {"num_cpus": 4}}) == "cpu"
    assert resolve_mode({"dwave": {}}) == "qpu"


def test_resolve_mode_multi_group_raises_for_single_caller():
    """resolve_mode rejects multi-group configs so the (rare) single-mode
    callers fail loudly instead of silently dropping a backend."""
    with pytest.raises(ModeResolutionError) as excinfo:
        resolve_mode({"cpu": {}, "dwave": {}})
    assert excinfo.value.code == "multi-backend-not-single-mode"


def test_resolve_mode_empty_with_default():
    assert resolve_mode({}, default="qpu") == "qpu"


def test_mode_names_matches_expected_subcommands():
    """Pin the canonical ordering — the entrypoint allocates --rest-port
    slots by index, so reordering MODE_NAMES would silently change which
    mode binds which port across container restarts."""
    assert MODE_NAMES == ("cpu", "gpu", "qpu")


# ----------------------------------------------------------------------
# Multi-backend acceptance + mine_mode removal (T8)
# ----------------------------------------------------------------------


def test_resolve_modes_multi_backend_unconditional():
    """Multi-backend configs resolve to every active group with no
    mempool guard — the one-solver-type-per-account constraint is
    handled by the config-derived owner election (`mempool_owner_group`;
    non-owner children resolve mempool off from the same TOML), not by
    mode resolution."""
    assert resolve_modes({"cpu": {}, "dwave": {}}) == ["cpu", "qpu"]
    backends = {"cpu": {}, "cuda": {"0": {}}, "dwave": {}}
    assert resolve_modes(backends) == ["cpu", "gpu", "qpu"]


def test_mempool_owner_group_is_first_non_qpu_configured():
    """The mempool owner is a pure function of the config's backend
    sections: first non-qpu group in canonical cpu,gpu,qpu order."""
    assert mempool_owner_group({"cpu": {}, "cuda": {"0": {}}}) == "cpu"
    assert mempool_owner_group({"cuda": {"0": {}}, "dwave": {}}) == "gpu"
    assert mempool_owner_group({"cpu": {}}) == "cpu"
    assert mempool_owner_group({"metal": {}}) == "gpu"


def test_mempool_owner_group_none_for_qpu_only_or_empty():
    """qpu-only configs elect nobody (per-kind default applies: opt-in
    via explicit `mempool = true`); empty backends likewise."""
    assert mempool_owner_group({"dwave": {}}) is None
    assert mempool_owner_group({"ibm": {}, "dwave": {}}) is None
    assert mempool_owner_group({}) is None


def test_resolve_modes_rejects_mine_mode_kwarg():
    """The mine_mode parameter is gone with the [miner] mode key —
    passing it is a caller bug, surfaced as a TypeError."""
    with pytest.raises(TypeError):
        resolve_modes({"cpu": {}}, mine_mode="mempool")


def test_resolve_mode_rejects_mine_mode_kwarg():
    """The singular wrapper dropped mine_mode along with the plural."""
    with pytest.raises(TypeError):
        resolve_mode({"cpu": {}}, mine_mode="pow")


def test_load_tolerates_stale_mode_key(tmp_path):
    """Legacy configs may still carry the removed [miner] `mode` key;
    the loader passes unknown keys through and nothing reads it."""
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\nmode = "mempool"\n')
    cfg = load_miner_config(p)
    assert cfg["validators"] == ["ws://a:9944"]


# ----------------------------------------------------------------------
# load_submission_config — [submission] table
# ----------------------------------------------------------------------


def test_submission_defaults_when_section_absent(tmp_path):
    """No [submission] table -> default tip=0, retries=3, backoff=250."""
    p = tmp_path / "config.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\n')
    cfg = load_submission_config(p)
    assert cfg == SubmissionConfig(
        tip_plancks=0, max_retries=3, retry_backoff_ms=250
    )


def test_submission_parses_all_keys(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[submission]\n"
        "tip_plancks = 5000\n"
        "max_retries = 7\n"
        "retry_backoff_ms = 100\n"
    )
    cfg = load_submission_config(p)
    assert cfg.tip_plancks == 5000
    assert cfg.max_retries == 7
    assert cfg.retry_backoff_ms == 100


def test_submission_partial_section_keeps_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[submission]\ntip_plancks = 42\n")
    cfg = load_submission_config(p)
    assert cfg.tip_plancks == 42
    assert cfg.max_retries == 3
    assert cfg.retry_backoff_ms == 250


def test_submission_rejects_negative_tip(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[submission]\ntip_plancks = -1\n")
    with pytest.raises(MinerConfigError, match="non-negative"):
        load_submission_config(p)


def test_submission_rejects_non_int(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[submission]\nmax_retries = "lots"\n')
    with pytest.raises(MinerConfigError, match="integer"):
        load_submission_config(p)


def test_submission_accepts_tip_at_u128_max(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(f"[submission]\ntip_plancks = {(1 << 128) - 1}\n")
    cfg = load_submission_config(p)
    assert cfg.tip_plancks == (1 << 128) - 1


def test_submission_rejects_tip_above_u128_max(tmp_path):
    # A tip above the chain's u128 Balance range encodes locally but the
    # runtime rejects it — fail at load time, not at the first proof.
    p = tmp_path / "config.toml"
    p.write_text(f"[submission]\ntip_plancks = {1 << 128}\n")
    with pytest.raises(MinerConfigError, match="u128 max"):
        load_submission_config(p)


def test_submission_rejects_bool_for_int_field(tmp_path):
    # bool is an int subclass; reject so `tip_plancks = true` fails loud.
    p = tmp_path / "config.toml"
    p.write_text("[submission]\ntip_plancks = true\n")
    with pytest.raises(MinerConfigError, match="integer"):
        load_submission_config(p)


def test_submission_rejects_non_table(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('submission = "oops"\n')
    with pytest.raises(MinerConfigError, match="must be a table"):
        load_submission_config(p)


def test_example_toml_submission_section_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "quip.network.qpu.example.toml"
    cfg = load_submission_config(example)
    # Pins the shipped defaults so a stray edit to the example file fails CI.
    assert cfg == SubmissionConfig(
        tip_plancks=0, max_retries=3, retry_backoff_ms=250
    )
