"""Unit tests for `shared.miner_config` — TOML loader + CLI merge."""
from __future__ import annotations

import pytest

from shared.miner_config import (
    MinerConfigError,
    load_miner_config,
    merge_config,
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
    toml = {"validators": ["ws://toml:9944"], "topology": "zephyr:9,2"}
    cli = {"validators": ("ws://cli:9944",), "topology": "zephyr:10,2"}
    merged = merge_config(toml, cli)
    # CLI list materialized to a list — order preserved.
    assert merged["validators"] == ["ws://cli:9944"]
    assert merged["topology"] == "zephyr:10,2"


def test_merge_falls_back_to_toml_when_cli_empty():
    """Empty tuples / None / unset CLI values must NOT clobber TOML."""
    toml = {
        "validators": ["ws://toml:9944"],
        "signer_key": "~/.quip-miner/signing.json",
        "topology": "zephyr:9,2",
    }
    cli = {"validators": (), "signer_key": None, "topology": None}
    merged = merge_config(toml, cli)
    assert merged["validators"] == ["ws://toml:9944"]
    assert merged["signer_key"] == "~/.quip-miner/signing.json"
    assert merged["topology"] == "zephyr:9,2"


def test_merge_unknown_cli_key_passes_through():
    """Extra CLI keys (e.g. num_cpus on the cpu command) get merged in."""
    merged = merge_config({}, {"validators": ("ws://a:9944",), "num_cpus": 4})
    assert merged["num_cpus"] == 4


def test_merge_rest_host_and_port_cli_overrides_toml():
    """`rest_host` / `rest_port` follow the same CLI > TOML precedence."""
    toml = {"rest_host": "127.0.0.1", "rest_port": 8086}
    cli = {"rest_host": "0.0.0.0", "rest_port": 9000}
    merged = merge_config(toml, cli)
    assert merged["rest_host"] == "0.0.0.0"
    assert merged["rest_port"] == 9000


def test_merge_rest_host_and_port_toml_when_cli_unset():
    """A TOML `rest_host = "0.0.0.0"` survives an unset CLI flag (None)."""
    toml = {"rest_host": "0.0.0.0", "rest_port": 8086}
    cli = {"rest_host": None, "rest_port": None}
    merged = merge_config(toml, cli)
    assert merged["rest_host"] == "0.0.0.0"
    assert merged["rest_port"] == 8086


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
