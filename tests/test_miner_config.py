"""Unit tests for `shared.miner_config` — TOML loader + CLI merge."""
from __future__ import annotations

from pathlib import Path

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
