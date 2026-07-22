"""CLI surface tests."""

from quip_miner_dwave import (
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
)
from quip_miner_dwave.cli import main, run_check


def test_version(capsys):
    assert main(["--version"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "quip-dwave-qa" in out
    assert "protocol 1" in out


def test_capabilities(capsys):
    assert main(["--capabilities"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "dwave-qpu" in out
    assert "quantum-anneal" in out


def test_missing_coordinator():
    assert main([]) == EXIT_CONFIG_INVALID


def test_check_mock_ok(monkeypatch):
    monkeypatch.setenv("QUIP_DWAVE_MOCK", "1")
    assert run_check(force_mock=True) == EXIT_CLEAN


def test_check_no_creds_fails(monkeypatch):
    monkeypatch.delenv("DWAVE_API_KEY", raising=False)
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    monkeypatch.delenv("QUIP_DWAVE_MOCK", raising=False)
    # Home without a dwave.conf token: force credentials_present False
    monkeypatch.setattr(
        "quip_miner_dwave.cli.credentials_present", lambda: False
    )
    monkeypatch.setattr(
        "quip_miner_dwave.cli.ocean_importable", lambda: True
    )
    assert run_check(force_mock=False) == EXIT_ENV_INCOMPATIBLE
