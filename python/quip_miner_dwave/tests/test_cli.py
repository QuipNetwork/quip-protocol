"""CLI surface tests."""

import signal

import pytest

from quip_miner_dwave import (
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
)
from quip_miner_dwave.cli import install_sigterm_handler, main, run_check


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


class _FakeSampler:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_sigterm_handler_closes_sampler_and_exits_clean():
    """SIGTERM must close the sampler and request a clean exit code, matching
    the v0.2 ``_cleanup_handler`` behavior (quip-w5p.8b)."""
    original = signal.getsignal(signal.SIGTERM)
    sampler = _FakeSampler()
    try:
        install_sigterm_handler(sampler)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not original
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGTERM, None)
        assert exc_info.value.code == EXIT_CLEAN
        assert sampler.closed
    finally:
        signal.signal(signal.SIGTERM, original)


def test_sigterm_handler_is_idempotent():
    """A second delivery (or re-entrant signal) must not raise/close twice."""
    original = signal.getsignal(signal.SIGTERM)
    sampler = _FakeSampler()
    try:
        install_sigterm_handler(sampler)
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
        # Second delivery: already triggered, must return quietly (no raise).
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, original)
