"""Tests for the uniform dwave config-override discipline."""
from __future__ import annotations

import logging

from quip_miner_dwave.budget import DWAVE_CONFIG_KEYS, warn_unknown_backend_keys
from quip_miner_dwave.config import config_override, warn_unknown_fields


def test_config_override_reports_only_on_change(caplog):
    with caplog.at_level(logging.WARNING):
        assert config_override("utilization", 100, 60) == 60  # changed -> config
        assert config_override("utilization", 60, 60) == 60  # same -> silent
        assert config_override("utilization", 100, None) == 100  # absent -> cli
    msgs = [r.message for r in caplog.records]
    assert any("config overrides utilization: 100 -> 60" in m for m in msgs)
    # only the effective change warned
    assert sum("config overrides" in m for m in msgs) == 1


def test_warn_unknown_fields_filters_session_keys(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unknown_fields("dwave", ["daily_budget", "num_sweeps", "typo"], DWAVE_CONFIG_KEYS)
    warned = [r.message for r in caplog.records]
    assert any("unknown field 'typo' for dwave" in m for m in warned)
    # recognized + session keys are not flagged
    assert not any("daily_budget" in m or "num_sweeps" in m for m in warned)


def test_warn_unknown_backend_keys_parses_toml(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unknown_backend_keys('daily_budget = "1h"\nnum_reads = 100\nbogus = 1\n')
    warned = [r.message for r in caplog.records]
    assert any("unknown field 'bogus'" in m for m in warned)
    # recognized keys are not flagged; connection creds live in D-Wave env, so
    # they are NOT recognized here and would warn (intentional).
    assert not any("daily_budget" in m or "num_reads" in m for m in warned)
