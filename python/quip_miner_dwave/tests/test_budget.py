"""Budget reservoir unit tests."""
from quip_miner_dwave.budget import (
    QPUTimeConfig,
    QPUTimeManager,
    budget_from_backend_toml,
    parse_duration,
)


def test_parse_duration():
    assert parse_duration("30s") == 30.0
    assert parse_duration("5m") == 300.0
    assert parse_duration("1h") == 3600.0
    assert parse_duration("2d") == 2 * 86400.0


def test_accumulate_then_burst():
    mgr = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=86400.0,  # 1s of QPU per wall-second
            min_block_budget_seconds=10.0,
            initial_budget_seconds=0.0,
        )
    )
    t0 = 1_000_000.0
    mgr.reset_clock(t0)
    # Dry: no mine
    est = mgr.should_mine(now=t0)
    assert not est.should_mine
    # Accrue 10s wall → 10s QPU budget
    est = mgr.should_mine(now=t0 + 10.0)
    assert est.should_mine
    assert est.burst_active
    # Spend down below zero mid-burst still continues while pool > 0
    mgr.record_access_time(5_000_000, now=t0 + 10.0)  # 5s
    est = mgr.should_mine(now=t0 + 10.0)
    assert est.should_mine
    # Drain to 0 ends burst
    mgr.record_access_time(5_000_000, now=t0 + 10.0)
    est = mgr.should_mine(now=t0 + 10.0)
    assert not est.should_mine


def test_budget_from_toml():
    mgr = budget_from_backend_toml('daily_budget = "30s"\nmin_block_budget = "5s"\n')
    assert mgr is not None
    assert mgr.config.daily_budget_seconds == 30.0
    assert mgr.config.min_block_budget_seconds == 5.0
    assert budget_from_backend_toml("") is None
