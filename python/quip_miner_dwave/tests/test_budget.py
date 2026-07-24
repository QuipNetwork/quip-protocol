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


def test_get_stats_includes_ema_eta_and_skip_fields():
    mgr = QPUTimeManager(
        QPUTimeConfig(
            daily_budget_seconds=86400.0,
            min_block_budget_seconds=10.0,
            initial_budget_seconds=0.0,
            min_blocks_for_estimation=2,
        )
    )
    t0 = 2_000_000.0
    mgr.reset_clock(t0)

    # Before any recorded access: EMA is unset, ETA falls back to the
    # no-history default, and the pool is empty so a wait is required.
    stats = mgr.get_stats(now=t0)
    assert stats["ema_estimate_seconds"] is None
    assert (
        stats["estimated_block_time_seconds"]
        == mgr.estimate_next_block_time() / 1_000_000
    )
    assert stats["seconds_until_can_mine"] > 0.0
    assert "pool_seconds" in stats
    assert "burst_active" in stats
    assert "daily_budget_seconds" in stats
    assert "min_block_budget_seconds" in stats
    assert "cumulative_used_seconds" in stats

    # Record enough accesses to seed the EMA; get_stats must reuse it, not
    # recompute independently.
    mgr.record_access_time(1_000_000, now=t0)
    mgr.record_access_time(1_500_000, now=t0)
    assert mgr.ema_estimate_us is not None
    stats = mgr.get_stats(now=t0)
    assert stats["ema_estimate_seconds"] == mgr.ema_estimate_us / 1_000_000
    assert stats["cumulative_used_seconds"] == mgr.cumulative_used_us / 1_000_000

    # Once budget accrues past the buffer, seconds_until_can_mine drops to 0
    # and matches should_mine()'s own value.
    est = mgr.should_mine(now=t0 + 20.0)
    stats = mgr.get_stats(now=t0 + 20.0)
    assert stats["seconds_until_can_mine"] == est.seconds_until_can_mine
    assert stats["burst_active"] == est.burst_active
