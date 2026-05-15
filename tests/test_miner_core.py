"""Unit tests for `shared.miner_core.MinerCore`.

These don't need a substrate chain. They cover MinerCore lifecycle
(handle construction + teardown), descriptor caching + resilience, and the
get_stats / record_dispatch / record_result aggregate counters that back
the preserved `/api/v1/stats` telemetry endpoint.
"""

from __future__ import annotations

import time

import pytest

from shared.miner_core import MinerCore


def test_miner_core_builds_no_handles_for_empty_config():
    core = MinerCore(node_id="empty", miners_config={})
    try:
        assert core.miner_handles == []
    finally:
        core.close()


def test_miner_core_builds_cpu_handles():
    core = MinerCore(node_id="cpu-test", miners_config={"cpu": {"num_cpus": 2}})
    try:
        assert len(core.miner_handles) == 2
        assert all(h.miner_type == "CPU" for h in core.miner_handles)
        assert {h.miner_id for h in core.miner_handles} == {
            "cpu-test-CPU-1",
            "cpu-test-CPU-2",
        }
    finally:
        core.close()


def test_descriptor_uses_builder_callback():
    """Custom descriptor builders are surfaced verbatim through descriptor()."""

    def builder():
        return {"node_id": "custom", "cpus": 8, "gpu": "metal"}

    core = MinerCore(node_id="d-test", miners_config={}, descriptor_builder=builder)
    try:
        desc = core.descriptor()
        assert desc["cpus"] == 8
        # Cached on second call (same object identity).
        assert core.descriptor() is desc
    finally:
        core.close()


def test_descriptor_falls_back_on_builder_failure():
    """If the descriptor builder raises, MinerCore surfaces the error in the
    cached descriptor rather than letting it propagate. Mirrors the
    resilience added to `Node.descriptor()` in commit 1859a35."""

    def bad_builder():
        raise RuntimeError("hardware probe failed")

    core = MinerCore(
        node_id="d-bad",
        miners_config={},
        descriptor_builder=bad_builder,
    )
    try:
        desc = core.descriptor()
        assert desc["node_id"] == "d-bad"
        assert "hardware probe failed" in desc["error"]
    finally:
        core.close()


def test_descriptor_default_when_no_builder():
    core = MinerCore(node_id="d-default", miners_config={"cpu": {"num_cpus": 1}})
    try:
        desc = core.descriptor()
        assert desc["node_id"] == "d-default"
        assert any(m["type"] == "CPU" for m in desc["miners"])
    finally:
        core.close()


def test_record_dispatch_and_result_aggregate_stats():
    core = MinerCore(node_id="stats-test", miners_config={})
    try:
        for _ in range(5):
            core.record_dispatch()
        core.record_result(winning_miner_id="stats-test-CPU-1", mining_time=1.5)
        core.record_result(winning_miner_id="stats-test-CPU-1", mining_time=2.5)
        stats = core.get_stats()
        assert stats["total_blocks_attempted"] == 5
        assert stats["total_blocks_won"] == 2
        assert stats["win_rate"] == 0.4
        assert stats["total_mining_time"] == 4.0
        assert stats["wins_per_miner"] == {"stats-test-CPU-1": 2}
    finally:
        core.close()


def test_get_stats_handles_zero_attempts():
    """Avoid ZeroDivisionError when no attempts have been recorded yet."""
    core = MinerCore(node_id="zero-test", miners_config={})
    try:
        stats = core.get_stats()
        assert stats["total_blocks_attempted"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_mining_time"] == 0.0
    finally:
        core.close()


def test_close_is_idempotent():
    core = MinerCore(node_id="close-test", miners_config={"cpu": {"num_cpus": 1}})
    core.close()
    # Second close on an already-shut-down core is a no-op.
    core.close()
    assert core.miner_handles == []


@pytest.mark.timeout(60)
def test_handles_terminate_on_close():
    """After close(), every worker process is dead (or terminating)."""
    core = MinerCore(node_id="term-test", miners_config={"cpu": {"num_cpus": 2}})
    procs = [h.proc for h in core.miner_handles]
    assert all(p.is_alive() for p in procs)
    core.close()
    # Give a moment for terminate to land.
    deadline = time.time() + 5.0
    while time.time() < deadline and any(p.is_alive() for p in procs):
        time.sleep(0.1)
    for p in procs:
        assert not p.is_alive(), f"worker {p.pid} did not terminate"
