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


# ── TOML round-trip integration ──────────────────────────────────────────
#
# Verify the loader output (`load_backend_config`) is consumable by the
# spec-builder helpers `MinerCore._initialize_miners` uses. These tests
# skip the full MinerCore lifecycle for GPU/QPU because spawning real
# CUDA / Metal / D-Wave worker processes requires hardware + SDKs that
# don't ship with the test harness. Instead they pin the *spec shape*,
# which is the part operators care about.


def test_toml_cpu_round_trip_through_miner_core(tmp_path):
    """End-to-end: TOML → load_backend_config → MinerCore → 4 CPU handles."""
    from shared.miner_config import load_backend_config

    p = tmp_path / "cpu.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 4\n'
    )
    backends = load_backend_config(p)
    core = MinerCore(node_id="toml-cpu", miners_config=backends)
    try:
        assert len(core.miner_handles) == 4
        assert all(h.miner_type == "CPU" for h in core.miner_handles)
    finally:
        core.close()


def test_toml_gpu_devices_produce_correct_spec_shape(tmp_path):
    """`[gpu]` defaults inherited by `[cuda.0]` + `[cuda.1]` per-device
    sections — covers the v0.1 layout `_normalize_gpu_config` handles."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_gpu_specs

    p = tmp_path / "gpu.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[gpu]\nutilization = 80\nsms_per_nonce = 4\n'
        '[cuda.0]\n'
        '[cuda.1]\nutilization = 50\n'
    )
    backends = load_backend_config(p)
    specs = _build_gpu_specs("rig", backends)
    assert len(specs) == 2
    assert {s["id"] for s in specs} == {"rig-GPU-CUDA-0", "rig-GPU-CUDA-1"}
    # Device 0 inherits `[gpu]` defaults verbatim.
    dev0 = next(s for s in specs if s["id"] == "rig-GPU-CUDA-0")
    assert dev0["cfg"]["utilization"] == 80
    assert dev0["cfg"]["sms_per_nonce"] == 4
    # Device 1 overrides `utilization` but inherits `sms_per_nonce`.
    dev1 = next(s for s in specs if s["id"] == "rig-GPU-CUDA-1")
    assert dev1["cfg"]["utilization"] == 50
    assert dev1["cfg"]["sms_per_nonce"] == 4


def test_toml_metal_section_produces_metal_spec(tmp_path):
    """A bare `[metal]` table → single Metal MPS spec. v0.1 schema for
    Apple Silicon rigs."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_gpu_specs

    p = tmp_path / "metal.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[metal]\nutilization = 100\n'
    )
    backends = load_backend_config(p)
    specs = _build_gpu_specs("mac-rig", backends)
    assert len(specs) == 1
    assert specs[0]["kind"] == "metal"
    assert specs[0]["id"] == "mac-rig-GPU-MPS"
    assert specs[0]["cfg"]["utilization"] == 100


def test_toml_modal_section_with_gpu_type_picks_correct_class(tmp_path):
    """`[modal] gpu_type = "a10g"` propagates to spec args."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_gpu_specs

    p = tmp_path / "modal.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[modal]\ngpu_type = "a10g"\n'
    )
    backends = load_backend_config(p)
    specs = _build_gpu_specs("cloud-rig", backends)
    assert len(specs) == 1
    assert specs[0]["kind"] == "modal"
    assert specs[0]["args"]["gpu_type"] == "a10g"
    assert specs[0]["id"] == "cloud-rig-GPU-MODAL-a10g"


def test_toml_dwave_section_round_trips_all_keys(tmp_path):
    """Full D-Wave key set survives the loader → spec-builder boundary.
    daily_budget / solver / qpu_min_blocks_for_estimation / qpu_ema_alpha
    must all land in the QPU sampler's cfg block — the indexer surfaces
    these in the system descriptor."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "dwave.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'solver = "Advantage2_system1"\n'
        'qpu_min_blocks_for_estimation = 7\n'
        'qpu_ema_alpha = 0.25\n'
    )
    backends = load_backend_config(p)
    specs = _build_qpu_specs("qpu-rig", backends)
    assert len(specs) == 1
    cfg = specs[0]["cfg"]
    assert cfg["qpu_type"] == "dwave"
    assert cfg["daily_budget"] == "60s"
    assert cfg["solver"] == "Advantage2_system1"
    assert cfg["qpu_min_blocks_for_estimation"] == 7
    assert cfg["qpu_ema_alpha"] == 0.25


def test_toml_ibm_section_produces_ibm_spec_with_token(tmp_path):
    """`[ibm]` token = "..." round-trips through the QPU spec builder."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "ibm.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[ibm]\ntoken = "ibm-secret-xyz"\ndaily_budget = "5m"\n'
    )
    backends = load_backend_config(p)
    specs = _build_qpu_specs("ibm-rig", backends)
    assert len(specs) == 1
    assert specs[0]["cfg"] == {
        "qpu_type": "ibm",
        "token": "ibm-secret-xyz",
        "daily_budget": "5m",
    }
