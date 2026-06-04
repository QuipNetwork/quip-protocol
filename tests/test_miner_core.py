"""Unit tests for `shared.miner_core.MinerCore`.

These don't need a substrate chain. They cover MinerCore lifecycle
(handle construction + teardown), descriptor caching + resilience, and the
get_stats / record_dispatch / record_result aggregate counters that back
the preserved `/api/v1/stats` telemetry endpoint.
"""

from __future__ import annotations

import time

import pytest

from dwave_topologies import DEFAULT_TOPOLOGY

from shared.miner_core import MinerCore


def test_miner_core_builds_no_handles_for_empty_config():
    core = MinerCore(node_id="empty", miners_config={})
    try:
        assert core.miner_handles == []
    finally:
        core.close()


def test_minercore_injects_topology_into_every_handle_spec():
    core = MinerCore(
        node_id="topo-cpu", miners_config={"cpu": {"num_cpus": 2}},
        topology=DEFAULT_TOPOLOGY,
    )
    try:
        assert len(core.miner_handles) == 2
        for h in core.miner_handles:
            assert h.spec["args"]["topology"] is DEFAULT_TOPOLOGY
    finally:
        core.close()


def test_minercore_requires_topology_when_building_handles():
    import pytest
    with pytest.raises(ValueError, match="requires a topology"):
        MinerCore(node_id="no-topo", miners_config={"cpu": {"num_cpus": 1}})


def test_minercore_empty_config_needs_no_topology():
    core = MinerCore(node_id="empty-ok", miners_config={})
    try:
        assert core.miner_handles == []
    finally:
        core.close()


def test_miner_core_builds_cpu_handles():
    core = MinerCore(node_id="cpu-test", miners_config={"cpu": {"num_cpus": 2}},
                     topology=DEFAULT_TOPOLOGY)
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
    core = MinerCore(node_id="d-default", miners_config={"cpu": {"num_cpus": 1}},
                     topology=DEFAULT_TOPOLOGY)
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
    core = MinerCore(node_id="close-test", miners_config={"cpu": {"num_cpus": 1}},
                     topology=DEFAULT_TOPOLOGY)
    core.close()
    # Second close on an already-shut-down core is a no-op.
    core.close()
    assert core.miner_handles == []


@pytest.mark.timeout(60)
def test_handles_terminate_on_close():
    """After close(), every worker process is dead (or terminating)."""
    core = MinerCore(node_id="term-test", miners_config={"cpu": {"num_cpus": 2}},
                     topology=DEFAULT_TOPOLOGY)
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
    core = MinerCore(node_id="toml-cpu", miners_config=backends,
                     topology=DEFAULT_TOPOLOGY)
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
    Token + region + solver translate into the SDK-canonical kwargs
    (token / region / solver_name) the DWaveSampler constructor
    expects; budget/tuning keys pass through verbatim for QPUTimeManager."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "dwave.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'solver = "Advantage2_system1"\n'
        'region = "na-west-1"\n'
        'token = "dwave-secret-xyz"\n'
        'qpu_min_blocks_for_estimation = 7\n'
        'qpu_ema_alpha = 0.25\n'
    )
    backends = load_backend_config(p)
    specs = _build_qpu_specs("qpu-rig", backends)
    assert len(specs) == 1
    cfg = specs[0]["cfg"]
    assert cfg["qpu_type"] == "dwave"
    assert cfg["daily_budget"] == "60s"
    # The cfg key matches the operator-facing TOML spelling (`solver`)
    # so the descriptor scrubber's whitelist surfaces it directly.
    # build_miner_from_spec translates to DWaveMiner's `solver_name`
    # kwarg at the constructor boundary.
    assert cfg["solver"] == "Advantage2_system1"
    assert cfg["region"] == "na-west-1"
    # `token` is now passed through — this was the root cause of the
    # "API token not defined" error operators hit when they set
    # `[dwave].token` in TOML expecting it to be honored.
    assert cfg["token"] == "dwave-secret-xyz"
    assert cfg["qpu_min_blocks_for_estimation"] == 7
    assert cfg["qpu_ema_alpha"] == 0.25


def test_toml_dwave_throughput_overrides_round_trip(tmp_path):
    """``[dwave].num_reads`` and ``annealing_time_us`` flow into the spec
    cfg so DWaveMiner.__init__ picks them up. Absent keys do not appear
    so the constructor's default of None drives the
    ``_adapt_mining_params`` fallback to the throughput-tuned hardcoded
    values (112 reads x 80us)."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "throughput.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'num_reads = 128\n'
        'annealing_time_us = 20.0\n'
    )
    backends = load_backend_config(p)
    specs = _build_qpu_specs("rig", backends)
    cfg = specs[0]["cfg"]
    assert cfg["num_reads"] == 128
    assert cfg["annealing_time_us"] == 20.0

    p2 = tmp_path / "no_overrides.toml"
    p2.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    backends2 = load_backend_config(p2)
    cfg2 = _build_qpu_specs("rig", backends2)[0]["cfg"]
    assert "num_reads" not in cfg2
    assert "annealing_time_us" not in cfg2


def test_toml_dwave_budget_overrides_round_trip(tmp_path):
    """``[dwave].min_block_budget`` and ``budget_cap`` flow into the spec cfg
    so the QPU time config (reservoir buffer + pool cap) picks them up. Absent
    keys do not appear, so miner_worker's ``cfg.get("min_block_budget", "90s")``
    default drives the buffer."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs

    p = tmp_path / "budget.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "15m"\n'
        'min_block_budget = "20m"\n'
        'budget_cap = "30m"\n'
    )
    backends = load_backend_config(p)
    cfg = _build_qpu_specs("rig", backends)[0]["cfg"]
    assert cfg["min_block_budget"] == "20m"
    assert cfg["budget_cap"] == "30m"

    p2 = tmp_path / "no_budget_overrides.toml"
    p2.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\ndaily_budget = "15m"\n'
    )
    backends2 = load_backend_config(p2)
    cfg2 = _build_qpu_specs("rig", backends2)[0]["cfg"]
    assert "min_block_budget" not in cfg2
    assert "budget_cap" not in cfg2


def test_toml_dwave_token_does_not_leak_to_descriptor(tmp_path):
    """Defense-in-depth regression: even though the dwave cfg now
    carries `token`, the descriptor pipeline's whitelist
    (`_QPU_HANDLE_FIELD_WHITELIST = {"solver", "daily_budget"}`)
    must still drop it before /api/v1/status canonicalization.
    Catches the case where someone widens the whitelist or routes the
    spec through a non-scrubbing code path."""
    from shared.miner_config import load_backend_config
    from shared.miner_core import _build_qpu_specs
    from shared.system_info import (
        build_descriptor,
        to_canonical_json,
        validate_descriptor,
    )

    p = tmp_path / "leaky.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'token = "dwave-MUST-NOT-LEAK-abc123"\n'
        'daily_budget = "60s"\n'
    )
    backends = load_backend_config(p)
    specs = _build_qpu_specs("rig", backends)
    # In-process: token IS in the spec (the sampler needs it).
    assert specs[0]["cfg"]["token"] == "dwave-MUST-NOT-LEAK-abc123"
    # On-chain: token must NOT appear in the canonical descriptor JSON.
    desc = build_descriptor(
        node_id="rig", node_name="rig",
        miner_specs=specs, include_system_info=False,
    )
    validate_descriptor(desc)
    payload = to_canonical_json(desc)
    assert "dwave-MUST-NOT-LEAK-abc123" not in payload.decode("utf-8")
    assert "token" not in desc.miners["dwave"]


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
