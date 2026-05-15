"""Unit tests for quip_cli helpers: topology parsing, injection, hashing, and subcommands."""
from __future__ import annotations

from typing import Any, Dict

import click
import pytest
from click.testing import CliRunner

import quip_cli


# ── _parse_topology tests ──────────────────────────────────────────────────


def test_parse_topology_valid_zephyr():
    topology = quip_cli._parse_topology("zephyr:9,2")
    assert topology.num_nodes > 0
    assert topology.num_edges > 0


def test_parse_topology_missing_colon():
    with pytest.raises(click.BadParameter, match="family:m,t"):
        quip_cli._parse_topology("zephyr92")


def test_parse_topology_unknown_family():
    with pytest.raises(click.BadParameter, match="zephyr"):
        quip_cli._parse_topology("pegasus:9,2")


def test_parse_topology_bad_params_non_integer():
    with pytest.raises(click.BadParameter):
        quip_cli._parse_topology("zephyr:abc,2")


# ── _inject_topology tests ─────────────────────────────────────────────────


def test_inject_topology_cpu_adds_to_args():
    sentinel = object()
    config = {"cpu": {"num_cpus": 2}}
    out = quip_cli._inject_topology(config, "cpu", sentinel)
    assert out["cpu"]["args"]["topology"] is sentinel


def test_inject_topology_cpu_preserves_existing_args():
    sentinel = object()
    config = {"cpu": {"num_cpus": 2, "args": {"existing": "value"}}}
    out = quip_cli._inject_topology(config, "cpu", sentinel)
    assert out["cpu"]["args"]["topology"] is sentinel
    assert out["cpu"]["args"]["existing"] == "value"


def test_inject_topology_does_not_mutate_input():
    sentinel = object()
    config = {"cpu": {"num_cpus": 1}}
    quip_cli._inject_topology(config, "cpu", sentinel)
    assert "args" not in config["cpu"]


def test_inject_topology_gpu_config_unchanged(capsys):
    config = {"cuda": [{"device": "0"}]}
    out = quip_cli._inject_topology(config, "gpu", object())
    assert out.get("cuda") == [{"device": "0"}]
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


# ── _zephyr_topology_hash tests ────────────────────────────────────────────


def test_zephyr_topology_hash_is_deterministic():
    topology = quip_cli._parse_topology("zephyr:2,2")
    h1 = quip_cli._zephyr_topology_hash(topology)
    h2 = quip_cli._zephyr_topology_hash(topology)
    assert h1 == h2
    assert len(h1) == 32


def test_zephyr_topology_hash_differs_across_specs():
    t1 = quip_cli._parse_topology("zephyr:2,2")
    t2 = quip_cli._parse_topology("zephyr:3,2")
    assert quip_cli._zephyr_topology_hash(t1) != quip_cli._zephyr_topology_hash(t2)


# ── quip-miner subcommand tests ────────────────────────────────────────────


def test_quip_miner_cpu_config(monkeypatch):
    """quip-miner cpu --num-cpus 3 wires the right miner_config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_cpu,
        ["--node-url", "ws://localhost:9944", "--num-cpus", "3"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"cpu": {"num_cpus": 3}}
    assert captured.get("kind") == "cpu"


def test_quip_miner_gpu_local_config(monkeypatch):
    """quip-miner gpu --gpu-backend local creates CUDA config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--node-url", "ws://localhost:9944", "--gpu-backend", "local"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"cuda": [{"device": "0"}]}
    assert captured.get("kind") == "gpu"


def test_quip_miner_gpu_metal_config(monkeypatch):
    """quip-miner gpu --gpu-backend metal creates Metal config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--node-url", "ws://localhost:9944", "--gpu-backend", "metal"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"metal": [{}]}


def test_quip_miner_gpu_modal_config(monkeypatch):
    """quip-miner gpu --gpu-backend modal creates Modal config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--node-url", "ws://localhost:9944", "--gpu-backend", "modal"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"modal": [{"gpu_type": "t4"}]}


def test_quip_miner_qpu_dwave_with_budget(monkeypatch):
    """quip-miner qpu --qpu-type dwave --daily-budget 40s creates DWave config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_qpu,
        ["--node-url", "ws://localhost:9944", "--qpu-type", "dwave", "--daily-budget", "40s"],
    )
    assert result.exit_code == 0, result.output
    cfg = captured.get("miner_config", {})
    assert "dwave" in cfg
    assert cfg["dwave"][0]["type"] == "dwave"
    assert cfg["dwave"][0]["daily_budget"] == "40s"


def test_quip_miner_qpu_ibm_config(monkeypatch):
    """quip-miner qpu --qpu-type ibm creates IBM QPU config."""
    captured: Dict[str, Any] = {}

    async def fake_run_miner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_miner", fake_run_miner)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_qpu,
        ["--node-url", "ws://localhost:9944", "--qpu-type", "ibm"],
    )
    assert result.exit_code == 0, result.output
    cfg = captured.get("miner_config", {})
    assert "ibm" in cfg
    assert cfg["ibm"][0]["type"] == "ibm"
