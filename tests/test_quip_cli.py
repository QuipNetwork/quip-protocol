from __future__ import annotations

import os
from typing import Any, Dict, Optional

import quip_cli
from click.testing import CliRunner


def write_toml(tmp_path, content: str):
    p = tmp_path / "cfg.toml"
    p.write_text(content)
    return p


def test_global_config_and_cpu_num_cpus(tmp_path, monkeypatch):
    # Arrange: config with [global] and [cpu].num_cpus
    cfg = """
[global]
default = "cpu"
host = "127.0.0.1"
port = 8123
peer = "localhost:8000"
auto_mine = 3

[cpu]
num_cpus = 2
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update(
            {
                "kind": kind,
                "host": host,
                "port": port,
                "peer": peer,
                "auto_mine": auto_mine,
                "env": env_overrides or {},
            }
        )
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path)])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "cpu"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["peer"] == "localhost:8000"
    assert captured["auto_mine"] == 3
    # num_cpus should propagate to env overrides for BLAS thread caps
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        assert captured["env"][k] == 2


def test_gpu_device_cli_overrides_toml(tmp_path, monkeypatch):
    # Arrange: config with devices array but CLI provides --device
    cfg = """
[global]
default = "gpu"
port = 9001

[gpu]
devices = ["1", "2"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update({"kind": kind, "port": port, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act: explicitly pass device 0
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu", "--device", "0"])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "gpu"
    assert captured["port"] == 9001
    assert captured["env"].get("CUDA_VISIBLE_DEVICES") == "0"


def test_gpu_device_from_toml_when_none(tmp_path, monkeypatch):
    # Arrange: default=gpu and devices list; no CLI device
    cfg = """
[global]
default = "gpu"

[gpu]
devices = ["3", "4"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update({"kind": kind, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path)])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "gpu"
    # With no --device, we export QUIP_GPU_DEVICES (comma-separated) rather than CUDA_VISIBLE_DEVICES
    assert captured["env"].get("QUIP_GPU_DEVICES") == "3,4"


def test_qpu_env_from_toml_and_defaults(tmp_path, monkeypatch):
    # Arrange: explicit key and solver; omit region to use default
    cfg = """
[global]
default = "qpu"

[qpu]
dwave_api_key = "TOKEN123"
dwave_api_solver = "Advantage_system6.4"
# dwave_region_url omitted to trigger default
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update({"kind": kind, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path)])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "qpu"
    assert captured["env"]["DWAVE_API_TOKEN"] == "TOKEN123"
    assert captured["env"]["DWAVE_API_SOLVER"] == "Advantage_system6.4"
    assert captured["env"]["DWAVE_API_ENDPOINT"] == "https://na-west-1.cloud.dwavesys.com/sapi/v2/"


def test_simulator_print_only_commands(tmp_path):
    runner = CliRunner()
    # CPU-only 2 nodes starting at 9000
    result = runner.invoke(
        quip_cli.quip_network_simulator,
        ["--scenario", "cpu", "--num-cpu", "2", "--base-port", "9000", "--print-only"],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Running: quip-network-node cpu --port 9000" in out
    assert "Running: quip-network-node cpu --port 9001 --peer localhost:9000" in out



def test_gpu_modal_types_env(tmp_path, monkeypatch):
    # Arrange: backend=modal with types
    cfg = """
[global]
default = "gpu"

[gpu]
backend = "modal"
types = ["t4", "a10g"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update({"kind": kind, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path)])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "gpu"
    assert captured["env"].get("QUIP_GPU_BACKEND") == "modal"
    assert captured["env"].get("QUIP_GPU_TYPES") == "t4,a10g"



def test_gpu_backend_cli_override(tmp_path, monkeypatch):
    # Arrange: TOML says local, CLI overrides to modal
    cfg = """
[global]
default = "gpu"

[gpu]
backend = "local"
types = ["t4"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None):
        captured.update({"kind": kind, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    # Act: override via CLI
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu", "--gpu-backend", "modal"])

    # Assert
    assert result.exit_code == 0, result.output
    assert captured["env"].get("QUIP_GPU_BACKEND") == "modal"

