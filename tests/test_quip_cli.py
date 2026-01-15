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
listen = "127.0.0.1"
port = 8123
peers = ["localhost:8000"]
auto_mine = true

[cpu]
num_cpus = 2
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "cpu"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "cpu" in config
    # Config flattens global section into top-level
    assert config.get("listen") == "127.0.0.1"
    assert config.get("port") == 8123
    assert config["cpu"]["num_cpus"] == 2


def test_gpu_device_cli_overrides_toml(tmp_path, monkeypatch):
    # Arrange: config with devices array but CLI provides --device
    cfg = """
[global]
port = 9001

[gpu]
devices = ["1", "2"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act: explicitly pass device 0
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu", "--device", "0"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "gpu" in config
    # Config flattens global section into top-level
    assert config.get("port") == 9001
    # Device is stored in devices list
    assert config["gpu"].get("devices") == ["0"]


def test_gpu_device_from_toml_when_none(tmp_path, monkeypatch):
    # Arrange: devices list in TOML; no CLI device
    cfg = """
[gpu]
devices = ["3", "4"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "gpu" in config
    # Devices should be preserved from TOML
    assert config["gpu"].get("devices") == ["3", "4"]


def test_qpu_env_from_toml_and_defaults(tmp_path, monkeypatch):
    # Arrange: explicit key and solver; omit region to use default
    cfg = """
[qpu]
dwave_api_key = "TOKEN123"
dwave_api_solver = "Advantage_system6.4"
# dwave_region_url omitted to trigger default
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "qpu"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "qpu" in config
    assert config["qpu"].get("dwave_api_key") == "TOKEN123"
    assert config["qpu"].get("dwave_api_solver") == "Advantage_system6.4"


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
[gpu]
backend = "modal"
types = ["t4", "a10g"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "gpu" in config
    assert config["gpu"].get("backend") == "modal"
    assert config["gpu"].get("types") == ["t4", "a10g"]



def test_gpu_backend_cli_override(tmp_path, monkeypatch):
    # Arrange: TOML says local, CLI overrides to modal
    cfg = """
[gpu]
backend = "local"
types = ["t4"]
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    # Act: override via CLI
    runner = CliRunner()
    result = runner.invoke(quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu", "--gpu-backend", "modal"])

    # Assert
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config["gpu"].get("backend") == "modal"

