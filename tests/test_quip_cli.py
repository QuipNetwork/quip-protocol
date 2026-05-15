from __future__ import annotations

from typing import Any, Dict

import pytest

import quip_cli
from click.testing import CliRunner
from shared.node import _normalize_gpu_config, _normalize_qpu_config


def write_toml(tmp_path, content: str):
    p = tmp_path / "cfg.toml"
    p.write_text(content)
    return p


def test_global_config_and_cpu_num_cpus(tmp_path, monkeypatch):
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

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node, ["--config", str(cfg_path), "cpu"]
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "cpu" in config
    assert config.get("listen") == "127.0.0.1"
    assert config.get("port") == 8123
    assert config["cpu"]["num_cpus"] == 2


def test_gpu_device_cli_creates_cuda_section(tmp_path, monkeypatch):
    """CLI --device produces top-level [cuda.N] sections."""
    cfg = """
[global]
port = 9001
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node,
        [
            "--config",
            str(cfg_path),
            "gpu",
            "--device",
            "0",
            "--device",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.get("port") == 9001
    assert "cuda" in config
    assert "0" in config["cuda"]
    assert "1" in config["cuda"]


def test_gpu_device_from_toml_new_format(tmp_path, monkeypatch):
    """[cuda.N] sections from TOML are forwarded."""
    cfg = """
[gpu]

[cuda.3]

[cuda.4]
utilization = 50
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node, ["--config", str(cfg_path), "gpu"]
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "cuda" in config
    assert "3" in config["cuda"]
    assert config["cuda"]["4"]["utilization"] == 50


def test_qpu_cli_creates_dwave_section(tmp_path, monkeypatch):
    """CLI --dwave-api-key/--dwave-api-solver produce [dwave] section."""
    cfg_path = write_toml(
        tmp_path,
        """
[global]
port = 9002
""",
    )

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node,
        [
            "--config",
            str(cfg_path),
            "qpu",
            "--dwave-api-key",
            "TOKEN123",
            "--dwave-api-solver",
            "Advantage_system6.4",
            "--qpu-daily-budget",
            "60s",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "dwave" in config
    assert config["dwave"]["token"] == "TOKEN123"
    assert config["dwave"]["solver"] == "Advantage_system6.4"
    assert config["dwave"]["daily_budget"] == "60s"


def test_qpu_from_toml_new_format(tmp_path, monkeypatch):
    """[dwave] section from TOML is forwarded."""
    cfg = """
[qpu]

[dwave]
token = "DW_TOKEN"
solver = "Advantage_system6.4"
daily_budget = "60s"
"""
    cfg_path = write_toml(tmp_path, cfg)

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node, ["--config", str(cfg_path), "qpu"]
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "dwave" in config
    assert config["dwave"]["token"] == "DW_TOKEN"
    assert config["dwave"]["solver"] == "Advantage_system6.4"


def test_simulator_print_only_commands(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_simulator,
        ["--scenario", "cpu", "--num-cpu", "2", "--base-port", "9000", "--print-only"],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    # Simulator pins listen/public-host to 127.0.0.1 and assigns each
    # child a unique REST port so children don't collide on 20050.
    assert (
        "Running: quip-network-node cpu --listen 127.0.0.1 --port 9000 "
        "--public-host 127.0.0.1 --rest-insecure-port 20050"
    ) in out
    assert (
        "Running: quip-network-node cpu --listen 127.0.0.1 --port 9001 "
        "--public-host 127.0.0.1 --rest-insecure-port 20051 "
        "--peer 127.0.0.1:9000"
    ) in out


def test_gpu_mps_backend_creates_metal_section(tmp_path, monkeypatch):
    """CLI --gpu-backend mps produces [metal] section."""
    cfg_path = write_toml(tmp_path, "")

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node,
        [
            "--config",
            str(cfg_path),
            "gpu",
            "--gpu-backend",
            "mps",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "metal" in config


def test_gpu_modal_backend_creates_modal_section(tmp_path, monkeypatch):
    """CLI --gpu-backend modal --gpu-type t4 produces [modal] section."""
    cfg_path = write_toml(tmp_path, "")

    captured: Dict[str, Any] = {}

    def fake_run(
        config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"
    ):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_node,
        [
            "--config",
            str(cfg_path),
            "gpu",
            "--gpu-backend",
            "modal",
            "--gpu-type",
            "t4",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert "modal" in config
    assert config["modal"]["gpu_type"] == "t4"


# ── GPU normalizer tests ──────────────────────────────────────────────


def test_normalize_gpu_cuda_sections():
    """[cuda.0] and [cuda.1] top-level sections → 2 cuda entries."""
    cfg = {
        "gpu": {"utilization": 100},
        "cuda": {
            "0": {},
            "1": {"utilization": 50, "yielding": True},
        },
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 2
    assert devs[0] == {"type": "cuda", "device": "0"}
    assert devs[1]["type"] == "cuda"
    assert devs[1]["device"] == "1"
    assert devs[1]["utilization"] == 50


def test_normalize_gpu_nvidia_alias():
    """[nvidia.0] is interchangeable with [cuda.0]."""
    cfg = {
        "gpu": {},
        "nvidia": {"0": {"yielding": True}},
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["type"] == "cuda"
    assert devs[0]["device"] == "0"
    assert devs[0]["yielding"] is True


def test_normalize_gpu_metal_section():
    """[metal] top-level section → 1 metal entry."""
    cfg = {
        "gpu": {"utilization": 80},
        "metal": {"yielding": True},
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["type"] == "metal"
    assert devs[0]["yielding"] is True


def test_normalize_gpu_metal_array_of_tables():
    """[[metal]] (array of tables) → 1 metal entry."""
    cfg = {
        "gpu": {},
        "metal": [{"utilization": 90}],
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["type"] == "metal"
    assert devs[0]["utilization"] == 90


def test_normalize_gpu_mixed_cuda_and_metal():
    """[cuda.0] + [metal] → cuda + metal entries."""
    cfg = {
        "gpu": {"utilization": 100},
        "cuda": {"0": {}},
        "metal": {},
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    types = [d["type"] for d in devs]
    assert "cuda" in types
    assert "metal" in types


def test_normalize_gpu_enabled_false():
    """Device with enabled=false is present in normalized list."""
    cfg = {
        "cuda": {"0": {"enabled": False}},
    }
    result = _normalize_gpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["enabled"] is False


def test_normalize_gpu_no_devices():
    """[gpu] with no device sections → empty devices list."""
    cfg = {"gpu": {"utilization": 100}}
    result = _normalize_gpu_config(cfg)
    assert "devices" not in result


# ── QPU normalizer tests ──────────────────────────────────────────────


def test_normalize_qpu_dwave_section():
    """[dwave] top-level section → 1 dwave entry."""
    cfg = {
        "qpu": {},
        "dwave": {
            "token": "DW_KEY",
            "solver": "Advantage_system6.4",
            "daily_budget": "60s",
        },
    }
    result = _normalize_qpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["type"] == "dwave"
    assert devs[0]["token"] == "DW_KEY"
    assert devs[0]["daily_budget"] == "60s"


def test_normalize_qpu_token_backend():
    """[ibm] with token → 1 ibm entry."""
    cfg = {
        "qpu": {},
        "ibm": {"token": "IBM_TOK"},
    }
    result = _normalize_qpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 1
    assert devs[0]["type"] == "ibm"
    assert devs[0]["token"] == "IBM_TOK"


def test_normalize_qpu_multiple_backends():
    """[dwave] + [ibm] + [ionq] → 3 entries."""
    cfg = {
        "qpu": {},
        "dwave": {"token": "DW"},
        "ibm": {"token": "IBM"},
        "ionq": {"token": "IONQ"},
    }
    result = _normalize_qpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 3
    types = [d["type"] for d in devs]
    assert "dwave" in types
    assert "ibm" in types
    assert "ionq" in types


def test_normalize_qpu_array_of_tables():
    """[[dwave]] (array of tables) → entries from list."""
    cfg = {
        "dwave": [
            {"token": "KEY1", "solver": "solver_a"},
            {"token": "KEY2", "solver": "solver_b"},
        ],
    }
    result = _normalize_qpu_config(cfg)
    devs = result["devices"]
    assert len(devs) == 2
    assert devs[0]["token"] == "KEY1"
    assert devs[1]["token"] == "KEY2"


def test_normalize_qpu_no_devices():
    """[qpu] with no device sections → empty devices list."""
    cfg = {"qpu": {}}
    result = _normalize_qpu_config(cfg)
    assert "devices" not in result


# ── _merge_globals_from_toml tests ─────────────────────────────────────────


def test_merge_globals_includes_miner_sections():
    """cpu/gpu/qpu sub-sections inside [global] must not be silently dropped."""
    cfg = {
        "global": {
            "listen": "127.0.0.1",
            "cpu": {"num_cpus": 4},
        }
    }
    out = quip_cli._merge_globals_from_toml(cfg)
    assert out.get("listen") == "127.0.0.1"
    assert out.get("cpu") == {"num_cpus": 4}
    assert "global.k" not in out


def test_merge_globals_empty():
    assert quip_cli._merge_globals_from_toml({}) == {}


# ── _parse_topology tests ──────────────────────────────────────────────────


def test_parse_topology_valid_zephyr():
    topology = quip_cli._parse_topology("zephyr:9,2")
    assert topology.num_nodes > 0
    assert topology.num_edges > 0


def test_parse_topology_missing_colon():
    import click

    with pytest.raises(click.BadParameter, match="family:m,t"):
        quip_cli._parse_topology("zephyr92")


def test_parse_topology_unknown_family():
    import click

    with pytest.raises(click.BadParameter, match="zephyr"):
        quip_cli._parse_topology("pegasus:9,2")


def test_parse_topology_bad_params_non_integer():
    import click

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
        [
            "--node-url",
            "ws://localhost:9944",
            "--num-cpus",
            "3",
        ],
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
        [
            "--node-url",
            "ws://localhost:9944",
            "--gpu-backend",
            "local",
        ],
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
        [
            "--node-url",
            "ws://localhost:9944",
            "--gpu-backend",
            "metal",
        ],
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
        [
            "--node-url",
            "ws://localhost:9944",
            "--gpu-backend",
            "modal",
        ],
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
        [
            "--node-url",
            "ws://localhost:9944",
            "--qpu-type",
            "dwave",
            "--daily-budget",
            "40s",
        ],
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
        [
            "--node-url",
            "ws://localhost:9944",
            "--qpu-type",
            "ibm",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = captured.get("miner_config", {})
    assert "ibm" in cfg
    assert cfg["ibm"][0]["type"] == "ibm"
