import os
from click.testing import CliRunner
from tempfile import TemporaryDirectory
from typing import Dict, Any

import quip_cli


def test_cpu_auto_mine_quick(monkeypatch):
    """Test CPU CLI wiring without actually starting the network node."""
    cfg = """
[cpu]
num_cpus = 1
"""
    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)

        # Patch to avoid actually spawning network node, just confirm config wiring
        captured: Dict[str, Any] = {}
        def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
            captured.update({"config": config, "genesis_config_file": genesis_config_file})
            return 0
        monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

        runner = CliRunner()
        res = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path, "cpu", "--auto-mine"])

        assert res.exit_code == 0, f"Unexpected exit code: {res.exit_code}"
        config = captured["config"]
        assert config["cpu"].get("num_cpus") == 1
        assert config.get("auto_mine") is True
        assert captured["genesis_config_file"] == "genesis_block.json"


def test_gpu_auto_mine_quick_env_only(monkeypatch):
    # Only assert CLI/env wiring for a quick GPU auto-mine run (no actual GPU required)
    cfg = """
[gpu]
backend = "local"
devices = ["0"]
"""
    from typing import Dict, Any
    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        # Patch runner to avoid actually spawning GPU work, just confirm exit path
        captured: Dict[str, Any] = {}
        def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
            captured.update({"config": config, "genesis_config_file": genesis_config_file})
            return 0
        monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)
        runner = CliRunner()
        res = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path, "gpu", "--auto-mine"])
        assert res.exit_code == 0
        config = captured["config"]
        assert config["gpu"].get("backend") == "local"
        assert config["gpu"].get("devices") == ["0"]
        assert config.get("auto_mine") is True

