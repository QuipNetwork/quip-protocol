import os
import socket
import contextlib
from tempfile import TemporaryDirectory
from typing import Optional

from click.testing import CliRunner

import quip_cli


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def test_network_cpu_only_smoke():
    # Validate simulator prints correct commands for a 3-CPU-node network
    base_port = _find_free_port()
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_network_simulator,
        ["--scenario", "cpu", "--num-cpu", "3", "--base-port", str(base_port), "--print-only"],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"Running: quip-network-node cpu --port {base_port}" in out
    assert f"Running: quip-network-node cpu --port {base_port+1} --peer localhost:{base_port}" in out
    assert f"Running: quip-network-node cpu --port {base_port+2} --peer localhost:{base_port}" in out


def test_network_cpu_plus_gpu_env_only(monkeypatch):
    # Validate GPU config propagation in a mixed network without launching processes
    from typing import Dict, Any
    port = _find_free_port()
    peer_port = _find_free_port()
    cfg = f"""
[global]
port = {port}
peers = ["localhost:{peer_port}"]

[gpu]
backend = "local"
devices = ["0"]
"""

    runner = CliRunner()

    captured: Dict[str, Any] = {}

    def fake_run(config: Dict[str, Any], genesis_config_file: str = "genesis_block.json"):
        captured.update({"config": config, "genesis_config_file": genesis_config_file})
        return 0

    monkeypatch.setattr(quip_cli, "_run_network_node_sync", fake_run)

    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        result = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path, "gpu"])
        assert result.exit_code == 0
        config = captured["config"]
        assert "gpu" in config
        assert config["gpu"].get("backend") == "local"
        assert config["gpu"].get("devices") == ["0"]

