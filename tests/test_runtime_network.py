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
    # Validate GPU env propagation in a mixed network without launching processes
    cfg = """
[global]
default = "gpu"
port = {PORT}
peer = "localhost:{PEER}"

[gpu]
backend = "local"
devices = ["0"]
""".replace("{PORT}", str(_find_free_port())).replace("{PEER}", str(_find_free_port()))

    runner = CliRunner()

    captured = {}

    def fake_run(kind: str, host: str, port: int, peer: Optional[str], auto_mine: int, env_overrides: Optional[dict] = None, genesis_config_file: str = "genesis_block.json"):
        captured.update({"kind": kind, "env": env_overrides or {}})
        return 0

    monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)

    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        result = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path])
        assert result.exit_code == 0
        assert captured["kind"] == "gpu"
        assert captured["env"].get("QUIP_GPU_BACKEND") == "local"
        assert captured["env"].get("QUIP_GPU_DEVICES") == "0"

