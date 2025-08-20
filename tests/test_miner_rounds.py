import os
from click.testing import CliRunner
import quip_cli


def test_cpu_auto_mine_quick(monkeypatch):
    # Limit rounds via --auto-mine and exit after auto-mine
    runner = CliRunner()
    monkeypatch.setenv("QUIP_TEST_EXIT_AFTER_AUTOMINE", "1")
    monkeypatch.setenv("QUIP_TEST_EASE_DIFFICULTY", "1")
    res = runner.invoke(quip_cli.quip_network_node, ["cpu", "--port", "0", "--auto-mine", "3"])
    assert res.exit_code == 0


def test_gpu_auto_mine_quick_env_only(monkeypatch):
    # Only assert CLI/env wiring for a quick GPU auto-mine run (no actual GPU required)
    cfg = """
[global]
default = "gpu"
port = 0

[gpu]
backend = "local"
devices = ["0"]
"""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "cfg.toml")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        # Patch runner to avoid actually spawning GPU work, just confirm exit path
        captured = {}
        def fake_run(kind, host, port, peer, auto_mine, env_overrides=None):
            captured.update({"env": env_overrides or {}, "auto": auto_mine})
            return 0
        monkeypatch.setattr(quip_cli, "_run_p2p_node", fake_run)
        runner = CliRunner()
        res = runner.invoke(quip_cli.quip_network_node, ["--config", cfg_path, "gpu", "--auto-mine", "5"])
        assert res.exit_code == 0
        assert captured["env"].get("QUIP_GPU_BACKEND") == "local"
        assert captured["env"].get("QUIP_GPU_DEVICES") == "0"
        assert captured["auto"] == 5

