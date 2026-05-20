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


def test_parse_topology_default_is_advantage2_hardware():
    """The CLI default 'advantage2_system1' resolves to the real QPU graph
    (DEFAULT_TOPOLOGY) — not a synthetic Zephyr — so the topology-hash
    guard accepts it on a chain seeded with hardware."""
    from dwave_topologies import DEFAULT_TOPOLOGY
    topology = quip_cli._parse_topology("advantage2_system1")
    assert topology.num_nodes == DEFAULT_TOPOLOGY.num_nodes
    assert topology.num_edges == DEFAULT_TOPOLOGY.num_edges
    assert quip_cli._topology_hash(topology) == quip_cli._topology_hash(
        DEFAULT_TOPOLOGY
    )


def test_parse_topology_unknown_bare_name():
    with pytest.raises(click.BadParameter, match="--topology"):
        quip_cli._parse_topology("nonexistent_topology")


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


# ── _topology_hash tests ───────────────────────────────────────────────────


def test_topology_hash_is_deterministic():
    topology = quip_cli._parse_topology("zephyr:2,2")
    h1 = quip_cli._topology_hash(topology)
    h2 = quip_cli._topology_hash(topology)
    assert h1 == h2
    assert len(h1) == 32


def test_topology_hash_differs_across_specs():
    t1 = quip_cli._parse_topology("zephyr:2,2")
    t2 = quip_cli._parse_topology("zephyr:3,2")
    assert quip_cli._topology_hash(t1) != quip_cli._topology_hash(t2)


def test_topology_hash_differs_zephyr_vs_hardware():
    """Hardware default and a synthetic Zephyr must hash to different
    bytes — otherwise the chain's mismatch guard couldn't tell them apart."""
    zephyr_topo = quip_cli._parse_topology("zephyr:9,2")
    hw_topo = quip_cli._parse_topology("advantage2_system1")
    assert quip_cli._topology_hash(zephyr_topo) != quip_cli._topology_hash(
        hw_topo
    )


# ── quip-miner subcommand tests ────────────────────────────────────────────


def test_quip_miner_cpu_config(monkeypatch):
    """quip-miner cpu --num-cpus 3 wires the right miner_config."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_cpu,
        ["--validator", "ws://localhost:9944", "--num-cpus", "3"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"cpu": {"num_cpus": 3}}
    assert captured.get("miner_kind") == "cpu"
    assert captured.get("mode") == "pow"


def test_quip_miner_gpu_local_config(monkeypatch):
    """quip-miner gpu --gpu-backend local creates CUDA config."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--validator", "ws://localhost:9944", "--gpu-backend", "local"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"cuda": [{"device": "0"}]}
    assert captured.get("miner_kind") == "gpu"
    assert captured.get("mode") == "pow"


def test_quip_miner_gpu_metal_config(monkeypatch):
    """quip-miner gpu --gpu-backend metal creates Metal config."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--validator", "ws://localhost:9944", "--gpu-backend", "metal"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"metal": [{}]}


def test_quip_miner_gpu_modal_config(monkeypatch):
    """quip-miner gpu --gpu-backend modal creates Modal config."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        ["--validator", "ws://localhost:9944", "--gpu-backend", "modal"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("miner_config") == {"modal": [{"gpu_type": "t4"}]}


def test_quip_miner_qpu_dwave_with_budget(monkeypatch):
    """quip-miner qpu --qpu-type dwave --daily-budget 40s creates DWave config."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_qpu,
        ["--validator", "ws://localhost:9944", "--qpu-type", "dwave", "--daily-budget", "40s"],
    )
    assert result.exit_code == 0, result.output
    cfg = captured.get("miner_config", {})
    assert "dwave" in cfg
    assert cfg["dwave"][0]["type"] == "dwave"
    assert cfg["dwave"][0]["daily_budget"] == "40s"


def test_quip_miner_qpu_ibm_config(monkeypatch):
    """quip-miner qpu --qpu-type ibm creates IBM QPU config with qpu_ibm miner_kind."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_qpu,
        ["--validator", "ws://localhost:9944", "--qpu-type", "ibm"],
    )
    assert result.exit_code == 0, result.output
    cfg = captured.get("miner_config", {})
    assert "ibm" in cfg
    assert cfg["ibm"][0]["type"] == "ibm"
    assert captured.get("miner_kind") == "qpu_ibm"


# ── --validator / --config plumbing ─────────────────────────────────────────


def test_quip_miner_cpu_multiple_validators_pass_through(monkeypatch):
    """Repeated --validator flags become a tuple passed to the runner."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        [
            "--validator", "ws://primary:9944",
            "--validator", "ws://standby:9944",
            "--num-cpus", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("validators") == ("ws://primary:9944", "ws://standby:9944")


def test_quip_miner_cpu_missing_validators_fails_fast(monkeypatch):
    """No --validator and no --config → ClickException with actionable text."""
    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", lambda **_: 0)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--num-cpus", "1"],
    )
    assert result.exit_code != 0
    assert "validator" in result.output.lower()


def test_quip_miner_cpu_config_file_supplies_validators(monkeypatch, tmp_path):
    """A --config TOML with [miner].validators is enough; no CLI flag needed."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\n'
        'validators = ["ws://toml:9944"]\n'
        'signer_key = "/tmp/signing.json"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--config", str(cfg), "--num-cpus", "1"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("validators") == ("ws://toml:9944",)
    assert captured.get("signer_key_path") == "/tmp/signing.json"


def test_quip_miner_cpu_cli_overrides_config_validators(monkeypatch, tmp_path):
    """CLI --validator wins over TOML validators."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\n'
        'validators = ["ws://toml:9944"]\n'
        'signer_key = "/tmp/signing.json"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        [
            "--config", str(cfg),
            "--validator", "ws://cli:9944",
            "--num-cpus", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("validators") == ("ws://cli:9944",)


# ── fail-fast guards (A / B / C) ────────────────────────────────────────────


def test_guard_a_wallet_not_configured_returns_machine_parseable_error(tmp_path):
    """Missing keystore at the resolved path → click error message
    `wallet-not-configured keystore=<path>`. Guard A fires inside
    `_run_concurrent_miner` before any network call, so no stubbing
    is needed."""
    missing = tmp_path / "no-keystore.json"
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        [
            "--validator", "ws://localhost:9944",
            "--signer-key", str(missing),
            "--num-cpus", "1",
        ],
    )
    assert result.exit_code != 0
    assert "wallet-not-configured" in result.output
    assert f"keystore={missing}" in result.output


def test_guard_b_validators_unreachable_renders_attempt_log(monkeypatch, tmp_path):
    """All validators refuse → `validators-unreachable urls=... reasons=...`."""
    # Stub: a real keystore file so Guard A passes, then a SubstrateClient
    # whose connect() raises NoValidatorReachable so Guard B fires.
    from shared.substrate_client import NoValidatorReachable, ValidatorAttempt

    fake_keystore_path = tmp_path / "signing.json"
    fake_keystore_path.write_text("{}")  # contents unused — _load is stubbed

    monkeypatch.setattr(
        quip_cli, "_load_keystore_or_fail",
        lambda _path: type("KS", (), {
            "signer": type("S", (), {
                "account_id_bytes": lambda self: b"\x00" * 32,
                "ss58_address": lambda self: "5Test",
            })(),
            "path": fake_keystore_path,
        })(),
    )

    fail = NoValidatorReachable(attempts=[
        ValidatorAttempt(url="ws://a:9944", exc_type="ConnectionRefusedError",
                         message="Connection refused"),
        ValidatorAttempt(url="ws://b:9944", exc_type="TimeoutError",
                         message="timed out"),
    ])

    # Simulate the real helper: convert NoValidatorReachable -> ClickException.
    # Signature mirrors the pool-aware `_connect_or_fail(pool, role)`.
    async def fake_connect_or_fail(_pool, role="rpc"):
        urls = ",".join(a.url for a in fail.attempts)
        reasons = ",".join(a.exc_type for a in fail.attempts)
        import click as _click
        raise _click.ClickException(
            f"validators-unreachable urls={urls} reasons={reasons}"
        )

    monkeypatch.setattr(quip_cli, "_connect_or_fail", fake_connect_or_fail)

    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        [
            "--validator", "ws://a:9944",
            "--validator", "ws://b:9944",
            "--signer-key", str(fake_keystore_path),
            "--num-cpus", "1",
        ],
    )
    assert result.exit_code != 0
    assert "validators-unreachable" in result.output
    assert "ws://a:9944" in result.output
    assert "ws://b:9944" in result.output
    assert "ConnectionRefusedError" in result.output
    assert "TimeoutError" in result.output


def test_guard_c_wallet_underfunded_without_faucet_fails_fast():
    """Direct test on the helper since wiring full _run_concurrent_miner
    needs a live chain. Guard C with balance=0 + no faucet should raise
    ClickException carrying the structured error code."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    fake_client = MagicMock()
    fake_client.query_balance = AsyncMock(return_value=0)
    fake_client.urls = ("ws://x:9944",)
    fake_keystore = MagicMock()
    fake_keystore.signer.account_id_bytes.return_value = b"\x00" * 32
    fake_keystore.signer.ss58_address.return_value = "5Test"
    fake_keystore.path = "/tmp/k.json"

    async def _do():
        with pytest.raises(
            click.ClickException,
            match=r"wallet-underfunded ss58=5Test balance=0 threshold=\d+",
        ):
            await quip_cli._ensure_funded_or_fail(
                fake_client,
                fake_keystore,
                faucet_url=None,
                min_balance=2_000_000_000_000,
            )

    asyncio.run(_do())


def test_guard_c_wallet_funded_returns_balance():
    """When balance is at/above threshold, helper just returns it (no raise)."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    fake_client = MagicMock()
    fake_client.query_balance = AsyncMock(return_value=5_000_000_000_000)
    fake_client.urls = ("ws://x:9944",)
    fake_keystore = MagicMock()
    fake_keystore.signer.account_id_bytes.return_value = b"\x00" * 32

    async def _do():
        balance = await quip_cli._ensure_funded_or_fail(
            fake_client,
            fake_keystore,
            faucet_url=None,
            min_balance=2_000_000_000_000,
        )
        assert balance == 5_000_000_000_000

    asyncio.run(_do())


def test_quip_miner_no_longer_accepts_node_url(monkeypatch):
    """--node-url was removed in v0.2; passing it must fail with click's
    'No such option' error so operators see the rename clearly."""
    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", lambda **_: 0)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--node-url", "ws://x:9944", "--num-cpus", "1"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower() or "--node-url" in result.output
