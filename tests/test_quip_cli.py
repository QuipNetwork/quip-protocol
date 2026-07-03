"""Unit tests for quip_cli helpers: topology parsing, injection, hashing, and subcommands."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import click
import pytest
from click.testing import CliRunner

import quip_cli
from shared.quantum_proof_of_work import (
    DEFAULT_ALLOWED_H,
    DEFAULT_ALLOWED_J,
    DEFAULT_ALLOWED_SPIN,
)
from shared.topology_hash import topology_hash


def _hash(topology) -> bytes:
    return topology_hash(
        topology.nodes,
        topology.edges,
        DEFAULT_ALLOWED_H,
        DEFAULT_ALLOWED_J,
        DEFAULT_ALLOWED_SPIN,
    )


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
    assert _hash(topology) == _hash(DEFAULT_TOPOLOGY)


def test_parse_topology_unknown_bare_name():
    with pytest.raises(click.BadParameter, match="--topology"):
        quip_cli._parse_topology("nonexistent_topology")


def test_parse_topology_unknown_family():
    with pytest.raises(click.BadParameter, match="zephyr"):
        quip_cli._parse_topology("pegasus:9,2")


def test_parse_topology_bad_params_non_integer():
    with pytest.raises(click.BadParameter):
        quip_cli._parse_topology("zephyr:abc,2")


# ── topology → MinerCore wiring ─────────────────────────────────────────────


def test_run_concurrent_miner_wires_topology_to_minercore():
    """Guard the exact wiring Task 5 added: _run_concurrent_miner must pass
    the resolved topology to MinerCore and no longer route through the removed
    per-kind helper. Driving the full async path needs chain/keystore mocks;
    assert against the function source instead (catches a dropped/renamed
    `topology=topology` or a reintroduced `_inject_topology`)."""
    import inspect

    src = inspect.getsource(quip_cli._run_concurrent_miner)
    assert "topology=topology" in src
    assert "_inject_topology" not in src


# ── topology_hash tests ────────────────────────────────────────────────────


def test_topology_hash_differs_zephyr_vs_hardware():
    """Hardware default and a synthetic Zephyr must hash to different
    bytes — otherwise the chain's mismatch guard couldn't tell them apart."""
    zephyr_topo = quip_cli._parse_topology("zephyr:9,2")
    hw_topo = quip_cli._parse_topology("advantage2_system1")
    assert _hash(zephyr_topo) != _hash(hw_topo)


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
    assert captured.get("mempool_enabled") is True  # cpu default: on


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
    assert captured.get("mempool_enabled") is True  # gpu default: on


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


# ----------------------------------------------------------------------
# Identification flags are threaded through to _run_concurrent_miner
# ----------------------------------------------------------------------


def test_quip_miner_cpu_identification_flags_threaded(monkeypatch):
    """--node-name/--public-host/--public-port/--node-log on the cpu
    subcommand reach _run_concurrent_miner so auto-identify can build
    the descriptor and setup_logging can attach the file handler."""
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_cpu,
        [
            "--validator", "ws://localhost:9944",
            "--node-name", "rig-01",
            "--public-host", "miner.example.com",
            "--public-port", "8086",
            "--node-log", "/tmp/quip-miner.log",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("node_name") == "rig-01"
    assert captured.get("public_host") == "miner.example.com"
    assert captured.get("public_port") == 8086
    assert captured.get("node_log") == "/tmp/quip-miner.log"


def test_quip_miner_gpu_identification_flags_threaded(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    runner = CliRunner()
    result = runner.invoke(
        quip_cli.quip_miner_gpu,
        [
            "--validator", "ws://localhost:9944",
            "--gpu-backend", "local",
            "--node-name", "gpu-rig",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("node_name") == "gpu-rig"
    assert captured.get("public_host") is None  # absent → propagates as None


def test_quip_miner_cpu_identification_via_toml(monkeypatch, tmp_path):
    """[miner] node_name / public_host / public_port / node_log in TOML
    are surfaced to _run_concurrent_miner."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://localhost:9944"]\n'
        'signer_key = "~/.quip-miner/signing.json"\n'
        'node_name = "toml-rig"\n'
        'public_host = "toml.example.com"\n'
        'public_port = 9099\n'
        'node_log = "/var/log/quip-miner.log"\n'
    )
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--config", str(p)],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("node_name") == "toml-rig"
    assert captured.get("public_host") == "toml.example.com"
    assert captured.get("public_port") == 9099
    assert captured.get("node_log") == "/var/log/quip-miner.log"


def _mempool_toml(tmp_path, extra: str = "") -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://localhost:9944"]\n'
        'signer_key = "~/.quip-miner/signing.json"\n'
        f'{extra}'
    )
    return p


def _capture_run(monkeypatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    return captured


def test_quip_miner_cpu_mempool_false_in_toml_disables(monkeypatch, tmp_path):
    """An explicit `mempool = false` always wins over the cpu default."""
    p = _mempool_toml(tmp_path, "mempool = false\n")
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_enabled") is False


def test_quip_miner_cpu_mempool_defaults_on(monkeypatch, tmp_path):
    """No `mempool` key → cpu/gpu default ON (pow + mempool priority)."""
    p = _mempool_toml(tmp_path)
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_enabled") is True


def test_quip_miner_qpu_mempool_defaults_off(monkeypatch, tmp_path):
    """QPU defaults mempool OFF (paid samples; opt-in only)."""
    p = _mempool_toml(tmp_path, '[dwave]\ndaily_budget = "60s"\n')
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("miner_kind") == "qpu"
    assert captured.get("mempool_enabled") is False


def test_quip_miner_qpu_mempool_explicit_true_opts_in(monkeypatch, tmp_path):
    """`mempool = true` on a QPU node overrides the qpu default-off."""
    p = _mempool_toml(
        tmp_path, 'mempool = true\n[dwave]\ndaily_budget = "60s"\n'
    )
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_enabled") is True


def test_quip_mempool_env_zero_force_disables(monkeypatch, tmp_path):
    """QUIP_MEMPOOL=0 (supervisor owner election, T8) beats config."""
    monkeypatch.setenv("QUIP_MEMPOOL", "0")
    p = _mempool_toml(tmp_path, "mempool = true\n")
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_enabled") is False


def test_quip_miner_cpu_mempool_min_reward_threaded(monkeypatch, tmp_path):
    """[miner] mempool_min_reward reaches the runner (→ producer)."""
    p = _mempool_toml(tmp_path, "mempool_min_reward = 250\n")
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_min_reward") == 250


def test_quip_miner_cpu_mempool_min_reward_defaults_zero(monkeypatch, tmp_path):
    p = _mempool_toml(tmp_path)
    captured = _capture_run(monkeypatch)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("mempool_min_reward") == 0


def test_quip_miner_cpu_rejects_removed_mode_flag(monkeypatch):
    """--mode was removed with the two-controller split; passing it must
    fail loudly so operators see the config-key change."""
    _capture_run(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--validator", "ws://x:9944", "--mode", "both"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_quip_miner_cpu_listen_port_aliased_to_rest(monkeypatch, tmp_path):
    """TOML alias resolution: listen → rest_host, port → rest_port. CLI
    receives the canonical values."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\n'
        'validators = ["ws://localhost:9944"]\n'
        'signer_key = "~/.quip-miner/signing.json"\n'
        'listen = "0.0.0.0"\n'
        'port = 8087\n'
    )
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert captured.get("rest_host") == "0.0.0.0"
    assert captured.get("rest_port") == 8087


# ----------------------------------------------------------------------
# _auto_identify behavior
# ----------------------------------------------------------------------


def test_default_node_name_falls_back_to_quip_miner(monkeypatch):
    """If gethostname raises OSError, fall back to 'quip-miner'."""
    import socket

    def boom():
        raise OSError("no hostname")

    monkeypatch.setattr(socket, "gethostname", boom)
    assert quip_cli._default_node_name() == "quip-miner"


def test_default_node_name_empty_string_falls_back(monkeypatch):
    """Empty hostname still falls back to 'quip-miner'."""
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "")
    assert quip_cli._default_node_name() == "quip-miner"


def test_auto_identify_submission_failure_is_fatal(monkeypatch):
    """Filing the descriptor is a fatal startup step: a submission that never
    succeeds raises ClickException(descriptor-failed) after the retries."""
    import asyncio

    class FakeClient:
        async def submit_extrinsic(self, *_a, **_kw):
            raise RuntimeError("validator rejected extrinsic")

        async def descriptor_schema_version(self):
            return 2

        async def encode_call_args(self, _module, _call, _params):
            return b"scale-args"

        async def query_descriptor_payload_hash(self, _account):
            return None  # nothing filed — startup must submit

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccountId00000000000000000000000000000000000"

        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    async def _no_probe():
        return None
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _no_probe)
    # One attempt (no backoff sleep) so the always-failing submit fails fast.
    monkeypatch.setattr(quip_cli, "_STARTUP_RETRY_ATTEMPTS", 1)
    with pytest.raises(
        click.ClickException,
        match=r"descriptor-failed ss58=5FakeAccountId.*validator rejected",
    ):
        asyncio.run(quip_cli._auto_identify(
            FakeClient(),
            FakeKeystore(),
            node_name="test-rig",
            public_host=None,
            public_port=None,
            log_level=None,
            miners_config={"cpu": {"num_cpus": 1}},
        ))


def test_auto_identify_retries_then_posts(monkeypatch, caplog):
    """A transient submission failure (e.g. stale nonce right after
    register_miner) is retried, and the descriptor still posts on start."""
    import asyncio
    import logging

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(quip_cli.asyncio, "sleep", _no_sleep)

    class FakeReceipt:
        error = None
        extrinsic_hash = "0xabc"
        block_hash = "0xdef"

    class FakeClient:
        def __init__(self):
            self.attempts = 0
            self.posted = False

        async def submit_extrinsic(self, _module, _call, _args, _signer, **_kw):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Priority is too low")  # stale-nonce shape
            self.posted = True
            return FakeReceipt()

        async def descriptor_schema_version(self):
            return 2

        async def encode_call_args(self, _module, _call, _params):
            return b"scale-args"

        async def query_descriptor_payload_hash(self, _account):
            from substrate.miner_registry import descriptor_payload_hash

            return descriptor_payload_hash(b"scale-args") if self.posted else None

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccountId00000000000000000000000000000000000"

        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    async def _no_probe():
        return None

    monkeypatch.setattr(quip_cli, "_detect_public_ip", _no_probe)
    caplog.set_level(logging.INFO, logger="quip_miner.auto_identify")
    client = FakeClient()
    asyncio.run(quip_cli._auto_identify(
        client,
        FakeKeystore(),
        node_name="test-rig",
        public_host="rig.example.com",
        public_port=None,
        log_level=None,
        miners_config={"cpu": {"num_cpus": 1}},
    ))
    assert client.attempts == 2, "should retry after the first transient failure"
    assert any(
        "auto-identify submitted" in rec.message for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_auto_identify_submits_miner_registry_descriptor(monkeypatch, caplog):
    """Startup identification submits MinerRegistry.set_descriptor."""
    import asyncio
    import logging

    class FakeReceipt:
        error = None
        extrinsic_hash = "0xabc"
        block_hash = "0xdef"

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def submit_extrinsic(self, module, call, args, _signer, **_kw):
            self.calls.append((module, call, args))
            return FakeReceipt()

        async def descriptor_schema_version(self):
            return 2

        async def encode_call_args(self, _module, _call, _params):
            return b"scale-args"

        async def query_descriptor_payload_hash(self, _account):
            from substrate.miner_registry import descriptor_payload_hash

            return descriptor_payload_hash(b"scale-args") if self.calls else None

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccountId00000000000000000000000000000000000"

        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    client = FakeClient()
    caplog.set_level(logging.WARNING, logger="quip_miner.auto_identify")
    async def _no_probe():
        return None
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _no_probe)
    asyncio.run(quip_cli._auto_identify(
        client,
        FakeKeystore(),
        node_name="test-rig",
        public_host=None,
        public_port=None,
        log_level=None,
        miners_config={"cpu": {"num_cpus": 1}},
    ))
    assert len(client.calls) == 1
    module, call, params = client.calls[0]
    assert (module, call) == ("MinerRegistry", "set_descriptor")
    # Stub runtime advertises descriptor schema V2, so the miner posts V2
    # (system_info + runtime) — the richest the chain accepts.
    body = params["descriptor"]["V2"]
    assert body["system_info"] is not None
    assert body["runtime"] is not None
    # BoundedVec fields are 1-tuple-wrapped for the runtime composite shape
    # (see test_miner_registry for the rationale).
    assert body["node_id"] == (b"test-rig",)
    assert body["node_name"] == (b"test-rig",)
    (miners,) = body["miners"]
    assert miners[0]["kind"] == "Cpu"


# ----------------------------------------------------------------------
# _auto_identify chain-state check: skip when current, verify after submit
# ----------------------------------------------------------------------


class _FakeIdentifySigner:
    def ss58_address(self):
        return "5FakeAccountId00000000000000000000000000000000000"

    def account_id_bytes(self):
        return b"\x00" * 32


class _FakeIdentifyKeystore:
    signer = _FakeIdentifySigner()


class _FakeIdentifyClient:
    """Client stub modelling the descriptor round-trip.

    ``encode_call_args`` returns fixed SCALE bytes; the chain "shows" the
    matching payload hash only once ``submits >= visible_after_submits``
    (before that, ``query_descriptor_payload_hash`` returns ``chain_hash``
    — ``None`` models a fresh miner, a different digest a stale record).
    """

    ENCODED = b"scale-encoded-descriptor-args"

    def __init__(self, *, chain_hash=None, visible_after_submits=1):
        self.chain_hash = chain_hash
        self.visible_after_submits = visible_after_submits
        self.submits = 0
        self.calls = []

    async def descriptor_schema_version(self):
        return 2

    async def encode_call_args(self, _module, _call, _params):
        return self.ENCODED

    async def query_descriptor_payload_hash(self, _account):
        from substrate.miner_registry import descriptor_payload_hash

        if self.submits >= self.visible_after_submits:
            return descriptor_payload_hash(self.ENCODED)
        return self.chain_hash

    async def submit_extrinsic(self, module, call, args, _signer, **_kw):
        self.submits += 1
        self.calls.append((module, call, args))
        return SimpleNamespace(error=None, extrinsic_hash="0xabc", block_hash="0xdef")


def _run_auto_identify(client):
    import asyncio

    asyncio.run(quip_cli._auto_identify(
        client,
        _FakeIdentifyKeystore(),
        node_name="test-rig",
        public_host="rig.example.com",
        public_port=None,
        log_level=None,
        miners_config={"cpu": {"num_cpus": 1}},
    ))


def test_auto_identify_skips_submit_when_chain_record_matches():
    """When the on-chain payload_hash already equals what we would submit,
    startup files nothing — the descriptor is current."""
    from substrate.miner_registry import descriptor_payload_hash

    client = _FakeIdentifyClient(
        chain_hash=descriptor_payload_hash(_FakeIdentifyClient.ENCODED),
    )
    _run_auto_identify(client)
    assert client.submits == 0


def test_auto_identify_submits_when_no_descriptor_on_chain():
    """A fresh miner (no NodeDescriptors entry) files the descriptor."""
    client = _FakeIdentifyClient(chain_hash=None)
    _run_auto_identify(client)
    assert client.submits == 1
    assert client.calls[0][:2] == ("MinerRegistry", "set_descriptor")


def test_auto_identify_submits_update_when_chain_record_differs():
    """A stale on-chain record (different payload_hash) triggers an update."""
    client = _FakeIdentifyClient(chain_hash=b"\x11" * 32)
    _run_auto_identify(client)
    assert client.submits == 1


def test_auto_identify_resubmits_until_descriptor_visible_on_chain(monkeypatch):
    """An in-block receipt is not enough: if chain state never shows our
    payload_hash (e.g. the block was orphaned), the attempt is retried
    until the descriptor is actually readable from storage."""
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(quip_cli.asyncio, "sleep", _no_sleep)
    client = _FakeIdentifyClient(chain_hash=None, visible_after_submits=2)
    _run_auto_identify(client)
    assert client.submits == 2


# ----------------------------------------------------------------------
# _detect_public_ip behaviour
# ----------------------------------------------------------------------


def test_detect_public_ip_uses_first_successful_service(monkeypatch):
    """Walk the service list in order, return the first parseable IP.
    Probes that error are skipped, not retried."""
    import asyncio
    import urllib.error

    call_order: list[str] = []

    def fake_urlopen(req, timeout=5.0, context=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        call_order.append(url)
        # First service errors, second returns a valid IP.
        if url == quip_cli._PUBLIC_IP_SERVICES[0]:
            raise urllib.error.URLError("simulated DNS failure")
        class _Resp:
            def read(self_inner):
                return b"203.0.113.42\n"
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *_a):
                return False
        return _Resp()

    monkeypatch.setattr(quip_cli.urllib.request, "urlopen", fake_urlopen)
    ip = asyncio.run(quip_cli._detect_public_ip(timeout=0.01))
    assert ip == "203.0.113.42"
    # First service was attempted then we moved on; we did NOT skip
    # straight to a later one.
    assert call_order[0] == quip_cli._PUBLIC_IP_SERVICES[0]
    assert call_order[1] == quip_cli._PUBLIC_IP_SERVICES[1]


def test_detect_public_ip_rejects_non_ip_response(monkeypatch):
    """A service that returns HTML or a hostname must be skipped — we
    only want a parseable IPv4/IPv6 address in the descriptor."""
    import asyncio

    bodies = iter([b"<html>nope</html>", b"2001:db8::1\n"])

    def fake_urlopen(req, timeout=5.0, context=None):
        body = next(bodies)
        class _Resp:
            def read(self_inner):
                return body
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *_a):
                return False
        return _Resp()

    monkeypatch.setattr(quip_cli.urllib.request, "urlopen", fake_urlopen)
    ip = asyncio.run(quip_cli._detect_public_ip(timeout=0.01))
    # IPv6 was the second response and parses cleanly.
    assert ip == "2001:db8::1"


def test_detect_public_ip_all_fail_returns_none(monkeypatch, caplog):
    """When every probe errors, we get None and a warning explaining
    how to skip detection."""
    import asyncio
    import logging
    import urllib.error

    def fake_urlopen(req, timeout=5.0, context=None):
        raise urllib.error.URLError("simulated offline")

    monkeypatch.setattr(quip_cli.urllib.request, "urlopen", fake_urlopen)
    caplog.set_level(logging.WARNING, logger="quip_miner.auto_identify")
    ip = asyncio.run(quip_cli._detect_public_ip(timeout=0.01))
    assert ip is None
    assert any(
        "could not detect public IP" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_auto_identify_uses_detected_public_ip_when_unset(monkeypatch):
    """When public_host is None, _auto_identify queries _detect_public_ip
    and threads the result into the NodeDescriptor it submits."""
    import asyncio

    async def _fake_probe():
        return "198.51.100.7"
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _fake_probe)

    captured: dict = {}
    real_build = quip_cli.build_descriptor
    def fake_build(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)
    monkeypatch.setattr(quip_cli, "build_descriptor", fake_build)

    asyncio.run(quip_cli._auto_identify(
        _FakeIdentifyClient(),
        _FakeIdentifyKeystore(),
        node_name="test-rig",
        public_host=None,
        public_port=None,
        log_level=None,
        miners_config={"cpu": {"num_cpus": 1}},
    ))
    assert captured.get("public_host") == "198.51.100.7"


def test_auto_identify_skips_detection_when_public_host_set(monkeypatch):
    """An explicitly configured public_host short-circuits detection —
    we never reach out to check.quip.network when the operator already
    told us what to advertise."""
    import asyncio

    called = []

    async def _fake_probe():
        called.append(True)
        return "should.not.be.used"
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _fake_probe)

    captured: dict = {}
    real_build = quip_cli.build_descriptor
    def fake_build(**kwargs):
        captured.update(kwargs)
        return real_build(**kwargs)
    monkeypatch.setattr(quip_cli, "build_descriptor", fake_build)

    asyncio.run(quip_cli._auto_identify(
        _FakeIdentifyClient(),
        _FakeIdentifyKeystore(),
        node_name="test-rig",
        public_host="miner.example.com",
        public_port=None,
        log_level=None,
        miners_config={"cpu": {"num_cpus": 1}},
    ))
    assert called == [], "detection ran despite explicit public_host"
    assert captured.get("public_host") == "miner.example.com"


# ── Backend-section TOML wiring (v0.1-shape inventory) ────────────────────


def _write_backend_toml(tmp_path, body: str, *, with_signer: bool = True):
    """Helper: write a TOML with [miner] + a backend section snippet."""
    cfg = tmp_path / "miner.toml"
    miner_section = '[miner]\nvalidators = ["ws://toml:9944"]\n'
    if with_signer:
        miner_section += 'signer_key = "/tmp/signing.json"\n'
    cfg.write_text(miner_section + body)
    return cfg


def _stub_runner(monkeypatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(quip_cli, "_run_concurrent_miner", fake_run)
    return captured


def test_quip_miner_cpu_toml_drives_num_cpus(monkeypatch, tmp_path):
    """`[cpu] num_cpus = 4` in TOML produces a 4-CPU miner_config without
    a --num-cpus flag. Restores the v0.1 path where the operator drives
    inventory from the file."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cpu]\nnum_cpus = 4\n")
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"] == {"cpu": {"num_cpus": 4}}


def test_quip_miner_cpu_toml_plus_num_cpus_flag_is_conflict(monkeypatch, tmp_path):
    """TOML `[cpu]` + explicit `--num-cpus` → UsageError. Matches the
    operator-requested rule: pick one source of truth per backend."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cpu]\nnum_cpus = 4\n")
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--config", str(cfg), "--num-cpus", "2"],
    )
    assert result.exit_code != 0
    assert "config-conflict" in result.output
    assert "--num-cpus" in result.output
    assert "[cpu]" in result.output


def test_quip_miner_cpu_default_num_cpus_with_toml_cpu_section_ok(monkeypatch, tmp_path):
    """The click default for --num-cpus (1) must NOT trip the conflict
    detector — only an explicit COMMANDLINE source does. This is what
    makes `quip-miner cpu --config x.toml` work without re-passing
    every flag."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cpu]\nnum_cpus = 8\n")
    # No --num-cpus on the cmdline → click fills in default 1, but
    # `get_parameter_source` reports DEFAULT, not COMMANDLINE.
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"] == {"cpu": {"num_cpus": 8}}


def test_quip_miner_cpu_cli_only_still_works(monkeypatch):
    """No --config → backends={} → CLI synthesis like before. Regression
    guard for the most common operator path."""
    captured = _stub_runner(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner_cpu,
        ["--validator", "ws://x:9944", "--num-cpus", "3"],
    )
    assert result.exit_code == 0, result.output
    assert captured["miner_config"] == {"cpu": {"num_cpus": 3}}


def test_quip_miner_gpu_toml_drives_cuda_devices(monkeypatch, tmp_path):
    """`[cuda.0]` + `[cuda.1]` produce a two-device inventory that
    `MinerCore._build_gpu_specs` consumes via _normalize_gpu_config."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(
        tmp_path,
        '[gpu]\nutilization = 80\n'
        '[cuda.0]\n'
        '[cuda.1]\nutilization = 50\n',
    )
    result = CliRunner().invoke(quip_cli.quip_miner_gpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    mc = captured["miner_config"]
    assert mc["gpu"] == {"utilization": 80}
    assert "0" in mc["cuda"] and "1" in mc["cuda"]
    assert mc["cuda"]["1"] == {"utilization": 50}


def test_quip_miner_gpu_toml_plus_backend_flag_is_conflict(monkeypatch, tmp_path):
    """`[cuda.0]` in TOML + `--gpu-backend local` → UsageError."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cuda.0]\n")
    result = CliRunner().invoke(
        quip_cli.quip_miner_gpu,
        ["--config", str(cfg), "--gpu-backend", "metal"],
    )
    assert result.exit_code != 0
    assert "config-conflict" in result.output
    assert "--gpu-backend" in result.output
    assert "[cuda]" in result.output


def test_quip_miner_gpu_metal_toml(monkeypatch, tmp_path):
    """A bare `[metal]` table — Apple Silicon rig — works without any
    `--gpu-backend` flag."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[metal]\nutilization = 100\n")
    result = CliRunner().invoke(quip_cli.quip_miner_gpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"] == {"metal": {"utilization": 100}}


def test_quip_miner_gpu_modal_toml_with_gpu_type(monkeypatch, tmp_path):
    """[modal] gpu_type = "a10g" picks the cloud GPU class from TOML."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, '[modal]\ngpu_type = "a10g"\n')
    result = CliRunner().invoke(quip_cli.quip_miner_gpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"]["modal"] == {"gpu_type": "a10g"}


def test_quip_miner_qpu_dwave_toml(monkeypatch, tmp_path):
    """`[dwave]` daily_budget/solver passed verbatim — what
    `_build_qpu_specs` reads to populate the QPU sampler's cfg block."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(
        tmp_path,
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'solver = "Advantage2_system1"\n',
    )
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"]["dwave"] == {
        "daily_budget": "60s",
        "solver": "Advantage2_system1",
    }
    # miner_kind defaults to bare "qpu" for D-Wave (matches the v0.2
    # mempool register-solver mapping QpuDwave).
    assert captured["miner_kind"] == "qpu"


def test_quip_miner_qpu_ibm_toml_gives_qpu_ibm_kind(monkeypatch, tmp_path):
    """`[ibm]` in TOML → miner_kind='qpu_ibm' so the mempool controller
    registers under MinerType.QpuIbm without operator needing --qpu-type."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, '[ibm]\ntoken = "ibm-xyz"\n')
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_kind"] == "qpu_ibm"
    assert captured["miner_config"]["ibm"] == {"token": "ibm-xyz"}


def test_quip_miner_qpu_toml_plus_qpu_type_flag_is_conflict(monkeypatch, tmp_path):
    """`[dwave]` + `--qpu-type ibm` is unambiguously contradictory →
    UsageError."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, '[dwave]\ndaily_budget = "60s"\n')
    result = CliRunner().invoke(
        quip_cli.quip_miner_qpu,
        ["--config", str(cfg), "--qpu-type", "ibm"],
    )
    assert result.exit_code != 0
    assert "config-conflict" in result.output


def test_quip_miner_qpu_toml_plus_daily_budget_flag_is_conflict(monkeypatch, tmp_path):
    """`[dwave]` + `--daily-budget 5m` → UsageError. --daily-budget is a
    QPU-section flag so it counts as a conflict alongside --qpu-type."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, '[dwave]\ndaily_budget = "60s"\n')
    result = CliRunner().invoke(
        quip_cli.quip_miner_qpu,
        ["--config", str(cfg), "--daily-budget", "5m"],
    )
    assert result.exit_code != 0
    assert "config-conflict" in result.output
    assert "--daily-budget" in result.output


def test_quip_miner_qpu_default_qpu_type_with_toml_dwave_ok(monkeypatch, tmp_path):
    """Click's default --qpu-type=dwave must NOT trip the conflict
    detector against a TOML `[dwave]` section (DEFAULT, not COMMANDLINE).
    Same DEFAULT-vs-COMMANDLINE rationale as the CPU test."""
    captured = _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, '[dwave]\ndaily_budget = "120s"\n')
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert captured["miner_config"]["dwave"]["daily_budget"] == "120s"


# ── _auto_identify secret-leakage guard (TOML → MinerRegistry) ─────────


def _capture_auto_identify_params(monkeypatch, miners_config: Dict[str, Any]) -> dict:
    """Run _auto_identify with a fake client and return set_descriptor params.

    Exercises the same
    code path that cpu/gpu/qpu subcommands invoke on every startup."""
    import asyncio
    captured: Dict[str, Any] = {}

    class FakeClient(_FakeIdentifyClient):
        async def submit_extrinsic(self, module, call, args, _signer, **_kw):
            captured["module"] = module
            captured["call"] = call
            captured["params"] = args
            return await super().submit_extrinsic(module, call, args, _signer, **_kw)

    async def _no_probe():
        return None
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _no_probe)

    asyncio.run(quip_cli._auto_identify(
        FakeClient(),
        _FakeIdentifyKeystore(),
        node_name="rig",
        public_host="rig.example.com",
        public_port=None,
        log_level=None,
        miners_config=miners_config,
    ))
    assert "params" in captured, "_auto_identify did not call submit_extrinsic"
    assert (captured["module"], captured["call"]) == (
        "MinerRegistry",
        "set_descriptor",
    )
    return captured["params"]


def test_auto_identify_does_not_leak_tokens_from_toml_loaded_miners_config(
    monkeypatch, tmp_path,
):
    """Full live-startup integration: write a TOML with every QPU
    vendor's `token`, load it through `load_backend_config`, hand the
    resulting dict to `_auto_identify`, and inspect the bytes that
    would land on chain via `MinerRegistry.set_descriptor`.

    This is the regression test the v0.2 backend-table restoration
    needs. The descriptor pipeline is the only layer between the
    in-memory `miners_config` dict and the on-chain descriptor; if any
    refactor in that layer regresses, this catches it."""
    from shared.miner_config import load_backend_config

    p = tmp_path / "all-vendors.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[ibm]\ntoken = "ibm-startup-sentinel-aaaaa"\n'
        '[braket]\ntoken = "braket-startup-sentinel-bbbbb"\n'
        '[pasqal]\ntoken = "pasqal-startup-sentinel-ccccc"\n'
        '[ionq]\ntoken = "ionq-startup-sentinel-ddddd"\n'
        '[origin]\ntoken = "origin-startup-sentinel-eeeee"\n'
    )
    miners_config = load_backend_config(p)
    params = _capture_auto_identify_params(monkeypatch, miners_config)

    text = repr(params)
    for sentinel in (
        "ibm-startup-sentinel-aaaaa",
        "braket-startup-sentinel-bbbbb",
        "pasqal-startup-sentinel-ccccc",
        "ionq-startup-sentinel-ddddd",
        "origin-startup-sentinel-eeeee",
    ):
        assert sentinel not in text, (
            f"secret leaked to MinerRegistry descriptor: {sentinel}\n"
            f"payload was: {text}"
        )

    # The vendor entries DID make it into the descriptor as backend labels
    # (the legitimate signal indexers need); just without credentials.
    # miners is a BoundedVec (1-tuple-wrapped); each backend is an
    # Option<BoundedVec> wrapped as (bytes,) when present.
    (miners,) = params["descriptor"]["V2"]["miners"]
    backends = {
        m["backend"][0].decode("utf-8")
        for m in miners
        if m.get("backend") is not None
    }
    for vendor in ("ibm", "braket", "pasqal", "ionq", "origin"):
        assert vendor in backends, f"{vendor} entry missing from descriptor"


def test_auto_identify_blocks_credential_smuggled_through_solver(
    monkeypatch, tmp_path,
):
    """Live-startup path: a TOML where the operator pastes a credential
    into the solver field. The strict solver-name regex at
    `_qpu_spec_entry` drops the value before it reaches the descriptor,
    so the descriptor gets submitted (with `solver` absent) — _auto_identify
    is best-effort by contract, it doesn't block mining on identify
    failures. Assert the bad value never leaves the process."""
    from shared.miner_config import load_backend_config

    p = tmp_path / "smuggle.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\n'
        'daily_budget = "60s"\n'
        'solver = "DWAVE_API_KEY=smuggle-via-solver-1234"\n'
    )
    miners_config = load_backend_config(p)
    params = _capture_auto_identify_params(monkeypatch, miners_config)
    text = repr(params)
    assert "DWAVE_API_KEY" not in text
    assert "smuggle-via-solver-1234" not in text


# ── top-level production supervisor: quip-miner --config ─────────────


def _write_supervisor_toml(tmp_path, extra_miner="", backend="[cpu]\nnum_cpus = 1\n"):
    p = tmp_path / "config.toml"
    p.write_text(
        '[miner]\nvalidators = ["ws://localhost:9944"]\n'
        'signer_key = "/data/keystore.json"\n'
        'rest_port = 8086\n'
        f'{extra_miner}'
        f'{backend}'
    )
    return p


def test_plan_processes_spawns_telemetry_and_backend_children(monkeypatch, tmp_path):
    """A config with rest_port > 0 and one backend group plans a telemetry
    aggregator plus one miner child, all driven by --config only."""
    monkeypatch.setattr(quip_cli, "_detect_image_supports", lambda: ["cpu", "qpu"])
    cfg = _write_supervisor_toml(tmp_path)
    runtime_dir, procs = quip_cli._plan_processes(cfg)
    assert runtime_dir == cfg.resolve().parent / "runtime"
    argvs = [p["args"] for p in procs]
    assert argvs[0][:1] == ["telemetry"]
    assert "--snapshot-dir" in argvs[0] and str(runtime_dir) in argvs[0]
    assert argvs[1] == ["cpu", "--config", str(cfg)]
    # miner children get the internal aggregator glue via env, invisibly
    assert procs[1]["env"] == {
        "QUIP_RUNTIME_DIR": str(runtime_dir),
        "QUIP_TELEMETRY_EXTERNAL": "1",
    }


def test_plan_processes_omits_telemetry_when_rest_disabled(monkeypatch, tmp_path):
    """rest_port = -1 → no aggregator process (and no external-glue env)."""
    monkeypatch.setattr(quip_cli, "_detect_image_supports", lambda: ["cpu"])
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://localhost:9944"]\nrest_port = -1\n'
        '[cpu]\nnum_cpus = 1\n'
    )
    _runtime_dir, procs = quip_cli._plan_processes(cfg)
    assert [p["args"][0] for p in procs] == ["cpu"]
    assert procs[0]["env"] == {}


def test_plan_processes_rejects_backend_without_library(monkeypatch, tmp_path):
    """[cuda.0] on an install without gpu libraries fails with the
    unsupported-mode error, before any process starts."""
    monkeypatch.setattr(quip_cli, "_detect_image_supports", lambda: ["cpu", "qpu"])
    cfg = _write_supervisor_toml(tmp_path, backend="[cuda.0]\n")
    with pytest.raises(click.ClickException, match="unsupported-mode"):
        quip_cli._plan_processes(cfg)


# ── mempool owner election (T8): one substrate account = one solver ──


def _plan_with_all_supports(monkeypatch, cfg):
    """_plan_processes with every backend group importable; returns the
    per-mode env map for the miner children plus the raw proc list."""
    monkeypatch.setattr(
        quip_cli, "_detect_image_supports", lambda: ["cpu", "gpu", "qpu"]
    )
    _runtime_dir, procs = quip_cli._plan_processes(cfg)
    envs = {p["args"][0]: p["env"] for p in procs if p["args"][0] != "telemetry"}
    return envs, procs


def test_plan_processes_elects_cpu_owner_over_gpu(monkeypatch, tmp_path, capsys):
    """cpu+gpu, mempool key absent → cpu owns mempool; the gpu child is
    force-disabled via QUIP_MEMPOOL=0 while its telemetry glue survives,
    and the telemetry aggregator itself is left alone. The election is
    echoed — the operator's only visibility into why a child has
    mempool off."""
    cfg = _write_supervisor_toml(
        tmp_path, backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n"
    )
    envs, procs = _plan_with_all_supports(monkeypatch, cfg)
    assert "QUIP_MEMPOOL" not in envs["cpu"]
    assert envs["gpu"]["QUIP_MEMPOOL"] == "0"
    for mode in ("cpu", "gpu"):
        assert envs[mode]["QUIP_TELEMETRY_EXTERNAL"] == "1"
        assert "QUIP_RUNTIME_DIR" in envs[mode]
    telemetry_env = next(p["env"] for p in procs if p["args"][0] == "telemetry")
    assert "QUIP_MEMPOOL" not in telemetry_env
    out = capsys.readouterr().out
    assert "supervisor: mempool owner is cpu" in out
    assert "gpu" in out


def test_plan_processes_three_backends_disables_all_non_owners(
    monkeypatch, tmp_path
):
    """EVERY non-owner is disabled, not just one: cpu+gpu+qpu → cpu owns,
    both gpu and qpu children get the force-off. Guards against a
    disable-only-the-last-mode mutant that every 2-backend test passes."""
    cfg = _write_supervisor_toml(
        tmp_path,
        backend='[cpu]\nnum_cpus = 1\n[cuda.0]\n[dwave]\ndaily_budget = "60s"\n',
    )
    envs, _procs = _plan_with_all_supports(monkeypatch, cfg)
    assert "QUIP_MEMPOOL" not in envs["cpu"]
    assert envs["gpu"]["QUIP_MEMPOOL"] == "0"
    assert envs["qpu"]["QUIP_MEMPOOL"] == "0"


def test_plan_processes_election_independent_of_telemetry(monkeypatch, tmp_path):
    """rest_port = -1 still elects: the force-off must not ride the
    telemetry-glue branch of the env construction."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://localhost:9944"]\n'
        'signer_key = "/data/keystore.json"\n'
        'rest_port = -1\n'
        '[cpu]\nnum_cpus = 1\n[cuda.0]\n'
    )
    envs, procs = _plan_with_all_supports(monkeypatch, cfg)
    assert all(p["args"][0] != "telemetry" for p in procs)
    assert envs["cpu"] == {}
    assert envs["gpu"] == {"QUIP_MEMPOOL": "0"}


def test_plan_processes_tolerates_stale_mode_key(monkeypatch, tmp_path):
    """A legacy volume's `mode = "pow"` is dead but harmless: the config
    loads, the supervisor plans, and the election still fires."""
    cfg = _write_supervisor_toml(
        tmp_path,
        extra_miner='mode = "pow"\n',
        backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n",
    )
    envs, _procs = _plan_with_all_supports(monkeypatch, cfg)
    assert set(envs) == {"cpu", "gpu"}
    assert envs["gpu"]["QUIP_MEMPOOL"] == "0"


def test_plan_processes_elects_gpu_owner_over_qpu(monkeypatch, tmp_path):
    """gpu+qpu → gpu is the first non-qpu mode, so qpu is disabled."""
    cfg = _write_supervisor_toml(
        tmp_path, backend='[cuda.0]\n[dwave]\ndaily_budget = "60s"\n'
    )
    envs, _procs = _plan_with_all_supports(monkeypatch, cfg)
    assert "QUIP_MEMPOOL" not in envs["gpu"]
    assert envs["qpu"]["QUIP_MEMPOOL"] == "0"


def test_plan_processes_elects_cpu_owner_over_qpu(monkeypatch, tmp_path):
    """cpu+qpu → cpu owns mempool, qpu child is disabled."""
    cfg = _write_supervisor_toml(
        tmp_path, backend='[cpu]\nnum_cpus = 1\n[dwave]\ndaily_budget = "60s"\n'
    )
    envs, _procs = _plan_with_all_supports(monkeypatch, cfg)
    assert "QUIP_MEMPOOL" not in envs["cpu"]
    assert envs["qpu"]["QUIP_MEMPOOL"] == "0"


def test_plan_processes_single_backend_no_election(monkeypatch, tmp_path):
    """A single miner child needs no election — no QUIP_MEMPOOL key on
    any planned process."""
    cfg = _write_supervisor_toml(tmp_path)
    _envs, procs = _plan_with_all_supports(monkeypatch, cfg)
    assert all("QUIP_MEMPOOL" not in p["env"] for p in procs)


def test_plan_processes_mempool_false_skips_election(monkeypatch, tmp_path):
    """Explicit `mempool = false` disables participation everywhere via
    config; the supervisor adds no force-off env at all."""
    cfg = _write_supervisor_toml(
        tmp_path,
        extra_miner="mempool = false\n",
        backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n",
    )
    _envs, procs = _plan_with_all_supports(monkeypatch, cfg)
    assert all("QUIP_MEMPOOL" not in p["env"] for p in procs)


def test_plan_processes_mempool_true_still_elects(monkeypatch, tmp_path):
    """`mempool = true` cannot lift the one-solver-type-per-account
    constraint — election still disables every non-owner child."""
    cfg = _write_supervisor_toml(
        tmp_path,
        extra_miner="mempool = true\n",
        backend='[cpu]\nnum_cpus = 1\n[dwave]\ndaily_budget = "60s"\n',
    )
    envs, _procs = _plan_with_all_supports(monkeypatch, cfg)
    assert "QUIP_MEMPOOL" not in envs["cpu"]
    assert envs["qpu"]["QUIP_MEMPOOL"] == "0"


def test_detect_image_supports_probes_imports(monkeypatch):
    """cpu is always supported; gpu/qpu depend on importable libraries."""
    available = {"dwave.system"}
    monkeypatch.setattr(
        quip_cli, "_importable", lambda name: name in available
    )
    assert quip_cli._detect_image_supports() == ["cpu", "qpu"]
    available.update({"cupy"})
    assert quip_cli._detect_image_supports() == ["cpu", "gpu", "qpu"]


# ── CLI miner-type selection: supervisor --mode + subcommand narrowing ──


def test_plan_processes_mode_narrows_to_requested_type(
    monkeypatch, tmp_path, capsys
):
    """--mode gpu on a cpu+gpu config plans ONLY the gpu child (the
    telemetry aggregator survives) and echoes which configured miner
    types were dropped."""
    monkeypatch.setattr(
        quip_cli, "_detect_image_supports", lambda: ["cpu", "gpu", "qpu"]
    )
    cfg = _write_supervisor_toml(
        tmp_path, backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n"
    )
    _runtime_dir, procs = quip_cli._plan_processes(cfg, mode="gpu")
    assert [p["args"][0] for p in procs] == ["telemetry", "gpu"]
    out = capsys.readouterr().out
    assert "--mode gpu keeps gpu only" in out
    assert "dropping configured miner types: cpu" in out


def test_plan_processes_mode_skips_election(monkeypatch, tmp_path, capsys):
    """Narrowing to one child means no owner election: the kept child
    keeps its own mempool default (no QUIP_MEMPOOL force-off, no
    election echo)."""
    monkeypatch.setattr(
        quip_cli, "_detect_image_supports", lambda: ["cpu", "gpu", "qpu"]
    )
    cfg = _write_supervisor_toml(
        tmp_path, backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n"
    )
    _runtime_dir, procs = quip_cli._plan_processes(cfg, mode="cpu")
    assert all("QUIP_MEMPOOL" not in p["env"] for p in procs)
    assert "mempool owner" not in capsys.readouterr().out


def test_plan_processes_mode_unconfigured_type_fails(monkeypatch, tmp_path):
    """--mode qpu with no qpu backend sections is an error that names
    what IS configured, instead of silently planning nothing."""
    monkeypatch.setattr(
        quip_cli, "_detect_image_supports", lambda: ["cpu", "gpu", "qpu"]
    )
    cfg = _write_supervisor_toml(tmp_path)  # cpu only
    with pytest.raises(
        click.ClickException, match="no qpu backend sections"
    ) as excinfo:
        quip_cli._plan_processes(cfg, mode="qpu")
    assert "cpu" in str(excinfo.value)


def test_plan_processes_mode_matching_single_backend_stays_quiet(
    monkeypatch, tmp_path, capsys
):
    """--mode cpu on a cpu-only config drops nothing → no drop echo."""
    monkeypatch.setattr(
        quip_cli, "_detect_image_supports", lambda: ["cpu", "gpu", "qpu"]
    )
    cfg = _write_supervisor_toml(tmp_path)
    _runtime_dir, procs = quip_cli._plan_processes(cfg, mode="cpu")
    assert [p["args"][0] for p in procs] == ["telemetry", "cpu"]
    assert "dropping configured miner types" not in capsys.readouterr().out


def test_plan_processes_mode_narrows_before_image_support_check(
    monkeypatch, tmp_path
):
    """--mode cpu must boot on an install that can't run the config's
    [cuda.0]: once a type is dropped by choice, its libraries are
    irrelevant (no unsupported-mode error for dropped types)."""
    monkeypatch.setattr(quip_cli, "_detect_image_supports", lambda: ["cpu"])
    cfg = _write_supervisor_toml(
        tmp_path, backend="[cpu]\nnum_cpus = 1\n[cuda.0]\n"
    )
    _runtime_dir, procs = quip_cli._plan_processes(cfg, mode="cpu")
    assert [p["args"][0] for p in procs] == ["telemetry", "cpu"]


def test_quip_miner_mode_requires_config():
    """--mode without --config has nothing to narrow → usage error."""
    result = CliRunner().invoke(quip_cli.quip_miner, ["--mode", "cpu"])
    assert result.exit_code != 0
    assert "--mode requires --config" in result.output


def test_quip_miner_mode_rejected_with_subcommand(tmp_path):
    """--mode is supervisor-only; with a subcommand the subcommand
    already selects the miner type."""
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["--mode", "cpu", "resolve-modes"]
    )
    assert result.exit_code != 0
    assert "already selects the miner type" in result.output


def test_quip_miner_mode_passes_through_to_supervisor(monkeypatch, tmp_path):
    """`quip-miner --config X --mode gpu` hands the mode to the
    supervisor run."""
    captured = {}

    def fake_supervisor(config_path, *, mode=None):
        captured["config_path"] = config_path
        captured["mode"] = mode
        return 0

    monkeypatch.setattr(quip_cli, "_run_supervisor", fake_supervisor)
    cfg = _write_supervisor_toml(tmp_path)
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["--config", str(cfg), "--mode", "gpu"]
    )
    assert result.exit_code == 0, result.output
    assert captured["mode"] == "gpu"


def test_quip_miner_cpu_warns_on_dropped_configured_types(
    monkeypatch, tmp_path
):
    """Direct `quip-miner cpu` against a config that also declares gpu
    and qpu sections keeps cpu and warns what was dropped — a
    wrong-subcommand launch must be visible, not silent."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(
        tmp_path,
        '[cpu]\nnum_cpus = 1\n[cuda.0]\n[dwave]\ndaily_budget = "60s"\n',
    )
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "config also declares miner types gpu, qpu" in result.output
    assert "running cpu only" in result.output


def test_quip_miner_gpu_warns_even_on_flag_defaults_path(
    monkeypatch, tmp_path
):
    """`quip-miner gpu` against a cpu-only config runs GPU from flag
    defaults — the configured cpu sections are still dropped, so the
    warning fires on that path too."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cpu]\nnum_cpus = 2\n")
    result = CliRunner().invoke(quip_cli.quip_miner_gpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "config also declares miner types cpu" in result.output
    assert "running gpu only" in result.output


def test_quip_miner_qpu_warns_on_dropped_types(monkeypatch, tmp_path):
    """`quip-miner qpu` with a cpu+dwave config warns about the dropped
    cpu sections."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(
        tmp_path, '[cpu]\nnum_cpus = 1\n[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(quip_cli.quip_miner_qpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "config also declares miner types cpu" in result.output
    assert "running qpu only" in result.output


def test_quip_miner_cpu_no_warning_on_single_backend_config(
    monkeypatch, tmp_path
):
    """A cpu-only config invoked via `quip-miner cpu` drops nothing —
    no warning noise on the matched path."""
    _stub_runner(monkeypatch)
    cfg = _write_backend_toml(tmp_path, "[cpu]\nnum_cpus = 1\n")
    result = CliRunner().invoke(quip_cli.quip_miner_cpu, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "config also declares miner types" not in result.output


def test_bare_quip_miner_with_config_runs_supervisor(monkeypatch, tmp_path):
    """`quip-miner --config x.toml` (no subcommand) is the production
    entry: it hands off to the supervisor."""
    cfg = _write_supervisor_toml(tmp_path)
    called: Dict[str, Any] = {}
    monkeypatch.setattr(
        quip_cli, "_run_supervisor",
        lambda path, mode=None: called.setdefault("path", path) and 0 or 0,
    )
    result = CliRunner().invoke(quip_cli.quip_miner, ["--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert called["path"] == Path(cfg).expanduser()


def test_bare_quip_miner_without_config_shows_help():
    """Bare `quip-miner` (no subcommand, no --config) prints help, not an
    error traceback."""
    result = CliRunner().invoke(quip_cli.quip_miner, [])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


# ── telemetry aggregator: --config fallback ───────────────────────────


def _capture_telemetry_main(monkeypatch):
    """Stub substrate.telemetry_process.telemetry_main; return capture dict."""
    import substrate.telemetry_process as tp

    captured: Dict[str, Any] = {}

    def fake_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tp, "telemetry_main", fake_main)
    return captured


def test_telemetry_reads_rest_and_validators_from_config(monkeypatch, tmp_path):
    """`telemetry --config` picks up rest_host/rest_port/validators from
    the [miner] table — same file that configures the miner children."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\n'
        'validators = ["ws://a:9944", "ws://b:9944"]\n'
        'rest_host = "127.0.0.5"\n'
        'rest_port = 9001\n'
    )
    captured = _capture_telemetry_main(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["telemetry", "--snapshot-dir", str(tmp_path), "--config", str(cfg)],
    )
    assert result.exit_code == 0, result.output
    assert captured["listen_host"] == "127.0.0.5"
    assert captured["listen_port"] == 9001
    assert captured["validator_urls"] == ["ws://a:9944", "ws://b:9944"]


def test_telemetry_flags_beat_config(monkeypatch, tmp_path):
    """Explicit --rest-host/--rest-port/--validator override the TOML."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\n'
        'validators = ["ws://toml:9944"]\n'
        'rest_host = "127.0.0.5"\n'
        'rest_port = 9001\n'
    )
    captured = _capture_telemetry_main(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        [
            "telemetry", "--snapshot-dir", str(tmp_path), "--config", str(cfg),
            "--rest-host", "0.0.0.0", "--rest-port", "8099",
            "--validator", "ws://flag:9944",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["listen_host"] == "0.0.0.0"
    assert captured["listen_port"] == 8099
    assert captured["validator_urls"] == ["ws://flag:9944"]


def test_telemetry_disabled_rest_port_collapses_to_8086(monkeypatch, tmp_path):
    """A legacy `rest_port = -1` (children-disabled sentinel) must not
    break the aggregator — it serves on the 8086 default instead."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text('[miner]\nrest_port = -1\n')
    captured = _capture_telemetry_main(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["telemetry", "--snapshot-dir", str(tmp_path), "--config", str(cfg)],
    )
    assert result.exit_code == 0, result.output
    assert captured["listen_port"] == 8086
    assert captured["listen_host"] == "0.0.0.0"


def test_telemetry_defaults_without_config(monkeypatch, tmp_path):
    """No --config, no flags → the documented 0.0.0.0:8086 defaults."""
    captured = _capture_telemetry_main(monkeypatch)
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["telemetry", "--snapshot-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert captured["listen_host"] == "0.0.0.0"
    assert captured["listen_port"] == 8086
    assert captured["validator_urls"] == []


# ── resolve-mode / resolve-modes CLI subcommands ──────────────────────


def test_resolve_mode_dwave_only(monkeypatch, tmp_path):
    """Operator hands a `[dwave]`-only config to the CPU image: the
    entrypoint asks resolve-mode, gets `qpu`, and dispatches there
    (instead of the legacy hardcoded `cpu`)."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-mode", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "qpu"


def test_resolve_mode_unsupported_in_cpu_image(tmp_path):
    """CUDA section in a config given to a cpu-only image → error code
    `unsupported-mode`, non-zero exit, descriptive stderr the entrypoint
    surfaces to the operator."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cuda.0]\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["resolve-mode", "--config", str(cfg), "--image-supports", "cpu,qpu"],
    )
    assert result.exit_code != 0
    assert "unsupported-mode" in result.output


def test_resolve_mode_empty_config_with_default(tmp_path):
    """No backend sections + --default cpu → cpu. Models the bare
    `docker run` first-startup case where config.toml is still the
    template (only `[miner]` populated)."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text('[miner]\nvalidators = ["ws://a:9944"]\n')
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["resolve-mode", "--config", str(cfg), "--default", "cpu"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "cpu"


def test_resolve_mode_multi_group_single_caller_errors(tmp_path):
    """resolve-mode (singular) refuses multi-group configs so the rare
    one-mode caller fails loudly instead of dropping a backend."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 2\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-mode", "--config", str(cfg)]
    )
    assert result.exit_code != 0
    assert "multi-backend-not-single-mode" in result.output


def test_resolve_modes_multi_group_returns_all_active(tmp_path):
    """resolve-modes (plural) is what the entrypoint calls: emits one
    mode per line so bash mapfile builds the supervisor's child list."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 2\n'
        '[cuda.0]\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-modes", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines == ["cpu", "gpu", "qpu"]


def test_resolve_modes_single_group(tmp_path):
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n[cpu]\nnum_cpus = 4\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-modes", "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().splitlines() == ["cpu"]


def test_resolve_modes_image_supports_partial_unsupported_errors(tmp_path):
    """`[cpu]` + `[cuda.0]` on a cpu-only image: even though cpu is
    runnable, the operator's intent (run both) can't be honoured.
    Better to fail fast than to silently drop the gpu child."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 2\n'
        '[cuda.0]\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        [
            "resolve-modes", "--config", str(cfg),
            "--image-supports", "cpu,qpu",
        ],
    )
    assert result.exit_code != 0
    assert "unsupported-mode" in result.output
    assert "cuda" in result.output


def test_resolve_modes_no_config_argument(monkeypatch, tmp_path):
    """Without --config: backends={}. With --default, returns [default]."""
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-modes", "--default", "cpu"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "cpu"


def test_resolve_modes_no_config_no_default_errors():
    result = CliRunner().invoke(quip_cli.quip_miner, ["resolve-modes"])
    assert result.exit_code != 0
    assert "no-mode-resolvable" in result.output


# ── mode resolution has no mempool guard (T8) ─────────────────────────


def test_resolve_modes_multi_backend_cli_ok(tmp_path):
    """Multi-backend configs resolve unconditionally — the old mempool
    multi-backend guard is replaced by supervisor owner election."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\nmempool = true\n'
        '[cpu]\nnum_cpus = 2\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner, ["resolve-modes", "--config", str(cfg)],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().splitlines() == ["cpu", "qpu"]


def test_resolve_modes_mine_mode_flag_removed(tmp_path):
    """--mine-mode is gone along with the [miner] mode key — from both
    the plural and singular resolve commands."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n[cpu]\nnum_cpus = 2\n'
    )
    for command in ("resolve-modes", "resolve-mode"):
        result = CliRunner().invoke(
            quip_cli.quip_miner,
            [command, "--config", str(cfg), "--mine-mode", "mempool"],
        )
        assert result.exit_code != 0, command
        assert "no such option" in result.output.lower(), command


# ---------------------------------------------------------------------------
# Guard D — transparent self-registration (_ensure_registered_or_fail)
# ---------------------------------------------------------------------------


def test_guard_d_registers_when_absent(capsys):
    """First run: _ensure_registered reports a fresh registration."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_client = MagicMock()
    fake_keystore = MagicMock()
    fake_keystore.signer.ss58_address.return_value = "5Test"

    with patch.object(quip_cli, "_ensure_registered", AsyncMock(return_value=True)):
        asyncio.run(quip_cli._ensure_registered_or_fail(fake_client, fake_keystore))

    assert "registered miner: 5Test" in capsys.readouterr().out


def test_guard_d_already_registered_reports(capsys):
    """Subsequent run: already-registered is a no-op that reports state."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_client = MagicMock()
    fake_keystore = MagicMock()
    fake_keystore.signer.ss58_address.return_value = "5Test"

    with patch.object(quip_cli, "_ensure_registered", AsyncMock(return_value=False)):
        asyncio.run(quip_cli._ensure_registered_or_fail(fake_client, fake_keystore))

    assert "miner already registered: 5Test" in capsys.readouterr().out


def test_guard_d_registration_failure_raises(monkeypatch):
    """A registration that never verifies is fatal (miner-registration-failed)
    after the retries."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_client = MagicMock()
    fake_keystore = MagicMock()
    fake_keystore.signer.ss58_address.return_value = "5Test"

    # One attempt (no backoff sleep) so the always-failing register fails fast.
    monkeypatch.setattr(quip_cli, "_STARTUP_RETRY_ATTEMPTS", 1)
    boom = AsyncMock(side_effect=RuntimeError("register_miner failed: DispatchError"))
    with patch.object(quip_cli, "_ensure_registered", boom):
        with pytest.raises(
            click.ClickException,
            match=r"miner-registration-failed ss58=5Test error=RuntimeError: register_miner failed",
        ):
            asyncio.run(quip_cli._ensure_registered_or_fail(fake_client, fake_keystore))


def test_guard_d_registration_retries_then_succeeds(monkeypatch, capsys):
    """A transient registration failure is retried, then verifies on start."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(quip_cli.asyncio, "sleep", _no_sleep)
    fake_client = MagicMock()
    fake_keystore = MagicMock()
    fake_keystore.signer.ss58_address.return_value = "5Test"

    # First call raises (transient), second verifies as newly registered.
    flaky = AsyncMock(side_effect=[RuntimeError("nonce stale"), True])
    with patch.object(quip_cli, "_ensure_registered", flaky):
        asyncio.run(quip_cli._ensure_registered_or_fail(fake_client, fake_keystore))

    assert flaky.await_count == 2
    assert "registered miner: 5Test" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Guard D+ — non-fatal mempool solver registration
# ---------------------------------------------------------------------------


def _guard_dplus(monkeypatch, outcome_or_exc) -> bool:
    """Run _ensure_solver_or_disable_mempool with a stubbed helper."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    if isinstance(outcome_or_exc, Exception):
        stub = AsyncMock(side_effect=outcome_or_exc)
    else:
        stub = AsyncMock(return_value=outcome_or_exc)
    monkeypatch.setattr(quip_cli, "ensure_solver_registered", stub)
    fake_keystore = MagicMock()
    return asyncio.run(
        quip_cli._ensure_solver_or_disable_mempool(
            MagicMock(), fake_keystore, "cpu",
        )
    )


def test_guard_dplus_success_keeps_mempool_enabled(monkeypatch):
    from substrate.solver_registration import SolverGuardOutcome

    assert _guard_dplus(monkeypatch, SolverGuardOutcome.REGISTERED) is True
    assert (
        _guard_dplus(monkeypatch, SolverGuardOutcome.ALREADY_REGISTERED)
        is True
    )


def test_guard_dplus_failed_disables_mempool_without_raising(monkeypatch, capsys):
    """FAILED → loud log + mempool off for the run; pow proceeds (no raise —
    a fatal exit here would trip supervisor terminate-all-siblings)."""
    from substrate.solver_registration import SolverGuardOutcome

    assert _guard_dplus(monkeypatch, SolverGuardOutcome.FAILED) is False
    assert "mempool DISABLED" in capsys.readouterr().err


def test_guard_dplus_type_mismatch_disables_mempool(monkeypatch, capsys):
    from substrate.solver_registration import SolverGuardOutcome

    assert _guard_dplus(monkeypatch, SolverGuardOutcome.TYPE_MISMATCH) is False
    err = capsys.readouterr().err
    assert "TYPE_MISMATCH" in err
    assert "deregister-solver" in err


def test_guard_dplus_unexpected_exception_disables_mempool(monkeypatch, capsys):
    assert _guard_dplus(monkeypatch, RuntimeError("rpc down")) is False
    assert "mempool DISABLED" in capsys.readouterr().err


def test_startup_guards_return_effective_mempool(monkeypatch):
    """_run_startup_guards runs D+ only when mempool is enabled and
    propagates its verdict; C/D/E still run unconditionally."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from substrate.solver_registration import SolverGuardOutcome

    monkeypatch.setattr(
        quip_cli, "_ensure_funded_or_fail", AsyncMock(return_value=1)
    )
    monkeypatch.setattr(
        quip_cli, "_ensure_registered_or_fail", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        quip_cli, "_auto_identify", AsyncMock(return_value=None)
    )
    solver_guard = AsyncMock(return_value=SolverGuardOutcome.FAILED)
    monkeypatch.setattr(quip_cli, "ensure_solver_registered", solver_guard)

    common = dict(
        faucet_url=None,
        node_name=None,
        public_host=None,
        public_port=None,
        miner_config={},
    )
    enabled = asyncio.run(quip_cli._run_startup_guards(
        MagicMock(), MagicMock(),
        mempool_enabled=True, miner_kind="cpu", **common,
    ))
    assert enabled is False  # Guard D+ FAILED → mempool off, no raise
    assert solver_guard.await_count == 1

    enabled = asyncio.run(quip_cli._run_startup_guards(
        MagicMock(), MagicMock(),
        mempool_enabled=False, miner_kind="cpu", **common,
    ))
    assert enabled is False
    assert solver_guard.await_count == 1  # D+ skipped when mempool is off


# ---------------------------------------------------------------------------
# Chain-pull topology (replaces --topology on the live path)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from dwave_topologies.topologies.json_loader import topology_from_nodes_edges  # noqa: E402


def test_topology_from_nodes_edges_builds_minimal_object():
    """The chain-built topology exposes the nodes/edges/solver surface
    consumers read, identical to a file-loaded topology."""
    topo = topology_from_nodes_edges([2, 0, 1], [(0, 1), (1, 2)], "SolverX")
    assert topo.solver_name == "SolverX"
    assert topo.num_nodes == 3
    assert topo.num_edges == 2
    assert topo.nodes == [2, 0, 1]            # order preserved (hash canonicalizes)
    assert topo.edges == [(0, 1), (1, 2)]     # tuples, as consumers expect


def test_solver_name_from_config_reads_qpu_device():
    assert quip_cli._solver_name_from_config(
        {"qpu": {"devices": [{"type": "dwave", "solver": "Advantage2_system1"}]}}
    ) == "Advantage2_system1"


def test_solver_name_from_config_none_for_cpu():
    assert quip_cli._solver_name_from_config({"cpu": {"num_cpus": 4}}) is None
    assert quip_cli._solver_name_from_config({}) is None


def test_run_concurrent_miner_rejects_topology_on_live_path():
    """--topology is tools-only; passing it to live mining errors (exit 2)
    before any chain connection is attempted."""
    code = asyncio.run(quip_cli._run_concurrent_miner(
        miner_kind="cpu",
        validators=("ws://unused:9944",),
        signer_key_path="/nonexistent",
        faucet_url=None,
        rest_port=0,
        rest_host="127.0.0.1",
        topology_spec="advantage2_system1",   # offending flag
        miner_config={"cpu": {"num_cpus": 1}},
    ))
    assert code == 2


def test_run_concurrent_miner_still_passes_topology_to_controllers():
    """Guard the chain-pull wiring: the topology built from the chain is still
    threaded to the controllers (keyword `topology=topology`)."""
    import inspect
    src = inspect.getsource(quip_cli._run_concurrent_miner)
    assert "_topology_from_chain(" in src
    assert "topology=topology" in src
