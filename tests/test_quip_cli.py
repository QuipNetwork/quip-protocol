"""Unit tests for quip_cli helpers: topology parsing, injection, hashing, and subcommands."""
from __future__ import annotations

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

        async def submit_extrinsic(self, _module, _call, _args, _signer, **_kw):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Priority is too low")  # stale-nonce shape
            return FakeReceipt()

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
    body = params["descriptor"]["V1"]
    # BoundedVec fields are 1-tuple-wrapped for the runtime composite shape
    # (see test_miner_registry for the rationale).
    assert body["node_id"] == (b"test-rig",)
    assert body["node_name"] == (b"test-rig",)
    (miners,) = body["miners"]
    assert miners[0]["kind"] == "Cpu"


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

    class FakeReceipt:
        error = None
        extrinsic_hash = "0xabc"
        block_hash = "0xdef"

    class FakeClient:
        async def submit_extrinsic(self, *_a, **_kw):
            return FakeReceipt()

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccountId00000000000000000000000000000000000"
        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    asyncio.run(quip_cli._auto_identify(
        FakeClient(),
        FakeKeystore(),
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

    class FakeReceipt:
        error = None
        extrinsic_hash = "0xabc"
        block_hash = "0xdef"

    class FakeClient:
        async def submit_extrinsic(self, *_a, **_kw):
            return FakeReceipt()

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccountId00000000000000000000000000000000000"
        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    asyncio.run(quip_cli._auto_identify(
        FakeClient(),
        FakeKeystore(),
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

    class FakeReceipt:
        error = None
        extrinsic_hash = "0xabc"
        block_hash = "0xdef"

    class FakeClient:
        async def submit_extrinsic(self, module, call, args, _signer, **_kw):
            captured["module"] = module
            captured["call"] = call
            captured["params"] = args
            return FakeReceipt()

    class FakeSigner:
        def ss58_address(self):
            return "5FakeAccount" + "0" * 38
        def account_id_bytes(self):
            return b"\x00" * 32

    class FakeKeystore:
        signer = FakeSigner()

    async def _no_probe():
        return None
    monkeypatch.setattr(quip_cli, "_detect_public_ip", _no_probe)

    asyncio.run(quip_cli._auto_identify(
        FakeClient(),
        FakeKeystore(),
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
    (miners,) = params["descriptor"]["V1"]["miners"]
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


# ── --mine-mode guard (W4a) ───────────────────────────────────────────


def test_resolve_modes_mempool_multi_backend_cli_errors(tmp_path):
    """End-to-end via the CLI: --mine-mode mempool + multi-backend
    config → non-zero exit with the kebab-case error code on stderr
    for the entrypoint to grep."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 2\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["resolve-modes", "--config", str(cfg), "--mine-mode", "mempool"],
    )
    assert result.exit_code != 0
    assert "multi-backend-not-allowed-in-mempool-mode" in result.output


def test_resolve_modes_pow_multi_backend_cli_ok(tmp_path):
    """--mine-mode pow with multi-backend → 2 modes, no error.
    Mirror of the test above with the safe mine-mode value."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[cpu]\nnum_cpus = 2\n'
        '[dwave]\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["resolve-modes", "--config", str(cfg), "--mine-mode", "pow"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().splitlines() == ["cpu", "qpu"]


def test_resolve_modes_mempool_single_backend_cli_ok(tmp_path):
    """Single backend + mempool is the supported path — no error."""
    cfg = tmp_path / "miner.toml"
    cfg.write_text(
        '[miner]\nvalidators = ["ws://a:9944"]\n'
        '[dwave]\ndaily_budget = "60s"\n'
    )
    result = CliRunner().invoke(
        quip_cli.quip_miner,
        ["resolve-modes", "--config", str(cfg), "--mine-mode", "mempool"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "qpu"


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
