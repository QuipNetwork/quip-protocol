"""End-to-end tests for the `quip-miner identify` CLI command.

The CLI is exercised through Click's CliRunner; the substrate connection
layer is monkeypatched with stubs so we don't need a live validator. The
goal is to cover the surface that's hardest to verify with pure unit
tests of `shared.system_info`: argument plumbing, --dry-run vs submit
branches, the `MinerRegistry.set_descriptor` submission path, and the
descriptor-invalid error code emitted to stderr.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from shared.keystore_hybrid import generate
from substrate.types import ExtrinsicReceipt


def _keystore(tmp_path: Path) -> Path:
    """Create a hybrid keystore at tmp_path/sk.json and return the path."""
    path = tmp_path / "sk.json"
    generate(path)
    return path


def _cpu_miner_toml(tmp_path: Path, *, num_cpus: int = 1) -> Path:
    path = tmp_path / "miner.toml"
    path.write_text(f"[cpu]\nnum_cpus = {num_cpus}\n")
    return path


# ----------------------------------------------------------------------
# --dry-run path
# ----------------------------------------------------------------------


def test_dry_run_emits_canonical_json_and_zero_exit(tmp_path):
    """--dry-run prints the canonical JSON to stdout (no chain access)."""
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--signer-key", str(keystore),
            "--node-name", "rig-01",
            "--rpc-endpoint", "ws://rig-01.example.com:9944",
            "--public-host", "rig-01.example.com",
            "--public-port", "20049",
            "--auto-mine",
            "--no-system-info",
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    body = json.loads(res.output.splitlines()[0])
    assert body["schema"] == "quip.node_descriptor.v1"
    assert body["descriptor_version"] == 1
    assert body["node_name"] == "rig-01"
    assert body["public_host"] == "rig-01.example.com"
    assert body["public_port"] == 20049
    assert body["rpc_endpoints"] == ["ws://rig-01.example.com:9944"]
    assert body["auto_mine"] is True
    # Footer carries account + size for operator inspection.
    assert "payload_size_bytes" in res.output


def test_dry_run_with_miner_config_populates_miners_block(tmp_path):
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    miner_cfg = _cpu_miner_toml(tmp_path, num_cpus=2)
    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--signer-key", str(keystore),
            "--node-name", "cpu-rig",
            "--miner-config", str(miner_cfg),
            "--no-system-info",
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    body = json.loads(res.output.splitlines()[0])
    assert body["miners"] == {
        "cpu": {"kind": "CPU", "miner_id": "cpu-rig-CPU-1", "num_cpus": 2}
    }


def test_dry_run_rejects_oversized_node_name(tmp_path):
    """A 65-byte node_name fails validation BEFORE any submit attempt."""
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--signer-key", str(keystore),
            "--node-name", "x" * 65,
            "--no-system-info",
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code != 0
    assert "descriptor-invalid" in res.output
    assert "node_name" in res.output


# ----------------------------------------------------------------------
# Submit path — MinerRegistry.set_descriptor
# ----------------------------------------------------------------------


class _StubClient:
    """Minimal SubstrateClient stand-in: tracks calls, returns scripted receipts."""

    def __init__(self):
        self.calls: list = []

    async def submit_extrinsic(self, module, function, params, signer, wait_for):  # noqa: ARG002
        self.calls.append((module, function, params, wait_for))
        return ExtrinsicReceipt(
            extrinsic_hash=f"0x{function}-hash",
            block_hash="0xblockhash",
            is_finalized=False,
        )


def _patch_pool_and_client(monkeypatch, client):
    """Wire `quip_cli.SubstrateClient` to return our stub.

    After the pool.get("rpc") removal, ``identify`` calls
    ``SubstrateClient(urls=...)`` directly via ``_connect_or_fail``. The
    stub client has its own ``connect()`` / ``close()`` to satisfy that
    contract; everything else mirrors the original ``_StubClient``.
    """
    import quip_cli

    async def _connect():
        return None

    async def _close():
        return None

    client.connect = _connect
    client.close = _close
    monkeypatch.setattr(quip_cli, "SubstrateClient", lambda urls: client)


def test_submit_posts_miner_registry_descriptor(tmp_path, monkeypatch):
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    miner_cfg = _cpu_miner_toml(tmp_path)
    client = _StubClient()
    _patch_pool_and_client(monkeypatch, client)

    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--validator", "ws://stub:9944",
            "--signer-key", str(keystore),
            "--node-name", "rig-01",
            "--miner-config", str(miner_cfg),
            "--no-system-info",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert len(client.calls) == 1
    module, function, params, _wait = client.calls[0]
    assert (module, function) == ("MinerRegistry", "set_descriptor")
    body = params["descriptor"]["V1"]
    assert body["node_id"] == b"rig-01"
    assert body["node_name"] == b"rig-01"
    assert body["miners"][0]["kind"] == "Cpu"
    assert "MinerRegistry.set_descriptor" in res.output
