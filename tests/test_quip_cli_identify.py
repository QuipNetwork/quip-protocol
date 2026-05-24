"""End-to-end tests for the `quip-miner identify` CLI command.

The CLI is exercised through Click's CliRunner; the substrate connection
layer is monkeypatched with stubs so we don't need a live validator. The
goal is to cover the surface that's hardest to verify with pure unit
tests of `shared.system_info`: argument plumbing, --dry-run vs submit
branches, the `remark_with_event` → `remark` fallback, and the
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
    # Footer carries account + size + hash for operator inspection.
    assert "payload_hash" in res.output
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
# Submit path — remark_with_event preferred, falls back to remark
# ----------------------------------------------------------------------


class _StubClient:
    """Minimal SubstrateClient stand-in: tracks calls, returns scripted receipts."""

    def __init__(self, *, has_remark_with_event: bool, fail_first_with_event: bool = False):
        self._has_event = has_remark_with_event
        self._fail_event = fail_first_with_event
        self.calls: list = []

    async def has_call(self, module: str, function: str) -> bool:
        return module == "System" and function == "remark_with_event" and self._has_event

    async def submit_extrinsic(self, module, function, params, signer, wait_for):  # noqa: ARG002
        self.calls.append((module, function, params, wait_for))
        if function == "remark_with_event" and self._fail_event:
            raise RuntimeError("simulated runtime mismatch")
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


def test_submit_prefers_remark_with_event(tmp_path, monkeypatch):
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    client = _StubClient(has_remark_with_event=True)
    _patch_pool_and_client(monkeypatch, client)

    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--validator", "ws://stub:9944",
            "--signer-key", str(keystore),
            "--node-name", "rig-01",
            "--no-system-info",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    # Exactly one extrinsic submitted; chose the eventful variant.
    assert len(client.calls) == 1
    module, function, params, _wait = client.calls[0]
    assert (module, function) == ("System", "remark_with_event")
    # Payload is bytes (canonical JSON, not a string)
    assert isinstance(params["remark"], bytes)
    body = json.loads(params["remark"])
    assert body["schema"] == "quip.node_descriptor.v1"
    # Output surfaces which call we used.
    assert "System.remark_with_event" in res.output


def test_submit_falls_back_to_plain_remark(tmp_path, monkeypatch):
    """When metadata says `remark_with_event` is missing, we go straight
    to `remark` without trying the eventful variant first."""
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    client = _StubClient(has_remark_with_event=False)
    _patch_pool_and_client(monkeypatch, client)

    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--validator", "ws://stub:9944",
            "--signer-key", str(keystore),
            "--node-name", "rig-01",
            "--no-system-info",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert len(client.calls) == 1
    assert client.calls[0][1] == "remark"
    assert "System.remark" in res.output


def test_submit_retries_with_plain_remark_when_event_call_throws(tmp_path, monkeypatch):
    """`has_call` says yes, but the actual submit_extrinsic throws — we
    degrade to plain remark rather than crash. This guards against
    metadata caches that report stale runtime info."""
    from quip_cli import quip_miner

    keystore = _keystore(tmp_path)
    client = _StubClient(has_remark_with_event=True, fail_first_with_event=True)
    _patch_pool_and_client(monkeypatch, client)

    runner = CliRunner()
    res = runner.invoke(
        quip_miner,
        [
            "identify",
            "--validator", "ws://stub:9944",
            "--signer-key", str(keystore),
            "--node-name", "rig-01",
            "--no-system-info",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    # Two calls: first remark_with_event (fails), then remark (succeeds).
    assert [c[1] for c in client.calls] == ["remark_with_event", "remark"]
    assert "System.remark" in res.output
