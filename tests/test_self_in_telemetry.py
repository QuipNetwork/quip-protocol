"""Tests for the ``_register_self_in_telemetry`` helper on NetworkNode.

The node registers itself in ``TelemetryManager`` under ``public_host``
so that the local node appears in ``nodes.json`` alongside remote peers.
The helper must be resilient: telemetry failures are logged, not raised.
"""

import json
import time

from unittest.mock import MagicMock

import pytest

from shared.telemetry import TelemetryManager


def _make_shell(public_host: str, telemetry_dir, descriptor=None):
    """Build a minimal NetworkNode with only the attributes the helper reads."""
    from shared.network_node import NetworkNode

    node = object.__new__(NetworkNode)
    node.public_host = public_host
    node.logger = MagicMock()
    node.telemetry = TelemetryManager(
        telemetry_dir=str(telemetry_dir),
        enabled=True,
        logger=node.logger,
    )
    node.descriptor = lambda: descriptor or {"node_name": "n1"}
    return node


def _read_nodes_json(telemetry_dir):
    return json.loads((telemetry_dir / "nodes.json").read_text())


def test_register_self_writes_record_under_public_host(tmp_path):
    node = _make_shell(
        "1.2.3.4:20049", tmp_path, descriptor={"node_name": "alpha"},
    )

    node._register_self_in_telemetry()

    data = _read_nodes_json(tmp_path)
    assert "1.2.3.4:20049" in data["nodes"]
    entry = data["nodes"]["1.2.3.4:20049"]
    assert entry["status"] == "active"
    assert entry["address"] == "1.2.3.4:20049"
    assert entry["node_name"] == "alpha"
    assert data["active_count"] == 1


def test_register_self_sets_last_heartbeat(tmp_path):
    node = _make_shell("127.0.0.1:20049", tmp_path)

    node._register_self_in_telemetry()

    entry = _read_nodes_json(tmp_path)["nodes"]["127.0.0.1:20049"]
    assert entry["last_heartbeat"] is not None
    assert entry["last_heartbeat"] > 0


def test_register_self_refresh_updates_last_seen(tmp_path):
    node = _make_shell("127.0.0.1:20049", tmp_path)

    node._register_self_in_telemetry()
    first = _read_nodes_json(tmp_path)["nodes"]["127.0.0.1:20049"]["last_seen"]

    time.sleep(0.02)
    node._register_self_in_telemetry()
    second = _read_nodes_json(tmp_path)["nodes"]["127.0.0.1:20049"]["last_seen"]

    assert second > first


def test_register_self_swallows_telemetry_errors(tmp_path):
    node = _make_shell("127.0.0.1:20049", tmp_path)
    node.telemetry = MagicMock()
    node.telemetry.update_node.side_effect = RuntimeError("disk full")

    node._register_self_in_telemetry()

    node.logger.warning.assert_called_once()
    args, kwargs = node.logger.warning.call_args
    assert "Failed to register self in telemetry" in args[0]
    assert kwargs.get("exc_info") is True


def test_register_self_embeds_descriptor(tmp_path):
    node = _make_shell(
        "10.0.0.1:20049",
        tmp_path,
        descriptor={"node_name": "beta", "public_host": "10.0.0.1", "auto_mine": True},
    )

    node._register_self_in_telemetry()

    entry = _read_nodes_json(tmp_path)["nodes"]["10.0.0.1:20049"]
    assert entry["node_name"] == "beta"
    assert entry["public_host"] == "10.0.0.1"
    assert entry["auto_mine"] is True
