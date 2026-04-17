# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""End-to-end scrubbing test.

Build a ``Node`` with intentionally sensitive values (secret,
peer list, QPU tokens) and assert that none of those strings
appear in:

    * ``node.info().to_json()``
    * the cached ``NodeDescriptor`` served on JOIN/STATUS/STATS
    * the ``nodes.json`` telemetry file produced by ``TelemetryManager``

This is the regression guard for the ``miner_type`` leak and
for any future whitelist mistake.
"""

import json

import pytest

from shared.node import Node
from shared.telemetry import TelemetryManager


_SECRETS = [
    "TOPSECRET-SEED",
    "dwave-ABCDEF-NEVER-SHIP",
    "ibm-ZZZ-PRIVATE",
    "10.9.8.7:65000",            # planted peer address
    "192.168.77.33",             # planted internal listen IP
    "/etc/quip/private-key.pem", # planted TLS private-key path
]


@pytest.fixture
def sensitive_miners_config():
    return {
        # Top-level "[global]" flattened keys (as the CLI produces)
        "node_name": "scrub-test",
        "secret": "TOPSECRET-SEED",
        "listen": "192.168.77.33",
        "port": 20049,
        "public_host": "public.example.com",
        "public_port": 8085,
        "peer": ["10.9.8.7:65000"],
        "heartbeat_interval": 15,
        "auto_mine": True,
        "log_level": "INFO",
        "tls_cert_file": "/etc/quip/cert.pem",
        "tls_key_file": "/etc/quip/private-key.pem",
        # Miner sections
        "cpu": {"num_cpus": 1},
        "dwave": {
            "solver": "Advantage_system6.4",
            "token": "dwave-ABCDEF-NEVER-SHIP",
            "daily_budget": "60s",
            "dwave_region_url": "https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        },
        "ibm": {"token": "ibm-ZZZ-PRIVATE"},
    }


@pytest.fixture
def node_with_secrets(sensitive_miners_config):
    from quip_cli import load_genesis_block
    genesis = load_genesis_block("genesis_block_public.json")
    node = Node(
        node_id="scrub-test",
        miners_config=sensitive_miners_config,
        genesis_block=genesis,
        secret="dev-seed",
    )
    try:
        yield node
    finally:
        # Without this, the QueueListener thread and MinerHandle child
        # processes leak and block pytest from exiting at session end.
        node.close()


def _assert_no_secrets(blob: str):
    for s in _SECRETS:
        assert s not in blob, f"LEAK: {s!r} in {blob[:200]}…"


def test_miner_type_is_short_label(node_with_secrets):
    info = node_with_secrets.info()
    assert info.miner_type in {
        "CPU", "GPU", "QPU", "CPU+GPU", "CPU+QPU", "GPU+QPU", "CPU+GPU+QPU",
    }
    _assert_no_secrets(info.miner_type)


def test_info_to_json_has_no_secrets(node_with_secrets):
    _assert_no_secrets(node_with_secrets.info().to_json())


def test_descriptor_has_no_secrets(node_with_secrets):
    desc = node_with_secrets.descriptor()
    _assert_no_secrets(json.dumps(desc))


def test_descriptor_whitelisted_globals_only(node_with_secrets):
    desc = node_with_secrets.descriptor()
    top_level = set(desc.keys())
    allowed = {
        "descriptor_version", "node_name", "public_host", "public_port",
        "auto_mine", "log_level", "runtime", "miners", "system_info",
    }
    assert top_level <= allowed, f"unexpected top-level keys: {top_level - allowed}"


def test_dwave_descriptor_fields(node_with_secrets):
    desc = node_with_secrets.descriptor()
    dwave = desc["miners"]["dwave"]
    assert dwave["provider"] == "dwave"
    assert dwave["solver"] == "Advantage_system6.4"
    assert dwave["daily_budget"] == "60s"
    assert "token" not in dwave
    assert "dwave_region_url" not in dwave


def test_telemetry_nodes_json_has_no_secrets(tmp_path, node_with_secrets):
    tele = TelemetryManager(telemetry_dir=str(tmp_path), enabled=True)
    tele.update_node(
        "peer.example.com:8085", "active",
        miner_info=node_with_secrets.info(),
        descriptor=node_with_secrets.descriptor(),
    )
    nodes_json = (tmp_path / "nodes.json").read_text()
    _assert_no_secrets(nodes_json)
    parsed = json.loads(nodes_json)
    rec = parsed["nodes"]["peer.example.com:8085"]
    # Descriptor propagated
    assert rec["descriptor"]["descriptor_version"] >= 1
    assert rec["miner_type"] in {"CPU+QPU", "CPU", "QPU"}
