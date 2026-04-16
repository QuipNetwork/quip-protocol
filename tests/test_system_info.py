# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for shared/system_info.py.

Covers: hardware probe shape, descriptor assembly, whitelist
filtering, and the ``_FORBIDDEN_SUBSTRINGS`` defense-in-depth pass.
"""

import json

from shared.system_info import (
    DESCRIPTOR_VERSION,
    GPUInfo,
    build_descriptor,
    collect_system_info,
    descriptor_summary_line,
    override_public_address,
    summarize_miners,
)


def test_collect_system_info_shape():
    si = collect_system_info()
    assert si.os["system"]
    assert si.os["machine"]
    assert isinstance(si.cpu.logical_cores, int) and si.cpu.logical_cores >= 1
    assert isinstance(si.cpu.brand, str) and si.cpu.brand
    assert isinstance(si.gpus, list)
    for gpu in si.gpus:
        assert isinstance(gpu, GPUInfo)
        assert gpu.vendor in {"NVIDIA", "Apple", "AMD"}
        assert gpu.name


def test_descriptor_version_is_current():
    desc = build_descriptor("n1", {"cpu": {"num_cpus": 1}})
    d = desc.to_dict()
    assert d["descriptor_version"] == DESCRIPTOR_VERSION


def test_cpu_only_descriptor_shape():
    miners_config = {
        "node_name": "rig-01",
        "auto_mine": True,
        "log_level": "INFO",
        "cpu": {"num_cpus": 4},
    }
    desc = build_descriptor("rig-01", miners_config).to_dict()
    assert desc["node_name"] == "rig-01"
    assert desc["auto_mine"] is True
    assert desc["miners"] == {
        "cpu": {"kind": "CPU", "miner_id": "rig-01-CPU-1", "num_cpus": 4},
    }


def test_cuda_device_ordinal_preserved():
    miners_config = {
        "cpu": {"num_cpus": 1},
        "cuda": {
            "0": {"utilization": 100, "yielding": False, "enabled": True},
            "1": {"utilization": 50, "yielding": True, "enabled": True},
        },
    }
    miners = build_descriptor("n1", miners_config).to_dict()["miners"]
    assert "cuda.0" in miners and "cuda.1" in miners
    assert miners["cuda.0"]["utilization"] == 100
    assert miners["cuda.0"]["yielding"] is False
    assert miners["cuda.1"]["utilization"] == 50
    assert miners["cuda.1"]["yielding"] is True
    assert miners["cuda.0"]["device_index"] == 0
    assert miners["cuda.1"]["device_index"] == 1


def test_metal_and_modal_keys():
    miners_config = {
        "metal": {"utilization": 30, "yielding": True},
        "modal": {"gpu_type": "a10g"},
    }
    miners = build_descriptor("n1", miners_config).to_dict()["miners"]
    assert "metal.0" in miners
    assert miners["metal.0"]["backend"] == "metal"
    assert miners["metal.0"]["utilization"] == 30
    assert "modal.0" in miners
    assert miners["modal.0"]["backend"] == "modal"
    assert miners["modal.0"]["gpu_type"] == "a10g"


def test_qpu_providers_reported_by_name():
    miners_config = {
        "dwave": {"solver": "Advantage_system6.4", "daily_budget": "60s"},
        "ibm": {},
    }
    miners = build_descriptor("n1", miners_config).to_dict()["miners"]
    assert "dwave" in miners and "ibm" in miners
    assert miners["dwave"]["provider"] == "dwave"
    assert miners["dwave"]["solver"] == "Advantage_system6.4"
    assert miners["ibm"]["provider"] == "ibm"


def test_whitelist_drops_internal_global_fields():
    miners_config = {
        "node_name": "rig-01",
        "listen": "0.0.0.0",
        "port": 20049,
        "peer": ["10.0.0.99:8085"],
        "heartbeat_interval": 15,
        "cpu": {"num_cpus": 1},
    }
    desc = build_descriptor("rig-01", miners_config).to_dict()
    top = {k for k in desc if k not in {"descriptor_version", "runtime",
                                        "miners", "system_info"}}
    # Only whitelisted global fields should appear as top-level keys
    assert top <= {
        "node_name", "public_host", "public_port", "auto_mine", "log_level",
    }
    assert "listen" not in desc
    assert "port" not in desc
    assert "peer" not in desc
    assert "heartbeat_interval" not in desc


def test_secret_substring_blocks_even_if_whitelisted():
    """A key containing 'secret' must not appear even if someone injects it."""
    miners_config = {
        "secret": "TOPSECRET",
        "cpu": {"num_cpus": 1, "secret_tuning": 99},
        "cuda": {"0": {"utilization": 100, "auth_token": "sk-x"}},
    }
    serialized = json.dumps(build_descriptor("n1", miners_config).to_dict())
    for bad in ("TOPSECRET", "sk-x", "secret_tuning", "auth_token"):
        assert bad not in serialized, f"LEAK: {bad!r}"


def test_qpu_token_never_reported():
    miners_config = {
        "dwave": {
            "solver": "Advantage_system6.4",
            "token": "dwave-SECRET",
            "daily_budget": "60s",
            "dwave_region_url": "https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        },
        "ibm": {"token": "ibm-SECRET"},
    }
    dwave = build_descriptor("n1", miners_config).to_dict()["miners"]["dwave"]
    assert "token" not in dwave
    assert "dwave_region_url" not in dwave
    assert dwave["solver"] == "Advantage_system6.4"
    serialized = json.dumps(build_descriptor("n1", miners_config).to_dict())
    assert "dwave-SECRET" not in serialized
    assert "ibm-SECRET" not in serialized


def test_docker_image_from_env(monkeypatch):
    monkeypatch.setenv("QUIP_DOCKER_IMAGE", "my.registry/quip:v9")
    desc = build_descriptor("n1", {"cpu": {"num_cpus": 1}}).to_dict()
    assert desc["runtime"]["docker_image"] == "my.registry/quip:v9"


def test_docker_image_absent_when_env_unset(monkeypatch):
    monkeypatch.delenv("QUIP_DOCKER_IMAGE", raising=False)
    desc = build_descriptor("n1", {"cpu": {"num_cpus": 1}}).to_dict()
    assert desc["runtime"]["docker_image"] is None


def test_descriptor_summary_line_formatting():
    desc = build_descriptor("n1", {"cpu": {"num_cpus": 2}})
    line = descriptor_summary_line(desc)
    assert desc.system_info.os["system"] in line
    assert "C" in line  # core count


def test_summarize_miners_skips_missing_sections():
    assert summarize_miners("n1", {}) == {}


def test_public_host_whitelist_passed_through():
    miners_config = {
        "public_host": "announce.example.com",
        "public_port": 12345,
        "cpu": {"num_cpus": 1},
    }
    desc = build_descriptor("n1", miners_config).to_dict()
    assert desc["public_host"] == "announce.example.com"
    assert desc["public_port"] == 12345


def test_override_public_address_replaces_host_port():
    desc = {"public_host": "liar.example.com", "public_port": 1, "foo": 42}
    patched = override_public_address(desc, "203.0.113.5:20049")
    assert patched["public_host"] == "203.0.113.5"
    assert patched["public_port"] == 20049
    assert patched["foo"] == 42
    # original not mutated
    assert desc["public_host"] == "liar.example.com"


def test_override_public_address_passthrough_on_none():
    assert override_public_address(None, "1.2.3.4:5") is None
    assert override_public_address({"a": 1}, None) == {"a": 1}
    assert override_public_address({"a": 1}, "") == {"a": 1}


def test_override_public_address_preserves_dns_when_that_is_validated():
    """If the validated address *is* a DNS name, the descriptor keeps it.

    _validate_peer_address returns the claimed DNS name when reachable,
    so passing it through override_public_address must yield the DNS form
    rather than silently converting to IP.
    """
    desc = {"public_host": "originally.wrong.example.com", "public_port": 1}
    patched = override_public_address(desc, "announce.example.com:20049")
    assert patched["public_host"] == "announce.example.com"
    assert patched["public_port"] == 20049
