"""Tests for Node.descriptor() fallback behavior.

The descriptor probes hardware via ``build_descriptor`` which can fail on
minimal containers, sandboxed environments, or unusual platforms. A
failure here previously poisoned every caller that depended on it
(STATUS_REQUEST, listener-snapshot serializer, telemetry register),
producing a silent-drop / blank-cache symptom. Verify the fallback path
returns a JSON-friendly dict and caches it.
"""
import logging

import pytest

from shared import node as node_module
from shared.node import Node


def _bare_node(node_id: str = "n1") -> Node:
    n = object.__new__(Node)
    n.node_id = node_id
    n.node_name = "rig-01"
    n.miners_config = {"cpu": {"num_cpus": 1}}
    n._descriptor_cache = None
    n.logger = logging.getLogger("test.node")
    return n


def test_descriptor_returns_real_when_build_succeeds():
    """Sanity: when build_descriptor works, callers get the real dict."""
    n = _bare_node()
    desc = n.descriptor()
    assert isinstance(desc, dict)
    assert desc.get("node_id") == "n1" or desc.get("node_name")
    # Real descriptor has a descriptor_version field; fallback does not.
    assert "descriptor_version" in desc


def test_descriptor_returns_fallback_when_build_raises(monkeypatch):
    """When build_descriptor raises, descriptor() returns a fallback dict."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("hardware probe blew up")

    monkeypatch.setattr(node_module, "build_descriptor", _boom)
    n = _bare_node()
    desc = n.descriptor()
    assert isinstance(desc, dict)
    assert desc["available"] is False
    assert "hardware probe blew up" in desc["error"]
    assert desc["node_id"] == "n1"
    assert desc["node_name"] == "rig-01"


def test_descriptor_fallback_is_cached(monkeypatch):
    """Fallback is cached so a persistent failure doesn't re-probe each call."""
    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("probe failed")

    monkeypatch.setattr(node_module, "build_descriptor", _boom)
    n = _bare_node()
    n.descriptor()
    n.descriptor()
    n.descriptor()
    assert calls["n"] == 1


def test_descriptor_fallback_is_json_serializable(monkeypatch):
    """Callers serialize descriptor() into status JSON; fallback must work."""
    import json

    monkeypatch.setattr(
        node_module, "build_descriptor",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("no /proc")),
    )
    n = _bare_node()
    desc = n.descriptor()
    # Must round-trip cleanly — STATUS_REQUEST handler does this.
    assert json.loads(json.dumps(desc)) == desc


def test_invalidate_then_retry(monkeypatch):
    """invalidate_descriptor clears the fallback so the next call retries."""
    state = {"fail": True}

    def _maybe(*_args, **_kwargs):
        if state["fail"]:
            raise RuntimeError("transient")

        class _Desc:
            def to_dict(self):
                return {"descriptor_version": 1, "node_name": "rig-01"}
        return _Desc()

    monkeypatch.setattr(node_module, "build_descriptor", _maybe)
    n = _bare_node()
    first = n.descriptor()
    assert first["available"] is False

    state["fail"] = False
    n.invalidate_descriptor()
    second = n.descriptor()
    assert second.get("available") is not False
    assert second["descriptor_version"] == 1
