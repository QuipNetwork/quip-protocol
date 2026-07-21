import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from quip_proto import session  # noqa: E402
from quip_proto import miner_pb2  # noqa: E402


def test_hello_requires_token(monkeypatch):
    monkeypatch.delenv("QUIP_SESSION_TOKEN", raising=False)
    with pytest.raises(session.MissingToken):
        session.build_hello("qpu-0", "dwave-qpu", "quantum-anneal", [miner_pb2.ISING_SAMPLE])
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "tok-abc")
    h = session.build_hello("qpu-0", "dwave-qpu", "quantum-anneal", [miner_pb2.ISING_SAMPLE])
    assert h.session_token == "tok-abc"
    assert h.protocol_version == 1


def test_configure_defaults():
    c = miner_pb2.Configure()  # all zero
    cfg = session.session_config_from_configure("qpu-0", c)
    assert (cfg.queue_depth, cfg.idle_timeout_s, cfg.heartbeat_s, cfg.reconnect_window_s) == (3, 300, 15, 60)
