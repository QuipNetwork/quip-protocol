"""Real-QPU connection must be deferred until the coordinator engages."""

import pytest

pytest.importorskip("numpy")  # OceanSampler imports numpy at module load

from quip_miner_dwave.ocean import OceanSampler  # noqa: E402


def test_real_mode_does_not_connect_at_construction():
    """Constructing a real (non-mock) sampler opens no QPU session; the
    connection waits for the coordinator's Configure via ensure_connected."""
    s = OceanSampler(mock=False)
    assert s.sampler is None
    assert s.native_topology_hash is None
    assert not s._connected
    assert callable(s.ensure_connected)
