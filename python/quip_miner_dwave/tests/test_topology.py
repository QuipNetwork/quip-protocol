"""Native topology hash tests."""
from quip_miner_dwave.topology import native_topology_hash


def test_hash_deterministic_and_order_insensitive_for_edges():
    h1 = native_topology_hash([0, 1, 2], [(0, 1), (1, 2)])
    h2 = native_topology_hash([2, 0, 1], [(1, 2), (0, 1)])
    assert h1 == h2
    assert len(h1) == 32
    assert h1 != native_topology_hash([0, 1], [(0, 1)])
