"""
Zephyr Z(8, 2) topology definition.

This is a generic Zephyr(8, 2) graph with 1,088 nodes and 6,068 edges.
Precomputed embedding available for Advantage2-System1.6.
"""

from typing import List, Tuple
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON (auto-detects .gz compression)
_json_topology = load_json_topology('zephyr_z8_t2.json')

# Topology object (new type system)
ZEPHYR_Z8_T2_TOPOLOGY = _json_topology

# Legacy dictionary format (backward compatibility)
ZEPHYR_Z8_T2 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = ZEPHYR_Z8_T2['nodes']
EDGES: List[Tuple[int, int]] = ZEPHYR_Z8_T2['edges']
NUM_NODES: int = ZEPHYR_Z8_T2['num_nodes']
NUM_EDGES: int = ZEPHYR_Z8_T2['num_edges']

__all__ = [
    'ZEPHYR_Z8_T2_TOPOLOGY',
    'ZEPHYR_Z8_T2',
    'NODES',
    'EDGES',
    'NUM_NODES',
    'NUM_EDGES',
]
