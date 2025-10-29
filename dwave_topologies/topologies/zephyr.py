"""
D-Wave Zephyr Z12 topology definition.

Loaded from static JSON file (zephyr_z12_t4.json).
This represents a generic D-Wave Advantage2 topology structure.

Topology Information:
- Type: zephyr
- Shape: [12, 4]
- Nodes: 4800
- Edges: 45864
"""

from typing import List, Tuple, Dict, Any
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON file
_json_topology = load_json_topology('zephyr_z12_t4.json')

# Export the topology instance directly
ZEPHYR_Z12_TOPOLOGY = _json_topology

# Legacy dictionary format for backward compatibility
ZEPHYR_Z12 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = ZEPHYR_Z12['nodes']
EDGES: List[Tuple[int, int]] = ZEPHYR_Z12['edges']
PROPERTIES: Dict[str, Any] = ZEPHYR_Z12['properties']
SOLVER_NAME: str = ZEPHYR_Z12['solver_name']
TOPOLOGY_TYPE: str = ZEPHYR_Z12['topology_type']
TOPOLOGY_SHAPE: str = ZEPHYR_Z12['topology_shape']
NUM_NODES: int = ZEPHYR_Z12['num_nodes']
NUM_EDGES: int = ZEPHYR_Z12['num_edges']
