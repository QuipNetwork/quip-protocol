"""
D-Wave Advantage2_system1.6 topology definition.

Loaded from static JSON file (advantage2_system1_6.json).
This is the real Advantage2-System1.6 solver topology with defects.

Topology Information:
- Solver: Advantage2_system1.6
- Type: zephyr
- Shape: [12, 4]
- Nodes: 4593
- Edges: 41796
"""

from typing import List, Tuple, Dict, Any
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system1_6.json')

# Export the topology instance directly
ADVANTAGE2_SYSTEM1_6_TOPOLOGY = _json_topology

# Legacy dictionary format for backward compatibility
ADVANTAGE2_SYSTEM1_6 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = ADVANTAGE2_SYSTEM1_6['nodes']
EDGES: List[Tuple[int, int]] = ADVANTAGE2_SYSTEM1_6['edges']
PROPERTIES: Dict[str, Any] = ADVANTAGE2_SYSTEM1_6['properties']
SOLVER_NAME: str = ADVANTAGE2_SYSTEM1_6['solver_name']
TOPOLOGY_TYPE: str = ADVANTAGE2_SYSTEM1_6['topology_type']
TOPOLOGY_SHAPE: str = ADVANTAGE2_SYSTEM1_6['topology_shape']
NUM_NODES: int = ADVANTAGE2_SYSTEM1_6['num_nodes']
NUM_EDGES: int = ADVANTAGE2_SYSTEM1_6['num_edges']
