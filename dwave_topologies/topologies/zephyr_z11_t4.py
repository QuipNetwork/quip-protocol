"""
D-Wave Zephyr Z(11, 4) topology definition.

Loaded from static JSON file (zephyr_z11_t4.json).
This is the most aggressive generic Zephyr topology that fits on Advantage2-System1.6.

Topology Information:
- Type: zephyr
- Shape: [11, 4]
- Nodes: 4048
- Edges: 38520
- Average degree: 19.03
- Advantage2 utilization: 88% nodes, 92% edges

This topology provides:
- Generic, QPU-agnostic graph structure (no solver-specific defect patterns)
- Maximum size that fits comfortably on Advantage2-System1.6
- Higher connectivity than Z(12, 4) which has 4800 nodes (exceeds Advantage2)
- Round parameters (11×4) for clean generation
- Loaded from JSON (no runtime dependency on dwave_networkx)
"""

from typing import List, Tuple, Dict, Any
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON file
_json_topology = load_json_topology('zephyr_z11_t4.json')

# Export the topology instance directly
ZEPHYR_Z11_T4_TOPOLOGY = _json_topology

# Legacy dictionary format for backward compatibility
ZEPHYR_Z11_T4 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = ZEPHYR_Z11_T4['nodes']
EDGES: List[Tuple[int, int]] = ZEPHYR_Z11_T4['edges']
PROPERTIES: Dict[str, Any] = ZEPHYR_Z11_T4['properties']
SOLVER_NAME: str = ZEPHYR_Z11_T4['solver_name']
TOPOLOGY_TYPE: str = ZEPHYR_Z11_T4['topology_type']
TOPOLOGY_SHAPE: str = ZEPHYR_Z11_T4['topology_shape']
NUM_NODES: int = ZEPHYR_Z11_T4['num_nodes']
NUM_EDGES: int = ZEPHYR_Z11_T4['num_edges']
