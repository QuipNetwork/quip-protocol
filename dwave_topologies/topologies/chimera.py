"""
D-Wave Chimera C16 topology definition.

Loaded from static JSON file (chimera_c16.json).
This represents the D-Wave 2000Q topology structure.

Topology Information:
- Type: chimera
- Shape: [16, 16, 4]
- Nodes: 2048
- Edges: 6016
"""

from typing import List, Tuple, Dict, Any
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON file
_json_topology = load_json_topology('chimera_c16.json')

# Export the topology instance directly
CHIMERA_C16_TOPOLOGY = _json_topology

# Legacy dictionary format for backward compatibility
CHIMERA_C16 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = CHIMERA_C16['nodes']
EDGES: List[Tuple[int, int]] = CHIMERA_C16['edges']
PROPERTIES: Dict[str, Any] = CHIMERA_C16['properties']
SOLVER_NAME: str = CHIMERA_C16['solver_name']
TOPOLOGY_TYPE: str = CHIMERA_C16['topology_type']
TOPOLOGY_SHAPE: str = CHIMERA_C16['topology_shape']
NUM_NODES: int = CHIMERA_C16['num_nodes']
NUM_EDGES: int = CHIMERA_C16['num_edges']
