"""
D-Wave Pegasus P16 topology definition.

Loaded from static JSON file (pegasus_p16.json).
This represents the D-Wave Advantage topology structure.

Topology Information:
- Type: pegasus
- Shape: [16]
- Nodes: 5640
- Edges: 40484
"""

import os
from typing import List, Tuple, Dict, Any
from .json_loader import load_json_topology, json_topology_to_dict

# Load topology from JSON file in topologies/ directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_json_topology = load_json_topology('pegasus_p16.json', topologies_dir=_current_dir)

# Export the topology instance directly
PEGASUS_P16_TOPOLOGY = _json_topology

# Legacy dictionary format for backward compatibility
PEGASUS_P16 = json_topology_to_dict(_json_topology)

# Convenience accessors
NODES: List[int] = PEGASUS_P16['nodes']
EDGES: List[Tuple[int, int]] = PEGASUS_P16['edges']
PROPERTIES: Dict[str, Any] = PEGASUS_P16['properties']
SOLVER_NAME: str = PEGASUS_P16['solver_name']
TOPOLOGY_TYPE: str = PEGASUS_P16['topology_type']
TOPOLOGY_SHAPE: str = PEGASUS_P16['topology_shape']
NUM_NODES: int = PEGASUS_P16['num_nodes']
NUM_EDGES: int = PEGASUS_P16['num_edges']
