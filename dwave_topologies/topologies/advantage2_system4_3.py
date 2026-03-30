"""
D-Wave Advantage2_system4.3 topology definition.

Loaded from static JSON file (advantage2_system4_3.json.gz).
This is the real Advantage2_system4.3 solver topology with defects.

Topology Information:
- Solver: Advantage2_system4.3
- Type: zephyr
- Shape: [6, 4]
- Nodes: 1203
- Edges: 10553
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system4_3.json.gz')

# Export the topology instance directly
ADVANTAGE2_SYSTEM4_3_TOPOLOGY = _json_topology
