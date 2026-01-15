"""
D-Wave Advantage2_system1.8 topology definition.

Loaded from static JSON file (advantage2_system1_8.json.gz).
This is the real Advantage2_system1.8 solver topology with defects.

Topology Information:
- Solver: Advantage2_system1.8
- Type: zephyr
- Shape: [12, 4]
- Nodes: 4591
- Edges: 41766
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system1_8.json.gz')

# Export the topology instance directly
ADVANTAGE2_SYSTEM1_8_TOPOLOGY = _json_topology
