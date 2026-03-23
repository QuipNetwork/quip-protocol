"""
D-Wave Advantage2_system1.13 topology definition.

Loaded from static JSON file (advantage2_system1_13.json.gz).
This is the real Advantage2_system1.13 solver topology with defects.

Topology Information:
- Solver: Advantage2_system1.13
- Type: zephyr
- Shape: [12, 4]
- Nodes: 4579
- Edges: 41549
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system1_13.json.gz')

# Export the topology instance directly
ADVANTAGE2_SYSTEM1_13_TOPOLOGY = _json_topology
