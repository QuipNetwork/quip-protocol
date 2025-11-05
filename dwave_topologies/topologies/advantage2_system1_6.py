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

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system1_6.json')

# Export the topology instance directly
ADVANTAGE2_SYSTEM1_6_TOPOLOGY = _json_topology
