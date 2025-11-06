"""
D-Wave Advantage2_system1.7 topology definition.

Loaded from static JSON file (advantage2_system1_7.json.gz).
This is the real Advantage2-System1.7 solver topology with defects.

Topology Information:
- Solver: Advantage2_system1.7
- Type: zephyr
- Shape: [12, 4]
- Nodes: 4592 (qubit 510 removed vs system1.6)
- Edges: 41779
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system1_7.json.gz')

# Export the topology instance directly
ADVANTAGE2_SYSTEM1_7_TOPOLOGY = _json_topology
