"""
D-Wave Advantage2_system4.1 topology definition.

Loaded from static JSON file (advantage2_system4_1.json.gz).
This is the real Advantage2-System4.1 solver topology with defects.

Topology Information:
- Solver: Advantage2_system4.1
- Type: zephyr
- Shape: [6, 4]
- Nodes: 1204
- Edges: 10571
- Region: na-east-1
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage2_system4_1.json.gz')

# Export the topology instance directly
ADVANTAGE2_SYSTEM4_1_TOPOLOGY = _json_topology
