"""
D-Wave Advantage_system4.1 topology definition.

Loaded from static JSON file (advantage_system4_1.json.gz).
This is the real Advantage_system4.1 solver topology with defects.

Topology Information:
- Solver: Advantage_system4.1
- Type: pegasus
- Shape: [16]
- Nodes: 5627
- Edges: 40279
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage_system4_1.json.gz')

# Export the topology instance directly
ADVANTAGE_SYSTEM4_1_TOPOLOGY = _json_topology
