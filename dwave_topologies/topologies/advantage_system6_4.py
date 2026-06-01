"""
D-Wave Advantage_system6.4 topology definition.

Loaded from static JSON file (advantage_system6_4.json.gz).
This is the real Advantage_system6.4 solver topology with defects.

Topology Information:
- Solver: Advantage_system6.4
- Type: pegasus
- Shape: [16]
- Nodes: 5612
- Edges: 40088
"""

from .json_loader import load_json_topology

# Load topology from JSON file
_json_topology = load_json_topology('advantage_system6_4.json.gz')

# Export the topology instance directly
ADVANTAGE_SYSTEM6_4_TOPOLOGY = _json_topology
