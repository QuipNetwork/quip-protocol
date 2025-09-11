"""
D-Wave Pegasus P16 topology definition.

Generated from dwave_networkx.pegasus_graph(16) for general-purpose use.
This represents the D-Wave Advantage topology structure.

Topology Information:
- Type: pegasus
- Shape: [16]
- Nodes: ~5627
- Edges: ~40279
"""

from typing import List, Tuple, Dict, Any
from .dwave_topology import DWaveTopology
import dwave_networkx as dnx
import networkx as nx

# Generate the actual topology data
_graph = dnx.pegasus_graph(16)
_nodes = list(_graph.nodes())
_edges = list(_graph.edges())

class PegasusP16Topology:
    """D-Wave Pegasus P16 topology implementation."""

    def __init__(self):
        # Basic topology information
        self.solver_name = 'Pegasus_P16_Generic'
        self.topology_type = 'pegasus'
        self.topology_shape = '[16]'
        self.num_nodes = len(_nodes)
        self.num_edges = len(_edges)

        # Topology data
        self.nodes = _nodes
        self.edges = _edges

        # D-Wave properties
        self.properties = {
            'topology': {
                'type': 'pegasus',
                'shape': [16]
            },
            'num_qubits': len(_nodes),
            'num_couplers': len(_edges),
            'chip_id': 'Advantage_system1.1',
            'supported_problem_types': ['qubo', 'ising'],
        }

        # Metadata
        self.generated_at = 'Generated from dwave_networkx'
        self.docs = {
            'topology': 'https://support.dwavesys.com/hc/en-us/articles/360003695354-What-Is-the-Pegasus-Topology',
            'solver': 'https://docs.dwavesys.com/docs/latest/c_solver_properties.html',
            'overview': 'https://docs.ocean.dwavesys.com/en/latest/concepts/topology.html'
        }

        # Create graph on initialization
        self._graph = _graph

    @property
    def graph(self) -> nx.Graph:
        """Get the NetworkX graph for this topology."""
        return self._graph

# Create the topology instance
PEGASUS_P16_TOPOLOGY = PegasusP16Topology()

# Legacy dictionary format for backward compatibility
PEGASUS_P16 = {
    'solver_name': 'Pegasus_P16_Generic',
    'topology_type': 'pegasus',
    'topology_shape': '[16]',
    'num_nodes': len(_nodes),
    'num_edges': len(_edges),
    'nodes': _nodes,
    'edges': _edges,
    'properties': PEGASUS_P16_TOPOLOGY.properties,
    'generated_at': 'Generated from dwave_networkx',
    'docs': PEGASUS_P16_TOPOLOGY.docs
}

# Convenience accessors
NODES: List[int] = PEGASUS_P16['nodes']
EDGES: List[Tuple[int, int]] = PEGASUS_P16['edges']
PROPERTIES: Dict[str, Any] = PEGASUS_P16['properties']
SOLVER_NAME: str = 'Pegasus_P16_Generic'
TOPOLOGY_TYPE: str = 'pegasus'
TOPOLOGY_SHAPE: str = '[16]'
NUM_NODES: int = len(_nodes)
NUM_EDGES: int = len(_edges)
