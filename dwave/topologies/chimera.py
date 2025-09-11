"""
D-Wave Chimera C16 topology definition.

Generated from dwave_networkx.chimera_graph(16, 16, 4) for general-purpose use.
This represents the D-Wave 2000Q topology structure.

Topology Information:
- Type: chimera
- Shape: [16, 16, 4]
- Nodes: ~2048
- Edges: ~6016
"""

from typing import List, Tuple, Dict, Any
from .dwave_topology import DWaveTopology
import dwave_networkx as dnx
import networkx as nx

# Generate the actual topology data
_graph = dnx.chimera_graph(16, 16, 4)
_nodes = list(_graph.nodes())
_edges = list(_graph.edges())

class ChimeraC16Topology:
    """D-Wave Chimera C16 topology implementation."""

    def __init__(self):
        # Basic topology information
        self.solver_name = 'Chimera_C16_Generic'
        self.topology_type = 'chimera'
        self.topology_shape = '[16, 16, 4]'
        self.num_nodes = len(_nodes)
        self.num_edges = len(_edges)

        # Topology data
        self.nodes = _nodes
        self.edges = _edges

        # D-Wave properties
        self.properties = {
            'topology': {
                'type': 'chimera',
                'shape': [16, 16, 4]
            },
            'num_qubits': len(_nodes),
            'num_couplers': len(_edges),
            'chip_id': 'DW_2000Q_VFYC_1',
            'supported_problem_types': ['qubo', 'ising'],
        }

        # Metadata
        self.generated_at = 'Generated from dwave_networkx'
        self.docs = {
            'topology': 'https://support.dwavesys.com/hc/en-us/articles/360003695354-What-Is-the-Chimera-Topology',
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
CHIMERA_C16_TOPOLOGY = ChimeraC16Topology()

# Legacy dictionary format for backward compatibility
CHIMERA_C16 = {
    'solver_name': 'Chimera_C16_Generic',
    'topology_type': 'chimera',
    'topology_shape': '[16, 16, 4]',
    'num_nodes': len(_nodes),
    'num_edges': len(_edges),
    'nodes': _nodes,
    'edges': _edges,
    'properties': CHIMERA_C16_TOPOLOGY.properties,
    'generated_at': 'Generated from dwave_networkx',
    'docs': CHIMERA_C16_TOPOLOGY.docs
}

# Convenience accessors
NODES: List[int] = CHIMERA_C16['nodes']
EDGES: List[Tuple[int, int]] = CHIMERA_C16['edges']
PROPERTIES: Dict[str, Any] = CHIMERA_C16['properties']
SOLVER_NAME: str = 'Chimera_C16_Generic'
TOPOLOGY_TYPE: str = 'chimera'
TOPOLOGY_SHAPE: str = '[16, 16, 4]'
NUM_NODES: int = len(_nodes)
NUM_EDGES: int = len(_edges)
