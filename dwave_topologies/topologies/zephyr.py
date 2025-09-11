"""
D-Wave Zephyr Z12 topology definition.

Generated from dwave_networkx.zephyr_graph(12, 4) for general-purpose use.
This represents the D-Wave Advantage2 topology structure.

Topology Information:
- Type: zephyr
- Shape: [12, 4]
- Nodes: ~4800
- Edges: ~45864
"""

from typing import List, Tuple, Dict, Any
from .dwave_topology import DWaveTopology
import dwave_networkx as dnx
import networkx as nx

# Generate the actual topology data
_graph = dnx.zephyr_graph(12, 4)
_nodes = list(_graph.nodes())
_edges = list(_graph.edges())

class ZephyrZ12Topology:
    """D-Wave Zephyr Z12 topology implementation."""

    def __init__(self):
        # Basic topology information
        self.solver_name = 'Zephyr_Z12_Generic'
        self.topology_type = 'zephyr'
        self.topology_shape = '[12, 4]'
        self.num_nodes = len(_nodes)
        self.num_edges = len(_edges)

        # Topology data
        self.nodes = _nodes
        self.edges = _edges

        # D-Wave properties
        self.properties = {
            'topology': {
                'type': 'zephyr',
                'shape': [12, 4]
            },
            'num_qubits': len(_nodes),
            'num_couplers': len(_edges),
            'chip_id': 'Generic_Z12',
            'supported_problem_types': ['qubo', 'ising'],
        }

        # Metadata
        self.generated_at = 'Generated from dwave_networkx'
        self.docs = {
            'topology': 'https://support.dwavesys.com/hc/en-us/articles/360003695354-What-Is-the-Zephyr-Topology',
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
ZEPHYR_Z12_TOPOLOGY = ZephyrZ12Topology()

# Legacy dictionary format for backward compatibility
ZEPHYR_Z12 = {
    'solver_name': 'Zephyr_Z12_Generic',
    'topology_type': 'zephyr',
    'topology_shape': '[12, 4]',
    'num_nodes': len(_nodes),
    'num_edges': len(_edges),
    'nodes': _nodes,
    'edges': _edges,
    'properties': ZEPHYR_Z12_TOPOLOGY.properties,
    'generated_at': 'Generated from dwave_networkx',
    'docs': ZEPHYR_Z12_TOPOLOGY.docs
}

# Convenience accessors
NODES: List[int] = ZEPHYR_Z12['nodes']
EDGES: List[Tuple[int, int]] = ZEPHYR_Z12['edges']
PROPERTIES: Dict[str, Any] = ZEPHYR_Z12['properties']
SOLVER_NAME: str = 'Zephyr_Z12_Generic'
TOPOLOGY_TYPE: str = 'zephyr'
TOPOLOGY_SHAPE: str = '[12, 4]'
NUM_NODES: int = len(_nodes)
NUM_EDGES: int = len(_edges)
TOPOLOGY_SHAPE: str = '[12, 4]'
NUM_NODES: int = len(_nodes)
NUM_EDGES: int = len(_edges)
