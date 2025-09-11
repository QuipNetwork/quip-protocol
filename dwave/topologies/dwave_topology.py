"""
D-Wave topology type definition and base functionality.

This module defines the standard DWaveTopology type that all topology files implement.
"""

from typing import List, Tuple, Dict, Any, Protocol
import networkx as nx


class DWaveTopology(Protocol):
    """Protocol defining the interface for D-Wave topology objects."""
    
    # Basic topology information
    solver_name: str
    topology_type: str  # 'chimera', 'pegasus', 'zephyr'
    topology_shape: str  # String representation like '[12, 4]'
    num_nodes: int
    num_edges: int
    
    # Topology data
    nodes: List[int]
    edges: List[Tuple[int, int]]
    
    # D-Wave properties (from API or mock)
    properties: Dict[str, Any]
    
    # Metadata
    generated_at: str
    docs: Dict[str, str]  # Optional documentation links


def create_graph_from_topology(topology: DWaveTopology) -> nx.Graph:
    """Create a NetworkX graph from a DWaveTopology object."""
    graph = nx.Graph()
    graph.add_nodes_from(topology.nodes)
    graph.add_edges_from(topology.edges)
    return graph


def get_topology_properties(topology: DWaveTopology) -> Dict[str, Any]:
    """Extract D-Wave compatible properties from a topology object."""
    return topology.properties


def validate_topology(topology: DWaveTopology) -> bool:
    """Validate that a topology object conforms to the expected structure."""
    try:
        # Check required attributes exist
        required_attrs = [
            'solver_name', 'topology_type', 'topology_shape',
            'num_nodes', 'num_edges', 'nodes', 'edges', 'properties'
        ]
        
        for attr in required_attrs:
            if not hasattr(topology, attr):
                return False
        
        # Check data consistency
        if len(topology.nodes) != topology.num_nodes:
            return False
        
        if len(topology.edges) != topology.num_edges:
            return False
        
        # Check properties structure
        if 'topology' not in topology.properties:
            return False
        
        topo_props = topology.properties['topology']
        if 'type' not in topo_props or 'shape' not in topo_props:
            return False
        
        return True
        
    except Exception:
        return False
