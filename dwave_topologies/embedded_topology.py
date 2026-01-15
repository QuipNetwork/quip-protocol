"""
Extract embedded hardware topology from precomputed embeddings.

This module allows you to create a topology based on the actual physical qubits
and couplers used in an embedding, rather than the perfect source topology.

This is useful for comparing:
- Perfect Zephyr Z(9,2): 1,368 logical variables, fully connected
- Embedded Hardware: ~3,423 physical qubits with hardware defects/constraints

Usage:
    from dwave_topologies.embedded_topology import create_embedded_topology

    # Create topology from Z(9,2) embedding
    hw_topology = create_embedded_topology("Z(9,2)")
    print(f"Physical qubits: {hw_topology.num_nodes}")
    print(f"Physical couplers: {hw_topology.num_edges}")
"""

import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dwave_topologies.embedding_loader import load_embedding
from dwave_topologies.topologies.json_loader import load_json_topology


class EmbeddedTopology:
    """Topology representing the physical hardware subgraph used in an embedding."""

    def __init__(self, topology_name: str, solver_name: str = "Advantage2_system1.10"):
        """
        Create an embedded topology from a precomputed embedding.

        Args:
            topology_name: Source topology name like "Z(9,2)"
            solver_name: Target solver name (default: "Advantage2_system1.10")
        """
        self.topology_name = topology_name
        self.solver_name = solver_name

        # Load embedding data
        embedding_data = load_embedding(topology_name, solver_name)
        if embedding_data is None:
            raise FileNotFoundError(f"No embedding found for {topology_name} on {solver_name}")

        self.embedding = embedding_data['embedding']
        self.metadata = embedding_data['metadata']
        self.statistics = embedding_data['statistics']

        # Load target hardware topology
        solver_filename = solver_name.replace('-', '_').replace('.', '_') + '.json'
        hardware_topology = load_json_topology(solver_filename)
        hardware_graph = hardware_topology.graph

        # Extract physical qubits used in embedding
        physical_qubits = set()
        for chains in self.embedding.values():
            physical_qubits.update(int(q) for q in chains)

        # Create subgraph of hardware containing only embedded qubits
        self._graph = hardware_graph.subgraph(physical_qubits).copy()
        self._nodes = None
        self._edges = None

    @property
    def graph(self) -> nx.Graph:
        """Get the NetworkX graph of physical qubits/couplers."""
        return self._graph

    @property
    def nodes(self) -> List[int]:
        """Get list of physical qubit nodes."""
        if self._nodes is None:
            self._nodes = list(self._graph.nodes())
        return self._nodes

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Get list of physical coupler edges."""
        if self._edges is None:
            self._edges = list(self._graph.edges())
        return self._edges

    @property
    def num_nodes(self) -> int:
        """Number of physical qubits."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of physical couplers."""
        return len(self.edges)

    @property
    def solver_name_prop(self) -> str:
        """Solver name property."""
        return f"{self.metadata['source_topology']}_Embedded_on_{self.solver_name}"

    @property
    def topology_type(self) -> str:
        """Topology type."""
        return "embedded_hardware"

    @property
    def topology_shape(self) -> str:
        """Topology shape (from source)."""
        return self.topology_name

    @property
    def properties(self) -> Dict[str, Any]:
        """Topology properties including embedding statistics.

        Note: Uses 'zephyr' as topology type to maintain compatibility with
        MockDWaveSampler, which only recognizes standard QPU architectures.
        """
        # Load target hardware topology to get its properties
        solver_filename = self.solver_name.replace('-', '_').replace('.', '_') + '.json'
        from dwave_topologies.topologies.json_loader import load_json_topology
        hardware_topology = load_json_topology(solver_filename)

        return {
            "topology": hardware_topology.properties['topology'],  # Use hardware topology type (zephyr)
            "num_qubits": self.num_nodes,
            "num_couplers": self.num_edges,
            "chip_id": f"{self.solver_name}_embedded_{self.topology_name}",
            "supported_problem_types": ["qubo", "ising"],
            "embedding_statistics": self.statistics,
            "metadata": self.metadata,
        }


def create_embedded_topology(topology_name: str, solver_name: str = "Advantage2_system1.10") -> EmbeddedTopology:
    """
    Create an embedded topology from a precomputed embedding.

    Args:
        topology_name: Source topology like "Z(9,2)"
        solver_name: Target solver (default: "Advantage2_system1.10")

    Returns:
        EmbeddedTopology instance representing the physical hardware subgraph

    Example:
        >>> hw_topo = create_embedded_topology("Z(9,2)")
        >>> print(f"Physical qubits: {hw_topo.num_nodes}")
        Physical qubits: 3423
        >>> print(f"Avg chain length: {hw_topo.statistics['avg_chain_length']}")
        Avg chain length: 2.5
    """
    return EmbeddedTopology(topology_name, solver_name)


__all__ = [
    'EmbeddedTopology',
    'create_embedded_topology',
]
