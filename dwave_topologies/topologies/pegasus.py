"""
Generic Pegasus topology factory and common Pegasus topologies.

Provides a factory function to create any Pegasus(m) topology on-the-fly,
plus pre-defined instances for commonly used configurations.

Usage:
    from dwave_topologies.topologies.pegasus import pegasus, PEGASUS_P16_TOPOLOGY

    # Use factory for any configuration
    p12 = pegasus(12)

    # Or use pre-defined common topologies
    default = PEGASUS_P16_TOPOLOGY
"""

from typing import List, Tuple, Dict, Any
import dwave_networkx as dnx
import networkx as nx


class PegasusTopology:
    """Generated Pegasus topology (creates graph on first access)."""

    def __init__(self, m: int):
        """
        Create a Pegasus(m) topology.

        Args:
            m: Pegasus m parameter (grid size)
        """
        self.m = m
        self._graph = None
        self._nodes = None
        self._edges = None

    @property
    def graph(self) -> nx.Graph:
        """Lazy-load the Pegasus graph."""
        if self._graph is None:
            self._graph = dnx.pegasus_graph(self.m)
        return self._graph

    @property
    def nodes(self) -> List[int]:
        """Get list of nodes."""
        if self._nodes is None:
            self._nodes = list(self.graph.nodes())
        return self._nodes

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Get list of edges."""
        if self._edges is None:
            self._edges = list(self.graph.edges())
        return self._edges

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def solver_name(self) -> str:
        return f"Pegasus_P{self.m}_Generic"

    @property
    def topology_type(self) -> str:
        return "pegasus"

    @property
    def topology_shape(self) -> str:
        return f"[{self.m}]"

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "topology": {"type": "pegasus", "shape": [self.m]},
            "num_qubits": self.num_nodes,
            "num_couplers": self.num_edges,
            "chip_id": f"Generic_P{self.m}",
            "supported_problem_types": ["qubo", "ising"]
        }


def pegasus(m: int) -> PegasusTopology:
    """
    Factory function to create a Pegasus(m) topology.

    Args:
        m: Pegasus m parameter

    Returns:
        PegasusTopology instance

    Example:
        >>> p = pegasus(16)
        >>> print(f"{p.num_nodes} nodes, {p.num_edges} edges")
        5640 nodes, 40484 edges
    """
    return PegasusTopology(m)


# Pre-defined common Pegasus topologies
PEGASUS_P16_TOPOLOGY = PegasusTopology(16)  # D-Wave Advantage: 5640 nodes




__all__ = [
    # Factory function
    'pegasus',
    'PegasusTopology',

    # Pre-defined topologies
    'PEGASUS_P16_TOPOLOGY',
]
