"""
Generic Chimera topology factory and common Chimera topologies.

Provides a factory function to create any Chimera(m, n, t) topology on-the-fly,
plus pre-defined instances for commonly used configurations.

Usage:
    from dwave_topologies.topologies.chimera import chimera, CHIMERA_C16_TOPOLOGY

    # Use factory for any configuration
    c12 = chimera(12, 12, 4)

    # Or use pre-defined common topologies
    default = CHIMERA_C16_TOPOLOGY
"""

from typing import List, Tuple, Dict, Any
import dwave_networkx as dnx
import networkx as nx


class ChimeraTopology:
    """Generated Chimera topology (creates graph on first access)."""

    def __init__(self, m: int, n: int = None, t: int = 4):
        """
        Create a Chimera(m, n, t) topology.

        Args:
            m: Number of rows of unit cells
            n: Number of columns of unit cells (defaults to m for square grid)
            t: Size of shore within each unit cell (default 4)
        """
        self.m = m
        self.n = n if n is not None else m
        self.t = t
        self._graph = None
        self._nodes = None
        self._edges = None

    @property
    def graph(self) -> nx.Graph:
        """Lazy-load the Chimera graph."""
        if self._graph is None:
            self._graph = dnx.chimera_graph(self.m, self.n, self.t)
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
        return f"Chimera_C{self.m}_{self.n}_T{self.t}_Generic"

    @property
    def topology_type(self) -> str:
        return "chimera"

    @property
    def topology_shape(self) -> str:
        return f"[{self.m}, {self.n}, {self.t}]"

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "topology": {"type": "chimera", "shape": [self.m, self.n, self.t]},
            "num_qubits": self.num_nodes,
            "num_couplers": self.num_edges,
            "chip_id": f"Generic_C{self.m}_{self.n}_T{self.t}",
            "supported_problem_types": ["qubo", "ising"]
        }


def chimera(m: int, n: int = None, t: int = 4) -> ChimeraTopology:
    """
    Factory function to create a Chimera(m, n, t) topology.

    Args:
        m: Number of rows
        n: Number of columns (defaults to m)
        t: Shore size (default 4)

    Returns:
        ChimeraTopology instance

    Example:
        >>> c = chimera(16)
        >>> print(f"{c.num_nodes} nodes, {c.num_edges} edges")
        2048 nodes, 6016 edges
    """
    return ChimeraTopology(m, n, t)


# Pre-defined common Chimera topologies
CHIMERA_C16_TOPOLOGY = ChimeraTopology(16, 16, 4)  # D-Wave 2000Q: 2048 nodes




__all__ = [
    # Factory function
    'chimera',
    'ChimeraTopology',

    # Pre-defined topologies
    'CHIMERA_C16_TOPOLOGY',
]
