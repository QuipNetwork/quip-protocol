"""
Generic Zephyr topology factory and common Zephyr topologies.

Provides a factory function to create any Zephyr(m, t) topology on-the-fly,
plus pre-defined instances for commonly used configurations.

Usage:
    from dwave_topologies.topologies.zephyr import zephyr, ZEPHYR_Z8_T2_TOPOLOGY

    # Use factory for any configuration
    z5_t3 = zephyr(5, 3)

    # Or use pre-defined common topologies
    default = ZEPHYR_Z8_T2_TOPOLOGY
"""

from typing import List, Tuple, Dict, Any
import dwave_networkx as dnx
import networkx as nx


class ZephyrTopology:
    """Generated Zephyr topology (creates graph on first access)."""

    def __init__(self, m: int, t: int):
        """
        Create a Zephyr(m, t) topology.

        Args:
            m: Zephyr m parameter (number of unit cells)
            t: Zephyr t parameter (tile size)
        """
        self.m = m
        self.t = t
        self._graph = None
        self._nodes = None
        self._edges = None

    @property
    def graph(self) -> nx.Graph:
        """Lazy-load the Zephyr graph."""
        if self._graph is None:
            self._graph = dnx.zephyr_graph(self.m, self.t)
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
        return f"Z({self.m},{self.t})"

    @property
    def topology_type(self) -> str:
        return "zephyr"

    @property
    def topology_shape(self) -> str:
        return f"[{self.m}, {self.t}]"

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "topology": {"type": "zephyr", "shape": [self.m, self.t]},
            "num_qubits": self.num_nodes,
            "num_couplers": self.num_edges,
            "chip_id": f"Generic_Z{self.m}_T{self.t}",
            "supported_problem_types": ["qubo", "ising"]
        }


def zephyr(m: int, t: int) -> ZephyrTopology:
    """
    Factory function to create a Zephyr(m, t) topology.

    Args:
        m: Zephyr m parameter
        t: Zephyr t parameter

    Returns:
        ZephyrTopology instance

    Example:
        >>> z = zephyr(8, 2)
        >>> print(f"{z.num_nodes} nodes, {z.num_edges} edges")
        1088 nodes, 6068 edges
    """
    return ZephyrTopology(m, t)


# Pre-defined common Zephyr topologies
ZEPHYR_Z8_T2_TOPOLOGY = ZephyrTopology(8, 2)   # 1,088 nodes - precomputed embedding available
ZEPHYR_Z9_T2_TOPOLOGY = ZephyrTopology(9, 2)   # 1,368 nodes - DEFAULT with precomputed embedding
ZEPHYR_Z10_T2_TOPOLOGY = ZephyrTopology(10, 2)  # 1,680 nodes
ZEPHYR_Z11_T4_TOPOLOGY = ZephyrTopology(11, 4)  # 4,048 nodes - largest that fits Advantage2 well
ZEPHYR_Z12_T4_TOPOLOGY = ZephyrTopology(12, 4)  # 4,800 nodes - exceeds Advantage2, for reference




__all__ = [
    # Factory function
    'zephyr',
    'ZephyrTopology',

    # Pre-defined topologies
    'ZEPHYR_Z8_T2_TOPOLOGY',
    'ZEPHYR_Z9_T2_TOPOLOGY',
    'ZEPHYR_Z10_T2_TOPOLOGY',
    'ZEPHYR_Z11_T4_TOPOLOGY',
    'ZEPHYR_Z12_T4_TOPOLOGY',
]
