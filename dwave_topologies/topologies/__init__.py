"""
D-Wave solver topologies package.

This package contains topology definitions for various D-Wave solvers,
including both real solver topologies extracted from the D-Wave API and
general-purpose topology definitions for development and testing.

Usage:
    # Import topology objects
    from dwave_topologies import DEFAULT_TOPOLOGY, ZEPHYR_Z9_T2_TOPOLOGY
    from dwave_topologies.topologies import ADVANTAGE2_SYSTEM1_6_TOPOLOGY

    # Access topology properties
    nodes = DEFAULT_TOPOLOGY.nodes
    edges = DEFAULT_TOPOLOGY.edges
    num_nodes = DEFAULT_TOPOLOGY.num_nodes
    properties = DEFAULT_TOPOLOGY.properties
"""

# Import topology objects (new type system)
from .chimera import CHIMERA_C16_TOPOLOGY
from .pegasus import PEGASUS_P16_TOPOLOGY
from .zephyr import (
    zephyr,
    ZephyrTopology,
    ZEPHYR_Z8_T2_TOPOLOGY,
    ZEPHYR_Z9_T2_TOPOLOGY,
    ZEPHYR_Z10_T2_TOPOLOGY,
    ZEPHYR_Z11_T4_TOPOLOGY,
    ZEPHYR_Z12_T4_TOPOLOGY,
)
from .advantage2_system1_7 import ADVANTAGE2_SYSTEM1_7_TOPOLOGY
from .advantage2_system4_1 import ADVANTAGE2_SYSTEM4_1_TOPOLOGY
from .json_loader import load_topology

# Default topology (Zephyr Z(9, 2) - logical problem size for all miners)
# Topology: 1,368 nodes, 7,692 edges (perfect Zephyr graph structure)
# - CPU/GPU/Metal: Sample on perfect Z(9,2) graph
# - QPU: Uses Z(9,2) embedding to map onto Advantage2 hardware (4,592 qubits)
# Precomputed embedding: dwave_topologies/embeddings/Advantage2_system1_7/zephyr_z9_t2.embed.json.gz
# All miners solve the same logical problem size, ensuring consistent GSE
DEFAULT_TOPOLOGY = ZEPHYR_Z9_T2_TOPOLOGY

__all__ = [
    # Topology objects
    "CHIMERA_C16_TOPOLOGY",
    "PEGASUS_P16_TOPOLOGY",
    "zephyr",
    "ZephyrTopology",
    "ZEPHYR_Z8_T2_TOPOLOGY",
    "ZEPHYR_Z9_T2_TOPOLOGY",
    "ZEPHYR_Z10_T2_TOPOLOGY",
    "ZEPHYR_Z11_T4_TOPOLOGY",
    "ZEPHYR_Z12_T4_TOPOLOGY",
    "ADVANTAGE2_SYSTEM1_7_TOPOLOGY",
    "ADVANTAGE2_SYSTEM4_1_TOPOLOGY",

    # Default
    "DEFAULT_TOPOLOGY",

    # Utilities
    "load_topology",
]
