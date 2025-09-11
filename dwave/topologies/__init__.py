"""
D-Wave solver topologies package.

This package contains topology definitions for various D-Wave solvers,
including both real solver topologies extracted from the D-Wave API and
general-purpose topology definitions for development and testing.

Usage:
    # Import a specific real solver topology
    from dwave.topologies.advantage2_system1_6 import ADVANTAGE2_SYSTEM1_6

    # Import general-purpose topologies
    from dwave.topologies.zephyr_z12 import ZEPHYR_Z12
    from dwave.topologies.pegasus_p16 import PEGASUS_P16
    from dwave.topologies.chimera_c16 import CHIMERA_C16

    # Access topology data
    nodes = ADVANTAGE2_SYSTEM1_6['nodes']
    edges = ADVANTAGE2_SYSTEM1_6['edges']

    # Or use convenience accessors
    from dwave.topologies.advantage2_system1_6 import NODES, EDGES
"""

# Import topology objects (new type system)
from .chimera_c16 import CHIMERA_C16_TOPOLOGY
from .pegasus_p16 import PEGASUS_P16_TOPOLOGY
from .zephyr_z12 import ZEPHYR_Z12_TOPOLOGY
from .advantage2_system1_6 import ADVANTAGE2_SYSTEM1_6_TOPOLOGY

# Legacy dictionary imports for backward compatibility
from .chimera_c16 import CHIMERA_C16
from .pegasus_p16 import PEGASUS_P16
from .zephyr_z12 import ZEPHYR_Z12
from .advantage2_system1_6 import ADVANTAGE2_SYSTEM1_6
from .advantage_system4_1 import ADVANTAGE_SYSTEM4_1
from .advantage_system6_4 import ADVANTAGE_SYSTEM6_4

# Default topology (Advantage2 Zephyr Z12)
DEFAULT_TOPOLOGY = ADVANTAGE2_SYSTEM1_6_TOPOLOGY

__all__ = [
    # Topology objects (new type system)
    "CHIMERA_C16_TOPOLOGY",
    "PEGASUS_P16_TOPOLOGY",
    "ZEPHYR_Z12_TOPOLOGY",
    "ADVANTAGE2_SYSTEM1_6_TOPOLOGY",

    # Legacy dictionary format (backward compatibility)
    "CHIMERA_C16",
    "PEGASUS_P16",
    "ZEPHYR_Z12",
    "ADVANTAGE2_SYSTEM1_6",
    "ADVANTAGE_SYSTEM4_1",
    "ADVANTAGE_SYSTEM6_4",

    # Default
    "DEFAULT_TOPOLOGY",
]
