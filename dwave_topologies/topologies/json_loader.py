"""
JSON topology loader for QUIP protocol.

Loads D-Wave topology definitions from JSON files (optionally gzipped) into DWaveTopology objects.
This allows for static, version-controlled topology definitions without
requiring code generation or dwave_networkx at runtime.

Supports both plain JSON (.json) and gzip-compressed JSON (.json.gz) files.
Gzipped files are preferred as they significantly reduce file size (typically 10x compression).
"""

import json
import gzip
import os
from typing import List, Tuple, Dict, Any, Union
import networkx as nx


class DWaveTopologyFromJSON:
    """D-Wave topology loaded from JSON file."""

    def __init__(self, json_data: Dict[str, Any]):
        """
        Initialize topology from JSON data.

        Args:
            json_data: Dictionary containing topology data from JSON file
        """
        metadata = json_data['metadata']
        properties = json_data['properties']

        # Basic topology information
        self.solver_name = metadata['solver_name']
        self.topology_type = metadata['topology_type']
        self.topology_shape = str(metadata['topology_shape'])
        self.num_nodes = metadata['num_nodes']
        self.num_edges = metadata['num_edges']

        # Topology data
        self.nodes = json_data['nodes']
        # Convert edge lists back to tuples
        self.edges = [tuple(edge) for edge in json_data['edges']]

        # D-Wave properties
        self.properties = properties

        # Metadata
        self.generated_at = metadata.get('generated_from', 'Loaded from JSON')
        self.docs = json_data.get('docs', {})

        # Lazy-load graph
        self._graph = None

    @property
    def graph(self) -> nx.Graph:
        """Get the NetworkX graph for this topology (lazy-loaded)."""
        if self._graph is None:
            self._graph = nx.Graph()
            self._graph.add_nodes_from(self.nodes)
            self._graph.add_edges_from(self.edges)
        return self._graph


def load_json_topology(filename: str, topologies_dir: str = None, from_embeddings: bool = False) -> DWaveTopologyFromJSON:
    """
    Load a topology from a JSON file (plain or gzipped).

    Automatically detects whether the file is gzipped based on the .gz extension.
    If the exact filename isn't found, will also try the .gz variant.

    Args:
        filename: Name of the JSON topology file (e.g., 'zephyr_z11_t4.json' or 'zephyr_z11_t4.json.gz')
        topologies_dir: Directory containing topology files. If None, uses default based on from_embeddings.
        from_embeddings: If True, load from embeddings/Advantage2_system1_10/. Otherwise load from topologies/.

    Returns:
        DWaveTopologyFromJSON instance

    Example:
        >>> topology = load_json_topology('zephyr_z11_t4.json', from_embeddings=True)  # Mined topologies
        >>> topology = load_json_topology('zephyr_z12_t4.json')  # Pregenerated topologies
        >>> print(f"Loaded {topology.num_nodes} nodes, {topology.num_edges} edges")
    """
    if topologies_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if from_embeddings:
            # Mined topologies are in embeddings/Advantage2_system1_10/
            parent_dir = os.path.dirname(current_dir)
            topologies_dir = os.path.join(parent_dir, 'embeddings', 'Advantage2_system1_10')
        else:
            # Pregenerated topologies are in topologies/
            topologies_dir = current_dir

    filepath = os.path.join(topologies_dir, filename)

    # Try gzipped version if plain file doesn't exist
    if not os.path.exists(filepath) and not filename.endswith('.gz'):
        filepath_gz = filepath + '.gz'
        if os.path.exists(filepath_gz):
            filepath = filepath_gz

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Topology file not found: {filepath} (also checked .gz variant)")

    # Load JSON data (automatically decompressing if gzipped)
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

    return DWaveTopologyFromJSON(json_data)


def load_topology(topology_name: str) -> Any:
    """
    Parse a topology name or filename and return the appropriate topology object.

    This function handles:
    - Zephyr topology names: "Z(9,2)" → generates ZephyrTopology on-the-fly
    - Hardware topology names: "Advantage2_system1.10" → loads from JSON
    - File paths: "path/to/topology.json" → loads from file
    - Named topologies: "ZEPHYR_Z9_T2_TOPOLOGY" → returns from dwave_topologies module

    Args:
        topology_name: String identifying the topology. Can be:
                      - Zephyr format: "Z(9,2)", "Z(10,2)", etc.
                      - Hardware name: "Advantage2_system1.10", "Advantage2_system4_1"
                      - File path: "path/to/custom_topology.json" or ".json.gz"
                      - Named constant: "ZEPHYR_Z9_T2_TOPOLOGY"

    Returns:
        DWaveTopology object (either ZephyrTopology or DWaveTopologyFromJSON)

    Raises:
        ValueError: If topology_name format is not recognized
        FileNotFoundError: If a file path is provided but file doesn't exist

    Examples:
        >>> # Zephyr topologies (generated on-the-fly)
        >>> topology = load_topology("Z(9,2)")
        >>> topology = load_topology("Z(10,2)")

        >>> # Hardware topologies (loaded from JSON)
        >>> topology = load_topology("Advantage2_system1.10")
        >>> topology = load_topology("Advantage2-system1.10")  # Also works

        >>> # Custom file
        >>> topology = load_topology("path/to/my_topology.json.gz")

        >>> # Named constants
        >>> topology = load_topology("ZEPHYR_Z9_T2_TOPOLOGY")
    """
    from .zephyr import ZephyrTopology

    # Case 1: File path (contains / or ends with .json/.json.gz)
    if '/' in topology_name or topology_name.endswith(('.json', '.json.gz')):
        if not os.path.exists(topology_name):
            raise FileNotFoundError(f"Topology file not found: {topology_name}")
        return load_json_topology(os.path.basename(topology_name),
                                  topologies_dir=os.path.dirname(topology_name) or None)

    # Case 2: Zephyr format Z(m,t) - generate on-the-fly
    if topology_name.startswith("Z(") and topology_name.endswith(")"):
        try:
            # Parse Z(m,t) format
            parts = topology_name[2:-1].split(",")
            m = int(parts[0].strip())
            t = int(parts[1].strip())

            # Create Zephyr topology (graph generated lazily)
            return ZephyrTopology(m, t)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid Zephyr format '{topology_name}'. Expected 'Z(m,t)' with integer m,t. Error: {e}")

    # Case 2b: Old Zephyr format "Zephyr_Zm_Tt_Generic" (deprecated, for backward compatibility)
    if topology_name.startswith("Zephyr_Z") and "_T" in topology_name:
        import re
        match = re.match(r'Zephyr_Z(\d+)_T(\d+)(?:_Generic)?', topology_name)
        if match:
            m = int(match.group(1))
            t = int(match.group(2))
            return ZephyrTopology(m, t)

    # Case 3: Hardware topology name (e.g., "Advantage2_system1.10" or "Advantage2-system1.10")
    # Normalize name to match JSON filename
    normalized_name = topology_name.replace('-', '_').replace('.', '_').lower()

    # Try to load as JSON file
    try:
        json_filename = f"{normalized_name}.json"
        return load_json_topology(json_filename, from_embeddings=False)
    except FileNotFoundError:
        pass

    # Case 4: Try loading from embeddings directory
    try:
        json_filename = f"{normalized_name}.json"
        return load_json_topology(json_filename, from_embeddings=True)
    except FileNotFoundError:
        pass

    # Case 5: Named constant from dwave_topologies module
    try:
        import dwave_topologies
        if hasattr(dwave_topologies, topology_name):
            return getattr(dwave_topologies, topology_name)
    except Exception:
        pass

    # No match found
    raise ValueError(
        f"Could not parse topology name '{topology_name}'. "
        f"Expected formats:\n"
        f"  - Zephyr: Z(m,t) (e.g., 'Z(9,2)')\n"
        f"  - Hardware: Advantage2_system1.10 or Advantage2-system1.10\n"
        f"  - File path: path/to/topology.json or .json.gz\n"
        f"  - Named constant: ZEPHYR_Z9_T2_TOPOLOGY"
    )


