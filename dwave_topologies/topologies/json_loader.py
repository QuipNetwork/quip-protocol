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
from typing import List, Tuple, Dict, Any
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
        from_embeddings: If True, load from embeddings/Advantage2_system1_7/. Otherwise load from topologies/.

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
            # Mined topologies are in embeddings/Advantage2_system1_7/
            parent_dir = os.path.dirname(current_dir)
            topologies_dir = os.path.join(parent_dir, 'embeddings', 'Advantage2_system1_7')
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


