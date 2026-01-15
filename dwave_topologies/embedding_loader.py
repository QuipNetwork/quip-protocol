"""
Embedding loader utility for precomputed topology embeddings.

Loads precomputed embeddings from JSON files saved in:
    dwave_topologies/embeddings/{solver_name}/zephyr_z{m}_t{t}.embed.json.gz
"""

import os
import json
import gzip
from typing import Dict, List, Optional, Tuple


def load_embedding(
    topology_name: str,
    solver_name: str = "Advantage2_system1_10",
    embeddings_dir: Optional[str] = None
) -> Optional[Dict]:
    """
    Load a precomputed embedding from JSON file.

    Args:
        topology_name: Name like "zephyr_z10_t2" or "Z(10,2)"
        solver_name: Target solver name (default: "Advantage2_system1_10")
        embeddings_dir: Optional custom embeddings directory

    Returns:
        Dict with 'metadata', 'statistics', and 'embedding' keys,
        or None if not found.

    Example:
        embedding_data = load_embedding("Z(10,2)")
        if embedding_data:
            embedding = embedding_data['embedding']
            # Convert string keys back to ints if needed
            embedding = {int(k): v for k, v in embedding.items()}
    """
    # Normalize topology name to filename format
    if topology_name.startswith("Z(") and topology_name.endswith(")"):
        # Parse Z(m,t) format
        parts = topology_name[2:-1].split(",")
        m, t = int(parts[0].strip()), int(parts[1].strip())
        filename = f"zephyr_z{m}_t{t}.embed.json.gz"
    elif topology_name.startswith("zephyr_z"):
        # Already in filename format
        if not topology_name.endswith(".embed.json.gz"):
            # Add .embed.json.gz extension
            base = topology_name.replace(".json.gz", "").replace(".embed.json.gz", "")
            filename = f"{base}.embed.json.gz"
        else:
            filename = topology_name
    else:
        raise ValueError(f"Invalid topology name format: {topology_name}")

    # Determine base directory
    if embeddings_dir is None:
        # Default: relative to this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_dir = os.path.join(this_dir, "embeddings")

    # Build full path
    filepath = os.path.join(embeddings_dir, solver_name, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        return None

    # Load and decompress
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            embedding_data = json.load(f)
        return embedding_data
    except Exception as e:
        raise IOError(f"Failed to load embedding from {filepath}: {e}")


def get_embedding_dict(
    topology_name: str,
    solver_name: str = "Advantage2_system1_10",
    convert_keys_to_int: bool = True
) -> Optional[Dict]:
    """
    Load embedding and return just the embedding dict (not metadata).

    Args:
        topology_name: Name like "zephyr_z10_t2" or "Z(10,2)"
        solver_name: Target solver name
        convert_keys_to_int: If True, convert string keys to int

    Returns:
        Embedding dict {source_var: [target_qubits]} or None if not found.

    Example:
        embedding = get_embedding_dict("Z(10,2)")
        if embedding:
            # Ready to use with D-Wave sampler
            sampler = FixedEmbeddingComposite(qpu, embedding)
    """
    data = load_embedding(topology_name, solver_name)
    if data is None:
        return None

    embedding = data['embedding']

    if convert_keys_to_int:
        # Convert string keys back to integers
        embedding = {int(k): v for k, v in embedding.items()}

    return embedding


def list_available_embeddings(solver_name: str = "Advantage2_system1_10") -> List[Tuple[str, Dict]]:
    """
    List all available precomputed embeddings for a solver.

    Args:
        solver_name: Target solver name

    Returns:
        List of (filename, metadata) tuples

    Example:
        embeddings = list_available_embeddings()
        for filename, metadata in embeddings:
            print(f"{filename}: {metadata['description']}")
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(this_dir, "embeddings", solver_name)

    if not os.path.exists(embeddings_dir):
        return []

    results = []
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.embed.json.gz'):
            filepath = os.path.join(embeddings_dir, filename)
            try:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append((filename, data.get('metadata', {})))
            except Exception:
                continue

    return results


def embedding_exists(topology_name: str, solver_name: str = "Advantage2_system1_10") -> bool:
    """
    Check if a precomputed embedding exists.

    Args:
        topology_name: Name like "zephyr_z10_t2" or "Z(10,2)"
        solver_name: Target solver name

    Returns:
        True if embedding file exists, False otherwise

    Example:
        if embedding_exists("Z(10,2)"):
            embedding = get_embedding_dict("Z(10,2)")
        else:
            print("Need to compute embedding first")
    """
    return load_embedding(topology_name, solver_name) is not None
