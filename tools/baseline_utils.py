# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared utilities for CUDA baseline testing tools.

Common topology loading, GPU info, energy classification,
and argparse setup used by cuda_baseline.py and
cuda_gibbs_baseline.py.
"""
import argparse
from typing import List, Optional, Tuple

from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.embedded_topology import (
    create_embedded_topology,
)
from dwave_topologies.topologies.json_loader import (
    load_topology,
)


def load_baseline_topology(
    topology_arg: Optional[str] = None,
    embedding_arg: Optional[str] = None,
) -> Tuple[List[int], list, str]:
    """Load topology from arg, embedding, or default.

    Args:
        topology_arg: Path to topology file, hardware name,
            or Zephyr format. Takes precedence.
        embedding_arg: Embedded topology spec (e.g., "Z(9,2)").

    Returns:
        (nodes, edges, description) tuple.
    """
    if topology_arg:
        if topology_arg.endswith('.embed.json.gz'):
            import os
            filename = os.path.basename(topology_arg)
            parts = filename.replace(
                "zephyr_z", "",
            ).replace(".embed.json.gz", "").split("_t")
            topology_name = f"Z({parts[0]},{parts[1]})"
            embedded = create_embedded_topology(
                topology_name,
            )
            nodes = embedded.nodes
            edges = embedded.edges
            desc = (
                f"{topology_name} embedded "
                f"({len(nodes)} qubits, "
                f"{len(edges)} couplers)"
            )
        else:
            topo_obj = load_topology(topology_arg)
            nodes = (
                list(topo_obj.graph.nodes)
                if hasattr(topo_obj, 'graph')
                else topo_obj.nodes
            )
            edges = (
                list(topo_obj.graph.edges)
                if hasattr(topo_obj, 'graph')
                else topo_obj.edges
            )
            topology_name = getattr(
                topo_obj, 'solver_name', 'unknown',
            )
            desc = (
                f"{topology_name} "
                f"({len(nodes)} nodes, "
                f"{len(edges)} edges)"
            )
    elif embedding_arg:
        embedded = create_embedded_topology(embedding_arg)
        nodes = embedded.nodes
        edges = embedded.edges
        desc = (
            f"{embedding_arg} embedded "
            f"({len(nodes)} qubits, "
            f"{len(edges)} couplers)"
        )
    else:
        topo_obj = DEFAULT_TOPOLOGY
        nodes = list(topo_obj.graph.nodes)
        edges = list(topo_obj.graph.edges)
        desc = (
            f"{topo_obj.solver_name} "
            f"({len(nodes)} nodes, {len(edges)} edges)"
        )

    return nodes, edges, desc


def get_gpu_info() -> Tuple[int, str]:
    """Query GPU SM count and name.

    Returns:
        (num_sms, gpu_name) tuple.
    """
    import cupy as cp
    dev = cp.cuda.Device()
    num_sms = dev.attributes['MultiProcessorCount']
    name = cp.cuda.runtime.getDeviceProperties(
        dev.id,
    )['name']
    if isinstance(name, bytes):
        name = name.decode()
    return num_sms, name


def classify_energy(min_energy: float) -> str:
    """Classify energy quality tier.

    Args:
        min_energy: Minimum energy achieved.

    Returns:
        Quality tier string: "excellent", "very_good",
        "good", "fair", or "none".
    """
    if min_energy <= -15650:
        return "excellent"
    if min_energy <= -15500:
        return "very_good"
    if min_energy <= -15400:
        return "good"
    if min_energy <= -15300:
        return "fair"
    return "none"


def build_baseline_argparser(
    description: str,
) -> argparse.ArgumentParser:
    """Build shared argparse for baseline tools.

    Args:
        description: Tool description string.

    Returns:
        ArgumentParser with common arguments.
    """
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        '--timeout', '-t', type=float, default=10.0,
        help='Timeout in minutes (default: 10.0)',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (only Light test)',
    )
    parser.add_argument(
        '--extended', action='store_true',
        help='Extended test mode (30 minute timeout)',
    )
    parser.add_argument(
        '--only', type=str,
        help='Run only this config label',
    )
    parser.add_argument(
        '--h-values', type=str, default='-1,0,1',
        help=(
            'Comma-separated h values '
            '(default: -1,0,1)'
        ),
    )
    parser.add_argument(
        '--topology', type=str,
        help=(
            'Topology: file path, hardware name, '
            'or Zephyr format'
        ),
    )
    return parser
