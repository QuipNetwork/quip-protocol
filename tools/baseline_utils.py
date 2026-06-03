# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared utilities for baseline testing tools.

Common topology loading, GPU info, energy classification,
argparse setup, problem/result printing, config filtering,
and sampleset evaluation used by all *_baseline.py tools.
"""
import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from shared.miner_types import BlockRequirements
from shared.quantum_proof_of_work import (
    calculate_diversity,
    evaluate_sampleset,
)
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.embedded_topology import (
    create_embedded_topology,
)
from dwave_topologies.topologies.json_loader import (
    load_topology,
)

logger = logging.getLogger(__name__)


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


def print_problem_summary(h: dict, J: dict) -> None:
    """Print problem size and h-field distribution.

    Args:
        h: Ising h-field coefficients dict.
        J: Ising coupling coefficients dict.
    """
    h_vals_set = sorted(set(h.values()))
    h_counts = {v: list(h.values()).count(v) for v in h_vals_set}
    h_dist_str = ", ".join([
        f"{v}: {h_counts[v]} ({100 * h_counts[v] / len(h):.1f}%)"
        for v in h_vals_set
    ])
    print(f"Problem: {len(h)} variables, {len(J)} couplings")
    print(f"   h distribution: {h_dist_str}")


def print_results_summary(results: dict, title: str) -> None:
    """Print best energy and time-vs-energy table.

    Args:
        results: Results dict with 'tests' list. Each test must
            have 'min_energy', 'runtime_minutes', and at least
            one of 'num_sweeps' or 'num_reads'.
        title: Tool-specific summary title string.
    """
    total_runtime = results.get('_total_runtime_seconds', 0.0)
    print(f"\n{title} (total time: {total_runtime / 60:.1f} min):")
    print("=" * 50)

    tests = results.get('tests', [])
    if not tests:
        return

    best = min(tests, key=lambda r: r['min_energy'])
    print(f"Best energy: {best['min_energy']:.1f}")
    if 'num_sweeps' in best:
        print(
            f"   Required: {best['num_sweeps']} sweeps, "
            f"{best['runtime_minutes']:.1f} min"
        )
    elif 'num_reads' in best:
        print(
            f"   Required: {best['num_reads']} reads, "
            f"{best['runtime_minutes']:.1f} min"
        )

    print(f"\nTime vs Energy Performance:")
    for r in tests:
        quality = (
            f"({r['target_reached']})"
            if r['target_reached'] != 'none' else ""
        )
        print(
            f"  {r['runtime_minutes']:5.1f} min: "
            f"{r['min_energy']:7.1f} energy {quality}"
        )


def filter_configs_by_label(
    test_configs: List[Tuple],
    only_label: Optional[str],
) -> Optional[List[Tuple]]:
    """Filter test_configs to only matching label.

    Args:
        test_configs: List of (sweeps, reads, desc) tuples.
        only_label: Label to match (case-insensitive), or None.

    Returns:
        Filtered list, or None if label not found.
        Returns original list unchanged if only_label is None.
    """
    if not only_label:
        return test_configs
    available = [desc for _, _, desc in test_configs]
    filtered = [
        cfg for cfg in test_configs
        if cfg[2].lower() == only_label.lower()
    ]
    if not filtered:
        print(
            f"No test config matched --only {only_label!r}; "
            f"available: {available}"
        )
        return None
    return filtered


def evaluate_baseline_sampleset(
    sampleset: Any,
    nodes: list,
    edges: list,
    nonce: int,
    start_time: float,
    label: str,
    miner_type: str,
    salt: bytes,
) -> Dict[str, Any]:
    """Evaluate a sampleset against lenient mining requirements.

    Computes top-10 diversity, mining result, and returns a
    dict of evaluation fields suitable for inclusion in a
    test_result record.

    Args:
        sampleset: dimod SampleSet from a sampler.
        nodes: Topology node list used to generate the problem.
        edges: Topology edge list used to generate the problem.
        nonce: Problem nonce.
        start_time: Wall-clock time when sampling started.
        label: Identifier string for logging (e.g. "cpu-sa-256-64").
        miner_type: Miner type string for evaluate_sampleset
            (e.g. "CPU", "CUDA", "Metal").
        salt: Salt bytes for evaluate_sampleset.

    Returns:
        Dict with keys: diversity, diversity_top_10,
        num_solutions, meets_requirements.
    """
    requirements = BlockRequirements(
        difficulty_energy=0.0,
        min_diversity=0.1,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=600,
    )
    prev_timestamp = int(time.time()) - 600

    mining_result = evaluate_sampleset(
        sampleset, requirements, nodes, edges,
        nonce, salt, prev_timestamp, start_time,
        label, miner_type,
    )

    solutions = list(sampleset.record.sample)
    energies = list(sampleset.record.energy)
    pairs = sorted(zip(solutions, energies), key=lambda x: x[1])
    top_10 = [sol for sol, _ in pairs[:10]]
    top_10_diversity = float(calculate_diversity(top_10))

    print(f"  diversity (top 10) = {top_10_diversity:.3f}")

    diversity = 0.0
    num_solutions = 0
    meets_requirements = False

    if mining_result:
        diversity = mining_result.diversity
        num_solutions = mining_result.num_valid
        meets_requirements = True
        print(f"  num_solutions = {num_solutions}")
        print(f"  Meets mining requirements!")
    else:
        print(f"  Does not meet mining requirements")

    return {
        'diversity': float(diversity),
        'diversity_top_10': top_10_diversity,
        'num_solutions': int(num_solutions),
        'meets_requirements': bool(meets_requirements),
    }
