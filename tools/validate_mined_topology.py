#!/usr/bin/env python3
"""
Validate that mined topologies are valid subgraphs of the Advantage2-System1.6 topology.

This tool verifies that:
1. All nodes in the mined topology exist in the real QPU topology
2. All edges in the mined topology exist in the real QPU topology
3. No extra nodes or edges are present that wouldn't work on actual hardware

Usage:
    python tools/validate_mined_topology.py zephyr_z10_t2.json.gz
    python tools/validate_mined_topology.py zephyr_z11_t4.json.gz
    python tools/validate_mined_topology.py --all  # Validate all mined topologies
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwave_topologies.topologies.json_loader import load_json_topology


def validate_topology(mined_topology_file: str, verbose: bool = True) -> bool:
    """
    Validate that a mined topology is a valid subgraph of Advantage2-System1.6.

    Args:
        mined_topology_file: Filename of the mined topology (e.g., 'zephyr_z10_t2.json.gz')
        verbose: If True, print detailed validation information

    Returns:
        True if topology is valid, False otherwise
    """
    # Load the real QPU topology
    qpu_topology = load_json_topology('advantage2_system1_6.json', from_embeddings=False)
    qpu_nodes = set(qpu_topology.nodes)
    qpu_edges = set((min(u, v), max(u, v)) for u, v in qpu_topology.edges)  # Normalize edge direction

    # Load the mined topology
    mined_topology = load_json_topology(mined_topology_file, from_embeddings=True)
    mined_nodes = set(mined_topology.nodes)
    mined_edges = set((min(u, v), max(u, v)) for u, v in mined_topology.edges)  # Normalize edge direction

    if verbose:
        print(f"\n{'='*80}")
        print(f"Validating: {mined_topology_file}")
        print(f"{'='*80}")
        print(f"\nReal QPU Topology (Advantage2-System1.6):")
        print(f"  Nodes: {len(qpu_nodes)}")
        print(f"  Edges: {len(qpu_edges)}")
        print(f"\nMined Topology:")
        print(f"  Name: {mined_topology.solver_name}")
        print(f"  Type: {mined_topology.topology_type}")
        print(f"  Shape: {mined_topology.topology_shape}")
        print(f"  Nodes: {len(mined_nodes)}")
        print(f"  Edges: {len(mined_edges)}")
        print(f"  Avg degree: {2 * len(mined_edges) / len(mined_nodes):.2f}")
        print(f"\nUtilization:")
        print(f"  Nodes: {len(mined_nodes)} / {len(qpu_nodes)} ({100 * len(mined_nodes) / len(qpu_nodes):.1f}%)")
        print(f"  Edges: {len(mined_edges)} / {len(qpu_edges)} ({100 * len(mined_edges) / len(qpu_edges):.1f}%)")

    # Validate nodes
    invalid_nodes = mined_nodes - qpu_nodes
    if invalid_nodes:
        print(f"\n❌ VALIDATION FAILED: Found {len(invalid_nodes)} nodes not in QPU topology:")
        if verbose and len(invalid_nodes) <= 10:
            print(f"  Invalid nodes: {sorted(invalid_nodes)}")
        elif verbose:
            print(f"  First 10 invalid nodes: {sorted(list(invalid_nodes))[:10]}")
        return False

    # Validate edges
    invalid_edges = mined_edges - qpu_edges
    if invalid_edges:
        print(f"\n❌ VALIDATION FAILED: Found {len(invalid_edges)} edges not in QPU topology:")
        if verbose and len(invalid_edges) <= 10:
            print(f"  Invalid edges: {sorted(invalid_edges)}")
        elif verbose:
            print(f"  First 10 invalid edges: {sorted(list(invalid_edges))[:10]}")
        return False

    # Success!
    if verbose:
        print(f"\n✅ VALIDATION PASSED: All nodes and edges are valid!")
        print(f"   This topology is a proper subgraph of Advantage2-System1.6")
        print(f"   and can be used for quantum annealing on the actual hardware.\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate mined topologies against Advantage2-System1.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific topology
  python tools/validate_mined_topology.py zephyr_z10_t2.json.gz

  # Validate all mined topologies
  python tools/validate_mined_topology.py --all

  # Quiet mode (only show pass/fail)
  python tools/validate_mined_topology.py zephyr_z11_t4.json.gz --quiet
        """
    )

    parser.add_argument(
        'topology',
        nargs='?',
        help='Mined topology filename (e.g., zephyr_z10_t2.json.gz)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all mined topologies in embeddings folder'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (only show pass/fail)'
    )

    args = parser.parse_args()

    if not args.topology and not args.all:
        parser.error("Either provide a topology filename or use --all")

    # Determine which topologies to validate
    if args.all:
        embeddings_dir = Path(__file__).parent.parent / 'dwave_topologies' / 'embeddings' / 'Advantage2_system1_6'
        topology_files = sorted(embeddings_dir.glob('*.json.gz'))
        if not topology_files:
            print("No mined topologies found in embeddings directory")
            return 1
        topology_files = [f.name for f in topology_files]
    else:
        topology_files = [args.topology]

    # Validate each topology
    results = {}
    for topology_file in topology_files:
        try:
            valid = validate_topology(topology_file, verbose=not args.quiet)
            results[topology_file] = valid
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            results[topology_file] = False
        except Exception as e:
            print(f"\n❌ ERROR validating {topology_file}: {e}")
            results[topology_file] = False

    # Print summary if validating multiple files
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        for topology_file, valid in results.items():
            status = "✅ PASS" if valid else "❌ FAIL"
            print(f"  {status}: {topology_file}")

        passed = sum(results.values())
        total = len(results)
        print(f"\nTotal: {passed}/{total} topologies valid")

    # Exit with error code if any validation failed
    if not all(results.values()):
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
