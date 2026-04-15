#!/usr/bin/env python3
"""Live QPU test for variable clamping with a stale topology.

Loads an old topology (System1.10, 4589 nodes) as the "protocol reference"
and connects to the live QPU (Advantage2_system1, ~4578 nodes). The mismatch
exercises the defect detection and variable clamping code path.

Usage:
    python tools/test_clamping_live.py
    python tools/test_clamping_live.py --topology /tmp/advantage2_system1_10.json.gz
    python tools/test_clamping_live.py --num-reads 20 --num-samples 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import dimod

from dwave_topologies.topologies.json_loader import load_json_topology
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    energy_of_solution,
)


def main():
    parser = argparse.ArgumentParser(
        description="Live QPU test for variable clamping",
    )
    default_topo = str(
        Path(__file__).parent.parent
        / "dwave_topologies" / "topologies"
        / "advantage2_system1_10.json.gz"
    )
    parser.add_argument(
        "--topology",
        default=default_topo,
        help="Path to stale topology file (default: System1.10 in repo)",
    )
    parser.add_argument(
        "--num-reads", type=int, default=10,
        help="QPU reads per sample call (default: 10)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of sample calls to make (default: 3)",
    )
    parser.add_argument(
        "--annealing-time", type=float, default=20.0,
        help="Annealing time in microseconds (default: 20.0)",
    )
    args = parser.parse_args()

    # Check topology file exists
    if not os.path.exists(args.topology):
        print(f"Topology file not found: {args.topology}")
        sys.exit(1)

    # Load stale topology
    topo_dir = os.path.dirname(args.topology)
    topo_file = os.path.basename(args.topology)
    topology = load_json_topology(topo_file, topologies_dir=topo_dir)
    print(f"Loaded stale topology: {topology.solver_name}")
    print(f"  Nodes: {topology.num_nodes}")
    print(f"  Edges: {topology.num_edges}")
    print()

    # Connect to live QPU with stale topology
    from QPU.dwave_sampler import DWaveSamplerWrapper

    print("Connecting to live QPU...")
    sampler = DWaveSamplerWrapper(topology=topology)
    print(f"  Connected to: {sampler.topology_name}")
    print(f"  Defective qubits: {len(sampler._defective_qubits)}")
    if sampler._defective_qubits:
        print(f"  First 20: {sampler._defective_qubits[:20]}")
    print(f"  Protocol nodes: {len(sampler.nodes)}")
    print(f"  Protocol edges: {len(sampler.edges)}")
    print()

    # Run sample calls
    nodes = sampler.nodes
    edges = sampler.edges

    for i in range(args.num_samples):
        nonce = 42 + i * 1000
        print(f"--- Sample {i + 1}/{args.num_samples} (nonce={nonce}) ---")

        # Generate full-topology Ising problem
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        print(f"  Ising model: {len(h)} h-fields, {len(J)} couplings")

        # Sample with clamping
        t0 = time.time()
        sampleset = sampler.sample_ising(
            h, J,
            num_reads=args.num_reads,
            annealing_time=args.annealing_time,
            nonce_seed=nonce,
        )
        elapsed = time.time() - t0

        # Verify results
        n_samples = len(sampleset)
        n_vars = len(sampleset.variables)
        energies = list(sampleset.record.energy)
        best_energy = min(energies)
        worst_energy = max(energies)

        print(f"  Samples returned: {n_samples}")
        print(f"  Variables per sample: {n_vars}")
        print(f"  Energy range: [{best_energy:.1f}, {worst_energy:.1f}]")
        print(f"  QPU + overhead: {elapsed:.2f}s")

        # Verify all protocol nodes are present
        expected_vars = set(nodes)
        actual_vars = set(sampleset.variables)
        if actual_vars != expected_vars:
            missing = expected_vars - actual_vars
            extra = actual_vars - expected_vars
            print(f"  FAIL: Variable mismatch!")
            print(f"    Missing: {len(missing)} vars")
            print(f"    Extra: {len(extra)} vars")
            sys.exit(1)
        else:
            print(f"  PASS: All {n_vars} protocol variables present")

        # Verify energy against manual calculation
        best_sample = dict(sampleset.first.sample)
        manual_energy = energy_of_solution(
            [best_sample[n] for n in nodes], h, J, nodes,
        )
        energy_diff = abs(sampleset.first.energy - manual_energy)
        if energy_diff > 0.01:
            print(f"  FAIL: Energy mismatch! "
                  f"sampleset={sampleset.first.energy:.4f}, "
                  f"manual={manual_energy:.4f}, "
                  f"diff={energy_diff:.4f}")
            sys.exit(1)
        else:
            print(f"  PASS: Energy verified (diff={energy_diff:.6f})")

        # Show timing if available
        if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
            timing = sampleset.info['timing']
            qpu_time = timing.get('qpu_access_time', 0)
            print(f"  QPU access time: {qpu_time}μs")

        print()

    # Summary
    print("=" * 50)
    print("ALL CHECKS PASSED")
    print(f"  Stale topology: {topology.solver_name} "
          f"({topology.num_nodes} nodes)")
    print(f"  Defective qubits clamped: "
          f"{len(sampler._defective_qubits)}")
    print(f"  Samples verified: {args.num_samples} x "
          f"{args.num_reads} reads")
    print("=" * 50)

    sampler.close()


if __name__ == "__main__":
    main()
