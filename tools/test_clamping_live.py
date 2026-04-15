#!/usr/bin/env python3
"""Live QPU test for variable clamping with a stale topology.

Loads the old System1.10 topology (4,589 nodes) as the "protocol reference"
and runs the full mining loop against the live QPU (Advantage2_system1,
~4,578 nodes). The mismatch exercises defect detection and variable clamping.

Each trial runs mine_block() until it finds a valid solution, then validates
the result against the full stale topology — exactly as a network validator
would.

Usage:
    python tools/test_clamping_live.py
    python tools/test_clamping_live.py --trials 5
    python tools/test_clamping_live.py --difficulty-energy -14900
"""

import argparse
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

from dwave_topologies.topologies.json_loader import load_json_topology
from shared.block import create_genesis_block, BlockRequirements
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    energy_of_solution,
    validate_quantum_proof,
)


@dataclass
class NodeInfo:
    """Minimal node info for mining."""
    miner_id: str


def run_trial(
    trial_num: int,
    miner,
    topology,
    difficulty_energy: float,
    timeout: float,
):
    """Run one mining trial and validate the result.

    Returns dict with trial results, or None on timeout.
    """
    prev_block = create_genesis_block()
    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=0.15,
        min_solutions=5,
        timeout_to_difficulty_adjustment_decay=0,
    )
    prev_block.next_block_requirements = requirements
    node_info = NodeInfo(miner_id=miner.miner_id)
    stop_event = multiprocessing.Event()

    print(f"\n{'=' * 60}")
    print(f"  Trial {trial_num}: target energy {difficulty_energy}")
    print(f"{'=' * 60}")

    start = time.time()
    result = miner.mine_block(
        prev_block=prev_block,
        node_info=node_info,
        requirements=requirements,
        prev_timestamp=prev_block.header.timestamp,
        stop_event=stop_event,
    )
    elapsed = time.time() - start

    if result is None:
        print(f"  TIMEOUT after {elapsed:.1f}s — no valid solution found")
        return None

    print(f"  Mined in {elapsed:.1f}s")
    print(f"  Energy: {result.energy:.1f}")
    print(f"  Solutions: {result.num_valid}")
    print(f"  Diversity: {result.diversity:.3f}")
    print(f"  Nonce: {result.nonce}")

    # --- Verification 1: All protocol nodes present ---
    expected_nodes = set(topology.nodes)
    actual_nodes = set(result.node_list)
    if actual_nodes != expected_nodes:
        missing = expected_nodes - actual_nodes
        extra = actual_nodes - expected_nodes
        print(f"  FAIL: Node list mismatch! "
              f"missing={len(missing)}, extra={len(extra)}")
        return {"trial": trial_num, "passed": False, "reason": "node_mismatch"}
    print(f"  PASS: All {len(expected_nodes)} protocol nodes in result")

    # --- Verification 2: Energy matches manual calculation ---
    h, J = generate_ising_model_from_nonce(
        result.nonce, result.node_list, result.edge_list,
    )
    for sol_idx, solution in enumerate(result.solutions):
        manual_e = energy_of_solution(
            solution, h, J, result.node_list,
        )
        if manual_e >= difficulty_energy:
            print(f"  FAIL: Solution {sol_idx} energy {manual_e:.1f} "
                  f">= threshold {difficulty_energy}")
            return {
                "trial": trial_num, "passed": False,
                "reason": "energy_above_threshold",
            }

    print(f"  PASS: All {len(result.solutions)} solution energies "
          f"below {difficulty_energy}")

    # --- Verification 3: Full validator check ---
    # Build a QuantumProof-like object for validate_quantum_proof
    @dataclass
    class MockProof:
        nonce: int
        salt: bytes
        nodes: list
        edges: list
        solutions: list
        mining_time: float
        h_values: list = None

    proof = MockProof(
        nonce=result.nonce,
        salt=result.salt,
        nodes=result.node_list,
        edges=result.edge_list,
        solutions=result.solutions,
        mining_time=result.mining_time,
    )

    valid = validate_quantum_proof(
        quantum_proof=proof,
        miner_id=miner.miner_id,
        requirements=requirements,
        block_index=prev_block.header.index + 1,
        previous_hash=prev_block.hash,
    )
    if not valid:
        print(f"  FAIL: validate_quantum_proof returned False")
        return {"trial": trial_num, "passed": False, "reason": "validation_failed"}
    print(f"  PASS: validate_quantum_proof succeeded")

    return {
        "trial": trial_num,
        "passed": True,
        "energy": result.energy,
        "solutions": result.num_valid,
        "diversity": result.diversity,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Live QPU clamping test — full mining loop with validation",
    )
    default_topo = str(
        Path(__file__).parent.parent
        / "dwave_topologies" / "topologies"
        / "advantage2_system1_10.json.gz"
    )
    parser.add_argument(
        "--topology", default=default_topo,
        help="Stale topology file (default: System1.10)",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of mining trials (default: 10)",
    )
    parser.add_argument(
        "--difficulty-energy", type=float, default=-14850.0,
        help="Target energy threshold (default: -14850.0)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.topology):
        print(f"Topology file not found: {args.topology}")
        sys.exit(1)

    # Load stale topology
    topo_dir = os.path.dirname(args.topology)
    topo_file = os.path.basename(args.topology)
    topology = load_json_topology(topo_file, topologies_dir=topo_dir)

    print(f"Stale topology: {topology.solver_name}")
    print(f"  Nodes: {topology.num_nodes}, Edges: {topology.num_edges}")
    print(f"Difficulty: {args.difficulty_energy}")
    print(f"Trials: {args.trials}")

    # Initialize QPU miner with stale topology
    from QPU.dwave_miner import DWaveMiner

    print("\nConnecting to live QPU...")
    miner = DWaveMiner(miner_id="clamping-test", topology=topology)
    n_defects = len(miner.sampler._defective_qubits)
    print(f"  Defective qubits: {n_defects}")
    if n_defects:
        print(f"  IDs: {miner.sampler._defective_qubits[:20]}"
              f"{'...' if n_defects > 20 else ''}")

    # Run trials
    results = []
    for i in range(1, args.trials + 1):
        trial = run_trial(
            trial_num=i,
            miner=miner,
            topology=topology,
            difficulty_energy=args.difficulty_energy,
            timeout=300.0,
        )
        results.append(trial)

    # Summary
    passed = [r for r in results if r and r["passed"]]
    failed = [r for r in results if r and not r["passed"]]
    timed_out = [r for r in results if r is None]

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(passed)} passed, {len(failed)} failed, "
          f"{len(timed_out)} timed out / {args.trials} total")
    print(f"{'=' * 60}")

    if passed:
        energies = [r["energy"] for r in passed]
        times = [r["time"] for r in passed]
        print(f"  Energy range: [{min(energies):.1f}, {max(energies):.1f}]")
        print(f"  Time range: [{min(times):.1f}s, {max(times):.1f}s]")
        print(f"  Avg time: {sum(times) / len(times):.1f}s")

    if failed:
        for r in failed:
            print(f"  Trial {r['trial']} FAILED: {r['reason']}")

    miner.sampler.close()

    if failed:
        sys.exit(1)
    if len(timed_out) == args.trials:
        print("  All trials timed out")
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
