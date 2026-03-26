#!/usr/bin/env python3
"""IBM QAOA baseline benchmarking tool.

Modeled on tools/qpu_baseline.py — runs QAOA solves against Qiskit's
AerSimulator (local quantum circuit simulation, no IBM account or API
keys required) to collect baseline metrics and prove the solver works.

Two modes
---------
--quick / standard / --extended / --stress
    Protocol-style problems (random h, J from generate_ising_model_from_nonce
    on a subgraph of the protocol topology).  AerSimulator can't handle the
    full ~4,580-qubit topology, so all modes extract a connected subgraph:
      --quick:     8 nodes
      standard:   10 nodes
      --extended: 14 nodes
      --stress:   28 nodes (memory stress test, ~4 GB state vector)
      (override with --subgraph-size)
    Proves the mining pipeline works end-to-end:
      nonce -> Ising problem -> solve_ising -> evaluate_sampleset -> MiningResult
    We don't know the optimal answer for random problems, so we can't
    measure exact solution quality — the point is to prove the pipeline
    runs without crashing and produces valid MiningResults.

--known-problems
    Tests against Ising problems with known optimal energies from two files:
      - tools/basic_ising_problems.py  (easy, 2-16 qubits)
      - tools/hard_ising_problems.py   (hard, 16-20 qubit spin glasses)
    Proves the solver finds good solutions:
      "optimal is -14.0, we found -13.5, that's 96.4%"
    Reports approximation ratios for each problem.  The easy problems
    should hit 1.0 (exact optimum).  The hard problems are designed to
    challenge low-depth QAOA — expect some ratios below 90%.

Both modes measure: energy levels, execution time, memory usage, and
solution diversity.

Getting Started
===============
Dependencies:
    pip install qiskit qiskit-aer dimod numpy scipy python-dotenv psutil

Run from the project root:
    python tools/ibm_qaoa_baseline.py --quick            # fast pipeline smoke test
    python tools/ibm_qaoa_baseline.py                    # standard pipeline test
    python tools/ibm_qaoa_baseline.py --known-problems   # solution quality check
    python tools/ibm_qaoa_baseline.py --extended         # thorough parameter sweep
    python tools/ibm_qaoa_baseline.py --stress           # memory stress test

What success looks like:
  --quick / standard / --extended:
    - Completes without errors
    - evaluate_sampleset returns a MiningResult (not None)
    - Energy scale depends on subgraph size (smaller graph = smaller energies)
    - --quick (8 nodes) solves in seconds
    - --extended (14 nodes) may take a few minutes per solve

  --stress:
    - 28-node subgraph, AerSimulator uses ~4 GB for the state vector
    - Reports peak memory usage — expect several GB
    - May take several minutes per solve
    - Useful for understanding RAM requirements at scale

  --known-problems:
    - Approximation ratios > 0.90 (finding >90% of optimal energy)
    - All problems return valid SampleSets
    - Small problems (2-16 qubits) solve in seconds

Output:
    Results are saved as JSON (auto-named or use --output).
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    evaluate_sampleset,
    calculate_diversity,
)
from shared.block_requirements import BlockRequirements
from shared.energy_utils import expected_solution_energy
from tools.basic_ising_problems import BASIC_ISING_PROBLEMS
from tools.hard_ising_problems import HARD_ISING_PROBLEMS

try:
    from QPU.IBM.ibm_qaoa_solver import QAOASolverWrapper
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

from dwave_topologies import DEFAULT_TOPOLOGY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_memory_mb():
    """Return peak process memory in MB.

    On Linux: resource.getrusage reports peak RSS (ru_maxrss).
    On macOS: same but in bytes instead of KB.
    On Windows: psutil.memory_info().peak_wset reports peak working set,
                which captures the high-water mark even after C++ memory
                (like AerSimulator's state vector) is freed.
    """
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = rusage.ru_maxrss
        if sys.platform == "darwin":
            return rss_kb / (1024 * 1024)  # bytes -> MB on macOS
        return rss_kb / 1024  # KB -> MB on Linux
    except ImportError:
        pass
    try:
        import psutil
        mem = psutil.Process(os.getpid()).memory_info()
        # peak_wset is Windows-only; falls back to rss on other platforms
        peak = getattr(mem, 'peak_wset', None) or mem.rss
        return peak / (1024 * 1024)
    except ImportError:
        return None


def _extract_subgraph(
    all_nodes: List[int],
    all_edges: List[Tuple[int, int]],
    size: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Extract a connected subgraph of ``size`` nodes via BFS.

    Starts from a random node and grows outward.  Returns
    (sub_nodes, sub_edges) where sub_edges only includes edges
    between nodes in the subgraph.
    """
    # Build adjacency list
    adj: Dict[int, List[int]] = {n: [] for n in all_nodes}
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)

    start = random.choice(all_nodes)
    visited = set()
    queue = [start]

    while queue and len(visited) < size:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    sub_nodes = sorted(visited)
    sub_node_set = set(sub_nodes)
    sub_edges = [(u, v) for u, v in all_edges
                 if u in sub_node_set and v in sub_node_set]
    return sub_nodes, sub_edges


def _load_known_problems(max_qubits: int = 20) -> List[Dict[str, Any]]:
    """Load known Ising problems from basic_ising_problems.py and hard_ising_problems.py.

    Filters to problems with <= max_qubits nodes since AerSimulator
    scales exponentially with qubit count.

    Each entry in BASIC_ISING_PROBLEMS is a tuple:
        (h, J, optimal_energy, description)
    """
    problems = []
    for idx, (h, J, optimal_energy, description) in enumerate(BASIC_ISING_PROBLEMS):
        nodes = sorted(h.keys())
        edges = list(J.keys())

        if len(nodes) > max_qubits:
            continue

        problems.append({
            'name': f'problem_{idx}_{len(nodes)}q',
            'description': description,
            'nodes': nodes,
            'edges': edges,
            'h': h,
            'J': J,
            'optimal_energy': optimal_energy,
        })

    # Add hard problems that QAOA is expected to struggle with
    for idx, (h, J, optimal_energy, description) in enumerate(HARD_ISING_PROBLEMS):
        nodes_h = sorted(h.keys())
        edges_h = list(J.keys())

        if len(nodes_h) > max_qubits:
            continue

        problems.append({
            'name': f'hard_{idx}_{len(nodes_h)}q',
            'description': description,
            'nodes': nodes_h,
            'edges': edges_h,
            'h': h,
            'J': J,
            'optimal_energy': optimal_energy,
        })

    return problems



# ---------------------------------------------------------------------------
# Mode 1: Pipeline test (--quick / standard / --extended)
# ---------------------------------------------------------------------------

def pipeline_test(
    timeout_minutes: float = 30.0,
    output_file: Optional[str] = None,
    p_values: Optional[List[int]] = None,
    optimizers: Optional[List[str]] = None,
    shots_list: Optional[List[int]] = None,
    num_solves: int = 3,
    subgraph_size: Optional[int] = None,
) -> Optional[Dict]:
    """Run protocol-style QAOA solves to prove the mining pipeline works.

    Extracts a connected subgraph from the protocol topology (AerSimulator
    can't handle the full ~4,580-qubit graph) and runs random nonce solves.
    Each solve goes through:
      nonce -> generate_ising_model_from_nonce -> solve_ising ->
      evaluate_sampleset -> MiningResult.

    We don't know the optimal energy for random problems, so this mode
    proves integration, not solution quality.

    Args:
        subgraph_size: Number of nodes to extract from the protocol
                       topology.  Defaults: --quick=8, standard=10,
                       --extended=14.
    """
    if p_values is None:
        p_values = [1, 2]
    if optimizers is None:
        optimizers = ['COBYLA']
    if shots_list is None:
        shots_list = [512, 1024]

    print("🔬 IBM QAOA Pipeline Test (protocol-style problems)")
    print("=" * 55)
    print(f"⏰ Timeout: {timeout_minutes} min | 🔁 Solves per config: {num_solves}")

    if not IBM_AVAILABLE:
        print("❌ IBM QAOA not available (missing qiskit / qiskit-aer)")
        return None

    nodes = DEFAULT_TOPOLOGY.nodes
    edges = DEFAULT_TOPOLOGY.edges

    if subgraph_size is not None:
        nodes, edges = _extract_subgraph(nodes, edges, subgraph_size)
        print(f"✂️ Subgraph: {len(nodes)} nodes, {len(edges)} edges "
              f"(from {len(DEFAULT_TOPOLOGY.nodes)}-node topology)")
    else:
        print(f"🕸️ Topology: {len(nodes)} nodes, {len(edges)} edges")

    expected_gse = expected_solution_energy(
        num_nodes=len(nodes), num_edges=len(edges)
    )
    print(f"📐 Expected GSE (empirical): {expected_gse:.1f}")

    mem_before = _get_memory_mb()
    if mem_before is not None:
        print(f"💾 Memory before solver init: {mem_before:.1f} MB")

    results = {
        'mode': 'pipeline',
        'timeout_minutes': float(timeout_minutes),
        'num_solves_per_config': num_solves,
        'topology': {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'expected_gse': expected_gse,
        },
        'tests': [],
    }

    configs = [
        {'p': p, 'optimizer': opt, 'shots': s}
        for p in p_values for opt in optimizers for s in shots_list
    ]

    print(f"\n🧪 {len(configs)} configs x {num_solves} solves "
          f"= {len(configs) * num_solves} total solves")
    print(f"  p:          {p_values}")
    print(f"  optimizers: {optimizers}")
    print(f"  shots:      {shots_list}")

    timeout_s = timeout_minutes * 60
    total_start = time.time()

    for cfg_idx, cfg in enumerate(configs):
        if time.time() - total_start > timeout_s:
            print(f"\n⏰ Timeout ({timeout_minutes} min) reached")
            break

        p, optimizer, shots = cfg['p'], cfg['optimizer'], cfg['shots']
        final_shots = 4 * shots

        print(f"\n--- Config {cfg_idx + 1}/{len(configs)}: "
              f"p={p}, optimizer={optimizer}, shots={shots} ---")

        try:
            solver = QAOASolverWrapper(
                nodes=nodes, edges=edges, backend=None,
                p=p, optimizer=optimizer, shots=shots,
                final_shots=final_shots,
            )
        except Exception as e:
            print(f"  ❌ Failed to build solver: {e}")
            continue

        mem_after_init = _get_memory_mb()
        if mem_after_init is not None:
            print(f"  💾 Memory after solver init: {mem_after_init:.1f} MB")

        solve_energies: List[float] = []
        solve_times: List[float] = []
        solve_diversities: List[float] = []
        solve_num_solutions: List[int] = []
        pipeline_pass_count = 0
        peak_mem = mem_after_init or 0.0

        for solve_idx in range(num_solves):
            if time.time() - total_start > timeout_s:
                print(f"  ⏰ Timeout reached during solve {solve_idx + 1}")
                break

            nonce = random.randint(0, 2**32 - 1)
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            print(f"  Solve {solve_idx + 1}/{num_solves}: nonce={nonce} ... ",
                  end="", flush=True)

            try:
                start = time.time()
                sampleset = solver.solve_ising(h, J)
                solve_time = time.time() - start

                if sampleset is None:
                    print("interrupted")
                    continue

                energies = list(sampleset.record.energy)
                min_energy = float(min(energies))
                avg_energy = float(sum(energies) / len(energies))
                solve_energies.append(min_energy)
                solve_times.append(solve_time)

                # Diversity
                solutions = list(sampleset.record.sample)
                pairs = sorted(zip(solutions, energies), key=lambda x: x[1])
                top_10 = [s for s, _ in pairs[:10]]
                diversity = calculate_diversity(top_10)
                solve_diversities.append(diversity)

                # Pipeline check: evaluate_sampleset -> MiningResult?
                requirements = BlockRequirements(
                    difficulty_energy=0.0,
                    min_diversity=0.1,
                    min_solutions=1,
                    timeout_to_difficulty_adjustment_decay=600,
                )
                salt = random.randbytes(32)
                prev_timestamp = int(time.time()) - 600
                mining_result = evaluate_sampleset(
                    sampleset, requirements, nodes, edges, nonce, salt,
                    prev_timestamp, start,
                    f"qaoa-baseline-{cfg_idx}", "IBM_QAOA",
                )

                num_valid = mining_result.num_valid if mining_result else 0
                solve_num_solutions.append(num_valid)
                pipeline_ok = mining_result is not None
                if pipeline_ok:
                    pipeline_pass_count += 1

                mem_now = _get_memory_mb()
                if mem_now is not None and mem_now > peak_mem:
                    peak_mem = mem_now

                status = "✅ MiningResult" if pipeline_ok else "⚠️  None"
                print(f"energy={min_energy:.1f}, avg={avg_energy:.1f}, "
                      f"diversity={diversity:.3f}, solutions={num_valid}, "
                      f"time={solve_time:.1f}s {status}")

            except Exception as e:
                print(f"error: {e}")
                continue

        if not solve_energies:
            continue

        best_energy = min(solve_energies)
        avg_time = sum(solve_times) / len(solve_times)
        avg_diversity = (sum(solve_diversities) / len(solve_diversities)
                         if solve_diversities else 0.0)

        print(f"\n  📊 Config summary:")
        print(f"    Best energy:       {best_energy:.1f}")
        print(f"    Avg solve time:    {avg_time:.1f}s ({avg_time/60:.1f} min)")
        print(f"    Avg diversity:     {avg_diversity:.3f}")
        print(f"    Pipeline pass:     {pipeline_pass_count}/{len(solve_energies)}")
        if peak_mem > 0:
            print(f"    Peak memory:       {peak_mem:.1f} MB")

        results['tests'].append({
            'p': p,
            'optimizer': optimizer,
            'shots': shots,
            'final_shots': final_shots,
            'num_solves': len(solve_energies),
            'best_energy': best_energy,
            'avg_min_energy': sum(solve_energies) / len(solve_energies),
            'all_min_energies': solve_energies,
            'avg_solve_time_seconds': avg_time,
            'all_solve_times': solve_times,
            'avg_diversity': avg_diversity,
            'avg_num_solutions': (sum(solve_num_solutions) / len(solve_num_solutions)
                                  if solve_num_solutions else 0),
            'pipeline_pass_rate': pipeline_pass_count / len(solve_energies),
            'peak_memory_mb': peak_mem if peak_mem > 0 else None,
        })

    # Summary
    total_runtime = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"📊 Pipeline Test Summary (total: {total_runtime/60:.1f} min)")
    print(f"{'=' * 55}")

    if results['tests']:
        best = min(results['tests'], key=lambda r: r['best_energy'])
        print(f"🏆 Best energy: {best['best_energy']:.1f}")
        print(f"  Config: p={best['p']}, {best['optimizer']}, "
              f"{best['shots']} shots")
        print(f"  Avg solve time: {best['avg_solve_time_seconds']:.1f}s")

        total_passes = sum(
            int(r['pipeline_pass_rate'] * r['num_solves'])
            for r in results['tests']
        )
        total_solves = sum(r['num_solves'] for r in results['tests'])
        print(f"\n✅ Pipeline pass rate: {total_passes}/{total_solves}")

        print(f"\n⏱️  Time vs Energy:")
        for r in results['tests']:
            print(f"  p={r['p']} {r['optimizer']:>10s} {r['shots']:5d} shots -> "
                  f"{r['best_energy']:8.1f} energy, "
                  f"{r['avg_solve_time_seconds']:6.1f}s avg, "
                  f"pass={r['pipeline_pass_rate']:.0%}")

        mem_values = [r['peak_memory_mb'] for r in results['tests']
                      if r['peak_memory_mb'] is not None]
        if mem_values:
            print(f"\n💾 Peak memory: {max(mem_values):.1f} MB")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


# ---------------------------------------------------------------------------
# Mode 2: Known-problems test (--known-problems)
# ---------------------------------------------------------------------------

def known_problems_test(
    output_file: Optional[str] = None,
    p: int = 1,
    optimizer: str = 'COBYLA',
    shots: int = 1024,
    num_solves: int = 3,
    max_qubits: int = 20,
) -> Optional[Dict]:
    """Test QAOA solution quality against problems with known optima.

    Loads easy problems from basic_ising_problems.py and hard problems
    from hard_ising_problems.py.  Runs num_solves independent solves per
    problem.  Reports approximation ratios:
      found_energy / optimal_energy
    Ratio of 1.0 = found exact ground state.  > 0.90 = good.
    """
    print("🔬 IBM QAOA Known-Problems Test (solution quality)")
    print("=" * 55)
    print(f"🔧 Solver: p={p}, optimizer={optimizer}, shots={shots}")
    print(f"🔁 Solves per problem: {num_solves}")
    print(f"📏 Max qubits: {max_qubits}")

    if not IBM_AVAILABLE:
        print("❌ IBM QAOA not available (missing qiskit / qiskit-aer)")
        return None

    print("\n📋 Loading problems from basic_ising_problems.py + hard_ising_problems.py...")
    problems = _load_known_problems(max_qubits=max_qubits)
    print(f"  {len(problems)} problems loaded "
          f"(filtered to <= {max_qubits} qubits)\n")

    results = {
        'mode': 'known_problems',
        'solver_config': {
            'p': p,
            'optimizer': optimizer,
            'shots': shots,
            'final_shots': 4 * shots,
        },
        'num_solves_per_problem': num_solves,
        'max_qubits': max_qubits,
        'problems': [],
    }

    total_start = time.time()

    for prob_idx, prob in enumerate(problems):
        name = prob['name']
        p_nodes = prob['nodes']
        p_edges = prob['edges']
        h = prob['h']
        J = prob['J']
        optimal = prob['optimal_energy']

        print(f"--- Problem {prob_idx + 1}/{len(problems)}: {name} ---")
        print(f"  {prob['description']}")
        print(f"  {len(p_nodes)} nodes, {len(p_edges)} edges, optimal = {optimal}")

        try:
            solver = QAOASolverWrapper(
                nodes=p_nodes, edges=p_edges, backend=None,
                p=p, optimizer=optimizer, shots=shots,
                final_shots=4 * shots,
            )
        except Exception as e:
            print(f"  ❌ Failed to build solver: {e}")
            continue

        solve_energies: List[float] = []
        solve_times: List[float] = []
        solve_ratios: List[float] = []
        peak_mem = _get_memory_mb() or 0.0

        for solve_idx in range(num_solves):
            print(f"  Solve {solve_idx + 1}/{num_solves} ... ", end="", flush=True)

            try:
                start = time.time()
                sampleset = solver.solve_ising(h, J)
                solve_time = time.time() - start

                if sampleset is None:
                    print("interrupted")
                    continue

                energies = list(sampleset.record.energy)
                min_energy = float(min(energies))
                solve_energies.append(min_energy)
                solve_times.append(solve_time)

                # Approximation ratio
                if optimal != 0:
                    ratio = min_energy / optimal
                else:
                    ratio = 1.0 if min_energy == 0 else 0.0
                solve_ratios.append(ratio)

                mem_now = _get_memory_mb()
                if mem_now is not None and mem_now > peak_mem:
                    peak_mem = mem_now

                gap = min_energy - optimal
                print(f"energy={min_energy:.1f}, optimal={optimal:.1f}, "
                      f"ratio={ratio:.3f}, gap={gap:+.1f}, "
                      f"time={solve_time:.2f}s")

            except Exception as e:
                print(f"error: {e}")
                continue

        if not solve_energies:
            continue

        best_energy = min(solve_energies)
        best_ratio = max(solve_ratios)
        avg_ratio = sum(solve_ratios) / len(solve_ratios)
        avg_time = sum(solve_times) / len(solve_times)

        if best_ratio >= 0.90:
            verdict = "✅🎉 Excellent!"
            verdict_label = "Excellent"
        elif best_ratio >= 0.80:
            verdict = "✅🌈 Very Good!"
            verdict_label = "Very Good"
        elif best_ratio >= 0.70:
            verdict = "✅🎖️ Good!"
            verdict_label = "Good"
        elif best_ratio >= 0.60:
            verdict = "✅⚡ Fair!"
            verdict_label = "Fair"
        else:
            verdict = "❌🥶 Poor!"
            verdict_label = "Poor"
        print(f"\n  📊 Summary: best={best_energy:.1f}, "
              f"best_ratio={best_ratio:.3f}, "
              f"avg_ratio={avg_ratio:.3f}, "
              f"avg_time={avg_time:.2f}s -> {verdict}\n")

        results['problems'].append({
            'name': name,
            'description': prob['description'],
            'num_nodes': len(p_nodes),
            'num_edges': len(p_edges),
            'optimal_energy': optimal,
            'best_energy': best_energy,
            'best_approx_ratio': best_ratio,
            'avg_approx_ratio': avg_ratio,
            'verdict': verdict_label,
            'all_min_energies': solve_energies,
            'all_approx_ratios': solve_ratios,
            'avg_solve_time_seconds': avg_time,
            'all_solve_times': solve_times,
            'peak_memory_mb': peak_mem if peak_mem > 0 else None,
        })

    # Summary
    total_runtime = time.time() - total_start
    print(f"{'=' * 55}")
    print(f"📊 Known-Problems Summary (total: {total_runtime:.1f}s)")
    print(f"{'=' * 55}")

    if results['problems']:
        print(f"\n{'Problem':<30s} {'Nodes':>5s} {'Optimal':>8s} "
              f"{'Found':>8s} {'Ratio':>7s} {'Time':>7s}")
        print(f"{'~' * 30} {'~' * 5} {'~' * 8} {'~' * 8} {'~' * 7} {'~' * 7}")
        for r in results['problems']:
            print(f"{r['name']:<30s} {r['num_nodes']:5d} "
                  f"{r['optimal_energy']:8.1f} "
                  f"{r['best_energy']:8.1f} "
                  f"{r['best_approx_ratio']:7.3f} "
                  f"{r['avg_solve_time_seconds']:6.2f}s")

        all_ratios = [r['best_approx_ratio'] for r in results['problems']]
        avg_overall = sum(all_ratios) / len(all_ratios)
        min_ratio = min(all_ratios)
        excellent = sum(1 for r in all_ratios if r >= 0.90)
        very_good = sum(1 for r in all_ratios if 0.80 <= r < 0.90)
        good = sum(1 for r in all_ratios if 0.70 <= r < 0.80)
        fair = sum(1 for r in all_ratios if 0.60 <= r < 0.70)
        poor = sum(1 for r in all_ratios if r < 0.60)

        print(f"\n🏆 Overall avg approx ratio: {avg_overall:.3f}")
        print(f"📉 Worst ratio:             {min_ratio:.3f}")
        print(f"🎉 Excellent (>= 90%%):      {excellent}/{len(all_ratios)}")
        print(f"🌈 Very Good (80-90%%):      {very_good}/{len(all_ratios)}")
        print(f"🎖️ Good (70-80%%):           {good}/{len(all_ratios)}")
        print(f"⚡ Fair (60-70%%):           {fair}/{len(all_ratios)}")
        print(f"🥶 Poor (< 60%%):            {poor}/{len(all_ratios)}")

        mem_values = [r['peak_memory_mb'] for r in results['problems']
                      if r['peak_memory_mb'] is not None]
        if mem_values:
            print(f"\n💾 Peak memory: {max(mem_values):.1f} MB")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='IBM QAOA baseline benchmarking tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/ibm_qaoa_baseline.py --quick            # fast pipeline smoke test
  python tools/ibm_qaoa_baseline.py                    # standard pipeline test
  python tools/ibm_qaoa_baseline.py --known-problems   # solution quality check
  python tools/ibm_qaoa_baseline.py --extended         # thorough parameter sweep
  python tools/ibm_qaoa_baseline.py --stress           # memory stress test (28 qubits)
  python tools/ibm_qaoa_baseline.py --quick -v         # show QAOA solver steps
  python tools/ibm_qaoa_baseline.py -p 1 2 3 --optimizer COBYLA SPSA
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--known-problems', action='store_true',
        help='Test against easy and hard problems with known optimal '
             'energies. Reports approximation ratios.',
    )
    mode_group.add_argument(
        '--quick', action='store_true',
        help='Quick pipeline smoke test: 8-node subgraph, p=1, COBYLA, '
             '512 shots, 1 solve.',
    )
    mode_group.add_argument(
        '--extended', action='store_true',
        help='Extended sweep: 14-node subgraph, p=1,2,3, COBYLA+SPSA, '
             'multiple shot counts, 5 solves.',
    )
    mode_group.add_argument(
        '--stress', action='store_true',
        help='Memory stress test: 28-node subgraph (~4 GB state vector). '
             'Requires 8+ GB RAM. p=1, COBYLA, 1024 shots, 1 solve.',
    )

    # Shared options
    parser.add_argument(
        '--timeout', '-t', type=float, default=30.0,
        help='Timeout in minutes for pipeline mode (default: 30.0)',
    )
    parser.add_argument(
        '-p', '--depths', type=int, nargs='+', default=None,
        help='QAOA circuit depths to test (default: 1 2)',
    )
    parser.add_argument(
        '--optimizer', type=str, nargs='+', default=None,
        help='Optimizers to test (COBYLA, NELDER_MEAD, POWELL, L_BFGS_B, SPSA)',
    )
    parser.add_argument(
        '--shots', type=int, nargs='+', default=None,
        help='Shot counts to test (default: 512 1024)',
    )
    parser.add_argument(
        '--num-solves', '-n', type=int, default=3,
        help='Independent solves per config/problem (default: 3)',
    )
    parser.add_argument(
        '--max-qubits', type=int, default=20,
        help='Max qubit count for --known-problems (default: 20). '
             'AerSimulator gets slow and may error beyond ~30 qubits.',
    )
    parser.add_argument(
        '--subgraph-size', type=int, default=None,
        help='Override the subgraph size (number of nodes extracted from '
             'the protocol topology). Defaults: --quick=8, standard=10, '
             '--extended=14.',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show QAOA solver steps (circuit building, optimization, sampling)',
    )

    args = parser.parse_args()

    # Configure logging — show QAOA solver steps if --verbose
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='  %(message)s',
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    timestamp = int(time.time())

    if args.known_problems:
        output_file = args.output or f"ibm_qaoa_known_problems_{timestamp}.json"
        p_val = (args.depths or [1])[0]
        opt = (args.optimizer or ['COBYLA'])[0]
        s = (args.shots or [1024])[0]

        results = known_problems_test(
            output_file=output_file,
            p=p_val,
            optimizer=opt,
            shots=s,
            num_solves=args.num_solves,
            max_qubits=args.max_qubits,
        )
    else:
        if args.quick:
            p_values = [1]
            optimizers = ['COBYLA']
            shots_list = [512]
            num_solves = 1
            timeout = 15.0
            subgraph_size = args.subgraph_size or 8
        elif args.extended:
            p_values = [1, 2, 3]
            optimizers = ['COBYLA', 'SPSA']
            shots_list = [512, 1024, 2048]
            num_solves = 5
            timeout = 120.0
            subgraph_size = args.subgraph_size or 14
        elif args.stress:
            p_values = [1]
            optimizers = ['COBYLA']
            shots_list = [1024]
            num_solves = 1
            timeout = 60.0
            subgraph_size = args.subgraph_size or 28
        else:
            p_values = args.depths or [1, 2]
            optimizers = args.optimizer or ['COBYLA']
            shots_list = args.shots or [512, 1024]
            num_solves = args.num_solves
            timeout = args.timeout
            subgraph_size = args.subgraph_size or 10

        output_file = args.output or f"ibm_qaoa_pipeline_{timestamp}.json"
        results = pipeline_test(
            timeout_minutes=timeout,
            output_file=output_file,
            p_values=p_values,
            optimizers=optimizers,
            shots_list=shots_list,
            num_solves=num_solves,
            subgraph_size=subgraph_size,
        )

    if results:
        print(f"\n✅ IBM QAOA baseline test complete!")
    else:
        print(f"\n❌ IBM QAOA baseline test failed!")


if __name__ == "__main__":
    main()