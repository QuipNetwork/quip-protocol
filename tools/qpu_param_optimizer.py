#!/usr/bin/env python3
"""QPU parameter optimization tool for finding optimal QPU parameters.

This tool systematically tests different parameter combinations to find optimal
settings for minimizing Ising model energies on D-Wave Advantage2 quantum annealers.

Supports multi-solver discovery and sweeps of: annealing_time, num_reads,
chain_strength_multiplier, reduce_intersample_correlation, reinitialize_state.
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
from typing import Any as AnyType
from typing import Tuple as TupleType

import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from shared.quantum_proof_of_work import generate_ising_model_from_nonce, calculate_diversity

try:
    from QPU.dwave_sampler import DWaveSamplerWrapper, EmbeddedFuture
    QPU_AVAILABLE = True
except ImportError:
    QPU_AVAILABLE = False


class QuotaExhaustedError(Exception):
    """Raised when D-Wave solver quota is exhausted."""
    pass


def is_quota_exhausted_error(error: Exception) -> bool:
    """Check if an exception indicates quota exhaustion."""
    error_str = str(error).lower()
    return "insufficient remaining solver access time" in error_str


DWAVE_REGIONS = ['na-west-1', 'na-east-1', 'eu-central-1']


def discover_qpu_solvers(
    regions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Discover all available QPU solvers across D-Wave regions."""
    from dwave.cloud import Client

    if regions is None:
        regions = DWAVE_REGIONS

    qpu_solvers = []
    seen_names: set = set()

    for region in regions:
        try:
            with Client.from_config(region=region) as client:
                solvers = client.get_solvers()
        except Exception as e:
            print(f"  Warning: Could not query region {region}: {e}")
            continue

        for solver in solvers:
            props = solver.properties
            if props.get('category', '') != 'qpu':
                continue
            name = solver.name
            if name in seen_names:
                continue
            seen_names.add(name)
            qpu_solvers.append({
                'name': name,
                'num_qubits': props.get('num_qubits', 0),
                'topology_type': props.get('topology', {}).get('type', 'unknown'),
                'chip_id': props.get('chip_id', name),
                'region': region,
            })

    return qpu_solvers


@dataclass
class PendingJob:
    """Tracks a pending QPU job with its associated metadata."""
    future: Any  # Future or EmbeddedFuture
    ising_seed: int
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    submit_time: float
    num_reads: int
    annealing_time: float
    test_id: str
    solver_name: str = ""
    chain_strength_multiplier: float = 1.5
    reduce_intersample_correlation: Optional[bool] = None
    reinitialize_state: Optional[bool] = None


# CSV column names (extended with new parameter columns)
CSV_COLUMNS = [
    'solver_name',
    'test_id', 'ising_seed', 'start_time', 'end_time',
    'annealing_time', 'num_reads',
    'chain_strength_multiplier',
    'reduce_intersample_correlation',
    'reinitialize_state',
    'min_energy', '5_smallest_energy_mean', '5_smallest_energy_median',
    '5_smallest_energy_stdev', '5_smallest_energy_variance',
    'diversity', 'qpu_time',
    'chain_break_fraction',
]

# Fixed seeds for reproducibility - same 1024 seeds used for ALL configurations
FIXED_SEEDS = list(range(1024))


def generate_test_id(
    num_reads: int,
    annealing_time: float,
    chain_strength_mult: float = 1.5,
    reduce_intersample: Optional[bool] = None,
    reinitialize: Optional[bool] = None,
) -> str:
    """Generate parseable test_id from parameters.

    Appends non-default parameter tags to keep old IDs compatible.
    """
    base = f"quip_{num_reads}sweep_{int(round(annealing_time))}time"
    if chain_strength_mult != 1.5:
        base += f"_cs{int(round(chain_strength_mult * 10))}"
    if reduce_intersample is not None:
        base += f"_ric{int(reduce_intersample)}"
    if reinitialize is not None:
        base += f"_rs{int(reinitialize)}"
    return base


def parse_test_id(test_id: str) -> Tuple[int, float]:
    """Parse test_id to extract num_reads and annealing_time.

    Handles both old format (quip_32sweep_10time) and new format with suffixes.
    """
    # Strip known suffixes before parsing base
    base = test_id
    for suffix_prefix in ('_cs', '_ric', '_rs'):
        idx = base.find(suffix_prefix)
        if idx != -1:
            base = base[:idx]
    parts = base.replace('quip_', '').replace('time', '').split('sweep_')
    num_reads = int(parts[0])
    annealing_time = float(parts[1])
    return num_reads, annealing_time


def normalize_test_id(test_id: str) -> str:
    """Normalize a test_id to current format (integer annealing_time).

    Preserves suffix tags (_cs, _ric, _rs) while normalizing the base.
    """
    try:
        # Extract suffix tags
        suffix = ""
        base = test_id
        for suffix_prefix in ('_cs', '_ric', '_rs'):
            idx = base.find(suffix_prefix)
            if idx != -1:
                suffix += base[idx:]
                base = base[:idx]
                break  # suffixes are contiguous after the base
        # Re-extract suffix from remaining if multiple
        if suffix:
            # All suffixes are in the suffix string already
            pass

        num_reads, annealing_time = parse_test_id(base)
        normalized_base = f"quip_{num_reads}sweep_{int(round(annealing_time))}time"
        return normalized_base + suffix
    except (ValueError, IndexError):
        return test_id  # Return unchanged if can't parse


def load_completed_tests(csv_path: str) -> set:
    """Load completed (test_id, ising_seed) pairs from existing CSV for recovery.

    Normalizes test_ids to handle old format (10.5) vs new format (10).
    """
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize test_id to current format
                normalized_id = normalize_test_id(row['test_id'])
                completed.add((normalized_id, int(row['ising_seed'])))
    return completed


def append_csv_row(csv_path: str, row: Dict[str, Any], write_header: bool = False):
    """Append a single row to CSV file."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header or not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_solver_properties(sampler: DWaveSamplerWrapper) -> Dict[str, Any]:
    """Query solver for parameter limits."""
    props = sampler.properties

    return {
        'annealing_time_range': props.get('annealing_time_range', [0.5, 2000.0]),
        'num_reads_range': props.get('num_reads_range', [1, 10000]),
        'default_annealing_time': props.get('default_annealing_time', 20.0),
        'problem_run_duration_range': props.get('problem_run_duration_range', [0, 1000000]),
        'chip_id': props.get('chip_id', 'unknown'),
        'num_qubits': len(sampler.nodes),
        'num_couplers': len(sampler.edges),
        # Timing overhead parameters
        'default_programming_thermalization': props.get('default_programming_thermalization', 1000.0),
        'default_readout_thermalization': props.get('default_readout_thermalization', 0.0),
        'readout_time_per_sample': props.get('readout_time_per_sample', 123.0),  # Typical value
    }


def estimate_qpu_time(
    annealing_time: float,
    num_reads: int,
    solver_props: Dict[str, Any]
) -> float:
    """Estimate total QPU access time for a job.

    Formula: programming_time + (annealing_time + readout_time + thermalization) × num_reads

    Returns estimated time in microseconds.
    """
    programming_time = solver_props.get('default_programming_thermalization', 1000.0)
    readout_per_sample = solver_props.get('readout_time_per_sample', 123.0)
    readout_therm = solver_props.get('default_readout_thermalization', 0.0)

    # Per-read time includes annealing + readout + thermalization overhead
    per_read_time = annealing_time + readout_per_sample + readout_therm

    # Total time
    total_time = programming_time + (per_read_time * num_reads)

    return total_time


# Default QPU time budget: 250ms (250,000 µs)
# Jobs taking longer are typically wasteful for optimization
DEFAULT_QPU_BUDGET_US = 250_000


def validate_configuration(
    annealing_time: float,
    num_reads: int,
    solver_props: Dict[str, Any],
    qpu_budget: float = DEFAULT_QPU_BUDGET_US
) -> Tuple[bool, float]:
    """Check if a configuration will fit within QPU time budget.

    Args:
        annealing_time: Annealing time in microseconds
        num_reads: Number of reads per job
        solver_props: Solver properties dict
        qpu_budget: Maximum QPU time budget in microseconds (default: 250ms)

    Returns:
        (is_valid, estimated_time) tuple
    """
    estimated_time = estimate_qpu_time(annealing_time, num_reads, solver_props)

    # Also respect hardware limit
    hw_max = solver_props['problem_run_duration_range'][1]
    effective_max = min(qpu_budget, hw_max)

    return estimated_time <= effective_max, estimated_time


def max_annealing_time_for_reads(
    num_reads: int,
    solver_props: Dict[str, Any],
    qpu_budget: float = DEFAULT_QPU_BUDGET_US
) -> float:
    """Calculate maximum annealing_time that fits within QPU budget for given num_reads.

    Solves: programming_time + (annealing_time + overhead) × num_reads <= qpu_budget
    For: annealing_time
    """
    # Respect hardware limit too
    hw_max_duration = solver_props['problem_run_duration_range'][1]
    max_duration = min(qpu_budget, hw_max_duration)

    programming_time = solver_props.get('default_programming_thermalization', 1000.0)
    readout_per_sample = solver_props.get('readout_time_per_sample', 123.0)
    readout_therm = solver_props.get('default_readout_thermalization', 0.0)

    overhead_per_read = readout_per_sample + readout_therm

    # max_duration = programming_time + (annealing_time + overhead) × num_reads
    # annealing_time = (max_duration - programming_time) / num_reads - overhead
    available = max_duration - programming_time
    max_time = (available / num_reads) - overhead_per_read

    # Cap at hardware limit
    hw_max = solver_props['annealing_time_range'][1]
    return min(max(max_time, 0), hw_max)


def max_num_reads_for_annealing_time(
    annealing_time: float,
    solver_props: Dict[str, Any],
    qpu_budget: float = DEFAULT_QPU_BUDGET_US
) -> int:
    """Calculate maximum num_reads that fits within QPU budget for given annealing_time.

    Solves: programming_time + (annealing_time + overhead) × num_reads <= qpu_budget
    For: num_reads
    """
    # Respect hardware limit too
    hw_max_duration = solver_props['problem_run_duration_range'][1]
    max_duration = min(qpu_budget, hw_max_duration)

    programming_time = solver_props.get('default_programming_thermalization', 1000.0)
    readout_per_sample = solver_props.get('readout_time_per_sample', 123.0)
    readout_therm = solver_props.get('default_readout_thermalization', 0.0)

    per_read_time = annealing_time + readout_per_sample + readout_therm

    # max_duration = programming_time + per_read_time × num_reads
    # num_reads = (max_duration - programming_time) / per_read_time
    available = max_duration - programming_time
    max_reads = int(available / per_read_time)

    # Cap at hardware limit
    hw_max = solver_props['num_reads_range'][1]
    return min(max(max_reads, 1), hw_max)


def submit_async_job(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    ising_seed: int,
    num_reads: int,
    annealing_time: float,
    test_id: str,
    solver_name: str = "",
    chain_strength_multiplier: float = 1.5,
    reduce_intersample_correlation: Optional[bool] = None,
    reinitialize_state: Optional[bool] = None,
) -> PendingJob:
    """Submit a single QPU job asynchronously.

    Returns a PendingJob with the future and metadata.
    """
    # Generate Ising model from seed
    h, J = generate_ising_model_from_nonce(ising_seed, nodes, edges)

    h_cast = cast(Mapping[AnyType, float], h)
    J_cast = cast(Mapping[TupleType[AnyType, AnyType], float], J)

    # Build sample kwargs
    sample_kwargs: Dict[str, Any] = {
        'num_reads': num_reads,
        'annealing_time': annealing_time,
        'answer_mode': 'raw',
        'chain_strength_multiplier': chain_strength_multiplier,
    }
    if reduce_intersample_correlation is not None:
        sample_kwargs['reduce_intersample_correlation'] = reduce_intersample_correlation
    if reinitialize_state is not None:
        sample_kwargs['reinitialize_state'] = reinitialize_state

    # Submit asynchronously (non-blocking)
    future = sampler.sample_ising_async(h_cast, J_cast, **sample_kwargs)

    return PendingJob(
        future=future,
        ising_seed=ising_seed,
        h=h,
        J=J,
        submit_time=time.time(),
        num_reads=num_reads,
        annealing_time=annealing_time,
        test_id=test_id,
        solver_name=solver_name,
        chain_strength_multiplier=chain_strength_multiplier,
        reduce_intersample_correlation=reduce_intersample_correlation,
        reinitialize_state=reinitialize_state,
    )


def process_completed_job(job: PendingJob) -> Dict[str, Any]:
    """Process a completed job and extract results.

    Returns a dict with all CSV columns populated.
    """
    end_time = datetime.now(timezone.utc).isoformat()
    start_time_iso = datetime.fromtimestamp(job.submit_time, tz=timezone.utc).isoformat()

    # Get the sampleset (should be ready since done() returned True)
    sampleset = job.future.sampleset

    # Extract energies
    energies = list(sampleset.record.energy)
    sorted_energies = sorted(energies)

    # Min energy
    min_energy = float(sorted_energies[0])

    # 5 smallest energies statistics
    five_smallest = sorted_energies[:5]
    five_mean = statistics.mean(five_smallest)
    five_median = statistics.median(five_smallest)
    five_stdev = statistics.stdev(five_smallest) if len(five_smallest) > 1 else 0.0
    five_variance = statistics.variance(five_smallest) if len(five_smallest) > 1 else 0.0

    # Diversity of top solutions
    solutions = list(sampleset.record.sample)
    solution_energy_pairs = list(zip(solutions, energies))
    solution_energy_pairs.sort(key=lambda x: x[1])
    top_solutions = [list(sol) for sol, _ in solution_energy_pairs[:10]]
    diversity = calculate_diversity(top_solutions)

    # QPU timing
    qpu_time = 0.0
    if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
        timing = sampleset.info['timing']
        qpu_time = timing.get('qpu_access_time', 0.0)

    # Chain break fraction (only meaningful for embedded problems)
    chain_break_fraction = 0.0
    if hasattr(sampleset, 'record') and 'chain_break_fraction' in sampleset.record.dtype.names:
        chain_break_fraction = float(np.mean(sampleset.record.chain_break_fraction))

    return {
        'solver_name': job.solver_name,
        'test_id': job.test_id,
        'ising_seed': job.ising_seed,
        'start_time': start_time_iso,
        'end_time': end_time,
        'annealing_time': job.annealing_time,
        'num_reads': job.num_reads,
        'chain_strength_multiplier': job.chain_strength_multiplier,
        'reduce_intersample_correlation': job.reduce_intersample_correlation if job.reduce_intersample_correlation is not None else '',
        'reinitialize_state': job.reinitialize_state if job.reinitialize_state is not None else '',
        'min_energy': min_energy,
        '5_smallest_energy_mean': five_mean,
        '5_smallest_energy_median': five_median,
        '5_smallest_energy_stdev': five_stdev,
        '5_smallest_energy_variance': five_variance,
        'diversity': diversity,
        'qpu_time': qpu_time,
        'chain_break_fraction': chain_break_fraction,
    }


def run_configuration(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    num_reads: int,
    annealing_time: float,
    csv_path: str,
    completed: set,
    solver_props: Dict[str, Any],
    num_ising_models: int = 1024,
    queue_depth: int = 30,
    solver_name: str = "",
    chain_strength_multiplier: float = 1.5,
    reduce_intersample_correlation: Optional[bool] = None,
    reinitialize_state: Optional[bool] = None,
) -> List[float]:
    """Run all Ising models for a single configuration using streaming.

    Maintains up to queue_depth jobs in-flight for maximum throughput.

    Returns list of min_energies for statistical analysis.
    """
    test_id = generate_test_id(
        num_reads, annealing_time,
        chain_strength_mult=chain_strength_multiplier,
        reduce_intersample=reduce_intersample_correlation,
        reinitialize=reinitialize_state,
    )
    min_energies = []

    # Validate configuration fits within QPU runtime limits
    is_valid, estimated_time = validate_configuration(annealing_time, num_reads, solver_props)
    if not is_valid:
        max_duration = solver_props['problem_run_duration_range'][1]
        print(f"\n  SKIPPING: num_reads={num_reads}, annealing_time={annealing_time:.1f}us")
        print(f"    Estimated QPU time: {estimated_time/1000:.1f}ms exceeds limit of {max_duration/1000:.1f}ms")
        return min_energies

    # Filter seeds to only those not yet completed
    seeds_to_run = [
        seed for seed in FIXED_SEEDS[:num_ising_models]
        if (test_id, seed) not in completed
    ]

    if not seeds_to_run:
        print(f"\n  Testing: num_reads={num_reads}, annealing_time={annealing_time:.1f}us")
        print(f"    All {num_ising_models} tests already completed (recovered from CSV)")
        return min_energies

    print(f"\n  Testing: num_reads={num_reads}, annealing_time={annealing_time:.1f}us (est. {estimated_time/1000:.1f}ms)")
    print(f"    Running {len(seeds_to_run)} tests with queue_depth={queue_depth}")

    # Track pending jobs: future -> PendingJob
    pending_jobs: Dict[Any, PendingJob] = {}
    seed_index = 0
    completed_count = 0
    total_to_run = len(seeds_to_run)

    # Initial queue fill
    while len(pending_jobs) < queue_depth and seed_index < total_to_run:
        seed = seeds_to_run[seed_index]
        try:
            job = submit_async_job(
                sampler, nodes, edges, seed, num_reads, annealing_time, test_id,
                solver_name=solver_name,
                chain_strength_multiplier=chain_strength_multiplier,
                reduce_intersample_correlation=reduce_intersample_correlation,
                reinitialize_state=reinitialize_state,
            )
            pending_jobs[job.future] = job
            seed_index += 1
        except Exception as e:
            if is_quota_exhausted_error(e):
                print(f"\n    ⚠️  QUOTA EXHAUSTED: {e}")
                raise QuotaExhaustedError(str(e))
            print(f"    Error submitting seed {seed}: {e}")
            seed_index += 1

    # Streaming result loop
    while pending_jobs:
        # Poll for completed futures
        completed_future = None

        while completed_future is None:
            for future in list(pending_jobs.keys()):
                if future.done():
                    completed_future = future
                    break
            if completed_future is None:
                time.sleep(0.05)  # 50ms polling interval

        # Process completed job
        job = pending_jobs.pop(completed_future)
        completed_count += 1

        try:
            result = process_completed_job(job)

            # Append to CSV immediately
            append_csv_row(csv_path, result)

            min_energies.append(result['min_energy'])

            # Print details for each test
            cbf = result.get('chain_break_fraction', 0.0)
            print(f"    [{completed_count:4d}/{total_to_run}] seed={job.ising_seed:4d} | "
                  f"min_E={result['min_energy']:8.1f} | "
                  f"5best_mean={result['5_smallest_energy_mean']:8.1f} | "
                  f"diversity={result['diversity']:.3f} | "
                  f"cbf={cbf:.3f} | "
                  f"qpu_time={result['qpu_time']/1000:.1f}ms | "
                  f"in_flight={len(pending_jobs)}")

        except Exception as e:
            if is_quota_exhausted_error(e):
                print(f"\n    ⚠️  QUOTA EXHAUSTED: {e}")
                # Cancel remaining pending jobs
                for future in pending_jobs.keys():
                    try:
                        future.cancel()
                    except Exception:
                        pass
                raise QuotaExhaustedError(str(e))
            print(f"    Error processing seed {job.ising_seed}: {e}")

        # Refill queue
        while len(pending_jobs) < queue_depth and seed_index < total_to_run:
            seed = seeds_to_run[seed_index]
            try:
                new_job = submit_async_job(
                    sampler, nodes, edges, seed, num_reads, annealing_time, test_id,
                    solver_name=solver_name,
                    chain_strength_multiplier=chain_strength_multiplier,
                    reduce_intersample_correlation=reduce_intersample_correlation,
                    reinitialize_state=reinitialize_state,
                )
                pending_jobs[new_job.future] = new_job
                seed_index += 1
            except Exception as e:
                if is_quota_exhausted_error(e):
                    print(f"\n    ⚠️  QUOTA EXHAUSTED: {e}")
                    # Cancel remaining pending jobs
                    for future in pending_jobs.keys():
                        try:
                            future.cancel()
                        except Exception:
                            pass
                    raise QuotaExhaustedError(str(e))
                print(f"    Error submitting seed {seed}: {e}")
                seed_index += 1

    if min_energies:
        avg = statistics.mean(min_energies)
        std = statistics.stdev(min_energies) if len(min_energies) > 1 else 0
        print(f"    Completed: {len(min_energies)} tests, avg min_energy: {avg:.1f} (std: {std:.1f})")

    return min_energies


def check_diminishing_returns(
    current_energies: List[float],
    previous_energies: List[float],
    threshold: float = 0.01
) -> bool:
    """Check if improvement is below threshold (1% by default).

    Returns True if we should stop (diminishing returns detected).
    """
    if not previous_energies or not current_energies:
        return False

    current_avg = statistics.mean(current_energies)
    previous_avg = statistics.mean(previous_energies)

    # Lower energy is better, so improvement = (previous - current) / |previous|
    if previous_avg == 0:
        return False

    improvement = (previous_avg - current_avg) / abs(previous_avg)

    print(f"    Improvement: {improvement*100:.2f}% (threshold: {threshold*100:.1f}%)")

    return improvement < threshold


def phase2_xaxis_sweep(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    num_ising_models: int = 1024,
    baseline_num_reads: int = 32,
    improvement_threshold: float = 0.01,
    queue_depth: int = 30
) -> Tuple[float, List[float]]:
    """Phase 2: Sweep annealing_time with fixed num_reads.

    Returns: (best_annealing_time, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 2: X-Axis Sweep (annealing_time)")
    print(f"  Fixed num_reads: {baseline_num_reads}")
    print("="*60)

    hw_min_time = solver_props['annealing_time_range'][0]
    # Calculate max annealing_time that fits within 250ms budget for this num_reads
    max_time = max_annealing_time_for_reads(baseline_num_reads, solver_props)
    step = max(hw_min_time, 10.0)  # Step size: max(min_solver_time, 10)

    # Start at round step value (snap up from hardware minimum)
    start_time = max(step, math.ceil(hw_min_time / step) * step)

    print(f"  Range: {start_time:.0f} - {max_time:.0f} us (budget-limited), step: {step:.0f} us")
    est_time = estimate_qpu_time(max_time, baseline_num_reads, solver_props)
    print(f"  Max config QPU time: {est_time/1000:.1f}ms")

    best_annealing_time = start_time
    best_energies: List[float] = []
    previous_energies: List[float] = []

    current_time = start_time
    while current_time <= max_time:
        energies = run_configuration(
            sampler, nodes, edges,
            baseline_num_reads, current_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth
        )

        if energies:
            # Track best
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_annealing_time = current_time
                best_energies = energies

            # Check for diminishing returns
            if check_diminishing_returns(energies, previous_energies, improvement_threshold):
                print(f"  Stopping: diminishing returns at annealing_time={current_time}")
                break

            previous_energies = energies

        current_time += step

    print(f"\n  Best annealing_time: {best_annealing_time} us")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_annealing_time, best_energies


def phase3_yaxis_sweep(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    best_annealing_time: float,
    num_ising_models: int = 1024,
    improvement_threshold: float = 0.01,
    queue_depth: int = 30
) -> Tuple[int, List[float]]:
    """Phase 3: Sweep num_reads with fixed annealing_time.

    Returns: (best_num_reads, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 3: Y-Axis Sweep (num_reads)")
    print(f"  Fixed annealing_time: {best_annealing_time} us")
    print("="*60)

    # Powers of 2 from 16 to 1024
    num_reads_values = [16, 32, 64, 128, 256, 512, 1024]

    # Cap at budget-limited max for this annealing_time
    max_reads = max_num_reads_for_annealing_time(best_annealing_time, solver_props)
    num_reads_values = [n for n in num_reads_values if n <= max_reads]

    print(f"  Values to test: {num_reads_values} (max: {max_reads} budget-limited)")
    if num_reads_values:
        est_time = estimate_qpu_time(best_annealing_time, num_reads_values[-1], solver_props)
        print(f"  Max config QPU time: {est_time/1000:.1f}ms")

    best_num_reads = num_reads_values[0] if num_reads_values else 16
    best_energies: List[float] = []
    previous_energies: List[float] = []

    for num_reads in num_reads_values:
        energies = run_configuration(
            sampler, nodes, edges,
            num_reads, best_annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth
        )

        if energies:
            # Track best
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_num_reads = num_reads
                best_energies = energies

            # Check for diminishing returns
            if check_diminishing_returns(energies, previous_energies, improvement_threshold):
                print(f"  Stopping: diminishing returns at num_reads={num_reads}")
                break

            previous_energies = energies

    print(f"\n  Best num_reads: {best_num_reads}")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_num_reads, best_energies


def phase4_diagonal_sweep(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    num_ising_models: int = 1024,
    improvement_threshold: float = 0.01,
    queue_depth: int = 30
) -> Tuple[float, int, List[float]]:
    """Phase 4: Sweep both annealing_time and num_reads together.

    Returns: (best_annealing_time, best_num_reads, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 4: Diagonal Sweep (both axes)")
    print("="*60)

    hw_min_time = solver_props['annealing_time_range'][0]
    step = max(hw_min_time, 10.0)
    start_time = max(step, math.ceil(hw_min_time / step) * step)

    # Powers of 2 from 16 to 1024
    num_reads_values = [16, 32, 64, 128, 256, 512, 1024]

    # Generate diagonal pairs: (time, reads) where both increase together
    # Only include pairs that fit within QPU budget
    diagonal_pairs = []
    current_time = start_time
    for num_reads in num_reads_values:
        is_valid, est_time = validate_configuration(current_time, num_reads, solver_props)
        if is_valid:
            diagonal_pairs.append((current_time, num_reads))
        current_time += step

    if not diagonal_pairs:
        print("  No valid diagonal pairs within budget!")
        return start_time, 16, []

    print(f"  Diagonal pairs (budget-filtered): {diagonal_pairs}")
    last_time, last_reads = diagonal_pairs[-1]
    est_time = estimate_qpu_time(last_time, last_reads, solver_props)
    print(f"  Max config QPU time: {est_time/1000:.1f}ms")

    best_time = diagonal_pairs[0][0]
    best_reads = diagonal_pairs[0][1]
    best_energies: List[float] = []
    previous_energies: List[float] = []

    for annealing_time, num_reads in diagonal_pairs:
        energies = run_configuration(
            sampler, nodes, edges,
            num_reads, annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth
        )

        if energies:
            # Track best
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_time = annealing_time
                best_reads = num_reads
                best_energies = energies

            # Check for diminishing returns
            if check_diminishing_returns(energies, previous_energies, improvement_threshold):
                print(f"  Stopping: diminishing returns at ({annealing_time}, {num_reads})")
                break

            previous_energies = energies

    print(f"\n  Best diagonal point: annealing_time={best_time}, num_reads={best_reads}")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_time, best_reads, best_energies


def phase5_max_validation(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    best_annealing_time: float,
    best_num_reads: int,
    num_ising_models: int = 1024,
    queue_depth: int = 30
):
    """Phase 5: Test at MAX budget-limited values as sanity check."""
    print("\n" + "="*60)
    print("PHASE 5: MAX Validation (sanity check)")
    print("="*60)

    # Calculate budget-limited max values
    max_time_for_best_reads = max_annealing_time_for_reads(best_num_reads, solver_props)
    max_reads_for_best_time = max_num_reads_for_annealing_time(best_annealing_time, solver_props)

    print(f"  Budget limits:")
    print(f"    Max annealing_time for num_reads={best_num_reads}: {max_time_for_best_reads:.0f} us")
    print(f"    Max num_reads for annealing_time={best_annealing_time:.0f}: {max_reads_for_best_time}")

    # Test 1: MAX annealing_time (budget-limited) with best num_reads
    print(f"\n  Test 1: MAX annealing_time ({max_time_for_best_reads:.0f}us) with best num_reads ({best_num_reads})")
    run_configuration(
        sampler, nodes, edges,
        best_num_reads, max_time_for_best_reads,
        csv_path, completed, solver_props, num_ising_models, queue_depth
    )

    # Test 2: MAX num_reads (budget-limited) with best annealing_time
    print(f"\n  Test 2: MAX num_reads ({max_reads_for_best_time}) with best annealing_time ({best_annealing_time:.0f}us)")
    run_configuration(
        sampler, nodes, edges,
        max_reads_for_best_time, best_annealing_time,
        csv_path, completed, solver_props, num_ising_models, queue_depth
    )

    # Test 3: Balanced max - find a combo that maxes both within budget
    # Use a moderate value for both that fits within budget
    balanced_reads = 256
    balanced_time = max_annealing_time_for_reads(balanced_reads, solver_props)
    print(f"\n  Test 3: Balanced max ({balanced_time:.0f}us, {balanced_reads} reads)")
    run_configuration(
        sampler, nodes, edges,
        balanced_reads, balanced_time,
        csv_path, completed, solver_props, num_ising_models, queue_depth
    )


def run_extra_annealing_times(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    extra_times: List[float],
    num_reads: int,
    num_ising_models: int = 1024,
    queue_depth: int = 30,
    solver_name: str = "",
):
    """Run specific annealing time values (spot checks)."""
    print("\n" + "="*60)
    print("EXTRA ANNEALING TIMES (spot checks)")
    print(f"  Fixed num_reads: {num_reads}")
    print(f"  Testing: {extra_times}")
    print("="*60)

    for annealing_time in extra_times:
        run_configuration(
            sampler, nodes, edges,
            num_reads, annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth,
            solver_name=solver_name,
        )
        # Reload completed after each config to ensure recovery works
        completed.update(load_completed_tests(csv_path))


def phase6_chain_strength_sweep(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    best_annealing_time: float,
    best_num_reads: int,
    num_ising_models: int = 1024,
    queue_depth: int = 30,
    solver_name: str = "",
    chain_strength_multipliers: Optional[List[float]] = None,
) -> Tuple[float, List[float]]:
    """Phase 6: Sweep chain_strength_multiplier with best annealing_time/num_reads.

    Auto-skips if sampler has no embedding (native hardware, no chains).

    Returns: (best_multiplier, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 6: Chain Strength Sweep")
    print(f"  Fixed annealing_time: {best_annealing_time} us")
    print(f"  Fixed num_reads: {best_num_reads}")
    print("="*60)

    if sampler.embedding is None:
        print("  SKIPPED: No embedding (native hardware) — chain strength is irrelevant")
        return 1.5, []

    if chain_strength_multipliers is None:
        chain_strength_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"  Multipliers to test: {chain_strength_multipliers}")

    best_mult = 1.5
    best_energies: List[float] = []

    for mult in chain_strength_multipliers:
        print(f"\n  --- chain_strength_multiplier = {mult} ---")
        energies = run_configuration(
            sampler, nodes, edges,
            best_num_reads, best_annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth,
            solver_name=solver_name,
            chain_strength_multiplier=mult,
        )

        if energies:
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_mult = mult
                best_energies = energies

    print(f"\n  Best chain_strength_multiplier: {best_mult}")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_mult, best_energies


def phase7_intersample_correlation(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    best_annealing_time: float,
    best_num_reads: int,
    num_ising_models: int = 1024,
    queue_depth: int = 30,
    solver_name: str = "",
) -> Tuple[Optional[bool], List[float]]:
    """Phase 7: Test reduce_intersample_correlation (False vs True).

    Returns: (best_setting, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 7: reduce_intersample_correlation")
    print(f"  Fixed annealing_time: {best_annealing_time} us")
    print(f"  Fixed num_reads: {best_num_reads}")
    print("="*60)

    # Check solver support
    supported_params = solver_props.get('parameters', {})
    if not isinstance(supported_params, dict):
        supported_params = {}
    # Also check raw properties for parameter support
    raw_params = sampler.properties.get('parameters', {})
    if 'reduce_intersample_correlation' not in raw_params:
        print("  SKIPPED: Solver does not support reduce_intersample_correlation")
        return None, []

    best_setting: Optional[bool] = None
    best_energies: List[float] = []

    for setting in [False, True]:
        print(f"\n  --- reduce_intersample_correlation = {setting} ---")
        energies = run_configuration(
            sampler, nodes, edges,
            best_num_reads, best_annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth,
            solver_name=solver_name,
            reduce_intersample_correlation=setting,
        )

        if energies:
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_setting = setting
                best_energies = energies

    print(f"\n  Best reduce_intersample_correlation: {best_setting}")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_setting, best_energies


def phase8_reinitialize_state(
    sampler: DWaveSamplerWrapper,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    solver_props: Dict[str, Any],
    csv_path: str,
    completed: set,
    best_annealing_time: float,
    best_num_reads: int,
    num_ising_models: int = 1024,
    queue_depth: int = 30,
    solver_name: str = "",
) -> Tuple[Optional[bool], List[float]]:
    """Phase 8: Test reinitialize_state (False vs True).

    Returns: (best_setting, energies at best point)
    """
    print("\n" + "="*60)
    print("PHASE 8: reinitialize_state")
    print(f"  Fixed annealing_time: {best_annealing_time} us")
    print(f"  Fixed num_reads: {best_num_reads}")
    print("="*60)

    # Check solver support
    raw_params = sampler.properties.get('parameters', {})
    if 'reinitialize_state' not in raw_params:
        print("  SKIPPED: Solver does not support reinitialize_state")
        return None, []

    best_setting: Optional[bool] = None
    best_energies: List[float] = []

    for setting in [False, True]:
        print(f"\n  --- reinitialize_state = {setting} ---")
        energies = run_configuration(
            sampler, nodes, edges,
            best_num_reads, best_annealing_time,
            csv_path, completed, solver_props, num_ising_models, queue_depth,
            solver_name=solver_name,
            reinitialize_state=setting,
        )

        if energies:
            if not best_energies or statistics.mean(energies) < statistics.mean(best_energies):
                best_setting = setting
                best_energies = energies

    print(f"\n  Best reinitialize_state: {best_setting}")
    if best_energies:
        print(f"  Best avg min_energy: {statistics.mean(best_energies):.1f}")

    return best_setting, best_energies


def estimate_total_configs(skip_phases: List[int]) -> int:
    """Estimate total number of configurations across all phases."""
    configs = 0
    if 2 not in skip_phases:
        configs += 8   # ~8 annealing_time steps
    if 3 not in skip_phases:
        configs += 7   # 7 num_reads values
    if 4 not in skip_phases:
        configs += 6   # ~6 diagonal pairs
    if 5 not in skip_phases:
        configs += 3   # 3 max validation tests
    if 6 not in skip_phases:
        configs += 6   # 6 chain_strength_multiplier values
    if 7 not in skip_phases:
        configs += 2   # 2 reduce_intersample_correlation values
    if 8 not in skip_phases:
        configs += 2   # 2 reinitialize_state values
    return configs


def run_solver_optimization(
    sampler: DWaveSamplerWrapper,
    csv_path: str,
    solver_display_name: str,
    num_ising_models: int = 1024,
    baseline_num_reads: int = 32,
    improvement_threshold: float = 0.01,
    skip_phases: Optional[List[int]] = None,
    queue_depth: int = 30,
    extra_annealing_times: Optional[List[float]] = None,
    solver_name: str = "",
    chain_strength_multipliers: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run the full parameter optimization workflow for a single solver.

    Returns dict with best parameters and stats.
    """
    nodes = sampler.nodes
    edges = sampler.edges

    # Phase 1: Get solver properties
    print("\n" + "="*60)
    print(f"PHASE 1: Parameter Range Discovery [{solver_display_name}]")
    print("="*60)

    solver_props = get_solver_properties(sampler)
    print(f"  Chip ID: {solver_props['chip_id']}")
    print(f"  Qubits: {solver_props['num_qubits']}")
    print(f"  Couplers: {solver_props['num_couplers']}")
    print(f"  Annealing time range: {solver_props['annealing_time_range']} us")
    print(f"  Num reads range: {solver_props['num_reads_range']}")
    print(f"  Default annealing time: {solver_props['default_annealing_time']} us")
    print(f"  Embedding: {'yes' if sampler.embedding else 'no (native)'}")

    # Show supported new parameters
    raw_params = sampler.properties.get('parameters', {})
    for param in ['reduce_intersample_correlation', 'reinitialize_state']:
        supported = param in raw_params
        print(f"  {param}: {'supported' if supported else 'not supported'}")

    # Load completed tests for recovery
    completed = load_completed_tests(csv_path)
    if completed:
        print(f"\n  Recovered {len(completed)} completed tests from {csv_path}")

    skip_phases = skip_phases or []

    # Phase 2: X-axis sweep
    best_annealing_time = solver_props['default_annealing_time']
    if 2 not in skip_phases:
        best_annealing_time, _ = phase2_xaxis_sweep(
            sampler, nodes, edges, solver_props, csv_path, completed,
            num_ising_models, baseline_num_reads, improvement_threshold, queue_depth
        )
        completed = load_completed_tests(csv_path)

    # Extra annealing times (spot checks) - runs after Phase 2
    if extra_annealing_times:
        run_extra_annealing_times(
            sampler, nodes, edges, solver_props, csv_path, completed,
            extra_annealing_times, baseline_num_reads, num_ising_models,
            queue_depth, solver_name=solver_name,
        )
        completed = load_completed_tests(csv_path)

    # Phase 3: Y-axis sweep
    best_num_reads = baseline_num_reads
    if 3 not in skip_phases:
        best_num_reads, _ = phase3_yaxis_sweep(
            sampler, nodes, edges, solver_props, csv_path, completed,
            best_annealing_time, num_ising_models, improvement_threshold, queue_depth
        )
        completed = load_completed_tests(csv_path)

    # Phase 4: Diagonal sweep
    if 4 not in skip_phases:
        diag_time, diag_reads, diag_energies = phase4_diagonal_sweep(
            sampler, nodes, edges, solver_props, csv_path, completed,
            num_ising_models, improvement_threshold, queue_depth
        )
        if diag_energies:
            print(f"  Diagonal best: time={diag_time}, reads={diag_reads}")
        completed = load_completed_tests(csv_path)

    # Phase 5: MAX validation
    if 5 not in skip_phases:
        phase5_max_validation(
            sampler, nodes, edges, solver_props, csv_path, completed,
            best_annealing_time, best_num_reads, num_ising_models, queue_depth
        )
        completed = load_completed_tests(csv_path)

    # Phase 6: Chain strength sweep
    best_chain_mult = 1.5
    if 6 not in skip_phases:
        best_chain_mult, _ = phase6_chain_strength_sweep(
            sampler, nodes, edges, solver_props, csv_path, completed,
            best_annealing_time, best_num_reads, num_ising_models, queue_depth,
            solver_name=solver_name,
            chain_strength_multipliers=chain_strength_multipliers,
        )
        completed = load_completed_tests(csv_path)

    # Phase 7: reduce_intersample_correlation
    best_ric = None
    if 7 not in skip_phases:
        best_ric, _ = phase7_intersample_correlation(
            sampler, nodes, edges, solver_props, csv_path, completed,
            best_annealing_time, best_num_reads, num_ising_models, queue_depth,
            solver_name=solver_name,
        )
        completed = load_completed_tests(csv_path)

    # Phase 8: reinitialize_state
    best_rs = None
    if 8 not in skip_phases:
        best_rs, _ = phase8_reinitialize_state(
            sampler, nodes, edges, solver_props, csv_path, completed,
            best_annealing_time, best_num_reads, num_ising_models, queue_depth,
            solver_name=solver_name,
        )

    # Summary for this solver
    print("\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE [{solver_display_name}]")
    print("="*60)
    print(f"  Best annealing_time: {best_annealing_time} us")
    print(f"  Best num_reads: {best_num_reads}")
    print(f"  Best chain_strength_multiplier: {best_chain_mult}")
    print(f"  Best reduce_intersample_correlation: {best_ric}")
    print(f"  Best reinitialize_state: {best_rs}")
    print(f"  Results saved to: {csv_path}")

    final_completed = load_completed_tests(csv_path)
    print(f"  Total tests completed: {len(final_completed)}")

    return {
        'solver_name': solver_display_name,
        'best_annealing_time': best_annealing_time,
        'best_num_reads': best_num_reads,
        'best_chain_strength_multiplier': best_chain_mult,
        'best_reduce_intersample_correlation': best_ric,
        'best_reinitialize_state': best_rs,
        'total_tests': len(final_completed),
        'output_file': csv_path,
        'quota_exhausted': False,
    }


def run_optimization(
    output_prefix: str,
    num_ising_models: int = 1024,
    baseline_num_reads: int = 32,
    improvement_threshold: float = 0.01,
    skip_phases: Optional[List[int]] = None,
    queue_depth: int = 30,
    extra_annealing_times: Optional[List[float]] = None,
    solvers: Optional[List[str]] = None,
    single_solver: bool = False,
    chain_strength_multipliers: Optional[List[float]] = None,
):
    """Run optimization across one or more solvers.

    Args:
        output_prefix: Output file prefix (solver name appended automatically).
        solvers: Explicit list of solver names. Overrides auto-discovery.
        single_solver: If True, only test the solver from DWAVE_API_SOLVER env var.
        chain_strength_multipliers: List of multipliers for phase 6.
    """
    print("=" * 60)
    print("QPU Parameter Optimization Tool (Multi-Solver)")
    print("=" * 60)

    if not QPU_AVAILABLE:
        print("ERROR: QPU not available (dwave-system not installed)")
        return None

    skip_phases = skip_phases or []

    # Determine solver list
    if single_solver:
        solver_list: List[Optional[Dict[str, Any]]] = [None]
        print("\nMode: Single solver (from DWAVE_API_SOLVER env var)")
    elif solvers:
        solver_list = [{'name': s, 'chip_id': s, 'region': None} for s in solvers]
        print(f"\nMode: Explicit solvers: {solvers}")
    else:
        print("\nDiscovering available QPU solvers across all regions...")
        discovered = discover_qpu_solvers()
        if not discovered:
            print("ERROR: No QPU solvers found")
            return None
        solver_list = discovered
        print(f"  Found {len(discovered)} QPU solver(s):")
        for s in discovered:
            print(f"    - {s['name']} ({s['num_qubits']} qubits, {s['topology_type']}, {s['region']})")

    # Print time estimate
    configs = estimate_total_configs(skip_phases)
    est_hours = (configs * num_ising_models * 0.25 * len(solver_list)) / 3600
    print(f"\nEstimated QPU time: {est_hours:.1f}h across {len(solver_list)} solver(s)")
    print(f"  {configs} configurations × {num_ising_models} models × ~250ms each")

    results = []

    for solver_info in solver_list:
        solver_name_arg = solver_info['name'] if solver_info else None
        solver_region = solver_info.get('region') if solver_info else None
        display_name = solver_name_arg or "default"

        # Build CSV path: prefix_solvername.csv
        safe_name = display_name.replace('.', '_').replace('-', '_')
        csv_path = f"{output_prefix}_{safe_name}.csv"

        region_str = f" ({solver_region})" if solver_region else ""
        print("\n" + "#"*60)
        print(f"# SOLVER: {display_name}{region_str}")
        print(f"# Output: {csv_path}")
        print("#"*60)

        # Initialize sampler for this solver
        print(f"\nInitializing QPU sampler for {display_name}...")
        try:
            sampler = DWaveSamplerWrapper(
                solver_name=solver_name_arg,
                region=solver_region,
            )
            print(f"  Connected: {len(sampler.nodes)} nodes, {len(sampler.edges)} edges")
        except Exception as e:
            print(f"ERROR: Failed to initialize sampler for {display_name}: {e}")
            continue

        try:
            result = run_solver_optimization(
                sampler=sampler,
                csv_path=csv_path,
                solver_display_name=display_name,
                num_ising_models=num_ising_models,
                baseline_num_reads=baseline_num_reads,
                improvement_threshold=improvement_threshold,
                skip_phases=skip_phases,
                queue_depth=queue_depth,
                extra_annealing_times=extra_annealing_times,
                solver_name=display_name,
                chain_strength_multipliers=chain_strength_multipliers,
            )
            results.append(result)
        except QuotaExhaustedError as e:
            print("\n" + "="*60)
            print(f"⚠️  QUOTA EXHAUSTED on {display_name}: {e}")
            print("="*60)
            print(f"  Results saved to: {csv_path}")
            print("  Run again later to resume from where you left off.")
            final_completed = load_completed_tests(csv_path)
            results.append({
                'solver_name': display_name,
                'total_tests': len(final_completed),
                'output_file': csv_path,
                'quota_exhausted': True,
            })
            # Continue to next solver (they may have separate quotas)
            continue
        finally:
            sampler.close()

    # Final summary
    print("\n" + "="*60)
    print("ALL SOLVERS COMPLETE")
    print("="*60)
    for r in results:
        status = "QUOTA EXHAUSTED" if r.get('quota_exhausted') else "OK"
        print(f"  {r['solver_name']}: {r['total_tests']} tests [{status}] → {r['output_file']}")

    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='QPU parameter optimization tool for finding optimal QPU parameters'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file prefix (solver name appended automatically). '
             'Default: qpu_param_optimization_{timestamp}'
    )

    parser.add_argument(
        '--num-models', '-n',
        type=int,
        default=1024,
        help='Number of Ising models to test per configuration (default: 1024)'
    )

    parser.add_argument(
        '--baseline-reads',
        type=int,
        default=32,
        help='Baseline num_reads for X-axis sweep (default: 32)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Improvement threshold for diminishing returns (default: 0.01 = 1%%)'
    )

    parser.add_argument(
        '--skip-phases',
        type=str,
        default=None,
        help='Comma-separated list of phases to skip (e.g., "4,5,6,7,8")'
    )

    parser.add_argument(
        '--queue-depth',
        type=int,
        default=30,
        help='Number of QPU jobs to keep in-flight for streaming (default: 30)'
    )

    parser.add_argument(
        '--extra-annealing-times',
        type=str,
        default=None,
        help='Comma-separated list of additional annealing times to test (e.g., "40,50,100,200")'
    )

    parser.add_argument(
        '--solvers',
        type=str,
        default=None,
        help='Comma-separated solver names. Default: auto-discover all QPU solvers.'
    )

    parser.add_argument(
        '--single-solver',
        action='store_true',
        help='Only test the solver from DWAVE_API_SOLVER env var.'
    )

    parser.add_argument(
        '--chain-strength-multipliers',
        type=str,
        default='0.5,1.0,1.5,2.0,3.0,5.0',
        help='Comma-separated chain strength multipliers for phase 6 '
             '(default: "0.5,1.0,1.5,2.0,3.0,5.0")'
    )

    args = parser.parse_args()

    # Generate output prefix if not specified
    output_prefix = args.output
    if not output_prefix:
        timestamp = int(time.time())
        output_prefix = f"qpu_param_optimization_{timestamp}"
    # Strip .csv extension if user provided it (we add solver suffix)
    if output_prefix.endswith('.csv'):
        output_prefix = output_prefix[:-4]

    # Parse skip phases
    skip_phases = None
    if args.skip_phases:
        skip_phases = [int(p.strip()) for p in args.skip_phases.split(',')]

    # Parse extra annealing times
    extra_annealing_times = None
    if args.extra_annealing_times:
        extra_annealing_times = [float(t.strip()) for t in args.extra_annealing_times.split(',')]

    # Parse solvers
    solvers = None
    if args.solvers:
        solvers = [s.strip() for s in args.solvers.split(',')]

    # Parse chain strength multipliers
    chain_strength_multipliers = [
        float(x.strip()) for x in args.chain_strength_multipliers.split(',')
    ]

    # Run optimization
    results = run_optimization(
        output_prefix=output_prefix,
        num_ising_models=args.num_models,
        baseline_num_reads=args.baseline_reads,
        improvement_threshold=args.threshold,
        skip_phases=skip_phases,
        queue_depth=args.queue_depth,
        extra_annealing_times=extra_annealing_times,
        solvers=solvers,
        single_solver=args.single_solver,
        chain_strength_multipliers=chain_strength_multipliers,
    )

    if results:
        print("\nOptimization completed successfully!")
        sys.exit(0)
    else:
        print("\nOptimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
