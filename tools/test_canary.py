#!/usr/bin/env python3
"""Test canary (fast) vs full mining runs to identify problematic model generations.

This tool generates Ising models using random nonces and runs them through:
1. Canary mode: Very low num_sweeps/num_reads (< 100ms) to quickly filter bad models
2. Full mode: Correct num_sweeps/num_reads for the given difficulty target

The intent is to create a filter for really bad model generations instead of wasting
time on problems where we know we won't hit the target energy.
"""
import argparse
import json
import logging
import multiprocessing
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import dimod

from shared.block import Block, BlockHeader, BlockRequirements, create_genesis_block
from shared.miner_types import MiningResult
from shared.time_utils import utc_timestamp
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
    evaluate_sampleset
)
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import load_topology


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to minutes.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 0.5 (minutes)
        "5m" -> 5.0
        "2h" -> 120.0
        "1d" -> 1440.0
        "1w" -> 10080.0
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith('s'):
        return int(duration_str[:-1]) / 60.0
    elif duration_str.endswith('m'):
        return float(duration_str[:-1])
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 60.0
    elif duration_str.endswith('d'):
        return int(duration_str[:-1]) * 1440.0
    elif duration_str.endswith('w'):
        return int(duration_str[:-1]) * 10080.0
    else:
        # Try parsing as raw minutes
        return float(duration_str)


def determine_canary_params(num_sweeps: int = 4, num_reads: int = 10) -> Dict[str, int]:
    """Determine canary parameters that run in < 100ms.

    Based on empirical testing with SA on typical hardware.

    Args:
        num_sweeps: Number of sweeps for canary (default: 4)
        num_reads: Number of reads for canary (default: 10)

    Returns:
        Dictionary with canary parameters
    """
    return {
        'num_sweeps': num_sweeps,
        'num_reads': num_reads,
    }


def process_batch(
    batch_h_list, batch_J_list, batch_nonces, batch_salts, batch_canary_results,
    miner, full_params, difficulty_energy, is_gpu, qpu_time_used, qpu_calls,
    full_passed, full_failed, full_energies, full_times, topology
):
    """Process a batch of problems through the full miner.

    Returns: (nonce_results, qpu_time_used, qpu_calls, full_passed, full_failed, full_energies, full_times)
    """
    nonce_results = []

    if len(batch_h_list) == 0:
        return nonce_results, qpu_time_used, qpu_calls, full_passed, full_failed, full_energies, full_times

    full_start = time.time()

    try:
        # For CUDA/GPU miners, use async_sampler with proper array conversion
        is_cuda = hasattr(miner, 'miner_type') and 'CUDA' in miner.miner_type

        if is_cuda and hasattr(miner, 'async_sampler'):
            # Convert dict representations to arrays for CUDA
            # Use topology from miner (which was initialized with the correct topology)
            nodes = miner.nodes
            edges = miner.edges
            N = max(max(nodes), max(max(i, j) for i, j in edges)) + 1

            h_arrays = []
            J_arrays = []

            for h_dict, J_dict in zip(batch_h_list, batch_J_list):
                # Convert h dict to array
                h = np.zeros(N, dtype=np.float32)
                for node, val in h_dict.items():
                    h[node] = val
                h_arrays.append(h)

                # Convert J dict to array
                J = np.zeros(len(edges), dtype=np.float32)
                edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
                for (i, j), val in J_dict.items():
                    if (i, j) in edge_to_idx:
                        J[edge_to_idx[(i, j)]] = val
                    elif (j, i) in edge_to_idx:
                        J[edge_to_idx[(j, i)]] = val
                J_arrays.append(J)

            # Debug: Log sampling params
            # Calculate batch number based on how many we've processed so far
            batch_num = (full_passed + full_failed) // len(h_arrays) + 1

            # Calculate num_betas from num_sweeps (like Metal does)
            num_sweeps_per_beta = 1  # Default (matches Metal)
            num_sweeps = full_params['num_sweeps']
            num_betas = num_sweeps // num_sweeps_per_beta

            print(f"")
            print(f"🔷 CUDA Batch #{batch_num}: Sampling {len(h_arrays)} problems")
            print(f"   Parameters: num_sweeps={num_sweeps}, num_reads={full_params['num_reads']}")
            print(f"   Internal: num_betas={num_betas}, num_sweeps_per_beta={num_sweeps_per_beta}")
            print(f"   Problem size: N={N}, nodes={len(nodes)}, edges={len(edges)}")
            print(f"   Starting GPU sampling at {time.strftime('%H:%M:%S')}...")

            # Use async_sampler for CUDA (with explicit timeout like the miner does)
            batch_start = time.time()
            full_samplesets = miner.async_sampler.sample_ising(
                h_list=h_arrays,
                J_list=J_arrays,
                num_reads=full_params['num_reads'],
                num_betas=num_betas,
                num_sweeps_per_beta=num_sweeps_per_beta,
                edges=edges,
                timeout=600.0  # 10 minute timeout for test workload
            )
            batch_elapsed = time.time() - batch_start
            print(f"   ✅ GPU sampling completed in {batch_elapsed:.2f}s ({batch_elapsed/len(h_arrays):.3f}s per problem)")

            # Filter samples for sparse topology (CUDA returns full N-sized samples)
            filtered_samplesets = []
            for sampleset in full_samplesets:
                filtered_samples = []
                for sample in sampleset.record.sample:
                    # Extract only values at node indices that exist in topology
                    filtered_sample = np.array([sample[node] for node in nodes], dtype=np.int8)
                    filtered_samples.append(filtered_sample)

                # Create new SampleSet with filtered samples
                filtered_sampleset = dimod.SampleSet.from_samples(
                    filtered_samples,
                    vartype='SPIN',
                    energy=sampleset.record.energy,
                    info=sampleset.info
                )
                filtered_samplesets.append(filtered_sampleset)

            full_samplesets = filtered_samplesets

            # Log detailed per-problem results
            batch_energies = []
            print(f"   📊 Individual problem results:")
            for prob_idx, (sampleset, canary_result) in enumerate(zip(full_samplesets, batch_canary_results)):
                prob_energy = float(np.min(sampleset.record.energy))
                batch_energies.append(prob_energy)
                canary_energy = canary_result.get('energy', 'N/A')
                passed_mark = "✅" if prob_energy < difficulty_energy else "❌"
                print(f"      Problem {prob_idx+1:2d}: energy={prob_energy:8.1f} {passed_mark}  (canary: {canary_energy:8.1f})")

            # Summary statistics
            min_batch_energy = min(batch_energies)
            max_batch_energy = max(batch_energies)
            avg_batch_energy = sum(batch_energies) / len(batch_energies)
            num_passed = sum(1 for e in batch_energies if e < difficulty_energy)
            print(f"   📈 Batch summary: min={min_batch_energy:.1f}, max={max_batch_energy:.1f}, avg={avg_batch_energy:.1f}")
            print(f"   Passed difficulty threshold ({difficulty_energy:.1f}): {num_passed}/{len(batch_energies)}")

        else:
            # For CPU/QPU/Metal, use regular sampler
            sampling_params = {
                'h': batch_h_list if is_gpu else batch_h_list[0],
                'J': batch_J_list if is_gpu else batch_J_list[0],
                'num_reads': full_params['num_reads']
            }

            # Add type-specific parameters
            if hasattr(miner, 'miner_type') and miner.miner_type == 'QPU':
                if 'annealing_time' in full_params:
                    sampling_params['annealing_time'] = full_params['annealing_time']
            elif hasattr(miner, 'miner_type') and any(t in miner.miner_type for t in ['CPU', 'Metal', 'GPU']):
                if 'num_sweeps' in full_params:
                    sampling_params['num_sweeps'] = full_params['num_sweeps']

            # Call sampler
            full_samplesets = miner.sampler.sample_ising(**sampling_params)

        # GPU returns list, others return single sampleset
        if not is_gpu:
            full_samplesets = [full_samplesets]

        full_time = time.time() - full_start

        # Process each result in batch
        for idx, (nonce, salt, canary_result, sampleset) in enumerate(
            zip(batch_nonces, batch_salts, batch_canary_results, full_samplesets)
        ):
            full_energy = float(np.min(sampleset.record.energy))
            full_energies.append(full_energy)
            full_times.append(full_time / len(batch_h_list))  # Amortize time across batch

            # Track QPU time if available
            qpu_access_time = 0.0
            if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                timing = sampleset.info['timing']
                if 'qpu_access_time' in timing:
                    qpu_access_time = timing['qpu_access_time'] / 1_000_000.0
                    qpu_time_used += qpu_access_time
                    qpu_calls += 1

            passed = full_energy < difficulty_energy
            if passed:
                full_passed += 1
            else:
                full_failed += 1

            nonce_result = {
                'nonce': nonce,
                'salt': salt,
                'canary': canary_result,
                'full': {
                    'energy': full_energy,
                    'time': full_time / len(batch_h_list),
                    'qpu_time': qpu_access_time if qpu_access_time > 0 else None,
                    'passed': passed
                }
            }
            nonce_results.append(nonce_result)

    except Exception as e:
        # If batch fails, mark all as errors
        for nonce, salt, canary_result in zip(batch_nonces, batch_salts, batch_canary_results):
            nonce_result = {
                'nonce': nonce,
                'salt': salt,
                'canary': canary_result,
                'full': {'error': str(e)}
            }
            nonce_results.append(nonce_result)
            full_failed += 1

    return nonce_results, qpu_time_used, qpu_calls, full_passed, full_failed, full_energies, full_times


def run_canary_test(
    miner,
    canary_miner,
    difficulty_energy: float,
    duration_minutes: float,
    min_diversity: float,
    min_solutions: int,
    seed: int,
    topology,
    canary_num_sweeps: int = 4,
    canary_num_reads: int = 10,
    log_file=None
) -> Dict:
    """Run canary tests to filter bad model generations.

    Args:
        miner: Full miner instance (CPU, CUDA, Metal, QPU) - used for full tests
        canary_miner: Canary miner instance (always CPU SA) - used for fast filtering
        difficulty_energy: Fixed difficulty energy threshold
        duration_minutes: How long to run tests (in minutes)
        min_diversity: Minimum solution diversity requirement
        min_solutions: Minimum number of solutions required
        seed: Random seed for reproducible nonce generation
        topology: Topology to use for model generation
        canary_num_sweeps: Number of sweeps for canary (default: 4)
        canary_num_reads: Number of reads for canary (default: 10)
        log_file: Optional file object to write logs to

    Returns:
        Dictionary with canary test statistics
    """
    def log(msg):
        """Print to console and optionally to log file."""
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"\n⛏️  Starting canary testing:")
    log(f"   Duration: {duration_minutes} minutes")
    log(f"   Difficulty: {difficulty_energy:.1f}")
    log(f"   Min diversity: {min_diversity}")
    log(f"   Min solutions: {min_solutions}")
    log(f"   Seed: {seed}")

    # Setup
    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=0  # Disable decay
    )

    # Determine canary parameters
    canary_params = determine_canary_params(
        num_sweeps=canary_num_sweeps,
        num_reads=canary_num_reads
    )
    log(f"\n🐤 Canary params: {canary_params}")

    # For full mining, we need to determine appropriate parameters
    nodes = list(topology.nodes)
    edges = list(topology.edges)

    full_params = miner.adapt_parameters(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        num_nodes=len(nodes),
        num_edges=len(edges),
    )

    # Cap num_reads at 256 for CUDA (hardware limit)
    is_cuda = hasattr(miner, 'miner_type') and 'CUDA' in miner.miner_type
    if is_cuda and full_params['num_reads'] > 256:
        full_params['num_reads'] = 256

    log(f"🎯 Full params: {full_params}")

    # Initialize random seed for reproducible nonces
    random.seed(seed)
    np.random.seed(seed)

    # Create genesis block as prev_block
    prev_block = create_genesis_block()
    prev_block.next_block_requirements = requirements

    # Stats tracking
    total_nonces = 0
    canary_passed = 0
    canary_failed = 0
    full_passed = 0
    full_failed = 0

    canary_energies = []
    full_energies = []

    canary_times = []
    full_times = []

    # QPU time tracking
    qpu_time_used = 0.0  # Actual QPU time in seconds
    qpu_calls = 0

    # Detailed results for each nonce
    nonce_results = []

    start_time = time.time()
    duration_seconds = duration_minutes * 60
    last_progress_time = start_time
    progress_interval = 10  # Print progress every 10 seconds

    log(f"\n⏱️  Testing started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    log(f"   Will run until {time.strftime('%H:%M:%S', time.localtime(start_time + duration_seconds))}")

    # Determine if we should use batching (for GPU miners)
    is_gpu = hasattr(miner, 'miner_type') and any(t in miner.miner_type for t in ['Metal', 'CUDA', 'GPU'])

    # For CUDA, use SM count for batch size (typically 48 for most GPUs)
    # For other miners, process one at a time
    if is_gpu and hasattr(miner, 'async_sampler') and hasattr(miner.async_sampler, 'get_num_sms'):
        batch_size = miner.async_sampler.get_num_sms()  # Full SM utilization
    elif is_gpu:
        batch_size = 48  # Default for most modern GPUs
    else:
        batch_size = 1  # CPU/QPU: one at a time

    if is_gpu:
        log(f"   🚀 GPU batching enabled: {batch_size} problems per batch")

    # Accumulate batch
    batch_nonces = []
    batch_salts = []
    batch_h_list = []
    batch_J_list = []
    batch_canary_results = []

    while time.time() - start_time < duration_seconds:
        total_nonces += 1

        # Generate random salt for nonce generation (reproducible via seed)
        salt = random.randbytes(32)

        # Generate nonce deterministically
        nonce = ising_nonce_from_block(
            prev_block.hash,
            f"canary-test-{seed}",
            1,  # block index
            salt
        )

        # Generate Ising model
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

        nonce_result = {
            'nonce': nonce,
            'salt': salt.hex(),
            'canary': {},
            'full': {}
        }

        # === CANARY TEST ===
        # Use canary_miner (CPU SA) for fast filtering
        canary_start = time.time()
        canary_result = {}
        try:
            canary_sampleset = canary_miner.sampler.sample_ising(
                h, J,
                num_reads=canary_params['num_reads'],
                num_sweeps=canary_params['num_sweeps']
            )
            canary_time = time.time() - canary_start
            canary_times.append(canary_time)

            # Get best energy from canary
            canary_energy = float(np.min(canary_sampleset.record.energy))
            canary_energies.append(canary_energy)

            canary_result = {
                'energy': canary_energy,
                'time': canary_time,
                'passed': canary_energy < difficulty_energy
            }

            if canary_energy < difficulty_energy:
                canary_passed += 1
            else:
                canary_failed += 1

        except Exception as e:
            log(f"   ❌ Canary error for nonce {nonce}: {e}")
            canary_failed += 1
            canary_result = {'error': str(e)}

        # === ACCUMULATE FOR BATCH ===
        # Add to batch regardless of canary result (for comparison analysis)
        batch_nonces.append(nonce)
        batch_salts.append(salt.hex())
        batch_h_list.append(h)
        batch_J_list.append(J)
        batch_canary_results.append(canary_result)

        # Process batch when it reaches batch_size or time is running out
        should_process_batch = (
            len(batch_h_list) >= batch_size or
            time.time() - start_time >= duration_seconds
        )

        if should_process_batch:
            # Process accumulated batch through full miner
            batch_results, qpu_time_used, qpu_calls, full_passed, full_failed, full_energies, full_times = process_batch(
                batch_h_list, batch_J_list, batch_nonces, batch_salts, batch_canary_results,
                miner, full_params, difficulty_energy, is_gpu, qpu_time_used, qpu_calls,
                full_passed, full_failed, full_energies, full_times, topology
            )

            nonce_results.extend(batch_results)

            # Clear batch
            batch_nonces = []
            batch_salts = []
            batch_h_list = []
            batch_J_list = []
            batch_canary_results = []

        # Progress update
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed = current_time - start_time
            elapsed_min = elapsed / 60

            # Rate based on completed full tests (both canary AND full completed)
            completed_nonces = full_passed + full_failed
            completed_per_min = completed_nonces / elapsed_min if elapsed_min > 0 else 0

            qpu_time_str = ""
            if qpu_calls > 0:
                qpu_time_str = f", QPU: {qpu_time_used:.1f}s/{qpu_calls} calls"

            # Show best energy achieved and gap to target
            best_energy_str = ""
            if full_energies:
                best_energy = min(full_energies)
                gap = best_energy - difficulty_energy
                best_energy_str = f", Best: {best_energy:.1f} (gap: {gap:+.1f})"
                if full_passed == 0 and completed_nonces >= 10:
                    best_energy_str += " ⚠️"

            log(f"   [{elapsed_min:.1f}/{duration_minutes:.0f} min] "
                f"Generated: {total_nonces}, Completed: {completed_nonces}, "
                f"Canary: {canary_passed}✅/{canary_failed}❌, "
                f"Full: {full_passed}✅/{full_failed}❌{best_energy_str}, "
                f"Rate: {completed_per_min:.2f} completed/min{qpu_time_str}")
            last_progress_time = current_time

    total_time = time.time() - start_time

    # Compute statistics
    stats = {
        'total_nonces': total_nonces,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'nonces_per_minute': total_nonces / (total_time / 60) if total_time > 0 else 0,
        'qpu_time_used_seconds': qpu_time_used if qpu_calls > 0 else None,
        'qpu_time_used_minutes': qpu_time_used / 60 if qpu_calls > 0 else None,
        'qpu_calls': qpu_calls if qpu_calls > 0 else None,
        'qpu_time_per_call': qpu_time_used / qpu_calls if qpu_calls > 0 else None,
        'qpu_efficiency': (qpu_time_used / total_time * 100) if qpu_calls > 0 and total_time > 0 else None,

        'canary': {
            'passed': canary_passed,
            'failed': canary_failed,
            'pass_rate': canary_passed / total_nonces if total_nonces > 0 else 0,
            'energy_stats': {
                'min': min(canary_energies) if canary_energies else None,
                'max': max(canary_energies) if canary_energies else None,
                'mean': sum(canary_energies) / len(canary_energies) if canary_energies else None,
                'all_energies': canary_energies
            },
            'time_stats': {
                'min': min(canary_times) if canary_times else None,
                'max': max(canary_times) if canary_times else None,
                'mean': sum(canary_times) / len(canary_times) if canary_times else None,
                'all_times': canary_times
            },
            'params': canary_params
        },

        'full': {
            'passed': full_passed,
            'failed': full_failed,
            'pass_rate': full_passed / total_nonces if total_nonces > 0 else 0,
            'energy_stats': {
                'min': min(full_energies) if full_energies else None,
                'max': max(full_energies) if full_energies else None,
                'mean': sum(full_energies) / len(full_energies) if full_energies else None,
                'all_energies': full_energies
            },
            'time_stats': {
                'min': min(full_times) if full_times else None,
                'max': max(full_times) if full_times else None,
                'mean': sum(full_times) / len(full_times) if full_times else None,
                'all_times': full_times
            },
            'params': full_params
        },

        'nonce_results': nonce_results,

        # Analysis of canary effectiveness
        'analysis': {
            'true_positives': len([r for r in nonce_results
                                  if r['canary'].get('passed') and r['full'].get('passed')]),
            'false_positives': len([r for r in nonce_results
                                   if r['canary'].get('passed') and not r['full'].get('passed')]),
            'true_negatives': len([r for r in nonce_results
                                  if not r['canary'].get('passed') and not r['full'].get('passed')]),
            'false_negatives': len([r for r in nonce_results
                                   if not r['canary'].get('passed') and r['full'].get('passed')]),
        }
    }

    # Calculate precision, recall, F1
    tp = stats['analysis']['true_positives']
    fp = stats['analysis']['false_positives']
    tn = stats['analysis']['true_negatives']
    fn = stats['analysis']['false_negatives']

    if tp + fp > 0:
        stats['analysis']['precision'] = tp / (tp + fp)
    else:
        stats['analysis']['precision'] = 0.0

    if tp + fn > 0:
        stats['analysis']['recall'] = tp / (tp + fn)
    else:
        stats['analysis']['recall'] = 0.0

    if tp > 0:
        p = stats['analysis']['precision']
        r = stats['analysis']['recall']
        stats['analysis']['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        stats['analysis']['f1'] = 0.0

    # === PREDICTIVE POWER ANALYSIS ===
    # Analyze how well canary energy predicts mining success
    if len(canary_energies) > 1 and len(full_energies) > 1:
        # Group nonces by canary energy into buckets
        canary_arr = np.array(canary_energies)
        full_arr = np.array(full_energies)

        # Create canary energy buckets (e.g., every 50 units)
        bucket_size = 50
        min_canary = np.floor(np.min(canary_arr) / bucket_size) * bucket_size
        max_canary = np.ceil(np.max(canary_arr) / bucket_size) * bucket_size

        buckets = []
        if max_canary > min_canary:
            for bucket_start in np.arange(min_canary, max_canary, bucket_size):
                bucket_end = bucket_start + bucket_size
                mask = (canary_arr >= bucket_start) & (canary_arr < bucket_end)

                if np.sum(mask) > 0:
                    bucket_canaries = canary_arr[mask]
                    bucket_fulls = full_arr[mask]
                    bucket_passed = np.sum(bucket_fulls < difficulty_energy)
                    bucket_total = len(bucket_fulls)

                    buckets.append({
                        'range': [float(bucket_start), float(bucket_end)],
                        'count': int(bucket_total),
                        'success_rate': float(bucket_passed / bucket_total),
                        'avg_canary_energy': float(np.mean(bucket_canaries)),
                        'avg_full_energy': float(np.mean(bucket_fulls)),
                    })

        stats['analysis']['predictive_buckets'] = buckets

        # Find optimal canary threshold (maximize success rate while filtering)
        if len(nonce_results) > 0:
            # Sort by canary energy (most negative first = best)
            sorted_results = sorted(
                nonce_results,
                key=lambda r: r['canary'].get('energy', float('inf'))
            )

            best_threshold = None
            best_metric = -float('inf')

            # For each potential threshold, calculate success rate and time savings
            for i in range(1, len(sorted_results)):
                threshold_energy = sorted_results[i]['canary'].get('energy', float('inf'))

                # Count how many would pass this threshold
                passed_threshold = i

                # Of those that passed, how many succeeded in full test?
                successes = sum(1 for r in sorted_results[:i]
                              if r['full'].get('passed', False))

                if passed_threshold > 0:
                    success_rate = successes / passed_threshold

                    # Metric: success rate weighted by selectivity
                    # We want high success rate but also want to filter some
                    selectivity = 1.0 - (passed_threshold / len(sorted_results))
                    metric = success_rate * (1 + selectivity)  # Reward both

                    if metric > best_metric:
                        best_metric = metric
                        best_threshold = {
                            'canary_energy': float(threshold_energy),
                            'nonces_passed': passed_threshold,
                            'nonces_total': len(sorted_results),
                            'selectivity': float(selectivity),
                            'success_rate': float(success_rate),
                            'expected_successes': successes,
                            'metric': float(metric)
                        }

            stats['analysis']['optimal_threshold'] = best_threshold

    # === CORRELATION ANALYSIS ===
    # Analyze relationship between canary and full energies
    if len(canary_energies) > 1 and len(full_energies) > 1:
        canary_arr = np.array(canary_energies)
        full_arr = np.array(full_energies)

        # Pearson correlation coefficient
        correlation = np.corrcoef(canary_arr, full_arr)[0, 1]

        # Linear regression: full = slope * canary + intercept
        # Use numpy's polyfit for simple linear regression
        slope, intercept = np.polyfit(canary_arr, full_arr, 1)

        # Calculate R² (coefficient of determination)
        predicted_full = slope * canary_arr + intercept
        ss_res = np.sum((full_arr - predicted_full) ** 2)  # Residual sum of squares
        ss_tot = np.sum((full_arr - np.mean(full_arr)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Energy difference analysis (full - canary)
        energy_diffs = full_arr - canary_arr
        mean_diff = float(np.mean(energy_diffs))
        std_diff = float(np.std(energy_diffs))
        min_diff = float(np.min(energy_diffs))
        max_diff = float(np.max(energy_diffs))

        # Check if relationship is roughly constant offset vs non-linear
        # If std of differences is small relative to mean, it's roughly constant
        diff_cv = abs(std_diff / mean_diff) if mean_diff != 0 else float('inf')

        stats['analysis']['correlation'] = {
            'pearson_r': float(correlation),
            'linear_regression': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                'equation': f"full = {slope:.3f} * canary + {intercept:.1f}"
            },
            'energy_difference': {
                'mean': mean_diff,
                'std': std_diff,
                'min': min_diff,
                'max': max_diff,
                'coefficient_of_variation': diff_cv,
                'interpretation': (
                    'roughly constant offset' if diff_cv < 0.1 else
                    'moderately variable offset' if diff_cv < 0.3 else
                    'highly variable (non-linear)'
                )
            },
            'interpretation': {
                'correlation_strength': (
                    'very strong positive' if correlation > 0.9 else
                    'strong positive' if correlation > 0.7 else
                    'moderate positive' if correlation > 0.5 else
                    'weak positive' if correlation > 0.3 else
                    'weak or no correlation'
                ),
                'relationship_type': (
                    'nearly perfect linear (constant offset)' if r_squared > 0.95 and diff_cv < 0.1 else
                    'strong linear relationship' if r_squared > 0.8 else
                    'moderate linear relationship' if r_squared > 0.5 else
                    'weak linear relationship (non-linear)'
                )
            }
        }
    else:
        stats['analysis']['correlation'] = None

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test canary (fast) vs full mining to identify bad model generations'
    )
    parser.add_argument(
        '--miner-type',
        type=str,
        choices=['cpu', 'cuda', 'metal', 'qpu'],
        default='cpu',
        help='Miner type to test (default: cpu)'
    )
    parser.add_argument(
        '--difficulty-energy',
        type=float,
        required=True,
        help='Fixed difficulty energy threshold (e.g., -3532.0)'
    )
    parser.add_argument(
        '--duration',
        type=str,
        default='1m',
        help='Test duration (default: 1m). Examples: 30s, 5m, 2h, 1d, 1w'
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.15,
        help='Minimum solution diversity (default: 0.15)'
    )
    parser.add_argument(
        '--min-solutions',
        type=int,
        default=5,
        help='Minimum number of solutions (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible nonce generation (default: 42)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device ID (for cuda miner, default: 0)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Topology name (default: DEFAULT_TOPOLOGY=Z(9,2)). Examples: "Z(9,2)", "Z(10,2)", "Advantage2_system1.13"'
    )
    parser.add_argument(
        '--canary-num-sweeps',
        type=int,
        default=4,
        help='Number of sweeps for canary (default: 4, very fast)'
    )
    parser.add_argument(
        '--canary-num-reads',
        type=int,
        default=10,
        help='Number of reads for canary (default: 10, very fast)'
    )

    args = parser.parse_args()

    # Parse duration
    try:
        duration_minutes = parse_duration(args.duration)
    except (ValueError, IndexError):
        print(f"❌ Invalid duration format: '{args.duration}'. Use formats like: 30s, 5m, 2h, 1d, 1w")
        return 1

    # Parse topology if specified
    if args.topology:
        try:
            topology = load_topology(args.topology)
            print(f"✅ Loaded topology: {topology.solver_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Failed to load topology '{args.topology}': {e}")
            return 1
    else:
        topology = DEFAULT_TOPOLOGY

    print("🔬 Canary Test Tool")
    print("=" * 50)
    print(f"Miner type: {args.miner_type.upper()}")
    print(f"Topology: {topology.solver_name} ({len(topology.nodes)} nodes, {len(topology.edges)} edges)")
    print(f"Difficulty: {args.difficulty_energy:.1f}")
    print(f"Duration: {args.duration} ({duration_minutes:.1f} minutes)")
    print(f"Seed: {args.seed}")

    # Initialize full miner (what we're testing)
    miner = None
    if args.miner_type == 'cpu':
        from CPU.sa_miner import SimulatedAnnealingMiner
        miner = SimulatedAnnealingMiner(miner_id="canary-test-cpu", topology=topology)
    elif args.miner_type == 'cuda':
        from GPU.cuda_miner import CudaMiner
        miner = CudaMiner(miner_id="canary-test-cuda", device=args.device, topology=topology)
    elif args.miner_type == 'metal':
        from GPU.metal_miner import MetalMiner
        miner = MetalMiner(miner_id="canary-test-metal", topology=topology)
    elif args.miner_type == 'qpu':
        from QPU.dwave_miner import DWaveMiner
        miner = DWaveMiner(miner_id="canary-test-qpu", topology=topology, qpu_timeout=0.0)

    if not miner:
        print(f"❌ Failed to initialize {args.miner_type} miner")
        return 1

    print(f"✅ {args.miner_type.upper()} miner initialized")

    # Initialize canary miner (always CPU SA for fast filtering)
    # IMPORTANT: Use same topology as full miner
    from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
    from CPU.sa_miner import SimulatedAnnealingMiner
    canary_sampler = SimulatedAnnealingStructuredSampler(topology=topology)
    canary_miner = SimulatedAnnealingMiner(miner_id="canary-cpu-sa", sampler=canary_sampler)
    print(f"✅ Canary (CPU SA) initialized")

    # Open log file
    log_file = None
    if args.output:
        log_filename = args.output.replace('.json', '.log')
        log_file = open(log_filename, 'w', encoding='utf-8')
        print(f"📝 Logging to {log_filename}")

    # Run canary tests
    start_time = time.time()
    try:
        stats = run_canary_test(
            miner=miner,
            canary_miner=canary_miner,
            difficulty_energy=args.difficulty_energy,
            duration_minutes=duration_minutes,
            min_diversity=args.min_diversity,
            min_solutions=args.min_solutions,
            seed=args.seed,
            topology=topology,
            canary_num_sweeps=args.canary_num_sweeps,
            canary_num_reads=args.canary_num_reads,
            log_file=log_file
        )
    finally:
        if log_file:
            log_file.close()
    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"✅ Canary testing completed:")
    print(f"   Total time: {stats['total_time_minutes']:.1f} min")
    print(f"   Nonces tested: {stats['total_nonces']}")
    print(f"   Rate: {stats['nonces_per_minute']:.2f} nonces/min")

    print(f"\n🐤 Canary results:")
    print(f"   Passed: {stats['canary']['passed']} ({stats['canary']['pass_rate']*100:.1f}%)")
    print(f"   Failed: {stats['canary']['failed']}")
    print(f"   Avg time: {stats['canary']['time_stats']['mean']*1000:.1f}ms")
    print(f"   Energy range: [{stats['canary']['energy_stats']['min']:.1f}, {stats['canary']['energy_stats']['max']:.1f}]")
    print(f"   Mean energy: {stats['canary']['energy_stats']['mean']:.1f}")

    print(f"\n🎯 Full results:")
    print(f"   Passed: {stats['full']['passed']} ({stats['full']['pass_rate']*100:.1f}%)")
    print(f"   Failed: {stats['full']['failed']}")
    print(f"   Avg time: {stats['full']['time_stats']['mean']:.3f}s")
    print(f"   Energy range: [{stats['full']['energy_stats']['min']:.1f}, {stats['full']['energy_stats']['max']:.1f}]")
    print(f"   Mean energy: {stats['full']['energy_stats']['mean']:.1f}")
    print(f"   Difficulty target: {args.difficulty_energy:.1f}")

    # Prominently warn if no solutions hit target
    if stats['full']['passed'] == 0 and stats['total_nonces'] > 0:
        gap = stats['full']['energy_stats']['min'] - args.difficulty_energy
        print(f"\n   ⚠️  WARNING: No solutions hit target energy!")
        print(f"   📊 Best energy achieved: {stats['full']['energy_stats']['min']:.1f}")
        print(f"   📉 Gap to target: {gap:.1f} energy units ({gap/args.difficulty_energy*100:.1f}% relative)")
        print(f"   🔍 Problems evaluated: {stats['total_nonces']}")
        if gap > 100:
            print(f"   💡 Consider: Target may be too aggressive, or num_sweeps too low")

    # Print QPU time usage if available
    if stats.get('qpu_calls') and stats['qpu_calls'] > 0:
        print(f"\n⚡ QPU Time Usage:")
        print(f"   Total QPU time: {stats['qpu_time_used_minutes']:.2f} min ({stats['qpu_time_used_seconds']:.1f}s)")
        print(f"   QPU calls: {stats['qpu_calls']}")
        print(f"   Avg time per call: {stats['qpu_time_per_call']:.3f}s")
        print(f"   QPU efficiency: {stats['qpu_efficiency']:.1f}% (QPU time / wall clock time)")
        print(f"   Wall clock time: {stats['total_time_minutes']:.2f} min")
        overhead_pct = 100 - stats['qpu_efficiency']
        print(f"   Overhead: {overhead_pct:.1f}% (network, queue, processing)")

    # Print predictive power analysis
    if stats['analysis'].get('predictive_buckets'):
        print(f"\n🎯 Predictive Power (Success Rate by Canary Energy):")
        buckets = stats['analysis']['predictive_buckets']
        print(f"   {'Canary Range':<20} {'Count':>6} {'Success':>8} {'Avg Canary':>12} {'Avg Full':>10}")
        print(f"   {'-'*66}")
        for bucket in buckets:
            range_str = f"[{bucket['range'][0]:.0f}, {bucket['range'][1]:.0f})"
            print(f"   {range_str:<20} {bucket['count']:>6} {bucket['success_rate']*100:>7.1f}% "
                  f"{bucket['avg_canary_energy']:>12.1f} {bucket['avg_full_energy']:>10.1f}")

    if stats['analysis'].get('optimal_threshold'):
        thresh = stats['analysis']['optimal_threshold']
        print(f"\n💡 Optimal Canary Threshold:")
        print(f"   Cutoff energy: {thresh['canary_energy']:.1f}")
        print(f"   Nonces accepted: {thresh['nonces_passed']}/{thresh['nonces_total']} ({(1-thresh['selectivity'])*100:.1f}%)")
        print(f"   Success rate: {thresh['success_rate']*100:.1f}%")
        print(f"   Expected successes: {thresh['expected_successes']}")
        print(f"   Selectivity: {thresh['selectivity']*100:.1f}% filtered")
        print(f"   Combined metric: {thresh['metric']:.3f}")

    print(f"\n📈 Classification Metrics (for reference):")
    print(f"   True Positives: {stats['analysis']['true_positives']}")
    print(f"   False Positives: {stats['analysis']['false_positives']}")
    print(f"   True Negatives: {stats['analysis']['true_negatives']}")
    print(f"   False Negatives: {stats['analysis']['false_negatives']}")
    print(f"   Precision: {stats['analysis']['precision']:.3f}")
    print(f"   Recall: {stats['analysis']['recall']:.3f}")
    print(f"   F1 Score: {stats['analysis']['f1']:.3f}")

    # Print correlation analysis
    if stats['analysis'].get('correlation'):
        corr = stats['analysis']['correlation']
        print(f"\n🔬 Correlation Analysis:")
        print(f"   Pearson correlation: {corr['pearson_r']:.4f} ({corr['interpretation']['correlation_strength']})")
        print(f"   R² (goodness of fit): {corr['linear_regression']['r_squared']:.4f}")
        print(f"   Linear equation: {corr['linear_regression']['equation']}")
        print(f"   Relationship: {corr['interpretation']['relationship_type']}")

        print(f"\n   Energy difference (full - canary):")
        print(f"      Mean: {corr['energy_difference']['mean']:.1f}")
        print(f"      Std dev: {corr['energy_difference']['std']:.1f}")
        print(f"      Range: [{corr['energy_difference']['min']:.1f}, {corr['energy_difference']['max']:.1f}]")
        print(f"      Variability: {corr['energy_difference']['interpretation']}")
        print(f"      CV: {corr['energy_difference']['coefficient_of_variation']:.3f}")

        # Interpretation summary
        if corr['pearson_r'] > 0.9:
            print(f"\n   ✅ Strong positive correlation: more negative canary → more negative full")
        elif corr['pearson_r'] > 0.7:
            print(f"\n   ✅ Good positive correlation: canary is a useful predictor")
        else:
            print(f"\n   ⚠️  Weak correlation: canary may not be a reliable predictor")

        if corr['energy_difference']['coefficient_of_variation'] < 0.1:
            print(f"   ✅ Relationship is roughly constant offset (~{corr['energy_difference']['mean']:.0f} energy units)")
        elif corr['energy_difference']['coefficient_of_variation'] < 0.3:
            print(f"   ⚠️  Relationship has moderate variability (not perfectly constant offset)")
        else:
            print(f"   ⚠️  Relationship is non-linear or highly variable")

    # Calculate time savings
    if stats['canary']['time_stats']['mean'] and stats['full']['time_stats']['mean']:
        avg_canary = stats['canary']['time_stats']['mean']
        avg_full = stats['full']['time_stats']['mean']

        # Time saved by using canary to filter
        fn = stats['analysis']['false_negatives']
        tn = stats['analysis']['true_negatives']

        # Without canary: run full on everything
        time_without_canary = stats['total_nonces'] * avg_full

        # With canary: run canary on everything, full only on canary passes
        time_with_canary = (stats['total_nonces'] * avg_canary +
                           stats['canary']['passed'] * avg_full)

        time_saved = time_without_canary - time_with_canary
        speedup = time_without_canary / time_with_canary if time_with_canary > 0 else 0

        print(f"\n⚡ Time savings:")
        print(f"   Without canary: {time_without_canary:.1f}s ({time_without_canary/60:.1f}m)")
        print(f"   With canary: {time_with_canary:.1f}s ({time_with_canary/60:.1f}m)")
        print(f"   Time saved: {time_saved:.1f}s ({time_saved/60:.1f}m)")
        print(f"   Speedup: {speedup:.2f}x")

    # Save results
    output_data = {
        'miner_type': args.miner_type,
        'difficulty_energy': args.difficulty_energy,
        'duration_spec': args.duration,
        'duration_minutes': duration_minutes,
        'min_diversity': args.min_diversity,
        'min_solutions': args.min_solutions,
        'seed': args.seed,
        'topology': topology.solver_name,
        'statistics': stats,
        'timestamp': utc_timestamp()
    }

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"canary_test_{args.miner_type}_{args.duration}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
