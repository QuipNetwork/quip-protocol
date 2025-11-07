"""GPU miner using Metal/MPS via GPUSampler('mps')."""
from __future__ import annotations

import math
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import subprocess
import sys
import time
from typing import Optional

import numpy as np

from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce
)
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from GPU.metal_sa import MetalSASampler
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


def get_gpu_core_count() -> int:
    """Detect Apple Silicon GPU core count programmatically."""
    try:
        # Use grep to filter ioreg output - much faster and avoids Unicode issues
        result = subprocess.run(
            "ioreg -l | grep gpu-core-count",
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.stdout:
            # Parse line like: | |   |   |   "gpu-core-count" = 40
            for line in result.stdout.splitlines():
                if 'gpu-core-count' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) == 2:
                        return int(parts[1].strip())
    except Exception as e:
        raise RuntimeError(f"Failed to detect GPU core count: {e}")

    raise RuntimeError("Could not find gpu-core-count in ioreg output")


class MetalMiner(BaseMiner):
    def __init__(self, miner_id: str, topology=None, **cfg):
        try:
            # Initialize base miner first to get the logger
            sampler = MetalSASampler(topology=topology)
            super().__init__(miner_id, sampler, miner_type="GPU-Metal")
            # Now update sampler with our logger
            sampler.logger = self.logger
            self.miner_type = "GPU-Metal"

            self.logger.info(f"Using MetalSASampler (Simulated Annealing)")
        except Exception as e:
            # For fallback case, we can't use logger yet since super().__init__() wasn't called
            sampler = SimulatedAnnealingStructuredSampler(topology=topology)
            super().__init__(miner_id, sampler, miner_type="CPU-FALLBACK")
            self.miner_type = "CPU-FALLBACK"
            # Now we can use logger
            self.logger.warning(f"Metal GPU initialization failed, falling back to CPU: {e}")

        # GPU utilization control (0-100, default 100)
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(f"gpu_utilization must be between 1-100, got {self.gpu_utilization}")

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of Metal resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"Metal miner {self.miner_id} received SIGTERM, cleaning up Metal resources...")

        # Metal-specific cleanup
        try:
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()

            # Reset sampler state if possible
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during Metal miner cleanup: {e}")

        # Exit gracefully
        sys.exit(0)
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using Metal GPU acceleration.

        Args:
            prev_block: Previous block in the chain
            node_info: Node information containing miner_id and other details
            requirements: NextBlockRequirements object with difficulty settings
            prev_timestamp: Timestamp from the previous block header
            stop_event: Multiprocessing event to signal stop

        Returns:
            MiningResult if successful, None if stopped or failed
        """
        self.mining = True
        progress = 0  # Progress counter for logging
        self.top_attempts = []
        start_time = time.time()

        # Mining statistics
        total_nonces_evaluated = 0
        last_report_time = start_time
        last_report_nonces = 0
        report_interval = 10.0  # Report every 10 seconds

        self.logger.debug(f"requirements: {requirements}")

        cur_index = prev_block.header.index + 1

        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index}...")

        # Apply difficulty decay based on elapsed time since previous block
        current_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
        difficulty_energy = current_requirements.difficulty_energy
        min_diversity = current_requirements.min_diversity
        min_solutions = current_requirements.min_solutions

        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        params = adapt_parameters(
            difficulty_energy,
            min_diversity,
            min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges)
        )
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        # Track current sweeps for incremental increase (reads stay constant)
        current_num_sweeps = params.get('num_sweeps', 64)
        num_reads = params.get('num_reads', 100)  # Constant - doesn't increment
        max_num_sweeps = params.get('num_sweeps', 64)

        # Increment rate: increase by 5% every 30 seconds
        increment_interval = 30.0
        last_increment_time = start_time

        # Batch size: number of nonces to evaluate simultaneously
        # Scale batch size by utilization percentage to control GPU load
        gpu_cores = get_gpu_core_count()
        batch_size = max(1, int(gpu_cores * (self.gpu_utilization / 100.0)))
        self.logger.info(f"Detected {gpu_cores} GPU cores, using {batch_size} nonces per batch ({self.gpu_utilization}% utilization)")

        # Pregenerate first batch to start
        next_batch_nonces = []
        next_batch_salts = []
        next_batch_problems = []
        for _ in range(batch_size):
            salt = random.randbytes(32)
            nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
            next_batch_nonces.append(nonce)
            next_batch_salts.append(salt)
            next_batch_problems.append((h, J))

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating models
            if stop_event.is_set():
                break

            # Update requirements if necessary
            updated_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
            if current_requirements != updated_requirements:
                current_requirements = updated_requirements
                # Recompute adaptive parameters based on updated requirements
                params = adapt_parameters(
                    current_requirements.difficulty_energy,
                    current_requirements.min_diversity,
                    current_requirements.min_solutions,
                    num_nodes=len(nodes),
                    num_edges=len(edges)
                )
                self.logger.info(f"{self.miner_id} - updated adaptive params: {params}")
                # Check if any existing results meet the new requirements
                for sample in self.top_attempts:
                    if min(sample.sampleset.record.energy) <= current_requirements.difficulty_energy:
                        result = self.evaluate_sampleset(sample.sampleset, current_requirements, nodes, edges,
                                                         sample.nonce, sample.salt, prev_timestamp, start_time)
                        if result:
                            self.logger.info(f"[Block-{cur_index}] Already Mined at this difficulty! Nonce: {sample.nonce}, Salt: {sample.salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                            return result
                difficulty_energy = current_requirements.difficulty_energy
                min_diversity = current_requirements.min_diversity
                min_solutions = current_requirements.min_solutions

            # Use pregenerated batch (overlapping CPU work with GPU work)
            batch_nonces = next_batch_nonces
            batch_salts = next_batch_salts
            batch_problems = next_batch_problems

            # Pregenerate next batch while GPU is working (overlap computation)
            next_batch_nonces = []
            next_batch_salts = []
            next_batch_problems = []

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start

            # Increment sweeps slowly over time (reads stay constant)
            current_time = time.time()
            if current_time - last_increment_time >= increment_interval:
                # Increase sweeps by 1% toward max
                current_num_sweeps = min(max_num_sweeps, int(current_num_sweeps * 1.05))
                last_increment_time = current_time

            # Sample from Metal GPU using batched evaluation
            try:
                # Use current sweeps (incrementing) and constant reads
                num_sweeps = current_num_sweeps
                # num_reads is constant throughout mining

                self.logger.debug(f"Batched: {batch_size} nonces × {num_reads} reads/nonce, {num_sweeps} sweeps")

                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start

                # Task 5: Batched kernel dispatch - evaluate all problems in one GPU call
                h_list = [h for h, J in batch_problems]
                J_list = [J for h, J in batch_problems]
                batch_samplesets = self.sampler.sample_ising(h_list, J_list, num_reads=num_reads, num_sweeps=num_sweeps)

                # Pregenerate next batch while waiting (or after GPU completes)
                if len(next_batch_nonces) == 0:
                    for _ in range(batch_size):
                        salt = random.randbytes(32)
                        nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)
                        h_next, J_next = generate_ising_model_from_nonce(nonce, nodes, edges)
                        next_batch_nonces.append(nonce)
                        next_batch_salts.append(salt)
                        next_batch_problems.append((h_next, J_next))

                sample_time = time.time() - sample_start

                self.logger.debug(f"Batched sampling completed: {sample_time:.2f}s for {batch_size} problems")

                # Estimate Metal timing components
                self.timing_stats['sampling'].append(sample_time * 1e6)  # Convert to microseconds
                self.timing_stats['preprocessing'].append((time.time() - preprocess_start) * 1e6)
            except Exception as e:
                if stop_event.is_set():
                    self.logger.info("Interrupted during sampling")
                    return None
                self.logger.error(f"Sampling error: {e}")
                continue

            # Check if interrupted before processing results
            if stop_event.is_set():
                self.logger.info("Interrupted")
                return None

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start

            # Evaluate all results from the batch
            for nonce, salt, sampleset in zip(batch_nonces, batch_salts, batch_samplesets):
                # Update sample counts
                self.timing_stats['total_samples'] += len(sampleset.record.energy)
                self.timing_stats['blocks_attempted'] += 1

                result = self.evaluate_sampleset(sampleset, current_requirements, nodes, edges, nonce, salt, prev_timestamp, start_time)

                if result:
                    # Track postprocessing time
                    self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)

                    self.logger.info(f"[Block-{cur_index}] Mined! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                    return result

                # Update top samples with this one
                self.update_top_samples(sampleset, nonce, salt, current_requirements)

            # Track postprocessing time
            self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)

            progress += 1
            total_nonces_evaluated += batch_size

            # Periodic progress report
            current_time = time.time()
            time_since_last_report = current_time - last_report_time

            if time_since_last_report >= report_interval:
                elapsed_total = current_time - start_time
                nonces_since_last_report = total_nonces_evaluated - last_report_nonces
                nonces_per_sec = nonces_since_last_report / time_since_last_report
                avg_nonces_per_sec = total_nonces_evaluated / elapsed_total

                best_energy = min(self.top_attempts[0].sampleset.record.energy) if self.top_attempts else float('inf')

                self.logger.info(
                    f"[Mining Stats] "
                    f"Nonces: {total_nonces_evaluated} total, "
                    f"{nonces_per_sec:.1f}/s recent, "
                    f"{avg_nonces_per_sec:.1f}/s average | "
                    f"Time: {elapsed_total:.1f}s | "
                    f"Best energy: {best_energy:.0f} | "
                    f"Sweeps: {current_num_sweeps}/{max_num_sweeps}, "
                    f"Reads: {num_reads}"
                )

                last_report_time = current_time
                last_report_nonces = total_nonces_evaluated

        self.logger.info("Stopping mining, no results found")
        return None
    

def adapt_parameters(
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    num_nodes: int = DEFAULT_NUM_NODES,
    num_edges: int = DEFAULT_NUM_EDGES
):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Metal MPS strategy: Fast convergence with MORE READS, FEWER SWEEPS.
    Based on calibration showing Metal benefits from multiple quick attempts
    rather than deep convergence per attempt.

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum solution diversity required (reserved)
        min_solutions: Minimum number of valid solutions required
        num_nodes: Number of nodes in topology (default: DEFAULT_TOPOLOGY)
        num_edges: Number of edges in topology (default: DEFAULT_TOPOLOGY)

    Returns:
        Dictionary with num_sweeps and num_reads parameters
    """
    # Get normalized difficulty [0, 1]
    difficulty = energy_to_difficulty(
        difficulty_energy,
        num_nodes=num_nodes,
        num_edges=num_edges
    )

    # Metal calibration: Prefers fewer sweeps, more reads per calibration
    min_sweeps = 64      # Easiest difficulty (very fast convergence)
    max_sweeps = 512     # Hardest difficulty (still relatively fast)

    # Direct linear scaling: difficulty × max_sweeps
    num_sweeps = max(min_sweeps, int(difficulty * max_sweeps))

    # Metal benefits from MORE READS (multiple nonce strategy)
    min_reads = 32
    max_reads = 1024
    num_reads = max(min_reads, int(difficulty * max_reads))

    return {
        'num_sweeps': num_sweeps,
        'num_reads': max(num_reads, min_solutions),
    }