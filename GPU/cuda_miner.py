"""GPU miner using CUDA via CudaSASampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import signal
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
from GPU.cuda_sa import CudaSASampler


class CudaMiner(BaseMiner):
    def __init__(self, miner_id: str, device: str = "0", **cfg):
        # Initialize CUDA SA sampler
        sampler = CudaSASampler()
        super().__init__(miner_id, sampler)
        # Now update sampler with our logger
        sampler.logger = self.logger
        self.miner_type = "GPU-CUDA-SA"
        self.device = device
        
        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of CUDA resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"CUDA miner {self.miner_id} received SIGTERM, cleaning up GPU device {self.device}...")
        
        # CUDA-specific cleanup
        try:
            # Import CUDA modules for cleanup
            try:
                import cupy as cp
                # Reset CUDA device to free all memory and contexts
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.runtime.deviceReset()
                if hasattr(self, 'logger'):
                    self.logger.info(f"CUDA device {self.device} reset completed")
            except ImportError:
                # Try alternative CUDA cleanup if CuPy not available
                try:
                    import pycuda.driver as cuda
                    # Initialize CUDA if not already done
                    if not hasattr(cuda, '_initialized') or not cuda._initialized:
                        cuda.init()
                    # Get device and reset
                    device = cuda.Device(int(self.device))
                    context = device.make_context()
                    context.pop()
                    context.detach()
                    if hasattr(self, 'logger'):
                        self.logger.info(f"PyCUDA device {self.device} context cleanup completed")
                except ImportError:
                    if hasattr(self, 'logger'):
                        self.logger.warning("No CUDA library available for device reset")
            
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
            
            # Reset sampler state if possible
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during CUDA miner cleanup: {e}")
        
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
        """Mine a block using CUDA GPU acceleration with batched multi-nonce evaluation.

        Args:
            prev_block: Previous block in the chain
            node_info: Node information containing miner_id and other details
            requirements: BlockRequirements object with difficulty settings
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

        params = adapt_parameters(difficulty_energy, min_diversity, min_solutions)
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        # Batch size: number of nonces to evaluate simultaneously
        # Constrained by both GPU hardware and workspace allocation
        num_reads = params.get('num_reads', 100)
        max_workspace_threads = getattr(self.sampler, 'max_threads_per_call', 1024)

        try:
            import cupy as cp
            # Get number of SM (streaming multiprocessors)
            device_id = cp.cuda.runtime.getDevice()
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)
            multiprocessor_count = device_props['multiProcessorCount']

            # Limit batch size by workspace constraint: total_reads <= max_workspace_threads
            # total_reads = batch_size * num_reads, so batch_size <= max_workspace_threads / num_reads
            max_batch_by_workspace = max_workspace_threads // num_reads

            # Use minimum of SM count and workspace limit
            batch_size = min(multiprocessor_count, max_batch_by_workspace)
            batch_size = max(1, batch_size)  # At least 1

            self.logger.info(f"Detected {multiprocessor_count} CUDA SMs, workspace limit {max_workspace_threads} threads")
            self.logger.info(f"Batch size: {batch_size} nonces/batch (limited by workspace: {max_batch_by_workspace}, reads/nonce: {num_reads})")
        except Exception as e:
            # Fallback: ensure we don't exceed workspace
            batch_size = max(1, max_workspace_threads // num_reads)
            self.logger.warning(f"Could not detect CUDA SM count ({e}), using batch_size={batch_size}")

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
                params = adapt_parameters(current_requirements.difficulty_energy, current_requirements.min_diversity, current_requirements.min_solutions)
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

            # Programming error if batch is invalid - fail fast
            assert len(batch_nonces) == len(batch_salts) == len(batch_problems), \
                f"Batch arrays must have same length: nonces={len(batch_nonces)}, salts={len(batch_salts)}, problems={len(batch_problems)}"
            assert len(batch_problems) > 0, \
                f"Batch should never be empty (batch_size={batch_size})"

            # Pregenerate next batch while GPU is working (overlap computation)
            next_batch_nonces = []
            next_batch_salts = []
            next_batch_problems = []

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start

            # Sample from CUDA GPU using batched evaluation
            try:
                # For CUDA, use adaptive parameters
                num_sweeps = params.get('num_sweeps', 512)
                num_reads = params.get('num_reads', 100)

                actual_batch_size = len(batch_problems)
                self.logger.debug(f"Batched: {actual_batch_size} nonces × {num_reads} reads/nonce, {num_sweeps} sweeps")

                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start

                # Batched kernel dispatch - evaluate all problems in one GPU call
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

                self.logger.debug(f"Batched sampling completed: {sample_time:.2f}s for {actual_batch_size} problems")

                # Estimate CUDA timing components
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
            total_nonces_evaluated += actual_batch_size

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
                    f"Best energy: {best_energy:.0f}"
                )

                last_report_time = current_time
                last_report_nonces = total_nonces_evaluated

        self.logger.info("Stopping mining, no results found")
        return None


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Optimized for GPU CUDA performance - uses much lower sweep counts than CPU
    since GPU parallelism allows faster convergence with fewer sweeps.
    """
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # GPU CUDA parameters - optimized for fast convergence
    # GPU can afford fewer sweeps due to massive parallelism
    base_sweeps = 512  # Much lower base than CPU (was 512, now 8x faster)
    # Use square root scaling instead of exponential to prevent explosion
    num_sweeps = int(base_sweeps * (difficulty_factor ** 0.5))  # Gentler scaling
    num_reads = max(int(min_solutions) * 2, 32)  # Reduce reads slightly

    return {
        'num_sweeps': max(32, min(num_sweeps, 512)),     # Much lower max (was 32768)
        'num_reads': max(32, min(num_reads, 200)),       # Reasonable bounds
    }