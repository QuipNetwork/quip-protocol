"""CPU miner using SimulatedAnnealingStructuredSampler."""
from __future__ import annotations

import math
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Optional

from shared.base_miner import BaseMiner, MiningResult
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
from shared.quantum_proof_of_work import generate_ising_model_from_nonce, ising_nonce_from_block
from shared.block_requirements import compute_current_requirements

class SimulatedAnnealingMiner(BaseMiner):
    def __init__(self, miner_id: str, sampler=None, topology=None, **cfg):
        if sampler is None:
            sampler = SimulatedAnnealingStructuredSampler(topology=topology)
        self.nodes = sampler.nodes
        self.edges = sampler.edges
        super().__init__(miner_id, sampler)
        self.miner_type = "CPU"

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of CPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"CPU miner {self.miner_id} received SIGTERM, cleaning up...")
        
        # CPU-specific cleanup
        try:
            # Reset any persistent library state
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()
            
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during CPU miner cleanup: {e}")
        
        # Exit gracefully
        sys.exit(0)
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        drain: bool = False,
    ) -> Optional[MiningResult]:
        """Mine a block using simulated annealing.

        Args:
            prev_block: Previous block object containing header, data, and other block information
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
        nodes = self.nodes
        edges = self.edges

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

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                break

            # Generate random salt for each attempt
            salt = random.randbytes(32)
            
            # Generate quantum model using deterministic block-based seeding
            # Use 64 variables to match original working energy scale
            nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            # Update requirements if necessary.
            updated_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
            if current_requirements != updated_requirements:
                current_requirements = updated_requirements
                params = adapt_parameters(
                    difficulty_energy,
                    min_diversity,
                    min_solutions,
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
                            self.logger.info(f"[Block-{cur_index}] Already Mined at this difficulty! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                            return result

            # Increment sweeps slowly over time (reads stay constant)
            current_time = time.time()
            if current_time - last_increment_time >= increment_interval:
                # Increase sweeps by 1% toward max
                current_num_sweeps = min(max_num_sweeps, int(current_num_sweeps * 1.05))
                last_increment_time = current_time

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start

            # Sample from simulated annealer
            try:
                # Use current sweeps (incrementing) and constant reads
                num_sweeps = current_num_sweeps
                # num_reads is constant throughout mining
                
                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start
                
                # Build sampling parameters based on sampler type
                sampling_params = {
                    'h': h,
                    'J': J,
                    'num_reads': num_reads,
                    'num_sweeps': num_sweeps
                }
                
                sampleset = self.sampler.sample_ising(**sampling_params)
                sample_time = time.time() - sample_start
                
                # Estimate SA timing components
                self.timing_stats['sampling'].append(sample_time * 1e6)  # Convert to microseconds
                self.timing_stats['preprocessing'].append((time.time() - preprocess_start) * 1e6)
            except Exception as e:
                if stop_event.is_set():
                    self.logger.info("Interrupted during sampling")
                    return None
                # Log detailed diagnostic info for debugging rare sampling errors
                import traceback
                self.logger.error(
                    f"Sampling error: {e}\n"
                    f"  Topology: nodes={len(nodes)}, edges={len(edges)}\n"
                    f"  h keys: {len(h)}, J keys: {len(J)}\n"
                    f"  Nonce: {nonce}, Salt: {salt.hex()[:8]}...\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                continue

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start

            # Update sample counts
            all_energies = sampleset.record.energy
            self.timing_stats['total_samples'] += len(all_energies)
            self.timing_stats['blocks_attempted'] += 1

            result = self.evaluate_sampleset(sampleset, current_requirements, nodes, edges, nonce, salt, prev_timestamp, start_time)
            self.logger.info(f"CPU len(nodes)={len(nodes)}, len(edges)={len(edges)}, len(energies)={len(all_energies)}")

            # Track postprocessing time
            self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)

            if result:
                self.logger.info(f"[Block-{cur_index}] Mined! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                return result
                        
            # Update top samples with this one
            self.update_top_samples(sampleset, nonce, salt, current_requirements)

            progress += 1

            # Progress update
            if progress % 10 == 0:
                best_energy = min(self.top_attempts[0].sampleset.record.energy) if self.top_attempts else float('inf')
                self.logger.info(
                    f"Progress: {progress} attempts, best energy: {best_energy:.2f} | "
                    f"Sweeps: {current_num_sweeps}/{max_num_sweeps}, Reads: {num_reads}"
                )

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

    Uses GSE-based difficulty calculation with log-linear interpolation
    in sweep space, calibrated from experimental data.

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

    # CPU SA calibration ranges (from experimental data)
    min_sweeps = 64      # Easiest difficulty
    max_sweeps = 4096    # Hardest difficulty (avoid overfitting beyond this)

    # Direct linear scaling: difficulty × max_sweeps
    # Example: difficulty=0.5 → 0.5 × 4096 = 2048 sweeps
    num_sweeps = max(min_sweeps, int(difficulty * max_sweeps))

    # Reads scale with both difficulty and min_solutions requirement
    base_reads = max(int(min_solutions) * 4, 64)
    max_reads = max(int(min_solutions) * 8, 512)
    num_reads = max(base_reads, int(difficulty * max_reads))

    return {
        'num_sweeps': num_sweeps,
        'num_reads': num_reads,
    }

