"""QPU miner using D-Wave sampler or mock via create_dwave_sampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import time
from typing import Optional, cast, Mapping, Tuple, Any

from QPU.dwave_sampler import DWaveSamplerWrapper
from shared.base_miner import BaseMiner, MiningResult
from shared.block_requirements import compute_current_requirements
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)

class DWaveMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        # cfg parameter is reserved for future configuration options
        _ = cfg  # Suppress unused parameter warning
        sampler = DWaveSamplerWrapper()
        super().__init__(miner_id, sampler, miner_type="QPU")
        self.miner_type = "QPU"
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using D-Wave QPU or mock sampler.

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

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                break

            attempt_no = progress + 1
            self.logger.info(f"[QPU] Starting mining attempt {attempt_no} (reads={params.get('num_reads', 100)}, anneal={params.get('quantum_annealing_time', 20.0)}µs)")

            # Generate random salt for each attempt
            salt = random.randbytes(32)

            # Generate quantum model using deterministic block-based seeding
            nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)

            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            # Update requirements if necessary.
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
                            self.logger.info(f"[Block-{cur_index}] Already Mined at this difficulty! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                            return result

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from QPU
            try:
                # For QPU, use adaptive parameters
                num_reads = params.get('num_reads', 100)
                annealing_time = params.get('quantum_annealing_time', 20.0)
                
                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start
                
                # Cast h and J to match protocol expectations (int is a valid Variable type)
                h_cast = cast(Mapping[Any, float], h)
                J_cast = cast(Mapping[Tuple[Any, Any], float], J)

                sampleset = self.sampler.sample_ising(
                    h_cast, J_cast,
                    num_reads=num_reads,
                    answer_mode='raw',
                    annealing_time=annealing_time
                )
                sample_time = time.time() - sample_start
                self.logger.debug(f"QPU sampling completed in {sample_time:.2f}s")
                
                # Estimate QPU timing components  
                self.timing_stats['sampling'].append(sample_time * 1e6)  # Convert to microseconds
                self.timing_stats['preprocessing'].append((time.time() - preprocess_start) * 1e6)
                # Extract QPU timing information if available
                if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                    timing = sampleset.info['timing']
                    if 'qpu_anneal_time_per_sample' in timing:
                        self.timing_stats['quantum_annealing_time'].append(
                            timing['qpu_anneal_time_per_sample']
                        )
                    if 'qpu_sampling_time' in timing:
                        self.timing_stats['sampling'].append(timing['qpu_sampling_time'])
                    if 'qpu_programming_time' in timing:
                        self.timing_stats['preprocessing'].append(timing['qpu_programming_time'])
            except Exception as e:
                if stop_event.is_set():
                    self.logger.info("Interrupted during sampling")
                    return None
                self.logger.error(f"Sampling error: {e}")
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
            self.logger.debug(f"QPU sampleset evaluated in {time.time() - postprocess_start:.2f}s")
            self.logger.info(f"QPU len(nodes)={len(nodes)}, len(edges)={len(edges)}, len(energies)={len(all_energies)}")

            # Track postprocessing time
            self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)

            if result:
                self.logger.info(f"[Block-{cur_index}] Mined! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                return result
                        
            # Update top samples with this one
            self.update_top_samples(sampleset, nonce, salt, current_requirements)

            progress += 1

            # Progress update every attempt (QPU attempts are slower)
            if self.top_attempts:
                best_energy = min(self.top_attempts[0].sampleset.record.energy)
                self.logger.info(f"Progress: {progress} attempts, best energy so far {best_energy:.2f}")
            else:
                self.logger.info(f"Progress: {progress} attempts, awaiting first results...")

        self.logger.info("Stopping mining, no results found")
        return None


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum diversity requirement (reserved for future use)
        min_solutions: Minimum number of valid solutions required
    """
    # min_diversity parameter is reserved for future adaptive parameter tuning
    _ = min_diversity  # Suppress unused parameter warning
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # QPU parameters (D-Wave quantum processor optimized)
    base_reads = 256  # QPU uses reads instead of sweeps
    num_reads = min(int(base_reads * (difficulty_factor ** 0.5)), 256)
    
    # QPU-specific annealing time (microseconds)
    base_annealing_time = 5.0
    annealing_time = min(base_annealing_time * (difficulty_factor ** 0.05), 1000.0)  # Max 1ms per D-WAve Docs

    return {
        # 'num_reads': max(min_solutions*2, num_reads),
        'num_reads': 2048,
        'quantum_annealing_time': max(5.0, annealing_time),  # Min 5 microseconds
    }