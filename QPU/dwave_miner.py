"""QPU miner using D-Wave sampler or mock via create_dwave_sampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import sys
import time
from typing import Optional

import numpy as np
import json

from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    calculate_diversity,
    filter_diverse_solutions,
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
    energy_of_solution,
)
from QPU.dwave_sampler import create_dwave_sampler


class DWaveMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        sampler = create_dwave_sampler()
        super().__init__(miner_id, sampler, miner_type="QPU")
        self.miner_type = "QPU"
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using D-Wave QPU or mock sampler.
        
        Args:
            prev_block: Previous block in the chain
            node_info: Node information containing miner_id and other details
            requirements: NextBlockRequirements object with difficulty settings
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
        """
        self.mining = True
        progress = 0  # Progress counter for logging
        start_time = time.time()
        
        self.logger.debug(f"requirements: {requirements}")

        cur_index = prev_block.header.index + 1

        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index}...")

        # Extract requirements from NextBlockRequirements object
        difficulty_energy = requirements.difficulty_energy
        min_diversity = requirements.min_diversity
        min_solutions = requirements.min_solutions

        params = adapt_parameters(difficulty_energy, min_diversity, min_solutions)
        self.logger.debug(f"Adaptive params: {params}")
        
        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                if self.top_results:
                    best_result = self.top_results[0]
                    self.logger.info(f"Stopping mining, best result was - Energy: {best_result.energy:.2f}, Valid Solutions: {best_result.num_valid}, Diversity: {best_result.diversity:.3f}")
                else:
                    self.logger.info("Stopping mining, no results found")
                return None

            # Generate random salt for each attempt
            salt = random.randbytes(32)
            
            # Generate quantum model using deterministic block-based seeding
            timestamp = int(time.time())
            nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)

            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            # Check again before sampling
            if stop_event.is_set():
                if self.top_results:
                    best_result = self.top_results[0]
                    self.logger.info(f"Stopping mining, best result was - Energy: {best_result.energy:.2f}, Valid Solutions: {best_result.num_valid}, Diversity: {best_result.diversity:.3f}")
                else:
                    self.logger.info("Stopping mining, no results found")
                return None

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
                
                sampleset = self.sampler.sample_ising(
                    h, J, 
                    num_reads=num_reads, 
                    answer_mode='raw',
                    annealing_time=annealing_time
                )
                sample_time = time.time() - sample_start
                
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

            # Check if interrupted before processing results
            if stop_event.is_set():
                self.logger.info("Interrupted")
                return None

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < difficulty_energy)[0]
            
            # Update sample counts
            self.timing_stats['total_samples'] += len(sampleset.record.energy)
            self.timing_stats['blocks_attempted'] += 1

            if len(valid_indices) >= min_solutions:
                # Get unique solutions
                valid_solutions = []
                seen = set()

                for idx in valid_indices:
                    solution = tuple(sampleset.record.sample[idx])
                    if solution not in seen:
                        seen.add(solution)
                        valid_solutions.append(list(solution))

                # Calculate diversity
                diversity = calculate_diversity(valid_solutions)
                min_energy = float(np.min(sampleset.record.energy))

                # Filter excess solutions to maintain diversity
                filtered_solutions = filter_diverse_solutions(valid_solutions, min_solutions)

                # Recalculate diversity after filtering
                final_diversity = calculate_diversity(filtered_solutions)
                self.logger.info(f"Found sufficient solutions! Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}, Diversity: {diversity:.3f}, Final Diversity: {final_diversity:.3f}")

                # Track postprocessing time
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                
                # Check if diversity requirement is met
                if final_diversity >= min_diversity and len(valid_solutions) >= min_solutions:
                    mining_time = time.time() - start_time
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))

                    energies = [energy_of_solution(sol, h, J, nodes) for sol in filtered_solutions]
                    min_energy = float(min(energies)) if energies else 0.0

                    result = MiningResult(
                        miner_id=self.miner_id,
                        miner_type=self.miner_type,
                        nonce=nonce,
                        salt=salt,
                        timestamp=timestamp,
                        solutions=filtered_solutions,
                        energy=min_energy,
                        diversity=final_diversity,
                        num_valid=len(filtered_solutions),
                        mining_time=mining_time,
                        node_list=nodes,
                        edge_list=edges,
                        variable_order=nodes
                    )

                    result_queue.put(result)
                    self.logger.info(f"Found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")

                    # Log mining attempt results
                    self.logger.info(f"Mining attempt completed - Best Energy: {result.energy:.2f}, Valid Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Params: {json.dumps(params)}")

                    # Update top results
                    self.update_top_results(result, requirements)

                    return result

            progress += 1

            # Progress update
            if progress % 10 == 0 and len(sampleset.record.energy) > 0:
                min_energy = float(np.min(sampleset.record.energy))
                self.logger.debug(f"Progress: {progress}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")

        # If we exit the loop due to stop event
        if stop_event.is_set():
            if self.top_results:
                best_result = self.top_results[0]
                self.logger.info(f"Stopping mining, best result was - Energy: {best_result.energy:.2f}, Valid Solutions: {best_result.num_valid}, Diversity: {best_result.diversity:.3f}")
            else:
                self.logger.info("Stopping mining, no results found")
        return None


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Supports either a NextBlockRequirements object or a dict with keys:
    'difficulty_energy', 'min_diversity', 'min_solutions'.
    """
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # QPU parameters (D-Wave quantum processor optimized)
    base_reads = 100  # QPU uses reads instead of sweeps
    num_reads = max(int(base_reads * (difficulty_factor ** 1.2)), int(min_solutions) * 2)  # Scale with difficulty
    
    # QPU-specific annealing time (microseconds)
    base_annealing_time = 20.0
    annealing_time = min(base_annealing_time * (difficulty_factor ** 0.5), 2000.0)  # Max 2ms

    return {
        'num_reads': max(64, min(num_reads, 10000)),     # Reasonable bounds for QPU
        'quantum_annealing_time': max(5.0, annealing_time),  # Min 5 microseconds
    }