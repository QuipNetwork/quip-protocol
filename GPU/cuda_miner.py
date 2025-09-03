"""GPU miner using CUDA via GPUSampler(device)."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import sys
import time
from typing import List, Optional, Tuple

import dimod
import numpy as np
import json

from shared.base_miner import BaseMiner, MiningResult
from shared.block_requirements import BlockRequirements, compute_current_requirements
from shared.quantum_proof_of_work import (
    calculate_diversity,
    select_diverse_solutions,
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)
from GPU.sampler import GPUSampler


class CudaMiner(BaseMiner):
    def __init__(self, miner_id: str, device: str = "0", **cfg):
        sampler = GPUSampler(str(device))
        super().__init__(miner_id, sampler)
        self.miner_type = f"GPU-LOCAL:{device}"
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using CUDA GPU acceleration.

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
        
        self.logger.debug(f"requirements: {requirements}")

        cur_index = prev_block.header.index + 1

        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index}...")

        # Extract requirements from BlockRequirements object
        difficulty_energy = requirements.difficulty_energy
        min_diversity = requirements.min_diversity
        min_solutions = requirements.min_solutions

        # Apply difficulty decay based on elapsed time since previous block
        current_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
        difficulty_energy = current_requirements.difficulty_energy
        min_diversity = current_requirements.min_diversity
        min_solutions = current_requirements.min_solutions

        params = adapt_parameters(difficulty_energy, min_diversity, min_solutions)
        self.logger.debug(f"Adaptive params: {params}")
        
        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        while self.mining and not stop_event.is_set():
            # Generate random salt for each attempt
            salt = random.randbytes(32)
            
            # Generate quantum model using deterministic block-based seeding
            timestamp = int(time.time())
            nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)

            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            # Update requirements if necessary.
            updated_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
            if current_requirements != updated_requirements:
                current_requirements = updated_requirements
                # Check if any existing results meet the new requirements
                for sample in self.top_attempts:
                    if min(sample.sampleset.record.energy) <= current_requirements.difficulty_energy:
                        result = self.evaluate_sampleset(sample.sampleset, current_requirements, nodes, edges, 
                                                         sample.nonce, sample.salt, prev_timestamp, start_time)
                        if result:
                            self.logger.info(f"Found valid block! Nonce: {sample.nonce}, Energy: {result.energy:.2f}, Time: {result.mining_time:.2f}s")
                            return result
                difficulty_energy = current_requirements.difficulty_energy
                min_diversity = current_requirements.min_diversity
                min_solutions = current_requirements.min_solutions

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from GPU
            try:
                # For GPU, use adaptive parameters
                num_sweeps = params.get('num_sweeps', 512)
                num_reads = params.get('num_reads', 100)
                
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
                
                # Estimate GPU timing components
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
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < difficulty_energy)[0]
            
            # Update sample counts
            self.timing_stats['total_samples'] += len(sampleset.record.energy)
            self.timing_stats['blocks_attempted'] += 1

            # Process results from this mining attempt
            all_energies = sampleset.record.energy
            best_energy = float(np.min(all_energies)) if len(all_energies) > 0 else float('inf')
            
            # Create mining result for this attempt
            mining_time = time.time() - start_time
            
            # Get unique solutions that meet energy threshold
            valid_solutions = []
            seen = set()
            for idx in valid_indices:
                solution = tuple(sampleset.record.sample[idx])
                if solution not in seen:
                    seen.add(solution)
                    valid_solutions.append(list(solution))
            
            # Calculate diversity for valid solutions
            diversity = calculate_diversity(valid_solutions) if valid_solutions else 0.0
            
            # Filter solutions if we have enough
            if len(valid_solutions) >= min_solutions:
                selected_solutions_indices = select_diverse_solutions(valid_solutions, min_solutions)
                filtered_solutions = [valid_solutions[i] for i in selected_solutions_indices]
                final_diversity = calculate_diversity(filtered_solutions)
                best_energy = min(all_energies[selected_solutions_indices])
            else:
                filtered_solutions = valid_solutions
                final_diversity = diversity
            
            # Create result for this attempt
            attempt_result = MiningResult(
                miner_id=self.miner_id,
                miner_type=self.miner_type,
                nonce=nonce,
                salt=salt,
                timestamp=int(time.time()),
                prev_timestamp=prev_timestamp,
                solutions=filtered_solutions,
                energy=best_energy,
                diversity=final_diversity,
                num_valid=len(valid_solutions),
                mining_time=mining_time,
                node_list=nodes,
                edge_list=edges,
                variable_order=nodes
            )
            
            # Track postprocessing time
            self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
            
            # Update top results with this attempt
            self.update_top_samples(sampleset, nonce, salt, requirements)
            
            # Check if this result meets current requirements
            if (len(valid_solutions) >= min_solutions and 
                final_diversity >= min_diversity and 
                best_energy <= difficulty_energy):
                
                self.logger.info(f"Found valid block! Nonce: {nonce}, Energy: {best_energy:.2f}, Time: {mining_time:.2f}s")
                self.logger.info(f"Mining attempt completed - Best Energy: {best_energy:.2f}, Valid Solutions: {len(valid_solutions)}, Diversity: {final_diversity:.3f}, Params: {json.dumps(params)}")
                return attempt_result
            else:
                # Log this attempt but continue mining
                self.logger.debug(f"Mining attempt - Energy: {best_energy:.2f}, Valid: {len(valid_solutions)}, Diversity: {final_diversity:.3f} (requirements: energy<={difficulty_energy:.2f}, valid>={min_solutions}, diversity>={min_diversity:.3f})")

            progress += 1

            # Progress update
            if progress % 10 == 0:
                if self.top_attempts:
                    best_result = self.top_attempts[0]
                    self.logger.info(f"Progress: {progress} attempts, best result so far - Energy: {min(best_result.sampleset.record.energy):.2f}")
                else:
                    self.logger.info(f"Progress: {progress} attempts, no results yet")

        # If we exit the loop due to stop event
        if stop_event.is_set():
            if self.top_attempts:
                best_result = self.top_attempts[0]
                self.logger.info(f"Stopping mining, best result was - Energy: {min(best_result.sampleset.record.energy):.2f}")
            else:
                self.logger.info("Stopping mining, no results found")
        return None

    def evaluate_sampleset(self, sampleset: dimod.SampleSet, requirements: BlockRequirements, nodes: List[int], edges: List[Tuple[int, int]], nonce: int, salt: bytes, prev_timestamp: int, start_time: float) -> Optional[MiningResult]:
        """Convert a sample set into a mining result if it meets requirements, otherwise return None."""
        difficulty_energy = requirements.difficulty_energy
        min_diversity = requirements.min_diversity
        min_solutions = requirements.min_solutions
        best_energy = float('inf')
        valid_solutions = []
        diversity = 0.0
        result = None

        try:
            # Best Energy
            all_energies = sampleset.record.energy
            if len(all_energies) == 0:
                raise ValueError("No samples in sampleset")

            best_energy = float(np.min(all_energies))
            if best_energy > difficulty_energy:
                raise ValueError(f"Best energy {best_energy} exceeds difficulty energy {difficulty_energy}")
                
            # Process results from this mining attempt
            # Find all solutions meeting energy threshold
            valid_indices = np.where(all_energies < difficulty_energy)[0]
            # Get unique solutions that meet energy threshold
            valid_solutions = []
            seen = set()
            for idx in valid_indices:
                solution = tuple(sampleset.record.sample[idx])
                if solution not in seen:
                    seen.add(solution)
                    valid_solutions.append(list(solution))
            if len(valid_solutions) < min_solutions:
                raise ValueError(f"Insufficient valid solutions: {len(valid_solutions)} < {min_solutions}")
            
            # Filter solutions if we have too many
            filtered_solutions = valid_solutions
            final_diversity = diversity
            if len(valid_solutions) >= min_solutions:
                selected_solutions_indices = select_diverse_solutions(valid_solutions, min_solutions)
                filtered_solutions = [valid_solutions[i] for i in selected_solutions_indices]
                final_diversity = calculate_diversity(filtered_solutions)
                best_energy = min(all_energies[selected_solutions_indices])

            if final_diversity < min_diversity:
                raise ValueError(f"Insufficient diversity: {final_diversity} < {min_diversity}")

            # Create mining result for this attempt
            mining_time = time.time() - start_time
            
            # Create result for this attempt
            result = MiningResult(
                miner_id=self.miner_id,
                miner_type=self.miner_type,
                nonce=nonce,
                salt=salt,
                timestamp=int(time.time()),
                prev_timestamp=prev_timestamp,
                solutions=filtered_solutions,
                energy=best_energy,
                diversity=final_diversity,
                num_valid=len(valid_solutions),
                mining_time=mining_time,
                node_list=nodes,
                edge_list=edges,
                variable_order=nodes
            )
        except ValueError as e:
            self.logger.debug(f"Failed to meet requirements: {e}")
        finally:
            self.logger.info(f"Mining attempt - Energy: {best_energy:.2f}, Valid: {len(valid_solutions)}, Diversity: {diversity:.3f} (requirements: energy<={difficulty_energy:.2f}, valid>={min_solutions}, diversity>={min_diversity:.3f})")
        return result


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Supports either a BlockRequirements object or a dict with keys:
    'difficulty_energy', 'min_diversity', 'min_solutions'.
    """
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # GPU CUDA parameters
    base_sweeps = 512
    num_sweeps = int(base_sweeps * (difficulty_factor ** 1.5))  # Exponential scaling
    num_reads = max(int(min_solutions) * 3, 64)  # At least 3x required solutions

    return {
        'num_sweeps': max(128, min(num_sweeps, 32768)),  # Reasonable bounds for GPU
        'num_reads': max(64, min(num_reads, 1000)),      # Reasonable bounds
    }