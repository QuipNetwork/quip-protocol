"""GPU miner using Metal/MPS via GPUSampler('mps')."""
from __future__ import annotations

import multiprocessing
import random
import sys
import time
from typing import Optional

import numpy as np

from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
    energies_for_solutions,
)
from GPU.sampler import LocalGPUSampler as GPUSampler  # temporary alias until rename


class MetalMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        sampler = GPUSampler("mps")
        super().__init__(miner_id, sampler)
        self.miner_type = "GPU-MPS"
        
    def mine_block(
        self,
        block,
        requirements,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using Metal/MPS GPU acceleration.
        
        Args:
            block: Block object containing header, data, and other block information
            requirements: NextBlockRequirements object with difficulty settings
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
        """
        self.mining = True
        progress = 0  # Progress counter for logging
        start_time = time.time()
        
        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        self.logger.info("Started...")

        # Extract requirements from NextBlockRequirements object
        difficulty_energy = requirements.difficulty_energy
        min_diversity = requirements.min_diversity
        min_solutions = requirements.min_solutions

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                self.logger.info("Interrupted")
                return None

            # Generate random nonce for each attempt
            nonce = random.randint(0, sys.maxsize)

            # Build topology lists from sampler
            nodes = [int(n) for n in self.sampler.nodelist]
            edges = [(int(u), int(v)) for (u, v) in self.sampler.edgelist]

            # Deterministic seed
            cur_index = block.header.index + 1
            seed = ising_nonce_from_block(block.hash, self.miner_id, cur_index, nonce)

            # Deterministic Ising model
            h, J = generate_ising_model_from_nonce(seed, nodes, edges)

            # Check again before sampling
            if stop_event.is_set():
                self.logger.info("Interrupted")
                return None

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from Metal GPU
            try:
                # For Metal, use adaptive parameters
                num_sweeps = self.adaptive_params.get('num_sweeps', 512)
                
                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start
                
                # Build sampling parameters
                sampling_params = {
                    'h': h,
                    'J': J,
                    'num_reads': 100,
                    'num_sweeps': num_sweeps
                }
                
                # Add beta parameters for Metal samplers
                sampling_params['beta_range'] = self.adaptive_params.get('beta_range', [0.1, 10.0])
                sampling_params['beta_schedule_type'] = self.adaptive_params.get('beta_schedule', 'geometric')
                
                sampleset = self.sampler.sample_ising(**sampling_params)
                sample_time = time.time() - sample_start
                
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
                diversity = self.calculate_diversity(valid_solutions)

                # Deterministic energies with shared function
                energies = energies_for_solutions(valid_solutions, h, J, nodes)
                min_energy = float(min(energies)) if energies else 0.0

                # Filter excess solutions to maintain diversity
                filtered_solutions = self.filter_diverse_solutions(valid_solutions, min_solutions)

                # Recalculate diversity after filtering
                final_diversity = self.calculate_diversity(filtered_solutions)
                self.logger.info(f"Found sufficient solutions! Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}, Diversity: {diversity:.3f}, Final Diversity: {final_diversity:.3f}")

                # Track postprocessing time
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                
                # Check if diversity requirement is met
                if final_diversity >= min_diversity and len(valid_solutions) >= min_solutions:
                    mining_time = time.time() - start_time
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))

                    result = MiningResult(
                        miner_id=self.miner_id,
                        miner_type=self.miner_type,
                        nonce=nonce,
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
                    print(f"{self.miner_id} found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                    return result

            progress += 1

            # Progress update
            if progress % 10 == 0 and len(sampleset.record.energy) > 0:
                min_energy = float(np.min(sampleset.record.energy))
                print(f"{self.miner_id} - Progress: {progress}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")

        # If we exit the loop due to stop event
        if stop_event.is_set():
            print(f"{self.miner_id} stopped")
        return None