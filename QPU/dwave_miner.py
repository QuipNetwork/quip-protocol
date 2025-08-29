"""QPU miner using D-Wave sampler or mock via create_dwave_sampler."""
from __future__ import annotations

import multiprocessing
import random
import sys
import time
from typing import Optional

import numpy as np

from shared.base_miner import BaseMiner, MiningResult
from QPU.dwave_sampler import create_dwave_sampler


class DWaveMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        super().__init__(miner_id)
        self.miner_type = "QPU"
        self.sampler = create_dwave_sampler()
        
    def mine_block(
        self,
        block_header: str,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using D-Wave QPU or mock sampler.
        
        Args:
            block_header: Format is f"{previous_hash}{index}{timestamp}{data}"
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
        """
        self.mining = True
        progress = 0  # Progress counter for logging
        start_time = time.time()
        
        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        print(f"{self.miner_id} started...")

        while self.mining and not stop_event.is_set():
            # Check if we should stop before generating model
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return None

            # Generate random nonce for each attempt
            nonce = random.randint(0, sys.maxsize)
            
            # Generate quantum model
            h, J = self.generate_quantum_model(block_header, nonce)

            # Check again before sampling
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return None

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from QPU
            try:
                sampleset = self.sampler.sample_ising(
                    h, J, 
                    num_reads=100, 
                    answer_mode='raw',
                    annealing_time=self.adaptive_params.get('quantum_annealing_time', 20.0)
                )
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
                    print(f"{self.miner_id} interrupted during sampling")
                    return None
                print(f"{self.miner_id} sampling error: {e}")
                continue

            # Check if interrupted before processing results
            if stop_event.is_set():
                print(f"{self.miner_id} interrupted")
                return None

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start
            
            # Find all solutions meeting energy threshold
            valid_indices = np.where(sampleset.record.energy < self.difficulty_energy)[0]
            
            # Update sample counts
            self.timing_stats['total_samples'] += len(sampleset.record.energy)
            self.timing_stats['blocks_attempted'] += 1

            if len(valid_indices) >= self.min_solutions:
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
                min_energy = float(np.min(sampleset.record.energy))

                # Filter excess solutions to maintain diversity
                filtered_solutions = self.filter_diverse_solutions(valid_solutions, self.min_solutions)

                # Recalculate diversity after filtering
                final_diversity = self.calculate_diversity(filtered_solutions)
                print(f"{self.miner_id} found sufficient solutions! Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}, Diversity: {diversity:.3f}, Final Diversity: {final_diversity:.3f}")

                # Track postprocessing time
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                
                # Check if diversity requirement is met
                if final_diversity >= self.min_diversity and len(valid_solutions) >= self.min_solutions:
                    mining_time = time.time() - start_time
                    min_energy = float(np.min(sampleset.record.energy[valid_indices]))

                    result = MiningResult(
                        miner_id=self.miner_id,
                        miner_type=self.miner_type,
                        nonce=nonce,
                        solutions=filtered_solutions,
                        energy=min_energy,
                        diversity=final_diversity,
                        num_valid=len(valid_solutions),
                        mining_time=mining_time
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