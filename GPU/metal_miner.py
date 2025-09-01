"""GPU miner using Metal/MPS via GPUSampler('mps')."""
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
from GPU.metal_sampler import MetalSampler


class MetalMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        try:
            sampler = MetalSampler("mps")
            super().__init__(miner_id, sampler, miner_type="GPU-MPS")
            self.miner_type = "GPU-MPS"
            print(f"INFO: Using optimized MetalSampler (64 sweeps, ~13s mining time)")
        except Exception as e:
            print(f"Metal GPU initialization failed, falling back to CPU: {e}")
            from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
            sampler = SimulatedAnnealingStructuredSampler()
            super().__init__(miner_id, sampler, miner_type="CPU-FALLBACK")
            self.miner_type = "CPU-FALLBACK"
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using Metal/MPS GPU acceleration.
        
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
            
            # Sample from Metal GPU
            try:
                # For Metal, use adaptive parameters
                # Metal compromise: Balance between performance and quality
                # CPU uses 4096, GPU uses 512, Metal uses 64 as reasonable middle ground
                num_sweeps = params.get('num_sweeps', 64)
                if self.sampler.sampler_type == "metal":
                    print(f"[METAL] Using {num_sweeps} sweeps (optimized for MPS performance)")
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
                
                # Metal performance info
                if self.sampler.sampler_type == "metal":
                    energies = sampleset.data_vectors['energy']
                    print(f"[METAL] Sampling completed: {sample_time:.2f}s, energy range: {min(energies):.1f} to {max(energies):.1f}")
                
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
            
            # Print intermediate results regardless of success
            all_energies = sampleset.record.energy
            best_energy = float(np.min(all_energies)) if len(all_energies) > 0 else float('inf')
            num_below_threshold = len(valid_indices)
            
            print(f"[METAL] Attempt {progress + 1}: nonce={nonce}")
            print(f"[METAL]   Samples: {len(all_energies)}, Best energy: {best_energy:.1f}")
            print(f"[METAL]   Below threshold ({difficulty_energy:.1f}): {num_below_threshold} samples")
            
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
                
                print(f"[METAL]   → Sufficient samples found! Processing...")
                print(f"[METAL]   → Unique solutions: {len(valid_solutions)}")
                print(f"[METAL]   → Initial diversity: {diversity:.3f}")
                print(f"[METAL]   → Final diversity: {final_diversity:.3f} (need {min_diversity:.3f})")
                
                self.logger.info(f"Found sufficient solutions! Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}, Diversity: {diversity:.3f}, Final Diversity: {final_diversity:.3f}")

                # Track postprocessing time
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)
                
                # Check if diversity requirement is met
                if final_diversity >= min_diversity and len(valid_solutions) >= min_solutions:
                    mining_time = time.time() - start_time

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
                    print(f"[METAL] ✅ BLOCK FOUND! Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")
                    self.logger.info(f"Found valid block! Nonce: {nonce}, Energy: {min_energy:.2f}, Time: {mining_time:.2f}s")

                    # Log mining attempt results
                    self.logger.info(f"Mining attempt completed - Best Energy: {result.energy:.2f}, Valid Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Params: {json.dumps(params)}")

                    # Update top results
                    self.update_top_results(result, requirements)

                    return result
                else:
                    print(f"[METAL]   ❌ Insufficient diversity: {final_diversity:.3f} < {min_diversity:.3f}")
            else:
                print(f"[METAL]   → Not enough samples below threshold (need {min_solutions}, got {num_below_threshold})")

            progress += 1

            # Progress update
            if progress % 10 == 0 and len(sampleset.record.energy) > 0:
                min_energy = float(np.min(sampleset.record.energy))
                self.logger.debug(f"Progress: {progress}, Best energy: {min_energy:.2f}, Valid: {len(valid_indices)}")

        # If we exit the loop due to stop event or completion
        total_time = time.time() - start_time
        if stop_event.is_set():
            print(f"[METAL] ⏹️ Mining stopped after {progress} attempts ({total_time:.1f}s)")
            if self.top_results:
                best_result = self.top_results[0]
                self.logger.info(f"Stopping mining, best result was - Energy: {best_result.energy:.2f}, Valid Solutions: {best_result.num_valid}, Diversity: {best_result.diversity:.3f}")
            else:
                self.logger.info("Stopping mining, no results found")
        else:
            print(f"[METAL] ⏰ Mining completed: {progress} attempts, {total_time:.1f}s, no valid block found")
        return None
    

def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Supports either a NextBlockRequirements object or a dict with keys:
    'difficulty_energy', 'min_diversity', 'min_solutions'.
    """
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # Metal MPS parameters - fast convergence for multiple nonce strategy
    # Strategy: Quick convergence to local minimum (~-13800), then try multiple nonces
    base_sweeps = 80   # Fast convergence - we'll do multiple attempts
    num_sweeps = int(base_sweeps * min(2.0, difficulty_factor ** 0.3))  # Light scaling
    num_reads = max(int(min_solutions), 32)  # Modest reads per attempt

    return {
        'num_sweeps': max(80, min(num_sweeps, 150)),    # Fast convergence range
        'num_reads': max(32, min(num_reads, 64)),       # Efficient reads per attempt
    }