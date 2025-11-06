"""QPU miner using D-Wave sampler or mock via create_dwave_sampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Optional, cast, Mapping, Tuple, Any

from QPU.dwave_sampler import DWaveSamplerWrapper
from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES

class DWaveMiner(BaseMiner):
    def __init__(self, miner_id: str, topology_name: Optional[str] = "Z(9,2)", qpu_timeout: float = 360.0, **cfg):
        """
        Initialize D-Wave QPU miner.

        Args:
            miner_id: Unique identifier for this miner
            topology_name: Topology to embed on QPU (default: "Z(9,2)" with precomputed embedding).
                          The QPU will use FixedEmbeddingComposite to map logical variables
                          onto the physical Advantage2 hardware topology.
            qpu_timeout: Minimum seconds between QPU attempts (default: 360.0). Set to 0.0 to disable rate limiting.
            **cfg: Additional configuration options (reserved for future use)
        """
        # Create sampler with embedding (default Z(9,2) matches DEFAULT_TOPOLOGY variable count)
        sampler = DWaveSamplerWrapper(topology_name=topology_name, job_label_prefix="QUIP_MINE")
        super().__init__(miner_id, sampler, miner_type="QPU")
        self.miner_type = "QPU"
        self.topology_name = topology_name

        # QPU rate limiting configuration
        self.qpu_timeout = qpu_timeout  # Minimum seconds between QPU attempts (0.0 = disabled)
        self.last_qpu_attempt_time = 0.0  # Track last QPU sampling time
        
        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of QPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"QPU miner {self.miner_id} received SIGTERM, cleaning up D-Wave resources...")
        
        # QPU-specific cleanup
        try:
            # Cancel any running D-Wave jobs
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cancel_jobs'):
                self.sampler.cancel_jobs()
                if hasattr(self, 'logger'):
                    self.logger.info("D-Wave jobs cancelled")
            
            # Close D-Wave connections
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'close'):
                self.sampler.close()
                if hasattr(self, 'logger'):
                    self.logger.info("D-Wave connections closed")
            
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during QPU miner cleanup: {e}")
        
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
                            self.logger.info(f"[Block-{cur_index}] Already Mined at this difficulty! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Attempt Time: {result.mining_time:.2f}s, Total Mining Time: {time.time() - start_time:.2f}s")
                            return result

            # Track preprocessing time
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start
            
            # Sample from QPU
            try:
                # QPU rate limiting: wait if we've run too recently
                current_time = time.time()
                time_since_last_attempt = current_time - self.last_qpu_attempt_time

                if time_since_last_attempt < self.qpu_timeout:
                    wait_time = self.qpu_timeout - time_since_last_attempt
                    self.logger.info(f"[QPU] Rate limiting: waiting {wait_time:.1f}s before next attempt (timeout={self.qpu_timeout}s)")

                    # Check stop event during wait to allow early termination
                    while wait_time > 0 and not stop_event.is_set():
                        sleep_duration = min(1.0, wait_time)  # Sleep in 1-second chunks
                        time.sleep(sleep_duration)
                        wait_time -= sleep_duration

                    # If stop event was set during wait, exit
                    if stop_event.is_set():
                        self.logger.info("Stop event received during QPU rate limiting wait")
                        break

                # For QPU, use adaptive parameters
                num_reads = params.get('num_reads', 100)
                annealing_time = params.get('quantum_annealing_time', 20.0)

                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start

                # Update last attempt time before sampling
                self.last_qpu_attempt_time = sample_start
                
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


def adapt_parameters(
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    num_nodes: int = DEFAULT_NUM_NODES,
    num_edges: int = DEFAULT_NUM_EDGES
):
    """Calculate adaptive mining parameters based on difficulty requirements.

    QPU strategy: Scales annealing time and read bonus (not sweeps).
    Linear interpolation for both parameters.

    Note: num_reads = min_solutions + bonus, where bonus ∈ [16, 64]

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum solution diversity required (reserved)
        min_solutions: Minimum number of valid solutions required
        num_nodes: Number of nodes in topology (default: DEFAULT_TOPOLOGY)
        num_edges: Number of edges in topology (default: DEFAULT_TOPOLOGY)

    Returns:
        Dictionary with num_reads and annealing_time parameters
    """
    # Get normalized difficulty [0, 1]
    difficulty = energy_to_difficulty(
        difficulty_energy,
        num_nodes=num_nodes,
        num_edges=num_edges
    )

    # QPU annealing time range (microseconds)
    min_annealing_time = 5.0    # Easiest difficulty
    max_annealing_time = 10.0   # Hardest difficulty

    # Linear interpolation for annealing time
    annealing_time = min_annealing_time + difficulty * (max_annealing_time - min_annealing_time)

    # QPU read bonus range (added to min_solutions)
    min_bonus = 16    # Easiest difficulty
    max_bonus = 64    # Hardest difficulty

    # Linear interpolation for bonus reads
    bonus_reads = int(min_bonus + difficulty * (max_bonus - min_bonus))
    num_reads = min_solutions + bonus_reads

    return {
        'num_reads': num_reads,
        'annealing_time': annealing_time,
    }