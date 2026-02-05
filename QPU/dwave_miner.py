"""QPU miner using D-Wave sampler with streaming queue for high-throughput mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, cast, Mapping, Tuple, Any, Dict, Union

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper, EmbeddedFuture
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology


@dataclass
class PendingJob:
    """Tracks a pending QPU job with its associated metadata."""
    future: Any  # Future or EmbeddedFuture - both have .done(), .cancel(), .sampleset
    nonce: int
    salt: bytes
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    submit_time: float


class DWaveMiner(BaseMiner):
    def __init__(
        self,
        miner_id: str,
        topology: DWaveTopology = DEFAULT_TOPOLOGY,
        embedding_file: Optional[str] = None,
        time_config: Optional[QPUTimeConfig] = None,
        queue_depth: int = 10,
        **cfg
    ):
        """
        Initialize D-Wave QPU miner with streaming queue support.

        Args:
            miner_id: Unique identifier for this miner
            topology: Topology object (default: DEFAULT_TOPOLOGY = Z(9,2)).
                     Can be any DWaveTopology (Zephyr, Advantage2, etc.)
            embedding_file: Optional path to embedding file. If None and topology requires
                          embedding, will auto-discover precomputed embedding.
            time_config: Optional QPUTimeConfig for time budget management. If provided,
                        the miner will track QPU time usage and skip mining when the
                        budget is exhausted (accounting for reserve time).
            queue_depth: Number of problems to keep in-flight in the QPU queue (default: 10).
                        Higher values increase throughput but may waste QPU time if
                        early results are valid.
            **cfg: Additional configuration options (reserved for future use)
        """
        # Create sampler (encapsulates embedding internally)
        # job_label_prefix will be auto-generated as "Quip_Z{m}_T{t}" by DWaveSamplerWrapper
        # Note: We can't use self.logger yet since super().__init__ hasn't been called
        init_logger.info(f"[QPU] Initializing DWaveMiner with topology: {topology.solver_name}")
        try:
            sampler = DWaveSamplerWrapper(
                topology=topology,
                embedding_file=embedding_file
            )
            init_logger.info(f"[QPU] Sampler ready: {len(sampler.nodes)} nodes, {len(sampler.edges)} edges")
        except Exception as e:
            init_logger.error(f"[QPU] Failed to initialize sampler: {e}")
            raise
        super().__init__(miner_id, sampler, miner_type="QPU")
        self.miner_type = "QPU"
        self.topology = topology

        # QPU time budget management
        self.time_manager: Optional[QPUTimeManager] = None
        if time_config is not None:
            self.time_manager = QPUTimeManager(time_config)
            self.logger.info(
                f"[QPU] Daily budget enabled: {time_config.daily_budget_seconds:.1f}s/day"
            )
        else:
            self.logger.info("[QPU] Daily budget management disabled - no budget configured")

        # Streaming queue configuration
        self.queue_depth = queue_depth  # Number of problems to keep in-flight
        self.pending_futures: Dict[Future, PendingJob] = {}  # Track pending jobs for cleanup

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of QPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"QPU miner {self.miner_id} received SIGTERM, cleaning up D-Wave resources...")

        # QPU-specific cleanup
        try:
            # Cancel all pending streaming futures first
            if hasattr(self, 'pending_futures') and self.pending_futures:
                cancelled_count = 0
                for future in list(self.pending_futures.keys()):
                    try:
                        future.cancel()
                        cancelled_count += 1
                    except Exception:
                        pass  # Best effort cancellation
                self.pending_futures.clear()
                if hasattr(self, 'logger'):
                    self.logger.info(f"Cancelled {cancelled_count} pending QPU jobs")

            # Cancel any running D-Wave jobs via sampler
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cancel_jobs'):
                self.sampler.cancel_jobs()
                if hasattr(self, 'logger'):
                    self.logger.info("D-Wave jobs cancelled via sampler")

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
        
    def _generate_and_submit_job(
        self,
        prev_block,
        node_info,
        cur_index: int,
        params: dict,
        nodes: list,
        edges: list,
    ) -> PendingJob:
        """Generate an Ising problem and submit it to the QPU asynchronously.

        Args:
            prev_block: Previous block for nonce generation
            node_info: Node info containing miner_id
            cur_index: Current block index
            params: Adaptive parameters (num_reads, annealing_time)
            nodes: Topology nodes
            edges: Topology edges

        Returns:
            PendingJob with future and metadata
        """
        # Generate random salt
        salt = random.randbytes(32)

        # Generate quantum model using deterministic block-based seeding
        nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

        # Cast h and J to match protocol expectations
        h_cast = cast(Mapping[Any, float], h)
        J_cast = cast(Mapping[Tuple[Any, Any], float], J)

        # Generate job label with topology and nonce
        topology_label = self.sampler.job_label  # e.g., "Quip_Z9_T2"
        nonce_hex = hex(nonce)[2:][:8]  # First 8 hex chars of nonce
        job_label = f"{topology_label}_{nonce_hex}"

        # Submit asynchronously (non-blocking)
        future = self.sampler.sample_ising_async(
            h_cast, J_cast,
            num_reads=params.get('num_reads', 100),
            answer_mode='raw',
            annealing_time=params.get('annealing_time', 20.0),
            label=job_label
        )

        return PendingJob(
            future=future,
            nonce=nonce,
            salt=salt,
            h=h,
            J=J,
            submit_time=time.time()
        )

    def _cancel_all_pending(self) -> Tuple[int, float]:
        """Cancel all pending futures and estimate their QPU time.

        Jobs that were submitted to the QPU have already consumed QPU time even
        if cancelled. This method attempts to get actual timing from completed
        jobs, or uses EMA estimates for pending jobs.

        Returns:
            Tuple of (cancelled_count, estimated_qpu_time_us)
        """
        cancelled = 0
        total_estimated_us = 0.0

        for future, job in list(self.pending_futures.items()):
            try:
                # Try to get actual time if job already completed
                if future.done():
                    try:
                        sampleset = future.sampleset
                        if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                            timing = sampleset.info['timing']
                            qpu_time = timing.get('qpu_programming_time', 0) + timing.get('qpu_sampling_time', 0)
                            total_estimated_us += qpu_time
                        elif self.time_manager:
                            # Use EMA estimate for completed job without timing
                            total_estimated_us += self.time_manager.estimate_next_block_time()
                    except Exception:
                        # Fallback to estimate
                        if self.time_manager:
                            total_estimated_us += self.time_manager.estimate_next_block_time()
                else:
                    # Job still pending - use EMA estimate
                    if self.time_manager:
                        total_estimated_us += self.time_manager.estimate_next_block_time()

                future.cancel()
                cancelled += 1
            except Exception:
                pass  # Best effort cancellation

        self.pending_futures.clear()
        return cancelled, total_estimated_us

    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using D-Wave QPU with streaming queue.

        Uses a rolling window pattern: maintains queue_depth problems in-flight,
        processes results as they stream back, and refills the queue on each result.
        Cancels all pending jobs immediately when a valid result is found.

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
        self.logger.info(f"Mining block {cur_index} with streaming queue (depth={self.queue_depth})...")

        # Apply difficulty decay based on elapsed time since previous block
        current_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)

        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        params = adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges)
        )
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        # Check QPU daily budget before starting
        if self.time_manager is not None:
            estimate = self.time_manager.should_mine_block()
            if not estimate.should_mine:
                # Pacing - wait for proportional limit to catch up
                wait_str = f"{estimate.seconds_until_can_mine:.0f}s" if estimate.seconds_until_can_mine < 3600 else f"{estimate.seconds_until_can_mine/3600:.1f}h"
                self.logger.info(
                    f"[QPU] Pacing block {cur_index} - waiting {wait_str} for limit to catch up. "
                    f"Used: {estimate.cumulative_used_us/1e6:.2f}s, "
                    f"Limit: {estimate.proportional_limit_us/1e6:.2f}s "
                    f"({estimate.elapsed_fraction*100:.1f}% of day)"
                )
                return None

            self.logger.info(
                f"[QPU] Budget check passed. Used: {estimate.cumulative_used_us/1e6:.2f}s / "
                f"{estimate.proportional_limit_us/1e6:.2f}s limit ({estimate.elapsed_fraction*100:.1f}% of day), "
                f"Estimated: {estimate.estimated_block_time_us/1e6:.2f}s ({estimate.confidence} confidence)"
            )

        # Mark start of streaming attempt
        self.current_stage = 'sampling'
        self.current_stage_start = time.time()

        # Clear any stale pending futures
        self.pending_futures.clear()

        # Track if we've paused submissions due to budget
        budget_paused = False

        # Initial queue fill
        self.logger.info(f"[QPU] Filling initial queue with {self.queue_depth} problems...")
        try:
            for i in range(self.queue_depth):
                if stop_event.is_set():
                    cancelled, cancelled_time_us = self._cancel_all_pending()
                    if self.time_manager and cancelled_time_us > 0:
                        self.time_manager.record_block_time(cancelled_time_us)
                    return None

                # Check budget before each submission
                if self.time_manager is not None:
                    estimate = self.time_manager.should_mine_block()
                    if not estimate.should_mine:
                        budget_paused = True
                        self.logger.info(
                            f"[QPU] Budget limit reached during queue fill at job {i+1}/{self.queue_depth}. "
                            f"Used: {estimate.cumulative_used_us/1e6:.2f}s / {estimate.proportional_limit_us/1e6:.2f}s limit. "
                            f"Continuing with {len(self.pending_futures)} pending jobs."
                        )
                        break

                job = self._generate_and_submit_job(prev_block, node_info, cur_index, params, nodes, edges)
                self.pending_futures[job.future] = job
                self.logger.debug(f"[QPU] Submitted job {i+1}/{self.queue_depth}")
        except Exception as e:
            self.logger.error(f"Error filling initial queue: {e}")
            cancelled, cancelled_time_us = self._cancel_all_pending()
            if self.time_manager and cancelled_time_us > 0:
                self.time_manager.record_block_time(cancelled_time_us)
            return None

        self.logger.info(f"[QPU] Queue filled, streaming results...")

        # Streaming result loop
        while not stop_event.is_set() and self.pending_futures:
            # Check for difficulty decay and update requirements
            updated_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
            if current_requirements != updated_requirements:
                current_requirements = updated_requirements
                params = adapt_parameters(
                    current_requirements.difficulty_energy,
                    current_requirements.min_diversity,
                    current_requirements.min_solutions,
                    num_nodes=len(nodes),
                    num_edges=len(edges)
                )
                self.logger.info(f"{self.miner_id} - Updated adaptive params due to difficulty decay: {params}")

                # Check if any existing top attempts now meet requirements
                for sample in self.top_attempts:
                    if min(sample.sampleset.record.energy) <= current_requirements.difficulty_energy:
                        result = self.evaluate_sampleset(
                            sample.sampleset, current_requirements, nodes, edges,
                            sample.nonce, sample.salt, prev_timestamp, start_time
                        )
                        if result:
                            self.logger.info(f"[Block-{cur_index}] Previous result now meets decayed difficulty!")
                            cancelled, cancelled_time_us = self._cancel_all_pending()
                            if self.time_manager and cancelled_time_us > 0:
                                self.time_manager.record_block_time(cancelled_time_us)
                                self.logger.info(f"Cancelled {cancelled} pending jobs, recorded ~{cancelled_time_us/1e6:.2f}s estimated QPU time")
                            else:
                                self.logger.info(f"Cancelled {cancelled} pending jobs")
                            return result

            # Poll for completed futures (works with both Future and EmbeddedFuture)
            completed_future = None
            poll_start = time.time()

            # Poll for up to 1 second to allow stop_event checking
            while time.time() - poll_start < 1.0:
                for future in list(self.pending_futures.keys()):
                    if future.done():
                        completed_future = future
                        break
                if completed_future is not None:
                    break
                time.sleep(0.05)  # 50ms polling interval

            if completed_future is None:
                # No results ready yet, continue to check stop_event
                continue

            job = self.pending_futures.pop(completed_future)
            progress += 1

            # Track postprocessing time
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start

            try:
                # Get the sampleset (should be ready since done() returned True)
                sampleset = completed_future.sampleset

                # Update timing stats
                sample_time = time.time() - job.submit_time
                self.timing_stats['sampling'].append(sample_time * 1e6)

                # Extract QPU timing information if available
                if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                    timing = sampleset.info['timing']
                    if 'qpu_anneal_time_per_sample' in timing:
                        self.timing_stats['quantum_annealing_time'].append(
                            timing['qpu_anneal_time_per_sample']
                        )
                    qpu_programming = timing.get('qpu_programming_time', 0)
                    qpu_sampling = timing.get('qpu_sampling_time', 0)
                    qpu_total_access = qpu_programming + qpu_sampling
                    if qpu_total_access > 0:
                        self.timing_stats['qpu_access_time'].append(qpu_total_access)
                        # Record time for budget tracking
                        if self.time_manager is not None:
                            self.time_manager.record_block_time(qpu_total_access)

                # Update sample counts
                all_energies = sampleset.record.energy
                self.timing_stats['total_samples'] += len(all_energies)
                self.timing_stats['blocks_attempted'] += 1

                # Evaluate the result
                result = self.evaluate_sampleset(
                    sampleset, current_requirements, nodes, edges,
                    job.nonce, job.salt, prev_timestamp, start_time
                )

                self.logger.debug(f"QPU sampleset evaluated in {time.time() - postprocess_start:.2f}s")
                self.timing_stats['postprocessing'].append((time.time() - postprocess_start) * 1e6)

                if result:
                    # Valid result! Cancel all pending and return
                    self.logger.info(
                        f"[Block-{cur_index}] Mined! Nonce: {job.nonce}, "
                        f"Salt: {job.salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, "
                        f"Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, "
                        f"Attempts: {progress}, Total Mining Time: {time.time() - start_time:.2f}s"
                    )
                    cancelled, cancelled_time_us = self._cancel_all_pending()
                    if self.time_manager and cancelled_time_us > 0:
                        self.time_manager.record_block_time(cancelled_time_us)
                        self.logger.info(f"Cancelled {cancelled} pending jobs, recorded ~{cancelled_time_us/1e6:.2f}s estimated QPU time")
                    else:
                        self.logger.info(f"Cancelled {cancelled} pending jobs")
                    return result

                # Invalid result - track and refill queue (rolling window)
                self.update_top_samples(sampleset, job.nonce, job.salt, current_requirements)

                # Log progress
                if self.top_attempts:
                    best_energy = min(self.top_attempts[0].sampleset.record.energy)
                    self.logger.info(
                        f"Progress: {progress} results, {len(self.pending_futures)} pending, "
                        f"best energy: {best_energy:.2f}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing result: {e}")

            # Refill queue: submit new job to maintain queue depth (if budget allows)
            if not stop_event.is_set() and not budget_paused:
                # Check budget before submitting replacement job
                if self.time_manager is not None:
                    estimate = self.time_manager.should_mine_block()
                    if not estimate.should_mine:
                        budget_paused = True
                        self.logger.info(
                            f"[QPU] Budget limit reached. Pausing new submissions. "
                            f"Used: {estimate.cumulative_used_us/1e6:.2f}s / {estimate.proportional_limit_us/1e6:.2f}s limit. "
                            f"Waiting for {len(self.pending_futures)} pending jobs and difficulty decay."
                        )
                    else:
                        try:
                            new_job = self._generate_and_submit_job(
                                prev_block, node_info, cur_index, params, nodes, edges
                            )
                            self.pending_futures[new_job.future] = new_job
                        except Exception as e:
                            self.logger.error(f"Error submitting replacement job: {e}")
                else:
                    # No time manager - always submit
                    try:
                        new_job = self._generate_and_submit_job(
                            prev_block, node_info, cur_index, params, nodes, edges
                        )
                        self.pending_futures[new_job.future] = new_job
                    except Exception as e:
                        self.logger.error(f"Error submitting replacement job: {e}")

        # Cleanup on exit
        cancelled, cancelled_time_us = self._cancel_all_pending()
        if cancelled > 0:
            if self.time_manager and cancelled_time_us > 0:
                self.time_manager.record_block_time(cancelled_time_us)
                self.logger.info(f"Cancelled {cancelled} pending jobs on exit, recorded ~{cancelled_time_us/1e6:.2f}s estimated QPU time")
            else:
                self.logger.info(f"Cancelled {cancelled} pending jobs on exit")

        self.logger.info("Stopping mining, no valid results found")
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
    max_annealing_time = 20.0   # Hardest difficulty

    # Linear interpolation for annealing time
    annealing_time = min_annealing_time + difficulty * (max_annealing_time - min_annealing_time)

    # QPU read bonus range (added to min_solutions)
    min_bonus = 32    # Easiest difficulty
    max_bonus = 64    # Hardest difficulty

    # Linear interpolation for bonus reads
    bonus_reads = int(min_bonus + difficulty * (max_bonus - min_bonus))
    num_reads = min_solutions + bonus_reads

    return {
        'num_reads': num_reads,
        'annealing_time': annealing_time,
    }