"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import multiprocessing
import multiprocessing.synchronize
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import dimod
import numpy as np

from shared.block_requirements import BlockRequirements, compute_current_requirements
from shared.miner_types import IsingSample, MiningResult, Sampler
from shared.quantum_proof_of_work import (
    evaluate_sampleset,
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)

# Global logger for this module
log = logging.getLogger(__name__)

class BaseMiner(ABC):
    """Abstract base class for concrete miners (Template Method pattern).

    Subclasses must implement:
      - _sample(h, J, **kwargs): backend-specific Ising sampling
      - _adapt_mining_params(requirements, nodes, edges): return parameter dict

    Subclasses may optionally override:
      - _pre_mine_setup(...): one-time setup before the mining loop
      - _post_sample(sampleset): post-process a SampleSet (e.g. sparse filtering)
      - _post_mine_cleanup(): cleanup after the mining loop exits
      - _on_sampling_error(error, stop_event): handle sampling exceptions
    """

    def __init__(
        self,
        miner_id: str,
        sampler: Sampler,
        miner_type: str = "UNKNOWN"
    ) -> None:
        if type(self) is BaseMiner:
            raise TypeError("BaseMiner is abstract; instantiate a concrete subclass")
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.mining = False
        self.blocks_won = 0
        self.total_rewards = 0
        self.sampler = sampler

        # Initialize logger that inherits parent process configuration
        self.logger = logging.getLogger(f'miner.{miner_id}')

        self.logger.debug(f"{miner_id} initialized ({self.miner_type})")

        # Initialize timing statistics
        self.timing_stats = {
            'preprocessing': [],
            'sampling': [],
            'postprocessing': [],
            'quantum_annealing_time': [],
            'per_sample_overhead': [],
            'qpu_access_time': [],  # Total QPU time (programming + sampling) in microseconds
            'total_samples': 0,
            'blocks_attempted': 0
        }

        # Track timing history for graphing (block_number, timing_value)
        self.timing_history = {
            'block_numbers': [],
            'preprocessing_times': [],
            'sampling_times': [],
            'postprocessing_times': [],
            'total_times': [],
            'win_rates': [],
            'adaptive_params_history': []  # Track adaptive params over time
        }

        # Track participation in current round
        self.current_round_attempted = False

        # Track current stage timing
        self.current_stage: Optional[str] = None
        self.current_stage_start: Optional[float] = None

        # Adaptive parameters for performance tuning
        # Initialize num_sweeps based on miner ID for SA miners
        initial_sweeps = 512
        if self.miner_id and self.miner_id[-1].isdigit():
            initial_sweeps = pow(2, 6 + int(self.miner_id[-1]))

        self.adaptive_params = {
            'quantum_annealing_time': 20.0,  # microseconds for QPU
            'beta_range': [0.1, 10.0],  # for SA
            'beta_schedule': 'geometric',  # or 'linear'
            'num_sweeps': initial_sweeps  # for SA
        }

        # Track top 3 mining results
        self.top_attempts: List[IsingSample] = []


    def update_top_samples(self, sampleset: dimod.SampleSet, nonce: int, salt: bytes, requirements: BlockRequirements):
        """Update the top 3 results list with a new mining result."""

        # Add current result
        attempt = IsingSample(nonce, salt, sampleset)
        self.top_attempts.append(attempt)
        self.top_attempts.sort(key=lambda r: compare_mining_samples(r, attempt, requirements))

        # Keep only top 3
        self.top_attempts = self.top_attempts[:3]

    def capture_partial_timing(self):
        """Capture timing for current mining attempt, including partial progress."""
        current_time = time.time()

        # Initialize with zeros
        preprocessing_time = 0
        sampling_time = 0
        postprocessing_time = 0

        # If we have completed preprocessing
        if len(self.timing_stats['preprocessing']) > len(self.timing_stats['sampling']):
            # Preprocessing was completed
            preprocessing_time = self.timing_stats['preprocessing'][-1]

            # Check if sampling was started
            if self.current_stage == 'sampling' and self.current_stage_start:
                # Sampling was in progress
                sampling_time = (current_time - self.current_stage_start) * 1e6
                postprocessing_time = 0  # Not started
            elif self.current_stage == 'postprocessing' and self.current_stage_start:
                # Sampling was completed, postprocessing in progress
                if self.timing_stats['sampling']:
                    sampling_time = self.timing_stats['sampling'][-1]
                postprocessing_time = (current_time - self.current_stage_start) * 1e6
        elif self.current_stage == 'preprocessing' and self.current_stage_start:
            # Still in preprocessing
            preprocessing_time = (current_time - self.current_stage_start) * 1e6
            sampling_time = 0
            postprocessing_time = 0

        return preprocessing_time, sampling_time, postprocessing_time

    def get_timing_summary(self) -> str:
        """Generate a summary of timing statistics for this miner."""
        summary_lines = [f"\nTiming Statistics for {self.miner_id}:"]

        if self.timing_stats['blocks_attempted'] > 0:
            summary_lines.append(f"  Blocks Attempted: {self.timing_stats['blocks_attempted']}")
            summary_lines.append(f"  Total Samples: {self.timing_stats['total_samples']}")
            summary_lines.append(f"  Blocks Won: {self.blocks_won}")
            summary_lines.append(f"  Win Rate: {self.blocks_won / self.timing_stats['blocks_attempted'] * 100:.1f}%")

        # Calculate averages for each timing component
        for component in ['preprocessing', 'sampling', 'postprocessing']:
            if self.timing_stats[component]:
                avg_time = np.mean(self.timing_stats[component])
                std_time = np.std(self.timing_stats[component])
                summary_lines.append(f"  {component.capitalize()} Time: {avg_time:.2f} ± {std_time:.2f} μs")

        # QPU-specific timing
        if self.timing_stats['quantum_annealing_time']:
            avg_anneal = np.mean(self.timing_stats['quantum_annealing_time'])
            summary_lines.append(f"  Quantum Annealing Time: {avg_anneal:.2f} μs")

        # Show adaptive parameters
        if self.miner_type == "QPU":
            summary_lines.append(f"  Current Annealing Time: {self.adaptive_params['quantum_annealing_time']:.2f} μs")
        else:
            summary_lines.append(f"  Current Num Sweeps: {self.adaptive_params['num_sweeps']}")
            summary_lines.append(f"  Beta Range: {self.adaptive_params['beta_range']}")
            summary_lines.append(f"  Beta Schedule: {self.adaptive_params['beta_schedule']}")

        return "\n".join(summary_lines)

    def adapt_parameters(self, network_stats: dict):
        """Adapt miner parameters based on performance relative to network.

        Args:
            network_stats: Dict containing total_blocks, total_miners, avg_win_rate
        """
        if self.timing_stats['blocks_attempted'] < 5:
            return  # Need enough data before adapting

        # Calculate expected win rate (fair share)
        expected_win_rate = 1.0 / network_stats['total_miners']
        actual_win_rate = self.blocks_won / self.timing_stats['blocks_attempted']

        # If winning less than expected, improve parameters
        if actual_win_rate < expected_win_rate * 0.8:  # 20% below expected
            if self.miner_type == "QPU":
                # Increase annealing time for better solutions
                self.adaptive_params['quantum_annealing_time'] *= 1.2
                self.logger.info(f"{self.miner_id} increasing annealing time to {self.adaptive_params['quantum_annealing_time']:.2f} μs")
            else:
                # For SA, increase sweeps or adjust beta range
                self.adaptive_params['num_sweeps'] = int(self.adaptive_params['num_sweeps'] * 1.1)
                # Widen beta range for better exploration
                self.adaptive_params['beta_range'][0] *= 0.9
                self.adaptive_params['beta_range'][1] *= 1.1
                self.logger.info(f"{self.miner_id} adapting: sweeps={self.adaptive_params['num_sweeps']}, beta_range={self.adaptive_params['beta_range']}")

        # If winning too much, can reduce parameters to save resources
        elif actual_win_rate > expected_win_rate * 1.5:  # 50% above expected
            if self.miner_type == "QPU":
                # Reduce annealing time to save QPU resources
                self.adaptive_params['quantum_annealing_time'] *= 0.9
                self.logger.info(f"{self.miner_id} reducing annealing time to {self.adaptive_params['quantum_annealing_time']:.2f} μs")
            else:
                # For SA, reduce sweeps for faster mining
                self.adaptive_params['num_sweeps'] = int(self.adaptive_params['num_sweeps'] * 0.95)
                self.logger.info(f"{self.miner_id} reducing sweeps to {self.adaptive_params['num_sweeps']}")

    # ------------------------------------------------------------------
    # Template Method: mine_block
    # ------------------------------------------------------------------

    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> Optional[MiningResult]:
        """Mine a block using the common mining loop skeleton.

        The loop generates (salt, nonce, h, J) each iteration, delegates
        sampling to ``_sample()``, evaluates the result, and tracks
        progress.  Subclasses customise behaviour via the hook methods
        listed in the class docstring.

        Args:
            prev_block: Previous block object
            node_info: Node information containing miner_id
            requirements: BlockRequirements with difficulty settings
            prev_timestamp: Timestamp from the previous block header
            stop_event: Multiprocessing event to signal stop
            **kwargs: Passed through to ``_pre_mine_setup``

        Returns:
            MiningResult if a valid solution is found, else None.
        """
        # -- setup --------------------------------------------------------
        self.mining = True
        progress = 0
        self.top_attempts = []
        start_time = time.time()

        cur_index = prev_block.header.index + 1

        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index}...")

        # One-time miner-specific initialisation
        if not self._pre_mine_setup(
            prev_block, node_info, requirements,
            prev_timestamp, stop_event, **kwargs,
        ):
            return None

        # Compute initial requirements (with difficulty decay)
        current_requirements = compute_current_requirements(
            requirements, prev_timestamp, self.logger,
        )

        # Topology
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        # Adaptive parameters (per-miner)
        params = self._adapt_mining_params(
            current_requirements, nodes, edges,
        )
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        # Sweep-increment tracking
        current_num_sweeps = params.get('num_sweeps', 64)
        num_reads = params.get('num_reads', 100)
        max_num_sweeps = current_num_sweeps
        increment_interval = 30.0
        last_increment_time = start_time

        # -- main mining loop ---------------------------------------------
        while self.mining and not stop_event.is_set():
            # Gradually increase sweeps over time
            current_time = time.time()
            if current_time - last_increment_time >= increment_interval:
                current_num_sweeps = min(
                    max_num_sweeps, int(current_num_sweeps * 1.05),
                )
                last_increment_time = current_time

            # Generate salt, nonce, Ising model
            salt = random.randbytes(32)
            nonce = ising_nonce_from_block(
                prev_block.hash, node_info.miner_id, cur_index, salt,
            )
            h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

            # Re-check requirements (difficulty decay)
            updated_requirements = compute_current_requirements(
                requirements, prev_timestamp, self.logger,
            )
            if current_requirements != updated_requirements:
                current_requirements = updated_requirements
                params = self._adapt_mining_params(
                    current_requirements, nodes, edges,
                )
                self.logger.info(
                    f"{self.miner_id} - updated adaptive params: {params}",
                )

                # Check if any cached top attempts now satisfy requirements
                for sample in self.top_attempts:
                    best_e = min(sample.sampleset.record.energy)
                    if best_e <= current_requirements.difficulty_energy:
                        result = self.evaluate_sampleset(
                            sample.sampleset, current_requirements,
                            nodes, edges, sample.nonce, sample.salt,
                            prev_timestamp, start_time,
                        )
                        if result:
                            self.logger.info(
                                f"[Block-{cur_index}] Already Mined at "
                                f"this difficulty! Nonce: {sample.nonce}, "
                                f"Salt: {sample.salt.hex()[:4]}..., "
                                f"Min Energy: {result.energy:.2f}, "
                                f"Solutions: {result.num_valid}, "
                                f"Diversity: {result.diversity:.3f}, "
                                f"Attempt Time: {result.mining_time:.2f}s, "
                                f"Total Mining Time: "
                                f"{time.time() - start_time:.2f}s",
                            )
                            self._post_mine_cleanup()
                            return result

            # Track preprocessing
            preprocess_start = time.time()
            self.current_stage = 'preprocessing'
            self.current_stage_start = preprocess_start

            # ---------- SAMPLING (delegated to subclass) ----------
            try:
                sample_start = time.time()
                self.current_stage = 'sampling'
                self.current_stage_start = sample_start

                sampleset = self._sample(
                    h, J,
                    num_reads=num_reads,
                    num_sweeps=current_num_sweeps,
                    **{k: v for k, v in params.items()
                       if k not in ('num_reads', 'num_sweeps')},
                )

                sample_time = time.time() - sample_start
                self.timing_stats['sampling'].append(sample_time * 1e6)
                self.timing_stats['preprocessing'].append(
                    (sample_start - preprocess_start) * 1e6,
                )
            except Exception as e:
                if self._on_sampling_error(e, stop_event):
                    return None
                continue

            # Post-sample hook (e.g. sparse-topology filter)
            sampleset = self._post_sample(sampleset)

            if stop_event.is_set():
                self._post_mine_cleanup()
                return None

            # ---------- EVALUATE ----------
            postprocess_start = time.time()
            self.current_stage = 'postprocessing'
            self.current_stage_start = postprocess_start

            self.timing_stats['total_samples'] += len(
                sampleset.record.energy,
            )
            self.timing_stats['blocks_attempted'] += 1

            result = self.evaluate_sampleset(
                sampleset, current_requirements, nodes, edges,
                nonce, salt, prev_timestamp, start_time,
            )

            self.timing_stats['postprocessing'].append(
                (time.time() - postprocess_start) * 1e6,
            )

            if result:
                self.logger.info(
                    f"[Block-{cur_index}] Mined! "
                    f"Nonce: {nonce}, Salt: {salt.hex()[:4]}..., "
                    f"Min Energy: {result.energy:.2f}, "
                    f"Solutions: {result.num_valid}, "
                    f"Diversity: {result.diversity:.3f}, "
                    f"Attempt Time: {result.mining_time:.2f}s, "
                    f"Total Mining Time: {time.time() - start_time:.2f}s",
                )
                self._post_mine_cleanup()
                return result

            # Track best attempts
            self.update_top_samples(
                sampleset, nonce, salt, current_requirements,
            )

            progress += 1
            if progress % 10 == 0:
                best_energy = (
                    min(self.top_attempts[0].sampleset.record.energy)
                    if self.top_attempts
                    else float('inf')
                )
                self.logger.info(
                    f"Progress: {progress} attempts, "
                    f"best energy: {best_energy:.2f} | "
                    f"Sweeps: {current_num_sweeps}/{max_num_sweeps}, "
                    f"Reads: {num_reads}",
                )

        # -- teardown -----------------------------------------------------
        self._post_mine_cleanup()
        self.logger.info("Stopping mining, no results found")
        return None

    # ------------------------------------------------------------------
    # Hook methods (override in subclasses as needed)
    # ------------------------------------------------------------------

    def _pre_mine_setup(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> bool:
        """Called once before the mining loop starts.

        Return False to abort mining (e.g. QPU budget exhausted).
        """
        return True

    @abstractmethod
    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> dimod.SampleSet:
        """Perform backend-specific Ising sampling.

        Must return a dimod.SampleSet.
        """

    @abstractmethod
    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """Return adaptive mining parameters for the current difficulty.

        The returned dict must include at least 'num_sweeps' and
        'num_reads'.  Extra keys are forwarded to ``_sample()`` as
        keyword arguments.
        """

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Post-process a SampleSet before evaluation.

        Default implementation is the identity function.
        Override in subclasses that need filtering (e.g. sparse topology).
        """
        return sampleset

    def _post_mine_cleanup(self) -> None:
        """Called after the mining loop exits (success or stop)."""

    def _on_sampling_error(
        self,
        error: Exception,
        stop_event: multiprocessing.synchronize.Event,
    ) -> bool:
        """Handle a sampling exception.

        Return True to abort mining, False to skip this iteration and
        continue.
        """
        if stop_event.is_set():
            self.logger.info("Interrupted during sampling")
            return True
        self.logger.error(f"Sampling error: {error}")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return machine-readable stats for this miner."""
        stats = dict(self.timing_stats)
        stats.update({
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
        })
        return stats

    def evaluate_sampleset(self, sampleset: dimod.SampleSet, requirements: BlockRequirements, nodes: List[int], edges: List[Tuple[int, int]], nonce: int, salt: bytes, prev_timestamp: int, start_time: float) -> Optional[MiningResult]:
        """Convert a sample set into a mining result if it meets requirements, otherwise return None."""
        return evaluate_sampleset(sampleset, requirements, nodes, edges, nonce, salt, prev_timestamp, start_time, self.miner_id, self.miner_type)


def compare_mining_samples(sample_a: IsingSample, sample_b: IsingSample, requirements: BlockRequirements) -> int:
    """
    Compare two mining results to determine which is better.

    Returns:
        -1 if A is better than B
         0 if A and B are equal
         1 if B is better than A

    Comparison logic:
    1. Compare average of top N energies
       where N = requirements.min_solutions
    2. If still equal, compare overall average solution energy
    """

    # 1. Compare average of top N solution energies
    a_energies = list(sample_a.sampleset.record.energy)
    b_energies = list(sample_b.sampleset.record.energy)
    n_energies = min(requirements.min_solutions, len(a_energies), len(b_energies))
    if n_energies > 0:
        energies_a = a_energies[:n_energies]
        energies_b = b_energies[:n_energies]
        avg_energy_a = np.mean(energies_a)
        avg_energy_b = np.mean(energies_b)

        if avg_energy_a < avg_energy_b:  # Lower energy is better
            return -1
        elif avg_energy_b < avg_energy_a:
            return 1

    # 2. If still equal, compare overall best energy (lower is better)
    best_energy_a = min(a_energies)
    best_energy_b = min(b_energies)
    if best_energy_a < best_energy_b:
        return -1
    elif best_energy_b < best_energy_a:
        return 1

    return 0  # Equal

