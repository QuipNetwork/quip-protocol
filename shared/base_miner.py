"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import multiprocessing
import multiprocessing.synchronize

import dimod
from shared.block_requirements import BlockRequirements
from shared.miner_types import IsingSample, MiningResult, Sampler

# Global logger for this module (set during initialization)
log = None
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from shared.quantum_proof_of_work import evaluate_sampleset
# Global logger for this module
log = logging.getLogger(__name__)

class BaseMiner(ABC):
    """Abstract base class for concrete miners.

    Subclasses must implement:
      - mine_block(): miner-specific mining logic
      - set self.miner_type and self.sampler in __init__
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

    @abstractmethod
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Abstract method for miner-specific mining implementation.

        Args:
            prev_block: Previous block object containing header, data, and other block information
            node_info: Node information containing miner_id and other details
            requirements: BlockRequirements object with difficulty settings
            prev_timestamp: Timestamp from the previous block header
            stop_event: Multiprocessing event to signal stop

        Returns:
            MiningResult if successful, None if stopped or failed
        """
        pass

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

