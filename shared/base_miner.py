"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import collections.abc
from blake3 import blake3
import logging
import multiprocessing
import multiprocessing.synchronize
from shared.block_requirements import BlockRequirements
from shared.logging_config import QuipFormatter

# Global logger for this module (set during initialization)
log = None
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Protocol, Any, Union

import numpy as np

# Type definitions for quantum computing
Variable = collections.abc.Hashable
Bias = float

@dataclass
class MiningResult:
    """Result of a mining operation."""
    miner_id: str
    miner_type: str
    nonce: int
    salt: bytes
    timestamp: int
    prev_timestamp: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    num_valid: int
    mining_time: float
    node_list: List[int]
    edge_list: List[Tuple[int, int]]
    variable_order: Optional[List[int]] = None

class Sampler(Protocol):
    """Protocol defining the D-Wave sampler interface."""
    nodelist: List[Variable]
    edgelist: List[Tuple[Variable, Variable]]
    properties: Dict[str, Any]
    sampler_type: str
    nodes: List[int]  # Integer nodes for quantum_proof_of_work functions
    edges: List[Tuple[int, int]]  # Integer edges for quantum_proof_of_work functions

    def sample_ising(
        self,
        h: Union[Mapping[Variable, Bias], Sequence[Bias]],
        J: Mapping[Tuple[Variable, Variable], Bias],
        **kwargs
    ) -> Any:
        ...
from shared.quantum_proof_of_work import (
    calculate_hamming_distance as _shared_hamming,
    calculate_diversity as _shared_diversity,
    filter_diverse_solutions as _shared_filter,
    generate_ising_model_from_nonce,
    energy_of_solution,
)
from shared.logging_config import get_logger, init_component_logger

# Global logger for this module
log = None

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

        # Initialize logger with helper function
        self.logger = init_component_logger('miner', miner_id)

        # Setup multiprocessing logging compatibility
        self._setup_multiprocess_logging()

        self.logger.debug(f"{miner_id} initialized ({self.miner_type})")

        # Initialize timing statistics
        self.timing_stats = {
            'preprocessing': [],
            'sampling': [],
            'postprocessing': [],
            'quantum_annealing_time': [],
            'per_sample_overhead': [],
            'total_samples': 0,
            'blocks_attempted': 0
        }

    def _setup_multiprocess_logging(self):
        """Ensure logger works in multiprocessing context."""
        # Force propagation to root logger
        self.logger.propagate = True

        # If in child process, ensure proper handler setup
        if (hasattr(multiprocessing, 'current_process') and
            multiprocessing.current_process().name != 'MainProcess'):
            self._configure_child_logging()

    def _configure_child_logging(self):
        """Configure logging for child processes."""
        formatter_class = QuipFormatter

        root_logger = logging.getLogger()

        # Check if QuipFormatter is present
        has_quip_formatter = any(
            isinstance(getattr(handler, 'formatter', None), formatter_class)
            for handler in root_logger.handlers
        )

        if not has_quip_formatter:
            # Add QuipFormatter handler
            formatter = QuipFormatter()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

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

        # Track top 3 mining results
        self.top_results: List[MiningResult] = []

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
        self.top_results: List[MiningResult] = []


    def update_top_results(self, result: MiningResult, requirements: BlockRequirements):
        """Update the top 3 results list with a new mining result."""

        # Add current result
        self.top_results.append(result)

        # Sort by quality using the comparison function
        def sort_key(r):
            # Create a dummy result to compare against for sorting
            dummy = MiningResult(
                miner_id="", miner_type="", nonce=0, salt=b"", timestamp=0, prev_timestamp=0,
                solutions=[], energy=float('inf'), diversity=0.0, num_valid=0,
                mining_time=0.0, node_list=[], edge_list=[]
            )
            comparison = compare_mining_results(r, dummy, requirements)
            return comparison

        self.top_results.sort(key=sort_key)

        # Keep only top 3
        self.top_results = self.top_results[:3]

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


def compare_mining_results(result_a: MiningResult, result_b: MiningResult, requirements: BlockRequirements) -> int:
    """
    Compare two mining results to determine which is better.

    Returns:
        -1 if A is better than B
         0 if A and B are equal
         1 if B is better than A

    Comparison logic:
    1. Higher num_valid_solutions is better
    2. If equal (or both 0), compare average of top N solution energies
       where N = requirements.min_solutions
    3. If still equal, compare overall average solution energy
    """
    # 1. Compare number of valid solutions (higher is better)
    if result_a.num_valid > result_b.num_valid:
        return -1
    elif result_b.num_valid > result_a.num_valid:
        return 1

    # 2. If equal (or both 0), compare average of top N solution energies
    if result_a.num_valid > 0 and result_b.num_valid > 0:
        # Generate Ising models for both results
        h_a, J_a = generate_ising_model_from_nonce(result_a.nonce, result_a.node_list, result_a.edge_list)
        h_b, J_b = generate_ising_model_from_nonce(result_b.nonce, result_b.node_list, result_b.edge_list)

        # Calculate energies for top N solutions
        n_solutions = min(requirements.min_solutions, len(result_a.solutions), len(result_b.solutions))
        if n_solutions > 0:
            energies_a = [energy_of_solution(sol, h_a, J_a, result_a.node_list)
                         for sol in result_a.solutions[:n_solutions]]
            energies_b = [energy_of_solution(sol, h_b, J_b, result_b.node_list)
                         for sol in result_b.solutions[:n_solutions]]

            avg_energy_a = np.mean(energies_a)
            avg_energy_b = np.mean(energies_b)

            if avg_energy_a < avg_energy_b:  # Lower energy is better
                return -1
            elif avg_energy_b < avg_energy_a:
                return 1

    # 3. If still equal, compare overall best energy (lower is better)
    if result_a.energy < result_b.energy:
        return -1
    elif result_b.energy < result_a.energy:
        return 1

    return 0  # Equal

