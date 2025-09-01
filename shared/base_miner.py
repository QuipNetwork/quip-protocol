"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import collections.abc
from blake3 import blake3
import multiprocessing
import multiprocessing.synchronize
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Protocol, Any, Union

import numpy as np
from shared.quantum_proof_of_work import (
    calculate_hamming_distance as _shared_hamming,
    calculate_diversity as _shared_diversity,
    filter_diverse_solutions as _shared_filter,
)
from shared.logging_config import get_logger



@dataclass
class MiningResult:
    miner_id: str
    miner_type: str
    nonce: int
    salt: bytes
    timestamp: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    num_valid: int
    mining_time: float
    node_list: List[int]
    edge_list: List[Tuple[int, int]]
    variable_order: Optional[List[int]] = None

Variable = collections.abc.Hashable
Bias = float
class Sampler(Protocol):
    """Protocol defining the D-Wave sampler interface."""
    nodelist: List[Variable]
    edgelist: List[Tuple[Variable, Variable]]
    properties: Dict[str, Any]
    sampler_type: str
    def sample_ising(
        self,
        h: Union[Mapping[Variable, Bias], Sequence[Bias]],
        J: Mapping[Tuple[Variable, Variable], Bias],
        **kwargs
    ) -> Any:
        ...

class BaseMiner(ABC):
    """Abstract base class for concrete miners.

    Subclasses must implement:
      - mine_block(): miner-specific mining logic
      - set self.miner_type and self.sampler in __init__
    """

    def __init__(
        self,
        miner_id: str,
        sampler: Sampler
    ) -> None:
        if type(self) is BaseMiner:
            raise TypeError("BaseMiner is abstract; instantiate a concrete subclass")
        self.miner_id = miner_id
        self.miner_type: str = "UNKNOWN"
        self.mining = False
        self.blocks_won = 0
        self.total_rewards = 0
        self.sampler = sampler

        # Initialize logger
        self.logger = get_logger('base_miner')

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
        if miner_id and miner_id[-1].isdigit():
            initial_sweeps = pow(2, 6 + int(miner_id[-1]))

        self.adaptive_params = {
            'quantum_annealing_time': 20.0,  # microseconds for QPU
            'beta_range': [0.1, 10.0],  # for SA
            'beta_schedule': 'geometric',  # or 'linear'
            'num_sweeps': initial_sweeps  # for SA
        }

    def calculate_hamming_distance(self, s1: List[int], s2: List[int]) -> int:
        return _shared_hamming(s1, s2)

    def calculate_diversity(self, solutions: List[List[int]]) -> float:
        return _shared_diversity(solutions)

    def filter_diverse_solutions(self, solutions: List[List[int]], target_count: int) -> List[List[int]]:
        return _shared_filter(solutions, target_count)

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
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Abstract method for miner-specific mining implementation.

        Args:
            block: Block object containing header, data, and other block information
            requirements: NextBlockRequirements object with difficulty settings
            result_queue: Multiprocessing queue for results
            stop_event: Multiprocessing event to signal stop
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

