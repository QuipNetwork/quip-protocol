"""QPU miner using D-Wave sampler for quantum mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, cast, Mapping, Any

import dimod

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology


class DWaveMiner(BaseMiner):
    # QPU annealing time + bonus reads (no sweeps)
    ADAPT_MIN_ANNEALING_TIME = 5.0    # μs, easiest difficulty
    ADAPT_MAX_ANNEALING_TIME = 20.0   # μs, hardest difficulty
    ADAPT_MIN_BONUS_READS = 32
    ADAPT_MAX_BONUS_READS = 64

    def __init__(
        self,
        miner_id: str,
        topology: DWaveTopology = DEFAULT_TOPOLOGY,
        embedding_file: Optional[str] = None,
        time_config: Optional[QPUTimeConfig] = None,
        queue_depth: int = 10,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
        **cfg
    ):
        """
        Initialize D-Wave QPU miner.

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
            solver_name: Optional explicit solver name to connect to (e.g.
                        "Advantage2_system1.13"). If None, uses DWAVE_API_SOLVER env var.
            region: Optional D-Wave region (e.g. "na-east-1").
                   If None, uses default from config.
            **cfg: Additional configuration options (reserved for future use)
        """
        # Create sampler (encapsulates embedding internally)
        init_logger.info(f"[QPU] Initializing DWaveMiner with topology: {topology.solver_name}")
        try:
            sampler = DWaveSamplerWrapper(
                topology=topology,
                embedding_file=embedding_file,
                solver_name=solver_name,
                region=region,
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

        # Queue depth (preserved for potential future streaming use)
        self.queue_depth = queue_depth

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of QPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"QPU miner {self.miner_id} received SIGTERM, cleaning up D-Wave resources...")

        # QPU-specific cleanup
        try:
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

    def _pre_mine_setup(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> bool:
        """Check QPU daily budget before starting."""
        if self.time_manager is not None:
            estimate = self.time_manager.should_mine_block()
            if not estimate.should_mine:
                cur_index = prev_block.header.index + 1
                wait_str = (
                    f"{estimate.seconds_until_can_mine:.0f}s"
                    if estimate.seconds_until_can_mine < 3600
                    else f"{estimate.seconds_until_can_mine/3600:.1f}h"
                )
                self.logger.info(
                    f"[QPU] Pacing block {cur_index} - waiting {wait_str} "
                    f"for limit to catch up. "
                    f"Used: {estimate.cumulative_used_us/1e6:.2f}s, "
                    f"Limit: {estimate.proportional_limit_us/1e6:.2f}s "
                    f"({estimate.elapsed_fraction*100:.1f}% of day)"
                )
                return False

            self.logger.info(
                f"[QPU] Budget check passed. Used: "
                f"{estimate.cumulative_used_us/1e6:.2f}s / "
                f"{estimate.proportional_limit_us/1e6:.2f}s limit "
                f"({estimate.elapsed_fraction*100:.1f}% of day), "
                f"Estimated: {estimate.estimated_block_time_us/1e6:.2f}s "
                f"({estimate.confidence} confidence)"
            )
        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        annealing_time: float = 20.0,
        **kwargs,
    ) -> dimod.SampleSet:
        """Submit a synchronous QPU sampling call."""
        h_cast = cast(Mapping[Any, float], h)
        J_cast = cast(Mapping[Tuple[Any, Any], float], J)

        # Generate job label
        topology_label = self.sampler.job_label
        job_label = f"{topology_label}_sync"

        sampleset = self.sampler.sample_ising(
            h_cast, J_cast,
            num_reads=num_reads,
            answer_mode='raw',
            annealing_time=annealing_time,
            label=job_label,
        )

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

        return sampleset
