"""QPU miner using D-Wave sampler for quantum mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple, cast, Mapping, Any

import dimod

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
)
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology


class QPUStream:
    """Streaming iterator that keeps queue_depth QPU jobs in-flight.

    Modeled on the GPU streaming pattern: each next() call polls for
    a completed future, yields its result, and submits a replacement
    to keep the pipeline full. This overlaps network round-trip latency
    (~2-3s per call) so the QPU stays saturated.
    """

    def __init__(
        self,
        sampler: DWaveSamplerWrapper,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        num_reads: int,
        annealing_time: float,
        queue_depth: int,
    ):
        self._sampler = sampler
        self._prev_hash = prev_hash
        self._miner_id = miner_id
        self._cur_index = cur_index
        self._nodes = nodes
        self._edges = edges
        self._num_reads = num_reads
        self._annealing_time = annealing_time
        self._queue_depth = queue_depth
        self._job_index = 0
        self._topology_label = sampler.job_label

        # pending: {future_id: (nonce, salt, future)}
        self._pending: Dict[int, Tuple[int, bytes, Any]] = {}

        # Fill initial queue
        for _ in range(queue_depth):
            self._submit_one()

    def _submit_one(self):
        """Generate and submit a single async QPU problem."""
        salt = random.randbytes(32)
        nonce = ising_nonce_from_block(
            self._prev_hash, self._miner_id,
            self._cur_index, salt,
        )
        h, J = generate_ising_model_from_nonce(
            nonce, self._nodes, self._edges,
        )
        future = self._sampler.sample_ising_async(
            h, J,
            num_reads=self._num_reads,
            answer_mode='raw',
            annealing_time=self._annealing_time,
            label=f"{self._topology_label}_s{self._job_index}",
            nonce_seed=nonce,
        )
        self._pending[id(future)] = (nonce, salt, future)
        self._job_index += 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, bytes, dimod.SampleSet]:
        """Poll for a completed future, return it, submit a replacement."""
        if not self._pending:
            raise StopIteration

        # Poll until one completes
        while True:
            for fid, (nonce, salt, fut) in self._pending.items():
                if fut.done():
                    self._pending.pop(fid)
                    sampleset = fut.sampleset

                    # Refill the slot
                    self._submit_one()

                    return nonce, salt, sampleset
            time.sleep(0.02)

    def close(self):
        """Cancel all pending futures."""
        for _, _, fut in self._pending.values():
            try:
                fut.cancel()
            except Exception:
                pass
        self._pending.clear()


class DWaveMiner(BaseMiner):

    def __init__(
        self,
        miner_id: str,
        topology: DWaveTopology = DEFAULT_TOPOLOGY,
        embedding_file: Optional[str] = None,
        time_config: Optional[QPUTimeConfig] = None,
        queue_depth: int = 30,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
        **cfg
    ):
        """Initialize D-Wave QPU miner.

        Args:
            miner_id: Unique identifier for this miner.
            topology: Topology object (default: DEFAULT_TOPOLOGY).
            embedding_file: Optional path to embedding file.
            time_config: Optional QPUTimeConfig for time budget management.
            queue_depth: Number of QPU jobs to keep in-flight (default: 30).
            solver_name: Optional solver name (e.g. "Advantage2_system1").
            region: Optional D-Wave region (e.g. "na-east-1").
        """
        init_logger.info(
            f"[QPU] Initializing DWaveMiner with topology: {topology.solver_name}"
        )
        try:
            sampler = DWaveSamplerWrapper(
                topology=topology,
                embedding_file=embedding_file,
                solver_name=solver_name,
                region=region,
            )
            init_logger.info(
                f"[QPU] Sampler ready: {len(sampler.nodes)} nodes, "
                f"{len(sampler.edges)} edges"
            )
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
                f"[QPU] Daily budget enabled: "
                f"{time_config.daily_budget_seconds:.1f}s/day"
            )
        else:
            self.logger.info(
                "[QPU] Daily budget management disabled - no budget configured"
            )

        self.queue_depth = queue_depth
        self._stream: Optional[QPUStream] = None

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of QPU resources."""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"QPU miner {self.miner_id} received SIGTERM, "
                f"cleaning up D-Wave resources..."
            )
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'close'):
                self.sampler.close()
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during QPU miner cleanup: {e}")
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
                    else f"{estimate.seconds_until_can_mine / 3600:.1f}h"
                )
                self.logger.info(
                    f"[QPU] Pacing block {cur_index} - waiting {wait_str} "
                    f"for limit to catch up. "
                    f"Used: {estimate.cumulative_used_us / 1e6:.2f}s, "
                    f"Limit: {estimate.proportional_limit_us / 1e6:.2f}s "
                    f"({estimate.elapsed_fraction * 100:.1f}% of day)"
                )
                return False

            self.logger.info(
                f"[QPU] Budget check passed. Used: "
                f"{estimate.cumulative_used_us / 1e6:.2f}s / "
                f"{estimate.proportional_limit_us / 1e6:.2f}s limit "
                f"({estimate.elapsed_fraction * 100:.1f}% of day), "
                f"Estimated: {estimate.estimated_block_time_us / 1e6:.2f}s "
                f"({estimate.confidence} confidence)"
            )
        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """Return fixed optimal QPU parameters.

        Based on "Multi-Solver QPU Parameter Grid Test" (2026-03-30):
        512 reads x 120us is the universal optimum across all D-Wave
        solvers and architectures, within 0.1% of absolute best while
        using ~4x less QPU time than higher settings.
        """
        return {
            'num_reads': 512,
            'annealing_time': 120.0,
        }

    def _sample_batch(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> Optional[List[Tuple[int, bytes, dimod.SampleSet]]]:
        """Stream one result from the QPU pipeline.

        Lazily creates the streaming iterator on first call. Returns one
        (nonce, salt, sampleset) per call — matching the GPU streaming
        pattern. The iterator maintains queue_depth in-flight futures.
        """
        annealing_time = kwargs.pop('annealing_time', 120.0)

        if self._stream is None:
            self._stream = QPUStream(
                sampler=self.sampler,
                prev_hash=prev_hash,
                miner_id=miner_id,
                cur_index=cur_index,
                nodes=nodes,
                edges=edges,
                num_reads=num_reads,
                annealing_time=annealing_time,
                queue_depth=self.queue_depth,
            )
            self.logger.info(
                f"[QPU] Streaming started: queue_depth={self.queue_depth}, "
                f"num_reads={num_reads}, annealing_time={annealing_time}μs"
            )

        try:
            nonce, salt, sampleset = next(self._stream)
        except StopIteration:
            return None

        self._record_qpu_timing(sampleset)
        return [(nonce, salt, sampleset)]

    def _record_qpu_timing(self, sampleset: dimod.SampleSet):
        """Extract and record QPU timing from a sampleset."""
        if not hasattr(sampleset, 'info') or 'timing' not in sampleset.info:
            return
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
            if self.time_manager is not None:
                self.time_manager.record_block_time(qpu_total_access)

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        annealing_time: float = 120.0,
        **kwargs,
    ) -> dimod.SampleSet:
        """Synchronous QPU sampling (fallback, not used in streaming)."""
        h_cast = cast(Mapping[Any, float], h)
        J_cast = cast(Mapping[Tuple[Any, Any], float], J)

        topology_label = self.sampler.job_label
        nonce_seed = kwargs.pop('nonce_seed', None)
        sampleset = self.sampler.sample_ising(
            h_cast, J_cast,
            num_reads=num_reads,
            answer_mode='raw',
            annealing_time=annealing_time,
            label=f"{topology_label}_sync",
            nonce_seed=nonce_seed,
        )
        self._record_qpu_timing(sampleset)
        return sampleset

    def _post_mine_cleanup(self) -> None:
        """Stop the streaming pipeline."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
