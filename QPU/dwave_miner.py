"""QPU miner using D-Wave sampler for quantum mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple, cast, Mapping, Any

import dimod

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper, DefectInfo
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from shared.ising_model import IsingModel
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology


def _best_effort_cancel(future: Any, label: str = "") -> None:
    """Best-effort cancel of a D-Wave future.

    D-Wave's remote cancel is advisory (the solver may have already
    run) but invoking it lets the client release network resources.
    Failures are logged at debug level so flaky networks during
    teardown don't spam production logs.
    """
    cancel_fn = getattr(future, "cancel", None)
    if not callable(cancel_fn):
        return
    try:
        cancel_fn()
    except Exception as exc:
        init_logger.debug(
            "D-Wave future.cancel() failed%s (best-effort): %s: %s",
            f" for {label}" if label else "",
            type(exc).__name__, exc,
        )


def _shift_energies(sampleset: dimod.SampleSet, offset: float) -> dimod.SampleSet:
    """Shift all energies in a sampleset by a constant offset.

    Cheap O(n) operation — no sample dict copying. Used when we don't
    need full reconstruction but want approximate full-topology energies.
    """
    new_energies = sampleset.record.energy + offset
    # Build new record with shifted energies
    import numpy as np
    new_record = np.recarray(
        sampleset.record.shape,
        dtype=sampleset.record.dtype,
    )
    for name in sampleset.record.dtype.names:
        if name == 'energy':
            new_record.energy = new_energies
        else:
            new_record[name] = sampleset.record[name]
    return dimod.SampleSet(
        new_record, sampleset.variables, sampleset.info, sampleset.vartype,
    )


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
        drain_on_stop: bool = False,
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
            drain_on_stop: When True and stop_event fires, stop submitting
                new QPU jobs but wait for in-flight ones to complete so
                their results can be inspected. Used by tests that need
                to examine partial results. Default False: on stop,
                abandon pending futures immediately to free the node to
                start the next block as fast as possible.
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
        self.drain_on_stop = drain_on_stop
        self._feeder: Optional[IsingFeeder] = None
        self._stream: Optional[Iterator] = None
        # Stashed by _pre_mine_setup so the streaming iterator can observe
        # cancellation during its inner poll loop. Needed because
        # base_miner's outer stop_event check only fires between batches.
        self._stop_event: Optional[multiprocessing.synchronize.Event] = None

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
            if self._stream is not None and hasattr(self._stream, 'close'):
                self._stream.close()
                self._stream = None
            if self._feeder is not None:
                self._feeder.stop()
                self._feeder = None
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
        """Check QPU daily budget and create IsingFeeder."""
        # Stash the stop_event so sample_ising_streaming can poll it
        # between QPU completions without plumbing it through every call.
        self._stop_event = stop_event
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

        # Create IsingFeeder (same pattern as GPU miners)
        cur_index = prev_block.header.index + 1
        feeder_seed = kwargs.pop('feeder_seed', None)
        self._feeder = IsingFeeder(
            prev_hash=prev_block.hash,
            miner_id=node_info.miner_id,
            cur_index=cur_index,
            nodes=self.sampler.nodes,
            edges=self.sampler.edges,
            buffer_size=self.queue_depth * 2,
            seed=feeder_seed,
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
            'energy_threshold': current_requirements.difficulty_energy,
        }

    def sample_ising_streaming(
        self,
        feeder: IsingFeeder,
        *,
        num_reads: int,
        annealing_time: float,
        queue_depth: int,
        energy_threshold: float = 0.0,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Stream Ising model solutions via async QPU submission.

        Maintains queue_depth jobs in-flight on the D-Wave cloud.
        As each completes, checks whether the best QPU energy (plus
        the defect offset) could meet the threshold. Only reconstructs
        the full-topology sampleset for promising candidates.

        Non-candidates get a minimal sampleset with just the best
        energy — enough for the mining loop's progress tracking but
        without the ~1s cost of copying 4,500+ spin dicts.

        Args:
            feeder: IsingFeeder providing pre-generated IsingModels.
            num_reads: QPU reads per problem.
            annealing_time: Annealing time in microseconds.
            queue_depth: Number of concurrent in-flight QPU jobs.
            energy_threshold: Current difficulty energy. Only samplesets
                whose best QPU energy + defect offset < threshold are
                fully reconstructed.

        Yields:
            (IsingModel, SampleSet) in completion order.
        """
        topology_label = self.sampler.job_label

        # pending: {future_id: (model, future, defect_info, job_index)}
        pending: Dict[int, Tuple[IsingModel, Any, Optional[DefectInfo], int]] = {}
        job_index = 0

        def submit_one():
            nonlocal job_index
            model = feeder.pop_blocking()
            future, defect_info = self.sampler.sample_ising_async(
                model.h, model.J,
                num_reads=num_reads,
                answer_mode='raw',
                annealing_time=annealing_time,
                label=f"{topology_label}_s{job_index}",
                nonce_seed=model.nonce,
            )
            pending[id(future)] = (model, future, defect_info, job_index)
            job_index += 1

        # Fill initial queue
        for _ in range(queue_depth):
            submit_one()

        stop_event = self._stop_event

        def _cancel_pending():
            """Abandon in-flight D-Wave futures on stop.

            Remote cancel is best-effort (see _best_effort_cancel).
            Clearing the local map is what frees the mining child to
            move on to the next block.
            """
            for _mdl, fut, _defect, fidx in pending.values():
                _best_effort_cancel(fut, label=f"job {fidx}")
            pending.clear()

        drain_engaged = False

        # Stream: poll for completions, yield, refill
        while pending:
            # Observe cancellation between polls. Default is discard-on-
            # stop: abandon pending futures and stop yielding so
            # base_miner's outer loop can exit promptly. With
            # drain_on_stop=True (test-only), stop submitting new work
            # but let in-flight jobs complete so callers can still read
            # their results.
            if stop_event is not None and stop_event.is_set():
                if not self.drain_on_stop:
                    _cancel_pending()
                    return
                if not drain_engaged:
                    # Suppress submit_one() for the remainder of this
                    # stream so the pipeline winds down naturally as
                    # pending empties. One-shot — repeated reassignment
                    # would just churn lambdas every poll.
                    self.logger.info(
                        f"[QPU] drain_on_stop engaged with "
                        f"{len(pending)} jobs in flight; no new "
                        f"submissions until stream completes"
                    )
                    submit_one = lambda: None  # noqa: E731
                    drain_engaged = True

            completed_id = None
            while completed_id is None:
                for fid, (_, fut, _, _) in pending.items():
                    if fut.done():
                        completed_id = fid
                        break
                if completed_id is None:
                    if stop_event is not None and stop_event.is_set() and not self.drain_on_stop:
                        _cancel_pending()
                        return
                    time.sleep(0.02)

            model, future, defect_info, _ = pending.pop(completed_id)
            raw_ss = future.sampleset

            # Refill the slot immediately (before any reconstruction).
            # In drain mode, submit_one is a no-op so the pipeline winds
            # down naturally as pending empties.
            submit_one()

            if defect_info is not None:
                # Check if best QPU energy + offset could meet threshold
                best_qpu_energy = min(raw_ss.record.energy)
                approx_energy = best_qpu_energy + defect_info.energy_offset

                if approx_energy < energy_threshold:
                    # Promising — full reconstruction
                    sampleset = self.sampler.reconstruct_full_sampleset(
                        raw_ss, defect_info,
                    )
                else:
                    # Not promising — yield raw QPU energies shifted by
                    # offset so the mining loop sees approximate values
                    # without paying reconstruction cost.
                    sampleset = _shift_energies(raw_ss, defect_info.energy_offset)
            else:
                sampleset = raw_ss

            yield model, sampleset

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
        (nonce, salt, sampleset) per call — matching the GPU miner pattern.
        """
        annealing_time = kwargs.pop('annealing_time', 120.0)
        energy_threshold = kwargs.pop('energy_threshold', 0.0)

        if self._stream is None:
            if self._feeder is None:
                return None  # _pre_mine_setup not called
            self._stream = self.sample_ising_streaming(
                feeder=self._feeder,
                num_reads=num_reads,
                annealing_time=annealing_time,
                queue_depth=self.queue_depth,
                energy_threshold=energy_threshold,
            )
            self.logger.info(
                f"[QPU] Streaming started: queue_depth={self.queue_depth}, "
                f"num_reads={num_reads}, annealing_time={annealing_time}μs"
            )

        try:
            model, sampleset = next(self._stream)
        except StopIteration:
            return None

        self._record_qpu_timing(sampleset)
        return [(model.nonce, model.salt, sampleset)]

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

    #: Hard wall-time cap on the synchronous fallback _sample path.
    #: Without this, the mining child can block indefinitely inside
    #: the D-Wave SDK (the underlying sample call has no timeout)
    #: and stop_event is never observed — which looks to the
    #: orchestrator like mining silently stopped.
    SYNC_SAMPLE_TIMEOUT = 60.0

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
        """Fallback synchronous QPU sampling with cancel + timeout.

        Uses sample_ising_async + polling so this path is cancellable
        via stop_event and bounded by SYNC_SAMPLE_TIMEOUT. Used when
        the streaming _sample_batch path is unavailable (e.g. after
        a feeder exception tore down the stream generator).
        """
        h_cast = cast(Mapping[Any, float], h)
        J_cast = cast(Mapping[Tuple[Any, Any], float], J)

        topology_label = self.sampler.job_label
        nonce_seed = kwargs.pop('nonce_seed', None)

        t0 = time.monotonic()
        future, defect_info = self.sampler.sample_ising_async(
            h_cast, J_cast,
            num_reads=num_reads,
            answer_mode='raw',
            annealing_time=annealing_time,
            label=f"{topology_label}_sync",
            nonce_seed=nonce_seed,
        )

        deadline = t0 + self.SYNC_SAMPLE_TIMEOUT
        stop_event = self._stop_event
        while not future.done():
            if stop_event is not None and stop_event.is_set():
                _best_effort_cancel(future)
                raise RuntimeError("mining cancelled during _sample")
            if time.monotonic() >= deadline:
                _best_effort_cancel(future)
                raise TimeoutError(
                    f"QPU _sample exceeded {self.SYNC_SAMPLE_TIMEOUT:.1f}s "
                    f"(label={topology_label}_sync)"
                )
            time.sleep(0.05)

        elapsed = time.monotonic() - t0
        raw_ss = future.sampleset
        if defect_info is not None:
            sampleset = self.sampler.reconstruct_full_sampleset(
                raw_ss, defect_info,
            )
        else:
            sampleset = raw_ss
        self._record_qpu_timing(sampleset)
        self.logger.info(
            f"[QPU] _sample (sync fallback) completed in {elapsed:.2f}s"
        )
        return sampleset

    def _post_mine_cleanup(self) -> None:
        """Stop the streaming pipeline and feeder."""
        if self._stream is not None:
            if hasattr(self._stream, 'close'):
                self._stream.close()
            self._stream = None
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None
