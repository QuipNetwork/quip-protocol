"""QPU miner using D-Wave sampler for quantum mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple, Any

import dimod

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper, DefectInfo
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements
from shared.ising_feeder import RandomIsingFeeder
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


def build_production_stream(
    *,
    miner_id: str,
    num_reads: int,
    annealing_time: float,
    queue_depth: int,
    energy_threshold: float,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    last_proof_block_hash: bytes,
    miner_bytes: bytes,
    feeder_buffer_size: int,
    solver_name: Optional[str] = None,
    region: Optional[str] = None,
    token: Optional[str] = None,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
) -> Tuple[Iterator[Tuple[IsingModel, dimod.SampleSet]], Any]:
    """Build the production QPU stream inside the stream-driver process.

    Constructs a :class:`DWaveMiner` (its own D-Wave client), a
    :class:`RandomIsingFeeder`, and returns ``(stream, cleanup)`` where
    ``stream`` is the iterator from
    :meth:`DWaveMiner.sample_ising_streaming` and ``cleanup()`` stops the
    feeder and closes the sampler. Runs ONLY in the stream-driver process;
    it is never instantiated in tests because it connects to D-Wave.

    Args:
        miner_id: Unique identifier for this miner.
        num_reads: QPU reads per submission.
        annealing_time: Annealing time in microseconds.
        queue_depth: Number of concurrent in-flight QPU jobs.
        energy_threshold: Current difficulty energy gate.
        nodes: Topology node list (must match the configured solver).
        edges: Topology edge list.
        last_proof_block_hash: 32-byte ``block_hash(LastProofBlock)`` seed.
        miner_bytes: Canonical 32-byte miner identity.
        feeder_buffer_size: Target ready + in-flight feeder depth.
        solver_name: Optional D-Wave solver name.
        region: Optional D-Wave region.
        token: Optional D-Wave API token (passed through verbatim).
        stop_event: Optional event the streaming loop polls for cancellation.

    Returns:
        Tuple of ``(stream, cleanup)``.
    """
    miner = DWaveMiner(
        miner_id=miner_id,
        queue_depth=queue_depth,
        solver_name=solver_name,
        region=region,
        token=token,
    )
    feeder = RandomIsingFeeder(
        last_proof_block_hash=last_proof_block_hash,
        miner_bytes=miner_bytes,
        nodes=nodes,
        edges=edges,
        buffer_size=feeder_buffer_size,
    )
    if stop_event is not None:
        miner._stop_event = stop_event
    stream = miner.sample_ising_streaming(
        feeder,
        num_reads=num_reads,
        annealing_time=annealing_time,
        queue_depth=queue_depth,
        energy_threshold=energy_threshold,
    )

    def cleanup() -> None:
        try:
            feeder.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(miner, "sampler") and hasattr(miner.sampler, "close"):
                miner.sampler.close()
        except Exception:  # noqa: BLE001
            pass

    return stream, cleanup


# Default interval between repeated pacing log lines (seconds).
_PACING_LOG_INTERVAL = 60.0


class _PacingRateLimiter:
    """Rate-limiter for the QPU budget-pacing log line.

    Emits at most once per ``log_interval`` seconds AND always on the
    first entry into the paced state or when the wait-bucket changes.
    Call :meth:`reset` when mining resumes so the next pacing episode
    is treated as a fresh entry.

    The clock is injectable (``now`` parameter on :meth:`should_log`)
    so unit tests can drive time deterministically without monkeypatching
    a global.
    """

    def __init__(self, log_interval: float = _PACING_LOG_INTERVAL) -> None:
        self._interval = log_interval
        self._last_log_time: Optional[float] = None  # None ↔ "not paced yet"
        self._last_bucket: Optional[str] = None

    def should_log(self, now: float, wait_bucket: str) -> bool:
        """Return True if a log line should be emitted.

        Args:
            now: Current monotonic time (seconds).
            wait_bucket: Human-readable wait estimate (e.g. ``"2h"`` or
                ``"45m"``).  A change in this value forces a log even if
                the interval has not elapsed.

        Returns:
            ``True`` when the caller should log; ``False`` to suppress.
        """
        if self._last_log_time is None:
            # First time entering the paced state — always log.
            self._last_log_time = now
            self._last_bucket = wait_bucket
            return True

        bucket_changed = wait_bucket != self._last_bucket
        interval_elapsed = (now - self._last_log_time) >= self._interval

        if bucket_changed or interval_elapsed:
            self._last_log_time = now
            self._last_bucket = wait_bucket
            return True

        return False

    def reset(self) -> None:
        """Mark that mining has resumed; the next paced call logs again."""
        self._last_log_time = None
        self._last_bucket = None


class DWaveMiner(BaseMiner):
    # Old code sized the feeder as ``queue_depth * 2`` (default 60).
    # Keep that headroom so the streaming sampler can saturate the
    # D-Wave cloud queue without blocking on Python-side derivation.
    FEEDER_BUFFER_SIZE = 60
    # QPU is the async-streaming backend: drive the stream on a pump thread
    # so per-result processing never blocks the cloud pipeline.
    STREAMING_PUMP = True

    def __init__(
        self,
        miner_id: str,
        topology: DWaveTopology = DEFAULT_TOPOLOGY,
        embedding_file: Optional[str] = None,
        time_config: Optional[QPUTimeConfig] = None,
        queue_depth: int = 30,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
        token: Optional[str] = None,
        drain_on_stop: bool = False,
        num_reads: Optional[int] = None,
        annealing_time_us: Optional[float] = None,
        connect: bool = True,
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
            token: Optional D-Wave API token. When unset the SDK falls
                back to the `DWAVE_API_KEY` env var; when set, it wins
                over env. Matches the operator's expectation that a
                TOML `[dwave].token` value is honored verbatim.
            drain_on_stop: When True and stop_event fires, stop submitting
                new QPU jobs but wait for in-flight ones to complete so
                their results can be inspected. Used by tests that need
                to examine partial results. Default False: on stop,
                abandon pending futures immediately to free the node to
                start the next block as fast as possible.
            num_reads: Optional override for QPU reads per submission.
                ``None`` (default) uses the hardcoded throughput-tuned
                value from ``_adapt_mining_params``. Set via TOML
                ``[dwave].num_reads`` to retune (e.g. raise for solution
                quality).
            annealing_time_us: Optional override for anneal duration in
                microseconds. ``None`` uses the hardcoded default. Set
                via TOML ``[dwave].annealing_time_us``.
            connect: When True (default) build a live ``DWaveSamplerWrapper``
                and own the D-Wave connection. When False, construct without
                a sampler (``self.sampler = None``) for the worker/orchestrator
                instance: the single D-Wave connection lives in the
                stream-driver process. All non-sampler machinery (budget gate,
                param overrides, time manager, pacing) still initializes so the
                connection-less miner can run the dispatch loop.
        """
        init_logger.info(
            f"[QPU] Initializing DWaveMiner with topology: {topology.solver_name}"
        )
        if connect:
            try:
                sampler = DWaveSamplerWrapper(
                    topology=topology,
                    embedding_file=embedding_file,
                    solver_name=solver_name,
                    region=region,
                    token=token,
                )
                init_logger.info(
                    f"[QPU] Sampler ready: {len(sampler.nodes)} nodes, "
                    f"{len(sampler.edges)} edges"
                )
            except Exception as e:
                init_logger.error(f"[QPU] Failed to initialize sampler: {e}")
                raise
        else:
            sampler = None
            init_logger.info(
                "[QPU] constructed without sampler (orchestrator mode)"
            )
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
        # Operator-tunable overrides for the per-submission cost knobs.
        # Validated lightly here; the D-Wave SDK rejects out-of-range
        # values per-solver with a clear error at first submission.
        if num_reads is not None and num_reads < 1:
            raise ValueError(
                f"num_reads must be >= 1 if set, got {num_reads}"
            )
        if annealing_time_us is not None and annealing_time_us <= 0:
            raise ValueError(
                "annealing_time_us must be > 0 if set, got "
                f"{annealing_time_us}"
            )
        self._num_reads_override = num_reads
        self._annealing_time_override = annealing_time_us
        if num_reads is not None or annealing_time_us is not None:
            self.logger.info(
                "[QPU] parameter override active: num_reads=%s "
                "annealing_time_us=%s",
                num_reads, annealing_time_us,
            )
        self._feeder: Optional[RandomIsingFeeder] = None
        self._stream: Optional[Iterator] = None
        # Stashed by _pre_mine_setup so the streaming iterator can observe
        # cancellation during its inner poll loop. Needed because
        # base_miner's outer stop_event check only fires between batches.
        self._stop_event: Optional[multiprocessing.synchronize.Event] = None

        # Rate-limiter for the budget-pacing log line. Suppresses the
        # per-head "[QPU] Pacing block …" spam when the daily budget is
        # exhausted; the first entry and any bucket/interval change still
        # surface. reset() is called when mining resumes.
        self._pacing_rl = _PacingRateLimiter()

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
            if getattr(self, 'sampler', None) is not None and hasattr(
                self.sampler, 'close'
            ):
                self.sampler.close()
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during QPU miner cleanup: {e}")
        # Guard against raising SystemExit during interpreter finalization
        # (would produce "Exception ignored" noise on stderr).
        self._graceful_exit()

    def _pre_mine_setup(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> bool:
        """Check the QPU daily budget before mining can begin.

        The Ising feeder is now built by ``BaseMiner.mine_work_item``
        via ``context.make_feeder(...)``; this hook is kept solely for
        the budget gate (``time_manager.should_mine_block()``), which
        must run before any QPU calls so we don't burn budget on an
        already-exhausted day.
        """
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
                if self._pacing_rl.should_log(
                    now=time.monotonic(), wait_bucket=wait_str
                ):
                    self.logger.info(
                        f"[QPU] Pacing block {cur_index} - waiting {wait_str} "
                        f"for limit to catch up. "
                        f"Used: {estimate.cumulative_used_us / 1e6:.2f}s, "
                        f"Limit: {estimate.proportional_limit_us / 1e6:.2f}s "
                        f"({estimate.elapsed_fraction * 100:.1f}% of day)"
                    )
                return False

            # Mining can proceed — reset the pacing rate-limiter so the
            # next pacing episode is treated as a fresh entry.
            self._pacing_rl.reset()
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
        """Return QPU sampler parameters for this submission.

        Defaults are the throughput-tuned values from the QPU
        time-to-solution study (``qpu_tts_test/``, 2026-05): 112 reads x
        80us — the QPU-TTS optimum (~22s normalized time-to-solution,
        ~5x faster than the older 512x120 quality-tuned setting).
        ``p_success`` plateaus above ~96-128 reads, so extra reads only
        add QPU access time; 80us maximizes top-5 diversity (the binding
        chain-validation gate) within the productive anneal range.

        Operators tuning for solution quality rather than throughput can
        override via TOML ``[dwave].num_reads`` and
        ``[dwave].annealing_time_us``; see
        ``tools/qpu_throughput_canary.py`` for the canary + sweep flow
        that informs those values.
        """
        num_reads = (
            self._num_reads_override
            if self._num_reads_override is not None
            else 112
        )
        annealing_time = (
            self._annealing_time_override
            if self._annealing_time_override is not None
            else 80.0
        )
        return {
            'num_reads': num_reads,
            'annealing_time': annealing_time,
            'energy_threshold': current_requirements.difficulty_energy,
        }

    def sample_ising_streaming(
        self,
        feeder: RandomIsingFeeder,
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
            feeder: RandomIsingFeeder providing pre-generated IsingModels.
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
        # getattr default guards tests that bypass BaseMiner.__init__ via
        # object.__new__; production always has self._pump_stop set (to None
        # until mine_work_item starts a pump).
        pump_stop = getattr(self, "_pump_stop", None)

        def _stop_requested() -> bool:
            # stop_event/pump_stop captured once at stream start; stable for
            # this stream's lifetime (a new dispatch builds a fresh stream).
            if stop_event is not None and stop_event.is_set():
                return True
            return pump_stop is not None and pump_stop.is_set()

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
            if _stop_requested():
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
                    if _stop_requested() and not self.drain_on_stop:
                        _cancel_pending()
                        return
                    # 5ms poll: at production throughput one job per ~60ms
                    # the difference vs 20ms is small per-cycle but worth a
                    # few percent over a long mining session, and matters
                    # more when shorter (num_reads, annealing_time) push
                    # the per-job time down.
                    time.sleep(0.005)

            model, future, defect_info, _ = pending.pop(completed_id)
            raw_ss = future.sampleset

            # Refill the slot immediately (before any reconstruction).
            # In drain mode, submit_one is a no-op so the pipeline winds
            # down naturally as pending empties.
            submit_one()

            if job_index % 100 == 0:
                feeder_stats = feeder.stats()
                self.logger.info(
                    "[QPU] stream depth: in_flight=%d/%d "
                    "feeder_ready=%d/%d drained=%d wait_total=%.2fs",
                    len(pending), queue_depth,
                    feeder_stats['ready'], feeder_stats['buffer_size'],
                    feeder_stats['drained_count'],
                    feeder_stats['pop_wait_total_s'],
                )

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
        annealing_time = kwargs.pop('annealing_time', 80.0)
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

    def _sample(self, h, J, *, num_reads, num_sweeps, **kwargs):
        """Unused on the QPU path — sampling runs in the stream-driver process.

        QPU mining is STREAMING_PUMP=True and is sourced exclusively from the
        stream-driver descriptor queue; the legacy synchronous fallback was
        removed. Kept only to satisfy the BaseMiner ABC.
        """
        raise NotImplementedError(
            "DWaveMiner does not sample synchronously; use the stream driver"
        )

    def _post_mine_cleanup(self) -> None:
        """Stop the streaming pipeline and feeder."""
        if self._stream is not None:
            if hasattr(self._stream, 'close'):
                self._stream.close()
            self._stream = None
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None
