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


def _should_reconstruct(
    best_qpu_energy: float,
    defect_offset: float,
    threshold_energy: float,
) -> bool:
    """Return True if a sampleset is promising enough to fully reconstruct.

    The gate is ``(best_qpu_energy + defect_offset) < threshold_energy``.
    Callers pass the *effective* threshold: ``sample_ising_streaming`` passes
    the fixed difficulty energy; the persistent driver passes the live
    (decayed) threshold widened by ``RATCHET_PRECHECK_MARGIN_MILLI`` so it
    reconstructs anything the worker's ratchet would want to stash.
    """
    return (best_qpu_energy + defect_offset) < threshold_energy


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


class PersistentStreamContext:
    """Long-lived D-Wave connection + feeder driving generation-tagged streams.

    Built ONCE per driver process by :func:`build_persistent_context` (the
    expensive ``DWaveSampler`` solver download happens there). A chain-head
    change calls :meth:`apply_command` with a ``switch`` tuple to swap the
    round seed (via ``feeder.reseed`` — no re-fork, no reconnect); a live
    threshold decay calls it with a ``threshold`` tuple to widen/narrow the
    reconstruction gate WITHOUT bumping the generation.

    :meth:`iter_results` is one long-lived generator. It maintains
    ``queue_depth`` in-flight QPU submissions, tagging each with the
    generation it was submitted under; on a switch it cancels in-flight
    futures and reseeds, so a completion from a superseded round is never
    yielded. Each completion is gated by :func:`_should_reconstruct` against
    the LIVE (decayed) threshold widened by ``precheck_margin_milli`` so the
    worker always receives full-width samplesets it can stash.
    """

    def __init__(
        self,
        *,
        miner: "DWaveMiner",
        nodes: List[int],
        edges: List[Tuple[int, int]],
        feeder_buffer_size: int,
        num_reads: int,
        annealing_time: float,
        energy_threshold_milli: int,
        precheck_margin_milli: int,
        queue_depth: int,
        stop_event: Optional[multiprocessing.synchronize.Event] = None,
    ) -> None:
        self._miner = miner
        self._nodes = nodes
        self._edges = edges
        self._feeder_buffer_size = feeder_buffer_size
        self._num_reads = num_reads
        self._annealing_time = annealing_time
        self._energy_threshold_milli = int(energy_threshold_milli)
        self._precheck_margin_milli = int(precheck_margin_milli)
        self._queue_depth = queue_depth
        self._stop_event = stop_event
        # Feeder is built lazily on the first 'switch' (it needs the round
        # seed); thereafter reseed() keeps the same pool.
        self._feeder: Optional[RandomIsingFeeder] = None
        self.generation: int = 0
        # pending[id(future)] = (model, future, defect_info, job_index, gen)
        self._pending: Dict[int, Tuple[Any, Any, Any, int, int]] = {}
        self._job_index = 0

    # -- control ----------------------------------------------------------

    def set_threshold(self, energy_threshold_milli: int) -> None:
        """Update the live reconstruction threshold (no reseed, no gen bump)."""
        self._energy_threshold_milli = int(energy_threshold_milli)

    def cancel_inflight(self) -> None:
        """Best-effort-cancel and drop all in-flight QPU futures."""
        for _mdl, fut, _d, fidx, _g in self._pending.values():
            _best_effort_cancel(fut, label=f"job {fidx}")
        self._pending.clear()

    def apply_command(self, cmd: Tuple[Any, ...]) -> None:
        """Apply one ctl_q command tuple.

        ``("switch", gen, last_proof_block_hash, miner_bytes, thr_milli,
        num_reads, annealing_time)`` bumps the generation, cancels in-flight
        work, and (re)seeds the feeder. ``("threshold", gen, thr_milli)``
        updates the gate only.
        """
        kind = cmd[0]
        if kind == "switch":
            (_, gen, lpbh, miner_bytes, thr_milli, num_reads,
             annealing_time) = cmd
            self.generation = int(gen)
            self._num_reads = int(num_reads)
            self._annealing_time = float(annealing_time)
            self._energy_threshold_milli = int(thr_milli)
            self.cancel_inflight()
            if self._feeder is None:
                self._feeder = RandomIsingFeeder(
                    last_proof_block_hash=lpbh,
                    miner_bytes=miner_bytes,
                    nodes=self._nodes,
                    edges=self._edges,
                    buffer_size=self._feeder_buffer_size,
                )
            else:
                self._feeder.reseed(lpbh, miner_bytes)
        elif kind == "threshold":
            self.set_threshold(cmd[2])

    # -- streaming --------------------------------------------------------

    def _submit_one(self) -> None:
        model = self._feeder.pop_blocking()
        future, defect_info = self._miner.sampler.sample_ising_async(
            model.h, model.J,
            num_reads=self._num_reads,
            answer_mode="raw",
            annealing_time=self._annealing_time,
            label=f"{self._miner.sampler.job_label}_s{self._job_index}",
            nonce_seed=model.nonce,
        )
        self._pending[id(future)] = (
            model, future, defect_info, self._job_index, self.generation,
        )
        self._job_index += 1

    def _stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _gate_sampleset(self, raw_ss: Any, defect_info: Any) -> Any:
        if defect_info is None:
            return raw_ss
        best_qpu_energy = float(min(raw_ss.record.energy))
        threshold_energy = (
            self._energy_threshold_milli + self._precheck_margin_milli
        ) / 1000.0
        if _should_reconstruct(
            best_qpu_energy, defect_info.energy_offset, threshold_energy,
        ):
            return self._miner.sampler.reconstruct_full_sampleset(
                raw_ss, defect_info,
            )
        return _shift_energies(raw_ss, defect_info.energy_offset)

    def iter_results(self) -> Iterator[Tuple[IsingModel, dimod.SampleSet, int]]:
        """Yield ``(model, sampleset, submit_generation)`` in completion order.

        Runs until ``stop_event`` fires. Requires at least one ``switch``
        command applied first (so a feeder exists). Submissions are tagged
        with the generation they were made under; the driver discards any
        completion whose tag no longer matches the live generation.
        """
        while not self._stop():
            if self._feeder is None:
                return  # no round yet; driver waits on ctl_q before iterating
            while len(self._pending) < self._queue_depth and not self._stop():
                self._submit_one()
            completed_id = None
            while completed_id is None and not self._stop():
                for fid, (_, fut, _, _, _) in self._pending.items():
                    if fut.done():
                        completed_id = fid
                        break
                if completed_id is None:
                    time.sleep(0.005)
            if completed_id is None:
                return  # stopped while polling
            model, future, defect_info, _, submit_gen = self._pending.pop(
                completed_id,
            )
            # Throughput diagnostic (restored from the legacy
            # sample_ising_streaming path): periodically log the in-flight
            # depth + feeder state so operators can tell a D-Wave-bound stream
            # (in_flight stays near queue_depth) from a driver/feeder stall
            # (in_flight collapses). Logger falls back to the module logger so
            # the no-QPU context tests (fake miner without .logger) don't break.
            if self._job_index % 50 == 0 and self._feeder is not None:
                fstats = self._feeder.stats()
                logger = getattr(self._miner, "logger", init_logger)
                logger.info(
                    "[QPU] stream depth: in_flight=%d/%d feeder_ready=%d/%d "
                    "drained=%d wait_total=%.2fs",
                    len(self._pending), self._queue_depth,
                    fstats["ready"], fstats["buffer_size"],
                    fstats["drained_count"], fstats["pop_wait_total_s"],
                )
            sampleset = self._gate_sampleset(future.sampleset, defect_info)
            yield model, sampleset, submit_gen

    def cleanup(self) -> None:
        """Cancel in-flight work, stop the feeder, close the sampler."""
        self.cancel_inflight()
        if self._feeder is not None:
            try:
                self._feeder.stop()
            except Exception as exc:  # noqa: BLE001 — log; a leak must show
                init_logger.warning("ctx cleanup: feeder.stop failed: %s", exc)
            self._feeder = None
        sampler = getattr(self._miner, "sampler", None)
        if sampler is not None and hasattr(sampler, "close"):
            try:
                sampler.close()
            except Exception as exc:  # noqa: BLE001 — connection may leak
                init_logger.warning("ctx cleanup: sampler.close failed: %s", exc)


def build_persistent_context(
    *,
    miner_id: str,
    queue_depth: int,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    feeder_buffer_size: int,
    num_reads: int,
    annealing_time: float,
    energy_threshold_milli: int,
    precheck_margin_milli: int,
    solver_name: Optional[str] = None,
    region: Optional[str] = None,
    token: Optional[str] = None,
    topology: Optional[DWaveTopology] = None,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
) -> PersistentStreamContext:
    """Build the persistent QPU context (the expensive D-Wave connect).

    Constructs a connected :class:`DWaveMiner` ONCE; the feeder is created
    lazily on the first ``switch`` command (it needs the round seed) and
    reseeded thereafter. Runs ONLY in the stream-driver process — never in
    tests (it connects to D-Wave). Requires a topology to wire the chain
    topology to the QPU sampler.

    Args:
        topology: The chain topology for the QPU (required).

    Returns:
        A :class:`PersistentStreamContext` ready to receive ctl_q commands.

    Raises:
        ValueError: If topology is None.
    """
    if topology is None:
        raise ValueError(
            "the QPU stream driver requires a topology; the chain topology "
            "was not wired to build_persistent_context"
        )
    miner = DWaveMiner(
        miner_id=miner_id,
        queue_depth=queue_depth,
        solver_name=solver_name,
        region=region,
        token=token,
        topology=topology,
    )
    return PersistentStreamContext(
        miner=miner,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        annealing_time=annealing_time,
        energy_threshold_milli=energy_threshold_milli,
        precheck_margin_milli=precheck_margin_milli,
        queue_depth=queue_depth,
        stop_event=stop_event,
    )


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
    # QPU is the async-streaming backend: drive the stream in a separate
    # process so per-result processing never blocks the cloud pipeline.
    STREAMING_PUMP = True
    # The sampler + feeder live in the persistent stream-driver process (see
    # QPU/stream_driver.py + build_persistent_context); BaseMiner skips the
    # worker-side feeder and _ensure_driver spawns/reuses that process.
    DRIVER_OWNS_FEEDER = True

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
        # Connection config for the persistent stream-driver process. The
        # worker-side miner is built with connect=False (no sampler);
        # _ensure_driver passes these to build_persistent_context so the
        # driver process can construct its own connected DWaveMiner.
        self.solver_name = solver_name
        self.region = region
        self.token = token
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

        Diagnostic/tooling path only — the production miner streams via
        ``PersistentStreamContext`` (see ``build_persistent_context``); this
        method is retained for the QPU canary/sweep tools and their tests.

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

        def _stop_requested() -> bool:
            # stop_event captured once at stream start; stable for this
            # stream's lifetime (a new dispatch builds a fresh stream).
            return stop_event is not None and stop_event.is_set()

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

                if _should_reconstruct(
                    best_qpu_energy, defect_info.energy_offset, energy_threshold,
                ):
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
