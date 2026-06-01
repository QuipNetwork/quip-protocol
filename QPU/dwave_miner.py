"""QPU miner using D-Wave sampler for quantum mining."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import signal
import time
from typing import Dict, Iterator, List, Optional, Tuple, Any

init_logger = logging.getLogger(__name__)

from QPU.dwave_sampler import DWaveSamplerWrapper
from QPU.qpu_time_manager import QPUTimeManager, QPUTimeConfig
from shared.base_miner import BaseMiner, MidstreamBudget, _energy_to_milli
from shared.miner_types import BlockRequirements
from shared.ising_feeder import RandomIsingFeeder
from shared.stream_context import StreamContext
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies.dwave_topology import DWaveTopology


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
    solver_name: Optional[str] = None,
    region: Optional[str] = None,
    token: Optional[str] = None,
    topology: Optional[DWaveTopology] = None,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
) -> StreamContext:
    """Build the persistent QPU context (the expensive D-Wave connect).

    Constructs a connected :class:`DWaveMiner` ONCE (for its
    :class:`~QPU.dwave_sampler.DWaveSamplerWrapper`) then hands the sampler
    to the generic :class:`~shared.stream_context.StreamContext`. The feeder
    is built lazily on the first ``switch`` command (via the spec). Runs ONLY
    in the stream-driver process — never in tests (it connects to D-Wave).
    Requires a topology to wire the chain topology to the QPU sampler.

    Args:
        topology: The chain topology for the QPU (required).

    Returns:
        A :class:`~shared.stream_context.StreamContext` ready to receive
        ctl_q commands.

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
    return StreamContext(
        sampler=miner.sampler,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=0,
        sampler_kwargs={
            "queue_depth": queue_depth,
            "annealing_time": annealing_time,
            "energy_threshold_milli": energy_threshold_milli,
        },
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
    # The sampler + feeder live in the persistent stream-driver process (see
    # QPU/stream_driver.py + build_persistent_context); _ensure_driver
    # spawns/reuses that process and the worker keeps no feeder.

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
        # Stashed by _pre_mine_setup; exposed for tooling that builds an
        # in-process miner and passes the event to the sampler's pump.
        self._stop_event: Optional[multiprocessing.synchronize.Event] = None

        # Rate-limiter for the budget-pacing log line. Suppresses the
        # per-head "[QPU] Pacing block …" spam when the daily budget is
        # exhausted; the first entry and any bucket/interval change still
        # surface. reset() is called when mining resumes.
        self._pacing_rl = _PacingRateLimiter()
        # Separate rate-limiter for the in-loop (mid-dispatch) budget stall
        # log, so it doesn't share state with the per-head pacing limiter.
        self._inloop_pacing_rl = _PacingRateLimiter()

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
        # Stash the stop_event so tooling (e.g. qpu_consumer_livefire) that
        # builds a miner and checks _stop_event can still find it here.
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
                        "[QPU] Accumulating block %d - pool %.2fs/%.2fs cap, "
                        "need %.0fs buffer; ~%s to go.",
                        cur_index,
                        estimate.pool_us / 1e6,
                        estimate.pool_cap_us / 1e6,
                        self.time_manager.config.min_block_budget_seconds,
                        wait_str,
                    )
                return False

            # Mining can proceed — reset the pacing rate-limiter so the
            # next pacing episode is treated as a fresh entry.
            self._pacing_rl.reset()
            self.logger.info(
                "[QPU] Burst budget ready: pool %.2fs (cap %.2fs), "
                "est/block %.2fs (%s confidence)",
                estimate.pool_us / 1e6,
                estimate.pool_cap_us / 1e6,
                estimate.estimated_block_time_us / 1e6,
                estimate.confidence,
            )
        return True

    def _midstream_budget_ok(
        self, solution_number: int,
    ) -> Optional[MidstreamBudget]:
        """In-loop QPU budget check (called at the progress-log cadence).

        Consults the ``QPUTimeManager`` mid-dispatch so a head whose budget is
        exhausted *while mining* stops submitting NEW QPU work without waiting
        for the next head. Returns ``None`` when no budget is configured (so
        the base loop logs its plain progress line). On exhaustion, emits a
        rate-limited stall WARNING; the base loop then pauses the driver
        (drain-and-idle) but keeps consuming the in-flight queue.
        """
        if self.time_manager is None:
            return None
        # Read-only snapshot: unlike should_mine_block(), get_stats() does not
        # mutate blocks_skipped, so polling every progress interval while we
        # keep draining doesn't inflate the counter. The in-loop decision is
        # the literal "have we exceeded the proportional limit now" — the
        # next-block lookahead reserve belongs to the per-head gate, not here.
        stats = self.time_manager.get_stats()
        should_mine = stats.get("budget_remaining_seconds", 0.0) > 0.0
        if not should_mine:
            # Burst drained: end it so the pool must re-accumulate the full
            # buffer before the next head restarts a burst (no micro-bursts as
            # accrual creeps the pool back above 0).
            self.time_manager.end_burst()
            pool = stats.get("pool_seconds", 0.0)
            buf = stats.get("min_block_budget_seconds", 0.0)
            wait_s = stats.get("seconds_until_buffer", 0.0)
            wait_str = (
                f"{wait_s:.0f}s" if wait_s < 3600 else f"{wait_s / 3600:.1f}h"
            )
            if self._inloop_pacing_rl.should_log(
                now=time.monotonic(), wait_bucket=wait_str
            ):
                self.logger.warning(
                    "mine_work_item: burst drained — QPU pool at %.2fs; pausing "
                    "driver (in-flight will drain), re-accumulating to %.0fs "
                    "buffer (~%s)",
                    pool, buf, wait_str,
                )
        else:
            # Mining proceeding again — reset so the next stall logs fresh.
            self._inloop_pacing_rl.reset()
        return MidstreamBudget(should_mine=should_mine, stats=stats)

    def _participation_extra(self) -> Dict[str, Any]:
        """QPU adds the reservoir pool at dispatch to the participation marker.

        The snapshot is the QPU runway on hand when mining began for this
        solution # (>= ``min_block_budget``). Returns ``{}`` when no budget is
        configured.
        """
        if self.time_manager is None:
            return {}
        return {"budget_seconds": self.time_manager.get_stats()["pool_seconds"]}

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

    def _stream_factory_kwargs(self, sample_ctx, nodes):
        return {
            "miner_id": self.miner_id,
            "queue_depth": self.queue_depth,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
            "num_reads": sample_ctx["num_reads"],
            "annealing_time": sample_ctx["annealing_time"],
            "energy_threshold_milli": _energy_to_milli(
                sample_ctx["energy_threshold"],
            ),
            "solver_name": getattr(self, "solver_name", None),
            "region": getattr(self, "region", None),
            "token": getattr(self, "token", None),
            "topology": getattr(self, "topology", None),
        }

    def _finalize_sample(self, sampleset: Any, defect_info: Any) -> Any:
        """Reconstruct a reduced D-Wave sampleset to full topology (survivor-only).

        Called by ``BaseMiner._run_substrate_ratchet`` when a promising sample
        has fewer variables than the topology — i.e. the QPU driver stripped
        offline qubits before writing to the ring.  ``defect_info`` carries the
        fixed-spin assignments and energy offset needed for reconstruction.

        Args:
            sampleset: The reduced sampleset from the QPU driver.
            defect_info: :class:`~QPU.dwave_sampler.DefectInfo` with
                ``fixed_spins``, ``energy_offset``, and ``removed_edges``.

        Returns:
            Full-topology sampleset with all variables present and energies
            corrected.
        """
        return self.sampler.reconstruct_full_sampleset(  # ty:ignore[unresolved-attribute]
            sampleset, defect_info,
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
