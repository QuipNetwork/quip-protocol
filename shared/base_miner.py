"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import math
import multiprocessing
import multiprocessing.synchronize
import pickle
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import dimod
import numpy as np

from shared.energy_utils import (
    energy_to_difficulty as _energy_to_difficulty,
    DEFAULT_NUM_NODES,
    DEFAULT_NUM_EDGES,
)
from shared.mempool_types import MempoolJobContext
from shared.miner_types import BlockRequirements, IsingSample, MiningResult, Sampler
from shared.mining_attempt_log import AttemptLogger, SolutionStore
from shared.quantum_proof_of_work import (
    compute_solution_meta,
    evaluate_sampleset,
    pack_spins_hex,
)
from substrate.difficulty_decay import step_for_energy
from substrate.types import SubstrateMiningContext
from shared.work_context import (
    WorkContext,
    requirements_from_context,
)
from shared.ring_views import SampleView
from shared.proc_util import spawn_worker, terminate_join

# Global logger for this module
log = logging.getLogger(__name__)

# Emit a progress log line every N sampling attempts. 10 is a balance between
# noise (one line per attempt is too chatty for the slow miners) and
# observability (one line per minute on the typical CPU/GPU cadence).
PROGRESS_LOG_INTERVAL = 10

# Sentinel for "no preview emitted yet" in the preview-throttle state.
# Kept as a plain int so the strict-improvement comparison stays integer.
_MILLI_INF = 1 << 62


class _SharedRecord:
    """record-like facade exposing ``.sample`` / ``.energy`` ndarrays."""

    __slots__ = ("sample", "energy")

    def __init__(self, sample, energy):
        self.sample = sample
        self.energy = energy


class _SharedSampleSet:
    """Minimal sampleset facade over zero-copy shared-ring views.

    ``evaluate_sampleset`` / ``compute_solution_meta`` only read
    ``record.sample`` and ``record.energy``; QPU timing is carried
    out-of-band in the descriptor, so ``.info`` is an empty dict on this
    path.
    """

    __slots__ = ("record", "info")

    def __init__(self, sample, energy):
        self.record = _SharedRecord(sample, energy)
        self.info: Dict[str, Any] = {}


# Maximum number of work-tags remembered by _SetupAbortThrottle.
_SETUP_ABORT_TAG_LIMIT = 32

# Result-acquisition control signals returned by ``_acquire_result`` so the
# loop in ``mine_work_item`` can reproduce the original inline ``return None``
# / ``break`` / ``continue`` semantics without a helper hijacking control flow.
_ACQUIRE_OK = "ok"  # payload present; process this iteration
_ACQUIRE_STOP = "stop"  # stop_event fired -> caller returns None
_ACQUIRE_DONE = "done"  # stream exhausted (trailing None) -> caller breaks
_ACQUIRE_CONTINUE = "continue"  # recoverable sampling error -> caller continues


@dataclass
class _AcquireResult:
    """Outcome of one ``_acquire_result`` call.

    ``action`` is one of the ``_ACQUIRE_*`` constants. The payload fields are
    only meaningful when ``action`` is :data:`_ACQUIRE_OK`.

    Intentionally NOT frozen: ``mine_work_item`` clears ``sampleset`` (the
    zero-copy ``_SharedSampleSet`` view over a ring slot) on every path that
    reaches its ``finally`` so no live view survives into the ring's
    ``close_unlink`` — a lingering export raises ``BufferError`` there.
    """

    action: str
    nonce: Any = None
    salt: bytes = b""
    sampleset: Any = None
    qpu_access_time_us: Optional[int] = None
    # Ring slot backing ``sampleset`` on the driver path; the consumer
    # releases it after ``_finalize_iteration_logging`` reads the set.
    # ``None`` on the inline path (no shared ring).
    ring_slot: Optional[int] = None
    # Unpickled DefectInfo from the descriptor's 8th element (driver path).
    # ``None`` for all current code paths (CPU/GPU/Metal/inline); will be
    # populated by DWaveMiner once that path lands. Consumer-side only —
    # never crosses a process boundary after this point.
    defect_info: Any = None


@dataclass(frozen=True)
class MidstreamBudget:
    """Result of the in-loop QPU budget check (`_midstream_budget_ok`).

    ``should_mine`` is False once the daily budget is exhausted (forward QPU
    submission should pause). ``stats`` is the live budget snapshot (the
    ``QPUTimeManager.get_stats`` shape) for the progress log + telemetry push.
    """

    should_mine: bool
    stats: Dict[str, Any]


@dataclass
class _DispatchSetup:
    """Everything ``_setup_dispatch`` hands back to ``mine_work_item``.

    Bundles the per-dispatch loop inputs and the (optional) result-pump
    handles so the main method stays a thin loop driver. ``None`` from
    ``_setup_dispatch`` means ``_pre_mine_setup`` aborted and the caller
    returns ``None`` immediately (same as the original early return).
    """

    loop_state: "_MiningLoopState"
    is_substrate: bool
    sample_ctx: Dict[str, Any]
    num_reads: int
    num_sweeps: int
    # Driver path (STREAMING_PUMP + DRIVER_OWNS_FEEDER): the shared-sample
    # ring, the descriptor queue, the stream-driver process and its stop
    # event. All ``None`` on the inline (CPU/GPU/Metal) path.
    ring: Optional[Any]
    desc_q: Optional[Any]
    driver_proc: Optional[Any]
    driver_stop: Optional[Any]
    # Round generation this dispatch accepts; descriptors tagged with any
    # other generation are stale and dropped by the consumer.
    generation: int = 0


@dataclass
class StashEntry:
    """A stashed candidate plus its decay-derived win-time.

    ``decay_num`` is the first decay step whose threshold clears the
    candidate's submit floor; ``valid_at_block`` is the absolute chain block
    that step lands on (``last_proof_block + decay_num * epoch_length``).
    Ordering key for the win-time-ranked stash (Task 5).
    """

    decay_num: int
    valid_at_block: int
    result: MiningResult


@dataclass
class _MiningLoopState:
    """Per-dispatch state bundle for ``mine_work_item``'s loop helpers.

    Groups the locals the extracted per-iteration helpers
    (``_run_substrate_ratchet``, ``_run_mempool_eval``,
    ``_finalize_iteration_logging``) need so each stays under the
    positional-param limit. Constructed once per ``mine_work_item`` call
    and mutated in place (``top_k``, ``previewed_wintime``) by the
    ratchet helper, mirroring the original inline locals exactly.
    """

    requirements: BlockRequirements
    nodes: List[int]
    edges: List[Tuple[int, int]]
    prev_timestamp: int
    start_time: float
    # On-disk archive key (chain-global solution ordinal).
    solution_number_for_log: int
    # Internal coordination handle — used only for the anticipatory preview
    # channel, where the controller resolves preview→context by
    # (miner_id, dispatch_id). Never persisted.
    dispatch_id_for_log: int
    attempt_log: AttemptLogger
    solution_store: SolutionStore
    live_threshold_var: Optional[Any]
    top_k_cap: int
    top_k: List[StashEntry]
    # Preview throttle state: (valid_at_block, floor_milli) — emit on strict
    # lexicographic improvement so an earlier-winning candidate always fires a
    # preview even when its floor isn't lower.  Initialised to (_MILLI_INF,
    # _MILLI_INF) = "nothing previewed yet".
    previewed_wintime: Tuple[int, int]
    # Last-logged best win-time (decay_num, valid_at_block) for the "Best
    # Solution …" throttle. None = never logged.
    last_best_wintime: Optional[Tuple[int, int]] = None
    # Round generation (mirrors _DispatchSetup.generation) so the ratchet can
    # forward same-generation live-threshold decay updates to the driver.
    generation: int = 0
    # Round-constant decay inputs (substrate PoW path). When decay_schedule
    # is None (CPU/GPU backends, or a controller that didn't attach one) the
    # stash falls back to legacy energy ranking — see is_decay_ranked.
    decay_schedule: Optional[List[int]] = None
    last_proof_block: int = 0
    epoch_length: int = 0

    @property
    def is_decay_ranked(self) -> bool:
        """True when the stash should rank by decay win-time (Task 5)."""
        return self.decay_schedule is not None


class _SetupAbortThrottle:
    """Rate-limiter for the ``_pre_mine_setup`` returned-False warning.

    Logs the first occurrence of a given work-tag at WARNING; subsequent
    identical tags are silently dropped until evicted from the bounded
    cache.  The cache is an :class:`~collections.OrderedDict` used as a
    FIFO so the oldest tag is evicted first when the cap is hit.

    This keeps ``mine_work_item``'s "aborting attempt" warning visible for
    the first paced head in each round without spamming thousands of lines
    over a long pacing window.
    """

    def __init__(self, max_tags: int = _SETUP_ABORT_TAG_LIMIT) -> None:
        self._max = max_tags
        # OrderedDict preserves insertion order; we evict the first item
        # (oldest) when full so the set stays bounded.
        self._seen: "OrderedDict[str, None]" = OrderedDict()

    def should_log(self, tag: str) -> bool:
        """Return True (and record tag) if the caller should emit a log line."""
        if tag in self._seen:
            return False
        # Evict oldest entry if at capacity before inserting the new one.
        while len(self._seen) >= self._max:
            self._seen.popitem(last=False)
        self._seen[tag] = None
        return True

    def __len__(self) -> int:
        return len(self._seen)


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

        # Feeder slot. ``mine_work_item`` writes here just before the
        # loop and clears it in ``finally``. Subclasses (GPU/QPU/Metal)
        # used to assign in their own ``_pre_mine_setup``; now the base
        # class is authoritative, but their pipelines still read this
        # attribute via ``self._feeder`` (e.g. streaming samplers and
        # SIGTERM handlers), so it must exist before any subclass runs.
        self._feeder: Optional[Any] = None

        # Count of QPU results dropped under result-queue backpressure.
        # Reset per dispatch by mine_work_item; logged at dispatch end.
        self._dropped_results: int = 0

        # Persistent driver-path handles (DRIVER_OWNS_FEEDER). ``_ensure_driver``
        # spawns ONE stream-driver process and keeps it alive across dispatches;
        # ``_close_driver`` (miner shutdown) is the only thing that reaps it and
        # close-unlinks the ring. All ``None`` on the inline (CPU/GPU/Metal) path
        # and before the first driver dispatch.
        self._ring: Optional[Any] = None
        self._desc_q: Optional[Any] = None
        self._ctl_q: Optional[Any] = None
        self._driver_stop: Optional[Any] = None
        self._driver_proc: Optional[Any] = None
        # ProblemView written by _mempool_feeder_spec for the current mempool
        # order. The worker owns the shared memory; the driver reads it on the
        # switch. Replaced (and previous freed) on each mempool dispatch;
        # close-unlinked in _close_driver on shutdown.
        self._mempool_problem_view: Optional[Any] = None
        # (max_rows, max_cols) the persistent ring was sized for; a dispatch
        # whose dims differ forces a driver respawn (rare — num_reads and the
        # topology are stable within a miner's life).
        self._ring_dims: Optional[Tuple[int, int]] = None
        # Monotonic per-miner round generation. Bumped once per dispatch in
        # _setup_dispatch; tags every switch_round and every descriptor so a
        # superseded round's results are dropped. Process-local; never persisted.
        self._generation: int = 0
        # Last live-threshold (milli) forwarded to the driver, so decay updates
        # are only sent on change. Reset on driver (re)spawn.
        self._last_forwarded_threshold_milli: Optional[int] = None
        # Generation for which a budget-exhaustion ("pause", gen) has already
        # been sent to the driver, so the in-loop gate sends it at most once
        # per dispatch. Reset when a new dispatch bumps the generation.
        self._budget_paused_generation: Optional[int] = None

        # In-flight QPU job count forwarded to the stream-driver factory.
        # QPU subclasses override; default 0 keeps the driver factory's
        # ``self.queue_depth`` reference safe for any STREAMING_PUMP subclass.
        self.queue_depth: int = 0

        # Throttle for the "_pre_mine_setup returned False" warning so
        # pacing during a budget exhaustion window doesn't produce one
        # WARNING line per head (~every 6 s). The first occurrence for
        # each work-tag still surfaces; repeats are suppressed until the
        # tag is evicted from the bounded cache.
        self._setup_abort_throttle = _SetupAbortThrottle()

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

    # --- Feeder buffer size (override in subclasses) ---
    # ``BaseMiner.mine_work_item`` builds a feeder via
    # ``context.make_feeder(nodes, edges, buffer_size=FEEDER_BUFFER_SIZE)``
    # right before the mining loop. CPU SA is single-threaded and doesn't
    # need much pipelining; GPU and QPU miners override this with larger
    # values so the background generator stays ahead of the kernel/QPU
    # pipeline (see GPUMiner / DWaveMiner overrides).
    FEEDER_BUFFER_SIZE: int = 4

    # --- Substrate ratchet: bounded top-K stash ---
    # The PoW path stashes the K best-energy candidates across iterations
    # so the submit gate has multiple shots when the chain's live decay
    # eases. Only the lowest-energy candidate is ever submitted, but the
    # spares carry insurance (best may fail evaluate_sampleset re-checks
    # under tighter thresholds, or its ``submit_floor_energy`` may sit
    # above the live floor while a worse-energy candidate's floor clears).
    # First K iters post-process unconditionally to fill the stash; after
    # that the ratchet gates on the heap's worst-energy entry.
    TOP_K_STORE: int = 5

    # --- Out-of-band result pump (streaming backends only) ---
    # When True, mine_work_item runs _sample_batch on a background pump
    # thread and consumes results off a bounded queue, so per-result
    # processing never blocks the QPU pipeline. CPU/GPU/Metal keep the
    # inline single-shot / batch path (STREAMING_PUMP stays False).
    STREAMING_PUMP: bool = False
    # When True (QPU), the sampler + feeder live in a separate stream-driver
    # PROCESS (see QPU/stream_driver.py); ``_setup_dispatch`` skips building a
    # worker-side feeder and ``_ensure_driver`` spawns the driver instead
    # of a pump thread. CPU/GPU keep this False and run the in-worker feeder.
    DRIVER_OWNS_FEEDER: bool = False
    # Dotted ``module:attr`` of the stream factory the driver process
    # resolves and calls. Points at ``QPU.dwave_miner:build_persistent_context``
    # for the QPU path; tests swap it for a fake so the driver path can be
    # exercised without a QPU connection.
    STREAM_FACTORY_DOTTED: str = "QPU.dwave_miner:build_persistent_context"
    # Bounded result queue depth. Sized to the in-flight QPU job count so
    # the consumer has headroom equal to what the cloud holds; on full the
    # pump drops the newest result (rare safety valve) and counts it.
    RESULT_QUEUE_MAXSIZE: int = 32

    # --- Adaptive parameter bounds (override in subclasses) ---
    # SA/GPU miners use sweeps + reads; QPU miners use annealing_time + reads.
    ADAPT_MIN_SWEEPS: int = 64
    ADAPT_MAX_SWEEPS: int = 4096
    ADAPT_MIN_READS: int = 64
    ADAPT_MAX_READS: int = 512
    # When > 0, reads range is derived from min_solutions instead of fixed
    # bounds: min = max(min_solutions * factor, ADAPT_MIN_READS),
    #         max = max(min_solutions * factor, ADAPT_MAX_READS).
    ADAPT_READS_SOLUTION_MIN_FACTOR: int = 0
    ADAPT_READS_SOLUTION_MAX_FACTOR: int = 0
    # Floor applied to final num_reads as max(num_reads, min_solutions * N).
    # Most miners use 1; Modal uses 3; SA uses 0 (no floor).
    ADAPT_READS_SOLUTION_FLOOR_FACTOR: int = 1
    # QPU miners set these instead of sweeps bounds
    ADAPT_MIN_ANNEALING_TIME: float = 0.0
    ADAPT_MAX_ANNEALING_TIME: float = 0.0
    ADAPT_MIN_BONUS_READS: int = 0
    ADAPT_MAX_BONUS_READS: int = 0
    # Extra keys merged into the returned dict (e.g. num_sweeps_per_beta)
    ADAPT_EXTRA_PARAMS: Dict[str, Any] = {}
    # Calibrated c_range for this miner's energy curve.
    # Override in subclasses with empirically measured values.
    # None = use DEFAULT_C_RANGE from energy_utils (SA baseline).
    ADAPT_C_RANGE: Optional[Tuple[float, float]] = None

    @classmethod
    def energy_to_difficulty(
        cls,
        target_energy: float,
        num_nodes: int = DEFAULT_NUM_NODES,
        num_edges: int = DEFAULT_NUM_EDGES,
    ) -> float:
        """Convert energy target to [0, 1] difficulty for this miner.

        Uses the miner's ADAPT_C_RANGE if set, otherwise falls
        back to the SA baseline. Override in subclasses for
        fundamentally different difficulty mappings.
        """
        kwargs = {}
        if cls.ADAPT_C_RANGE is not None:
            kwargs['c_range'] = cls.ADAPT_C_RANGE
        return _energy_to_difficulty(
            target_energy,
            num_nodes=num_nodes,
            num_edges=num_edges,
            **kwargs,
        )

    @classmethod
    def adapt_parameters(
        cls,
        difficulty_energy: float,
        min_diversity: float,
        min_solutions: int,
        num_nodes: int = DEFAULT_NUM_NODES,
        num_edges: int = DEFAULT_NUM_EDGES,
    ) -> dict:
        """Calculate adaptive mining parameters based on difficulty.

        Uses GSE-based difficulty with linear interpolation. Each miner
        subclass declares its own parameter bounds as class attributes.

        Can be called on an instance (``self.adapt_parameters(...)``) or on
        the class directly (``SimulatedAnnealingMiner.adapt_parameters(...)``).

        Args:
            difficulty_energy: Target energy threshold.
            min_diversity: Minimum solution diversity (reserved).
            min_solutions: Minimum number of valid solutions required.
            num_nodes: Number of nodes in topology.
            num_edges: Number of edges in topology.

        Returns:
            Dictionary with miner-specific parameter keys.
        """
        difficulty = cls.energy_to_difficulty(
            difficulty_energy,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

        # QPU path: annealing_time + bonus reads
        if cls.ADAPT_MAX_ANNEALING_TIME > 0:
            annealing_time = (
                cls.ADAPT_MIN_ANNEALING_TIME
                + difficulty
                * (cls.ADAPT_MAX_ANNEALING_TIME - cls.ADAPT_MIN_ANNEALING_TIME)
            )
            bonus_reads = int(
                cls.ADAPT_MIN_BONUS_READS
                + difficulty
                * (cls.ADAPT_MAX_BONUS_READS - cls.ADAPT_MIN_BONUS_READS)
            )
            result: dict = {
                'num_reads': min_solutions + bonus_reads,
                'annealing_time': annealing_time,
            }
        else:
            # SA / GPU path: sweeps + reads
            num_sweeps = max(
                cls.ADAPT_MIN_SWEEPS,
                int(difficulty * cls.ADAPT_MAX_SWEEPS),
            )

            # Reads range: solution-factor or fixed bounds
            if cls.ADAPT_READS_SOLUTION_MIN_FACTOR > 0:
                min_reads = max(
                    int(min_solutions) * cls.ADAPT_READS_SOLUTION_MIN_FACTOR,
                    cls.ADAPT_MIN_READS,
                )
                max_reads = max(
                    int(min_solutions) * cls.ADAPT_READS_SOLUTION_MAX_FACTOR,
                    cls.ADAPT_MAX_READS,
                )
            else:
                min_reads = cls.ADAPT_MIN_READS
                max_reads = cls.ADAPT_MAX_READS

            num_reads = max(min_reads, int(difficulty * max_reads))

            floor = min_solutions * cls.ADAPT_READS_SOLUTION_FLOOR_FACTOR
            result = {
                'num_sweeps': num_sweeps,
                'num_reads': max(num_reads, floor),
            }

        if cls.ADAPT_EXTRA_PARAMS:
            result.update(cls.ADAPT_EXTRA_PARAMS)

        return result


    # ------------------------------------------------------------------
    # Substrate-mode entry point (Phase 3)
    # ------------------------------------------------------------------

    def mine_work_item(
        self,
        context: WorkContext,
        stop_event: multiprocessing.synchronize.Event,
        preview_cb: Optional[Any] = None,
        budget_cb: Optional[Any] = None,
        participating_cb: Optional[Any] = None,
        **kwargs,
    ) -> Optional[MiningResult]:
        """Protocol-neutral mining loop.

        Accepts either work-source flavor:

        - ``SubstrateMiningContext`` (PoW path) — identity is the SCALE
          ``AccountId32`` in ``miner_account_bytes``; the loop derives a
          fresh nonce per iteration via
          ``derive_nonce(last_proof_block_hash, miner, salt)`` and
          regenerates ``(h, J)`` from it (`generate_ising_model_from_nonce`).
          The chain checks this exact derivation in ``submit_proof``.
        - ``MempoolJobContext`` (mempool path) — identity is the
          ``order_id``; the Ising problem is carried directly in the job
          order (``h_values``, ``j_values`` as i32 millivalues). No nonce
          derivation; the chain only validates submitted spins solve the
          stored model.

        Both paths share:

        - difficulty / quality floors converted to ``BlockRequirements`` at
          this seam (see ``shared.work_context.requirements_from_context``).
          No per-loop decay — the controller cancels work on head or
          job-state change.
        - the same sampler, adaptive param, evaluate-sampleset, top-attempts
          surface used by subclasses (CPU/GPU/QPU).
        - batch sampling (``_sample_batch``) is intentionally bypassed
          (see Phase 4 follow-on note on feeder identity).

        Args:
            context: Either a ``SubstrateMiningContext`` (PoW) or
                ``MempoolJobContext`` (mempool job).
            stop_event: Worker cancellation event. The controller sets
                this on new-head, deregistration, job-expiry, or shutdown.
            preview_cb: Optional callable invoked (substrate/PoW path only)
                whenever the best-by-floor stashed candidate improves —
                i.e. a strictly lower ``submit_floor_energy`` enters/leads
                ``top_k``. Receives a lightweight, picklable payload dict
                (see the call site) carrying enough for the controller to
                encode+submit the candidate later. Default ``None`` = no-op
                so existing callers are unaffected. A failing callback never
                breaks mining (wrapped in try/except, logged at debug).
            budget_cb: Optional callable invoked at the progress-log cadence
                with the live QPU budget stats dict (the
                ``QPUTimeManager.get_stats`` shape) so the controller can
                surface live usage on telemetry. Default ``None`` = no-op. A
                failing callback never breaks mining.
            participating_cb: Optional callable invoked exactly once per
                accepted dispatch (after ``_pre_mine_setup`` passes its gate,
                so a budget-starved QPU dispatch that aborts never fires it).
                Receives ``(solution_number, extra_dict)`` where ``extra_dict``
                is ``self._participation_extra()`` (QPU adds ``budget_seconds``).
                Drives the controller's write-once participation remark.
                Default ``None`` = no-op; a failing callback never breaks mining.
            **kwargs: Forwarded to ``_pre_mine_setup``.

        Returns:
            A ``MiningResult`` if a valid solution is found before the
            stop event fires; ``None`` otherwise.
        """
        setup = self._setup_dispatch(context, stop_event, **kwargs)
        if setup is None:
            return None  # _pre_mine_setup aborted (e.g. QPU budget exhausted)
        loop_state = setup.loop_state
        is_substrate = setup.is_substrate
        sample_ctx = setup.sample_ctx
        desc_q = setup.desc_q

        # Dispatch accepted (gate passed): emit the write-once participation
        # signal. For QPU this only runs once the reservoir buffer is reached,
        # since a budget-starved dispatch aborts in _setup_dispatch above.
        if participating_cb is not None:
            try:
                participating_cb(
                    loop_state.solution_number_for_log,
                    self._participation_extra(),
                )
            except Exception as exc:  # noqa: BLE001 — observability path
                self.logger.debug("participating_cb failed (ignored): %s", exc)

        progress = 0
        try:
            while self.mining and not stop_event.is_set():
                # Each iteration sources one (nonce, salt, sampleset):
                # streaming backends via ``_sample_batch`` (which pulls
                # models from the feeder internally), or the single-shot
                # path which pops one model. The PoW feeder derives a
                # fresh ``salt -> nonce -> (h, J)`` per model in a
                # background process; the mempool feeder cycles the
                # order's stored ``(h, J)``.
                preprocess_start = time.time()
                self.current_stage = 'preprocessing'
                self.current_stage_start = preprocess_start

                # Source one (nonce, salt, sampleset) plus the per-iteration
                # QPU access time. The returned signal directs the loop's
                # stop / stream-exhausted / sampling-error control flow.
                acquired = self._acquire_result(
                    stop_event, desc_q, preprocess_start,
                    sample_ctx=sample_ctx, driver_proc=setup.driver_proc,
                    generation=setup.generation,
                )
                if acquired.action == _ACQUIRE_STOP:
                    return None
                if acquired.action == _ACQUIRE_DONE:
                    break  # stream exhausted; exit the loop
                if acquired.action == _ACQUIRE_CONTINUE:
                    continue
                nonce = acquired.nonce
                salt = acquired.salt
                sampleset = acquired.sampleset
                qpu_access_time_us = acquired.qpu_access_time_us
                defect_info = acquired.defect_info

                sampleset = self._post_sample(sampleset)
                if stop_event.is_set():
                    # Drop the shared-ring view before teardown: release the
                    # slot and clear both the local and the _AcquireResult
                    # references so close_unlink doesn't trip BufferError.
                    if acquired.ring_slot is not None and self._ring is not None:
                        self._ring.release(acquired.ring_slot)
                    sampleset = None
                    acquired.sampleset = None
                    return None

                postprocess_start = time.time()
                self.current_stage = 'postprocessing'
                self.current_stage_start = postprocess_start

                self.timing_stats['total_samples'] += len(
                    sampleset.record.energy,
                )
                self.timing_stats['blocks_attempted'] += 1

                # Per-iteration log fields (filled below per code path).
                attempt_log_kwargs = self._init_attempt_log_kwargs(
                    loop_state.solution_number_for_log, progress, nonce, salt,
                    sampleset,
                )

                if is_substrate:
                    result = self._run_substrate_ratchet(
                        loop_state, sampleset, nonce, salt, postprocess_start,
                        preview_cb=preview_cb,
                        attempt_log_kwargs=attempt_log_kwargs,
                        defect_info=defect_info,
                    )
                else:
                    result = self._run_mempool_eval(
                        loop_state, sampleset, nonce, salt, postprocess_start,
                        attempt_log_kwargs=attempt_log_kwargs,
                        defect_info=defect_info,
                    )

                self._finalize_iteration_logging(
                    loop_state, sampleset, nonce, salt, progress,
                    preprocess_start=preprocess_start,
                    qpu_access_time_us=qpu_access_time_us,
                    attempt_log_kwargs=attempt_log_kwargs,
                )

                # Driver path: the consumer is done with the shared-ring
                # views (compute_solution_meta / evaluate copied the top-5
                # out as Python lists), so return the slot to the free-list
                # and drop the local view before teardown — otherwise a
                # lingering export would trip BufferError on close_unlink.
                if acquired.ring_slot is not None and self._ring is not None:
                    self._ring.release(acquired.ring_slot)
                sampleset = None
                # Clear the view held by ``acquired`` too: on the win / stop-
                # after-result early returns below, ``acquired`` survives into
                # the ``finally``; a live _SharedSampleSet view would make the
                # ring's close_unlink raise BufferError.
                acquired.sampleset = None

                if result:
                    # Post-evaluation cancel check. evaluate_sampleset can
                    # take meaningful time on dense graphs; if cancel
                    # raced with a valid result, return None so the next
                    # dispatch can decide what to do — the controller is
                    # already moving on. Without this check, a result
                    # produced after stop_event was set surfaces as
                    # "fresh" against the new dispatch and may submit a
                    # proof against a stale context.
                    if stop_event.is_set():
                        self.logger.info(
                            "mine_work_item: valid result produced after "
                            "cancel; discarding (stop_event set)"
                        )
                        return None
                    # Use result.nonce / result.salt — the submitted
                    # candidate — not the loop-local nonce/salt, which may
                    # belong to a different iteration when the submit gate
                    # returns a stashed top-k entry.
                    result_nonce_disp = f"0x{result.nonce.hex()[:16]}..."
                    self.logger.info(
                        f"[work-item {_work_tag(context)}] mined! "
                        f"nonce={result_nonce_disp} "
                        f"salt=0x{result.salt.hex()[:8]}... "
                        f"energy={result.energy:.2f} "
                        f"solutions={result.num_valid} "
                        f"diversity={result.diversity:.3f} "
                        f"attempt_time={result.mining_time:.2f}s "
                        f"total_time={time.time() - loop_state.start_time:.2f}s"
                    )
                    return result

                progress += 1
                if progress % PROGRESS_LOG_INTERVAL == 0:
                    # `self.top_attempts` is intentionally not maintained in
                    # substrate mode — no best-energy field to surface here.
                    budget = self._midstream_budget_ok(
                        loop_state.solution_number_for_log,
                    )
                    if budget is None:
                        self.logger.info(
                            "mine_work_item progress: %d attempts | "
                            "sweeps=%d reads=%d",
                            progress, setup.num_sweeps, setup.num_reads,
                        )
                    else:
                        s = budget.stats
                        self.logger.info(
                            "mine_work_item progress: %d attempts | "
                            "sweeps=%d reads=%d | qpu_pool=%.2fs/%.2fs cap "
                            "(buffer %.0fs) burst=%s used=%.2fs skipped=%d",
                            progress, setup.num_sweeps, setup.num_reads,
                            s.get("pool_seconds", 0.0),
                            s.get("pool_cap_seconds", 0.0),
                            s.get("min_block_budget_seconds", 0.0),
                            s.get("burst_active", False),
                            s.get("cumulative_used_seconds", 0.0),
                            s.get("blocks_skipped", 0),
                        )
                        if budget_cb is not None:
                            try:
                                budget_cb(s)
                            except Exception as exc:  # noqa: BLE001
                                self.logger.debug(
                                    "budget_cb failed (ignored): %s", exc,
                                )
                        if not budget.should_mine:
                            # Budget exhausted: stop the driver submitting NEW
                            # work (idempotent per dispatch), but KEEP consuming
                            # so the draining in-flight attempts still flow
                            # through the decay/stash/submit ratchet — a winner
                            # already paid for can still be submitted.
                            self._pause_driver(loop_state.generation)
            self.logger.info("mine_work_item: stopped, no valid result")
            return None
        finally:
            self._teardown_dispatch()

    @staticmethod
    def _init_attempt_log_kwargs(
        solution_number: int,
        progress: int,
        nonce: Any,
        salt: bytes,
        sampleset: Any,
    ) -> Dict[str, Any]:
        """Build the per-iteration attempt-log kwargs (pre-eval defaults).

        Per-path fields (threshold, num_valid, result_kind, ...) are filled
        in later by the eval helpers. ``solution_number`` is the on-disk
        archive key (chain-global ordinal of the solution being mined).
        """
        return {
            "solution_number": solution_number,
            "iter_num": progress + 1,
            "nonce_hex": (
                f"0x{nonce.hex()}"
                if isinstance(nonce, (bytes, bytearray))
                else hex(int(nonce))
            ),
            "salt_hex": f"0x{salt.hex()}",
            "best_energy_milli": int(
                float(np.min(sampleset.record.energy)) * 1000
            ),
            "num_samples": len(sampleset.record.energy),
            "post_processed": False,
            "stored_as_best": False,
            "result_kind": "rejected",
        }

    def _setup_dispatch(
        self,
        context: WorkContext,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> Optional[_DispatchSetup]:
        """One-time per-dispatch setup for ``mine_work_item``.

        Runs ``_pre_mine_setup`` (returning ``None`` to abort, exactly as the
        original early ``return None`` did), adapts params, builds the
        ``_MiningLoopState`` and sampling context, creates the feeder, and
        starts the result pump for streaming backends. Behaviour matches the
        original inline setup block.
        """
        self.mining = True
        self.top_attempts = []
        start_time = time.time()
        self.current_round_attempted = True
        self._log_work_start(context)

        # ``BlockRequirements`` synthesized from whichever context flavor
        # this is — PoW difficulty or mempool quality floors. The bridges
        # below preserve the existing ``_pre_mine_setup(prev_block,
        # node_info, ...)`` signature so subclass hooks (notably QPU's
        # budget check) keep working unchanged.
        requirements = requirements_from_context(context)
        bridge_prev_block = _BridgePrevBlock.from_work_context(context)
        bridge_node_info = _BridgeNodeInfo.from_work_context(context)

        # ``prev_timestamp`` is used by the legacy difficulty-decay code path
        # (which we don't run here) and propagated into ``MiningResult`` for
        # telemetry. Pass 0 — Phase 6 will swap to a real chain-derived
        # timestamp once the controller plumbs one through.
        prev_timestamp = 0

        # One-time miner-specific initialisation. Bridge objects expose
        # ``.header.index``, ``.hash``, ``.miner_id`` so existing hooks work.
        if not self._pre_mine_setup(
            bridge_prev_block, bridge_node_info, requirements,
            prev_timestamp, stop_event, **kwargs,
        ):
            # QPU budget exhausted, GPU device init failed, etc. The legacy
            # `mine_block` swallowed this case silently; here we surface it
            # so the worker's resp_q sentinel actually means "tried and
            # got nothing", not "couldn't start".
            # Rate-limited: only log the first occurrence for each work-tag
            # so pacing during budget exhaustion doesn't produce a WARNING
            # line on every chain head (~every 6 s).
            tag = _work_tag(context)
            if self._setup_abort_throttle.should_log(tag):
                self.logger.warning(
                    "mine_work_item: _pre_mine_setup returned False, aborting "
                    "attempt for %s", tag,
                )
            # A budget-exhausted head must also idle a driver still running the
            # PRIOR round — otherwise it keeps burning QPU on a generation the
            # worker no longer consumes. Drain-and-idle (no cancel); the next
            # eligible head's switch resumes it.
            self._pause_driver(self._generation)
            return None

        # Topology comes from the chain snapshot, not the local sampler.
        # The Phase 4 controller validates `self.sampler.topology_hash ==
        # context.topology_hash` at startup, but threading context's nodes
        # / edges through here directly means a misconfigured sampler
        # surfaces as a clear sampling exception rather than silently
        # producing proofs the chain rejects for InvalidTopology.
        nodes = list(context.nodes)
        edges = list(context.edges)
        params = self._adapt_mining_params(requirements, nodes, edges)
        self.logger.info(f"{self.miner_id} - adaptive params: {params}")

        current_num_sweeps = params.get('num_sweeps', 64)
        num_reads = params.get('num_reads', 100)
        extra_params = {
            k: v for k, v in params.items()
            if k not in ('num_reads', 'num_sweeps')
        }

        # Substrate ratchet state. Only the PoW path opts into the
        # decay-aware "store best, submit when chain catches up" loop;
        # mempool jobs have hard quality floors that don't decay so
        # they keep the strict-energy behaviour.
        is_substrate = isinstance(context, SubstrateMiningContext)
        live_threshold_var = getattr(self, '_live_max_energy_milli', None)
        # Seed the worker's live-threshold view with the snapshot value
        # in case the controller hasn't written one yet (e.g. tests, or
        # the very first dispatch before the head-handler runs).
        if is_substrate and live_threshold_var is not None:
            with live_threshold_var.get_lock():
                if live_threshold_var.value == 0:
                    live_threshold_var.value = context.difficulty.max_energy_milli
        # Win-time/energy-ordered stash; len <= TOP_K_STORE. ``top_k[0]`` is
        # the best candidate; the worst entry is the eviction target.
        top_k: List[StashEntry] = []
        top_k_cap: int = self.TOP_K_STORE
        # Anticipatory-submission preview throttle. Tracks
        # (valid_at_block, floor_milli) of the last-emitted preview;
        # we emit on a strict lexicographic improvement so an
        # earlier-winning candidate always fires a preview even if its
        # floor isn't lower.  ``(_MILLI_INF, _MILLI_INF)`` = nothing
        # previewed yet.
        previewed_wintime: Tuple[int, int] = (_MILLI_INF, _MILLI_INF)
        # Lazily build the per-worker attempt log. Mempool jobs share
        # the same writer for cross-mode forensics; the ``result_kind``
        # field on each row distinguishes the two paths.
        attempt_log: AttemptLogger = (
            getattr(self, '_attempt_logger', None)
            or AttemptLogger(self.miner_id, miner_type=self.miner_type)
        )
        if not hasattr(self, '_attempt_logger'):
            self._attempt_logger = attempt_log
        # SolutionStore is the per-worker writer for top-5 spin
        # archives. Only called when an iter is stored or submitted —
        # see the call sites further down. Same lazy-cache idiom as
        # attempt_log so we re-use one instance across dispatches.
        solution_store: SolutionStore = (
            getattr(self, '_solution_store', None)
            or SolutionStore(self.miner_id)
        )
        if not hasattr(self, '_solution_store'):
            self._solution_store = solution_store
        # On-disk archive key = the chain-global solution number. dispatch_id
        # stays internal (controller↔worker pairing); it is NOT persisted.
        # Solution 0 is the "unresolved" bucket (controller couldn't read the
        # WinningSolutions count and had no prior — rare).
        solution_number = getattr(self, '_current_solution_number', None)
        solution_number_for_log = int(solution_number) if solution_number is not None else 0
        dispatch_id_for_log = int(getattr(self, '_current_dispatch_id', 0))

        loop_state = _MiningLoopState(
            requirements=requirements,
            nodes=nodes,
            edges=edges,
            prev_timestamp=prev_timestamp,
            start_time=start_time,
            solution_number_for_log=solution_number_for_log,
            dispatch_id_for_log=dispatch_id_for_log,
            attempt_log=attempt_log,
            solution_store=solution_store,
            live_threshold_var=live_threshold_var,
            top_k_cap=top_k_cap,
            top_k=top_k,
            previewed_wintime=previewed_wintime,
            decay_schedule=getattr(context, "decay_schedule", None),
            last_proof_block=int(getattr(context, "last_proof_block", 0) or 0),
            epoch_length=int(getattr(context, "epoch_length", 0) or 0),
        )

        # Build the feeder for this attempt. Each context flavor picks
        # the right backing implementation (RandomIsingFeeder for PoW,
        # FixedIsingFeeder for mempool). We own the lifecycle here —
        # ``finally`` stops and clears it so a subsequent dispatch starts
        # from a clean state. Streaming-capable backends (QPU async
        # pipeline, GPU multi-problem dispatch) read ``self._feeder`` from
        # ``_sample_batch`` and keep ``queue_depth`` jobs in flight — that
        # is what gives the QPU its throughput. Backends without batch
        # streaming pop the feeder one model at a time (see the loop).
        if self.DRIVER_OWNS_FEEDER:
            # The stream-driver process builds its own feeder (RandomIsingFeeder
            # for PoW, FixedIsingFeeder for mempool via ProblemView); the worker
            # keeps no feeder so the model is derived in exactly one place.
            # Capture the construction inputs into sample_ctx below.
            self._feeder = None
        else:
            self._feeder = context.make_feeder(
                nodes, edges, buffer_size=self.FEEDER_BUFFER_SIZE,
            )

        # Positional args for the legacy ``_sample_batch`` signature. The
        # QPU/GPU streaming impls ignore them — their feeder already
        # encapsulates the round seed and miner identity — but the base
        # contract requires (prev_hash, miner_id, cur_index). The driver
        # path additionally forwards the feeder-construction inputs and the
        # QPU dispatch knobs (annealing_time / energy_threshold come from
        # _adapt_mining_params' extra params) so _ensure_driver can hand
        # them to the stream-driver process.
        sample_ctx = {
            "prev_hash": bridge_prev_block.hash,
            "miner_id": bridge_node_info.miner_id,
            "cur_index": bridge_prev_block.header.index,
            "nodes": nodes,
            "edges": edges,
            "num_reads": num_reads,
            "num_sweeps": current_num_sweeps,
            "extra": extra_params,
            "annealing_time": extra_params.get("annealing_time"),
            "energy_threshold": extra_params.get("energy_threshold"),
            "last_proof_block_hash": getattr(
                context, "last_proof_block_hash", None,
            ),
            "miner_bytes": getattr(context, "miner_account_bytes", None),
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
        }

        self._dropped_results = 0
        generation = 0
        if self._ensure_driver(sample_ctx):
            # New round: bump the generation and tell the persistent driver to
            # switch (reseed feeder, cancel in-flight, set threshold). The
            # driver keeps its D-Wave connection — no reconnect, no respawn.
            self._generation += 1
            generation = self._generation
            # Fresh round: clear the budget-pause tracker so the in-loop gate
            # can send a new ("pause", gen) for this generation if needed.
            self._budget_paused_generation = None
            # Streaming backends that do no reconstruction gating (Metal/SA)
            # have no energy_threshold/annealing_time in their adapted params;
            # default both rather than crash on _energy_to_milli(None). The
            # switch also carries num_sweeps so the persistent driver re-adapts
            # the sweep count per round (Metal); QPU ignores num_sweeps.
            thr = sample_ctx["energy_threshold"]
            threshold_milli = _energy_to_milli(thr) if thr is not None else 0
            anneal = sample_ctx["annealing_time"]
            # The 9th element is the feeder spec the generic StreamContext builds
            # its feeder from (PoW: random-model derivation seed; mempool: a
            # ProblemView slot written by the worker with the order's fixed h/J).
            if is_substrate:
                feeder_spec = (
                    "pow", sample_ctx["last_proof_block_hash"],
                    sample_ctx["miner_bytes"],
                )
            else:
                feeder_spec = self._mempool_feeder_spec(context)
            try:
                self._ctl_q.put(
                    ("switch", generation, sample_ctx["last_proof_block_hash"],
                     sample_ctx["miner_bytes"], threshold_milli,
                     int(sample_ctx["num_reads"]),
                     float(anneal) if anneal is not None else 0.0,
                     int(sample_ctx["num_sweeps"]), feeder_spec),
                )
            except Exception as exc:  # noqa: BLE001 — surface, don't hang
                self.logger.error("switch_round send failed: %s", exc)
                return None
            self._last_forwarded_threshold_milli = threshold_milli
        loop_state.generation = generation
        return _DispatchSetup(
            loop_state=loop_state,
            is_substrate=is_substrate,
            sample_ctx=sample_ctx,
            num_reads=num_reads,
            num_sweeps=current_num_sweeps,
            ring=self._ring,
            desc_q=self._desc_q,
            driver_proc=self._driver_proc,
            driver_stop=self._driver_stop,
            generation=generation,
        )

    def _stream_factory_kwargs(
        self, sample_ctx: Dict[str, Any], nodes: List[int]
    ) -> Dict[str, Any]:
        """Kwargs for the stream-driver context factory (backend-specific).

        Base raises: a backend that sets ``STREAMING_PUMP`` must override this
        to supply exactly the kwargs its ``build_persistent_context`` accepts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets STREAMING_PUMP but does not override "
            "_stream_factory_kwargs"
        )

    def _ensure_driver(self, sample_ctx: Dict[str, Any]) -> bool:
        """Ensure ONE persistent stream-driver process is running.

        Idempotent. Spawns the driver (and its persistent ring / ctl_q /
        desc_q / stop event) on first use, and respawns it if it died or if a
        new dispatch needs a differently-sized ring (rare — ``num_reads`` and
        the topology are stable within a miner's life). The driver builds its
        own connected context via ``STREAM_FACTORY_DOTTED`` and keeps it alive
        across dispatches; only ``_close_driver`` (miner shutdown) reaps it.

        Returns:
            ``True`` if a live driver is ready (streaming path); ``False`` for
            the inline (CPU/GPU/Metal) path (``STREAMING_PUMP`` is False).
        """
        if not self.STREAMING_PUMP:
            return False
        nodes = sample_ctx["nodes"]
        dims = (int(sample_ctx["num_reads"]), len(nodes))
        alive = self._driver_proc is not None and self._driver_proc.is_alive()
        if alive and self._ring_dims == dims:
            return True  # reuse the running driver
        # Dead, first-time, or dims changed: tear down any stale driver, spawn.
        self._close_driver()
        import multiprocessing as _mp
        from QPU.stream_driver import stream_driver_main
        ctx = _mp.get_context("spawn")
        ring = SampleView(
            slots=self.RESULT_QUEUE_MAXSIZE, max_rows=dims[0], max_cols=dims[1],
        )
        desc_q = ctx.Queue(maxsize=self.RESULT_QUEUE_MAXSIZE)
        ctl_q = ctx.Queue()
        driver_stop = ctx.Event()
        factory_kwargs = self._stream_factory_kwargs(sample_ctx, nodes)
        driver_proc = spawn_worker(
            stream_driver_main,
            (ring.attach_args(), desc_q, ctl_q, driver_stop,
             self.STREAM_FACTORY_DOTTED, factory_kwargs),
            name=f"qpu-stream-driver-{self.miner_id}",
            # Non-daemon: the driver's RandomIsingFeeder forks pool children,
            # which a daemon process is forbidden from doing. _close_driver
            # reaps it explicitly so it never blocks interpreter shutdown.
            daemon=False,
        )
        self._ring = ring
        self._desc_q = desc_q
        self._ctl_q = ctl_q
        self._driver_stop = driver_stop
        self._driver_proc = driver_proc
        self._ring_dims = dims
        self._last_forwarded_threshold_milli = None
        return True

    def _close_driver(self) -> None:
        """Reap the persistent driver and close+unlink the ring (shutdown only).

        Sends the ctl_q shutdown sentinel, sets the stop event, joins the
        driver, then close-unlinks the ring after a GC pass (so no lingering
        ``_SharedSampleSet`` view keeps a segment exported — the MR !110
        BufferError contract). Safe to call when no driver exists.
        """
        if self._ctl_q is not None:
            try:
                self._ctl_q.put(None)  # shutdown sentinel
            except Exception:  # noqa: BLE001 — best-effort
                pass
        if self._driver_stop is not None:
            self._driver_stop.set()
        if self._driver_proc is not None:
            terminate_join(self._driver_proc, 5.0)
            self._driver_proc = None
        if self._ring is not None:
            import gc
            gc.collect()
            try:
                self._ring.close_unlink()
            except BufferError:
                gc.collect()
                try:
                    self._ring.close_unlink()
                except BufferError:
                    self.logger.error(
                        "shared ring close_unlink still blocked after GC; a "
                        "sample view leaked. Closing handles without unlink "
                        "(segments %s may persist).",
                        getattr(self._ring, "names", "?"),
                    )
                    self._ring.close()
            self._ring = None
        self._desc_q = None
        self._ctl_q = None
        self._driver_stop = None
        self._ring_dims = None
        pv = getattr(self, "_mempool_problem_view", None)
        if pv is not None:
            try:
                pv.close_unlink()
            except Exception:  # noqa: BLE001 — best-effort
                pass
            self._mempool_problem_view = None

    def _mempool_feeder_spec(self, context) -> Tuple[Any, ...]:
        """Write the order's fixed (h, J) into a ProblemView slot for the driver.

        The mempool model is fixed per order, so it is transferred once via a
        zero-copy ProblemView (the worker owns the slot; the driver reads it on
        the switch and reconstructs a FixedIsingFeeder). The previous slot is
        freed here — the driver read it on the prior mempool switch.

        Args:
            context: A MempoolJobContext with .nodes, .edges, .h_values,
                .j_values (i32 millivalues).

        Returns:
            A ``("mempool", attach_args, slot)`` tuple suitable for passing as
            ``feeder_spec`` in the ``("switch", ...)`` ctl_q command.

        Raises:
            RuntimeError: If no free slot is available in the new ProblemView
                (should not happen for a freshly created 1-slot view).
        """
        import numpy as np
        from shared.ring_views import ProblemView
        h_vec = np.asarray(
            [hv / 1000.0 for hv in context.h_values], dtype=np.float64,
        )
        j_vec = np.asarray(
            [jv / 1000.0 for jv in context.j_values], dtype=np.float64,
        )
        pv = ProblemView(
            slots=1, n_nodes=len(context.nodes), n_edges=len(context.edges),
        )
        slot = pv.claim_free(timeout=1.0)
        if slot is None:
            pv.close_unlink()
            raise RuntimeError(
                "ProblemView.claim_free returned None for a fresh 1-slot view; "
                "cannot write mempool model for driver"
            )
        pv.write(slot, h_vec, j_vec)
        prev = getattr(self, "_mempool_problem_view", None)
        if prev is not None:
            try:
                prev.close_unlink()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self._mempool_problem_view = pv
        return ("mempool", pv.attach_args(), slot)

    def close(self) -> None:
        """Release all persistent resources (call on miner/worker shutdown)."""
        self._close_driver()

    def _await_descriptor(
        self,
        stop_event: multiprocessing.synchronize.Event,
        desc_q: Any,
        driver_proc: Optional[Any],
    ) -> Any:
        """Block on the descriptor queue until an item, stop, or driver death.

        Returns the dequeued descriptor (which may be the ``None``
        end-of-stream sentinel), or one of :data:`_ACQUIRE_STOP` /
        :data:`_ACQUIRE_DONE` when the dispatch should end. The driver
        normally signals end-of-stream by enqueuing ``None`` from its
        ``finally``; but a hard crash (SIGKILL / OOM / C-extension abort)
        skips that ``finally`` entirely. Without a liveness check the
        consumer would drain an empty queue forever — the silent "miner
        looks stuck" failure mode. Detecting a dead ``driver_proc`` (after a
        final non-blocking drain in case the sentinel raced the exit) ends
        the dispatch instead.
        """
        while not stop_event.is_set():
            try:
                return desc_q.get(timeout=0.1)
            except queue.Empty:
                if driver_proc is not None and not driver_proc.is_alive():
                    try:
                        return desc_q.get_nowait()
                    except queue.Empty:
                        self.logger.error(
                            "stream driver pid=%s exited (exitcode=%s) without "
                            "end-of-stream sentinel; ending dispatch",
                            driver_proc.pid, driver_proc.exitcode,
                        )
                        return _ACQUIRE_DONE
        return _ACQUIRE_STOP

    def _acquire_result(
        self,
        stop_event: multiprocessing.synchronize.Event,
        desc_q: Optional[Any],
        preprocess_start: float,
        *,
        sample_ctx: Dict[str, Any],
        driver_proc: Optional[Any] = None,
        generation: int = 0,
    ) -> _AcquireResult:
        """Source one ``(nonce, salt, sampleset)`` for a loop iteration.

        Driver mode (``desc_q`` set) blocks on the descriptor queue, waking to
        check ``stop_event``; each descriptor names a ring slot whose
        sample/energy matrices the consumer reads zero-copy into a
        :class:`_SharedSampleSet`. The inline path calls ``_sample_batch``
        (falling back to a single ``_sample`` on the popped feeder model).
        Appends the sampling / preprocessing timings on success. Returns an
        :class:`_AcquireResult` whose ``action`` reproduces the original
        inline control flow exactly:

        - :data:`_ACQUIRE_STOP` — ``stop_event`` fired (caller returns None).
        - :data:`_ACQUIRE_DONE` — stream exhausted (caller breaks the loop).
        - :data:`_ACQUIRE_CONTINUE` — recoverable sampling error and
          ``_on_sampling_error`` returned falsey (caller continues).
        - :data:`_ACQUIRE_OK` — payload populated; process the iteration.

        A sampling error where ``_on_sampling_error`` returns truthy maps to
        :data:`_ACQUIRE_STOP` (the original ``return None``).
        """
        qpu_access_time_us: Optional[int] = None
        ring_slot: Optional[int] = None
        defect_info: Any = None
        try:
            sample_start = time.time()
            self.current_stage = 'sampling'
            self.current_stage_start = sample_start
            if desc_q is not None:
                # Driver mode: block on the descriptor queue, waking to check
                # stop_event and driver liveness. A trailing None — or a dead
                # driver — means the stream ended. Descriptors tagged with a
                # superseded generation are released + skipped (consumer-side
                # generation filter, the second of the three).
                while True:
                    item = self._await_descriptor(
                        stop_event, desc_q, driver_proc,
                    )
                    if item == _ACQUIRE_STOP:
                        return _AcquireResult(_ACQUIRE_STOP)
                    if item is None or item == _ACQUIRE_DONE:
                        return _AcquireResult(_ACQUIRE_DONE)
                    if self._ring is None:
                        self.logger.error(
                            "driver descriptor received but shared ring is "
                            "None; ending dispatch",
                        )
                        return _AcquireResult(_ACQUIRE_DONE)
                    slot, n_rows, n_cols, nonce, salt, qpu_us, desc_gen = item[:7]
                    defect_pickle = item[7] if len(item) > 7 else None
                    if desc_gen != generation:
                        # Straggler from a prior round; free the slot, keep
                        # waiting for a current-generation descriptor.
                        self._ring.release(slot)
                        continue
                    ring_slot = slot
                    break
                # Stop raced with delivery: release the slot and stop.
                if stop_event.is_set():
                    self._ring.release(ring_slot)
                    return _AcquireResult(_ACQUIRE_STOP)
                sampleset = _SharedSampleSet(
                    *self._ring.read(ring_slot, n_rows, n_cols),
                )
                # pickle.loads is safe here: defect_pickle was written by the
                # stream-driver subprocess (same operator-controlled process
                # tree) and travels only over a local multiprocessing.Queue —
                # there is no network boundary or untrusted input source.
                defect_info = (
                    pickle.loads(defect_pickle)
                    if defect_pickle is not None else None
                )
                qpu_access_time_us = int(qpu_us)
                if qpu_us > 0:
                    self.timing_stats['qpu_access_time'].append(int(qpu_us))
                    time_manager = getattr(self, "time_manager", None)
                    if time_manager is not None:
                        time_manager.record_block_time(int(qpu_us))
            else:
                # Inline path (CPU/GPU/Metal): unchanged.
                qpu_access_len_before = len(
                    self.timing_stats['qpu_access_time'],
                )
                batch = self._sample_batch(
                    sample_ctx["prev_hash"], sample_ctx["miner_id"],
                    sample_ctx["cur_index"],
                    sample_ctx["nodes"], sample_ctx["edges"],
                    num_reads=sample_ctx["num_reads"],
                    num_sweeps=sample_ctx["num_sweeps"],
                    **sample_ctx["extra"],
                )
                if batch is not None:
                    nonce, salt, sampleset = batch[0]
                else:
                    model = self._feeder.pop_blocking()
                    nonce, salt = model.nonce, model.salt
                    sampleset = self._sample(
                        model.h, model.J,
                        num_reads=sample_ctx["num_reads"],
                        num_sweeps=sample_ctx["num_sweeps"],
                        nonce_seed=nonce,
                        **sample_ctx["extra"],
                    )
                qpu_access_list = self.timing_stats['qpu_access_time']
                if len(qpu_access_list) > qpu_access_len_before:
                    qpu_access_time_us = int(qpu_access_list[-1])
            sample_time = time.time() - sample_start
            # In pump mode sample_time is queue-dequeue latency (~0µs),
            # not QPU access time — qpu_access_time_us carries the real
            # QPU figure.
            self.timing_stats['sampling'].append(sample_time * 1e6)
            self.timing_stats['preprocessing'].append(
                (sample_start - preprocess_start) * 1e6,
            )
        except Exception as exc:
            if self._on_sampling_error(exc, stop_event):
                return _AcquireResult(_ACQUIRE_STOP)
            return _AcquireResult(_ACQUIRE_CONTINUE)
        return _AcquireResult(
            _ACQUIRE_OK, nonce, salt, sampleset, qpu_access_time_us,
            ring_slot=ring_slot,
            defect_info=defect_info,
        )

    def _teardown_dispatch(self) -> None:
        """Tear down PER-DISPATCH state in ``mine_work_item``'s ``finally``.

        On the persistent driver path this does NOT kill the driver or close
        the ring — both persist across dispatches (that is the whole point of
        the redesign; ``_close_driver`` reaps them on miner shutdown). On the
        inline path it stops and clears the worker feeder. Always flushes the
        attempt logger and delegates to subclass ``_post_mine_cleanup``.
        """
        self.mining = False
        if self._dropped_results:
            self.logger.info(
                "mine_work_item: %d QPU results dropped under "
                "result-queue backpressure", self._dropped_results,
            )
        # Inline path only: tear down the worker feeder. The driver path has
        # no worker feeder (the persistent driver owns it).
        if self._feeder is not None:
            try:
                self._feeder.stop()
            finally:
                self._feeder = None
        attempt_logger = getattr(self, "_attempt_logger", None)
        if attempt_logger is not None:
            attempt_logger.flush()
        self._post_mine_cleanup()

    @staticmethod
    def _stash_insert(
        top_k: List[StashEntry],
        top_k_cap: int,
        entry: StashEntry,
        is_decay_ranked: bool,
    ) -> bool:
        """Insert ``entry`` into the bounded, win-time-ordered ``top_k``.

        Ranked by ``(decay_num, energy)`` when decay-ranked, else by raw
        ``energy``. Admits unconditionally while there's room; once full,
        evicts the furthest/worst entry only when ``entry`` strictly beats it
        (a nearer win-time, or the same step with a lower energy). Re-sorts in
        place. Returns True when the stash changed.
        """
        sort_key = (
            (lambda e: (e.decay_num, e.result.energy)) if is_decay_ranked
            else (lambda e: e.result.energy)
        )
        if len(top_k) < top_k_cap:
            top_k.append(entry)
            top_k.sort(key=sort_key)
            return True
        worst = max(top_k, key=sort_key)
        if sort_key(entry) < sort_key(worst):
            top_k.remove(worst)
            top_k.append(entry)
            top_k.sort(key=sort_key)
            return True
        return False

    @staticmethod
    def _select_submittable_candidate(
        top_k: List[StashEntry],
        live_threshold_milli: int,
    ) -> Optional[MiningResult]:
        """Return the first stash entry whose chain floor clears the threshold.

        Walks ``top_k`` (already win-time/energy ordered, best first) and
        returns the first candidate's ``MiningResult`` whose
        ``submit_floor_energy`` (falling back to ``energy``) is strictly below
        ``live_threshold_milli``, or ``None``.
        """
        for entry in top_k:
            r = entry.result
            if int(r.effective_floor * 1000) < live_threshold_milli:
                return r
        return None

    def _midstream_budget_ok(
        self, solution_number: int,
    ) -> Optional[MidstreamBudget]:
        """Per-loop QPU budget check; ``None`` when no budget applies.

        Base no-op (CPU/GPU have no ``time_manager``). The QPU subclass
        overrides this to consult its ``QPUTimeManager`` and return a
        :class:`MidstreamBudget` carrying the live decision + stats. Returning
        ``None`` means "no budget gating for this backend".
        """
        return None

    def _participation_extra(self) -> Dict[str, Any]:
        """Extra fields for the per-dispatch participation marker.

        Base returns ``{}`` (CPU/GPU have no reservoir). The QPU subclass adds
        ``{"budget_seconds": <pool at dispatch>}`` so the controller's remark
        records the QPU runway committed to this solution #.
        """
        return {}

    def _pause_driver(self, generation: int) -> None:
        """Tell the persistent driver to stop submitting NEW work (drain-idle).

        Sends a same-generation ``("pause", gen)`` so the driver stops calling
        ``_submit_one`` but leaves in-flight work to complete — the consumer
        keeps draining every already-paid-for attempt. Idempotent per
        dispatch (guarded by ``_budget_paused_generation``). No-op on the
        inline path (no ctl_q) or when no live driver exists. Best-effort: a
        queue hiccup must never break mining.
        """
        if self._ctl_q is None:
            return
        if self._budget_paused_generation == generation:
            return  # already paused this dispatch
        driver = self._driver_proc
        if driver is None or not driver.is_alive():
            return
        try:
            self._ctl_q.put(("pause", int(generation)))
            self._budget_paused_generation = generation
        except Exception as exc:  # noqa: BLE001 — best-effort
            self.logger.debug("budget pause forward failed (ignored): %s", exc)

    def _forward_threshold_to_driver(
        self, generation: int, live_threshold_milli: int,
    ) -> None:
        """Send a same-generation threshold update to the driver, on change.

        No-op on the inline path (no ctl_q) or when the value is unchanged.
        Best-effort: a queue hiccup must never break mining.
        """
        if self._ctl_q is None:
            return
        if live_threshold_milli == self._last_forwarded_threshold_milli:
            return
        try:
            self._ctl_q.put(("threshold", generation, live_threshold_milli))
            self._last_forwarded_threshold_milli = live_threshold_milli
        except Exception as exc:  # noqa: BLE001 — best-effort
            self.logger.debug("threshold forward failed (ignored): %s", exc)

    def _stash_pre_check(
        self,
        state: _MiningLoopState,
        sampleset: Any,
        iter_best_energy: float,
        iter_best_milli: int,
    ) -> bool:
        """Decide whether this iter's best sample earns ``evaluate_sampleset``.

        Legacy (non-decay) path: admit when the iter's best energy beats the
        worst stashed energy (or the stash isn't full). Decay path: admit
        while the stash is under cap, else when the iter's best energy could
        clear within the stash's furthest win-time (``s_min <= s_max``);
        ``iter_best_milli`` is an optimistic proxy for the not-yet-computed
        submit floor — a sample whose best energy can't clear by the furthest
        stashed step can never produce a stashable floor.

        Energy-only gate; width handling has moved to ``_run_substrate_ratchet``
        where ``_finalize_sample`` can reconstruct a reduced QPU sampleset after
        the pre-check but before evaluation. Pure read of ``state.top_k``; does
        not mutate.
        """
        decay_schedule = state.decay_schedule
        if decay_schedule is None:
            ratchet_threshold = (
                state.top_k[-1].result.energy
                if len(state.top_k) >= state.top_k_cap
                else float("inf")
            )
            admit = (iter_best_milli / 1000.0) < ratchet_threshold
        elif len(state.top_k) < state.top_k_cap:
            admit = True
        else:
            s_max = max(e.decay_num for e in state.top_k)
            s_min = step_for_energy(decay_schedule, iter_best_milli)
            admit = s_min is not None and s_min <= s_max

        return admit

    def _compute_stash_entry(
        self,
        state: _MiningLoopState,
        result: MiningResult,
    ) -> Optional[StashEntry]:
        """Wrap a valid ``result`` into a ``StashEntry`` for the active ranking.

        Decay path: derive the first decay step whose threshold clears the
        result's effective floor and the absolute block it lands on; returns
        ``None`` when the floor never clears within the schedule horizon (so
        the caller stashes nothing). Legacy path: ``StashEntry(0, 0, result)``.
        """
        decay_schedule = state.decay_schedule
        if decay_schedule is None:
            return StashEntry(0, 0, result)
        s_floor = step_for_energy(
            decay_schedule, int(result.effective_floor * 1000),
        )
        if s_floor is None:
            # Never clears within the schedule horizon — not stashed.
            return None
        valid_at = state.last_proof_block + s_floor * state.epoch_length
        return StashEntry(s_floor, valid_at, result)

    def _run_substrate_ratchet(
        self, state: _MiningLoopState, sampleset: Any, nonce: Any, salt: bytes,
        postprocess_start: float, *, preview_cb: Optional[Any],
        attempt_log_kwargs: Dict[str, Any],
        defect_info: Any = None,
    ) -> Optional[MiningResult]:
        """Substrate / PoW ratchet path (the ``is_substrate`` branch).

        Runs the ``_stash_pre_check``-gated ``evaluate_sampleset``, maintains
        ``state.top_k``, emits a preview, applies the submit gate, and records
        the attempt. Returns the submittable result (or ``None``).
        """
        # Live (decay-applied) threshold the controller pushed via shared mem;
        # a later iter's submit gate returns the stored result once it decays.
        if state.live_threshold_var is not None:
            with state.live_threshold_var.get_lock():
                live_threshold_milli = int(state.live_threshold_var.value)
        else:
            live_threshold_milli = int(
                state.requirements.difficulty_energy * 1000,
            )
        # Forward the decayed threshold to the persistent driver (rule 2 of the
        # decay contract): same-generation update, no reseed/cancel/bump.
        self._forward_threshold_to_driver(state.generation, live_threshold_milli)

        # ``_stash_pre_check`` gates the expensive ``evaluate_sampleset``.
        # Apply the energy offset for QPU defect clamping: reduced samplesets
        # omit clamped spins whose fixed-spin energy contribution is in the
        # offset. With defect_info=None (all current paths) offset=0 so
        # behaviour is identical to before.
        offset = float(defect_info.energy_offset) if defect_info is not None else 0.0
        iter_best_energy = float(np.min(sampleset.record.energy)) + offset
        iter_best_milli = int(iter_best_energy * 1000)
        # ratchet_threshold (legacy energy gate the attempt log records):
        # +inf until the stash is full, then the worst stashed energy.
        ratchet_threshold = (
            state.top_k[-1].result.energy
            if len(state.top_k) >= state.top_k_cap
            else float("inf")
        )
        improves_stash = self._stash_pre_check(
            state, sampleset, iter_best_energy, iter_best_milli,
        )
        # Width handling: a reduced sampleset (clamped QPU qubits dropped)
        # must be reconstructed before evaluate_sampleset indexes topology
        # positions. With defect_info, reconstruct here; without it, the
        # sample is unexpectedly narrow — skip evaluation and log a warning.
        if improves_stash and sampleset.record.sample.shape[1] != len(state.nodes):
            if defect_info is not None:
                sampleset = self._finalize_sample(sampleset, defect_info)
            else:
                self.logger.info(
                    "[%s] Mining attempt - Energy: %.0f (under-reconstructed: "
                    "sample width %d != topology %d; skipping evaluation)",
                    self.miner_id, iter_best_energy,
                    sampleset.record.sample.shape[1], len(state.nodes),
                )
                improves_stash = False

        result = None
        stored_replaced = False
        # Captured from the post-processed eval so the submit-gate's later
        # rebind of `result` doesn't destroy them (logged on every
        # ``post_processed=true`` iter, not just submitted ones).
        post_num_valid: Optional[int] = None
        post_diversity_milli: Optional[int] = None
        if improves_stash:
            # Lenient eval (diversity + min_solutions required, no energy gate).
            result = self.evaluate_sampleset(
                sampleset, state.requirements, state.nodes, state.edges,
                nonce, salt, state.prev_timestamp, state.start_time,
                strict_energy=False,
                live_threshold_energy=(live_threshold_milli / 1000.0),
            )
            attempt_log_kwargs["post_processed"] = True
            if result is not None:
                post_num_valid = result.num_valid
                post_diversity_milli = int(result.diversity * 1000)
                entry = self._compute_stash_entry(state, result)
                stored_replaced = entry is not None and self._stash_insert(
                    state.top_k, state.top_k_cap, entry, state.is_decay_ranked,
                )
        else:
            # Pre-check skipped the lenient evaluate (and its log line). Emit a
            # lightweight heartbeat sharing the "[id] Mining attempt - Energy:"
            # prefix so one grep catches both; num_valid / diversity omitted.
            self.logger.info(
                "[%s] Mining attempt - Energy: %.0f (pre-check skip: not in "
                "top-5, worst stashed=%.0f, live threshold<=%d)",
                self.miner_id, iter_best_energy,
                (state.top_k[-1].result.energy
                 if len(state.top_k) >= state.top_k_cap
                 else float("inf")),
                live_threshold_milli,
            )

        # Anticipatory-submission preview on a stash improvement; the submit
        # gate below still owns the returned result.
        if stored_replaced:
            # Throttled "Best Solution" log line when the earliest-winning
            # entry changes (decay path only).
            if state.is_decay_ranked and state.top_k:
                best = min(
                    state.top_k, key=lambda e: (e.decay_num, e.result.energy),
                )
                new_wintime = (best.decay_num, best.valid_at_block)
                if new_wintime != state.last_best_wintime:
                    self.logger.info(
                        "[%s] Best Solution: floor=%.0f energy=%.0f "
                        "div=%.3f -> submittable at block %d (decay #%d)",
                        self.miner_id,
                        best.result.effective_floor,
                        best.result.energy,
                        best.result.diversity,
                        best.valid_at_block,
                        best.decay_num,
                    )
                    state.last_best_wintime = new_wintime
            if preview_cb is not None:
                state.previewed_wintime = self._maybe_emit_preview(
                    preview_cb, state.top_k, state.previewed_wintime,
                    state.dispatch_id_for_log,
                )

        self.timing_stats['postprocessing'].append(
            (time.time() - postprocess_start) * 1e6,
        )

        # Submit gate: the chain filters each solution with strict
        # ``< max_energy_milli``, so compare the chain-equivalent floor (not
        # best-of-set ``energy``). ``_select_submittable_candidate`` walks the
        # stash in win-time order and returns the first floor that clears.
        result = self._select_submittable_candidate(
            state.top_k, live_threshold_milli,
        )

        self._record_ratchet_attempt(
            attempt_log_kwargs, live_threshold_milli, ratchet_threshold,
            post_num_valid, post_diversity_milli,
            submitted=result is not None, stored=stored_replaced,
        )
        return result

    @staticmethod
    def _record_ratchet_attempt(
        attempt_log_kwargs: Dict[str, Any],
        live_threshold_milli: int,
        ratchet_threshold: float,
        num_valid: Optional[int],
        diversity_milli: Optional[int],
        *,
        submitted: bool,
        stored: bool,
    ) -> None:
        """Populate ``attempt_log_kwargs`` with this iter's ratchet outcome.

        ``ratchet_threshold`` is +inf before anything is stored — logged as
        None there (no meaningful prior).
        """
        ratchet_threshold_milli = (
            int(ratchet_threshold * 1000)
            if math.isfinite(ratchet_threshold)
            else None
        )
        result_kind = (
            "submitted" if submitted else "stored" if stored else "rejected"
        )
        attempt_log_kwargs.update(
            threshold_milli=live_threshold_milli,
            ratchet_threshold_milli=ratchet_threshold_milli,
            num_valid=num_valid,
            diversity_milli=diversity_milli,
            stored_as_best=stored,
            result_kind=result_kind,
        )

    def _run_mempool_eval(
        self,
        state: _MiningLoopState,
        sampleset: Any,
        nonce: Any,
        salt: bytes,
        postprocess_start: float,
        *,
        attempt_log_kwargs: Dict[str, Any],
        defect_info: Any = None,
    ) -> Optional[MiningResult]:
        """Mempool-path strict evaluation (the ``else`` of ``is_substrate``).

        Runs ``evaluate_sampleset`` with strict semantics, appends the
        post-processing timing, and updates ``attempt_log_kwargs`` in place.
        Returns the evaluated ``MiningResult`` (or ``None``). Behaviour is
        identical to the original inline mempool branch.

        QPU mempool samples arrive reduced (offline qubits stripped) carrying
        ``defect_info``.  ``evaluate_sampleset`` requires the full-topology
        sample, so a reduced sample is reconstructed via ``_finalize_sample``
        before evaluation.  Metal/CUDA mempool samples are full-width with
        ``defect_info=None`` and are unaffected.
        """
        if sampleset.record.sample.shape[1] != len(state.nodes):
            if defect_info is not None:
                sampleset = self._finalize_sample(sampleset, defect_info)
            else:
                self.logger.info(
                    "[%s] mempool attempt skipped (under-reconstructed: "
                    "width %d != topology %d)",
                    self.miner_id, sampleset.record.sample.shape[1],
                    len(state.nodes),
                )
                return None

        result = self.evaluate_sampleset(
            sampleset, state.requirements, state.nodes, state.edges,
            nonce, salt, state.prev_timestamp, state.start_time,
        )

        self.timing_stats['postprocessing'].append(
            (time.time() - postprocess_start) * 1e6,
        )
        # Mempool path may have ``difficulty_energy = +inf``
        # (unbounded threshold for jobs with no min_energy
        # floor). int() would overflow — use None instead.
        threshold_milli_log: Optional[int] = None
        if math.isfinite(state.requirements.difficulty_energy):
            threshold_milli_log = int(
                state.requirements.difficulty_energy * 1000,
            )
        attempt_log_kwargs.update(
            post_processed=True,
            threshold_milli=threshold_milli_log,
            num_valid=(result.num_valid if result is not None else None),
            diversity_milli=(
                int(result.diversity * 1000)
                if result is not None else None
            ),
            result_kind=(
                "submitted" if result is not None else "rejected"
            ),
        )
        return result

    def _finalize_iteration_logging(
        self,
        state: _MiningLoopState,
        sampleset: Any,
        nonce: Any,
        salt: bytes,
        progress: int,
        *,
        preprocess_start: float,
        qpu_access_time_us: Optional[int],
        attempt_log_kwargs: Dict[str, Any],
    ) -> None:
        """Finalise + persist per-iteration logging.

        Fills the timing/feeder/solution-meta fields on
        ``attempt_log_kwargs``, records the attempt row, and archives the
        top-5 spin configs to the solution store when the iter was stored or
        submitted. Pure side effects — no control flow. Behaviour matches
        the original inline logging block exactly.
        """
        attempt_log_kwargs["mining_time_us"] = int(
            (time.time() - preprocess_start) * 1e6
        )
        attempt_log_kwargs["qpu_access_time_us"] = qpu_access_time_us
        if self._feeder is not None:
            fstats = self._feeder.stats()
            attempt_log_kwargs["feeder_ready"] = fstats["ready"]
            attempt_log_kwargs["feeder_drained_count"] = (
                fstats["drained_count"]
            )
            attempt_log_kwargs["feeder_pop_wait_total_s"] = (
                fstats["pop_wait_total_s"]
            )
        else:
            # Driver path (QPU): the feeder lives in the stream-driver
            # process, so its stats aren't reachable here. Record None.
            attempt_log_kwargs["feeder_ready"] = None
            attempt_log_kwargs["feeder_drained_count"] = None
            attempt_log_kwargs["feeder_pop_wait_total_s"] = None
        # Compute solution_meta scalars + capture top-5 solutions. Meta is
        # always embedded in the attempt log; top-5 spins go to disk only
        # when this iter is stored or submitted (see write below).
        sol_meta, top_5_sols, top_5_es = compute_solution_meta(
            sampleset, state.requirements.difficulty_energy,
        )
        attempt_log_kwargs["solution_meta"] = sol_meta
        state.attempt_log.record(**attempt_log_kwargs)

        # SolutionStore — archive top-5 spin configs only when the chain
        # ratchet kept this candidate. The result_kind set on the attempt
        # drives the gate; "submitted" and "stored" both qualify since both
        # produce a candidate we (or someone) might reproduce for analysis.
        attempt_result_kind = attempt_log_kwargs.get("result_kind")
        if attempt_result_kind in ("stored", "submitted") and top_5_sols:
            nonce_hex = (
                nonce.hex() if isinstance(nonce, (bytes, bytearray))
                else f"{int(nonce):064x}"
            )
            state.solution_store.record(
                solution_number=state.solution_number_for_log,
                iter_num=progress + 1,
                nonce_hex=nonce_hex,
                salt_hex=salt.hex(),
                top_5_solutions_hex=[
                    pack_spins_hex(s) for s in top_5_sols
                ],
                top_5_energies=top_5_es,
                result_kind=attempt_result_kind,
            )

    # ------------------------------------------------------------------
    # Anticipatory-submission preview (substrate / PoW ratchet path)
    # ------------------------------------------------------------------

    def _maybe_emit_preview(
        self,
        preview_cb: Any,
        top_k: List[StashEntry],
        previewed_wintime: Tuple[int, int],
        dispatch_id: int,
    ) -> Tuple[int, int]:
        """Emit a best-candidate preview on win-time or floor improvement.

        On the decay path selects the **earliest-winning** entry by
        ``(decay_num, energy)`` and emits when ``(valid_at_block,
        floor_milli)`` is lexicographically smaller than the last-previewed
        pair — a candidate that will win EARLIER always fires a fresh preview
        even if its energy/floor is not lower.

        On the legacy (non-decay) path all entries have ``valid_at_block=0``
        and ``decay_num=0``, so the tiebreak collapses to ``floor_milli``
        alone, preserving the prior floor-improvement-only behaviour.

        Returns the (possibly updated) ``(valid_at_block, floor_milli)``
        throttle state so the caller can persist it. A failing callback or
        payload build never propagates — mining must not break on a preview
        hiccup.
        """
        if not top_k:
            return previewed_wintime
        # Earliest-winning entry: (decay_num, energy) ascending — nearest
        # win-time first, then lower energy as a tiebreak.
        best = min(top_k, key=lambda e: (e.decay_num, e.result.energy))
        best_result = best.result
        best_floor = best_result.effective_floor
        best_floor_milli = int(best_floor * 1000)
        new_wintime = (best.valid_at_block, best_floor_milli)
        if new_wintime >= previewed_wintime:
            # No lexicographic improvement over what we already previewed.
            return previewed_wintime
        try:
            payload = {
                "dispatch_id": dispatch_id,
                "miner_type": self.miner_type,
                "nonce": best_result.nonce,
                "salt": best_result.salt,
                "solutions": best_result.solutions,
                "submit_floor_energy": best_floor,
                "energy": best_result.energy,
                "num_valid": best_result.num_valid,
                "diversity": best_result.diversity,
                "valid_at_block": best.valid_at_block,
                "decay_num": best.decay_num,
            }
            preview_cb(payload)
        except Exception as exc:  # noqa: BLE001 — preview is best-effort
            self.logger.debug("preview_cb failed (ignored): %s", exc)
            # Don't advance the throttle on failure so a later identical
            # improvement still gets a chance to emit.
            return previewed_wintime
        return new_wintime

    # ------------------------------------------------------------------
    # Hook methods (override in subclasses as needed)
    # ------------------------------------------------------------------

    def _finalize_sample(self, sampleset: Any, defect_info: Any) -> Any:
        """Reconstruct a reduced sampleset to full topology (survivor-only).

        Called in ``_run_substrate_ratchet`` only when a sample passes the
        energy pre-check but its width is narrower than the topology (i.e. the
        QPU stream driver clamped offline qubits out before writing to the
        ring). ``defect_info`` carries the fixed-spin assignments and energy
        offset needed for reconstruction.

        The base implementation is the identity (CPU/GPU/Metal samples are
        always full width; the ratchet's width guard never triggers for them).
        ``DWaveMiner`` overrides this to call ``reconstruct_full_sampleset``.
        """
        return sampleset

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
        """Sample multiple nonces in a single kernel launch.

        Returns list of (nonce, salt, sampleset) tuples, or None
        to fall through to single-nonce _sample() path.
        Override in miners that support multi-nonce dispatch.
        """
        return None

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

    def _log_work_start(self, context: WorkContext) -> None:
        """Emit the per-iteration setup banner for the active work source.

        Different shapes per context flavor:
          - PoW: block_number / parent_hash / topology_hash / nodes / edges
          - Mempool: order_id / nodes / edges
        Both flavors print enough to grep logs for a specific work item.
        """
        if isinstance(context, MempoolJobContext):
            self.logger.info(
                f"mine_work_item: order_id={context.order_id} "
                f"nodes={len(context.nodes)} edges={len(context.edges)} "
                f"(mempool)"
            )
        else:
            self.logger.info(
                f"mine_work_item: last_proof_block_hash=0x{context.last_proof_block_hash.hex()[:16]}... "
                f"topology=0x{context.topology_hash.hex()[:16]}... "
                f"nodes={len(context.nodes)} edges={len(context.edges)}"
            )

    @staticmethod
    def _graceful_exit() -> None:
        """Exit the process gracefully, guarding against interpreter finalization.

        SIGTERM handlers that call ``sys.exit(0)`` during interpreter shutdown
        raise ``SystemExit`` in a context where it cannot propagate, producing
        the "Exception ignored in: <module 'threading' ...>" noise on stderr.
        This guard suppresses that by returning early when the interpreter is
        already finalizing — the process is exiting anyway, so no action is
        needed.
        """
        if sys.is_finalizing():
            return
        sys.exit(0)

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
        self.logger.error(
            f"Sampling error: {error}\n{traceback.format_exc()}"
        )
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return machine-readable stats for this miner."""
        stats = dict(self.timing_stats)
        stats.update({
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
        })
        if self._feeder is not None:
            stats["feeder_stats"] = self._feeder.stats()
        return stats

    def evaluate_sampleset(self, sampleset: dimod.SampleSet, requirements: BlockRequirements, nodes: List[int], edges: List[Tuple[int, int]], nonce: int, salt: bytes, prev_timestamp: int, start_time: float, *, strict_energy: bool = True, live_threshold_energy: Optional[float] = None) -> Optional[MiningResult]:
        """Convert a sample set into a mining result if it meets requirements, otherwise return None.

        ``strict_energy=False`` enables the substrate ratchet's lenient
        mode: diversity + min_solutions are still required but the
        energy gate is dropped, so candidates that don't yet beat the
        chain's snapshot threshold can still be stashed in the
        ``top_k`` heap for later submission when decay catches up.

        ``live_threshold_energy`` (float, ratchet path only) is the
        chain's *live* (decay-applied) target. When provided, the
        returned result's ``num_valid`` counts samples whose
        chain-recomputed energy clears that live target (rather than
        the snapshot ``difficulty_energy``).
        """
        return evaluate_sampleset(sampleset, requirements, nodes, edges, nonce, salt, prev_timestamp, start_time, self.miner_id, self.miner_type, strict_energy=strict_energy, live_threshold_energy=live_threshold_energy)


# ----------------------------------------------------------------------
# Helpers for ``mine_work_item``
# ----------------------------------------------------------------------


def _work_tag(context: WorkContext) -> str:
    """Short identifier for a work context, used in per-iteration log lines."""
    if isinstance(context, MempoolJobContext):
        return f"order={context.order_id}"
    return f"last_proof_block_hash=0x{context.last_proof_block_hash.hex()[:16]}"


def _energy_to_milli(energy: float) -> int:
    """Convert a difficulty energy (float) to integer milli-units.

    Non-finite thresholds clamp to a large positive sentinel. This branch is
    effectively unreachable on the driver path (mempool jobs, the only source
    of a non-finite energy floor, are never routed to the QPU driver), so the
    value only needs to be a safe, JSON/pickle-friendly int; it is not a tuned
    gate value.
    """
    if not math.isfinite(energy):
        return 1 << 62
    return int(energy * 1000)


@dataclass(frozen=True)
class _BridgePrevBlockHeader:
    index: int


@dataclass(frozen=True)
class _BridgePrevBlock:
    """Minimal duck-typed `prev_block` for substrate-mode hook compatibility.

    Concrete miners read ``prev_block.header.index`` and ``prev_block.hash``
    inside their ``_pre_mine_setup`` overrides. Phase 3 introduced this
    shim for the PoW path; Phase 8b extends it to mempool: for a
    ``MempoolJobContext`` the index is the ``order_id`` and the hash is
    zeros (mempool has no parent hash). The values flow into log
    messages and budget checks but never feed back into the chain, so
    the placeholders are safe.
    """

    header: _BridgePrevBlockHeader
    hash: bytes

    @classmethod
    def from_work_context(cls, context: WorkContext) -> "_BridgePrevBlock":
        if isinstance(context, MempoolJobContext):
            return cls(
                header=_BridgePrevBlockHeader(index=context.order_id),
                hash=b"\x00" * 32,
            )
        # PoW: the nonce no longer depends on `prev_block.header.index`
        # (the round-seed contract replaced the block-number input), so
        # this index is now a non-binding placeholder for legacy subclass
        # hooks that read it for logging or budget checks. `hash` carries
        # the last proof block hash so any hook needing a stable per-round identifier
        # still gets a meaningful 32-byte value.
        return cls(
            header=_BridgePrevBlockHeader(index=0),
            hash=context.last_proof_block_hash,
        )


@dataclass(frozen=True)
class _BridgeNodeInfo:
    """Minimal duck-typed `node_info` for substrate-mode hook compatibility.

    Legacy hooks read ``node_info.miner_id`` (a string). PoW exposes the
    chain account as a hex string; mempool uses a synthetic
    ``mempool-order-<id>`` tag since the order itself, not the solver,
    identifies the work. ``miner_account_bytes`` is preserved for the
    PoW path (used by the feeder) and zero-filled for mempool.
    """

    miner_id: str
    miner_account_bytes: bytes

    @classmethod
    def from_work_context(cls, context: WorkContext) -> "_BridgeNodeInfo":
        if isinstance(context, MempoolJobContext):
            return cls(
                miner_id=f"mempool-order-{context.order_id}",
                miner_account_bytes=b"\x00" * 32,
            )
        return cls(
            miner_id="0x" + context.miner_account_bytes.hex(),
            miner_account_bytes=context.miner_account_bytes,
        )


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

