"""Abstract base miner for quantum blockchain mining.

Contains core mining logic and defines abstract methods for miner-specific implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import math
import multiprocessing
import multiprocessing.synchronize
import queue
import threading
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
from substrate.types import SubstrateMiningContext
from shared.work_context import (
    WorkContext,
    requirements_from_context,
)

# Global logger for this module
log = logging.getLogger(__name__)

# Emit a progress log line every N sampling attempts. 10 is a balance between
# noise (one line per attempt is too chatty for the slow miners) and
# observability (one line per minute on the typical CPU/GPU cadence).
PROGRESS_LOG_INTERVAL = 10


@dataclass(frozen=True)
class _PumpedResult:
    """One QPU result handed from the pump thread to the consumer loop.

    Carries the QPU access time extracted on the pump thread so the
    consumer doesn't have to race the shared ``timing_stats`` list.
    """

    nonce: int | bytes
    salt: bytes
    sampleset: Any
    qpu_access_time_us: Optional[int]


# Sentinel pushed by the pump when the stream is exhausted so the
# consumer's blocking get() unblocks and the loop can exit. Module-level
# (not a class attribute) so subclasses can't accidentally shadow it and
# break the consumer's ``is`` identity check.
_PUMP_DONE = object()


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

        # Streaming-pump shutdown signal. Set while a pump thread is
        # running so the QPU stream can observe shutdown (Task 3 reads
        # this); None when no pump is active.
        self._pump_stop: Optional[threading.Event] = None

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
    # Bounded result queue depth. Sized to the in-flight QPU job count so
    # the consumer has headroom equal to what the cloud holds; on full the
    # pump drops the newest result (rare safety valve) and counts it.
    RESULT_QUEUE_MAXSIZE: int = 32
    # Ratchet pre-check: skip the expensive lenient evaluate unless the
    # iter's cheap best-energy is within this many milli of the live
    # (decayed) threshold — i.e. it could plausibly submit now or after a
    # little decay. Larger = more evaluates (safer, slower); 0 = only
    # iters already at/below the live target. Tunable per backend.
    RATCHET_PRECHECK_MARGIN_MILLI: int = 2000

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
            **kwargs: Forwarded to ``_pre_mine_setup``.

        Returns:
            A ``MiningResult`` if a valid solution is found before the
            stop event fires; ``None`` otherwise.
        """
        # -- setup --------------------------------------------------------
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
            self.logger.warning(
                "mine_work_item: _pre_mine_setup returned False, aborting "
                "attempt for %s", _work_tag(context),
            )
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
        # Sorted ASC by energy; len <= TOP_K_STORE. ``top_k[0]`` is the
        # best candidate; ``top_k[-1]`` is the eviction target.
        top_k: List[MiningResult] = []
        top_k_cap: int = self.TOP_K_STORE
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
        dispatch_id_for_log = int(getattr(self, '_current_dispatch_id', 0))

        # Build the feeder for this attempt. Each context flavor picks
        # the right backing implementation (RandomIsingFeeder for PoW,
        # FixedIsingFeeder for mempool). We own the lifecycle here —
        # ``finally`` stops and clears it so a subsequent dispatch starts
        # from a clean state. Streaming-capable backends (QPU async
        # pipeline, GPU multi-problem dispatch) read ``self._feeder`` from
        # ``_sample_batch`` and keep ``queue_depth`` jobs in flight — that
        # is what gives the QPU its throughput. Backends without batch
        # streaming pop the feeder one model at a time (see the loop).
        self._feeder = context.make_feeder(
            nodes, edges, buffer_size=self.FEEDER_BUFFER_SIZE,
        )

        # Positional args for the legacy ``_sample_batch`` signature. The
        # QPU/GPU streaming impls ignore them — their feeder already
        # encapsulates the round seed and miner identity — but the base
        # contract requires (prev_hash, miner_id, cur_index).
        batch_prev_hash = bridge_prev_block.hash
        batch_miner_id = bridge_node_info.miner_id
        batch_cur_index = bridge_prev_block.header.index

        # --- Out-of-band result pump (streaming backends only) ----------
        # CPU/GPU/Metal keep the inline path below (STREAMING_PUMP=False).
        self._dropped_results = 0
        result_queue: Optional["queue.Queue"] = None
        pump_thread: Optional[threading.Thread] = None
        pump_stop: Optional[threading.Event] = None
        if self.STREAMING_PUMP:
            result_queue = queue.Queue(maxsize=self.RESULT_QUEUE_MAXSIZE)
            pump_stop = threading.Event()
            # The QPU stream observes this so its in-flight next() unblocks
            # on shutdown before we join the pump and close the generator.
            self._pump_stop = pump_stop
            sample_kwargs = {
                "prev_hash": batch_prev_hash,
                "miner_id": batch_miner_id,
                "cur_index": batch_cur_index,
                "nodes": nodes,
                "edges": edges,
                "num_reads": num_reads,
                "num_sweeps": current_num_sweeps,
                "extra": extra_params,
            }
            pump_thread = threading.Thread(
                target=self._result_pump,
                args=(result_queue, pump_stop, sample_kwargs),
                name=f"qpu-pump-{self.miner_id}",
                daemon=True,
            )
            pump_thread.start()

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

                # Per-iteration QPU access time in microseconds (D-Wave
                # ``qpu_programming_time + qpu_sampling_time``). In pump mode
                # the pump thread extracts it and ships it on the result; on
                # the inline path the branch below snapshots the shared
                # ``timing_stats['qpu_access_time']`` list around the sample.
                # ``None`` when no entry was produced (non-QPU backends, or a
                # QPU sampleset without timing info).
                qpu_access_time_us: Optional[int] = None
                try:
                    sample_start = time.time()
                    self.current_stage = 'sampling'
                    self.current_stage_start = sample_start
                    if result_queue is not None:
                        # Pump mode: block on the queue, but wake to check
                        # stop_event. _PUMP_DONE means the stream ended.
                        item = None
                        while not stop_event.is_set():
                            try:
                                item = result_queue.get(timeout=0.1)
                                break
                            except queue.Empty:
                                continue
                        if stop_event.is_set():
                            return None
                        if item is _PUMP_DONE:
                            break  # stream exhausted; exit the loop
                        nonce = item.nonce
                        salt = item.salt
                        sampleset = item.sampleset
                        qpu_access_time_us = item.qpu_access_time_us
                    else:
                        # Inline path (CPU/GPU/Metal): unchanged.
                        qpu_access_len_before = len(
                            self.timing_stats['qpu_access_time'],
                        )
                        batch = self._sample_batch(
                            batch_prev_hash, batch_miner_id, batch_cur_index,
                            nodes, edges,
                            num_reads=num_reads,
                            num_sweeps=current_num_sweeps,
                            **extra_params,
                        )
                        if batch is not None:
                            nonce, salt, sampleset = batch[0]
                        else:
                            model = self._feeder.pop_blocking()
                            nonce, salt = model.nonce, model.salt
                            sampleset = self._sample(
                                model.h, model.J,
                                num_reads=num_reads,
                                num_sweeps=current_num_sweeps,
                                nonce_seed=nonce,
                                **extra_params,
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
                        return None
                    continue

                sampleset = self._post_sample(sampleset)
                if stop_event.is_set():
                    return None

                postprocess_start = time.time()
                self.current_stage = 'postprocessing'
                self.current_stage_start = postprocess_start

                self.timing_stats['total_samples'] += len(
                    sampleset.record.energy,
                )
                self.timing_stats['blocks_attempted'] += 1

                # Per-iteration log fields (filled below per code path).
                attempt_log_kwargs: Dict[str, Any] = {
                    "dispatch_id": dispatch_id_for_log,
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
                stored_replaced = False

                if is_substrate:
                    # ---- Ratchet path (substrate / PoW) -------------
                    # Read the chain's *live* (decay-applied) energy
                    # threshold the controller pushed via shared mem.
                    # When the chain decays past our stored best, the
                    # next iteration's check returns the stored result.
                    if live_threshold_var is not None:
                        with live_threshold_var.get_lock():
                            live_threshold_milli = int(live_threshold_var.value)
                    else:
                        live_threshold_milli = int(
                            requirements.difficulty_energy * 1000,
                        )

                    # Post-processing gate: stash the K best candidates
                    # we've mined, not the best the chain wants. The
                    # ratchet is about *local progress* — until the
                    # stash is full, every iter post-processes
                    # unconditionally so we build a baseline; once full,
                    # only iters beating the heap's worst-energy entry
                    # earn the expensive ``evaluate_sampleset`` call.
                    # Submission against the chain's live (decayed)
                    # threshold is a separate decision handled by the
                    # submit gate below — gating storage on the chain
                    # target would lock the miner out of building a
                    # baseline whenever the live threshold is harder
                    # than what SA is producing.
                    iter_best_energy = float(
                        np.min(sampleset.record.energy),
                    )
                    ratchet_threshold = (
                        top_k[-1].energy
                        if len(top_k) >= top_k_cap
                        else float("inf")
                    )
                    # Pre-check: only pay the lenient evaluate when the iter
                    # both (a) would improve the running top_k baseline and
                    # (b) is within RATCHET_PRECHECK_MARGIN_MILLI of the live
                    # (decayed) threshold — i.e. could submit now or after a
                    # little decay. No-hope iters (best far worse than the
                    # live target) skip the expensive diversity/selection
                    # work and log a lightweight rejected row. Mirrors the
                    # strict fast-bail the canary gets for free.
                    # NOTE: if the miner never lands within margin of the live
                    # target (e.g. hardware underperforming), the stash stays
                    # empty and nothing submits — expected; every skipped iter
                    # still logs as "rejected".
                    iter_best_milli = int(iter_best_energy * 1000)
                    near_live = iter_best_milli <= (
                        live_threshold_milli + self.RATCHET_PRECHECK_MARGIN_MILLI
                    )
                    improves_stash = iter_best_energy < ratchet_threshold

                    result = None
                    # Values captured from the post-processed eval so the
                    # submit-gate's later rebind of `result` doesn't
                    # destroy them — we want them logged on every
                    # ``post_processed=true`` iteration, not just the
                    # submitted ones. No new computation: these are
                    # already produced by ``evaluate_sampleset``.
                    post_num_valid: Optional[int] = None
                    post_diversity_milli: Optional[int] = None
                    if improves_stash and near_live:
                        # Lenient eval — diversity + min_solutions
                        # still required, but no energy gate. The
                        # ratchet itself enforces "only improvements
                        # land" via the comparison below.
                        result = self.evaluate_sampleset(
                            sampleset, requirements, nodes, edges,
                            nonce, salt, prev_timestamp, start_time,
                            strict_energy=False,
                            live_threshold_energy=(
                                live_threshold_milli / 1000.0
                            ),
                        )
                        attempt_log_kwargs["post_processed"] = True
                        if result is not None:
                            post_num_valid = result.num_valid
                            post_diversity_milli = int(result.diversity * 1000)
                            # Insert into the bounded heap. Always
                            # admits when there's room; otherwise the
                            # iter only got here because it beat the
                            # worst-energy entry (ratchet gate above),
                            # so we evict that tail and re-sort.
                            if len(top_k) < top_k_cap:
                                top_k.append(result)
                                top_k.sort(key=lambda r: r.energy)
                                stored_replaced = True
                            elif result.energy < top_k[-1].energy:
                                top_k[-1] = result
                                top_k.sort(key=lambda r: r.energy)
                                stored_replaced = True

                    self.timing_stats['postprocessing'].append(
                        (time.time() - postprocess_start) * 1e6,
                    )

                    # Submit gate. The chain re-derives each submitted
                    # solution's energy and filters with strict
                    # ``< max_energy_milli`` before counting; gating on
                    # ``energy`` (best of set) would let through proofs
                    # whose mid-pack solutions get rejected, producing
                    # ``InsufficientSolutions`` despite the headline
                    # energy clearing. ``submit_floor_energy`` is the
                    # chain-equivalent recompute, so it's what the gate
                    # must compare against.
                    #
                    # Walk the stash in energy-ascending order (best
                    # first). Prefer the best candidate whose floor
                    # clears; fall back to worse-energy stash entries
                    # if a better candidate's floor sits above the
                    # live threshold (the two can diverge — a tighter
                    # solution set can have a lower best but a higher
                    # worst, leaving a wider-spread candidate eligible
                    # for submission while the headline winner isn't).
                    result = None
                    for candidate in top_k:
                        floor_energy = (
                            candidate.submit_floor_energy
                            if candidate.submit_floor_energy is not None
                            else candidate.energy
                        )
                        if int(floor_energy * 1000) < live_threshold_milli:
                            result = candidate
                            break

                    # ratchet_threshold is float("inf") on the first iter
                    # before anything is stored — log None there since
                    # there's no meaningful prior to display.
                    ratchet_threshold_milli_log = (
                        int(ratchet_threshold * 1000)
                        if math.isfinite(ratchet_threshold)
                        else None
                    )
                    attempt_log_kwargs.update(
                        threshold_milli=live_threshold_milli,
                        ratchet_threshold_milli=ratchet_threshold_milli_log,
                        num_valid=post_num_valid,
                        diversity_milli=post_diversity_milli,
                        stored_as_best=stored_replaced,
                        result_kind=(
                            "submitted" if result is not None
                            else "stored" if stored_replaced
                            else "rejected"
                        ),
                    )
                else:
                    # ---- Mempool path (unchanged strict semantics) ---
                    result = self.evaluate_sampleset(
                        sampleset, requirements, nodes, edges,
                        nonce, salt, prev_timestamp, start_time,
                    )

                    self.timing_stats['postprocessing'].append(
                        (time.time() - postprocess_start) * 1e6,
                    )
                    # Mempool path may have ``difficulty_energy = +inf``
                    # (unbounded threshold for jobs with no min_energy
                    # floor). int() would overflow — use None instead.
                    threshold_milli_log: Optional[int] = None
                    if math.isfinite(requirements.difficulty_energy):
                        threshold_milli_log = int(
                            requirements.difficulty_energy * 1000,
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
                # Compute solution_meta scalars + capture top-5
                # solutions. Meta is always embedded in the attempt
                # log; top-5 spins go to disk only when this iter is
                # stored or submitted (see write below).
                sol_meta, top_5_sols, top_5_es = compute_solution_meta(
                    sampleset, requirements.difficulty_energy,
                )
                attempt_log_kwargs["solution_meta"] = sol_meta
                attempt_log.record(**attempt_log_kwargs)

                # SolutionStore — archive top-5 spin configs only when
                # the chain ratchet kept this candidate. The result_kind
                # set on the attempt drives the gate; "submitted" and
                # "stored" both qualify since both produce a candidate
                # we (or someone) might reproduce for analysis.
                attempt_result_kind = attempt_log_kwargs.get("result_kind")
                if attempt_result_kind in ("stored", "submitted") and top_5_sols:
                    nonce_hex = (
                        nonce.hex() if isinstance(nonce, (bytes, bytearray))
                        else f"{int(nonce):064x}"
                    )
                    solution_store.record(
                        dispatch_id=dispatch_id_for_log,
                        iter_num=progress + 1,
                        nonce_hex=nonce_hex,
                        salt_hex=salt.hex(),
                        top_5_solutions_hex=[
                            pack_spins_hex(s) for s in top_5_sols
                        ],
                        top_5_energies=top_5_es,
                        result_kind=attempt_result_kind,
                    )

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
                    nonce_disp = (
                        f"0x{nonce.hex()[:16]}..."
                        if isinstance(nonce, (bytes, bytearray))
                        else str(nonce)
                    )
                    self.logger.info(
                        f"[work-item {_work_tag(context)}] mined! "
                        f"nonce={nonce_disp} salt=0x{salt.hex()[:8]}... "
                        f"energy={result.energy:.2f} "
                        f"solutions={result.num_valid} "
                        f"diversity={result.diversity:.3f} "
                        f"attempt_time={result.mining_time:.2f}s "
                        f"total_time={time.time() - start_time:.2f}s"
                    )
                    return result

                progress += 1
                if progress % PROGRESS_LOG_INTERVAL == 0:
                    # `self.top_attempts` is intentionally not maintained in
                    # substrate mode — no best-energy field to surface here.
                    self.logger.info(
                        "mine_work_item progress: %d attempts | "
                        "sweeps=%d reads=%d",
                        progress, current_num_sweeps, num_reads,
                    )
            self.logger.info("mine_work_item: stopped, no valid result")
            return None
        finally:
            self.mining = False
            if self._dropped_results:
                self.logger.info(
                    "mine_work_item: %d QPU results dropped under "
                    "result-queue backpressure", self._dropped_results,
                )
            # Stop the pump first: signal it, let the QPU stream observe the
            # stop and return from its in-flight next(), then join. Only
            # after the pump thread is dead is it safe for _post_mine_cleanup
            # to close the generator (close() on a generator executing in
            # another thread raises "already executing").
            if pump_stop is not None:
                pump_stop.set()
            if pump_thread is not None:
                pump_thread.join(timeout=5.0)
                if pump_thread.is_alive():
                    self.logger.warning(
                        "mine_work_item: result pump did not join in 5s — "
                        "pump thread still alive (QPU stream pump-stop wiring "
                        "closes this race)",
                    )
            self._pump_stop = None
            # Tear down the feeder before delegating to subclass cleanup.
            if self._feeder is not None:
                try:
                    self._feeder.stop()
                finally:
                    self._feeder = None
            # Persist any buffered aggregate metadata for this dispatch.
            attempt_logger = getattr(self, "_attempt_logger", None)
            if attempt_logger is not None:
                attempt_logger.flush()
            self._post_mine_cleanup()

    # ------------------------------------------------------------------
    # Out-of-band result pump
    # ------------------------------------------------------------------

    def _result_pump(
        self,
        result_queue: "queue.Queue",
        pump_stop: threading.Event,
        sample_kwargs: Dict[str, Any],
    ) -> None:
        """Drive the streaming sampler on a background thread.

        Repeatedly calls ``_sample_batch`` (the only ``next()`` driver of
        the QPU generator) and pushes ``_PumpedResult`` onto a bounded
        queue. On a full queue the newest result is dropped and counted —
        the QPU pump must never block. Pushes ``_PUMP_DONE`` on exit so the
        consumer's blocking ``get`` always unblocks.
        """
        try:
            while not pump_stop.is_set():
                before = len(self.timing_stats["qpu_access_time"])
                batch = self._sample_batch(
                    sample_kwargs["prev_hash"],
                    sample_kwargs["miner_id"],
                    sample_kwargs["cur_index"],
                    sample_kwargs["nodes"],
                    sample_kwargs["edges"],
                    num_reads=sample_kwargs["num_reads"],
                    num_sweeps=sample_kwargs["num_sweeps"],
                    **sample_kwargs["extra"],
                )
                if batch is None:
                    break  # stream exhausted
                nonce, salt, sampleset = batch[0]
                qpu_list = self.timing_stats["qpu_access_time"]
                qpu_us = (
                    int(qpu_list[-1]) if len(qpu_list) > before else None
                )
                item = _PumpedResult(nonce, salt, sampleset, qpu_us)
                try:
                    result_queue.put_nowait(item)
                except queue.Full:
                    self._dropped_results += 1
        except Exception as exc:  # pump must not crash silently
            self.logger.error("result pump error: %s", exc)
        finally:
            try:
                result_queue.put_nowait(_PUMP_DONE)
            except queue.Full:
                # Consumer already gone; drain one slot and signal. The
                # evicted slot held a real _PumpedResult, so count it as
                # dropped before reusing the freed space for the sentinel.
                try:
                    result_queue.get_nowait()
                    self._dropped_results += 1
                    result_queue.put_nowait(_PUMP_DONE)
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------
    # Hook methods (override in subclasses as needed)
    # ------------------------------------------------------------------

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

