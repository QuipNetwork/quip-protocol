# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the process-model stream driver path in BaseMiner.

The in-worker pump thread was replaced by a stream-driver PROCESS that
writes samplesets into a SharedSampleRing and enqueues small descriptors
(see QPU/stream_driver.py). These tests exercise the consumer side
(``_acquire_result`` reading the ring, slot release, budget feed, clean
teardown) without contacting a QPU — ``build_fake_persistent_context``
stands in for the persistent stream-driver context.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from shared.allowed_value_spec import AllowedValueSet
from shared.base_miner import (
    BaseMiner,
    _ACQUIRE_DONE,
    _ACQUIRE_OK,
    _ACQUIRE_STOP,
    _SharedSampleSet,
)
from shared.miner_types import MiningResult
from shared.proc_util import terminate_join
from shared.shared_sample_ring import SharedSampleRing
from substrate.types import SubstrateDifficulty, SubstrateMiningContext

_FAKE_CTX = "tests.fakes.fake_stream:build_fake_persistent_context"
_FAKE_RAISING = "tests.fakes.fake_stream:build_fake_raising_context"

_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))


def _streaming_context(nodes=(0, 1, 2)) -> SubstrateMiningContext:
    """Small synthetic substrate context for the driver-path tests."""
    nodes = list(nodes)
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return SubstrateMiningContext(
        last_proof_block_hash=b"\xab" * 32,
        topology_hash=b"\xcd" * 32,
        nodes=nodes,
        edges=edges,
        difficulty=SubstrateDifficulty(
            min_solutions=1,
            max_energy_milli=0,
            min_diversity_milli=0,
        ),
        miner_account_bytes=b"\x42" * 32,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x55" * 32,
        block_number=1,
    )


class _DriverMiner(BaseMiner):
    """STREAMING_PUMP + DRIVER_OWNS_FEEDER miner driven by the fake context."""

    STREAMING_PUMP = True
    DRIVER_OWNS_FEEDER = True
    STREAM_FACTORY_DOTTED = _FAKE_CTX
    RESULT_QUEUE_MAXSIZE = 4

    def __init__(self, factory: str = _FAKE_CTX) -> None:
        super().__init__("driver-test", sampler=object(), miner_type="QPU")
        self.queue_depth = 2
        self.time_manager = None
        self.STREAM_FACTORY_DOTTED = factory

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {
            "num_reads": 8,
            "annealing_time": 80.0,
            "energy_threshold": requirements.difficulty_energy,
        }

    def _sample(self, *a, **k):
        raise AssertionError("single-shot _sample must not run on driver path")

    def _sample_batch(self, *a, **k):
        raise AssertionError("_sample_batch must not run on driver path")

    def evaluate_sampleset(self, *args, **kwargs):
        return None  # never a winner


# ----------------------------------------------------------------------
# Consumer-side _acquire_result against a hand-built ring + descriptor queue
# ----------------------------------------------------------------------


class _RingConsumer(BaseMiner):
    """Bare consumer used to drive ``_acquire_result`` directly."""

    STREAMING_PUMP = True
    DRIVER_OWNS_FEEDER = True

    def __init__(self) -> None:
        super().__init__("ring-consumer", sampler=object(), miner_type="QPU")
        self.time_manager = None

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        return {"num_reads": 4}

    def _sample(self, *a, **k):
        raise AssertionError("no sample on driver path")

    def _sample_batch(self, *a, **k):
        raise AssertionError("no batch on driver path")


def _sample_ctx() -> dict:
    return {
        "prev_hash": b"\x00" * 32, "miner_id": "m", "cur_index": 0,
        "nodes": [0, 1, 2], "edges": [], "num_reads": 4, "num_sweeps": 1,
        "extra": {},
    }


def _noop() -> None:
    """Module-level target so spawn can pickle it (used for the dead-driver
    liveness test — the process exits immediately)."""
    return None


def test_acquire_result_dead_driver_without_sentinel_is_done():
    """A driver that died without enqueuing None must not hang the consumer.

    The cooperative end-of-stream path sends a trailing ``None`` from the
    driver's ``finally``; a hard crash (SIGKILL/OOM/C-extension abort) skips
    it. ``_acquire_result`` must detect the dead ``driver_proc`` and return
    DONE rather than draining an empty queue forever.
    """
    consumer = _RingConsumer()
    # Empty queue: no descriptor and no None sentinel ever arrives.
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    dead = mp.get_context("spawn").Process(target=_noop)
    dead.start()
    dead.join(timeout=5.0)
    assert not dead.is_alive()
    try:
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            driver_proc=dead,
        )
        assert acquired.action == _ACQUIRE_DONE
    finally:
        terminate_join(dead, 2.0)


def test_ensure_driver_spawns_non_daemon_driver():
    """The persistent driver must be non-daemon (its feeder forks children)."""
    miner = _DriverMiner()
    sample_ctx = {
        "num_reads": 8, "nodes": [0, 1, 2], "edges": [],
        "annealing_time": 80.0, "energy_threshold": -1.0,
        "last_proof_block_hash": b"\xab" * 32, "miner_bytes": b"\x42" * 32,
    }
    try:
        assert miner._ensure_driver(sample_ctx) is True
        assert miner._driver_proc is not None
        assert miner._driver_proc.daemon is False
        # Idempotent: a second call reuses the same process.
        pid = miner._driver_proc.pid
        assert miner._ensure_driver(sample_ctx) is True
        assert miner._driver_proc.pid == pid
    finally:
        miner._close_driver()
    assert miner._ring is None
    assert miner._driver_proc is None


def test_acquire_result_reads_descriptor_from_ring():
    consumer = _RingConsumer()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        sample = np.ones((4, 3), np.int8)
        energy = np.array([-1.0, -2.0, -3.0, -4.0], np.float64)
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, sample, energy)
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 51010, 1))

        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=1,
        )
        assert acquired.action == _ACQUIRE_OK
        assert acquired.ring_slot == slot
        assert isinstance(acquired.sampleset, _SharedSampleSet)
        # Zero-copy view sees what the producer wrote.
        np.testing.assert_array_equal(acquired.sampleset.record.energy, energy)
        np.testing.assert_array_equal(acquired.sampleset.record.sample, sample)
        # QPU access time rode the descriptor and fed timing_stats.
        assert acquired.qpu_access_time_us == 51010
        assert consumer.timing_stats["qpu_access_time"][-1] == 51010
        # Release the slot before closing (drop the exported view first).
        acquired = None
        ring.release(slot)
    finally:
        ring.close_unlink()


def test_acquire_result_none_descriptor_is_done():
    consumer = _RingConsumer()
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    desc_q.put(None)
    stop = mp.Event()
    acquired = consumer._acquire_result(
        stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
    )
    assert acquired.action == _ACQUIRE_DONE


def test_acquire_result_stop_event_returns_stop():
    consumer = _RingConsumer()
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()  # left empty
    stop = mp.Event()
    stop.set()
    acquired = consumer._acquire_result(
        stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
    )
    assert acquired.action == _ACQUIRE_STOP


def test_acquire_result_feeds_budget_time_manager():
    class _Mgr:
        def __init__(self):
            self.recorded = []

        def record_block_time(self, us):
            self.recorded.append(us)

    consumer = _RingConsumer()
    consumer.time_manager = _Mgr()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q: "mp.Queue" = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, np.ones((4, 3), np.int8),
                   np.zeros(4, np.float64))
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 4242, 1))
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=1,
        )
        assert acquired.action == _ACQUIRE_OK
        assert consumer.time_manager.recorded == [4242]
        acquired = None
        ring.release(slot)
    finally:
        ring.close_unlink()


def test_acquire_result_drops_stale_generation_descriptor():
    """A descriptor from a superseded round is released + skipped, not OK."""
    consumer = _RingConsumer()
    ring = SharedSampleRing(slots=2, max_rows=4, max_cols=3)
    consumer._ring = ring
    desc_q = mp.get_context("spawn").Queue()
    stop = mp.Event()
    try:
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, np.ones((4, 3), np.int8), np.zeros(4, np.float64))
        # Stale gen=1 descriptor, then end-of-stream. Consumer wants gen=2.
        desc_q.put((slot, 4, 3, b"\x01" * 32, b"\x02" * 32, 0, 1))
        desc_q.put(None)
        acquired = consumer._acquire_result(
            stop, desc_q, preprocess_start=0.0, sample_ctx=_sample_ctx(),
            generation=2,
        )
        # Stale one skipped; stream then ended -> DONE.
        assert acquired.action == _ACQUIRE_DONE
        # The skipped slot was released back to the free-list (no leak): both
        # slots of the 2-slot ring are claimable again. (claim_free is a FIFO
        # so the reclaimed index need not equal ``slot``.)
        claimed = {ring.claim_free(timeout=1.0) for _ in range(2)}
        assert claimed == {0, 1}
        for s in claimed:
            ring.release(s)
    finally:
        ring.close_unlink()


# ----------------------------------------------------------------------
# Integration: mine_work_item spawns the driver, consumes, and tears down
# ----------------------------------------------------------------------


def test_mine_work_item_persists_driver_across_dispatches():
    """Two dispatches reuse ONE driver process (same pid, not respawned)."""
    ctx = _streaming_context()
    miner = _DriverMiner(factory=_FAKE_CTX)  # infinite stream (n=0)
    try:
        import threading
        import time as _t

        pids = []
        for _ in range(2):
            stop = mp.Event()
            done = threading.Event()

            def _run(s=stop, d=done):
                miner.mine_work_item(ctx, s)
                d.set()

            t = threading.Thread(target=_run)
            t.start()
            _t.sleep(0.4)
            pids.append(miner._driver_proc.pid)
            stop.set()
            t.join(timeout=15.0)
            assert not t.is_alive()
            assert miner._driver_proc is not None
            assert miner._driver_proc.is_alive()
            assert miner._ring is not None
        # The SECOND dispatch reused the SAME driver process (not respawned).
        assert pids[0] == pids[1]
        assert miner._generation == 2
    finally:
        miner._close_driver()
    assert miner._ring is None
    assert miner._driver_proc is None


def test_mine_work_item_stops_promptly_on_stop_event():
    ctx = _streaming_context()
    miner = _DriverMiner(factory=_FAKE_CTX)  # infinite
    stop = mp.Event()

    import threading
    import time

    done = threading.Event()
    raised: list = []

    def _run():
        try:
            miner.mine_work_item(ctx, stop)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_run, name="mine-loop")
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=15.0)
    try:
        assert not t.is_alive(), "mine_work_item did not stop on stop_event"
        assert done.is_set()
        assert raised == [], f"mine_work_item raised: {raised[0] if raised else None}"
        # The driver is NOT torn down by a dispatch stop — it persists.
        assert miner._driver_proc is not None and miner._driver_proc.is_alive()
    finally:
        miner._close_driver()


# ----------------------------------------------------------------------
# Headline regression: lookahead -> decay -> aggressive submit (end-to-end)
# ----------------------------------------------------------------------


class _DecayCandidateMiner(_DriverMiner):
    """Driver-path miner whose stream candidate clears only a looser decay.

    ``evaluate_sampleset`` returns the stashed candidate (floor -14500) on the
    FIRST sampleset only, then ``None`` — so the candidate is stashed exactly
    once and must PERSIST in ``top_k`` across later iterations for the submit
    gate to fire after decay. (If a rule-1 regression cleared the stash on a
    decay iteration, the candidate would be gone and the gate would never
    fire — the test would hang→fail, which is the contract.)
    """

    def __init__(self):
        super().__init__(factory=_FAKE_CTX)  # infinite fake stream
        self._eval_calls = 0

    def evaluate_sampleset(self, sampleset, requirements, nodes, edges,
                           nonce, salt, *args, **kwargs):
        self._eval_calls += 1
        if self._eval_calls > 1:
            return None  # stash exactly once; candidate must persist
        return MiningResult(
            miner_id=self.miner_id, miner_type=self.miner_type,
            nonce=bytes(nonce) if not isinstance(nonce, bytes) else nonce,
            salt=salt, timestamp=0, prev_timestamp=0,
            solutions=[[1, -1, 1]], energy=-14600.0, diversity=1.0,
            num_valid=1, mining_time=0, node_list=list(nodes),
            edge_list=list(edges), submit_floor_energy=-14500.0,
        )


def test_aggressive_submit_on_decay_end_to_end():
    """HEADLINE GUARD: lookahead → decay → aggressive submit survives the
    persistent multiprocessing model.

    The candidate clears a future, looser decay level (floor -14500) but not
    the strict initial live threshold (-14800). Asserts, in order:
      1. Under the strict threshold it is stashed, not submitted (the dispatch
         keeps running) and preview_cb fired (anticipatory path).
      2. Decaying the live threshold past the floor — with NO head change —
         makes mine_work_item return that exact candidate.
      3. Throughout, the driver pid is unchanged and the generation did not
         bump (decay stayed within one round); decay was forwarded as a
         'threshold' command, never a second 'switch'.
    """
    import threading
    import time as _t

    ctx = _streaming_context()
    miner = _DecayCandidateMiner()
    # Strict live threshold the candidate's floor (-14500) does NOT clear.
    miner._live_max_energy_milli = mp.Value("q", -14800_000)

    previews: list = []
    result_box: list = []

    # Spy on ctl_q puts to prove decay is a threshold update, not a switch.
    sent: list = []

    orig_ensure = miner._ensure_driver

    def _ensure_spy(sample_ctx):
        ready = orig_ensure(sample_ctx)
        if ready and miner._ctl_q is not None and not getattr(
            miner._ctl_q, "_quip_spied", False,
        ):
            real_put = miner._ctl_q.put

            def _put(item, *a, **k):
                sent.append(item[0] if isinstance(item, tuple) else item)
                return real_put(item, *a, **k)

            miner._ctl_q.put = _put
            miner._ctl_q._quip_spied = True
        return ready

    miner._ensure_driver = _ensure_spy

    stop = mp.Event()

    def _run():
        result_box.append(
            miner.mine_work_item(ctx, stop, preview_cb=previews.append),
        )

    t = threading.Thread(target=_run, name="decay-mine")
    t.start()
    try:
        # Phase 1: candidate stashed + previewed, but NOT submitted (strict).
        deadline = _t.monotonic() + 10.0
        while not previews and _t.monotonic() < deadline:
            _t.sleep(0.02)
        assert previews, "candidate was never previewed (lookahead broken)"
        assert not result_box, "submitted under the strict threshold (rule 1 broke)"
        gen_at_stash = miner._generation
        pid = miner._driver_proc.pid

        # Phase 2: decay past the floor with NO head change -> submit fires.
        with miner._live_max_energy_milli.get_lock():
            miner._live_max_energy_milli.value = -14400_000
        t.join(timeout=15.0)
        assert not t.is_alive(), "submit gate did not fire on decay (rule 3)"
        result = result_box[0]
        assert result is not None
        assert result.submit_floor_energy == -14500.0

        # Phase 3: same generation, same driver pid; decay was a 'threshold'.
        assert miner._generation == gen_at_stash
        assert miner._driver_proc.pid == pid
        assert sent.count("switch") == 1, "decay caused a spurious round switch"
        assert "threshold" in sent, "decay was not forwarded to the driver gate"
    finally:
        stop.set()
        t.join(timeout=15.0)
        miner._close_driver()


# ----------------------------------------------------------------------
# Teardown semantics against the persistent model
# ----------------------------------------------------------------------


class _WinningMiner(_DriverMiner):
    """First evaluated sampleset is a winner (exercises win early-return)."""

    def __init__(self):
        super().__init__(factory=_FAKE_CTX)

    def evaluate_sampleset(self, sampleset, requirements, nodes, edges,
                           nonce, salt, *args, **kwargs):
        return MiningResult(
            miner_id=self.miner_id, miner_type=self.miner_type,
            nonce=bytes(nonce) if not isinstance(nonce, bytes) else nonce,
            salt=salt, timestamp=0, prev_timestamp=0,
            solutions=[[1, -1, 1]], energy=-15000.0, diversity=1.0,
            num_valid=1, mining_time=0, node_list=list(nodes),
            edge_list=list(edges), submit_floor_energy=-15000.0,
        )


def test_win_then_close_no_buffererror():
    """A win returns cleanly with the driver still alive; close() unlinks."""
    ctx = _streaming_context()
    miner = _WinningMiner()
    stop = mp.Event()
    try:
        result = miner.mine_work_item(ctx, stop)
        assert result is not None and result.energy == -15000.0
        # Driver + ring persist after a win (not torn down per-dispatch).
        assert miner._driver_proc is not None and miner._driver_proc.is_alive()
        assert miner._ring is not None
    finally:
        miner._close_driver()  # must not raise BufferError
    assert miner._ring is None
    assert miner._driver_proc is None


def test_close_driver_reaps_and_unlinks():
    """_close_driver reaps the process and unlinks the ring with no leak."""
    ctx = _streaming_context()
    miner = _DriverMiner()
    stop = mp.Event()
    import threading
    import time as _t
    t = threading.Thread(target=lambda: miner.mine_work_item(ctx, stop))
    t.start()
    try:
        # Poll for the persistent ring to come up (driver spawned) rather than
        # a fixed sleep, so a loaded box can't race us into a None deref.
        deadline = _t.monotonic() + 5.0
        while miner._ring is None and _t.monotonic() < deadline:
            _t.sleep(0.02)
        assert miner._ring is not None, "driver ring never came up"
        names = list(miner._ring.names)
    finally:
        stop.set()
        t.join(timeout=15.0)
        miner._close_driver()
    assert miner._driver_proc is None
    assert miner._ring is None
    import pytest
    from multiprocessing import shared_memory
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0])
