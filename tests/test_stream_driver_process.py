# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Integration tests: stream_driver_main over a fake persistent context."""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time

from shared.proc_util import terminate_join
from shared.ring_views import SampleView

_FAKE_CTX = "tests.fakes.fake_stream:build_fake_persistent_context"
_FAKE_RAISING = "tests.fakes.fake_stream:build_fake_raising_context"


class _RecordingCtx:
    """Records apply_command calls for _coalesce_ctl_q precedence tests."""

    def __init__(self, generation=0):
        self.generation = generation
        self.applied = []

    def apply_command(self, cmd):
        self.applied.append(cmd)
        if cmd[0] == "switch":
            self.generation = int(cmd[1])


def _drain_queue_into(cmds):
    q = _queue.Queue()
    for c in cmds:
        q.put(c)
    return q


def test_coalesce_switch_beats_pause_in_one_drain():
    """A switch and a pause coalesced together → switch wins, pause ignored."""
    from QPU.stream_driver import _coalesce_ctl_q

    ctx = _RecordingCtx(generation=1)
    q = _drain_queue_into([
        ("pause", 1),
        ("switch", 2, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0),
    ])
    assert _coalesce_ctl_q(ctx, q) == "ok"
    kinds = [c[0] for c in ctx.applied]
    assert "switch" in kinds and "pause" not in kinds


def test_coalesce_applies_lone_pause():
    """A pause with no competing switch is applied."""
    from QPU.stream_driver import _coalesce_ctl_q

    ctx = _RecordingCtx(generation=5)
    q = _drain_queue_into([("pause", 5)])
    assert _coalesce_ctl_q(ctx, q) == "ok"
    assert ctx.applied == [("pause", 5)]


def test_coalesce_ignores_stale_pause_for_old_generation():
    """A pause for a superseded generation is dropped (the live round won)."""
    from QPU.stream_driver import _coalesce_ctl_q

    ctx = _RecordingCtx(generation=7)
    q = _drain_queue_into([("pause", 4)])  # gen 4 < live gen 7 → stale
    assert _coalesce_ctl_q(ctx, q) == "ok"
    assert ctx.applied == []


def _spawn_driver(ring, desc_q, ctl_q, stop, factory, factory_kwargs):
    from QPU.stream_driver import stream_driver_main

    proc = mp.get_context("spawn").Process(
        target=stream_driver_main,
        args=(ring.attach_args(), desc_q, ctl_q, stop, factory, factory_kwargs),
        daemon=True,
    )
    proc.start()
    return proc


def test_driver_produces_generation_tagged_descriptors():
    ctx = mp.get_context("spawn")
    ring = SampleView(slots=4, max_rows=8, max_cols=3)
    desc_q, ctl_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = _spawn_driver(
        ring,
        desc_q,
        ctl_q,
        stop,
        _FAKE_CTX,
        {"num_reads": 8, "nodes": [0, 1, 2], "n": 5},
    )
    ctl_q.put(("switch", 3, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    seen = 0
    try:
        for _ in range(5):
            item = desc_q.get(timeout=10.0)
            assert item is not None
            slot, nr, nc, nonce, salt, qpu, gen = item[:7]
            assert gen == 3  # tagged with the live generation
            s, _e = ring.read(slot, nr, nc)
            assert s.shape == (nr, nc)
            del s, _e
            ring.release(slot)
            seen += 1
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()
    assert seen == 5


def test_driver_shuts_down_on_ctl_q_none():
    ctx = mp.get_context("spawn")
    ring = SampleView(slots=2, max_rows=8, max_cols=3)
    desc_q, ctl_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = _spawn_driver(
        ring,
        desc_q,
        ctl_q,
        stop,
        _FAKE_CTX,
        {"num_reads": 8, "nodes": [0, 1, 2], "n": 0},  # infinite
    )
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    try:
        assert desc_q.get(timeout=10.0) is not None  # streaming
        ctl_q.put(None)  # shutdown sentinel
        saw_none = False
        for _ in range(200):
            if desc_q.get(timeout=10.0) is None:
                saw_none = True
                break
        assert saw_none, "driver never enqueued end-of-stream None"
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()


def test_driver_factory_error_yields_none_not_hang():
    ctx = mp.get_context("spawn")
    ring = SampleView(slots=2, max_rows=8, max_cols=3)
    desc_q, ctl_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = _spawn_driver(
        ring,
        desc_q,
        ctl_q,
        stop,
        _FAKE_RAISING,
        {"num_reads": 8, "nodes": [0, 1, 2]},
    )
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    try:
        assert desc_q.get(timeout=10.0) is None, (
            "factory error did not produce end-of-stream None"
        )
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()


def test_driver_pauses_then_resumes_on_switch():
    """A ('pause', gen) idles the driver (production stops); a later switch
    resumes it under the new generation."""
    ctx = mp.get_context("spawn")
    ring = SampleView(slots=4, max_rows=8, max_cols=3)
    desc_q, ctl_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = _spawn_driver(
        ring,
        desc_q,
        ctl_q,
        stop,
        _FAKE_CTX,
        {"num_reads": 8, "nodes": [0, 1, 2], "n": 0},  # infinite
    )
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    try:
        first = desc_q.get(timeout=10.0)
        assert first[6] == 1
        ring.release(first[0])

        # Pause: production must stop. Drain any already-queued descriptors,
        # then assert the queue stays empty for a quiet window.
        ctl_q.put(("pause", 1))
        time.sleep(0.3)
        while True:
            try:
                item = desc_q.get_nowait()
            except _queue.Empty:
                break
            if item is not None:
                ring.release(item[0])
        quiet = False
        try:
            stale = desc_q.get(timeout=1.0)
            if stale is not None:
                ring.release(stale[0])
        except _queue.Empty:
            quiet = True
        assert quiet, "driver kept producing after pause"

        # Resume via a new head: generation advances and production restarts.
        ctl_q.put(("switch", 2, b"\x09" * 32, b"\x02" * 32, 0, 8, 80.0))
        resumed = desc_q.get(timeout=10.0)
        assert resumed[6] == 2
        ring.release(resumed[0])
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()


def test_driver_switch_mid_stream_advances_generation():
    """A switch mid-stream advances the descriptor generation; the driver
    drops the straddler from the old generation (its stale-discard filter)."""
    ctx = mp.get_context("spawn")
    ring = SampleView(slots=4, max_rows=8, max_cols=3)
    desc_q, ctl_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = _spawn_driver(
        ring,
        desc_q,
        ctl_q,
        stop,
        _FAKE_CTX,
        {"num_reads": 8, "nodes": [0, 1, 2], "n": 0},  # infinite
    )
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))
    try:
        first = desc_q.get(timeout=10.0)
        assert first[6] == 1
        ring.release(first[0])
        # Advance to generation 2 mid-stream.
        ctl_q.put(("switch", 2, b"\x09" * 32, b"\x02" * 32, 0, 8, 80.0))
        deadline = time.monotonic() + 10.0
        saw_gen2 = False
        while time.monotonic() < deadline:
            item = desc_q.get(timeout=10.0)
            ring.release(item[0])
            gen = item[6]
            assert gen in (1, 2), f"unexpected generation {gen}"
            if gen == 2:
                saw_gen2 = True
                break
        assert saw_gen2, "descriptors never advanced to the new generation"
    finally:
        stop.set()
        assert terminate_join(proc, 5.0)
        ring.close_unlink()
