# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Generic StreamContext: switch builds feeder from spec; yields tagged sets."""
from __future__ import annotations

from types import SimpleNamespace

from shared.stream_context import StreamContext


class _FakeModel:
    def __init__(self, n):
        self.h, self.J = {0: 0.0}, {}
        self.nonce = bytes([n % 256]) * 32
        self.salt = bytes([n % 256]) * 32


class _FakeFeeder:
    def __init__(self, *_a, **_k):
        self.reseeds = 0
        self.stopped = False

    def reseed(self, *_a):
        self.reseeds += 1

    def stop(self):
        self.stopped = True


class _FakeSampler:
    def sample_ising_streaming(self, feeder, **_k):
        i = 0
        while True:
            yield _FakeModel(i), SimpleNamespace(
                record=SimpleNamespace(sample=[[1, -1]], energy=[0.0]))
            i += 1

    def close(self):
        pass


def _fake_builder(spec, nodes, edges, buffer_size):
    return _FakeFeeder()


def _ctx(**kw):
    return StreamContext(
        sampler=_FakeSampler(), nodes=[0, 1], edges=[(0, 1)],
        feeder_buffer_size=4, num_reads=8, num_sweeps=64,
        feeder_builder=_fake_builder, **kw)


def _switch(gen, num_reads=8, num_sweeps=64):
    # ("switch", gen, lpbh, miner_bytes, thr, num_reads, anneal, num_sweeps, spec)
    return ("switch", gen, b"\x00" * 32, b"\x01" * 32, 0, num_reads, 0.0,
            num_sweeps, ("pow", b"\x00" * 32, b"\x01" * 32))


def test_switch_builds_feeder_and_bumps_generation():
    ctx = _ctx()
    assert ctx.generation == 0
    ctx.apply_command(_switch(1))
    assert ctx.generation == 1
    first = ctx._feeder
    ctx.apply_command(_switch(2))  # same pow kind -> reseed, not rebuild
    assert ctx.generation == 2
    assert ctx._feeder is first and first.reseeds == 1
    ctx.cleanup()


def test_switch_updates_num_sweeps_and_reads():
    ctx = _ctx()
    ctx.apply_command(_switch(1, num_reads=16, num_sweeps=128))
    assert ctx._num_reads == 16 and ctx._num_sweeps == 128
    ctx.cleanup()


def test_iter_results_yields_tagged_until_stop():
    import multiprocessing as mp
    stop = mp.get_context("spawn").Event()
    ctx = _ctx(stop_event=stop)
    ctx.apply_command(_switch(1))
    out = []
    for model, ss, gen in ctx.iter_results():
        out.append(gen)
        if len(out) >= 3:
            stop.set()
    assert out == [1, 1, 1]
    ctx.cleanup()


def test_threshold_noop_and_pause_stops():
    ctx = _ctx()
    ctx.apply_command(_switch(1))
    ctx.apply_command(("threshold", 1, 123))
    ctx.apply_command(("pause", 1))
    assert ctx._paused is True
    assert list(ctx.iter_results()) == []
    ctx.cleanup()
