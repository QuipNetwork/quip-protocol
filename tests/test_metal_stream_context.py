# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""MetalStreamContext: switch reseeds + bumps gen; iter_results yields tagged sets."""
from __future__ import annotations

from types import SimpleNamespace

from GPU.metal_stream import MetalStreamContext


class _FakeModel:
    def __init__(self, n):
        self.h = {0: 0.0}
        self.J = {}
        self.nonce = bytes([n % 256]) * 32
        self.salt = bytes([n % 256]) * 32


class _FakeFeeder:
    """Stands in for RandomIsingFeeder; records reseed/stop."""

    def __init__(self, **_kw):
        self.reseeds = 0
        self.stopped = False

    def reseed(self, lpbh, miner_bytes):
        self.reseeds += 1

    def stop(self):
        self.stopped = True


class _FakeSampler:
    """Yields (model, sampleset) pairs; sampleset has .record.sample/.energy."""

    def __init__(self):
        self.closed = False

    def sample_ising_streaming(self, feeder, **_kw):
        i = 0
        while True:
            ss = SimpleNamespace(record=SimpleNamespace(sample=[[1, -1]], energy=[0.0]))
            yield _FakeModel(i), ss
            i += 1

    def close(self):
        self.closed = True


def _ctx():
    return MetalStreamContext(
        sampler=_FakeSampler(),
        nodes=[0, 1],
        edges=[(0, 1)],
        feeder_buffer_size=4,
        num_reads=8,
        num_sweeps=64,
        max_threadgroups=4,
        feeder_factory=_FakeFeeder,
    )


def test_switch_builds_feeder_and_bumps_generation():
    ctx = _ctx()
    assert ctx.generation == 0
    ctx.apply_command(("switch", 1, b"\x00" * 32, b"\x01" * 32, 0, 8, 0.0))
    assert ctx.generation == 1
    ctx.apply_command(("switch", 2, b"\x02" * 32, b"\x01" * 32, 0, 8, 0.0))
    assert ctx.generation == 2
    assert ctx._feeder.reseeds == 1  # built on switch 1, reseeded on switch 2
    ctx.cleanup()


def test_iter_results_yields_tagged_until_stop():
    import multiprocessing as mp

    stop = mp.get_context("spawn").Event()
    ctx = MetalStreamContext(
        sampler=_FakeSampler(), nodes=[0, 1], edges=[(0, 1)],
        feeder_buffer_size=4, num_reads=8, num_sweeps=64, max_threadgroups=4,
        feeder_factory=_FakeFeeder, stop_event=stop,
    )
    ctx.apply_command(("switch", 1, b"\x00" * 32, b"\x01" * 32, 0, 8, 0.0))
    out = []
    for model, ss, gen in ctx.iter_results():
        out.append((model, ss, gen))
        if len(out) >= 3:
            stop.set()
    assert len(out) == 3
    assert all(gen == 1 for _m, _s, gen in out)
    ctx.cleanup()


def test_threshold_is_noop_and_pause_stops_production():
    ctx = _ctx()
    ctx.apply_command(("switch", 1, b"\x00" * 32, b"\x01" * 32, 0, 8, 0.0))
    ctx.apply_command(("threshold", 1, 1234))  # no error, no effect
    ctx.apply_command(("pause", 1))
    assert ctx._paused is True
    assert list(ctx.iter_results()) == []
    ctx.cleanup()
