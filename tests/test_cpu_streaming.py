"""Unit tests for CPU SA streaming generator + persistent-context factory."""
from __future__ import annotations

import dimod

from CPU.sa_sampler import SimulatedAnnealingStructuredSampler
from shared.ising_feeder import FixedIsingFeeder
from shared.ising_model import IsingModel
from shared.stream_context import StreamContext
from dwave_topologies.topologies.zephyr import zephyr

_TOPO = zephyr(2, 2)


def _fixed_feeder(sampler):
    nodes = list(sampler.nodes)
    edges = list(sampler.edges)
    # Small non-zero biases keep the SA sampler from warning on an all-zero
    # bqm while leaving the sampleset shape (reads x nodes) unchanged.
    h = {int(n): 0.0 for n in nodes}
    for n in nodes[:2]:
        h[int(n)] = 1.0
    J = {(int(u), int(v)): 0.0 for u, v in edges}
    for u, v in edges[:2]:
        J[(int(u), int(v))] = -1.0
    model = IsingModel(
        h=h, J=J,
        nonce=b"\x01" * 32, salt=b"\x02" * 32,
    )
    return FixedIsingFeeder(models=[model]), model


def test_sample_ising_streaming_yields_model_and_sampleset():
    sampler = SimulatedAnnealingStructuredSampler(topology=_TOPO)
    feeder, model = _fixed_feeder(sampler)
    gen = sampler.sample_ising_streaming(feeder, num_reads=4, num_sweeps=8)
    try:
        out_model, ss = next(gen)
        assert out_model is model
        assert isinstance(ss, dimod.SampleSet)
        assert ss.record.sample.shape[0] == 4
        assert ss.record.sample.shape[1] == len(sampler.nodes)
        out_model2, ss2 = next(gen)
        assert out_model2 is model
    finally:
        gen.close()
        feeder.stop()


def test_build_persistent_context_returns_stream_context():
    from CPU.sa_stream import build_persistent_context
    nodes = [int(n) for n in _TOPO.nodes]
    edges = [(int(u), int(v)) for u, v in _TOPO.edges]
    ctx = build_persistent_context(
        miner_id="cpu-test", nodes=nodes, edges=edges,
        feeder_buffer_size=4, num_reads=4, num_sweeps=8,
        topology=_TOPO, stop_event=None,
    )
    try:
        assert isinstance(ctx, StreamContext)
    finally:
        ctx.cleanup()
