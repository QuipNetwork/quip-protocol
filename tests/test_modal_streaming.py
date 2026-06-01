"""Unit tests for Modal streaming generator (stubbed sample_ising — no cloud)."""
from __future__ import annotations

import dimod

from shared.ising_feeder import FixedIsingFeeder
from shared.ising_model import IsingModel
from shared.stream_context import StreamContext


class _StubModalSampler:
    """Stand-in exposing sample_ising; the real streaming method is bound onto it."""
    def __init__(self):
        self.calls = []

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kw):
        self.calls.append((num_reads, num_sweeps))
        n = len(h)
        return dimod.SampleSet.from_samples(
            [[-1] * n] * num_reads, vartype=dimod.SPIN, energy=[0.0] * num_reads,
        )


def _feeder():
    model = IsingModel(
        h={0: 1.0, 1: -1.0}, J={(0, 1): -1.0},
        nonce=b"\x01" * 32, salt=b"\x02" * 32,
    )
    return FixedIsingFeeder(models=[model])


def test_modal_sample_ising_streaming_yields_model_and_sampleset():
    from GPU.modal_sampler import ModalSampler
    stub = _StubModalSampler()
    feeder = _feeder()
    gen = ModalSampler.sample_ising_streaming(
        stub, feeder, num_reads=3, num_sweeps=16,
    )
    try:
        out_model, ss = next(gen)
        assert isinstance(ss, dimod.SampleSet)
        assert ss.record.sample.shape[0] == 3
        assert stub.calls == [(3, 16)]
    finally:
        gen.close()
        feeder.stop()


def test_modal_build_persistent_context_returns_stream_context(monkeypatch):
    import GPU.modal_sampler as ms

    class _NoConnectSampler(_StubModalSampler):
        def __init__(self, gpu_type="t4"):
            super().__init__()
            self.gpu_type = gpu_type

    monkeypatch.setattr(ms, "ModalSampler", _NoConnectSampler)
    from GPU.modal_stream import build_persistent_context
    ctx = build_persistent_context(
        miner_id="modal-test", nodes=[0, 1], edges=[(0, 1)],
        feeder_buffer_size=4, num_reads=4, num_sweeps=8, gpu_type="t4",
        stop_event=None,
    )
    try:
        assert isinstance(ctx, StreamContext)
    finally:
        ctx.cleanup()
