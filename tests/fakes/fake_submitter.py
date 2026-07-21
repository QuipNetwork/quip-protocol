# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Fake connected sampler for the QPU submitter-split tests (no D-Wave).

``build_fake_sampler`` is the ``SAMPLER_FACTORY_DOTTED`` stand-in the isolated
submitter resolves: it exposes the live-topology attributes the handshake reads
plus a ``sample_ising_streaming`` that consumes the real ``RingProblemFeeder``
and yields fake samplesets — so the whole A→ProblemView→B→SampleView path runs
on real shared memory without a connection.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# Small topology shared with the integration test (both sides must agree on the
# node/edge ordering for the reduced-array layout).
NODES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]


class _FakeSampler:
    """No-D-Wave stand-in for DWaveSamplerWrapper (submitter side)."""

    def __init__(self) -> None:
        self._defective_qubits: list = []
        self._defective_edges: set = set()
        self.live_nodes = list(NODES)
        self.live_edges = list(EDGES)
        self.closed = False

    def sample_ising_streaming(
        self,
        feeder,
        *,
        num_reads,
        queue_depth=2,
        annealing_time=None,
        stop_event=None,
        **_kw,
    ):
        """Pop reduced problems from the ring feeder; yield fake samplesets."""
        rng = np.random.default_rng(0)
        n_cols = len(self.live_nodes)
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                rp = feeder.pop_blocking()
            except StopIteration:
                return
            sample = rng.choice(
                np.array([-1, 1], np.int8), size=(int(num_reads), n_cols)
            )
            energy = rng.normal(-100.0, 5.0, size=int(num_reads)).astype(np.float64)
            ss = SimpleNamespace(
                record=SimpleNamespace(sample=sample, energy=energy),
                info={"timing": {
                    "qpu_programming_time": 10, "qpu_sampling_time": 5000,
                }},
            )
            yield rp, ss

    def close(self) -> None:
        self.closed = True


def build_fake_sampler(**_kwargs):
    """Drop-in fake for ``QPU.dwave_miner:build_sampler`` (no D-Wave)."""
    return _FakeSampler()
