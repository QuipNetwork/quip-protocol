# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared no-GPU fakes for MetalStreamContext tests + a driver-resolvable factory."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from GPU.metal_stream import MetalStreamContext


class FakeModel:
    def __init__(self, n: int):
        self.h = {0: 0.0}
        self.J = {}
        self.nonce = bytes([n % 256]) * 32
        self.salt = bytes([n % 256]) * 32


class FakeFeeder:
    def __init__(self, **_kw: Any):
        self.reseeds = 0
        self.stopped = False

    def reseed(self, lpbh: Any, miner_bytes: Any) -> None:
        self.reseeds += 1

    def stop(self) -> None:
        self.stopped = True


class FakeSampler:
    def __init__(self) -> None:
        self.closed = False

    def sample_ising_streaming(self, feeder: Any, **_kw: Any):  # type: ignore[return]
        i = 0
        while True:
            # 1 read x 2 spins (int8 sample row, f64 energy)
            ss = SimpleNamespace(
                record=SimpleNamespace(sample=[[1, -1]], energy=[0.0])
            )
            yield FakeModel(i), ss
            i += 1

    def close(self) -> None:
        self.closed = True


def build_fake_context(
    *, stop_event: Any = None, **_ignored: Any
) -> MetalStreamContext:
    """A no-GPU MetalStreamContext factory resolvable by stream_driver_main."""
    return MetalStreamContext(
        sampler=FakeSampler(),
        nodes=[0, 1],
        edges=[(0, 1)],
        feeder_buffer_size=4,
        num_reads=1,
        num_sweeps=8,
        max_threadgroups=4,
        feeder_factory=FakeFeeder,
        stop_event=stop_event,
    )
