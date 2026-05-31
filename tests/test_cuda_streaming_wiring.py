# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CudaMiner is wired to the stream-driver producer (class-level, no GPU)."""
from __future__ import annotations

from types import SimpleNamespace

from GPU.cuda_miner import CudaMiner


def test_cuda_streaming_flags_are_class_level():
    assert CudaMiner.STREAMING_PUMP is True
    assert CudaMiner.DRIVER_OWNS_FEEDER is True
    assert CudaMiner.STREAM_FACTORY_DOTTED == "GPU.cuda_stream:build_persistent_context"


def test_cuda_stream_factory_kwargs_keys():
    stub = SimpleNamespace(
        miner_id="C-1", FEEDER_BUFFER_SIZE=16, topology=None,
        gpu_utilization=100, _update_mode="sa", sms_per_nonce=4, device="0",
        _yielding=False,
    )
    kw = CudaMiner._stream_factory_kwargs(
        stub, {"edges": [(0, 1)], "num_reads": 64, "num_sweeps": 256}, [0, 1])
    assert kw["num_reads"] == 64 and kw["num_sweeps"] == 256
    assert kw["nodes"] == [0, 1]
    assert kw["update_mode"] == "sa" and kw["sms_per_nonce"] == 4
