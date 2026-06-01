# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CudaMiner is wired to the stream-driver producer (class-level, no GPU)."""
from __future__ import annotations

from types import SimpleNamespace

from GPU.cuda_miner import CudaMiner


def test_cuda_streaming_factory_is_class_level():
    assert CudaMiner.STREAM_FACTORY_DOTTED == "GPU.cuda_stream:build_persistent_context"
    # The inline sampling path is gone — no _sample/_sample_batch on the class.
    assert not hasattr(CudaMiner, "_sample")
    assert not hasattr(CudaMiner, "_sample_batch")


def test_cuda_stream_factory_kwargs_keys():
    stub = SimpleNamespace(
        miner_id="C-1", FEEDER_BUFFER_SIZE=16, topology=None,
        gpu_utilization=100, _update_mode="sa", sms_per_nonce=4, device="0",
        _yielding=False,
    )
    # The hook reads only plain attrs, so a stand-in stands in for ``self``
    # (a real CudaMiner can't be built without cupy/CUDA here).
    kw = CudaMiner._stream_factory_kwargs(
        stub,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        {"edges": [(0, 1)], "num_reads": 64, "num_sweeps": 256}, [0, 1])
    assert kw["num_reads"] == 64 and kw["num_sweeps"] == 256
    assert kw["nodes"] == [0, 1]
    assert kw["update_mode"] == "sa" and kw["sms_per_nonce"] == 4
