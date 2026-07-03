# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""MetalMiner mines through the stream driver; init crashes without Metal."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from GPU.metal_miner import MetalMiner


def test_streaming_wiring_on_success():
    with patch("GPU.metal_miner.MetalSASampler"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=10):
        m = MetalMiner("M-1", topology=None)
    assert m.STREAM_FACTORY_DOTTED == "GPU.metal_stream:build_persistent_context"
    # The inline sampling path is gone — no _sample/_sample_batch on the class.
    assert not hasattr(type(m), "_sample")
    assert not hasattr(type(m), "_sample_batch")


def test_metal_init_raises_when_metal_unavailable():
    with patch("GPU.metal_miner.MetalSASampler", side_effect=RuntimeError("no metal")):
        with pytest.raises(RuntimeError):
            MetalMiner("M-1", topology=None)


def test_stream_factory_kwargs_carries_worker_detected_gpu_cores():
    """The worker probes the core count ONCE at init and hands it to every
    driver (re)spawn — a respawn must never re-run the ioreg probe."""
    with patch("GPU.metal_miner.MetalSASampler"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=12):
        m = MetalMiner("M-1", topology=None)
    kwargs = m._stream_factory_kwargs(
        {"edges": [], "num_reads": 8, "num_sweeps": 128}, [0, 1],
    )
    assert kwargs["gpu_cores"] == 12


def test_build_persistent_context_skips_probe_when_gpu_cores_given():
    from GPU import metal_stream
    with patch("GPU.metal_sa.MetalSASampler"), \
         patch("GPU.metal_scheduler.MetalScheduler") as sched, \
         patch.object(metal_stream, "StreamContext"), \
         patch(
             "GPU.metal_miner.get_gpu_core_count",
             side_effect=AssertionError("respawn re-probed ioreg"),
         ):
        metal_stream.build_persistent_context(
            miner_id="M-1", nodes=[0, 1], edges=[], feeder_buffer_size=4,
            num_reads=8, num_sweeps=128, gpu_cores=12,
        )
    assert sched.call_args.kwargs["gpu_core_count"] == 12


def test_build_persistent_context_probes_when_gpu_cores_absent():
    """Standalone callers (tools) that pass no gpu_cores keep working."""
    from GPU import metal_stream
    with patch("GPU.metal_sa.MetalSASampler"), \
         patch("GPU.metal_scheduler.MetalScheduler") as sched, \
         patch.object(metal_stream, "StreamContext"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=7):
        metal_stream.build_persistent_context(
            miner_id="M-1", nodes=[0, 1], edges=[], feeder_buffer_size=4,
            num_reads=8, num_sweeps=128,
        )
    assert sched.call_args.kwargs["gpu_core_count"] == 7
