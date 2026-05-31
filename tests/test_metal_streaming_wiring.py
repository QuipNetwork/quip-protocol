# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""MetalMiner enables the streaming producer only when Metal init succeeds."""
from __future__ import annotations

from unittest.mock import patch

from GPU.metal_miner import MetalMiner


def test_streaming_flags_set_on_success():
    with patch("GPU.metal_miner.MetalSASampler"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=10):
        m = MetalMiner("M-1", topology=None)
    assert m.STREAMING_PUMP is True
    assert m.DRIVER_OWNS_FEEDER is True
    assert m.STREAM_FACTORY_DOTTED == "GPU.metal_stream:build_persistent_context"
    kw = m._stream_factory_kwargs(
        {"nodes": [0, 1], "edges": [(0, 1)], "num_reads": 8, "num_sweeps": 64},
        [0, 1],
    )
    assert kw["num_reads"] == 8 and kw["num_sweeps"] == 64
    assert kw["nodes"] == [0, 1]


def test_streaming_flags_off_on_cpu_fallback():
    with patch("GPU.metal_miner.MetalSASampler", side_effect=RuntimeError("no metal")):
        m = MetalMiner("M-2", topology=None)
    assert m.STREAMING_PUMP is False
    assert m.DRIVER_OWNS_FEEDER is False
