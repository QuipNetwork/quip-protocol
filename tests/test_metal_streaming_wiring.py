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
