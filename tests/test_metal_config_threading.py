# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Metal-only adaptive-cap config keys thread end-to-end.

The keys (active_util, idle_after_s) live in the ``[metal]`` device
section. They must:
  - survive normalization into the metal device cfg,
  - NOT leak through when set only in the shared ``[gpu]`` section
    (proving Metal stays independent of the shared registry),
  - reach ``MetalMiner`` and be forwarded by ``_stream_factory_kwargs``,
  - reach ``build_persistent_context`` and configure the scheduler.
The shared ``_GPU_CFG_KEYS`` registry is not modified.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from shared.miner_config import load_backend_config
from shared.miner_core import _GPU_CFG_KEYS, _build_gpu_specs


# ── config round-trip ───────────────────────────────────────────────────

def _metal_cfg(tmp_path, body: str) -> dict:
    p = tmp_path / "metal.toml"
    p.write_text('[miner]\nvalidators = ["ws://a:9944"]\n' + body)
    specs = _build_gpu_specs("rig", load_backend_config(p))
    assert len(specs) == 1 and specs[0]["kind"] == "metal"
    return specs[0]["cfg"]


def test_metal_keys_survive_normalization(tmp_path):
    cfg = _metal_cfg(
        tmp_path,
        "[metal]\nutilization = 100\nactive_util = 50\nidle_after_s = 45\n",
    )
    assert cfg["active_util"] == 50
    assert cfg["idle_after_s"] == 45
    assert cfg["utilization"] == 100


def test_metal_keys_in_gpu_section_do_not_leak(tmp_path):
    """active_util set only in [gpu] must NOT reach the metal device cfg."""
    cfg = _metal_cfg(
        tmp_path,
        "[gpu]\nactive_util = 50\nidle_after_s = 45\n"
        "[metal]\nutilization = 100\n",
    )
    assert "active_util" not in cfg
    assert "idle_after_s" not in cfg


def test_shared_registry_unchanged():
    """Guardrail: the CUDA-shared registry must not have grown metal keys."""
    assert _GPU_CFG_KEYS == ("utilization", "yielding", "enabled", "sms_per_nonce")


def test_yielding_and_utilization_still_wire(tmp_path):
    cfg = _metal_cfg(tmp_path, "[metal]\nutilization = 50\nyielding = true\n")
    assert cfg["utilization"] == 50
    assert cfg["yielding"] is True


# ── MetalMiner threading ────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Metal miner init")
def test_metal_miner_forwards_new_keys():
    from GPU.metal_miner import MetalMiner

    with patch("GPU.metal_miner.MetalSASampler"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=10):
        m = MetalMiner(
            "M-1", topology=None, utilization=100, yielding=True,
            active_util=50, idle_after_s=45,
        )
    kw = m._stream_factory_kwargs(
        {"edges": [(0, 1)], "num_reads": 8, "num_sweeps": 64}, [0, 1],
    )
    assert kw["active_util"] == 50
    assert kw["idle_after_s"] == 45
    assert kw["utilization"] == 100
    assert kw["yielding"] is True


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal miner init")
def test_active_util_defaults_to_85():
    from GPU.metal_miner import MetalMiner

    with patch("GPU.metal_miner.MetalSASampler"), \
         patch("GPU.metal_miner.get_gpu_core_count", return_value=10):
        m = MetalMiner("M-1", topology=None)
    assert m.active_util == 85
    kw = m._stream_factory_kwargs(
        {"edges": [(0, 1)], "num_reads": 8, "num_sweeps": 64}, [0, 1],
    )
    assert kw["active_util"] == 85


# ── active_util % -> thread budget conversion ───────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Metal device required")
def test_active_util_pct_converts_to_thread_budget():
    from GPU.metal_miner import get_gpu_core_count
    from GPU.metal_sa import MetalSASampler
    from GPU.metal_stream import active_threads_for_util, build_persistent_context

    sampler = MetalSASampler()
    cores = get_gpu_core_count()
    per_tg = int(sampler._pipeline.maxTotalThreadsPerThreadgroup())
    expected = round(0.50 * per_tg * cores)
    assert active_threads_for_util(50, sampler, cores) == expected

    # yielding=False avoids spawning the monitor process in the test.
    ctx = build_persistent_context(
        miner_id="M-1", nodes=[0, 1], edges=[(0, 1)],
        feeder_buffer_size=4, num_reads=8, num_sweeps=64,
        utilization=100, yielding=False, active_util=50, idle_after_s=45,
    )
    sched = ctx._sampler_kwargs["scheduler"]
    assert sched._active_threads == expected
    assert sched._idle_after_s == 45
