# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the Metal-only QoS clamp.

On first entry into the Metal sampling path (in the stream-driver child), the
sampler lowers its thread QoS to UTILITY so it competes less with the
foreground UI on the P-cores. The clamp must use QOS_CLASS_UTILITY (0x11) — NOT
USER_INTERACTIVE (0x21), which would be a priority *boost* — and must never be
applied from the shared QPU stream-driver path.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import GPU.metal_sa as metal_sa
from GPU.metal_sa import QOS_CLASS_UTILITY, apply_qos_utility


def test_qos_constant_is_utility_not_interactive():
    # <sys/qos.h>: QOS_CLASS_UTILITY = 0x11; USER_INTERACTIVE = 0x21.
    assert QOS_CLASS_UTILITY == 0x11
    assert QOS_CLASS_UTILITY != 0x21


@pytest.mark.skipif(sys.platform != "darwin", reason="QoS clamp is darwin-only")
def test_apply_qos_uses_utility_constant():
    fake_libc = MagicMock()
    with patch.object(metal_sa, "_qos_applied", False), \
         patch.object(metal_sa.ctypes, "CDLL", return_value=fake_libc):
        assert apply_qos_utility() is True
    fake_libc.pthread_set_qos_class_self_np.assert_called_once_with(
        QOS_CLASS_UTILITY, 0,
    )


def test_apply_qos_is_idempotent():
    fake_libc = MagicMock()
    with patch.object(metal_sa, "_qos_applied", True), \
         patch.object(metal_sa.ctypes, "CDLL", return_value=fake_libc):
        # Already applied -> no-op, no syscall.
        assert apply_qos_utility() is False
    fake_libc.pthread_set_qos_class_self_np.assert_not_called()


def test_apply_qos_is_non_fatal_on_error():
    with patch.object(metal_sa, "_qos_applied", False), \
         patch.object(metal_sa.ctypes, "CDLL", side_effect=OSError("boom")):
        # Must swallow the error, not raise into the sampler.
        assert apply_qos_utility() is False


def test_qpu_stream_driver_never_clamps_qos():
    """The shared QPU driver path must not touch QoS (Metal-only concern)."""
    import inspect

    import QPU.stream_driver as sd

    src = inspect.getsource(sd).lower()
    assert "qos" not in src
    assert "apply_qos_utility" not in src


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal device required")
def test_sample_ising_streaming_applies_qos_once():
    from GPU.metal_sa import MetalSASampler
    from tests.test_metal_yielding import _make_models

    s = MetalSASampler()
    models = _make_models(s, 2)
    calls = {"n": 0}

    def counting_clamp():
        calls["n"] += 1
        return True

    with patch.object(metal_sa, "apply_qos_utility", counting_clamp):
        list(s.sample_ising_streaming(
            iter(models), num_reads=8, num_sweeps=64, max_threadgroups=2,
            seed=1,
        ))
    assert calls["n"] == 1
