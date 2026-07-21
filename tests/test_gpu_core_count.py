# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""GPU core-count detection must use a targeted ioreg query.

``ioreg -l`` dumps the ENTIRE IO registry (~1.1s on an idle M4 Max) and blew
its 2s timeout under CPU contention, crash-looping the Metal stream driver.
``ioreg -rc AGXAccelerator -d1`` asks for just the GPU node (~0.05s).
"""
from __future__ import annotations

import subprocess

import pytest

import GPU.metal_miner as metal_miner
import shared.system_info as system_info

# Realistic `ioreg -rc AGXAccelerator -d1` output (abridged).
_AGX_OUTPUT = """\
+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X, id 0x100000abc, registered>
    {
      "gpu-core-count" = 40
      "IOClass" = "AGXAcceleratorG16X"
    }
"""


def _capture_run(calls, stdout, returncode=0):
    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr="",
        )
    return fake_run


def _assert_targeted_query(cmd):
    """The command must query the AGXAccelerator class, not dump the registry."""
    argv = cmd if isinstance(cmd, list) else cmd.split()
    assert "AGXAccelerator" in " ".join(argv)
    assert "-l" not in argv, "full-registry dump (ioreg -l) is too slow under load"
    assert "grep" not in " ".join(argv), "no shell pipe needed for a targeted query"


def test_get_gpu_core_count_uses_targeted_query(monkeypatch):
    calls = []
    monkeypatch.setattr(
        metal_miner.subprocess, "run", _capture_run(calls, _AGX_OUTPUT),
    )
    assert metal_miner.get_gpu_core_count() == 40
    (cmd, kwargs), = calls
    _assert_targeted_query(cmd)
    # Headroom: the query takes ~0.05s; anything under 5s invites the same
    # contention-timeout crash loop the -l variant hit.
    assert kwargs.get("timeout", 0) >= 5


def test_get_gpu_core_count_raises_when_absent(monkeypatch):
    monkeypatch.setattr(
        metal_miner.subprocess, "run",
        _capture_run([], "+-o SomethingElse\n"),
    )
    with pytest.raises(RuntimeError):
        metal_miner.get_gpu_core_count()


def test_system_info_gpu_core_count_uses_targeted_query(monkeypatch):
    calls = []
    monkeypatch.setattr(
        system_info.subprocess, "run", _capture_run(calls, _AGX_OUTPUT),
    )
    assert system_info._apple_gpu_core_count() == 40
    (cmd, kwargs), = calls
    _assert_targeted_query(cmd)
    assert kwargs.get("timeout", 0) >= 5


def test_system_info_gpu_core_count_none_on_timeout(monkeypatch):
    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))
    monkeypatch.setattr(system_info.subprocess, "run", raise_timeout)
    assert system_info._apple_gpu_core_count() is None
