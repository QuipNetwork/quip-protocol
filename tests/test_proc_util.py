# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for shared.proc_util spawn-context helpers."""

import multiprocessing as mp

from shared.proc_util import spawn_worker, terminate_join


def _set_flag(val):
    val.value = 7


def _block(ev):
    ev.wait()


def test_spawn_worker_runs_and_joins():
    ctx = mp.get_context("spawn")
    val = ctx.Value("i", 0)
    proc = spawn_worker(_set_flag, (val,), name="t")
    proc.join(timeout=5.0)
    assert val.value == 7


def test_terminate_join_stops_a_blocked_worker():
    ctx = mp.get_context("spawn")
    ev = ctx.Event()
    proc = spawn_worker(_block, (ev,), name="blocked")
    assert terminate_join(proc, timeout=2.0) is True
    assert not proc.is_alive()
