# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for GPU.util_monitor generic utilization monitor process."""

import multiprocessing as mp
import time

from shared.proc_util import terminate_join


def _fake_poll():
    return 42


def test_util_monitor_publishes_into_value():
    from GPU.util_monitor import util_monitor_main

    ctx = mp.get_context("spawn")
    val = ctx.Value("i", -1)
    stop = ctx.Event()
    proc = ctx.Process(
        target=util_monitor_main,
        args=(val, stop, 0.01,
              "tests.test_util_monitor_process:_fake_poll"),
        daemon=True,
    )
    proc.start()
    deadline = time.monotonic() + 3.0
    while val.value != 42 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    assert terminate_join(proc, 2.0)
    assert val.value == 42
