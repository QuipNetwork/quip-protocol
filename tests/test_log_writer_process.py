# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the log-writer process introduced in the threads→multiprocessing refactor.

Verifies that `log_writer_main` drains a multiprocessing queue to a rotating
log file and exits cleanly when signalled.
"""

import logging
import multiprocessing as mp
import time

from shared.proc_util import terminate_join


def test_log_writer_process_drains_queue_to_file(tmp_path):
    from shared.logging_config import log_writer_main

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    stop = ctx.Event()
    log_file = tmp_path / "out.log"
    proc = ctx.Process(
        target=log_writer_main,
        args=(q, stop, str(log_file), logging.INFO),
        daemon=True,
    )
    proc.start()
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "hello-mp", None, None)
    q.put(rec)
    deadline = time.monotonic() + 3.0
    while not (log_file.exists() and "hello-mp" in log_file.read_text()) \
            and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    q.put(None)
    assert terminate_join(proc, 2.0)
    assert "hello-mp" in log_file.read_text()
