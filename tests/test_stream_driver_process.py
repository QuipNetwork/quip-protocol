# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Integration test: stream_driver_main produces descriptors via SharedSampleRing."""
from __future__ import annotations

import multiprocessing as mp

from shared.shared_sample_ring import SharedSampleRing
from shared.proc_util import terminate_join


def test_driver_produces_descriptors_via_ring():
    from QPU.stream_driver import stream_driver_main
    ctx = mp.get_context("spawn")
    ring = SharedSampleRing(slots=4, max_rows=8, max_cols=16)
    desc_q = ctx.Queue()
    stop = ctx.Event()
    proc = ctx.Process(
        target=stream_driver_main,
        args=(ring.attach_args(), desc_q, stop,
              "tests.fakes.fake_stream:make_stream",
              {"n": 5, "rows": 8, "cols": 16}),
        daemon=True)
    proc.start()
    seen = 0
    try:
        for _ in range(5):
            item = desc_q.get(timeout=5.0)
            assert item is not None
            slot, nr, nc, nonce, salt, qpu = item
            s, e = ring.read(slot, nr, nc)
            assert s.shape == (nr, nc)
            del s, e
            ring.release(slot)
            seen += 1
    finally:
        stop.set()
        assert terminate_join(proc, 3.0)
        ring.close_unlink()
    assert seen == 5
