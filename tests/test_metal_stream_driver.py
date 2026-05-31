# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Integration test: stream_driver_main drives StreamContext (Metal) into a ring."""
from __future__ import annotations

import multiprocessing as mp

import pytest

from shared.proc_util import terminate_join
from shared.ring_views import SampleView

_FAKE_FACTORY = "tests._metal_stream_fakes:build_fake_context"


def _spawn_driver(ring, desc_q, ctl_q, stop, factory, factory_kwargs):
    from QPU.stream_driver import stream_driver_main

    proc = mp.get_context("spawn").Process(
        target=stream_driver_main,
        args=(ring.attach_args(), desc_q, ctl_q, stop, factory, factory_kwargs),
        daemon=True,
    )
    proc.start()
    return proc


@pytest.mark.timeout(30)
def test_metal_driver_produces_descriptor_and_ring_sample():
    """stream_driver_main + generic StreamContext writes samples into the ring.

    The fake yields 1 read × 2 spins: sample=[[1, -1]], energy=[0.0].
    We assert the descriptor 7-tuple fields and round-trip one sample from the
    ring to confirm zero-copy write/read works end-to-end.
    """
    ctx = mp.get_context("spawn")
    # 1 read x 2 spins — matches FakeSampler's fixed yield shape.
    ring = SampleView(slots=4, max_rows=1, max_cols=2)
    desc_q: mp.Queue = ctx.Queue()
    ctl_q: mp.Queue = ctx.Queue()
    stop = ctx.Event()

    proc = _spawn_driver(ring, desc_q, ctl_q, stop, _FAKE_FACTORY, {})
    # 9-tuple: gen, lpbh, miner_bytes, threshold, num_reads, anneal, num_sweeps, feeder_spec
    ctl_q.put(
        ("switch", 1, b"\x00" * 32, b"\x01" * 32, 0, 1, 0.0, 8,
         ("pow", b"\x00" * 32, b"\x01" * 32))
    )

    try:
        item = desc_q.get(timeout=15.0)
        assert item is not None, "expected a descriptor, got end-of-stream None"

        slot, n_rows, n_cols, nonce, salt, qpu_us, generation = item

        assert generation == 1
        assert n_rows == 1
        assert n_cols == 2

        sample, _energy = ring.read(slot, n_rows, n_cols)
        assert list(sample[0]) == [1, -1]
        del sample, _energy
        ring.release(slot)

    finally:
        stop.set()
        # Drain until the end-of-stream None sentinel so the driver can exit
        # cleanly (it only sends None after stop_event fires and cleanup runs).
        for _ in range(500):
            try:
                tail = desc_q.get(timeout=0.05)
            except Exception:
                break
            if tail is None:
                break
            # Release any still-held slots so the driver's ring.close_unlink()
            # doesn't block on live references.
            ring.release(tail[0])
        assert terminate_join(proc, 5.0), "stream driver process did not exit"
        ring.close_unlink()
