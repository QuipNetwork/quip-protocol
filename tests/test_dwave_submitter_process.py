# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Integration test: the QPU submitter split end-to-end (no D-Wave).

Spawns the connection-less feeder driver (A) and the isolated submitter (B) as
real processes wired by a full-size ProblemView + SampleView (real shared
memory). A's RandomIsingFeeder reduces problems into the ring; B (fake sampler)
reads them and writes results back. Verifies a generation-tagged result
descriptor flows all the way through, and that both processes tear down.
"""

from __future__ import annotations

import multiprocessing as mp

from shared.allowed_value_spec import AllowedValueSet
from shared.proc_util import terminate_join
from shared.ring_views import ProblemView, SampleView
from tests.fakes.fake_submitter import EDGES, NODES

_FAKE_SAMPLER = "tests.fakes.fake_submitter:build_fake_sampler"


def test_split_streams_result_through_problemview_and_sampleview():
    from QPU.dwave_submitter import dwave_submitter_main
    from shared.feeder_driver import feeder_driver_main

    ctx = mp.get_context("spawn")
    num_reads, queue_depth = 4, 2
    slots = queue_depth + 8

    ring = SampleView(slots=slots, max_rows=num_reads, max_cols=len(NODES))
    prob_ring = ProblemView(slots=slots, n_nodes=len(NODES), n_edges=len(EDGES))
    desc_q = ctx.Queue(maxsize=slots)
    prob_desc_q = ctx.Queue(maxsize=slots)
    ctl_q = ctx.Queue()
    handshake_q = ctx.Queue()
    stop = ctx.Event()

    submitter = ctx.Process(
        target=dwave_submitter_main,
        args=(_FAKE_SAMPLER, {}, prob_ring.recycling_attach_args(), prob_desc_q,
              ring.attach_args(), desc_q, handshake_q, stop,
              num_reads, queue_depth, 80.0, None),
        daemon=False,
    )
    feeder = ctx.Process(
        target=feeder_driver_main,
        args=(prob_ring.recycling_attach_args(), prob_desc_q, ctl_q, stop,
              handshake_q, NODES, EDGES, AllowedValueSet((0,)), 4, None),
        daemon=False,
    )
    submitter.start()
    feeder.start()

    # Drive a round: switch carries (gen, lpbh, miner_bytes, thr, num_reads,
    # anneal, num_sweeps, feeder_spec).
    lpbh = b"\x01" * 32
    miner_bytes = b"\x02" * 32
    feeder_spec = ("pow", lpbh, miner_bytes)
    ctl_q.put((
        "switch", 3, lpbh, miner_bytes, 0, num_reads, 80.0, 0, feeder_spec,
    ))

    try:
        item = desc_q.get(timeout=30.0)
        assert item is not None, "got end-of-stream before any result"
        slot, n_rows, n_cols, nonce, salt, qpu_us, generation, defect = item[:8]
        assert generation == 3                  # tagged with the live round
        assert n_rows == num_reads
        assert n_cols == len(NODES)
        assert len(nonce) == 32 and len(salt) == 32
        assert qpu_us > 0                        # fake reports QPU timing
        s, e = ring.read(slot, n_rows, n_cols)
        assert s.shape == (n_rows, n_cols)
        del s, e
        ring.release(slot)
    finally:
        ctl_q.put(None)       # shutdown sentinel -> feeder driver A
        stop.set()
        assert terminate_join(feeder, 10.0)
        assert terminate_join(submitter, 10.0)
        ring.close_unlink()
        prob_ring.close_unlink()
