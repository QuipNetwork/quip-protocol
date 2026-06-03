# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Cross-process transport tests for the mempool ProblemView feeder spec.

The mempool mining path sends ``("mempool", attach_args, slot)`` from the worker
to a stream-driver subprocess over a live ``multiprocessing.Queue``. A
``multiprocessing.Queue`` cannot itself be pickled onto another live Queue
(``RuntimeError: Queue objects should only be shared between processes through
inheritance``), so ``ProblemView.attach_args()`` must NOT carry one.

These tests close the coverage gap that let the bug ship: the previous mempool
tests round-tripped the spec only in-process, which never serializes the spec
across a real Queue.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from shared.ring_views import ProblemView

_SPAWN = mp.get_context("spawn")


def _child_read_slot(ctl_q, result_q) -> None:
    """Spawn-child target: reconstruct a ProblemView from the spec and read it.

    Module-level so it is picklable as a spawn ``Process`` target. Gets the
    ``("mempool", attach_args, slot)`` spec off ``ctl_q``, reconstructs the
    read-only ProblemView, reads slot 0, and sends ``(h, J)`` back as lists on
    ``result_q``. On any error the exception text is sent instead so the parent
    can surface it rather than hang.
    """
    try:
        kind, attach_args, slot = ctl_q.get(timeout=10)
        assert kind == "mempool"
        pv = ProblemView(**attach_args)
        try:
            h_vec, j_vec = pv.read(slot)
            result_q.put(("ok", h_vec.tolist(), j_vec.tolist()))
        finally:
            pv.close()
    except Exception as exc:  # noqa: BLE001 — relay to parent for assertion
        result_q.put(("err", f"{type(exc).__name__}: {exc}"))


def test_attach_args_has_no_queue():
    """attach_args() must not carry a multiprocessing.Queue (unpicklable)."""
    pv = ProblemView(slots=1, n_nodes=4, n_edges=3)
    try:
        args = pv.attach_args()
        assert "free_q" not in args
        assert set(args) == {"slots", "n_nodes", "n_edges", "names"}
    finally:
        pv.close_unlink()


def test_spec_round_trips_over_spawn_queue():
    """A real spawn-subprocess round-trip of the mempool spec reads back h/J.

    Reproduces the shipped bug: with ``free_q`` in attach_args the parent
    ``ctl_q.put`` fails to serialize and the child never receives the spec
    (result_q stays empty → timeout). With the fix the child reconstructs the
    read-only ProblemView and reads back exactly what the parent wrote.
    """
    h = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    j = np.array([0.5, -0.5, 0.25], dtype=np.float64)

    pv = ProblemView(slots=1, n_nodes=4, n_edges=3)
    ctl_q = _SPAWN.Queue()
    result_q = _SPAWN.Queue()
    proc = _SPAWN.Process(target=_child_read_slot, args=(ctl_q, result_q))
    try:
        slot = pv.claim_free(timeout=1.0)
        assert slot == 0
        pv.write(slot, h, j)

        proc.start()
        ctl_q.put(("mempool", pv.attach_args(), slot))

        status, *payload = result_q.get(timeout=15)
        assert status == "ok", f"child failed: {payload}"
        got_h, got_j = payload
        np.testing.assert_allclose(got_h, h)
        np.testing.assert_allclose(got_j, j)
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
        pv.close_unlink()
