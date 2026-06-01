# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for typed ring views (SampleView, ProblemView)."""
from __future__ import annotations

import numpy as np
import pytest

from shared.ring_views import ProblemView, SampleView


def test_sample_roundtrip_zero_copy():
    ring = SampleView(slots=4, max_rows=112, max_cols=4578)
    try:
        sample = np.random.default_rng(0).choice(
            np.array([-1, 1], np.int8), size=(112, 4578))
        energy = np.arange(112, dtype=np.float64)
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, sample, energy)
        s_view, e_view = ring.read(slot, 112, 4578)
        assert np.array_equal(s_view, sample)
        assert np.array_equal(e_view, energy)
        del s_view, e_view
        ring.release(slot)
        assert ring.claim_free(timeout=1.0) is not None
    finally:
        ring.close_unlink()


def test_sample_claim_blocks_when_full_then_frees():
    ring = SampleView(slots=1, max_rows=4, max_cols=4)
    try:
        s = ring.claim_free(timeout=1.0)
        assert ring.claim_free(timeout=0.05) is None
        ring.release(s)
        assert ring.claim_free(timeout=1.0) is not None
    finally:
        ring.close_unlink()


def test_sample_write_rejects_oversized():
    """A sample larger than the slot must raise, not silently overflow."""
    ring = SampleView(slots=1, max_rows=4, max_cols=3)
    try:
        slot = ring.claim_free(timeout=1.0)
        with pytest.raises(ValueError):
            ring.write(slot, np.ones((5, 3), np.int8), np.zeros(5, np.float64))
        with pytest.raises(ValueError):
            ring.write(slot, np.ones((4, 4), np.int8), np.zeros(4, np.float64))
    finally:
        ring.close_unlink()


def test_sample_attach_args_keys_unchanged():
    """attach_args must keep the exact keys the stream driver reconstructs from."""
    ring = SampleView(slots=2, max_rows=8, max_cols=3)
    try:
        args = ring.attach_args()
        assert set(args) == {"slots", "max_rows", "max_cols", "names", "free_q"}
        attached = SampleView(**args)
        slot = attached.claim_free(timeout=1.0)
        attached.write(slot, np.ones((8, 3), np.int8), np.zeros(8, np.float64))
        s, _e = ring.read(slot, 8, 3)
        assert int(s[0, 0]) == 1  # same shared segment
        attached.close()
    finally:
        ring.close_unlink()


def test_problem_roundtrip_zero_copy():
    ring = ProblemView(slots=3, n_nodes=16, n_edges=40)
    try:
        h = np.arange(16, dtype=np.float64)
        j = np.linspace(-1.0, 1.0, 40, dtype=np.float64)
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, h, j)
        h_view, j_view = ring.read(slot)
        assert np.array_equal(h_view, h)
        assert np.array_equal(j_view, j)
        del h_view, j_view
        ring.release(slot)
        assert ring.claim_free(timeout=1.0) is not None
    finally:
        ring.close_unlink()


def test_problem_write_rejects_wrong_size():
    ring = ProblemView(slots=1, n_nodes=4, n_edges=6)
    try:
        slot = ring.claim_free(timeout=1.0)
        with pytest.raises(ValueError):
            ring.write(slot, np.zeros(5, np.float64), np.zeros(6, np.float64))
        with pytest.raises(ValueError):
            ring.write(slot, np.zeros(4, np.float64), np.zeros(7, np.float64))
    finally:
        ring.close_unlink()


def test_problem_attach_args_reconstructs():
    # ProblemView is a write-once/read-once transport: the owner claims a slot
    # and writes h/J; a non-owner reconstructed from attach_args reads it by
    # name. attach_args carries no free_q (the consumer never claims/releases),
    # so the descriptor stays picklable over a live multiprocessing.Queue.
    ring = ProblemView(slots=2, n_nodes=4, n_edges=6)
    try:
        slot = ring.claim_free(timeout=1.0)
        ring.write(slot, np.full(4, 2.0), np.full(6, 3.0))

        args = ring.attach_args()
        assert set(args) == {"slots", "n_nodes", "n_edges", "names"}
        attached = ProblemView(**args)  # read-only non-owner (free_q=None)
        h, j = attached.read(slot)
        assert h[0] == 2.0 and j[0] == 3.0  # same shared segment
        attached.close()
    finally:
        ring.close_unlink()
