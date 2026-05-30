# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for SharedSampleRing zero-copy shared-memory transport."""
from __future__ import annotations

import numpy as np
import pytest

from shared.shared_sample_ring import SharedSampleRing


def test_roundtrip_zero_copy():
    ring = SharedSampleRing(slots=4, max_rows=112, max_cols=4578)
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


def test_claim_blocks_when_full_then_frees():
    ring = SharedSampleRing(slots=1, max_rows=4, max_cols=4)
    try:
        s = ring.claim_free(timeout=1.0)
        assert ring.claim_free(timeout=0.05) is None
        ring.release(s)
        assert ring.claim_free(timeout=1.0) is not None
    finally:
        ring.close_unlink()


def test_write_rejects_oversized():
    """A sample larger than the slot must raise, not silently overflow."""
    ring = SharedSampleRing(slots=1, max_rows=4, max_cols=3)
    try:
        slot = ring.claim_free(timeout=1.0)
        # Too many rows (D-Wave can return more rows than num_reads).
        with pytest.raises(ValueError):
            ring.write(slot, np.ones((5, 3), np.int8), np.zeros(5, np.float64))
        # Too many cols.
        with pytest.raises(ValueError):
            ring.write(slot, np.ones((4, 4), np.int8), np.zeros(4, np.float64))
    finally:
        ring.close_unlink()
