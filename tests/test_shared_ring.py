# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the generic SharedRing slot/free-list/lifecycle core."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from shared.shared_ring import SharedRing


def _create_and_teardown_ring() -> None:
    """Owner-side ring lifecycle in a child process (must exit cleanly).

    Constructing the ring puts the initial slot ints into ``free_q``, which
    starts that queue's feeder thread. Without ``close_unlink`` detaching the
    feeder-thread join, this process would hang forever at interpreter exit.
    Module-level so spawn can pickle it.
    """
    ring = SharedRing(slots=4, slot_bytes=64)
    slot = ring.claim_free(timeout=1.0)
    ring.release(slot)
    ring.close_unlink()


def test_claim_release_and_full_then_free():
    ring = SharedRing(slots=1, slot_bytes=16)
    try:
        s = ring.claim_free(timeout=1.0)
        assert s is not None
        assert ring.claim_free(timeout=0.05) is None  # full
        ring.release(s)
        assert ring.claim_free(timeout=1.0) is not None
    finally:
        ring.close_unlink()


def test_buf_is_writable_zero_copy():
    ring = SharedRing(slots=2, slot_bytes=32)
    try:
        slot = ring.claim_free(timeout=1.0)
        a = np.ndarray((8,), np.int32, ring.buf(slot), 0)
        a[:] = np.arange(8, dtype=np.int32)
        b = np.ndarray((8,), np.int32, ring.buf(slot), 0)
        assert np.array_equal(b, np.arange(8, dtype=np.int32))  # same memory
    finally:
        ring.close_unlink()


def test_attach_args_reconstructs_nonowner():
    owner = SharedRing(slots=2, slot_bytes=16)
    try:
        args = owner.attach_args()
        assert set(args) == {"slots", "slot_bytes", "names", "free_q"}
        attached = SharedRing(**args)  # non-owner view of the same segments
        slot = attached.claim_free(timeout=1.0)
        np.ndarray((4,), np.int32, attached.buf(slot), 0)[:] = 7
        assert np.ndarray((4,), np.int32, owner.buf(slot), 0)[0] == 7
        attached.close()  # non-owner: close only, never unlink
    finally:
        owner.close_unlink()


def test_owner_exits_cleanly_after_teardown():
    proc = mp.get_context("spawn").Process(target=_create_and_teardown_ring)
    proc.start()
    proc.join(timeout=15.0)
    assert not proc.is_alive(), (
        "ring owner hung at exit — free_q feeder-thread join was not cancelled"
    )
    assert proc.exitcode == 0
