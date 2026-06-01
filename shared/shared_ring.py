# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Generic fixed-byte shared-memory ring (slot allocation + lifecycle).

A ``SharedRing`` owns ``slots`` shared-memory segments of ``slot_bytes`` each,
plus a free-list queue and the cross-process lifecycle (picklable
``attach_args`` for reconstruction in a child, the feeder-thread-join cancel,
and close/unlink). It is *layout-agnostic*: typed views (``SampleView``,
``ProblemView`` in ``shared.ring_views``) own the byte layout of a slot and
delegate slot management here. Metadata (nonce/salt/generation) rides a
separate descriptor queue; the ring carries only bulk arrays.
See docs/miner-architecture.md.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from multiprocessing import shared_memory
from typing import Optional

_SPAWN = mp.get_context("spawn")


class SharedRing:
    """K shared-memory slots of fixed byte size, with a free-list."""

    def __init__(self, slots: int, slot_bytes: int,
                 *, names: Optional[list] = None, free_q=None):
        self.slots = slots
        self.slot_bytes = slot_bytes
        # Branch on ``names`` directly (not a derived flag) so the owner /
        # non-owner attribute types narrow cleanly.
        if names is None:
            self._owner = True
            self._shm = [shared_memory.SharedMemory(create=True, size=slot_bytes)
                         for _ in range(slots)]
            self.names = [s.name for s in self._shm]
            self.free_q = _SPAWN.Queue()
            for i in range(slots):
                self.free_q.put(i)
        else:
            # A non-owner is reconstructed from ``attach_args()``, which always
            # carries the owner's free-queue. A missing ``free_q`` here means a
            # caller built a non-owner by hand — fail fast rather than crash
            # later on the first ``claim_free`` / ``release``.
            if free_q is None:
                raise ValueError(
                    "non-owner SharedRing requires free_q (from attach_args())"
                )
            self._owner = False
            self._shm = [shared_memory.SharedMemory(name=n) for n in names]
            self.names = list(names)
            self.free_q = free_q

    def attach_args(self) -> dict:
        """Picklable kwargs to reconstruct this ring in another process."""
        return {"slots": self.slots, "slot_bytes": self.slot_bytes,
                "names": self.names, "free_q": self.free_q}

    def claim_free(self, timeout: float) -> Optional[int]:
        """Return a free slot index, or None if none free within timeout.

        Only a genuine timeout means "no free slot". A broken/closed free-queue
        (OSError, ValueError) must propagate so the caller's outer handler logs
        and tears down, rather than masquerading as permanent backpressure that
        silently drops every item.
        """
        try:
            return self.free_q.get(timeout=timeout)
        except _queue.Empty:
            return None

    def release(self, slot: int) -> None:
        """Return a slot to the free-list for reuse."""
        self.free_q.put(slot)

    def buf(self, slot: int):
        """Return the slot's raw shared-memory buffer (for view ndarrays)."""
        return self._shm[slot].buf

    def _release_free_q(self) -> None:
        """Detach this process's free-queue feeder thread.

        ``free_q`` is an ``mp.Queue``; every ``put`` starts a background feeder
        thread, and the queue registers an ``atexit`` finalizer that *joins* it.
        That join blocks forever because the feeder never receives a close
        sentinel — so an owner that exits cleanly would hang at interpreter
        shutdown. ``cancel_join_thread`` drops the join (the buffered free-slot
        ints are worthless once we're tearing the ring down).
        """
        try:
            self.free_q.cancel_join_thread()
        except Exception:  # noqa: BLE001 — best-effort; nothing left to flush
            pass

    def close(self) -> None:
        """Close all slot handles without unlinking (best-effort).

        Used as the fallback when ``close_unlink`` cannot release a segment (a
        buffer view survived): close the handles we can so this process leaks
        nothing further, even though the named segments may persist.
        """
        self._release_free_q()
        for s in self._shm:
            try:
                s.close()
            except Exception:  # noqa: BLE001 — best-effort; segment may leak
                pass

    def close_unlink(self) -> None:
        """Close all slots; unlink them if this instance owns them."""
        self._release_free_q()
        for s in self._shm:
            s.close()
            if self._owner:
                s.unlink()
