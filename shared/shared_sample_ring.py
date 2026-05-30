# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Fixed ring of shared-memory slots for zero-copy sampleset transfer.

Producer (stream-driver process) writes a sample matrix + energy vector into
a free slot and enqueues a small descriptor; consumer (miner worker) reads a
zero-copy numpy view and returns the slot to the free-list. Avoids pickling
the ~512 KB sample matrix per attempt. See AGENTS.md.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

_SPAWN = mp.get_context("spawn")


class SharedSampleRing:
    """K shared-memory slots sized for (int8 max_rows×max_cols + f64 max_rows)."""

    def __init__(self, slots: int, max_rows: int, max_cols: int,
                 *, names: Optional[list] = None, free_q=None):
        self.slots = slots
        self.max_rows = max_rows
        self.max_cols = max_cols
        self._sample_bytes = max_rows * max_cols
        self._energy_bytes = max_rows * 8
        self._slot_bytes = self._sample_bytes + self._energy_bytes
        self._owner = names is None
        if self._owner:
            self._shm = [shared_memory.SharedMemory(create=True, size=self._slot_bytes)
                         for _ in range(slots)]
            self.names = [s.name for s in self._shm]
            self.free_q = _SPAWN.Queue()
            for i in range(slots):
                self.free_q.put(i)
        else:
            self._shm = [shared_memory.SharedMemory(name=n) for n in names]
            self.names = list(names)
            self.free_q = free_q

    def attach_args(self) -> dict:
        """Picklable kwargs to reconstruct this ring in another process."""
        return {"slots": self.slots, "max_rows": self.max_rows,
                "max_cols": self.max_cols, "names": self.names,
                "free_q": self.free_q}

    def claim_free(self, timeout: float) -> Optional[int]:
        """Return a free slot index, or None if none free within timeout."""
        try:
            return self.free_q.get(timeout=timeout)
        except Exception:
            return None

    def release(self, slot: int) -> None:
        """Return a slot to the free-list for reuse."""
        self.free_q.put(slot)

    def write(self, slot: int, sample: np.ndarray, energy: np.ndarray) -> None:
        """Copy sample (int8) + energy (f64) into the slot's shared buffer.

        Raises:
            ValueError: if ``sample`` is larger than the slot
                (``max_rows``×``max_cols``). Writing past the sample region
                would silently corrupt the adjacent energy region, so reject
                oversized samples loudly instead.
        """
        n_rows, n_cols = sample.shape
        if n_rows > self.max_rows or n_cols > self.max_cols:
            raise ValueError(
                f"sample {n_rows}x{n_cols} exceeds slot capacity "
                f"{self.max_rows}x{self.max_cols}"
            )
        buf = self._shm[slot].buf
        np.ndarray((n_rows, n_cols), np.int8, buf, 0)[:] = sample
        np.ndarray((n_rows,), np.float64, buf, self._sample_bytes)[:] = energy

    def read(self, slot: int, n_rows: int, n_cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero-copy (sample, energy) views over the slot's buffer."""
        buf = self._shm[slot].buf
        s = np.ndarray((n_rows, n_cols), np.int8, buf, 0)
        e = np.ndarray((n_rows,), np.float64, buf, self._sample_bytes)
        return s, e

    def close_unlink(self) -> None:
        """Close all slots; unlink them if this instance owns them."""
        for s in self._shm:
            s.close()
            if self._owner:
                s.unlink()
