# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Typed views over a generic ``SharedRing``.

Each view owns only the *byte layout* of a slot and delegates slot allocation,
the free-list, ``attach_args`` reconstruction, and close/unlink to a
``SharedRing``. Bulk arrays live in the ring; small metadata (nonce/salt/
generation/qpu_us) rides the separate descriptor queue.

- ``SampleView``  — producer→evaluator transport: int8 sample matrix + f64
  energy vector.
- ``ProblemView`` — feeder→producer transport: f64 ``h`` vector + f64 ``J``
  vector (dense, in the caller's canonical node/edge order). Consumed by §4.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shared.shared_ring import SharedRing


class SampleView:
    """int8 ``max_rows``×``max_cols`` samples + f64 ``max_rows`` energies."""

    def __init__(self, slots: int, max_rows: int, max_cols: int,
                 *, names: Optional[list] = None, free_q=None):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self._sample_bytes = max_rows * max_cols
        slot_bytes = self._sample_bytes + max_rows * 8
        self._ring = SharedRing(slots, slot_bytes, names=names, free_q=free_q)

    # ── ring delegation ──────────────────────────────────────────────────
    @property
    def slots(self) -> int:
        """Total number of slots in the ring."""
        return self._ring.slots

    @property
    def names(self) -> list:
        """Shared-memory segment names (one per slot)."""
        return self._ring.names

    @property
    def free_q(self):
        """Cross-process free-list queue."""
        return self._ring.free_q

    def attach_args(self) -> dict:
        """Picklable kwargs to reconstruct this view in another process."""
        return {"slots": self._ring.slots, "max_rows": self.max_rows,
                "max_cols": self.max_cols, "names": self._ring.names,
                "free_q": self._ring.free_q}

    def claim_free(self, timeout: float) -> Optional[int]:
        """Return a free slot index, or None if none free within timeout."""
        return self._ring.claim_free(timeout)

    def release(self, slot: int) -> None:
        """Return a slot to the free-list for reuse."""
        self._ring.release(slot)

    def close(self) -> None:
        """Close all slot handles without unlinking (best-effort)."""
        self._ring.close()

    def close_unlink(self) -> None:
        """Close all slots; unlink them if this instance owns them."""
        self._ring.close_unlink()

    # ── sample layout ────────────────────────────────────────────────────
    def write(self, slot: int, sample: np.ndarray, energy: np.ndarray) -> None:
        """Copy sample (int8) + energy (f64) into the slot's shared buffer.

        Args:
            slot: Free slot index obtained from ``claim_free``.
            sample: int8 array of shape ``(n_rows, n_cols)``; must not exceed
                ``max_rows`` × ``max_cols``.
            energy: float64 array of shape ``(n_rows,)``; parallel to sample rows.

        Raises:
            ValueError: if ``sample`` exceeds the slot (``max_rows``×
                ``max_cols``). Writing past the sample region would silently
                corrupt the adjacent energy region, so reject oversized
                samples loudly instead.
        """
        n_rows, n_cols = sample.shape
        if n_rows > self.max_rows or n_cols > self.max_cols:
            raise ValueError(
                f"sample {n_rows}x{n_cols} exceeds slot capacity "
                f"{self.max_rows}x{self.max_cols}"
            )
        buf = self._ring.buf(slot)
        np.ndarray((n_rows, n_cols), np.int8, buf, 0)[:] = sample
        np.ndarray((n_rows,), np.float64, buf, self._sample_bytes)[:] = energy

    def read(self, slot: int, n_rows: int, n_cols: int,
             ) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero-copy (sample, energy) views over the slot's buffer.

        Args:
            slot: Slot index to read from.
            n_rows: Number of rows in the stored sample.
            n_cols: Number of columns in the stored sample.

        Returns:
            Tuple of ``(sample_view, energy_view)`` — zero-copy numpy arrays
            backed by the shared-memory buffer. Release the slot only after
            all views are deleted or otherwise unreferenced.
        """
        buf = self._ring.buf(slot)
        s = np.ndarray((n_rows, n_cols), np.int8, buf, 0)
        e = np.ndarray((n_rows,), np.float64, buf, self._sample_bytes)
        return s, e


class ProblemView:
    """f64 ``h`` vector (``n_nodes``) + f64 ``J`` vector (``n_edges``).

    Both arrays are dense, in the caller's canonical node / edge order. The
    sizes are fixed by the topology, so unlike ``SampleView`` (where D-Wave may
    return a variable row count) ``read`` needs no dimensions. nonce / salt /
    generation ride the descriptor queue, not the slot.
    """

    def __init__(self, slots: int, n_nodes: int, n_edges: int,
                 *, names: Optional[list] = None, free_q=None):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self._h_bytes = n_nodes * 8
        slot_bytes = self._h_bytes + n_edges * 8
        if names is not None and free_q is None:
            # Read-only non-owner reconstruction (mempool driver reading a slot
            # the worker wrote). ProblemView is a write-once/read-once
            # transport: the consumer reads slot 0 by name and never
            # claims/releases across the process boundary, so it needs no
            # shared free-list. attach_args() therefore omits free_q (an
            # mp.Queue can't ride a live ctl_q.put); give SharedRing a throwaway
            # local queue to satisfy its non-owner invariant. Never used for
            # this transport.
            import multiprocessing as _mp
            free_q = _mp.get_context("spawn").Queue()
        self._ring = SharedRing(slots, slot_bytes, names=names, free_q=free_q)

    # ── ring delegation ──────────────────────────────────────────────────
    @property
    def slots(self) -> int:
        """Total number of slots in the ring."""
        return self._ring.slots

    @property
    def names(self) -> list:
        """Shared-memory segment names (one per slot)."""
        return self._ring.names

    @property
    def free_q(self):
        """Cross-process free-list queue."""
        return self._ring.free_q

    def attach_args(self) -> dict:
        """Read-only descriptor to reconstruct this view in another process.

        Unlike ``SampleView.attach_args``, this omits ``free_q``: ProblemView is
        a write-once/read-once fixed-problem transport whose consumer only reads
        a slot by name, so it never needs the owner's shared free-list. The
        returned dict is therefore picklable over a live ``multiprocessing.Queue``
        (it contains no ``mp.Queue``), which the mempool driver path requires —
        the spec rides ``ctl_q.put`` rather than process inheritance.
        """
        return {"slots": self._ring.slots, "n_nodes": self.n_nodes,
                "n_edges": self.n_edges, "names": self._ring.names}

    def claim_free(self, timeout: float) -> Optional[int]:
        """Return a free slot index, or None if none free within timeout."""
        return self._ring.claim_free(timeout)

    def release(self, slot: int) -> None:
        """Return a slot to the free-list for reuse."""
        self._ring.release(slot)

    def close(self) -> None:
        """Close all slot handles without unlinking (best-effort)."""
        self._ring.close()

    def close_unlink(self) -> None:
        """Close all slots; unlink them if this instance owns them."""
        self._ring.close_unlink()

    # ── problem layout ───────────────────────────────────────────────────
    def write(self, slot: int, h: np.ndarray, j: np.ndarray) -> None:
        """Copy the ``h`` (f64) + ``J`` (f64) vectors into the slot.

        Args:
            slot: Free slot index obtained from ``claim_free``.
            h: float64 array of shape ``(n_nodes,)``; linear Ising biases.
            j: float64 array of shape ``(n_edges,)``; quadratic couplings in
                the caller's canonical edge order.

        Raises:
            ValueError: if ``h``/``j`` do not match the view's fixed
                ``n_nodes``/``n_edges``. A mismatch would read back a wrong /
                corrupt model, so reject loudly.
        """
        if h.shape != (self.n_nodes,) or j.shape != (self.n_edges,):
            raise ValueError(
                f"problem ({h.shape}, {j.shape}) != view "
                f"({self.n_nodes!r} nodes, {self.n_edges!r} edges)"
            )
        buf = self._ring.buf(slot)
        np.ndarray((self.n_nodes,), np.float64, buf, 0)[:] = h
        np.ndarray((self.n_edges,), np.float64, buf, self._h_bytes)[:] = j

    def read(self, slot: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero-copy (h, J) views over the slot's buffer.

        Args:
            slot: Slot index to read from.

        Returns:
            Tuple of ``(h_view, j_view)`` — zero-copy float64 numpy arrays
            backed by the shared-memory buffer. Release the slot only after
            all views are deleted or otherwise unreferenced.
        """
        buf = self._ring.buf(slot)
        h = np.ndarray((self.n_nodes,), np.float64, buf, 0)
        j = np.ndarray((self.n_edges,), np.float64, buf, self._h_bytes)
        return h, j
