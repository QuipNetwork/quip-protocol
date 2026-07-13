# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Pure host-side slot bookkeeping for the CUDA self-feeding streaming loop.

Extracted from ``base_cuda_sampler`` so it carries no cupy dependency and is
unit-testable on any machine. Each nonce owns ``slots_per_nonce`` physical
slots (indices ``0..N-1``); this tracks which slot currently plays the ACTIVE
role (kernel processing / host-awaited) and which plays the NEXT role (queued
READY behind it). Free slots are *derived* from the complement of the in-use
slots, so a slot can never be lost to a stale sentinel.

The rotation deliberately keeps a nonce alive when its feeder is momentarily
empty at a completion boundary: it goes IDLE rather than being dropped. The
gh-19 / QUI-828 throughput decay was exactly such a dropped nonce — the old
rotation advanced ``active`` into an empty NEXT slot (``-1`` / ``None``), after
which the nonce was skipped forever and its GPU block sat idle, so the miner's
attempt rate fell block-by-block over a long run until a restart reset it.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class SlotState:
    """Per-nonce ACTIVE/NEXT slot assignment for the streaming rotation.

    ``active_slot`` / ``next_slot`` hold the physical slot index playing each
    role, or ``-1`` when the role is unfilled. Free slots are derived, never
    stored, so no slot can leak.
    """

    slots_per_nonce: int = 3
    active_slot: int = -1
    active_model: Any = None
    next_slot: int = -1
    next_model: Any = None

    @property
    def is_idle(self) -> bool:
        """True when nothing is being processed (awaiting feeder input)."""
        return self.active_model is None

    def needs_next(self) -> bool:
        """Active but nothing queued behind it — a NEXT slot should be filled."""
        return self.active_model is not None and self.next_model is None

    def free_slot(self) -> int:
        """Lowest physical slot not currently ACTIVE or NEXT, else ``-1``.

        Derived from the complement of the in-use slots. The old code stored a
        single ``free_slot`` int and overwrote it on rotation, which leaked a
        slot on a second starved completion; deriving it makes that impossible.
        """
        used = {s for s in (self.active_slot, self.next_slot) if s >= 0}
        for i in range(self.slots_per_nonce):
            if i not in used:
                return i
        return -1

    def assign_active(self, slot: int, model: Any) -> None:
        """Mark ``slot`` (already uploaded READY) as the ACTIVE slot."""
        self.active_slot, self.active_model = slot, model

    def assign_next(self, slot: int, model: Any) -> None:
        """Mark ``slot`` (already uploaded READY) as the queued NEXT slot."""
        self.next_slot, self.next_model = slot, model

    def rotate_on_completion(self) -> None:
        """Consume the completed ACTIVE slot.

        Promote the queued NEXT model to ACTIVE if one is present; otherwise go
        IDLE (clear ACTIVE) *without* dropping the nonce, so a momentarily-empty
        feeder can no longer permanently starve this nonce's GPU block. A caller
        revives an idle nonce by uploading a fresh model into ``free_slot()``
        and calling :meth:`assign_active`.
        """
        if self.next_model is not None:
            self.active_slot, self.active_model = self.next_slot, self.next_model
        else:
            self.active_slot, self.active_model = -1, None
        self.next_slot, self.next_model = -1, None
