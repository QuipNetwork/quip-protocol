# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Pure decay-step math shared by the chain port and the miner.

``step_for_energy`` maps a candidate's energy floor to the first decay step
whose threshold strictly clears it — a binary search over a monotonic schedule
with no substrate/chain dependency. It lives in ``shared/`` so ``base_miner``
can rank candidates by decay step without importing the ``substrate`` package;
``substrate.difficulty_decay`` re-exports it for the chain-side callers that
build the schedule.
"""
from __future__ import annotations

import bisect
from typing import Optional


def step_for_energy(
    decay_schedule: list[int], floor_energy_milli: int
) -> Optional[int]:
    """First decay step ``s`` where ``schedule[s] > floor_energy_milli``.

    Mirrors the chain's strict ``best_energy_milli < max_energy_milli`` gate:
    the candidate clears at the first step whose threshold is *strictly* above
    its floor. ``None`` when it never clears within the schedule's horizon.
    The schedule is monotonic non-decreasing, so this is a binary search.
    """
    i = bisect.bisect_right(decay_schedule, floor_energy_milli)
    return i if i < len(decay_schedule) else None
