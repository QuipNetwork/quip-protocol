# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Fake (model, sampleset) stream for stream-driver tests (no QPU)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def make_stream(n: int, rows: int, cols: int):
    rng = np.random.default_rng(0)
    for i in range(n):
        sample = rng.choice(np.array([-1, 1], np.int8), size=(rows, cols))
        energy = rng.normal(-14800, 50, size=rows).astype(np.float64)
        ss = SimpleNamespace(record=SimpleNamespace(sample=sample, energy=energy),
                             info={"timing": {"qpu_programming_time": 10,
                                              "qpu_sampling_time": 51000}})
        model = SimpleNamespace(nonce=bytes([i]) * 32, salt=b"\2" * 32)
        yield model, ss
