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


def build_fake_production_stream(
    *,
    num_reads: int,
    nodes,
    n: int = 5,
    stop_event=None,
    **_ignored,
):
    """Drop-in fake for ``build_production_stream`` (no QPU).

    Matches the ``(stream, cleanup)`` contract the stream-driver process
    expects. ``stream`` yields ``n`` ``(model, sampleset)`` pairs sized to
    ``(num_reads, len(nodes))`` so the consumer's ring (max_rows=num_reads,
    max_cols=len(nodes)) fits them exactly. Extra production kwargs
    (miner_id, token, ...) are accepted and ignored. ``stop_event`` is
    honoured so the driver can be reaped promptly on teardown.

    ``n <= 0`` makes the stream run indefinitely until ``stop_event`` fires
    — used by teardown / prompt-stop tests.
    """
    rows, cols = int(num_reads), len(nodes)

    def _gen():
        rng = np.random.default_rng(0)
        produced = 0
        # Loop indefinitely (bounded by ``n`` only when n > 0) so teardown
        # tests can stop a still-running stream via stop_event.
        while True:
            if n > 0 and produced >= n:
                return
            if stop_event is not None and stop_event.is_set():
                return
            sample = rng.choice(np.array([-1, 1], np.int8), size=(rows, cols))
            energy = rng.normal(-14800, 50, size=rows).astype(np.float64)
            ss = SimpleNamespace(
                record=SimpleNamespace(sample=sample, energy=energy),
                info={"timing": {"qpu_programming_time": 10,
                                 "qpu_sampling_time": 51000}},
            )
            model = SimpleNamespace(
                nonce=bytes([produced % 256]) * 32, salt=b"\3" * 32,
            )
            produced += 1
            yield model, ss

    def cleanup():
        return None

    return _gen(), cleanup


def build_fake_infinite_stream(**kwargs):
    """Like :func:`build_fake_production_stream` but never self-terminates.

    Forces ``n=0`` so the stream only ends when ``stop_event`` fires — used
    by the prompt-stop / teardown tests that need a still-running driver.
    """
    kwargs["n"] = 0
    return build_fake_production_stream(**kwargs)
