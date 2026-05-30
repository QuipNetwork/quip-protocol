# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Fake persistent stream context for stream-driver tests (no QPU)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np


class _FakePersistentContext:
    """No-QPU stand-in for ``PersistentStreamContext``.

    Implements the duck-typed driver contract: ``apply_command`` /
    ``iter_results`` / ``cleanup`` / ``generation``. ``iter_results`` yields
    ``(model, sampleset, submit_generation)`` sized to ``(num_reads,
    len(nodes))`` so the consumer's ring fits them exactly. With ``n <= 0``
    it runs until ``stop_event`` fires (teardown / switch tests).

    ``energy_mean`` lets a test push the best energy below or above a
    threshold so the worker's evaluate/ratchet path can be driven.
    """

    def __init__(
        self,
        *,
        num_reads,
        nodes,
        n=0,
        energy_mean=-14800.0,
        stop_event=None,
        **_ignored,
    ):
        self._rows = int(num_reads)
        self._cols = len(nodes)
        self._n = n
        self._energy_mean = energy_mean
        self._stop_event = stop_event
        self.generation = 0
        self._seeded = False
        self._produced = 0
        self._rng = np.random.default_rng(0)
        self.cleaned_up = False

    def apply_command(self, cmd):
        kind = cmd[0]
        if kind == "switch":
            self.generation = int(cmd[1])
            self._seeded = True
        # 'threshold' is a no-op for the fake (no real reconstruction gate).

    def _stop(self):
        return self._stop_event is not None and self._stop_event.is_set()

    def iter_results(self):
        while self._seeded and not self._stop():
            if self._n > 0 and self._produced >= self._n:
                return
            sample = self._rng.choice(
                np.array([-1, 1], np.int8),
                size=(self._rows, self._cols),
            )
            energy = self._rng.normal(
                self._energy_mean,
                50,
                size=self._rows,
            ).astype(np.float64)
            ss = SimpleNamespace(
                record=SimpleNamespace(sample=sample, energy=energy),
                info={
                    "timing": {"qpu_programming_time": 10, "qpu_sampling_time": 51000}
                },
            )
            model = SimpleNamespace(
                nonce=bytes([self._produced % 256]) * 32,
                salt=b"\3" * 32,
            )
            self._produced += 1
            yield model, ss, self.generation
            time.sleep(0.001)  # let the driver poll ctl_q between yields

    def cleanup(self):
        self.cleaned_up = True


def build_fake_persistent_context(
    *, num_reads, nodes, n=0, energy_mean=-14800.0, stop_event=None, **_ignored
):
    """Drop-in fake for ``build_persistent_context`` (no QPU)."""
    return _FakePersistentContext(
        num_reads=num_reads,
        nodes=nodes,
        n=n,
        energy_mean=energy_mean,
        stop_event=stop_event,
    )


def build_fake_raising_context(**_ignored):
    """Context factory that raises (simulated D-Wave auth/topology failure)."""
    raise RuntimeError("simulated D-Wave factory failure")
