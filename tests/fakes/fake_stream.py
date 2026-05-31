# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Fake persistent stream context for stream-driver tests (no QPU)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np


class _FakePersistentContext:
    """No-QPU stand-in for the stream driver context.

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
        idle_on_pause=True,
        energies=None,
        qpu_sampling_time=51000,
        **_ignored,
    ):
        self._rows = int(num_reads)
        self._cols = len(nodes)
        self._n = n
        self._energy_mean = energy_mean
        # Optional deterministic per-iteration best energy. When provided the
        # context produces exactly len(energies) samplesets (row-0 carries the
        # given energy, the rest are slightly higher so np.min picks it) then
        # stops — used by precheck/preview tests that count iterations.
        self._energies = list(energies) if energies is not None else None
        if self._energies is not None:
            self._n = len(self._energies)
        # Lets a test zero out the QPU timing so the descriptor's qpu_us is 0
        # (non-QPU-backend behaviour) instead of the default ~51010.
        self._qpu_sampling_time = int(qpu_sampling_time)
        self._stop_event = stop_event
        # When False, a 'pause' is recorded but production continues — models a
        # continuously-draining in-flight queue so worker-side tests can drive
        # decay-after-pause deterministically (the driver's real stop-on-pause
        # is covered separately in test_stream_driver_process).
        self._idle_on_pause = idle_on_pause
        self.generation = 0
        self._seeded = False
        self._paused = False
        self._produced = 0
        self._rng = np.random.default_rng(0)
        self.cleaned_up = False

    def apply_command(self, cmd):
        kind = cmd[0]
        if kind == "switch":
            self.generation = int(cmd[1])
            self._seeded = True
            self._paused = False  # a new head resumes a paused driver
        elif kind == "pause":
            # Drain-and-idle: the fake has no in-flight concept, so pausing
            # just stops production (iter_results returns → driver idles).
            self._paused = True
        # 'threshold' is a no-op for the fake (no real reconstruction gate).

    def _stop(self):
        return self._stop_event is not None and self._stop_event.is_set()

    def iter_results(self):
        while self._seeded and not self._stop() and not (
            self._paused and self._idle_on_pause
        ):
            if self._n > 0 and self._produced >= self._n:
                return
            sample = self._rng.choice(
                np.array([-1, 1], np.int8),
                size=(self._rows, self._cols),
            )
            if self._energies is not None:
                best = float(self._energies[self._produced])
                energy = np.full(self._rows, best + 1.0, np.float64)
                energy[0] = best
            else:
                energy = self._rng.normal(
                    self._energy_mean,
                    50,
                    size=self._rows,
                ).astype(np.float64)
            ss = SimpleNamespace(
                record=SimpleNamespace(sample=sample, energy=energy),
                info={
                    "timing": {
                        "qpu_programming_time": 10 if self._qpu_sampling_time else 0,
                        "qpu_sampling_time": self._qpu_sampling_time,
                    }
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


def build_fake_nonstop_persistent_context(
    *, num_reads, nodes, n=0, energy_mean=-14800.0, stop_event=None, **_ignored
):
    """Fake whose production does NOT stop on 'pause' (continuous drain).

    For worker-side budget-gate tests that need the stream to keep delivering
    after the gate sends a pause, so decay-after-pause submission can be driven
    deterministically. The pause is still recorded (``_paused``); only
    production is unaffected.
    """
    return _FakePersistentContext(
        num_reads=num_reads,
        nodes=nodes,
        n=n,
        energy_mean=energy_mean,
        stop_event=stop_event,
        idle_on_pause=False,
    )


def build_fake_energies_context(
    *, num_reads, nodes, energies, stop_event=None, **_ignored
):
    """Fake whose row-0 best energy follows a deterministic ``energies`` list.

    Produces exactly ``len(energies)`` samplesets (one per entry, in order)
    then stops — so worker-side tests can count iterations and present an
    exact best-energy per iteration without the random ``energy_mean`` draw.
    """
    return _FakePersistentContext(
        num_reads=num_reads,
        nodes=nodes,
        energies=energies,
        stop_event=stop_event,
    )


def build_fake_zero_qpu_context(
    *, num_reads, nodes, n=0, energy_mean=-14800.0, stop_event=None, **_ignored
):
    """Fake whose descriptor carries ZERO qpu time (non-QPU-backend timing).

    Models a CPU/CUDA/Metal stream-driver: the sampleset info reports no QPU
    sampling time, so ``QPU/stream_driver._extract_qpu_us`` yields 0 and the
    consumer records ``qpu_access_time_us == 0`` (not a positive figure).
    """
    return _FakePersistentContext(
        num_reads=num_reads,
        nodes=nodes,
        n=n,
        energy_mean=energy_mean,
        stop_event=stop_event,
        qpu_sampling_time=0,
    )


def build_fake_raising_context(**_ignored):
    """Context factory that raises (simulated D-Wave auth/topology failure)."""
    raise RuntimeError("simulated D-Wave factory failure")
