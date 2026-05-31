# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Metal stream-driver context: GPU sampling in the producer process.

Mirrors QPU/dwave_miner.py::PersistentStreamContext for the Metal backend, but
is synchronous (one ``sample_ising_streaming`` generator, no async futures, no
reconstruction gating, no queue depth). Run ONLY in the stream-driver process
(QPU/stream_driver.py); the worker consumes the resulting ring zero-copy.
"""
from __future__ import annotations

import logging
import multiprocessing.synchronize
from typing import Any, Iterator, List, Optional, Tuple

import dimod

from shared.ising_feeder import RandomIsingFeeder
from shared.ising_model import IsingModel

log = logging.getLogger(__name__)


class MetalStreamContext:
    """Owns a feeder + Metal sampler; yields generation-tagged samplesets.

    ``apply_command``: ``switch`` (re)seeds the feeder and bumps the
    generation (and updates num_reads); ``threshold`` is a no-op (Metal does no
    reconstruction gating); ``pause`` stops production until the next switch
    (Metal is never paused today — the QPU budget gate is the only pause source
    — but it is honored harmlessly).
    """

    def __init__(
        self,
        *,
        sampler: Any,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        feeder_buffer_size: int,
        num_reads: int,
        num_sweeps: int,
        max_threadgroups: int,
        feeder_factory: Any = RandomIsingFeeder,
        duty_cycle: Any = None,
        scheduler: Any = None,
        stop_event: Optional[multiprocessing.synchronize.Event] = None,
    ) -> None:
        self._sampler = sampler
        self._nodes = nodes
        self._edges = edges
        self._feeder_buffer_size = feeder_buffer_size
        self._num_reads = num_reads
        self._num_sweeps = num_sweeps
        self._max_threadgroups = max_threadgroups
        self._feeder_factory = feeder_factory
        self._duty_cycle = duty_cycle
        self._scheduler = scheduler
        self._stop_event = stop_event
        self._feeder: Optional[Any] = None
        self._stream: Optional[Any] = None
        self.generation: int = 0
        self._paused = False

    def apply_command(self, cmd: Tuple[Any, ...]) -> None:
        """Dispatch a control command from the driver's ctl_q.

        Supported commands:
            ``("switch", gen, lpbh, miner_bytes, threshold, num_reads, anneal)``
                — (re)seed feeder, bump generation.
            ``("threshold", gen, value)``  — no-op on Metal.
            ``("pause", gen)``             — halt iter_results until next switch.
        """
        kind = cmd[0]
        if kind == "switch":
            (_, gen, lpbh, miner_bytes, _thr, num_reads, _anneal) = cmd[:7]
            self.generation = int(gen)
            self._num_reads = int(num_reads)
            # The switch carries the per-round adapted sweep count as an 8th
            # element (absent in older/test 7-tuples); closing the stream below
            # makes the new value take effect on the next iter_results.
            if len(cmd) > 7:
                self._num_sweeps = int(cmd[7])
            self._paused = False
            self._close_stream()
            if self._feeder is None:
                self._feeder = self._feeder_factory(
                    last_proof_block_hash=lpbh,
                    miner_bytes=miner_bytes,
                    nodes=self._nodes,
                    edges=self._edges,
                    buffer_size=self._feeder_buffer_size,
                )
            else:
                self._feeder.reseed(lpbh, miner_bytes)
        elif kind == "threshold":
            pass  # Metal has no reconstruction gate.
        elif kind == "pause":
            self._paused = True

    def _stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _close_stream(self) -> None:
        if self._stream is not None and hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self._stream = None

    def iter_results(
        self,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet, int]]:
        """Yield ``(model, sampleset, generation)`` until stop or pause."""
        while not self._stop() and not self._paused:
            if self._feeder is None:
                return  # no round yet; driver idles on ctl_q before iterating
            if self._stream is None:
                self._stream = self._sampler.sample_ising_streaming(
                    self._feeder,
                    num_reads=self._num_reads,
                    num_sweeps=self._num_sweeps,
                    max_threadgroups=self._max_threadgroups,
                    duty_cycle=self._duty_cycle,
                    scheduler=self._scheduler,
                )
            try:
                model, ss = next(self._stream)
            except StopIteration:
                self._close_stream()
                continue
            yield model, ss, self.generation

    def cleanup(self) -> None:
        """Release the stream, feeder, and sampler."""
        self._close_stream()
        if self._feeder is not None:
            try:
                self._feeder.stop()
            except Exception as exc:  # noqa: BLE001 — log; a leak must show
                log.warning("metal ctx cleanup: feeder.stop failed: %s", exc)
            self._feeder = None
        if hasattr(self._sampler, "close"):
            try:
                self._sampler.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("metal ctx cleanup: sampler.close failed: %s", exc)


def build_persistent_context(
    *,
    miner_id: str,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    feeder_buffer_size: int,
    num_reads: int,
    num_sweeps: int,
    topology: Any = None,
    utilization: int = 100,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
    **_ignored: Any,
) -> MetalStreamContext:
    """Build the Metal producer context (runs in the stream-driver process).

    Constructs the ``MetalSASampler`` + a ``MetalScheduler``/``DutyCycleController``
    ONCE here (the scheduler gives the GPU core budget = ``max_threadgroups`` and
    drives IOKit-feedback throttling to honor ``utilization``); the feeder is
    created lazily on the first ``switch``. ``**_ignored`` absorbs extra kwargs
    the generic driver passes that Metal does not need.
    """
    from GPU.metal_miner import get_gpu_core_count
    from GPU.metal_sa import MetalSASampler
    from GPU.metal_scheduler import DutyCycleController, MetalScheduler

    sampler = MetalSASampler(topology=topology)
    scheduler = MetalScheduler(
        gpu_core_count=get_gpu_core_count(),
        gpu_utilization_pct=utilization,
        yielding=True,
    )
    duty_cycle = DutyCycleController(target_pct=utilization)
    return MetalStreamContext(
        sampler=sampler,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        max_threadgroups=scheduler.get_core_budget(),
        duty_cycle=duty_cycle,
        scheduler=scheduler,
        stop_event=stop_event,
    )
