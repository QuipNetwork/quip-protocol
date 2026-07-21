# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Backend-agnostic stream-driver context.

Owns a feeder + generation + pause; drives ``sampler.sample_ising_streaming``
and yields generation-tagged samplesets into the ring (via QPU/stream_driver).
The sampler is the only backend-specific object; the feeder is built from the
switch's ``feeder_spec``. Replaces the per-backend contexts (Metal now; QPU in
a later unification step).
"""
from __future__ import annotations

import logging
import multiprocessing.synchronize
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import dimod

from shared.ising_feeder import build_feeder
from shared.ising_model import IsingModel

log = logging.getLogger(__name__)


def stream_from_feeder(sampler, feeder, *, num_reads, num_sweeps):
    """Stream ``(model, sampleset)`` pairs from a caller-owned feeder.

    Shared body for the trivial ``sample_ising_streaming`` generators (CPU SA,
    Modal): pop a model, sample it via ``sampler.sample_ising``, yield the pair
    until the feeder drains. The feeder is owned by ``StreamContext`` and never
    stopped here. PEP 479: returning on ``StopIteration`` lets the driver
    reseed cleanly instead of turning a bare StopIteration into a RuntimeError.
    """
    while True:
        try:
            model = feeder.pop_blocking()
        except StopIteration:
            return
        ss = sampler.sample_ising(
            h=model.h, J=model.J,
            num_reads=num_reads, num_sweeps=num_sweeps,
        )
        yield model, ss


class StreamContext:
    """Generic producer context: feeder (from spec) + sampler -> tagged sets."""

    def __init__(
        self,
        *,
        sampler: Any = None,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        feeder_buffer_size: int,
        num_reads: int,
        num_sweeps: int,
        allowed_h: Optional[Any] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        feeder_builder: Optional[Callable] = None,
        feeder_prep_fn: Optional[Callable] = None,
        feeder_prep_args: tuple = (),
        stop_event: Optional[multiprocessing.synchronize.Event] = None,
    ) -> None:
        self._sampler = sampler
        self._nodes = nodes
        self._edges = edges
        # The problem's per-node field spec (chain ``allowed_h_values``). It is
        # topology-bound like nodes/edges, so it is fixed for the driver's
        # lifetime and forwarded to the PoW feeder. ``None`` would make the
        # feeder silently build a ternary h model the chain scores as h=0, so
        # ``build_feeder`` rejects a None on the PoW path.
        self._allowed_h = allowed_h
        self._feeder_buffer_size = feeder_buffer_size
        self._num_reads = num_reads
        self._num_sweeps = num_sweeps
        self._sampler_kwargs = sampler_kwargs or {}
        # ``num_reads``/``num_sweeps`` are passed explicitly to the sampler;
        # a backend that also put them in sampler_kwargs would otherwise hit a
        # cryptic "multiple values for keyword argument" at stream-open time.
        clash = {"num_reads", "num_sweeps"} & self._sampler_kwargs.keys()
        if clash:
            raise ValueError(
                f"sampler_kwargs must not contain {sorted(clash)} "
                "(passed explicitly per round)"
            )
        self._feeder_builder = feeder_builder or build_feeder
        # Optional per-model post-processing the feeder workers run (QPU PoW path:
        # prepare_reduced — defect-clamp + array reduction off the submit path).
        self._feeder_prep_fn = feeder_prep_fn
        self._feeder_prep_args = feeder_prep_args
        self._stop_event = stop_event
        self._feeder: Optional[Any] = None
        self._feeder_kind: Optional[str] = None
        self._stream: Optional[Any] = None
        self.generation: int = 0
        self._paused = False

    def apply_command(self, cmd: Tuple[Any, ...]) -> None:
        """Dispatch a control command from the driver's ctl_q.

        Supported commands:
            ``("switch", gen, lpbh, miner_bytes, thr, num_reads, anneal,
               num_sweeps, feeder_spec)``
                — build/reseed feeder from spec, bump generation.
            ``("threshold", gen, value)`` — no-op (backends gate via sampler_kwargs).
            ``("pause", gen)``            — halt iter_results until next switch.
        """
        kind = cmd[0]
        if kind == "switch":
            self.generation = int(cmd[1])
            self._num_reads = int(cmd[5])
            self._num_sweeps = int(cmd[7])
            feeder_spec = cmd[8]
            self._paused = False
            self._close_stream()
            self._apply_feeder_spec(feeder_spec)
        elif kind == "threshold":
            pass  # backends that gate on threshold do so via sampler_kwargs
        elif kind == "pause":
            self._paused = True

    def _apply_feeder_spec(self, spec: Tuple[Any, ...]) -> None:
        skind = spec[0]
        if skind == "pow" and self._feeder is not None and self._feeder_kind == "pow":
            self._feeder.reseed(spec[1], spec[2])
        else:
            if self._feeder is not None:
                try:
                    self._feeder.stop()
                except Exception:  # noqa: BLE001 — best-effort
                    pass
            self._feeder = self._feeder_builder(
                spec, self._nodes, self._edges, self._feeder_buffer_size,
                allowed_h=self._allowed_h,
                prep_fn=self._feeder_prep_fn,
                prep_args=self._feeder_prep_args,
            )
            self._feeder_kind = skind

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
        """Yield ``(model, sampleset, generation)`` until stop or pause.

        Requires a sampler (the in-process / single-driver path). The split
        feeder-driver has no connection and uses :meth:`iter_problems` instead.
        """
        if self._sampler is None:
            raise RuntimeError(
                "iter_results requires a sampler; the connection-less feeder "
                "driver must use iter_problems()"
            )
        while not self._stop() and not self._paused:
            if self._feeder is None:
                return
            if self._stream is None:
                self._stream = self._sampler.sample_ising_streaming(
                    self._feeder, num_reads=self._num_reads,
                    num_sweeps=self._num_sweeps, **self._sampler_kwargs,
                )
            try:
                model, ss = next(self._stream)
            except StopIteration:
                self._close_stream()
                continue
            yield model, ss, self.generation

    def iter_problems(self) -> Iterator[Tuple[Any, int]]:
        """Yield ``(ReducedProblem, generation)`` from the feeder until stop/pause.

        The connection-less feeder-driver (process A) side: no sampler, no pump
        — just drain the feeder so the driver can relay reduced problems into the
        ProblemView ring. ``pop_blocking`` returns promptly (the feeder keeps a
        ready buffer); stop/pause are observed between pops.
        """
        while not self._stop() and not self._paused:
            if self._feeder is None:
                return
            try:
                rp = self._feeder.pop_blocking()
            except StopIteration:
                return
            yield rp, self.generation

    def cleanup(self) -> None:
        """Release the stream, feeder, and sampler."""
        self._close_stream()
        if self._feeder is not None:
            try:
                self._feeder.stop()
            except Exception as exc:  # noqa: BLE001 — log; a leak must show
                log.warning("stream ctx cleanup: feeder.stop failed: %s", exc)
            self._feeder = None
        if hasattr(self._sampler, "close"):
            try:
                self._sampler.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("stream ctx cleanup: sampler.close failed: %s", exc)
