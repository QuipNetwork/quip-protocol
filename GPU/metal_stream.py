# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Metal stream-driver context: GPU sampling in the producer process.

Builds a generic ``StreamContext`` (shared/stream_context.py) wired with the
Metal sampler + Metal-specific sampler_kwargs (max_threadgroups, scheduler).
Run ONLY in the stream-driver process (QPU/stream_driver.py); the worker
consumes the resulting ring zero-copy.
"""
from __future__ import annotations

import multiprocessing.synchronize
from typing import Any, List, Optional, Tuple

from shared.stream_context import StreamContext


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
    yielding: bool = True,
    active_threads: int = 2048,
    idle_after_s: float = 60.0,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
    **_ignored: Any,
) -> StreamContext:
    """Build the Metal producer context (runs in the stream-driver process).

    Constructs the ``MetalSASampler`` + a ``MetalScheduler`` ONCE here. When
    ``yielding`` the scheduler runs the adaptive cap monitor, which senses
    presence/thermal/battery and publishes an *occupancy budget* (max threads
    per command buffer); the sampler re-reads it per batch and splits reads so
    ``problems x reads`` stays under it (keeping the full batch + sweeps).
    ``active_threads`` is the budget while the user is present; idle/headless
    runs uncapped; battery/critical pauses. ``utilization`` sizes the problem
    batch (core budget). ``**_ignored`` absorbs extra driver kwargs.
    """
    from GPU.metal_miner import get_gpu_core_count
    from GPU.metal_sa import MetalSASampler
    from GPU.metal_scheduler import MetalScheduler

    sampler = MetalSASampler(topology=topology)
    scheduler = MetalScheduler(
        gpu_core_count=get_gpu_core_count(),
        gpu_utilization_pct=utilization,
        yielding=yielding,
        active_threads=active_threads,
        idle_after_s=idle_after_s,
    )
    return StreamContext(
        sampler=sampler,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        sampler_kwargs={
            "max_threadgroups": scheduler.get_core_budget(),
            "scheduler": scheduler,
            # The sampler checks this during a PAUSE so a battery / critical-
            # thermal full-stop never blocks teardown.
            "stop_event": stop_event,
        },
        stop_event=stop_event,
    )
