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


def active_threads_for_util(active_util: int, sampler: Any, cores: int) -> int:
    """Convert an ``active_util`` % into an absolute thread budget.

    The budget is ``active_util`` percent of the GPU's max concurrent thread
    capacity for this kernel = ``maxTotalThreadsPerThreadgroup x cores`` (1024 x
    40 = 40960 on an M4 Max). Falls back to a 1024 per-threadgroup max if the
    pipeline can't be queried. Returns at least 1.
    """
    try:
        per_tg = int(sampler._pipeline.maxTotalThreadsPerThreadgroup())
    except Exception:  # noqa: BLE001 — degrade to the common Apple GPU max
        per_tg = 1024
    max_threads = max(1, per_tg * max(1, cores))
    return max(1, round(active_util / 100.0 * max_threads))


def build_persistent_context(
    *,
    miner_id: str,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    feeder_buffer_size: int,
    num_reads: int,
    num_sweeps: int,
    allowed_h: Any = None,
    topology: Any = None,
    utilization: int = 100,
    yielding: bool = True,
    active_util: int = 85,
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
    ``active_util`` is the budget while the user is present, expressed as a
    percentage of the GPU's max thread capacity (``maxTotalThreadsPerThreadgroup
    x cores``); idle/headless runs uncapped; battery/critical pauses.
    ``utilization`` sizes the problem batch (core budget). ``**_ignored``
    absorbs extra driver kwargs.
    """
    from GPU.metal_miner import get_gpu_core_count
    from GPU.metal_sa import MetalSASampler
    from GPU.metal_scheduler import MetalScheduler

    sampler = MetalSASampler(topology=topology)
    cores = get_gpu_core_count()
    active_threads = active_threads_for_util(active_util, sampler, cores)
    scheduler = MetalScheduler(
        gpu_core_count=cores,
        gpu_utilization_pct=utilization,
        yielding=yielding,
        active_threads=active_threads,
        idle_after_s=idle_after_s,
    )
    return StreamContext(
        sampler=sampler,
        nodes=nodes,
        edges=edges,
        allowed_h=allowed_h,
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
