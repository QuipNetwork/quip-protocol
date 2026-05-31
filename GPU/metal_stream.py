# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Metal stream-driver context: GPU sampling in the producer process.

Builds a generic ``StreamContext`` (shared/stream_context.py) wired with the
Metal sampler + Metal-specific sampler_kwargs (max_threadgroups, duty_cycle,
scheduler). Run ONLY in the stream-driver process (QPU/stream_driver.py); the
worker consumes the resulting ring zero-copy.
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
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
    **_ignored: Any,
) -> StreamContext:
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
    return StreamContext(
        sampler=sampler,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        sampler_kwargs={
            "max_threadgroups": scheduler.get_core_budget(),
            "duty_cycle": duty_cycle,
            "scheduler": scheduler,
        },
        stop_event=stop_event,
    )
