# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA producer factory: builds a CudaMiner's sampler + scheduler in the
stream-driver process and drives the generic StreamContext. Runs ONLY on a
CUDA machine (the producer process); CI never imports cupy via this path.
"""
from __future__ import annotations

import multiprocessing.synchronize
from typing import Any, List, Optional, Tuple

from shared.stream_context import StreamContext

_CUDA_POLL_TIMEOUT_S = 1800.0


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
    device: str = "0",
    update_mode: str = "sa",
    sms_per_nonce: int = 4,
    utilization: int = 100,
    yielding: bool = False,
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
    **_ignored: Any,
) -> StreamContext:
    """Build the CUDA producer context (device + sampler + scheduler).

    Constructs a ``CudaMiner`` (which initialises the CUDA device and
    creates the appropriate sampler + ``KernelScheduler``) ONCE here in
    the stream-driver process. ``**_ignored`` absorbs extra kwargs the
    generic driver passes that CUDA does not need.
    """
    from GPU.cuda_miner import CudaMiner

    # The scheduler is passed into the sampler via sampler_kwargs so the
    # streaming generator honors should_throttle (yielding mode) — restoring the
    # back-off the old inline _sample_batch did — AND so the StreamContext
    # retains the scheduler + its NVML monitor (the built CudaMiner is dropped
    # after we take its sampler). The inline path's adaptive num_kernels rescale
    # is intentionally NOT carried over (fixed per round), mirroring Metal's
    # dropped threadgroup-rescale.
    miner = CudaMiner(
        miner_id,
        device=device,
        topology=topology,
        update_mode=update_mode,
        sms_per_nonce=sms_per_nonce,
        utilization=utilization,
        yielding=yielding,
    )
    num_kernels = max(
        1,
        miner._scheduler.get_sm_budget()
        // miner.sampler._sms_per_nonce,  # type: ignore[union-attr]  # CUDA-only attr  # ty:ignore[unresolved-attribute]
    )
    return StreamContext(
        sampler=miner.sampler,
        nodes=nodes,
        edges=edges,
        allowed_h=allowed_h,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        sampler_kwargs={
            "num_kernels": num_kernels,
            "poll_timeout": _CUDA_POLL_TIMEOUT_S,
            "scheduler": miner._scheduler,
        },
        stop_event=stop_event,
    )
