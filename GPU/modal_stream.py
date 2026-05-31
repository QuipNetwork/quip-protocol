# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Modal stream-driver context: cloud-GPU sampling in the producer process.

Builds a generic ``StreamContext`` wired with the Modal sampler. Run ONLY in
the stream-driver process (QPU/stream_driver.py); the worker consumes the ring.
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
    gpu_type: str = "t4",
    stop_event: Optional[multiprocessing.synchronize.Event] = None,
    **_ignored: Any,
) -> StreamContext:
    """Build the Modal producer context (runs in the stream-driver process).

    Constructs the ``ModalSampler`` ONCE here; the feeder is created lazily on
    the first ``switch`` (via the feeder spec). ``**_ignored`` absorbs extra
    kwargs the generic driver passes that Modal does not need.
    """
    from GPU.modal_sampler import ModalSampler

    sampler = ModalSampler(gpu_type)
    return StreamContext(
        sampler=sampler,
        nodes=nodes,
        edges=edges,
        feeder_buffer_size=feeder_buffer_size,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        sampler_kwargs={},
        stop_event=stop_event,
    )
