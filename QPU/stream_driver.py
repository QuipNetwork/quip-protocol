# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Stream-driver process: owns the sampler/feeder/generator and produces
samplesets into a SharedSampleRing for the consumer (miner worker).

Replaces the in-worker QPU pump thread (AGENTS.md). The generator is driven
by exactly one caller (this process's loop). Only a small descriptor crosses
the descriptor queue; the sample matrix is written zero-copy into the ring.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

import numpy as np

from shared.shared_sample_ring import SharedSampleRing


def _resolve(dotted: str):
    module_name, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _extract_qpu_us(sampleset) -> int:
    info = getattr(sampleset, "info", None) or {}
    t = info.get("timing", {}) if info else {}
    return int(t.get("qpu_programming_time", 0) + t.get("qpu_sampling_time", 0))


def stream_driver_main(ring_args: Dict[str, Any], desc_q, stop_event,
                       stream_factory_dotted: str,
                       factory_kwargs: Dict[str, Any]) -> None:
    """Drive the stream; write each result into the ring; enqueue descriptor.

    stream_factory_dotted resolves to a callable returning an iterator of
    (model, sampleset). Descriptor tuple:
    (slot, n_rows, n_cols, nonce_bytes, salt_bytes, qpu_us). A trailing None
    on the queue signals end-of-stream.
    """
    ring = SharedSampleRing(**ring_args)
    make_stream = _resolve(stream_factory_dotted)
    stream = make_stream(**factory_kwargs)
    dropped = 0
    try:
        for model, sampleset in stream:
            if stop_event.is_set():
                break
            sample = np.asarray(sampleset.record.sample, dtype=np.int8)
            energy = np.asarray(sampleset.record.energy, dtype=np.float64)
            n_rows, n_cols = sample.shape
            slot = ring.claim_free(timeout=0.0 if dropped else 0.005)
            if slot is None:
                dropped += 1
                continue
            ring.write(slot, sample, energy)
            try:
                desc_q.put_nowait(
                    (slot, n_rows, n_cols, bytes(model.nonce), bytes(model.salt),
                     _extract_qpu_us(sampleset)))
            except Exception:
                ring.release(slot)
                dropped += 1
    finally:
        desc_q.put(None)
        ring.close_unlink()
