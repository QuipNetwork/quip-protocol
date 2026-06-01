#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Fail if deleted inline-sampling symbols reappear in source.

After the unified-streaming-stack refactor the producer->ring->consumer driver
is the ONLY mining path; ``_sample``/``_sample_batch`` and the
``STREAMING_PUMP``/``DRIVER_OWNS_FEEDER`` toggles no longer exist. Every backend
(QPU, Metal, CUDA, CPU, Modal) and both job types (PoW, mempool) mine through a
stream-driver subprocess (``shared/stream_context.py``) feeding a shared-memory
ring (``shared/ring_views.py``). This guard keeps the deleted inline path and
its mode toggles from creeping back in. See docs/miner-architecture.md.
"""
from __future__ import annotations

import pathlib
import re
import sys

FORBIDDEN = (
    r"def _sample\(",
    r"def _sample_batch\(",
    r"\bSTREAMING_PUMP\b",
    r"\bDRIVER_OWNS_FEEDER\b",
)
ROOTS = ("shared", "QPU", "GPU", "CPU", "substrate")
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    """Scan non-test source for forbidden inline-sampling symbols.

    Returns:
        ``1`` if any forbidden pattern is found (printing each offending
        ``file: matches pattern``), ``0`` when the source is clean.
    """
    bad: list[str] = []
    for root in ROOTS:
        for p in (REPO_ROOT / root).rglob("*.py"):
            text = p.read_text()
            for pat in FORBIDDEN:
                if re.search(pat, text):
                    bad.append(f"{p}: matches {pat!r}")
    if bad:
        print("Inline-sampling symbols are forbidden (unified streaming stack):")
        print("\n".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
