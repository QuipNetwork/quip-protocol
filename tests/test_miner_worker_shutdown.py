# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""The worker shutdown op must close the miner (reap its persistent driver)."""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import MagicMock

from shared import miner_worker


def test_shutdown_op_calls_miner_close(monkeypatch):
    """`op: shutdown` must call miner.close() before the worker returns."""
    fake_miner = MagicMock()
    fake_miner.miner_id = "QPU-1"
    monkeypatch.setattr(
        miner_worker,
        "build_miner_from_spec",
        lambda spec: fake_miner,
    )

    req: mp.Queue = mp.Queue()
    resp: mp.Queue = mp.Queue()
    stop = mp.Event()
    req.put({"op": "shutdown"})

    miner_worker.miner_worker_main(
        req,
        resp,
        {"id": "QPU-1", "kind": "qpu"},
        stop,
    )

    fake_miner.close.assert_called_once()
    assert stop.is_set()
