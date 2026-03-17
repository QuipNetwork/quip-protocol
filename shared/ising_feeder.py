"""Background Ising model generator with ProcessPoolExecutor."""
from __future__ import annotations

import logging
import os
import queue
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)

logger = logging.getLogger(__name__)


def _generate_one_model(
    prev_hash: bytes,
    miner_id: str,
    cur_index: int,
    nodes: list,
    edges: list,
    salt: bytes,
) -> IsingModel:
    """Generate one IsingModel in a worker process.

    Module-level function for ProcessPoolExecutor pickling.
    """
    nonce = ising_nonce_from_block(
        prev_hash, miner_id, cur_index, salt,
    )
    h, J = generate_ising_model_from_nonce(
        nonce, nodes, edges,
    )
    return IsingModel(h=h, J=J, nonce=nonce, salt=salt)


class IsingFeeder:
    """Keeps a buffer of pre-generated IsingModels full.

    Uses a ProcessPoolExecutor to generate models in
    background worker processes, avoiding GIL contention
    with GPU work on the main thread.

    Args:
        prev_hash: Current block's previous hash.
        miner_id: Miner identifier string.
        cur_index: Current block index.
        nodes: Topology node list.
        edges: Topology edge list.
        buffer_size: Target number of ready + in-flight models.
        max_workers: Worker processes for model generation.
        seed: Optional seed for deterministic salt generation.
    """

    def __init__(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: list,
        edges: list,
        buffer_size: int = 8,
        max_workers: int = 2,
        seed: Optional[int] = None,
    ):
        self._prev_hash = prev_hash
        self._miner_id = miner_id
        self._cur_index = cur_index
        self._nodes = nodes
        self._edges = edges
        self._buffer_size = buffer_size
        self._rng = (
            random.Random(seed) if seed is not None
            else None
        )
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
        )
        self._futures: list = []
        self._queue: queue.Queue[IsingModel] = queue.Queue()
        self._stopped = False
        self._fill()

    def _make_salt(self) -> bytes:
        """Generate a 32-byte salt."""
        if self._rng is not None:
            return self._rng.randbytes(32)
        return os.urandom(32)

    def _fill(self) -> None:
        """Harvest done futures, submit new work."""
        if self._stopped:
            return
        still_pending = []
        for f in self._futures:
            if f.done():
                try:
                    self._queue.put_nowait(f.result())
                except Exception as exc:
                    logger.warning(
                        "IsingFeeder worker failed: %s",
                        exc,
                    )
            else:
                still_pending.append(f)
        self._futures = still_pending

        while (
            len(self._futures) + self._queue.qsize()
            < self._buffer_size
        ):
            salt = self._make_salt()
            f = self._pool.submit(
                _generate_one_model,
                self._prev_hash,
                self._miner_id,
                self._cur_index,
                self._nodes,
                self._edges,
                salt,
            )
            self._futures.append(f)

    def __iter__(self):
        return self

    def __next__(self) -> IsingModel:
        return self.pop()

    def pop(self) -> IsingModel:
        """Pop one model, blocking if necessary."""
        self._fill()
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._futures:
            logger.debug(
                "IsingFeeder queue empty, waiting",
            )
            model = self._futures.pop(0).result(
                timeout=5.0,
            )
            self._fill()
            return model
        raise RuntimeError(
            "IsingFeeder: no pending work and empty queue",
        )

    def try_pop(self) -> Optional[IsingModel]:
        """Non-blocking pop, returns None if empty."""
        self._fill()
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def pop_n(self, n: int) -> list[IsingModel]:
        """Pop up to n models, blocking only for the first."""
        assert n > 0, "n must be positive"
        models = [self.pop()]
        for _ in range(n - 1):
            m = self.try_pop()
            if m is None:
                break
            models.append(m)
        return models

    def stop(self) -> None:
        """Shutdown pool and cancel pending futures."""
        self._stopped = True
        for f in self._futures:
            f.cancel()
        self._futures.clear()
        self._pool.shutdown(
            wait=False, cancel_futures=True,
        )

    def update_block(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
    ) -> None:
        """Update generation args when block changes.

        Drains stale futures and queue, then refills
        with new block parameters.
        """
        self._prev_hash = prev_hash
        self._miner_id = miner_id
        self._cur_index = cur_index
        for f in self._futures:
            f.cancel()
        self._futures.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._fill()
