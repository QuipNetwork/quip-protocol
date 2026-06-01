"""Background Ising model generator with ProcessPoolExecutor.

Uses processes with 'spawn' context to avoid inheriting CUDA
state from the parent. Ising model generation involves
Python-level dict comprehension that holds the GIL, so threads
would serialize the work.
"""
from __future__ import annotations

import logging
import multiprocessing as _mp
import os
import queue
import random
import signal as _signal
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)

logger = logging.getLogger(__name__)

_SPAWN_CTX = _mp.get_context('spawn')


def _generate_one_model(
    prev_hash: bytes,
    miner_id: str,
    cur_index: int,
    nodes: list,
    edges: list,
    salt: bytes,
) -> IsingModel:
    """Generate one IsingModel in a worker process."""
    nonce = ising_nonce_from_block(
        prev_hash, miner_id, cur_index, salt,
    )
    h, J = generate_ising_model_from_nonce(
        nonce, nodes, edges,
    )
    return IsingModel(h=h, J=J, nonce=nonce, salt=salt)


def _kill_workers(pids: list[int], timeout: float = 3.0):
    """SIGTERM workers, wait, then SIGKILL survivors.

    Logs a warning if SIGKILL is needed — that indicates a
    bug in the worker shutdown path.
    """
    alive = []
    for pid in pids:
        try:
            os.kill(pid, _signal.SIGTERM)
            alive.append(pid)
        except OSError:
            pass

    if not alive:
        return

    deadline = time.monotonic() + timeout
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = [
            p for p in alive if _pid_alive(p)
        ]

    for pid in alive:
        logger.warning(
            "IsingFeeder: worker %d did not exit after "
            "SIGTERM, sending SIGKILL", pid,
        )
        try:
            os.kill(pid, _signal.SIGKILL)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    """Check if a process is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class IsingFeeder:
    """Keeps a buffer of pre-generated IsingModels full.

    Uses a ProcessPoolExecutor (spawn context) to generate
    models in background processes. Spawn avoids inheriting
    CUDA driver state from the parent process.

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
            mp_context=_SPAWN_CTX,
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
        failures = 0
        for f in self._futures:
            if f.done():
                try:
                    self._queue.put_nowait(f.result())
                except Exception as exc:
                    failures += 1
                    logger.warning(
                        "IsingFeeder worker failed: %s (pending=%d, "
                        "queue=%d, buffer_size=%d)",
                        exc, len(still_pending),
                        self._queue.qsize(), self._buffer_size,
                    )
            else:
                still_pending.append(f)
        self._futures = still_pending

        submitted = 0
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
            submitted += 1

        # Buffer state visibility: log when the feeder is drained
        # (callers fighting for queue slots) or when workers failed.
        ready = self._queue.qsize()
        pending = len(self._futures)
        if failures or ready == 0:
            logger.info(
                "IsingFeeder state: ready=%d pending=%d "
                "buffer_size=%d submitted=%d failures=%d",
                ready, pending, self._buffer_size,
                submitted, failures,
            )

    def __iter__(self):
        return self

    def __next__(self) -> IsingModel:
        return self.pop_blocking()

    def pop(self) -> IsingModel:
        """Pop one model. Never blocks.

        The buffer should always have ready models. If not,
        that's a programming error — the buffer_size is too
        small or _fill() isn't being called frequently enough.
        """
        self._fill()
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            pass
        # Check if any future is already done
        for i, f in enumerate(self._futures):
            if f.done():
                model = self._futures.pop(i).result()
                self._fill()
                return model
        assert self._futures, (
            "IsingFeeder: no pending work and empty queue"
        )
        assert False, (
            f"IsingFeeder buffer underrun: "
            f"{len(self._futures)} futures pending, "
            f"none ready. Increase buffer_size."
        )

    def pop_blocking(self) -> IsingModel:
        """Pop one model, waiting for a worker if needed.

        Used only during cold start when the buffer hasn't
        filled yet. Once the pipeline is running, use pop()
        (non-blocking) or try_pop() instead.

        No timeout on .result() — cold start of spawn-context
        workers on a loaded node can exceed any arbitrary bound
        (heavy imports + topology parse happen on first task).
        A shorter timeout only orphans the future: pop(0) has
        already removed it from self._futures, so a timeout means
        the worker keeps running with no one reading its result.
        Cancellation is handled one level up via the mining
        loop's stop_event.
        """
        self._fill()
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            pass
        assert self._futures, (
            "IsingFeeder: no pending work and empty queue"
        )
        fut = self._futures.pop(0)
        t0 = time.monotonic()
        model = fut.result()
        waited = time.monotonic() - t0
        if waited > 1.0:
            logger.info(
                "IsingFeeder.pop_blocking waited %.2fs for a "
                "worker (pending=%d, queue=%d)",
                waited, len(self._futures), self._queue.qsize(),
            )
        self._fill()
        return model

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
        models = [self.pop_blocking()]
        for _ in range(n - 1):
            m = self.try_pop()
            if m is None:
                break
            models.append(m)
        return models

    def stop(self) -> None:
        """Shutdown pool and force-kill any surviving workers."""
        self._stopped = True
        for f in self._futures:
            f.cancel()
        self._futures.clear()

        # Collect worker PIDs before shutdown — after
        # shutdown() the process objects may be gone.
        pids = [
            p.pid for p in self._pool._processes.values()
            if p.pid is not None
        ]

        self._pool.shutdown(
            wait=False, cancel_futures=True,
        )

        _kill_workers(pids)

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
