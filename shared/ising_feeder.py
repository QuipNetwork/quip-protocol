"""Pluggable Ising-model feeders shared by every miner backend.

Two implementations live here:

- :class:`RandomIsingFeeder` — background generator using a
  :class:`concurrent.futures.ProcessPoolExecutor` (spawn context, to avoid
  inheriting CUDA driver state). Used by the PoW path, where every iteration
  needs a fresh ``(salt -> nonce -> h, J)`` derivation.
- :class:`FixedIsingFeeder` — in-memory adapter for mempool jobs, where the
  Ising problem is carried directly in the job order and ``BaseMiner`` just
  needs to keep replaying the same model(s) across sampler iterations.

Both expose the same ``pop`` / ``pop_blocking`` / ``try_pop`` / ``pop_n`` /
``__iter__`` / ``stop`` surface so ``BaseMiner.mine_work_item`` can pop
models from a feeder without caring which backend supplied it.
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing as _mp
import os
import queue
import random
import signal as _signal
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, List, Optional, Sequence

from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    derive_nonce,
    generate_ising_model_from_nonce,
)

logger = logging.getLogger(__name__)

_SPAWN_CTX = _mp.get_context('spawn')


def _generate_one_model(
    last_proof_block_hash: bytes,
    miner_bytes: bytes,
    nodes: list,
    edges: list,
    salt: bytes,
) -> IsingModel:
    """Generate one IsingModel in a worker process.

    ``last_proof_block_hash`` and ``miner_bytes`` are fixed 32-byte inputs to
    ``derive_nonce``. The hash is ``block_hash(LastProofBlock)`` — the
    round-scoped seed that only changes when a new proof wins. Callers
    must supply the canonical miner identity
    (``blake2_256(SCALE(account))`` for substrate accounts).
    """
    nonce = derive_nonce(last_proof_block_hash, miner_bytes, salt)
    h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
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
            "RandomIsingFeeder: worker %d did not exit after "
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


class RandomIsingFeeder:
    """Keeps a buffer of freshly-derived ``IsingModel``s ready to pop.

    Uses a ProcessPoolExecutor (spawn context) to generate
    models in background processes. Spawn avoids inheriting
    CUDA driver state from the parent process.

    Args:
        last_proof_block_hash: ``block_hash(LastProofBlock)`` (32 bytes) —
            the time-bound input to ``derive_nonce`` that changes only when
            a new winning proof lands.
        miner_bytes: Canonical 32-byte miner identity
            (``blake2_256(SCALE(account_id))`` for substrate accounts).
        nodes: Topology node list.
        edges: Topology edge list.
        buffer_size: Target number of ready + in-flight models.
        max_workers: Worker processes for model generation.
        seed: Optional seed for deterministic salt generation.
    """

    def __init__(
        self,
        last_proof_block_hash: bytes,
        miner_bytes: bytes,
        nodes: list,
        edges: list,
        buffer_size: int = 8,
        max_workers: int = 2,
        seed: Optional[int] = None,
    ):
        if len(last_proof_block_hash) != 32:
            raise ValueError(
                "last_proof_block_hash must be 32 bytes, got "
                f"{len(last_proof_block_hash)}"
            )
        if len(miner_bytes) != 32:
            raise ValueError(
                f"miner_bytes must be 32 bytes, got {len(miner_bytes)}"
            )
        self._last_proof_block_hash = last_proof_block_hash
        self._miner_id = miner_bytes
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
                        "RandomIsingFeeder worker failed: %s (pending=%d, "
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
                self._last_proof_block_hash,
                self._miner_id,
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
                "RandomIsingFeeder state: ready=%d pending=%d "
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
            "RandomIsingFeeder: no pending work and empty queue"
        )
        assert False, (
            f"RandomIsingFeeder buffer underrun: "
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
            "RandomIsingFeeder: no pending work and empty queue"
        )
        fut = self._futures.pop(0)
        t0 = time.monotonic()
        model = fut.result()
        waited = time.monotonic() - t0
        if waited > 1.0:
            logger.info(
                "RandomIsingFeeder.pop_blocking waited %.2fs for a "
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

class FixedIsingFeeder:
    """Cycles through a fixed list of pre-baked ``IsingModel``s forever.

    Adapter used by the mempool mining path, where the Ising problem is
    carried directly in the job order rather than derived per-iteration.
    The job currently always supplies a single ``(h, J)`` pair, so the
    typical instance wraps a one-element list; cycling that list with
    :func:`itertools.cycle` keeps yielding the same model so the sampler
    can re-roll its internal RNG and find different solutions over many
    iterations. The list-of-models shape is forward-looking — future
    mempool jobs may carry several problems per order, and this feeder
    will round-robin them without changes to ``BaseMiner``.

    No background pool, no salt generation, no round seeding: the values
    were already finalised when the order landed on chain. ``stop()`` is
    a no-op kept for API parity with :class:`RandomIsingFeeder` so
    ``BaseMiner.mine_work_item`` can call it unconditionally.

    Args:
        models: One or more pre-built ``IsingModel`` instances. The list
            is cycled in-order; length must be ``>= 1``.

    Raises:
        ValueError: If ``models`` is empty.
    """

    def __init__(self, models: Sequence[IsingModel]) -> None:
        if len(models) < 1:
            raise ValueError(
                "FixedIsingFeeder requires at least one IsingModel, "
                f"got {len(models)}"
            )
        # Materialise the input — accepting a Sequence keeps the
        # constructor flexible (lists, tuples, generators that have
        # already been list()-ed). Cycle works on the materialised copy
        # so the feeder is independent of the caller's container.
        self._models: List[IsingModel] = list(models)
        self._cycle: Iterator[IsingModel] = itertools.cycle(self._models)
        self._stopped = False

    def __iter__(self) -> "FixedIsingFeeder":
        return self

    def __next__(self) -> IsingModel:
        return self.pop_blocking()

    def pop(self) -> IsingModel:
        """Return the next model in the cycle. Never blocks."""
        return next(self._cycle)

    def pop_blocking(self) -> IsingModel:
        """Same as :meth:`pop` — no async path to wait on."""
        return next(self._cycle)

    def try_pop(self) -> Optional[IsingModel]:
        """Non-blocking pop. Always returns a model (the list is non-empty)."""
        return next(self._cycle)

    def pop_n(self, n: int) -> List[IsingModel]:
        """Pop ``n`` models, cycling as needed.

        Args:
            n: Number of models to return; must be positive.

        Returns:
            A list of ``n`` models drawn in cycle order.
        """
        assert n > 0, "n must be positive"
        return [next(self._cycle) for _ in range(n)]

    def stop(self) -> None:
        """No-op shutdown. Idempotent — safe to call multiple times."""
        self._stopped = True


__all__ = [
    "FixedIsingFeeder",
    "RandomIsingFeeder",
]
