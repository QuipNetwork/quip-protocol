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
import weakref
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterator, List, Optional, Sequence

from shared.allowed_value_spec import AllowedValueSpec
from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    derive_nonce,
    generate_ising_model_from_nonce,
)

logger = logging.getLogger(__name__)

_SPAWN_CTX = _mp.get_context("spawn")


def _require_len(name: str, value: bytes, n: int = 32) -> None:
    """Raise ValueError if *value* is not exactly *n* bytes long."""
    if len(value) != n:
        raise ValueError(f"{name} must be {n} bytes, got {len(value)}")


def _default_max_workers() -> int:
    """Generator-process count to use when the caller passes ``max_workers=None``.

    Scales to the node: one worker per core minus one reserved for the
    consuming process (the QPU submit-pool / GPU / CPU sampler all run there),
    with a floor of 2. Workers only run when ``_fill`` needs to refill the
    buffer, so over-provisioning on a many-core box is harmless — surplus
    workers idle rather than burning CPU.
    """
    return max(2, (os.cpu_count() or 4) - 1)


def _empty_feeder_counters() -> dict:
    """Fresh zeroed cumulative-counter dict for RandomIsingFeeder stats."""
    return {
        "max_depth_seen": 0,
        "min_depth_seen": 0,
        "drained_count": 0,
        "pop_wait_total_s": 0.0,
        "pop_wait_count": 0,
    }


def _generate_one_model(
    last_proof_block_hash: bytes,
    miner_bytes: bytes,
    nodes: list,
    edges: list,
    salt: bytes,
    allowed_h: Optional[AllowedValueSpec] = None,
    prep_fn: Optional[Callable] = None,
    prep_args: tuple = (),
) -> Any:
    """Generate one model in a worker process, optionally post-processing it.

    ``last_proof_block_hash`` and ``miner_bytes`` are fixed 32-byte inputs to
    ``derive_nonce``. The hash is ``block_hash(LastProofBlock)`` — the
    round-scoped seed that only changes when a new proof wins. Callers
    must supply the canonical miner identity
    (``blake2_256(SCALE(account))`` for substrate accounts).

    ``allowed_h`` overrides the per-node field distribution (``None`` keeps the
    chain's ternary default). It is a frozen ``AllowedValueSet`` and therefore
    picklable across the spawn-context ``ProcessPoolExecutor``.

    ``prep_fn`` (with ``prep_args``) runs in this worker on the fresh model and
    its return value is what the feeder buffers — used by the QPU PoW path to do
    the GIL-held defect-clamp + array reduction here (off the submit path)
    instead of in the consuming process. Must be a module-level picklable
    callable. ``None`` returns the raw :class:`IsingModel`.
    """
    nonce = derive_nonce(last_proof_block_hash, miner_bytes, salt)
    h, J = generate_ising_model_from_nonce(nonce, nodes, edges, allowed_h=allowed_h)
    model = IsingModel(h=h, J=J, nonce=nonce, salt=salt)
    if prep_fn is not None:
        return prep_fn(model, *prep_args)
    return model


def _force_shutdown_pool(pool: ProcessPoolExecutor) -> None:
    """Shut down a ProcessPoolExecutor so its manager thread is guaranteed
    to terminate.

    ``concurrent.futures`` registers an interpreter-exit hook
    (``_python_exit``) that joins every live executor's manager thread with
    NO timeout. A still-running manager therefore deadlocks interpreter
    shutdown. ``shutdown(wait=False)`` alone only *signals* shutdown and can
    leave the manager blocked in ``wait_result_broken_or_wakeup`` waiting on
    a worker, so this:

    1. signals shutdown (non-blocking) so no new workers are spawned,
    2. SIGKILLs the workers — uncatchable, so a worker that inherited a
       SIGTERM handler via spawn re-import (e.g. the CPU miner's
       ``_cleanup_handler``) can't swallow it and stay alive, and the
       manager observes the broken pool,
    3. polls the manager thread, re-killing any worker that was still
       mid-spawn (no PID yet) at step 2 and only appears in
       ``_processes`` afterwards — that late worker would otherwise keep
       the manager alive,
    4. bounded-joins so a genuinely wedged manager leaks (logged) rather
       than hanging the caller.

    Never raises — it is also invoked from a ``weakref.finalize`` callback,
    where a propagated exception is swallowed by the GC but can corrupt
    interpreter state during finalization.
    """
    try:
        manager_thread = getattr(pool, "_executor_manager_thread", None)
        # Signal shutdown first so the executor stops spawning replacements.
        pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + 10.0
        while True:
            # ``_processes`` flips to None once the executor finishes its own
            # internal join — tolerate that.
            procs = getattr(pool, "_processes", None) or {}
            for proc in list(procs.values()):
                if proc.pid is not None:
                    try:
                        os.kill(proc.pid, _signal.SIGKILL)
                    except OSError:
                        pass
            if manager_thread is None:
                break
            manager_thread.join(timeout=0.2)
            if not manager_thread.is_alive() or time.monotonic() > deadline:
                break
        if manager_thread is not None and manager_thread.is_alive():
            logger.warning(
                "RandomIsingFeeder: executor manager thread still alive "
                "10s after worker SIGKILL; pool may block interpreter "
                "shutdown",
            )
    except Exception:
        pass


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
        max_workers: Worker processes for model generation. ``None``
            (the default) auto-scales to the node via
            :func:`_default_max_workers` (``max(2, cpu_count - 1)``) so a
            fast submit/consume path can't out-drain a fixed 2-worker pool
            on a multi-core box.
        seed: Optional seed for deterministic salt generation.
        allowed_h: Optional per-node field distribution override forwarded to
            ``generate_ising_model_from_nonce``. ``None`` keeps the chain's
            ternary default (``h in {-1, 0, +1}``); pass
            ``AllowedValueSet((0,))`` for the zero-field (J-only) problem class.
    """

    def __init__(
        self,
        last_proof_block_hash: bytes,
        miner_bytes: bytes,
        nodes: list,
        edges: list,
        buffer_size: int = 8,
        max_workers: Optional[int] = None,
        seed: Optional[int] = None,
        allowed_h: Optional[AllowedValueSpec] = None,
        prep_fn: Optional[Callable] = None,
        prep_args: tuple = (),
    ):
        _require_len("last_proof_block_hash", last_proof_block_hash)
        _require_len("miner_bytes", miner_bytes)
        self._last_proof_block_hash = last_proof_block_hash
        self._miner_id = miner_bytes
        self._nodes = nodes
        self._edges = edges
        self._buffer_size = buffer_size
        self._allowed_h = allowed_h
        self._prep_fn = prep_fn
        self._prep_args = prep_args
        self._rng = random.Random(seed) if seed is not None else None
        workers = max_workers if max_workers is not None else _default_max_workers()
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_SPAWN_CTX,
        )
        # Backstop: shut down the pool even if stop() is never called (the
        # feeder is dropped and GC-ed mid-run). stop() detaches this
        # finalizer to avoid double-shutdown. Note: at true interpreter exit
        # this runs too late to help — concurrent.futures' _python_exit join
        # fires first — which is why robust teardown lives in stop().
        self._finalizer = weakref.finalize(
            self,
            _force_shutdown_pool,
            self._pool,
        )
        self._futures: list = []
        # Holds IsingModel by default, or whatever ``prep_fn`` returns (e.g. a
        # ReducedProblem on the QPU PoW path).
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stopped = False
        # Throughput diagnostics: continuously updated in _fill() and
        # pop_blocking(). Read by telemetry to confirm the buffer stays
        # full under load (drained_count > 0 means the QPU stream out-ran
        # the worker pool).
        self._stats: dict = _empty_feeder_counters()
        self._min_depth_init = False
        self._fill()

    def reseed(self, last_proof_block_hash: bytes, miner_bytes: bytes) -> None:
        """Swap the round seed in place, keeping the worker pool alive.

        A new chain head changes ``last_proof_block_hash`` (and in principle
        the miner identity). Re-deriving models requires only the new seed —
        not a new ``ProcessPoolExecutor`` (that fork is the only expensive
        part of feeder construction). This:

        1. Cancels + drops all in-flight futures (they were submitted under
           the OLD seed; their results must never reach the new round).
        2. Drains the ready buffer (old-seed models).
        3. Installs the new seed and refills under it.

        Args:
            last_proof_block_hash: New 32-byte ``block_hash(LastProofBlock)``.
            miner_bytes: Canonical 32-byte miner identity (unchanged in
                practice, but accepted for a clean interface).

        Raises:
            ValueError: If either argument is not exactly 32 bytes.
        """
        _require_len("last_proof_block_hash", last_proof_block_hash)
        _require_len("miner_bytes", miner_bytes)
        # Abandon old-seed in-flight work. cancel() is best-effort (a
        # running worker keeps going), but clearing _futures means _fill
        # never harvests the result, so no old-seed model can reach the new
        # round.
        for f in self._futures:
            f.cancel()
        self._futures.clear()
        # Drain ready buffer of old-seed models.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._last_proof_block_hash = last_proof_block_hash
        self._miner_id = miner_bytes
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
                        exc,
                        len(still_pending),
                        self._queue.qsize(),
                        self._buffer_size,
                    )
            else:
                still_pending.append(f)
        self._futures = still_pending

        submitted = 0
        while len(self._futures) + self._queue.qsize() < self._buffer_size:
            salt = self._make_salt()
            f = self._pool.submit(
                _generate_one_model,
                self._last_proof_block_hash,
                self._miner_id,
                self._nodes,
                self._edges,
                salt,
                self._allowed_h,
                self._prep_fn,
                self._prep_args,
            )
            self._futures.append(f)
            submitted += 1

        # Buffer state visibility: log when the feeder is drained
        # (callers fighting for queue slots) or when workers failed.
        ready = self._queue.qsize()
        pending = len(self._futures)
        self._record_depth(ready)
        if failures or ready == 0:
            logger.info(
                "RandomIsingFeeder state: ready=%d pending=%d "
                "buffer_size=%d submitted=%d failures=%d",
                ready,
                pending,
                self._buffer_size,
                submitted,
                failures,
            )

    def _record_depth(self, ready: int) -> None:
        s = self._stats
        if ready == 0:
            s["drained_count"] += 1
        if ready > s["max_depth_seen"]:
            s["max_depth_seen"] = ready
        if not self._min_depth_init or ready < s["min_depth_seen"]:
            s["min_depth_seen"] = ready
            self._min_depth_init = True

    def stats(self) -> dict:
        """Snapshot of feeder activity counters plus current depth.

        Returns:
            Dict with cumulative counters (``max_depth_seen``,
            ``min_depth_seen``, ``drained_count``, ``pop_wait_total_s``,
            ``pop_wait_count``) plus point-in-time ``ready``, ``pending``,
            and ``buffer_size``. Safe to call from the same process that
            owns the feeder.
        """
        snap = dict(self._stats)
        snap["ready"] = self._queue.qsize()
        snap["pending"] = len(self._futures)
        snap["buffer_size"] = self._buffer_size
        return snap

    def reset_stats(self) -> None:
        """Zero the cumulative counters (point-in-time fields unaffected)."""
        self._stats = _empty_feeder_counters()
        self._min_depth_init = False

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
        raise AssertionError(
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
        assert self._futures, "RandomIsingFeeder: no pending work and empty queue"
        fut = self._futures.pop(0)
        t0 = time.monotonic()
        model = fut.result()
        waited = time.monotonic() - t0
        self._stats["pop_wait_total_s"] += waited
        self._stats["pop_wait_count"] += 1
        if waited > 1.0:
            logger.info(
                "RandomIsingFeeder.pop_blocking waited %.2fs for a "
                "worker (pending=%d, queue=%d)",
                waited,
                len(self._futures),
                self._queue.qsize(),
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
        """Shutdown pool and force-kill any surviving workers.

        Idempotent: a second call returns early. The first call already
        shut down the pool (after which ``_pool._processes`` can be ``None``)
        and detached the finalizer, so re-running the body would raise
        ``AttributeError`` on ``_processes.values()``.
        """
        if self._stopped:
            return
        self._stopped = True
        for f in self._futures:
            f.cancel()
        self._futures.clear()

        # SIGKILL the workers and join the manager thread (see
        # _force_shutdown_pool) so concurrent.futures' no-timeout
        # interpreter-exit join can't deadlock on a surviving manager.
        _force_shutdown_pool(self._pool)

        # Detach the weakref finalizer so it doesn't double-shutdown the
        # already-closed pool when this feeder is later garbage-collected.
        self._finalizer.detach()


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
                f"FixedIsingFeeder requires at least one IsingModel, got {len(models)}"
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

    def stats(self) -> dict:
        """Static snapshot matching :meth:`RandomIsingFeeder.stats`.

        The cycling adapter has no buffer to drain, so the depth and
        wait counters are pinned at values that say "always full, never
        waited" — callers downstream can read the same dict shape
        regardless of which feeder backend is active.
        """
        n = len(self._models)
        return {
            "max_depth_seen": n,
            "min_depth_seen": n,
            "drained_count": 0,
            "pop_wait_total_s": 0.0,
            "pop_wait_count": 0,
            "ready": n,
            "pending": 0,
            "buffer_size": n,
        }

    def reset_stats(self) -> None:
        """No-op for API parity with :class:`RandomIsingFeeder`."""


def build_feeder(spec, nodes, edges, buffer_size, allowed_h=None,
                 prep_fn=None, prep_args=()):
    """Build an IsingFeeder from a switch ``feeder_spec`` tuple.

    ``("pow", last_proof_block_hash, miner_bytes)`` -> ``RandomIsingFeeder``.
    ``("mempool", attach_args, slot)`` -> ``FixedIsingFeeder`` (non-owner
    attaches a ``ProblemView`` from ``attach_args`` and reads h/J from ``slot``).

    Args:
        spec: A feeder-spec tuple whose first element is the kind string.
        nodes: Topology node list passed through to the feeder.
        edges: Topology edge list passed through to the feeder.
        buffer_size: Target number of ready + in-flight models.
        allowed_h: The chain topology's ``allowed_h_values`` spec. Required on
            the PoW path: a ``None`` would make ``RandomIsingFeeder`` fall back
            to the legacy ternary spec, so the sampler optimizes ``h in
            {-1,0,+1}`` while the chain scores the registered spec (e.g. h=0),
            yielding 0 valid solutions. Ignored for the mempool path (fixed
            h/J come from the ProblemView).

    Returns:
        A configured feeder implementing the pop/stop interface.

    Raises:
        ValueError: If ``spec[0] == "pow"`` and ``allowed_h`` is None.
        NotImplementedError: If ``spec[0]`` is neither ``"pow"`` nor
            ``"mempool"``.
    """
    kind = spec[0]
    if kind == "pow":
        if allowed_h is None:
            raise ValueError(
                "build_feeder('pow', ...) requires allowed_h (the chain "
                "topology's allowed_h_values). None silently builds a ternary "
                "h model the chain scores as the registered spec, producing 0 "
                "valid solutions. Thread requirements.allowed_h_values through "
                "sample_ctx -> _stream_factory_kwargs -> build_persistent_context."
            )
        _, last_proof_block_hash, miner_bytes = spec
        return RandomIsingFeeder(
            last_proof_block_hash=last_proof_block_hash,
            miner_bytes=miner_bytes,
            nodes=nodes,
            edges=edges,
            buffer_size=buffer_size,
            allowed_h=allowed_h,
            prep_fn=prep_fn,
            prep_args=prep_args,
        )
    elif kind == "mempool":
        from shared.ring_views import ProblemView
        from shared.ising_model import IsingModel
        _, attach_args, slot = spec
        pv = ProblemView(**attach_args)  # non-owner attach
        try:
            h_vec, j_vec = pv.read(slot)
            model = IsingModel(
                h={int(n): float(h_vec[i]) for i, n in enumerate(nodes)},
                J={(int(e[0]), int(e[1])): float(j_vec[k])
                   for k, e in enumerate(edges)},
                nonce=b"\x00" * 32,
                salt=b"\x00" * 32,
            )
        finally:
            pv.close()  # non-owner: close (never unlink — the worker owns it)
        return FixedIsingFeeder(models=[model])
    raise NotImplementedError(f"feeder spec kind not yet supported: {kind!r}")


__all__ = [
    "FixedIsingFeeder",
    "RandomIsingFeeder",
    "build_feeder",
]
