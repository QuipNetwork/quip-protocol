"""Per-dispatch on-disk attempts archive.

Layout (one directory per dispatch_id):

    {base_dir}/
      next_solution_id              # monotonic counter persisted across restarts
      submissions_index.jsonl       # append-only {solution_id, dispatch_id, ...}
                                    # — fast solution_id → dispatch_id lookup
      {dispatch_id}/
        attempts-{miner_id}.jsonl   # append-on-event: one line per annealer return
                                    # Includes solution_meta scalars + submission ref
                                    # on the iter that submitted (if any).
        metadata-{miner_id}.json    # plain JSON, rewritten on every event:
                                    # aggregate per (dispatch, miner) — n_attempts,
                                    # n_stored, n_submitted, best_energy_seen, ...
        submission.json             # written when controller submits to chain
                                    # (only one per dispatch, by definition).
        solutions/
          {iter:06d}-{nonce8}       # binary file: hex of top-5 packed spins
                                    # Written ONLY when an attempt is
                                    # "stored" or "submitted".

The previous flat date-keyed layout
(``attempts-{miner_id}-{date}.jsonl`` + ``submissions-{date}.jsonl``)
is replaced wholesale. Operators querying historical attempts will
need data migration if running this on an existing log directory.

Query primitives:

  - ``query_by_solution_id(n)`` → ``{submission, attempts}`` —
    looks up dispatch_id via ``submissions_index.jsonl`` then reads
    the per-dispatch dir.
  - ``query_by_dispatch(miner_id, dispatch_id, limit)`` → ``[attempts]``
    — reads the per-dispatch ``attempts-{miner_id}.jsonl`` directly.
  - ``query_stored_solutions(dispatch_id, miner_id=None)`` → list of
    ``{iter, nonce_hex, top_5_solutions_hex, top_5_energies}`` — reads
    files from the per-dispatch ``solutions/`` folder.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def _default_log_dir() -> Path:
    """Resolve the per-dispatch attempts archive root.

    Precedence:
      1. ``QUIP_MINING_ATTEMPTS_DIR`` — explicit override.
      2. ``$QUIP_RUNTIME_DIR/mining_attempts`` — the Docker image exports
         ``QUIP_RUNTIME_DIR=/data/runtime`` (the mounted volume), so
         attempts persist there and are visible to operators / readable
         by the telemetry aggregator instead of being written to the
         container's ephemeral home.
      3. ``~/.quip-miner/mining_attempts`` — bare/local/dev default.

    Resolved once at import. Both the worker (``AttemptLogger`` /
    ``SolutionStore``), the controller snapshot's ``attempts_dir``, and
    the telemetry reader pivot on this value, so they stay consistent as
    long as every process shares the same environment (they do: the
    entrypoint exports ``QUIP_RUNTIME_DIR`` before spawning children).
    """
    explicit = os.environ.get("QUIP_MINING_ATTEMPTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    runtime = os.environ.get("QUIP_RUNTIME_DIR")
    if runtime:
        return Path(runtime).expanduser() / "mining_attempts"
    return Path("~/.quip-miner/mining_attempts").expanduser()


DEFAULT_LOG_DIR: Path = _default_log_dir()


# ----------------------------------------------------------------------
# Filename helpers
# ----------------------------------------------------------------------


def _safe_filename_part(s: str) -> str:
    """Strip filesystem-unsafe chars from a string for filename use."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _now_ns() -> int:
    """Nanosecond-precision UTC timestamp. Iteration-level events are
    sub-microsecond on GPU paths; ms granularity would collapse them
    onto a single bucket and make per-attempt ordering impossible to
    reconstruct."""
    return time.time_ns()


def _dispatch_dir(log_dir: Path, dispatch_id: int) -> Path:
    """Per-dispatch directory: ``{log_dir}/{dispatch_id}/``."""
    return log_dir / str(dispatch_id)


def _solution_filename(iter_num: int, nonce_hex: str) -> str:
    """``{iter:06d}-{nonce_first8}`` — sortable by iter, content-addressed."""
    nonce_short = nonce_hex[:8] if nonce_hex else "0" * 8
    return f"{iter_num:06d}-{nonce_short}"


# ----------------------------------------------------------------------
# Append helper (worker-side files)
# ----------------------------------------------------------------------


class _JsonlAppender:
    """Append-only JSONL file with a tiny in-process lock.

    The lock is per-instance, not per-file — concurrent writers to the
    same file from different processes would race. Each instance is
    owned by exactly one process (the worker for attempts, the
    controller for submissions). Each ``write`` is a single small line
    well under POSIX's ``PIPE_BUF`` atomicity guarantee, so even if
    cross-process writes landed in the same file the line boundaries
    wouldn't tear.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._ensured_dir = False

    def _ensure_dir(self) -> None:
        if not self._ensured_dir:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._ensured_dir = True

    def write(self, record: dict) -> None:
        self._ensure_dir()
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)


# ----------------------------------------------------------------------
# AttemptLogger — per-iteration attempt records (worker-side)
# ----------------------------------------------------------------------


class AttemptLogger:
    """Per-iteration mining attempt logger.

    One instance per worker process; not safe to share across processes
    (the in-process lock won't synchronize). Writes one JSONL line per
    annealer return to ``{base}/{dispatch_id}/attempts-{miner_id}.jsonl``.

    ``record`` is non-throwing — a failed write logs a warning but
    never blocks mining.
    """

    def __init__(
        self,
        miner_id: str,
        log_dir: Path = DEFAULT_LOG_DIR,
        *,
        miner_type: str = "",
    ) -> None:
        self.miner_id = miner_id
        self.miner_type = miner_type
        self.log_dir = log_dir
        self._appender_by_dispatch: dict[int, _JsonlAppender] = {}
        self._metadata_by_dispatch: dict[int, MetadataLogger] = {}

    def _appender(self, dispatch_id: int) -> _JsonlAppender:
        existing = self._appender_by_dispatch.get(dispatch_id)
        if existing is not None:
            return existing
        path = (
            _dispatch_dir(self.log_dir, dispatch_id)
            / f"attempts-{_safe_filename_part(self.miner_id)}.jsonl"
        )
        new = _JsonlAppender(path)
        self._appender_by_dispatch[dispatch_id] = new
        return new

    def _metadata(self, dispatch_id: int) -> "MetadataLogger":
        existing = self._metadata_by_dispatch.get(dispatch_id)
        if existing is not None:
            return existing
        ml = MetadataLogger(
            self.miner_id, dispatch_id,
            log_dir=self.log_dir, miner_type=self.miner_type,
        )
        self._metadata_by_dispatch[dispatch_id] = ml
        return ml

    def record(
        self,
        *,
        dispatch_id: int,
        iter_num: int,
        nonce_hex: str,
        salt_hex: str,
        best_energy_milli: int,
        num_samples: int,
        post_processed: bool,
        stored_as_best: bool,
        num_valid: Optional[int] = None,
        diversity_milli: Optional[int] = None,
        threshold_milli: Optional[int] = None,
        ratchet_threshold_milli: Optional[int] = None,
        result_kind: str = "rejected",
        mining_time_us: Optional[int] = None,
        qpu_access_time_us: Optional[int] = None,
        feeder_ready: Optional[int] = None,
        feeder_drained_count: Optional[int] = None,
        feeder_pop_wait_total_s: Optional[float] = None,
        solution_meta: Optional[dict] = None,
        solution_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write one attempt record + update the dispatch metadata.

        ``result_kind`` is the per-iteration outcome:
          - ``"rejected"`` — iteration didn't beat the ratchet's gate
          - ``"stored"`` — became the new ``stored_best`` candidate
          - ``"submitted"`` — chain threshold crossed; returned for submit
          - ``"error"`` — sampling / post-processing raised

        ``num_valid`` is the count of unique samples meeting the energy
        threshold (snapshot or live decayed). The chain accepts when
        ``num_valid >= min_solutions``.

        ``solution_meta`` is the scalar diagnostic dict produced by
        :func:`shared.quantum_proof_of_work.compute_solution_meta` —
        embedded inline in the JSONL line.

        ``solution_id`` is set only when this iter resulted in a
        submission; lets ``query_by_solution_id`` back-resolve.

        Side effect: metadata-{miner_id}.json gets rewritten with the
        updated aggregate (n_attempts, n_stored, n_submitted,
        best_energy_seen, qpu_time_total_us).
        """
        record = {
            "type": "attempt",
            "ts_ns": _now_ns(),
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
            "dispatch_id": dispatch_id,
            "iter": iter_num,
            "nonce": nonce_hex,
            "salt": salt_hex,
            "best_energy_milli": best_energy_milli,
            "num_samples": num_samples,
            "num_valid": num_valid,
            "diversity_milli": diversity_milli,
            "threshold_milli": threshold_milli,
            "ratchet_threshold_milli": ratchet_threshold_milli,
            "post_processed": post_processed,
            "stored_as_best": stored_as_best,
            "result_kind": result_kind,
            "mining_time_us": mining_time_us,
            "qpu_access_time_us": qpu_access_time_us,
            "feeder_ready": feeder_ready,
            "feeder_drained_count": feeder_drained_count,
            "feeder_pop_wait_total_s": feeder_pop_wait_total_s,
            "solution_meta": solution_meta,
            "solution_id": solution_id,
            "error": error,
        }
        try:
            self._appender(dispatch_id).write(record)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "AttemptLogger.record: write failed: %s", exc,
            )
            return

        # Update aggregate metadata. Failures here also don't block.
        try:
            self._metadata(dispatch_id).update_from_attempt(
                best_energy_milli=best_energy_milli,
                result_kind=result_kind,
                qpu_access_time_us=qpu_access_time_us,
                mining_time_us=mining_time_us,
            )
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "AttemptLogger.metadata update failed: %s", exc,
            )

    def flush(self) -> None:
        """Flush all per-dispatch metadata loggers owned by this worker."""
        for ml in self._metadata_by_dispatch.values():
            try:
                ml.flush()
            except OSError as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "AttemptLogger.flush: %s", exc,
                )


# ----------------------------------------------------------------------
# MetadataLogger — per-dispatch per-miner aggregate (rewritten on update)
# ----------------------------------------------------------------------


class MetadataLogger:
    """Per-dispatch per-miner aggregate JSON file.

    Lives at ``{base}/{dispatch_id}/metadata-{miner_id}.json``. Not a
    JSONL — it's a single JSON object rewritten on every update. Each
    rewrite is via tmp-file + os.replace so concurrent readers never
    see a partial file.
    """

    # Flush the aggregate JSON to disk at most once per this many attempts
    # (plus immediately on stored/submitted/error events and on final
    # flush()). Rewriting on every rejected attempt was the per-iter floor
    # on slow mounted volumes.
    FLUSH_EVERY: int = 25

    def __init__(
        self,
        miner_id: str,
        dispatch_id: int,
        log_dir: Path = DEFAULT_LOG_DIR,
        *,
        miner_type: str = "",
    ) -> None:
        self.miner_id = miner_id
        self.dispatch_id = dispatch_id
        self.log_dir = log_dir
        self.miner_type = miner_type
        self._lock = threading.Lock()
        self._path = (
            _dispatch_dir(log_dir, dispatch_id)
            / f"metadata-{_safe_filename_part(miner_id)}.json"
        )
        self._tmp_path = self._path.with_suffix(".json.tmp")
        # In-memory snapshot of the aggregate. Loaded lazily from disk
        # so restarts pick up where we left off within a dispatch.
        self._state: Optional[dict] = None
        self._pending_since_flush = 0

    def _initial_state(self) -> dict:
        return {
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
            "dispatch_id": self.dispatch_id,
            "n_attempts": 0,
            "n_stored": 0,
            "n_submitted": 0,
            "n_errored": 0,
            "best_energy_milli": None,
            "qpu_time_total_us": 0,
            "mining_time_total_us": 0,
            "first_ts_ns": None,
            "last_ts_ns": None,
            "submission": None,
        }

    def _load_or_init(self) -> dict:
        if self._state is not None:
            return self._state
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                return self._state
            except (OSError, json.JSONDecodeError):
                pass
        self._state = self._initial_state()
        return self._state

    def _write_atomic(self, state: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, separators=(",", ":"), default=str)
        import os
        os.replace(self._tmp_path, self._path)

    def update_from_attempt(
        self,
        *,
        best_energy_milli: int,
        result_kind: str,
        qpu_access_time_us: Optional[int],
        mining_time_us: Optional[int],
    ) -> None:
        with self._lock:
            state = self._load_or_init()
            state["n_attempts"] += 1
            if result_kind == "stored":
                state["n_stored"] += 1
            elif result_kind == "submitted":
                state["n_submitted"] += 1
            elif result_kind == "error":
                state["n_errored"] += 1
            best = state.get("best_energy_milli")
            if best is None or best_energy_milli < best:
                state["best_energy_milli"] = best_energy_milli
            if qpu_access_time_us is not None:
                state["qpu_time_total_us"] = (
                    state.get("qpu_time_total_us", 0) + qpu_access_time_us
                )
            if mining_time_us is not None:
                state["mining_time_total_us"] = (
                    state.get("mining_time_total_us", 0) + mining_time_us
                )
            now = _now_ns()
            if state.get("first_ts_ns") is None:
                state["first_ts_ns"] = now
            state["last_ts_ns"] = now
            self._pending_since_flush += 1
            force = result_kind in ("stored", "submitted", "error")
            if force or self._pending_since_flush >= self.FLUSH_EVERY:
                self._write_atomic(state)
                self._pending_since_flush = 0

    def flush(self) -> None:
        """Write any buffered aggregate state to disk.

        Called at dispatch end so the final counts land even when the last
        batch hadn't reached FLUSH_EVERY. Idempotent; cheap when clean.
        """
        with self._lock:
            if self._state is not None and self._pending_since_flush:
                self._write_atomic(self._state)
                self._pending_since_flush = 0

    def attach_submission(self, submission: dict) -> None:
        """Record that this dispatch was submitted (called by controller)."""
        with self._lock:
            state = self._load_or_init()
            state["submission"] = submission
            self._write_atomic(state)


# ----------------------------------------------------------------------
# SolutionStore — binary archive of top-5 spin configs (worker-side)
# ----------------------------------------------------------------------


class SolutionStore:
    """Top-5 packed-spin archive for stored/submitted attempts.

    Files at ``{base}/{dispatch_id}/solutions/{iter:06d}-{nonce8}``.
    Content is JSON (small, ~6KB per file) carrying
    ``{nonce_hex, top_5_solutions_hex, top_5_energies}``. We use JSON
    not raw bytes because operators need to see the energies alongside
    the spin packing.

    One instance per worker. The per-dispatch dir is created lazily;
    write failures log a warning but never block mining.
    """

    def __init__(
        self, miner_id: str, log_dir: Path = DEFAULT_LOG_DIR,
    ) -> None:
        self.miner_id = miner_id
        self.log_dir = log_dir

    def _path(self, dispatch_id: int, iter_num: int, nonce_hex: str) -> Path:
        return (
            _dispatch_dir(self.log_dir, dispatch_id)
            / "solutions"
            / _solution_filename(iter_num, nonce_hex)
        )

    def record(
        self,
        *,
        dispatch_id: int,
        iter_num: int,
        nonce_hex: str,
        salt_hex: str,
        top_5_solutions_hex: List[str],
        top_5_energies: List[float],
        result_kind: str,
    ) -> None:
        """Write a stored solution file. ``result_kind`` is preserved
        for auditing (so a reader can tell whether this attempt also
        submitted to chain)."""
        path = self._path(dispatch_id, iter_num, nonce_hex)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "miner_id": self.miner_id,
                "dispatch_id": dispatch_id,
                "iter": iter_num,
                "nonce_hex": nonce_hex,
                "salt_hex": salt_hex,
                "result_kind": result_kind,
                "top_5_solutions_hex": top_5_solutions_hex,
                "top_5_energies": top_5_energies,
            }
            with path.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, separators=(",", ":"), default=str)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "SolutionStore.record: write failed: %s", exc,
            )


# ----------------------------------------------------------------------
# SubmissionLogger — controller-side per-dispatch submission record
# ----------------------------------------------------------------------


class SubmissionLogger:
    """Controller-side submission archive.

    One instance per controller process. Owns the monotonic
    ``solution_id`` counter, persisted in ``{base}/next_solution_id``
    so restarts don't reset.

    Writes two artifacts per submission:

    - ``{base}/{dispatch_id}/submission.json`` — the submission record
      (single object, rewritten if called more than once for the same
      dispatch). Also attached to the winner miner's metadata.json.
    - ``{base}/submissions_index.jsonl`` — append-only
      ``{solution_id, dispatch_id, miner_id, ts_ns}`` for
      fast solution_id → dispatch_id lookup.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._counter_path = log_dir / "next_solution_id"
        self._index_path = log_dir / "submissions_index.jsonl"
        self._index_appender = _JsonlAppender(self._index_path)
        self._next_solution_id: int = self._restore_counter()

    def _restore_counter(self) -> int:
        """Read the persisted counter, falling back to scanning the
        index file (then defaulting to 1). The counter file lets
        startup be O(1); the scan is the resilience path for cases
        where the counter file went missing but the index survived."""
        try:
            return int(self._counter_path.read_text().strip()) + 1
        except (OSError, ValueError):
            pass
        highest = 0
        if self._index_path.exists():
            try:
                with self._index_path.open("r") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sid = rec.get("solution_id")
                        if isinstance(sid, int) and sid > highest:
                            highest = sid
            except OSError:
                pass
        return highest + 1

    def _persist_counter(self, used: int) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._counter_path.write_text(str(used))
        except OSError:
            pass

    def assign_id(self) -> int:
        """Reserve the next ``solution_id`` for an imminent submit."""
        with self._lock:
            sid = self._next_solution_id
            self._next_solution_id += 1
            self._persist_counter(sid)
            return sid

    def record(
        self,
        *,
        solution_id: int,
        miner_id: str,
        dispatch_id: int,
        energy_milli: int,
        diversity_milli: int,
        threshold_milli: int,
        last_proof_block_hash_hex: str,
        outcome: str,
        miner_type: str = "",
        extrinsic_hash: Optional[str] = None,
        chain_block_hash: Optional[str] = None,
        chain_block_number: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write the per-dispatch submission record + index entry.

        Also attaches the submission to the winning miner's
        ``metadata-{miner_id}.json`` so dispatch-level queries see the
        whole picture.
        """
        ts = _now_ns()
        record = {
            "type": "submission",
            "ts_ns": ts,
            "solution_id": solution_id,
            "miner_id": miner_id,
            "miner_type": miner_type,
            "dispatch_id": dispatch_id,
            "energy_milli": energy_milli,
            "diversity_milli": diversity_milli,
            "threshold_milli": threshold_milli,
            "last_proof_block_hash": last_proof_block_hash_hex,
            "extrinsic_hash": extrinsic_hash,
            "chain_block_hash": chain_block_hash,
            "chain_block_number": chain_block_number,
            "outcome": outcome,
            "error": error,
        }
        # Per-dispatch submission.json.
        sub_path = _dispatch_dir(self.log_dir, dispatch_id) / "submission.json"
        try:
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            with sub_path.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, separators=(",", ":"), default=str)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "SubmissionLogger.record: write %s failed: %s",
                sub_path, exc,
            )

        # Attach to winning miner's metadata.json (best effort).
        try:
            MetadataLogger(
                miner_id, dispatch_id, log_dir=self.log_dir,
                miner_type=miner_type,
            ).attach_submission(record)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "SubmissionLogger.record: metadata attach failed: %s", exc,
            )

        # Append to global index for solution_id → dispatch_id lookup.
        try:
            self._index_appender.write({
                "solution_id": solution_id,
                "dispatch_id": dispatch_id,
                "miner_id": miner_id,
                "ts_ns": ts,
            })
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "SubmissionLogger.record: index append failed: %s", exc,
            )


# ----------------------------------------------------------------------
# Query
# ----------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _resolve_solution_id(
    solution_id: int, log_dir: Path,
) -> Optional[Tuple[int, str]]:
    """Look up ``(dispatch_id, miner_id)`` for a given solution_id via
    the submissions_index. Returns None if not found."""
    index_path = log_dir / "submissions_index.jsonl"
    for rec in _iter_jsonl(index_path):
        if rec.get("solution_id") == solution_id:
            return (rec.get("dispatch_id"), rec.get("miner_id"))
    return None


def query_by_solution_id(
    solution_id: int,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> Optional[dict]:
    """Return ``{submission, attempts}`` for the given ``solution_id``.

    Resolves the solution_id via the global submissions_index, then
    reads the per-dispatch ``submission.json`` and
    ``attempts-{miner_id}.jsonl`` from the dispatch directory.

    Returns None when solution_id has no index entry (treat as 404).
    """
    resolved = _resolve_solution_id(solution_id, log_dir)
    if resolved is None:
        return None
    dispatch_id, miner_id = resolved
    if dispatch_id is None or miner_id is None:
        return None

    sub_path = _dispatch_dir(log_dir, dispatch_id) / "submission.json"
    submission: Optional[dict] = None
    try:
        with sub_path.open("r", encoding="utf-8") as fh:
            submission = json.load(fh)
    except (OSError, json.JSONDecodeError):
        submission = None

    attempts = list(query_by_dispatch(miner_id, dispatch_id, log_dir=log_dir))
    return {"submission": submission, "attempts": attempts}


def query_by_dispatch(
    miner_id: str,
    dispatch_id: int,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    limit: Optional[int] = None,
) -> List[dict]:
    """Return attempt records for a single (miner, dispatch)."""
    path = (
        _dispatch_dir(log_dir, dispatch_id)
        / f"attempts-{_safe_filename_part(miner_id)}.jsonl"
    )
    out: List[dict] = []
    for rec in _iter_jsonl(path):
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def query_stored_solutions(
    dispatch_id: int,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    miner_id: Optional[str] = None,
) -> List[dict]:
    """List archived top-5 spin configs for stored/submitted attempts.

    Reads the per-dispatch ``solutions/`` folder. Optionally filters
    by miner_id (matched against the record's ``miner_id`` field).
    Returns records sorted by iter ascending.
    """
    sol_dir = _dispatch_dir(log_dir, dispatch_id) / "solutions"
    if not sol_dir.is_dir():
        return []
    out: List[dict] = []
    for path in sorted(sol_dir.iterdir()):
        try:
            with path.open("r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if miner_id is not None and rec.get("miner_id") != miner_id:
            continue
        out.append(rec)
    return out


__all__ = [
    "DEFAULT_LOG_DIR",
    "AttemptLogger",
    "MetadataLogger",
    "SolutionStore",
    "SubmissionLogger",
    "query_by_solution_id",
    "query_by_dispatch",
    "query_stored_solutions",
]
