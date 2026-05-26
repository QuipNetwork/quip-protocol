"""Append-only JSONL log for mining attempts and submissions.

Two parallel write paths feed the same query API:

  - **Attempts** (worker process, one per mining iteration). Tagged
    with ``(miner_id, dispatch_id, iter_num)`` and the per-iteration
    metrics ``evaluate_sampleset`` already produces. Written from the
    child mining-worker process, so each worker owns its own file —
    no inter-process locking needed.
  - **Submissions** (controller process, one per ``submit_proof``
    extrinsic). Tagged with a monotonic ``solution_id`` the controller
    assigns at submit time. Carries ``(miner_id, dispatch_id)`` so the
    query layer can correlate back to the attempts that fed it.

Files live under ``~/.quip-miner/mining_attempts/`` and rotate by date.
Per-worker files use ``attempts-{miner_id}-{YYYY-MM-DD}.jsonl``;
controller files use ``submissions-{YYYY-MM-DD}.jsonl``. The naming is
deliberately readable rather than compact — operators dig through
these by hand when triaging a missed-reward complaint.

Query primitives (consumed by the telemetry API in
``shared.telemetry_api``):

  - ``query_by_solution_id(n)`` → ``{submission, attempts}`` view
  - ``query_by_dispatch(miner_id, dispatch_id, limit)`` → just the
    attempts that fed that dispatch

"Solution #" is the controller's monotonic counter, not the chain's
winning block number — many submissions never make it to a win (stale
context, sibling raced us, chain rejected). Tracking attempts against
the locally-assigned id lets you ask "what did we try when we
submitted #42?" even if #42 didn't win.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


DEFAULT_LOG_DIR: Path = Path("~/.quip-miner/mining_attempts").expanduser()


def _today_utc() -> str:
    """``YYYY-MM-DD`` in UTC. Matches how operators read these files
    across timezones — dispatching at 23:55 local vs. 00:05 local
    shouldn't fragment a debugging session across two files for
    arbitrary reasons."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_filename_part(s: str) -> str:
    """Strip filesystem-unsafe chars from miner_id for the filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------


class _JsonlAppender:
    """Append-only JSONL file with a tiny in-process lock.

    The lock is per-instance, not per-file — concurrent writers to the
    same file from different processes would race. In practice each
    instance is owned by exactly one process (the worker for attempts,
    the controller for submissions) so this is sufficient. Each
    ``write`` is a single small line, well under POSIX's
    ``PIPE_BUF`` atomicity guarantee, so even if cross-process writes
    landed in the same file the line boundaries wouldn't tear.
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


class AttemptLogger:
    """Worker-side per-iteration logger.

    Constructed lazily in the worker process (so each ``MinerHandle``
    has its own file). ``record`` is non-throwing — a failed write
    logs a warning but never blocks mining. The file path is chosen
    once at construction and reused; rotation across UTC midnight
    creates a new file on the next call.
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
        self._appender_by_date: dict[str, _JsonlAppender] = {}

    def _appender(self) -> _JsonlAppender:
        date = _today_utc()
        existing = self._appender_by_date.get(date)
        if existing is not None:
            return existing
        path = self.log_dir / (
            f"attempts-{_safe_filename_part(self.miner_id)}-{date}.jsonl"
        )
        new = _JsonlAppender(path)
        self._appender_by_date[date] = new
        return new

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
        num_solutions_meeting_target: Optional[int] = None,
        diversity_milli: Optional[int] = None,
        threshold_milli: Optional[int] = None,
        ratchet_threshold_milli: Optional[int] = None,
        result_kind: str = "rejected",
        mining_time_us: Optional[int] = None,
        qpu_access_time_us: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write one attempt record.

        ``result_kind`` is the per-iteration outcome:
          - ``"rejected"`` — iteration didn't beat the ratchet's gate
          - ``"stored"`` — became the new ``stored_best`` candidate
          - ``"submitted"`` — chain threshold crossed; returned for submit
          - ``"error"`` — sampling / post-processing raised

        ``qpu_access_time_us`` is D-Wave's ``qpu_programming_time +
        qpu_sampling_time`` (microseconds) for this iteration's sampleset
        when a QPU backend produced it. ``None`` for CPU/CUDA/etc. backends
        and for QPU iterations whose sampleset arrived without timing info.
        """
        record = {
            "type": "attempt",
            "ts_ns": _now_ns(),
            "miner_id": self.miner_id,
            # `miner_type` mirrors `MinerHandle.miner_type` (CPU / CUDA /
            # METAL / MODAL / QPU). Lets the indexer + dashboard show
            # which backend produced an attempt without parsing miner_id.
            # Empty string for legacy callers that didn't pass it; the
            # parser tolerates both.
            "miner_type": self.miner_type,
            "dispatch_id": dispatch_id,
            "iter": iter_num,
            "nonce": nonce_hex,
            "salt": salt_hex,
            "best_energy_milli": best_energy_milli,
            "num_samples": num_samples,
            "num_valid": num_valid,
            "num_solutions_meeting_target": num_solutions_meeting_target,
            "diversity_milli": diversity_milli,
            "threshold_milli": threshold_milli,
            "ratchet_threshold_milli": ratchet_threshold_milli,
            "post_processed": post_processed,
            "stored_as_best": stored_as_best,
            "result_kind": result_kind,
            "mining_time_us": mining_time_us,
            "qpu_access_time_us": qpu_access_time_us,
            "error": error,
        }
        try:
            self._appender().write(record)
        except OSError as exc:
            # Don't let a disk-full / readonly fs break the mining
            # loop. Surface as a warning so operators notice.
            import logging
            logging.getLogger(__name__).warning(
                "AttemptLogger.record: write failed: %s", exc,
            )


class SubmissionLogger:
    """Controller-side per-submission logger.

    One instance per controller. Owns the monotonic ``solution_id``
    counter — callers don't pass an id, they receive one back.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        self.log_dir = log_dir
        self._appender_by_date: dict[str, _JsonlAppender] = {}
        self._lock = threading.Lock()
        self._next_solution_id: int = self._scan_for_max_solution_id() + 1

    def _scan_for_max_solution_id(self) -> int:
        """Recover the high-water mark from existing files on disk.

        The ``solution_id`` must be monotonic across process restarts
        too — a restart should not reset to 1 and shadow older
        entries. Scan the most recent files for the highest id and
        resume from there + 1.
        """
        if not self.log_dir.exists():
            return 0
        highest = 0
        for path in sorted(self.log_dir.glob("submissions-*.jsonl"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sol = record.get("solution_id")
                        if isinstance(sol, int) and sol > highest:
                            highest = sol
                # One file is enough; older files can only contain
                # smaller ids by definition.
                if highest > 0:
                    break
            except OSError:
                continue
        return highest

    def _appender(self) -> _JsonlAppender:
        date = _today_utc()
        existing = self._appender_by_date.get(date)
        if existing is not None:
            return existing
        path = self.log_dir / f"submissions-{date}.jsonl"
        new = _JsonlAppender(path)
        self._appender_by_date[date] = new
        return new

    def assign_id(self) -> int:
        """Reserve the next ``solution_id`` for an imminent submit."""
        with self._lock:
            sid = self._next_solution_id
            self._next_solution_id += 1
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
        """Write one submission record.

        ``outcome`` is one of:
          - ``"submitted_inblock"`` / ``"submitted_finalized"`` —
            chain saw it (the storming-prevention check passes still
            log under these)
          - ``"rejected_stale"`` — controller dropped before submit
          - ``"rejected_duplicate"`` — sibling already won this round
          - ``"chain_error"`` — extrinsic failed (see ``error``)
        """
        record = {
            "type": "submission",
            "ts_ns": _now_ns(),
            "solution_id": solution_id,
            "miner_id": miner_id,
            # `miner_type` of the WINNING backend (which MinerHandle
            # produced the submitted result). Lets dashboards report
            # which backend type cleared the chain target without
            # parsing miner_id. Empty string for legacy rows; the
            # parser tolerates both.
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
        try:
            self._appender().write(record)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "SubmissionLogger.record: write failed: %s", exc,
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


def query_by_solution_id(
    solution_id: int,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> Optional[dict]:
    """Return ``{submission, attempts}`` for the given ``solution_id``.

    Linear scan over submission files (descending by date) until we
    find the submission. Then scan the relevant per-miner attempt file
    for entries matching the submission's ``(miner_id, dispatch_id)``.

    Returns ``None`` if the solution_id was never assigned.
    """
    if not log_dir.exists():
        return None
    submission: Optional[dict] = None
    for path in sorted(log_dir.glob("submissions-*.jsonl"), reverse=True):
        for record in _iter_jsonl(path):
            if record.get("solution_id") == solution_id:
                submission = record
                break
        if submission is not None:
            break
    if submission is None:
        return None
    attempts = list(
        _query_attempts_for(
            miner_id=submission["miner_id"],
            dispatch_id=submission["dispatch_id"],
            log_dir=log_dir,
        )
    )
    return {"submission": submission, "attempts": attempts}


def query_by_dispatch(
    miner_id: str,
    dispatch_id: int,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    limit: int = 1000,
) -> List[dict]:
    """Return attempt records for a given ``(miner_id, dispatch_id)``."""
    attempts = list(_query_attempts_for(
        miner_id=miner_id, dispatch_id=dispatch_id, log_dir=log_dir,
    ))
    return attempts[:limit]


def _query_attempts_for(
    *,
    miner_id: str,
    dispatch_id: int,
    log_dir: Path,
) -> Iterable[dict]:
    if not log_dir.exists():
        return
    safe_miner = _safe_filename_part(miner_id)
    pattern = f"attempts-{safe_miner}-*.jsonl"
    # Search most recent file first; older dispatches naturally live
    # in older files, but the typical query is for a recent one.
    for path in sorted(log_dir.glob(pattern), reverse=True):
        for record in _iter_jsonl(path):
            if (
                record.get("miner_id") == miner_id
                and record.get("dispatch_id") == dispatch_id
            ):
                yield record


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _now_ns() -> int:
    """Nanosecond-precision UTC timestamp. ``ts_ns`` not ``ts`` because
    iteration-level events are sub-microsecond on GPU paths; using ms
    would collapse them onto a single bucket and make per-attempt
    ordering impossible to reconstruct."""
    return time.time_ns()


__all__ = [
    "DEFAULT_LOG_DIR",
    "AttemptLogger",
    "SubmissionLogger",
    "query_by_solution_id",
    "query_by_dispatch",
]
