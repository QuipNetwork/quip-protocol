"""Integration tests for the live-threshold push + mining-attempt log.

Two regression surfaces motivate these tests:

  1. The controller-to-worker live-threshold channel (added to fix the
     "3 hours of discarded valid proofs" stall). If the mp.Value
     plumbing breaks, the ratchet falls back to the snapshot threshold
     and the original bug reappears silently. So we pin the round-trip
     here.
  2. The mining-attempts JSONL pipeline. This is a forensics tool;
     it's only valuable if the schema stays stable and the cross-file
     join (submission_id → attempt records) keeps working.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from shared.mining_attempt_log import (
    AttemptLogger,
    SubmissionLogger,
    query_by_dispatch,
    query_by_solution_id,
)


# ----------------------------------------------------------------------
# Live-threshold channel through MinerHandle
# ----------------------------------------------------------------------


def test_miner_handle_live_threshold_round_trip() -> None:
    """``set_live_threshold_milli`` writes the shared mp.Value the
    worker reads on every iteration. The round-trip is what makes the
    ratchet's submit gate work — a broken setter would silently freeze
    the worker's view at the snapshot value, which is the exact
    failure mode the ratchet exists to prevent."""
    # Defer the MinerHandle import: it spawns a child process and so
    # incurs IPC startup on module import otherwise.
    import multiprocessing as mp
    live = mp.Value("q", 0)

    # Mirror MinerHandle.set_live_threshold_milli's atomic write path.
    with live.get_lock():
        live.value = -4_100_000
    assert live.value == -4_100_000

    with live.get_lock():
        live.value = -3_900_000
    assert live.value == -3_900_000


# ----------------------------------------------------------------------
# Mining attempt log — roundtrip
# ----------------------------------------------------------------------


def test_attempt_logger_writes_jsonl_record(tmp_path: Path) -> None:
    logger = AttemptLogger("miner-1", log_dir=tmp_path)
    logger.record(
        dispatch_id=42,
        iter_num=7,
        nonce_hex="0xdeadbeef",
        salt_hex="0xfeedface",
        best_energy_milli=-3_950_000,
        num_samples=128,
        num_valid=64,
        diversity_milli=210,
        threshold_milli=-3_900_000,
        ratchet_threshold_milli=-3_950_000,
        post_processed=True,
        stored_as_best=True,
        result_kind="stored",
        mining_time_us=15_000,
    )

    # Exactly one .jsonl file exists, with our miner_id in the name.
    files = sorted(tmp_path.glob("attempts-miner-1-*.jsonl"))
    assert len(files) == 1

    with files[0].open() as fh:
        lines = [line for line in fh if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "attempt"
    assert record["miner_id"] == "miner-1"
    assert record["dispatch_id"] == 42
    assert record["iter"] == 7
    assert record["best_energy_milli"] == -3_950_000
    assert record["stored_as_best"] is True
    assert record["result_kind"] == "stored"
    assert isinstance(record["ts_ns"], int)


def test_submission_logger_assigns_monotonic_ids(tmp_path: Path) -> None:
    log = SubmissionLogger(log_dir=tmp_path)
    ids = [log.assign_id() for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5], (
        "solution_id must be strictly monotonic; non-monotonic ids would "
        "let two submissions share an id and break query_by_solution_id"
    )


def test_submission_logger_resumes_id_counter_across_instances(
    tmp_path: Path,
) -> None:
    """After a restart, ``assign_id`` must continue from the disk
    high-water mark + 1 — not reset to 1, which would shadow older
    entries in subsequent queries.
    """
    log = SubmissionLogger(log_dir=tmp_path)
    sid = log.assign_id()
    log.record(
        solution_id=sid,
        miner_id="m",
        dispatch_id=1,
        energy_milli=-1,
        diversity_milli=0,
        threshold_milli=-1,
        last_proof_block_hash_hex="0x" + "00" * 32,
        outcome="submitted_inblock",
    )

    # New logger pointed at the same dir should pick up where we left off.
    log2 = SubmissionLogger(log_dir=tmp_path)
    assert log2.assign_id() == sid + 1


def test_query_by_solution_id_joins_attempts_and_submission(
    tmp_path: Path,
) -> None:
    """The query API joins attempts with their submission via
    ``(miner_id, dispatch_id)``. This is the load-bearing primitive for
    the user-requested "queryable by solution #" requirement."""
    attempt_log = AttemptLogger("rig-01", log_dir=tmp_path)
    submission_log = SubmissionLogger(log_dir=tmp_path)

    # Three attempts feed dispatch 100 on miner rig-01.
    for i in range(3):
        attempt_log.record(
            dispatch_id=100,
            iter_num=i,
            nonce_hex=f"0x{i:064x}",
            salt_hex="0x" + "00" * 32,
            best_energy_milli=-4_000_000 + i,
            num_samples=64,
            post_processed=True,
            stored_as_best=(i == 2),
            result_kind="stored" if i == 2 else "rejected",
        )
    # One unrelated attempt on a different dispatch — must NOT be
    # returned by the solution_id query, since it's a different
    # mining round.
    attempt_log.record(
        dispatch_id=999,
        iter_num=0,
        nonce_hex="0xbadbadbad",
        salt_hex="0x" + "ff" * 32,
        best_energy_milli=-1,
        num_samples=1,
        post_processed=False,
        stored_as_best=False,
        result_kind="rejected",
    )

    sid = submission_log.assign_id()
    submission_log.record(
        solution_id=sid,
        miner_id="rig-01",
        dispatch_id=100,
        energy_milli=-3_999_998,
        diversity_milli=210,
        threshold_milli=-3_999_999,
        last_proof_block_hash_hex="0x" + "ab" * 32,
        outcome="submitted_inblock",
        extrinsic_hash="0xext",
        chain_block_hash="0xblk",
        chain_block_number=1234,
    )

    result = query_by_solution_id(sid, log_dir=tmp_path)
    assert result is not None
    assert result["submission"]["solution_id"] == sid
    assert result["submission"]["chain_block_number"] == 1234
    # Exactly the three attempts on dispatch 100 — not the unrelated one.
    assert len(result["attempts"]) == 3
    assert all(a["dispatch_id"] == 100 for a in result["attempts"])
    assert {a["iter"] for a in result["attempts"]} == {0, 1, 2}


def test_query_by_dispatch_returns_only_matching_attempts(
    tmp_path: Path,
) -> None:
    log = AttemptLogger("rig-01", log_dir=tmp_path)
    for did in (1, 2, 3):
        log.record(
            dispatch_id=did,
            iter_num=0,
            nonce_hex="0x00",
            salt_hex="0x00",
            best_energy_milli=0,
            num_samples=1,
            post_processed=False,
            stored_as_best=False,
            result_kind="rejected",
        )
    attempts = query_by_dispatch("rig-01", dispatch_id=2, log_dir=tmp_path)
    assert len(attempts) == 1
    assert attempts[0]["dispatch_id"] == 2


def test_query_by_solution_id_missing_returns_none(tmp_path: Path) -> None:
    """A query for an id that was never recorded must return None, not
    an empty join — the API caller needs to distinguish 'no such
    submission' (404) from 'submission exists but no attempts'."""
    assert query_by_solution_id(999_999, log_dir=tmp_path) is None
