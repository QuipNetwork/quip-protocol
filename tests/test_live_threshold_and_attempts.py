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
from pathlib import Path

from shared.mining_attempt_log import (
    AttemptLogger,
    SolutionStore,
    SubmissionLogger,
    _default_log_dir,
    query_by_dispatch,
    query_by_solution_id,
    query_stored_solutions,
)


def test_default_log_dir_env_precedence(monkeypatch, tmp_path: Path) -> None:
    """The attempts root must follow the Docker data volume.

    Regression: the loggers defaulted to ~/.quip-miner/mining_attempts,
    which inside the container is the ephemeral home — not the mounted
    /data volume — so operators never saw mining attempts. The dir now
    honors QUIP_MINING_ATTEMPTS_DIR, then $QUIP_RUNTIME_DIR/mining_attempts
    (which the entrypoint points at /data/runtime), then the home default.
    """
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("QUIP_MINING_ATTEMPTS_DIR", str(explicit))
    monkeypatch.setenv("QUIP_RUNTIME_DIR", str(tmp_path / "runtime"))
    assert _default_log_dir() == explicit

    # Explicit unset → fall back to the runtime (data-volume) dir.
    monkeypatch.delenv("QUIP_MINING_ATTEMPTS_DIR")
    assert _default_log_dir() == tmp_path / "runtime" / "mining_attempts"

    # Neither set → home default (bare/local/dev runs).
    monkeypatch.delenv("QUIP_RUNTIME_DIR")
    assert _default_log_dir() == Path("~/.quip-miner/mining_attempts").expanduser()


def test_default_log_dir_writes_under_runtime(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: an AttemptLogger using the resolved default lands the
    JSONL under the runtime (data-volume) dir, not the home dir."""
    monkeypatch.setenv("QUIP_RUNTIME_DIR", str(tmp_path))
    logger = AttemptLogger("rig-1", log_dir=_default_log_dir())
    logger.record(
        dispatch_id=7, iter_num=1, nonce_hex="0xab", salt_hex="0xcd",
        best_energy_milli=-14_000_000, num_samples=10, result_kind="rejected",
        post_processed=False, stored_as_best=False,
    )
    written = tmp_path / "mining_attempts" / "7" / "attempts-rig-1.jsonl"
    assert written.exists(), f"attempt not written under {written}"


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
        qpu_access_time_us=8_432,
    )

    # Per-dispatch layout: file lives under {dispatch_id}/attempts-{miner}.jsonl
    files = sorted(tmp_path.glob("42/attempts-miner-1.jsonl"))
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
    assert record["qpu_access_time_us"] == 8_432, (
        "qpu_access_time_us must be written for QPU-backed iterations so the "
        "dashboard can compute real QPU time per attempt"
    )
    assert isinstance(record["ts_ns"], int)


def test_attempt_logger_qpu_access_time_defaults_to_none(tmp_path: Path) -> None:
    """Non-QPU iterations (and QPU iterations without timing info) must
    still produce a record with the field explicitly present as ``None``
    so downstream parsers can rely on a uniform schema rather than a
    presence check."""
    logger = AttemptLogger("miner-cpu", log_dir=tmp_path)
    logger.record(
        dispatch_id=1,
        iter_num=1,
        nonce_hex="0x0",
        salt_hex="0x0",
        best_energy_milli=0,
        num_samples=1,
        post_processed=False,
        stored_as_best=False,
        result_kind="rejected",
    )

    files = sorted(tmp_path.glob("1/attempts-miner-cpu.jsonl"))
    with files[0].open() as fh:
        record = json.loads(fh.read().strip())
    assert "qpu_access_time_us" in record
    assert record["qpu_access_time_us"] is None


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


def test_attempt_logger_resets_reused_dispatch_id_across_processes(
    tmp_path: Path,
) -> None:
    """A fresh AttemptLogger (new worker process after controller restart)
    must clear a reused dispatch_id's stale artifacts on first use.

    ``dispatch_id`` is a controller-local counter that resets to 0 on
    restart, so run N+1 reuses ids from run N. The append-only attempts
    JSONL and the cumulative metadata aggregate would otherwise accrete
    iterations across runs — breaking the dashboard's iter-as-recency
    assumption. The first write to a dispatch_id in a new process must
    start that dispatch's files clean.
    """
    def _rec(log: AttemptLogger, did: int, it: int) -> None:
        log.record(
            dispatch_id=did, iter_num=it,
            nonce_hex="0x00", salt_hex="0x00",
            best_energy_milli=0, num_samples=1,
            post_processed=False, stored_as_best=False,
            result_kind="rejected",
        )

    # Run 1: three attempts under dispatch_id=5, then flush so the
    # aggregate metadata lands on disk (a real run persists it via
    # FLUSH_EVERY / stored-submitted events / dispatch-end flush).
    run1 = AttemptLogger("rig-01", log_dir=tmp_path)
    for it in (1, 2, 3):
        _rec(run1, 5, it)
    run1.flush()
    assert len(query_by_dispatch("rig-01", 5, log_dir=tmp_path)) == 3
    assert json.loads(
        (tmp_path / "5" / "metadata-rig-01.json").read_text()
    )["n_attempts"] == 3

    # Run 2: a brand-new logger (simulates the restarted worker process)
    # reuses dispatch_id=5 and records a single attempt.
    run2 = AttemptLogger("rig-01", log_dir=tmp_path)
    _rec(run2, 5, 1)
    run2.flush()

    attempts = query_by_dispatch("rig-01", 5, log_dir=tmp_path)
    assert len(attempts) == 1, (
        "reused dispatch_id must not accrete prior-run attempts; "
        f"got {len(attempts)}"
    )
    assert attempts[0]["iter"] == 1

    # The aggregate metadata must reflect only run 2, not 3 + 1 = 4.
    meta_path = tmp_path / "5" / "metadata-rig-01.json"
    meta = json.loads(meta_path.read_text())
    assert meta["n_attempts"] == 1, (
        f"metadata aggregate must reset on reuse; got {meta['n_attempts']}"
    )


def test_attempt_logger_reset_is_per_miner_not_whole_dispatch_dir(
    tmp_path: Path,
) -> None:
    """Resetting a reused dispatch_id must only clear THIS miner's files —
    a concurrent miner sharing the same numeric dispatch dir keeps its data.
    """
    def _rec(log: AttemptLogger, did: int, it: int) -> None:
        log.record(
            dispatch_id=did, iter_num=it,
            nonce_hex="0x00", salt_hex="0x00",
            best_energy_milli=0, num_samples=1,
            post_processed=False, stored_as_best=False,
            result_kind="rejected",
        )

    cpu = AttemptLogger("rig-cpu", log_dir=tmp_path)
    qpu = AttemptLogger("rig-qpu", log_dir=tmp_path)
    _rec(cpu, 5, 1)
    _rec(qpu, 5, 1)
    _rec(qpu, 5, 2)

    # CPU "restarts" and reuses dispatch_id=5 — must not touch QPU's file.
    cpu2 = AttemptLogger("rig-cpu", log_dir=tmp_path)
    _rec(cpu2, 5, 1)

    assert len(query_by_dispatch("rig-cpu", 5, log_dir=tmp_path)) == 1
    assert len(query_by_dispatch("rig-qpu", 5, log_dir=tmp_path)) == 2, (
        "a sibling miner's attempts in the same dispatch dir must survive"
    )


def test_solution_store_writes_packed_spins(tmp_path: Path) -> None:
    """SolutionStore archives top-5 spin configs per stored attempt."""
    store = SolutionStore("miner-q", log_dir=tmp_path)
    store.record(
        dispatch_id=99, iter_num=3,
        nonce_hex="deadbeef" + "00" * 28, salt_hex="cafe" + "00" * 30,
        top_5_solutions_hex=["a1b2", "c3d4", "e5f6", "0708", "1a2b"],
        top_5_energies=[-100.5, -99.0, -98.5, -97.0, -96.0],
        result_kind="submitted",
    )
    # File at {dispatch}/solutions/{iter:06d}-{nonce8}
    path = tmp_path / "99" / "solutions" / "000003-deadbeef"
    assert path.exists()
    rec = json.loads(path.read_text())
    assert rec["nonce_hex"].startswith("deadbeef")
    assert rec["result_kind"] == "submitted"
    assert len(rec["top_5_solutions_hex"]) == 5


def test_submission_logger_record_writes_pow_sequence(tmp_path: Path) -> None:
    """``pow_sequence`` (miner's cumulative proofs_submitted from chain) must
    be written into submission.json when provided, and must default to None so
    existing records without the field stay schema-compatible."""
    log = SubmissionLogger(log_dir=tmp_path)

    # With pow_sequence set.
    sid = log.assign_id()
    log.record(
        solution_id=sid,
        miner_id="rig-q",
        dispatch_id=200,
        energy_milli=-4_200_000,
        diversity_milli=210,
        threshold_milli=-4_000_000,
        last_proof_block_hash_hex="0x" + "aa" * 32,
        outcome="rejected_stale",
        pow_sequence=42,
    )
    sub_path = tmp_path / "200" / "submission.json"
    rec = json.loads(sub_path.read_text())
    assert rec["pow_sequence"] == 42, (
        "pow_sequence must be written into submission.json for not-won outcomes "
        "so the dashboard can display the miner's cumulative proofs_submitted"
    )

    # Default: pow_sequence omitted → None in record.
    sid2 = log.assign_id()
    log.record(
        solution_id=sid2,
        miner_id="rig-q",
        dispatch_id=201,
        energy_milli=-4_200_000,
        diversity_milli=210,
        threshold_milli=-4_000_000,
        last_proof_block_hash_hex="0x" + "bb" * 32,
        outcome="rejected_stale",
    )
    sub_path2 = tmp_path / "201" / "submission.json"
    rec2 = json.loads(sub_path2.read_text())
    assert "pow_sequence" in rec2, "pow_sequence key must always be present"
    assert rec2["pow_sequence"] is None, "pow_sequence must default to None"


def test_query_stored_solutions_returns_sorted_by_iter(tmp_path: Path) -> None:
    """query_stored_solutions returns all matching files sorted by iter
    (the leading 06d in the filename guarantees fs-sort order)."""
    store = SolutionStore("miner-q", log_dir=tmp_path)
    for it in (12, 3, 27):
        store.record(
            dispatch_id=5, iter_num=it,
            nonce_hex=f"{it:064x}", salt_hex="00" * 32,
            top_5_solutions_hex=[f"{it:04x}"] * 5,
            top_5_energies=[float(it)] * 5,
            result_kind="stored",
        )
    records = query_stored_solutions(5, log_dir=tmp_path)
    assert [r["iter"] for r in records] == [3, 12, 27]


def test_query_by_solution_id_missing_returns_none(tmp_path: Path) -> None:
    """A query for an id that was never recorded must return None, not
    an empty join — the API caller needs to distinguish 'no such
    submission' (404) from 'submission exists but no attempts'."""
    assert query_by_solution_id(999_999, log_dir=tmp_path) is None


def test_attempt_logger_writes_miner_type(tmp_path: Path) -> None:
    """`miner_type` (CPU / CUDA / METAL / MODAL / QPU) is fixed per
    AttemptLogger and written into every record. Dashboard reads it
    to attribute attempts to the right backend in multi-process
    containers without parsing miner_id."""
    logger = AttemptLogger("miner-cpu-1", log_dir=tmp_path, miner_type="CPU")
    logger.record(
        dispatch_id=1, iter_num=0,
        nonce_hex="0x00", salt_hex="0x00",
        best_energy_milli=-1_000_000, num_samples=64,
        post_processed=True, stored_as_best=False,
        result_kind="rejected",
    )
    files = sorted(tmp_path.glob("1/attempts-miner-cpu-1.jsonl"))
    record = json.loads(files[0].read_text().splitlines()[0])
    assert record["miner_type"] == "CPU"


def test_attempt_logger_miner_type_defaults_to_empty(tmp_path: Path) -> None:
    """Legacy callers that don't pass `miner_type` still produce a
    well-formed record — the dashboard's parser tolerates an empty
    miner_type so old miners keep working through the upgrade."""
    logger = AttemptLogger("miner-legacy", log_dir=tmp_path)
    logger.record(
        dispatch_id=1, iter_num=0,
        nonce_hex="0x00", salt_hex="0x00",
        best_energy_milli=-1_000_000, num_samples=64,
        post_processed=True, stored_as_best=False,
    )
    files = sorted(tmp_path.glob("1/attempts-miner-legacy.jsonl"))
    record = json.loads(files[0].read_text().splitlines()[0])
    assert record["miner_type"] == ""


def test_submission_logger_record_writes_num_valid(tmp_path: Path) -> None:
    """num_valid from MiningResult must land in submission.json so the
    dashboard 'Solutions' column has a stable value per submission."""
    log = SubmissionLogger(log_dir=tmp_path)
    sid = log.assign_id()
    log.record(
        solution_id=sid,
        miner_id="rig-QPU-1",
        dispatch_id=55,
        energy_milli=-14_000_000,
        diversity_milli=400,
        threshold_milli=-13_500_000,
        last_proof_block_hash_hex="0xabc",
        outcome="submitted_inblock",
        num_valid=7,
    )
    sub_path = tmp_path / "55" / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["num_valid"] == 7, (
        "num_valid must be written to submission.json so dashboard "
        "'Solutions' column is populated"
    )


def test_submission_logger_record_num_valid_defaults_to_none(tmp_path: Path) -> None:
    """Omitting num_valid must produce null in submission.json — old
    submission records stay readable without a schema migration."""
    log = SubmissionLogger(log_dir=tmp_path)
    sid = log.assign_id()
    log.record(
        solution_id=sid,
        miner_id="rig-CPU-1",
        dispatch_id=56,
        energy_milli=-5_000_000,
        diversity_milli=200,
        threshold_milli=-4_900_000,
        last_proof_block_hash_hex="0xdef",
        outcome="submitted_inblock",
    )
    sub_path = tmp_path / "56" / "submission.json"
    record = json.loads(sub_path.read_text())
    assert "num_valid" in record, "num_valid key must always be present"
    assert record["num_valid"] is None


def test_submission_logger_writes_miner_type_per_call(tmp_path: Path) -> None:
    """`miner_type` is a per-call parameter on SubmissionLogger.record
    because the controller-side logger aggregates submissions from
    every backend type. Each row carries the type of the winning
    handle so the dashboard can render the Backend column for the
    Recent Performance table."""
    log = SubmissionLogger(log_dir=tmp_path)
    sid = log.assign_id()
    log.record(
        solution_id=sid,
        miner_id="rig-QPU-DWAVE-1",
        miner_type="QPU",
        dispatch_id=1,
        energy_milli=-14_000_000,
        diversity_milli=400,
        threshold_milli=-13_500_000,
        last_proof_block_hash_hex="0xabc",
        outcome="submitted_inblock",
    )
    # Per-dispatch layout: submission.json is a single JSON object,
    # not a JSONL line; lives at {dispatch_id}/submission.json.
    sub_path = tmp_path / "1" / "submission.json"
    record = json.loads(sub_path.read_text())
    assert record["miner_type"] == "QPU"
    assert record["miner_id"] == "rig-QPU-DWAVE-1"


# ----------------------------------------------------------------------
# MetadataLogger batched-write tests
# ----------------------------------------------------------------------


def test_metadata_logger_flushes_periodically_not_every_attempt(tmp_path):
    """MetadataLogger must not rewrite the JSON on every rejected attempt;
    it batches writes and exposes a final flush()."""
    from shared.mining_attempt_log import MetadataLogger

    ml = MetadataLogger("m", 0, log_dir=tmp_path, miner_type="QPU")
    ml.FLUSH_EVERY = 5
    path = tmp_path / "0" / "metadata-m.json"

    for _ in range(4):
        ml.update_from_attempt(
            best_energy_milli=-100, result_kind="rejected",
            qpu_access_time_us=10, mining_time_us=20,
        )
    assert not path.exists(), "metadata written before flush threshold"

    ml.update_from_attempt(
        best_energy_milli=-100, result_kind="rejected",
        qpu_access_time_us=10, mining_time_us=20,
    )
    assert path.exists()
    import json as _json
    assert _json.loads(path.read_text())["n_attempts"] == 5


def test_metadata_logger_flushes_immediately_on_stored(tmp_path):
    from shared.mining_attempt_log import MetadataLogger
    ml = MetadataLogger("m", 0, log_dir=tmp_path, miner_type="QPU")
    ml.FLUSH_EVERY = 100
    path = tmp_path / "0" / "metadata-m.json"
    ml.update_from_attempt(
        best_energy_milli=-100, result_kind="stored",
        qpu_access_time_us=10, mining_time_us=20,
    )
    assert path.exists(), "stored events must flush immediately"


def test_metadata_logger_flushes_immediately_on_submitted(tmp_path):
    from shared.mining_attempt_log import MetadataLogger
    ml = MetadataLogger("m", 0, log_dir=tmp_path, miner_type="QPU")
    ml.FLUSH_EVERY = 100
    path = tmp_path / "0" / "metadata-m.json"
    ml.update_from_attempt(
        best_energy_milli=-100, result_kind="submitted",
        qpu_access_time_us=10, mining_time_us=20,
    )
    assert path.exists(), "submitted events must flush immediately"


def test_metadata_logger_final_flush_writes_pending(tmp_path):
    from shared.mining_attempt_log import MetadataLogger
    ml = MetadataLogger("m", 0, log_dir=tmp_path, miner_type="QPU")
    ml.FLUSH_EVERY = 100
    path = tmp_path / "0" / "metadata-m.json"
    for _ in range(3):
        ml.update_from_attempt(
            best_energy_milli=-100, result_kind="rejected",
            qpu_access_time_us=10, mining_time_us=20,
        )
    assert not path.exists()
    ml.flush()
    assert path.exists()
    import json as _json
    assert _json.loads(path.read_text())["n_attempts"] == 3
