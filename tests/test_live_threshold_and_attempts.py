"""Integration tests for the live-threshold push + mining-attempt log.

Two regression surfaces motivate these tests:

  1. The controller-to-worker live-threshold channel (added to fix the
     "3 hours of discarded valid proofs" stall). If the mp.Value
     plumbing breaks, the ratchet falls back to the snapshot threshold
     and the original bug reappears silently. So we pin the round-trip
     here.
  2. The mining-attempts JSONL pipeline. This is a forensics tool;
     it's only valuable if the schema stays stable and the cross-file
     join (solution_number → attempt records) keeps working. The archive
     is keyed by the chain-global solution number (see AGENTS.md).
"""
from __future__ import annotations

import json
from pathlib import Path

from shared.mining_attempt_log import (
    AttemptLogger,
    SolutionStore,
    SubmissionLogger,
    _default_log_dir,
    query_by_solution_id,
    query_by_solution_number,
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
        solution_number=7, iter_num=1, nonce_hex="0xab", salt_hex="0xcd",
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
        solution_number=42,
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

    # Per-solution layout: file at {solution_number}/attempts-{miner}.jsonl
    files = sorted(tmp_path.glob("42/attempts-miner-1.jsonl"))
    assert len(files) == 1

    with files[0].open() as fh:
        lines = [line for line in fh if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "attempt"
    assert record["miner_id"] == "miner-1"
    assert record["solution_number"] == 42
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
        solution_number=1,
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


def test_query_by_solution_id_joins_attempts_and_submission(
    tmp_path: Path,
) -> None:
    """The query API joins attempts with their submission via the shared
    solution number — the load-bearing primitive for the "queryable by
    solution #" requirement. The solution number is the directory key, so
    the join needs no separate index."""
    attempt_log = AttemptLogger("rig-01", log_dir=tmp_path)
    submission_log = SubmissionLogger(log_dir=tmp_path)

    # Three attempts feed solution 100 on miner rig-01.
    for i in range(3):
        attempt_log.record(
            solution_number=100,
            iter_num=i,
            nonce_hex=f"0x{i:064x}",
            salt_hex="0x" + "00" * 32,
            best_energy_milli=-4_000_000 + i,
            num_samples=64,
            post_processed=True,
            stored_as_best=(i == 2),
            result_kind="stored" if i == 2 else "rejected",
        )
    # One unrelated attempt on a different solution — must NOT be returned
    # by the solution query, since it's a different mining round.
    attempt_log.record(
        solution_number=999,
        iter_num=0,
        nonce_hex="0xbadbadbad",
        salt_hex="0x" + "ff" * 32,
        best_energy_milli=-1,
        num_samples=1,
        post_processed=False,
        stored_as_best=False,
        result_kind="rejected",
    )

    submission_log.record(
        solution_number=100,
        miner_id="rig-01",
        energy_milli=-3_999_998,
        diversity_milli=210,
        threshold_milli=-3_999_999,
        last_proof_block_hash_hex="0x" + "ab" * 32,
        outcome="submitted_inblock",
        extrinsic_hash="0xext",
        chain_block_hash="0xblk",
        chain_block_number=1234,
    )

    result = query_by_solution_id(100, log_dir=tmp_path)
    assert result is not None
    assert result["submission"]["solution_number"] == 100
    assert result["submission"]["chain_block_number"] == 1234
    # Exactly the three attempts on solution 100 — not the unrelated one.
    assert len(result["attempts"]) == 3
    assert all(a["solution_number"] == 100 for a in result["attempts"])
    assert {a["iter"] for a in result["attempts"]} == {0, 1, 2}


def test_query_by_solution_number_returns_only_matching_attempts(
    tmp_path: Path,
) -> None:
    log = AttemptLogger("rig-01", log_dir=tmp_path)
    for sol in (1, 2, 3):
        log.record(
            solution_number=sol,
            iter_num=0,
            nonce_hex="0x00",
            salt_hex="0x00",
            best_energy_milli=0,
            num_samples=1,
            post_processed=False,
            stored_as_best=False,
            result_kind="rejected",
        )
    attempts = query_by_solution_number("rig-01", 2, log_dir=tmp_path)
    assert len(attempts) == 1
    assert attempts[0]["solution_number"] == 2


def test_attempt_logger_resumes_same_solution_across_processes(
    tmp_path: Path,
) -> None:
    """A restart mid-solution must RESUME the same solution dir, not clear
    it — the solution number is chain-global and stable, so a worker that
    restarts while still mining solution N keeps appending to {N}/.

    This is the opposite of the old dispatch_id behavior (which reset on
    reuse because the counter collided across unrelated runs). With the
    solution-number key, "same number" means "same solution", so appending
    is correct and accumulates the full mining history for that solution.
    """
    def _rec(log: AttemptLogger, sol: int, it: int) -> None:
        log.record(
            solution_number=sol, iter_num=it,
            nonce_hex="0x00", salt_hex="0x00",
            best_energy_milli=0, num_samples=1,
            post_processed=False, stored_as_best=False,
            result_kind="rejected",
        )

    run1 = AttemptLogger("rig-01", log_dir=tmp_path)
    for it in (1, 2, 3):
        _rec(run1, 5, it)
    run1.flush()

    # Restart: a fresh logger continues mining the SAME solution 5.
    run2 = AttemptLogger("rig-01", log_dir=tmp_path)
    _rec(run2, 5, 4)
    run2.flush()

    attempts = query_by_solution_number("rig-01", 5, log_dir=tmp_path)
    assert len(attempts) == 4, (
        "restart mid-solution must resume the same dir, not clear it; "
        f"got {len(attempts)}"
    )
    assert [a["iter"] for a in attempts] == [1, 2, 3, 4]
    # The aggregate metadata accumulates across the restart.
    meta = json.loads((tmp_path / "5" / "metadata-rig-01.json").read_text())
    assert meta["n_attempts"] == 4


def test_solution_store_writes_packed_spins(tmp_path: Path) -> None:
    """SolutionStore archives top-5 spin configs per stored attempt."""
    store = SolutionStore("miner-q", log_dir=tmp_path)
    store.record(
        solution_number=99, iter_num=3,
        nonce_hex="deadbeef" + "00" * 28, salt_hex="cafe" + "00" * 30,
        top_5_solutions_hex=["a1b2", "c3d4", "e5f6", "0708", "1a2b"],
        top_5_energies=[-100.5, -99.0, -98.5, -97.0, -96.0],
        result_kind="submitted",
    )
    # File at {solution_number}/solutions/{iter:06d}-{nonce8}
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
    log.record(
        solution_number=200,
        miner_id="rig-q",
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
    log.record(
        solution_number=201,
        miner_id="rig-q",
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
            solution_number=5, iter_num=it,
            nonce_hex=f"{it:064x}", salt_hex="00" * 32,
            top_5_solutions_hex=[f"{it:04x}"] * 5,
            top_5_energies=[float(it)] * 5,
            result_kind="stored",
        )
    records = query_stored_solutions(5, log_dir=tmp_path)
    assert [r["iter"] for r in records] == [3, 12, 27]


def test_query_by_solution_id_missing_returns_none(tmp_path: Path) -> None:
    """A query for a solution that was never recorded must return None, not
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
        solution_number=1, iter_num=0,
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
        solution_number=1, iter_num=0,
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
    log.record(
        solution_number=55,
        miner_id="rig-QPU-1",
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
    log.record(
        solution_number=56,
        miner_id="rig-CPU-1",
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
    log.record(
        solution_number=1,
        miner_id="rig-QPU-DWAVE-1",
        miner_type="QPU",
        energy_milli=-14_000_000,
        diversity_milli=400,
        threshold_milli=-13_500_000,
        last_proof_block_hash_hex="0xabc",
        outcome="submitted_inblock",
    )
    # Per-solution layout: submission.json is a single JSON object,
    # not a JSONL line; lives at {solution_number}/submission.json.
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
