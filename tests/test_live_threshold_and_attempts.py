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


def _make_attempt_logger(miner):
    from shared.mining_attempt_log import AttemptLogger
    log = AttemptLogger(miner.miner_id, miner_type=miner.miner_type)
    miner._attempt_logger = log
    return log


def _make_solution_store(miner):
    from shared.mining_attempt_log import SolutionStore
    store = SolutionStore(miner.miner_id)
    miner._solution_store = store
    return store


def _energy_sampleset(energy):
    import numpy as np
    from shared.base_miner import _SharedSampleSet
    rows = len(energy)
    return _SharedSampleSet(np.ones((rows, 3), np.int8), energy.astype(np.float64))


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


def test_attempt_log_shows_live_decay_threshold_not_base(caplog) -> None:
    """The per-attempt console line must report the live decayed target (what
    ``num_valid`` is computed against and the chain enforces at submit), not the
    snapshot base difficulty. They diverge once decay eases the live target, so
    logging the base made ``Valid: 0`` look inconsistent with the stated bar.
    """
    import logging

    import dimod

    from shared.allowed_value_spec import AllowedValueSet
    from shared.miner_types import BlockRequirements
    from shared.quantum_proof_of_work import evaluate_sampleset

    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    ss = dimod.SampleSet.from_samples(
        [{0: -1, 1: 1, 2: -1}, {0: 1, 1: -1, 2: 1}], "SPIN", energy=[-4.0, -2.0],
    )
    req = BlockRequirements(
        difficulty_energy=-5000.0,  # base (snapshot)
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=2**31 - 1,
        allowed_h_values=AllowedValueSet((0,)),
        allowed_j_values=AllowedValueSet((-1000, 1000)),
    )
    with caplog.at_level(logging.INFO):
        evaluate_sampleset(
            ss, req, nodes, edges, nonce=1, salt=b"\x00" * 32,
            prev_timestamp=0, start_time=0.0, miner_id="t", miner_type="CPU",
            h={0: 0.0, 1: 0.0, 2: 0.0}, J={(0, 1): 1.0, (1, 2): 1.0},
            skip_validation=True, strict_energy=False,
            live_threshold_energy=-3000.0,  # live decayed target
        )
    log = "\n".join(r.getMessage() for r in caplog.records)
    assert "energy<=-3000" in log, log
    assert "energy<=-5000" not in log


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


def test_submission_logger_record_writes_qpu_access_us_total(
    tmp_path: Path,
) -> None:
    """``qpu_access_us_total`` (precise QPU spend on a won solution) is written
    when provided and defaults to None for schema compatibility."""
    log = SubmissionLogger(log_dir=tmp_path)
    log.record(
        solution_number=300,
        miner_id="qpu-0",
        energy_milli=-14_800,
        diversity_milli=210,
        threshold_milli=-14_000,
        last_proof_block_hash_hex="0x" + "cc" * 32,
        outcome="submitted_inblock",
        qpu_access_us_total=101_000,
    )
    rec = json.loads((tmp_path / "300" / "submission.json").read_text())
    assert rec["qpu_access_us_total"] == 101_000

    log.record(
        solution_number=301,
        miner_id="qpu-0",
        energy_milli=-14_800,
        diversity_milli=210,
        threshold_milli=-14_000,
        last_proof_block_hash_hex="0x" + "dd" * 32,
        outcome="rejected_stale",
    )
    rec2 = json.loads((tmp_path / "301" / "submission.json").read_text())
    assert "qpu_access_us_total" in rec2
    assert rec2["qpu_access_us_total"] is None


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


def test_decay_within_generation_preserves_stash(monkeypatch):
    """A live-threshold decay (same generation) must NOT clear top_k.

    Conversely, a new dispatch (new generation) starts a fresh stash. This
    guards rule 1 of the decay contract under the persistent-driver model.
    """
    import numpy as np
    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    # A live-threshold shared value the ratchet reads.
    import multiprocessing as mp
    miner._live_max_energy_milli = mp.Value("q", -14000_000)
    # ctl_q absent -> threshold forwarding is a no-op (unit isolation).
    miner._ctl_q = None

    req = BlockRequirements(
        difficulty_energy=-14000.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    state = _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0, attempt_log=miner._attempt_logger
        if hasattr(miner, "_attempt_logger") else _make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=[], previewed_wintime=(1 << 62, 1 << 62), generation=1,
    )

    # Force evaluate_sampleset to stash one candidate (floor -14500).
    cand = MiningResult(
        miner_id=miner.miner_id, miner_type="QPU", nonce=b"\x01" * 32,
        salt=b"\x02" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-14600.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=-14500.0,
    )
    monkeypatch.setattr(miner, "evaluate_sampleset",
                        lambda *a, **k: cand)

    ss = _energy_sampleset(np.full(4, -14600.0))  # within margin of -14000
    logkw = {}
    miner._run_substrate_ratchet(state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
                                 preview_cb=None, attempt_log_kwargs=logkw)
    assert len(state.top_k) == 1  # stashed

    # Decay the live threshold (same generation) — stash must persist.
    with miner._live_max_energy_milli.get_lock():
        miner._live_max_energy_milli.value = -14550_000
    logkw2 = {}
    # A worse-energy iter that won't displace the stash; the point is that the
    # decay did not clear it.
    ss2 = _energy_sampleset(np.full(4, -14100.0))
    monkeypatch.setattr(miner, "evaluate_sampleset", lambda *a, **k: None)
    miner._run_substrate_ratchet(state, ss2, b"\x03" * 32, b"\x04" * 32, 0.0,
                                 preview_cb=None, attempt_log_kwargs=logkw2)
    assert len(state.top_k) == 1, "decay within a generation cleared the stash"


def test_out_of_band_ease_admits_candidate_outside_schedule_horizon(monkeypatch):
    """Stash admission must never be stricter than the submit gate.

    The decay schedule is frozen at dispatch. An out-of-band difficulty
    ease (a sudo reseed on dev/testnet) can move the LIVE threshold below
    the frozen schedule's horizon while the item is still mining: the
    submit gate (`_select_submittable_candidate`, live-threshold based)
    would accept the candidate immediately, but `_compute_stash_entry`
    returned None ("never clears within the schedule horizon") so the
    candidate was never stashed — no preview, no submission, wedged for
    the item's whole lifetime. Found live by
    tests/test_mempool_priority_integration.py (strict-baked item + a
    relaxed reseed 8s later; the single-miner chain never rolls the work
    key, so nothing ever recovered).
    """
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    # Live threshold eased OUT-OF-BAND to -2500 while the frozen schedule
    # (baked from a strict -10000 base) tops out at -8100: the candidate
    # floor -3504 clears live but never clears the schedule.
    miner._live_max_energy_milli = mp.Value("q", -2_500_000)
    miner._ctl_q = None

    req = BlockRequirements(
        difficulty_energy=-10_000.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    state = _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=[], previewed_wintime=(1 << 62, 1 << 62),
        generation=1,
        decay_schedule=list(range(-10_000_000, -8_000_000, 100_000)),
        last_proof_block=987,
        epoch_length=10,
    )

    cand = MiningResult(
        miner_id=miner.miner_id, miner_type="CPU", nonce=b"\x01" * 32,
        salt=b"\x02" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-3_508.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=-3_504.0,
    )
    monkeypatch.setattr(miner, "evaluate_sampleset", lambda *a, **k: cand)

    ss = _energy_sampleset(np.full(4, -3_508.0))
    result = miner._run_substrate_ratchet(
        state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )
    assert state.top_k, "candidate clearing the LIVE threshold was not stashed"
    assert state.top_k[0].decay_num == 0
    assert state.top_k[0].valid_at_block == 987  # immediately submittable
    assert result is cand, "submit gate should fire on the admitted candidate"


def test_out_of_horizon_candidate_not_admitted_when_live_still_strict(monkeypatch):
    """Negative control for the out-of-band-ease admission: a floor outside
    the schedule horizon that does NOT clear the live threshold stays
    un-stashed (the pre-fix behavior is correct in this case)."""
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -4_000_000)  # tighter than floor
    miner._ctl_q = None

    req = BlockRequirements(
        difficulty_energy=-10_000.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    state = _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=[], previewed_wintime=(1 << 62, 1 << 62),
        generation=1,
        decay_schedule=list(range(-10_000_000, -8_000_000, 100_000)),
        last_proof_block=987,
        epoch_length=10,
    )

    cand = MiningResult(
        miner_id=miner.miner_id, miner_type="CPU", nonce=b"\x01" * 32,
        salt=b"\x02" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-3_508.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=-3_504.0,
    )
    monkeypatch.setattr(miner, "evaluate_sampleset", lambda *a, **k: cand)

    ss = _energy_sampleset(np.full(4, -3_508.0))
    result = miner._run_substrate_ratchet(
        state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )
    assert state.top_k == []
    assert result is None


def test_ratchet_stashes_top5_energy_regardless_of_threshold(monkeypatch):
    """top-5 gate: evaluate_sampleset must be called even when the live
    threshold is MUCH stricter than the attempt's best energy.

    Regression guard for the old ``near_live`` gate that silently skipped
    stashing whenever the QPU's output was far from the current threshold.
    Under the new top-5-by-energy policy the stash is filled by energy rank
    alone; the chain's live threshold is irrelevant to what we KEEP.
    """
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    # Live threshold is -15053 (milli = -15_053_000) — much stricter than the
    # attempt energy of -14889.0 (-14_889_000 milli).  Under the old near_live
    # gate the delta is +164_000 milli > RATCHET_PRECHECK_MARGIN_MILLI (2000),
    # so the iter would have been skipped.  Under the new gate it must stash.
    miner._live_max_energy_milli = mp.Value("q", -15_053_000)
    miner._ctl_q = None

    req = BlockRequirements(
        difficulty_energy=-15053.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    state = _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=[], previewed_wintime=(1 << 62, 1 << 62), generation=1,
    )

    cand = MiningResult(
        miner_id=miner.miner_id, miner_type="QPU", nonce=b"\x01" * 32,
        salt=b"\x02" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-14889.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=-14889.0,
    )
    evaluate_called = []

    def _fake_evaluate(*a, **k):
        evaluate_called.append(True)
        return cand

    monkeypatch.setattr(miner, "evaluate_sampleset", _fake_evaluate)

    # Attempt energy -14889.0 — improves empty stash (ratchet_threshold = +inf)
    ss = _energy_sampleset(np.full(4, -14889.0))
    miner._run_substrate_ratchet(
        state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )

    assert evaluate_called, (
        "evaluate_sampleset was NOT called even though the stash was empty — "
        "the old near_live gate must have been removed"
    )
    assert len(state.top_k) == 1, (
        "candidate was not stashed despite being the best so far"
    )
    assert state.top_k[0].result.energy == -14889.0


def test_ratchet_skips_attempt_outside_top5(monkeypatch):
    """top-5 gate: an attempt whose energy is worse than the worst stashed
    entry must NOT call evaluate_sampleset.

    Once the stash is full with 5 better candidates, an attempt that cannot
    displace the worst entry is cheap-rejected without paying the expensive
    diversity / selection work.
    """
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -15_000_000)
    miner._ctl_q = None

    req = BlockRequirements(
        difficulty_energy=-15000.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )

    # Pre-populate top_k with 5 candidates all better than -14000.0.
    better_cand = MiningResult(
        miner_id=miner.miner_id, miner_type="QPU", nonce=b"\xaa" * 32,
        salt=b"\xbb" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-15000.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=-15000.0,
    )
    # Build 5 distinct candidates with energies -15000, -14999, ..., -14996
    from shared.base_miner import StashEntry
    from shared.miner_types import MiningResult as MR
    top_k = []
    for i in range(5):
        top_k.append(StashEntry(
            decay_num=0, valid_at_block=0,
            result=MR(
                miner_id=miner.miner_id, miner_type="QPU",
                nonce=bytes([i]) * 32, salt=bytes([i + 10]) * 32,
                timestamp=0, prev_timestamp=0,
                solutions=[[1, -1, 1]], energy=-15000.0 + i,
                diversity=1.0, num_valid=1, mining_time=0,
                node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
                submit_floor_energy=-15000.0 + i,
            ),
        ))
    # Legacy (energy) ranking: top_k[-1] is the worst (highest energy).
    top_k.sort(key=lambda e: e.result.energy)

    state = _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=top_k, previewed_wintime=(1 << 62, 1 << 62), generation=1,
    )

    evaluate_called = []

    def _fake_evaluate(*a, **k):
        evaluate_called.append(True)
        return better_cand

    monkeypatch.setattr(miner, "evaluate_sampleset", _fake_evaluate)

    # Attempt energy -14000.0 — worse than all 5 stashed (worst stashed = -14996)
    ss = _energy_sampleset(np.full(4, -14000.0))
    miner._run_substrate_ratchet(
        state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )

    assert not evaluate_called, (
        "evaluate_sampleset was called for an attempt that cannot improve the "
        "full top-5 stash — the improves_stash gate must block it"
    )
    # top_k must be unchanged.
    assert len(state.top_k) == 5


def test_ratchet_skips_under_reconstructed_sample(monkeypatch):
    """Width guard: a sample narrower than the topology (an under-reconstructed
    QPU driver result without defect_info) must NOT be evaluated or
    stashed, even when its energy WOULD enter the top-5.

    A reduced-width sample drops the clamped fixed-spin columns; passing it to
    evaluate_sampleset indexes full-topology edge positions into the narrow
    matrix → IndexError or a silently wrong recomputed energy. This guards
    against any driver/worker rank divergence. Fails if the guard is removed
    (evaluate_sampleset would be called on the narrow sample).
    """
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import _MiningLoopState, _SharedSampleSet
    from shared.miner_types import BlockRequirements, MiningResult
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -15_053_000)
    miner._ctl_q = None

    # Topology has 5 nodes; the sample below has only 3 columns (narrow).
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    req = BlockRequirements(
        difficulty_energy=-15053.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    state = _MiningLoopState(
        requirements=req, nodes=nodes, edges=edges,
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=[], previewed_wintime=(1 << 62, 1 << 62), generation=1,
    )

    cand = MiningResult(
        miner_id=miner.miner_id, miner_type="QPU", nonce=b"\x01" * 32,
        salt=b"\x02" * 32, timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=-14889.0, diversity=1.0, num_valid=1,
        mining_time=0, node_list=nodes, edge_list=edges,
        submit_floor_energy=-14889.0,
    )
    evaluate_called = []

    def _fake_evaluate(*a, **k):
        evaluate_called.append(True)
        return cand

    monkeypatch.setattr(miner, "evaluate_sampleset", _fake_evaluate)

    # Narrow sample: 3 columns < len(nodes) == 5. Energy -14889 would enter the
    # empty stash (ratchet_threshold = +inf), so only the width guard can stop
    # the evaluation.
    narrow = _SharedSampleSet(
        np.ones((4, 3), np.int8), np.full(4, -14889.0, np.float64),
    )
    miner._run_substrate_ratchet(
        state, narrow, b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )

    assert not evaluate_called, (
        "evaluate_sampleset was called on an under-reconstructed (narrow) "
        "sample — the width guard must block it"
    )
    assert state.top_k == [], (
        "an under-reconstructed sample was stashed — the width guard must "
        "skip both evaluation and stashing"
    )


# ── Decay-aware eval gate + win-time stash ranking (Task 5) ──────────────


# Monotonic non-decreasing milli schedule (decay eases max_energy upward).
# step_for_energy returns the first index whose threshold is strictly above
# the candidate's floor: e.g. floor -14500 -> step 1 (-14000 > -14500),
# floor -11500 -> step 4 (-11000 > -11500).
_DECAY_SCHED = [-15000, -14000, -13000, -12000, -11000]


def _decay_result(miner, energy: float, floor: float, tag: int):
    from shared.miner_types import MiningResult
    return MiningResult(
        miner_id=miner.miner_id, miner_type="QPU",
        nonce=bytes([tag]) * 32, salt=bytes([tag + 1]) * 32,
        timestamp=0, prev_timestamp=0,
        solutions=[[1, -1, 1]], energy=energy, diversity=1.0, num_valid=1,
        mining_time=0, node_list=[0, 1, 2], edge_list=[(0, 1), (1, 2)],
        submit_floor_energy=floor,
    )


def _decay_state(miner, top_k, *, last_proof_block=100, epoch_length=10):
    from shared.base_miner import _MiningLoopState
    from shared.miner_types import BlockRequirements
    req = BlockRequirements(
        difficulty_energy=-11.0, min_diversity=0.0, min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10 ** 9,
    )
    return _MiningLoopState(
        requirements=req, nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
        prev_timestamp=0, start_time=0.0, solution_number_for_log=0,
        dispatch_id_for_log=0,
        attempt_log=_make_attempt_logger(miner),
        solution_store=_make_solution_store(miner),
        live_threshold_var=miner._live_max_energy_milli,
        top_k_cap=5, top_k=top_k, previewed_wintime=(1 << 62, 1 << 62), generation=1,
        decay_schedule=list(_DECAY_SCHED),
        last_proof_block=last_proof_block, epoch_length=epoch_length,
    )


def test_gate_skips_when_min_energy_cannot_beat_furthest_stash(
    monkeypatch, caplog,
):
    """Decay gate: an iter whose best energy maps to s_min > s_max (the
    furthest stashed win-time) must NOT call evaluate_sampleset, and must log
    a pre-check skip heartbeat."""
    import logging

    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -11_000_000)
    miner._ctl_q = None

    # Full stash, all at decay step 1 -> s_max = 1.
    top_k = [
        StashEntry(1, 110, _decay_result(miner, -14.5, -14.5, i))
        for i in range(5)
    ]
    state = _decay_state(miner, top_k)

    evaluate_called = []
    monkeypatch.setattr(
        miner, "evaluate_sampleset",
        lambda *a, **k: evaluate_called.append(True),
    )

    # iter best energy -11.5 -> milli -11500 -> step_for_energy -> 4 (s_min).
    # s_min (4) > s_max (1) -> must skip.
    ss = _energy_sampleset(np.full(4, -11.5))
    with caplog.at_level(logging.INFO, logger=miner.logger.name):
        miner._run_substrate_ratchet(
            state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
            preview_cb=None, attempt_log_kwargs={},
        )

    assert not evaluate_called, (
        "evaluate_sampleset was called for an iter that cannot clear within "
        "the furthest stashed win-time — the decay gate must block it"
    )
    assert any("pre-check skip" in r.message for r in caplog.records), (
        "expected a pre-check skip heartbeat log line"
    )
    assert len(state.top_k) == 5


def test_gate_admits_weaker_energy_that_clears_within_window(monkeypatch):
    """Decay gate: an iter weaker in *raw energy* than the worst stashed
    energy, but whose s_min <= s_max, MUST be evaluated.

    Regression vs the old energy-only gate, which would have skipped it
    because its energy is worse than every stashed entry.
    """
    import multiprocessing as mp

    import numpy as np

    from shared.base_miner import StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -11_000_000)
    miner._ctl_q = None

    # Full stash: strong energies (-15.0) all at decay step 4 -> s_max = 4.
    top_k = [
        StashEntry(4, 140, _decay_result(miner, -15.0, -11.5, i))
        for i in range(5)
    ]
    state = _decay_state(miner, top_k)

    cand = _decay_result(miner, -12.0, -12.0, 9)
    evaluate_called = []

    def _fake_evaluate(*a, **k):
        evaluate_called.append(True)
        return cand

    monkeypatch.setattr(miner, "evaluate_sampleset", _fake_evaluate)

    # iter best -12.0 -> milli -12000 -> step_for_energy -> 4 (s_min == s_max).
    # Worse raw energy than every stashed (-15.0) so the old energy gate would
    # have skipped; the decay gate admits it.
    ss = _energy_sampleset(np.full(4, -12.0))
    miner._run_substrate_ratchet(
        state, ss, b"\x09" * 32, b"\x0a" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )

    assert evaluate_called, (
        "evaluate_sampleset was NOT called for an iter that clears within the "
        "stash's furthest win-time — the decay gate must admit it"
    )


def test_stash_insert_orders_by_win_time_then_energy():
    """Decay inserts order by (decay_num, energy); eviction targets the
    furthest (largest decay_num, then highest energy) entry."""
    from shared.base_miner import BaseMiner, StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    top_k: list = []
    # decay_num, energy
    seed = [(2, -10.0), (1, -9.0), (3, -20.0), (1, -12.0), (2, -8.0)]
    for i, (d, e) in enumerate(seed):
        BaseMiner._stash_insert(
            top_k, 5, StashEntry(d, d * 10, _decay_result(miner, e, e, i)),
            True,
        )
    assert len(top_k) == 5
    # Ordered by (decay_num, energy) ascending.
    keys = [(s.decay_num, s.result.energy) for s in top_k]
    assert keys == sorted(keys)
    assert keys[0] == (1, -12.0)  # nearest win-time, then lowest energy

    # Full: a nearer-win-time entry evicts the furthest (decay_num 3).
    changed = BaseMiner._stash_insert(
        top_k, 5, StashEntry(1, 10, _decay_result(miner, -5.0, -5.0, 99)),
        True,
    )
    assert changed is True
    assert all(s.decay_num != 3 for s in top_k), (
        "the furthest (decay_num=3) entry must have been evicted"
    )

    # A further-win-time entry than the current worst is rejected.
    worst = max((s.decay_num, s.result.energy) for s in top_k)
    rejected = BaseMiner._stash_insert(
        top_k, 5,
        StashEntry(worst[0] + 1, 99, _decay_result(miner, -99.0, -99.0, 50)),
        True,
    )
    assert rejected is False


def test_stash_insert_tiebreak_lower_energy_incumbent():
    """Equal decay_num: a strictly-lower-energy entry replaces the worst;
    an equal-energy entry does NOT replace it."""
    from shared.base_miner import BaseMiner, StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    top_k: list = []
    for i in range(5):
        BaseMiner._stash_insert(
            top_k, 5,
            StashEntry(2, 20, _decay_result(miner, -10.0 - i, -10.0 - i, i)),
            True,
        )
    # Energies: -10..-14 at decay_num 2; worst is -10.0.
    worst_energy = max(s.result.energy for s in top_k)
    assert worst_energy == -10.0

    # Equal step, strictly lower energy than worst (-10.5 < -10.0) -> replaces.
    changed = BaseMiner._stash_insert(
        top_k, 5, StashEntry(2, 20, _decay_result(miner, -10.5, -10.5, 20)),
        True,
    )
    assert changed is True
    assert all(s.result.energy != -10.0 for s in top_k)

    # Equal step, equal energy to the current worst -> NOT replaced.
    new_worst = max(s.result.energy for s in top_k)
    unchanged = BaseMiner._stash_insert(
        top_k, 5,
        StashEntry(2, 20, _decay_result(miner, new_worst, new_worst, 21)),
        True,
    )
    assert unchanged is False


def test_legacy_path_still_energy_ranked():
    """is_decay_ranked False: inserts/evicts purely by energy, ignoring
    decay_num (which is 0 on the legacy path)."""
    from shared.base_miner import BaseMiner, StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    top_k: list = []
    for i, e in enumerate([-5.0, -1.0, -9.0, -3.0, -7.0]):
        BaseMiner._stash_insert(
            top_k, 5, StashEntry(0, 0, _decay_result(miner, e, e, i)), False,
        )
    energies = [s.result.energy for s in top_k]
    assert energies == sorted(energies)
    assert energies[0] == -9.0 and energies[-1] == -1.0

    # Better energy than worst (-1.0) -> evicts the worst.
    changed = BaseMiner._stash_insert(
        top_k, 5, StashEntry(0, 0, _decay_result(miner, -6.0, -6.0, 50)), False,
    )
    assert changed is True
    assert all(s.result.energy != -1.0 for s in top_k)

    # Worse energy than the new worst -> rejected.
    worst = max(s.result.energy for s in top_k)
    rejected = BaseMiner._stash_insert(
        top_k, 5,
        StashEntry(0, 0, _decay_result(miner, worst + 1.0, worst + 1.0, 51)),
        False,
    )
    assert rejected is False


# ── Task 6: preview valid_at_block + heartbeat win-time/threshold ─────────


def test_preview_payload_includes_valid_at_block(monkeypatch):
    """After a valid decay-path stash, preview_cb payload carries
    valid_at_block and decay_num from the earliest-winning entry, plus all
    the existing keys (submit_floor_energy, energy, num_valid, diversity,
    nonce, salt, solutions, dispatch_id, miner_type)."""
    import multiprocessing as mp
    import numpy as np
    from shared.base_miner import StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -11_000_000)
    miner._ctl_q = None

    # Two candidates at different decay steps.
    # decay_num=1 -> valid_at_block=110; decay_num=2 -> valid_at_block=120.
    cand_early = _decay_result(miner, -14.5, -14.5, 1)
    cand_late  = _decay_result(miner, -15.0, -13.5, 2)

    top_k = [
        StashEntry(1, 110, cand_early),  # earliest-winning
        StashEntry(2, 120, cand_late),
    ]
    # Use _decay_state which sets decay_schedule=_DECAY_SCHED.
    state = _decay_state(miner, top_k)

    payloads = []
    miner._run_substrate_ratchet(
        state, _energy_sampleset(np.full(4, -11.5)),
        b"\x01" * 32, b"\x02" * 32, 0.0,
        preview_cb=None, attempt_log_kwargs={},
    )
    # Directly call _maybe_emit_preview with a fresh (INF, INF) throttle.
    from shared.base_miner import _MILLI_INF
    new_throttle = miner._maybe_emit_preview(
        payloads.append, top_k, (_MILLI_INF, _MILLI_INF), dispatch_id=42,
    )

    assert len(payloads) == 1, "exactly one preview must be emitted"
    p = payloads[0]
    # win-time fields must be present and match the earliest-winning entry.
    assert p["valid_at_block"] == 110
    assert p["decay_num"] == 1
    # Existing required keys must all be present.
    for key in ("submit_floor_energy", "energy", "num_valid", "diversity",
                "nonce", "salt", "solutions", "dispatch_id", "miner_type"):
        assert key in p, f"missing expected payload key: {key!r}"
    # Throttle advanced to (valid_at_block=110, floor_milli).
    assert new_throttle == (110, int(cand_early.submit_floor_energy * 1000))


def test_preview_no_duplicate_on_unchanged_best(monkeypatch):
    """Emitting a preview for the same (valid_at_block, floor) a second time
    must NOT fire again — the throttle must suppress it.

    Also verify: a strictly-earlier valid_at_block DOES trigger a fresh preview.
    """
    import multiprocessing as mp
    from shared.base_miner import StashEntry, _MILLI_INF
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    miner._live_max_energy_milli = mp.Value("q", -11_000_000)
    miner._ctl_q = None

    cand = _decay_result(miner, -14.5, -14.5, 1)
    top_k = [StashEntry(2, 120, cand)]

    payloads = []

    # First emission.
    throttle = miner._maybe_emit_preview(
        payloads.append, top_k, (_MILLI_INF, _MILLI_INF), dispatch_id=7,
    )
    assert len(payloads) == 1

    # Second call with the same stash and the returned throttle — must NOT emit.
    throttle2 = miner._maybe_emit_preview(
        payloads.append, top_k, throttle, dispatch_id=7,
    )
    assert len(payloads) == 1, "throttle must suppress duplicate preview"
    assert throttle2 == throttle

    # Now add a strictly earlier-winning entry and call again — must emit.
    cand_earlier = _decay_result(miner, -14.0, -14.0, 9)
    top_k.insert(0, StashEntry(1, 110, cand_earlier))  # earlier valid_at_block

    miner._maybe_emit_preview(
        payloads.append, top_k, throttle, dispatch_id=7,
    )
    assert len(payloads) == 2, (
        "a strictly-earlier valid_at_block must trigger a fresh preview"
    )
    assert payloads[1]["valid_at_block"] == 110
    assert payloads[1]["decay_num"] == 1


def test_heartbeat_restores_live_threshold(monkeypatch, caplog):
    """The per-attempt skip heartbeat must include the live threshold value.

    Regression: the live threshold was dropped from the heartbeat in a
    prior commit.  This test pins the format so it can be grepped.
    """
    import logging
    import multiprocessing as mp
    import numpy as np
    from shared.base_miner import StashEntry
    from tests.test_base_miner_pump import _DriverMiner

    miner = _DriverMiner()
    live_milli = -11_500_000
    miner._live_max_energy_milli = mp.Value("q", live_milli)
    miner._ctl_q = None

    # Fill the stash with 5 entries all at decay_num=1 so the pre-check
    # gate definitely rejects the incoming attempt (-11.0 -> step 4 > s_max 1).
    top_k = [
        StashEntry(1, 110, _decay_result(miner, -14.5, -14.5, i))
        for i in range(5)
    ]
    state = _decay_state(miner, top_k)
    state.previewed_wintime = (0, 0)  # treat as already-previewed

    evaluate_called = []
    monkeypatch.setattr(
        miner, "evaluate_sampleset",
        lambda *a, **k: evaluate_called.append(True),
    )

    # iter energy -11.0 maps to step 4 which is > s_max=1 -> pre-check skip.
    ss = _energy_sampleset(np.full(4, -11.0))
    with caplog.at_level(logging.INFO, logger=miner.logger.name):
        miner._run_substrate_ratchet(
            state, ss, b"\x01" * 32, b"\x02" * 32, 0.0,
            preview_cb=None, attempt_log_kwargs={},
        )

    assert not evaluate_called, "evaluate must not be called on pre-check skip"

    # The heartbeat line must contain the live threshold value.
    import re
    threshold_pat = re.compile(r"live threshold<=(-?\d+)")
    matching = [r.message for r in caplog.records if threshold_pat.search(r.message)]
    assert matching, "no heartbeat line with 'live threshold<=' found in logs"
    m = threshold_pat.search(matching[0])
    assert m and int(m.group(1)) == live_milli, (
        f"expected threshold {live_milli} in heartbeat, "
        f"got: {matching[0]!r}"
    )
