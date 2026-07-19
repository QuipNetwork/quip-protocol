# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 QUIP Protocol Contributors

"""SubmissionArchiver: durable full-sampleset capture for QPU archive mode.

The archiver underpins the QPU reads x diversity x count study: every
per-read spin vector and energy must survive bit-packing, column
permutation, crash-interrupted flushes, and resume — silently corrupted
or mis-ordered spins would poison every offline (E', r, k, D) re-scoring
built on top. These tests pin:

  * exact spins/energies round-trip through packed npz shards, including
    sampleset variable orderings that differ from topology node order;
  * the pack convention matches ``pack_spins_hex`` (bit 1 = +1, big order);
  * ``valid_live`` set on the returned entry before flush is persisted;
  * short samplesets become error meta lines, never ragged shard rows;
  * a variable SET mismatch (solver retune mid-study) aborts loudly;
  * resume counts only committed meta lines — torn final lines and
    orphan shards (crash between shard rename and meta append) re-run;
  * resume refuses parameter drift against metadata.json and refuses to
    continue an existing cell without --resume.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.quantum_proof_of_work import pack_spins_hex

_SPEC = importlib.util.spec_from_file_location(
    "qpu_throughput_canary",
    Path(__file__).resolve().parent.parent
    / "tools" / "qpu_throughput_canary.py",
)
qtc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qtc)

N_NODES = 11
NODES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]  # non-contiguous labels
R = 4


def _model(seed: int):
    return SimpleNamespace(
        nonce=bytes([seed]) * 32, salt=bytes([255 - seed]) * 32,
    )


def _sampleset(rng, *, variables=None, num_reads=R, timing=None):
    spins = rng.choice([-1, 1], size=(num_reads, N_NODES)).astype(np.int8)
    energies = rng.normal(-100.0, 5.0, size=num_reads)
    return SimpleNamespace(
        record=SimpleNamespace(sample=spins, energy=energies),
        variables=list(variables if variables is not None else NODES),
        info={
            "problem_id": "fake-problem",
            "timing": timing or {
                "qpu_programming_time": 14000,
                "qpu_sampling_time": 1500,
                "qpu_anneal_time_per_sample": 80.0,
                "qpu_readout_time_per_sample": 245.0,
                "qpu_delay_time_per_sample": 20.5,
            },
        },
    ), spins, energies


def _archiver(tmp_path, *, n_target=100, resume=False, num_reads=R):
    return qtc.SubmissionArchiver(
        tmp_path / "cell",
        nodes=NODES,
        num_reads=num_reads,
        annealing_time_us=80.0,
        n_target=n_target,
        metadata_extra={"cell": "test"},
        flush_every=3,
        resume=resume,
    )


def _load_rows(cell_dir: Path):
    """Reconstruct (idx -> spins, energies, meta) the way the offline
    pipeline does: meta.jsonl pointers only."""
    out = {}
    shards = {}
    with open(cell_dir / "meta.jsonl") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # healed torn fragment — loader contract: skip
            if rec.get("error") is not None:
                out[rec["idx"]] = (None, None, rec)
                continue
            name = rec["shard"]
            if name not in shards:
                shards[name] = np.load(cell_dir / "shards" / name)
            sh = shards[name]
            packed = sh["spins_packed"][rec["shard_row"]]
            bits = np.unpackbits(packed, axis=-1)[..., :N_NODES]
            spins = (bits.astype(np.int8) * 2) - 1
            out[rec["idx"]] = (spins, sh["energies"][rec["shard_row"]], rec)
    return out


def test_round_trip_exact_spins_and_energies(tmp_path):
    rng = np.random.default_rng(7)
    arch = _archiver(tmp_path)
    originals = {}
    for i in range(7):  # crosses two flush windows (flush_every=3)
        if i % 2:
            # Permuted variable order — archiver must un-permute back
            # to topology node order.
            perm = rng.permutation(N_NODES)
            variables = [NODES[j] for j in perm]
            ss, spins, energies = _sampleset(rng, variables=variables)
            originals[i] = (spins[:, np.argsort(perm)], energies)
        else:
            ss, spins, energies = _sampleset(rng)
            originals[i] = (spins, energies)
        entry = arch.record(_model(i), ss, qpu_us=63000 + i, local_idx=i)
        entry["valid_live"] = bool(i == 4)  # caller sets post-validation
    arch.flush()

    rows = _load_rows(tmp_path / "cell")
    assert sorted(rows) == list(range(7))
    for i, (want_spins, want_e) in originals.items():
        got_spins, got_e, rec = rows[i]
        np.testing.assert_array_equal(got_spins, want_spins)
        np.testing.assert_allclose(got_e, want_e)
        assert rec["valid_live"] == (i == 4)
        assert rec["nonce_hex"] == (bytes([i]) * 32).hex()
        assert rec["timing"]["qpu_programming_time"] == 14000
        assert rec["best_energy"] == pytest.approx(want_e.min())


def test_pack_convention_matches_pack_spins_hex(tmp_path):
    rng = np.random.default_rng(11)
    arch = _archiver(tmp_path)
    ss, spins, _ = _sampleset(rng)
    arch.record(_model(0), ss, qpu_us=1, local_idx=0)
    arch.flush()
    shard_path = sorted((tmp_path / "cell" / "shards").iterdir())[0]
    packed = np.load(shard_path)["spins_packed"][0]  # (R, bytes_per_read)
    for read in range(R):
        want = bytes.fromhex(pack_spins_hex(spins[read].tolist()))
        assert packed[read].tobytes() == want


def test_short_sampleset_becomes_error_line(tmp_path):
    rng = np.random.default_rng(3)
    arch = _archiver(tmp_path)
    ss, _, _ = _sampleset(rng, num_reads=R - 1)
    entry = arch.record(_model(0), ss, qpu_us=1, local_idx=0)
    assert entry is None
    arch.flush()
    rows = _load_rows(tmp_path / "cell")
    assert rows[0][2]["error"].startswith("short sampleset")
    assert not any((tmp_path / "cell" / "shards").iterdir())


def test_variable_set_mismatch_aborts(tmp_path):
    rng = np.random.default_rng(5)
    arch = _archiver(tmp_path)
    bad_nodes = list(NODES)
    bad_nodes[-1] = 999  # qubit retuned away mid-study
    ss, _, _ = _sampleset(rng, variables=bad_nodes)
    with pytest.raises(RuntimeError, match="does not match"):
        arch.record(_model(0), ss, qpu_us=1, local_idx=0)


def test_resume_counts_committed_and_tolerates_torn_tail(tmp_path):
    rng = np.random.default_rng(9)
    arch = _archiver(tmp_path)
    for i in range(5):
        ss, _, _ = _sampleset(rng)
        arch.record(_model(i), ss, qpu_us=1, local_idx=i)
    arch.flush()

    arch2 = _archiver(tmp_path, resume=True)
    assert arch2.start_idx == 5

    # Torn final line (crash mid-append): ignored, that idx re-runs.
    with open(tmp_path / "cell" / "meta.jsonl", "a") as fh:
        fh.write('{"idx": 5, "truncated')
    arch3 = _archiver(tmp_path, resume=True)
    assert arch3.start_idx == 5

    # Orphan shard (crash between shard rename and meta append): the
    # unreferenced rows don't advance the resume point.
    orphan = tmp_path / "cell" / "shards" / "spins_000005_000007.npz"
    np.savez(orphan, spins_packed=np.zeros((3, R, 2), dtype=np.uint8),
             energies=np.zeros((3, R)), idx=np.arange(5, 8))
    arch4 = _archiver(tmp_path, resume=True)
    assert arch4.start_idx == 5

    # Resumed instance keeps writing at the right global index.
    ss, spins, energies = _sampleset(rng)
    arch4.record(_model(5), ss, qpu_us=1, local_idx=0)
    arch4.flush()
    rows = _load_rows(tmp_path / "cell")
    np.testing.assert_array_equal(rows[5][0], spins)


def test_resume_refuses_param_drift_and_fresh_refuses_existing(tmp_path):
    rng = np.random.default_rng(13)
    arch = _archiver(tmp_path)
    ss, _, _ = _sampleset(rng)
    arch.record(_model(0), ss, qpu_us=1, local_idx=0)
    arch.flush()

    with pytest.raises(SystemExit, match="never truncated"):
        _archiver(tmp_path, resume=False)
    with pytest.raises(SystemExit, match="resume refused"):
        _archiver(tmp_path, resume=True, num_reads=R * 2)
