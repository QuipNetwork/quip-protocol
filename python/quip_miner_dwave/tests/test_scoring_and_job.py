"""Consensus scoring + job reject paths (offline mock sampler)."""
import time
from typing import Dict, Optional, Tuple

from quip_proto import miner_pb2, scoring, wire

from quip_miner_dwave.job import handle_job
from quip_miner_dwave.ocean import OceanSampler, SampleResult


def test_energy_milli_matches_golden_shape():
    # Two-spin ferromagnetic: h=[1,-1], J=[0.5], spins=[+1,+1]
    spins = [1, 1]
    h = [1.0, -1.0]
    j = [0.5]
    edges = [(0, 1)]
    e = scoring.energy_milli(spins, h, j, edges)
    # E = 1*1 + (-1)*1 + 0.5*1*1 = 0.5 → 500 milli
    assert e == 500


def test_malformed_and_expired_rejects():
    sampler = OceanSampler(mock=True)
    # MALFORMED h
    bad = miner_pb2.Job(
        job_id=b"bad",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            h_milli_le32=b"\x01\x02\x03",
            j_milli_le32=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(bad, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert len(msgs) == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED

    # EXPIRED
    expired = miner_pb2.Job(
        job_id=b"old",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) - 60_000,
        ising=miner_pb2.IsingProblem(
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(expired, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert msgs[0].reject.reason == miner_pb2.EXPIRED
    sampler.close()


def test_valid_job_produces_result_with_access_time():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"job-1",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(job, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    kinds = [m.WhichOneof("msg") for m in msgs]
    assert "result" in kinds
    assert "job_request" in kinds
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    assert result.job_id == b"job-1"
    assert len(result.solutions) >= 1
    assert result.meta.device_access_time_us > 0
    # Consensus energy matches scorer on returned spins
    sol = result.solutions[0]
    spins = wire.decode_spins(sol.spins_bytes)
    expected = scoring.energy_milli(
        spins, [1.0, -1.0], [0.5], [(0, 1)]
    )
    assert sol.energy_milli == expected
    sampler.close()


def test_sparse_session_topology_scores_edges():
    # Session topology with non-dense qubit ids (10, 20): the consensus energy
    # must map edges to spin-vector positions, otherwise the J coupling is
    # silently dropped and the reported energy diverges from the coordinator's.
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"sparse",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            # No inline EdgeList -> resolve against the session topology.
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            num_reads=1,
        ),
    )
    msgs = handle_job(
        job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)]
    )
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    sol = result.solutions[0]
    spins = wire.decode_spins(sol.spins_bytes)
    # Positional scoring: edge (10,20) -> spin positions (0,1).
    expected = scoring.energy_milli(spins, [1.0, -1.0], [0.5], [(0, 1)])
    assert sol.energy_milli == expected
    # The coupling must actually be counted (ground state is -2500 milli, not
    # the -2000 an unmapped/dropped edge would yield).
    assert sol.energy_milli == -2500
    sampler.close()


def test_unsupported_kind():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"gate",
        kind=miner_pb2.GATE_CIRCUIT,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            h_milli_le32=wire.encode_i32_le([1000]),
        ),
    )
    msgs = handle_job(job, sampler, session_nodes=[], session_edges=[])
    assert msgs[0].reject.reason == miner_pb2.UNSUPPORTED_KIND
    sampler.close()


def _hash_job(job_id: bytes, topology_hash: bytes) -> "miner_pb2.Job":
    return miner_pb2.Job(
        job_id=job_id,
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            topology_hash=topology_hash,
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            num_reads=1,
        ),
    )


def test_topology_hash_job_without_cache_rejects_missing():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-missing", b"\x11" * 32)
    msgs = handle_job(
        job, sampler, session_nodes=[], session_edges=[], session_hash=None
    )
    assert len(msgs) == 1
    assert msgs[0].reject.reason == miner_pb2.TOPOLOGY_MISSING
    sampler.close()


def test_topology_hash_job_wrong_hash_rejects_mismatch():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-wrong", b"\x22" * 32)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_hash=b"\x11" * 32,
    )
    assert len(msgs) == 1
    assert msgs[0].reject.reason == miner_pb2.TOPOLOGY_MISMATCH
    sampler.close()


def test_topology_hash_job_match_resolves_to_result():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-ok", b"\x11" * 32)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_hash=b"\x11" * 32,
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


def test_set_target_num_reads_override_is_honored():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"st",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            num_reads=0,  # unset -> falls through to SetTarget
        ),
    )
    target = miner_pb2.SetTarget(max_energy_milli=1000, min_solutions=1, num_reads=3)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


class _RecordingSampler:
    """Duck-typed stand-in for OceanSampler that records ``sample()`` kwargs.

    Avoids exercising the real dimod/D-Wave path — only checks that job.py
    resolves and forwards ``anneal_time_us`` correctly (quip-w5p.3).
    """

    def __init__(self):
        self.calls: list = []

    def sample(
        self,
        h: Dict[int, float],
        j: Dict[Tuple[int, int], float],
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "",
    ) -> SampleResult:
        self.calls.append({"num_reads": num_reads, "anneal_time_us": anneal_time_us})
        return SampleResult(
            samples=[{0: 1, 1: -1}],
            energies=[0.0],
            device_access_time_us=1,
            num_reads=1,
            extra={},
        )

    def close(self) -> None:
        pass


def _make_job(job_id: bytes, *, anneal_time_us: int = 0) -> "miner_pb2.Job":
    return miner_pb2.Job(
        job_id=job_id,
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            num_reads=1,
            anneal_time_us=anneal_time_us,
        ),
    )


def test_anneal_time_us_unset_passes_none_to_sampler():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-unset")
    handle_job(job, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert sampler.calls[0]["anneal_time_us"] is None


def test_anneal_time_us_per_job_override_is_honored():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-job", anneal_time_us=250)
    target = miner_pb2.SetTarget(anneal_time_us=999)
    handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    # Per-job override wins over SetTarget.
    assert sampler.calls[0]["anneal_time_us"] == 250


def test_anneal_time_us_session_target_fallback_is_honored():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-target")  # unset -> falls through to SetTarget
    target = miner_pb2.SetTarget(anneal_time_us=777)
    handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    assert sampler.calls[0]["anneal_time_us"] == 777
