"""Job validation, sampling, and Result/Reject construction."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

from quip_proto import miner_pb2, scoring, wire

from quip_miner_dwave.ocean import OceanSampler, SampleResult

logger = logging.getLogger(__name__)


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def decode_milli_f64(raw: bytes) -> List[float]:
    """Decode little-endian i32 milli array to floats. Raises ValueError if bad."""
    return [v / 1000.0 for v in wire.decode_i32_le(raw)]


def edges_of(ising: miner_pb2.IsingProblem) -> List[Tuple[int, int]]:
    which = ising.WhichOneof("graph")
    if which == "edges":
        e = ising.edges
        return list(zip(e.u, e.v))
    return []


def resolve_graph(
    ising: miner_pb2.IsingProblem,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    n_h: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (nodes, edges) for a job.

    Inline EdgeList uses dense 0..n-1 node ids from ``n_h``; topology-hash
    jobs use the session-cached Topology.
    """
    which = ising.WhichOneof("graph")
    if which == "edges":
        return list(range(n_h)), edges_of(ising)
    return list(session_nodes), list(session_edges)


def spins_to_bytes(spins: Sequence[int]) -> bytes:
    return wire.encode_spins(list(spins))


def sample_dict_to_vector(
    sample: Dict[int, int], nodes: Sequence[int]
) -> List[int]:
    """Map a label-keyed sample into a dense spin vector in ``nodes`` order."""
    out: List[int] = []
    for n in nodes:
        s = sample.get(int(n), 1)
        out.append(1 if s >= 0 else -1)
    return out


def handle_job(
    job: miner_pb2.Job,
    sampler: OceanSampler,
    *,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    session_hash: Optional[bytes] = None,
    session_target: Optional["miner_pb2.SetTarget"] = None,
) -> List[miner_pb2.MinerMsg]:
    """Validate and solve one job; return Result+JobRequest or Reject messages.

    ``session_hash`` is the hash of the cached ``Topology`` (``None`` until one
    arrives). A ``topology_hash`` job with no cache rejects ``TOPOLOGY_MISSING``;
    a hash that differs from the cache rejects ``TOPOLOGY_MISMATCH`` — matching
    the Rust miners so both reject a topology desync identically.
    """
    job_id = job.job_id

    if job.kind not in (
        miner_pb2.ISING_SAMPLE,
        miner_pb2.JOB_KIND_UNSPECIFIED,
    ):
        return [
            miner_pb2.MinerMsg(
                reject=miner_pb2.Reject(
                    job_id=job_id,
                    reason=miner_pb2.UNSUPPORTED_KIND,
                )
            )
        ]

    ising = job.ising if job.HasField("ising") else miner_pb2.IsingProblem()

    # MALFORMED: h or j length not a multiple of 4
    try:
        h = decode_milli_f64(ising.h_milli_le32)
    except ValueError:
        return [
            miner_pb2.MinerMsg(
                reject=miner_pb2.Reject(
                    job_id=job_id, reason=miner_pb2.MALFORMED
                )
            )
        ]
    try:
        j_vals = decode_milli_f64(ising.j_milli_le32) if ising.j_milli_le32 else []
    except ValueError:
        return [
            miner_pb2.MinerMsg(
                reject=miner_pb2.Reject(
                    job_id=job_id, reason=miner_pb2.MALFORMED
                )
            )
        ]

    if job.deadline_ms and job.deadline_ms < now_unix_ms():
        return [
            miner_pb2.MinerMsg(
                reject=miner_pb2.Reject(
                    job_id=job_id, reason=miner_pb2.EXPIRED
                )
            )
        ]

    if ising.WhichOneof("graph") == "topology_hash":
        if session_hash is None:
            return [
                miner_pb2.MinerMsg(
                    reject=miner_pb2.Reject(
                        job_id=job_id, reason=miner_pb2.TOPOLOGY_MISSING
                    )
                )
            ]
        if bytes(ising.topology_hash) != bytes(session_hash):
            return [
                miner_pb2.MinerMsg(
                    reject=miner_pb2.Reject(
                        job_id=job_id, reason=miner_pb2.TOPOLOGY_MISMATCH
                    )
                )
            ]

    nodes, edges = resolve_graph(ising, session_nodes, session_edges, len(h))
    if not nodes and h:
        nodes = list(range(len(h)))

    h_dict = {int(nodes[i]): float(h[i]) for i in range(min(len(nodes), len(h)))}
    j_dict: Dict[Tuple[int, int], float] = {}
    for k, (u, v) in enumerate(edges):
        if k < len(j_vals):
            j_dict[(int(u), int(v))] = float(j_vals[k])

    # Sampling budget precedence: per-job override > SetTarget override >
    # default. (Full energy-based adapt for the QPU path is a follow-up;
    # see quip-asx.* — it needs the shared GSE model and QPU credits.)
    num_reads = int(ising.num_reads)
    if num_reads == 0 and session_target is not None and session_target.num_reads:
        num_reads = int(session_target.num_reads)
    if num_reads == 0:
        num_reads = 1
    # Same precedence for anneal_time_us: per-job override > SetTarget
    # override > unset (0 = let the QPU use its hardware-default anneal).
    anneal_time_us = int(ising.anneal_time_us)
    if (
        anneal_time_us == 0
        and session_target is not None
        and session_target.anneal_time_us
    ):
        anneal_time_us = int(session_target.anneal_time_us)
    # Use job_id bytes as clamp seed when present
    nonce_seed = bytes(job_id) if job_id else None

    result: SampleResult = sampler.sample(
        h_dict,
        j_dict,
        num_reads=num_reads,
        anneal_time_us=anneal_time_us or None,
        nonce_seed=nonce_seed,
        label=f"quip-{job_id.hex()[:8] if job_id else 'job'}",
    )

    # ``scoring.energy_milli`` indexes ``spins[u]``/``spins[v]`` by edge
    # endpoint, so edges must be positions into the dense spin vector (which is
    # ordered by ``nodes``), not raw qubit ids. These coincide for inline
    # EdgeList jobs (nodes == 0..n-1) but not for sparse session topologies.
    pos = {int(n): i for i, n in enumerate(nodes)}
    scored_edges: List[Tuple[int, int]] = []
    scored_j: List[float] = []
    for k, (u, v) in enumerate(edges):
        if k >= len(j_vals):
            break
        iu = pos.get(int(u))
        iv = pos.get(int(v))
        if iu is not None and iv is not None:
            scored_edges.append((iu, iv))
            scored_j.append(j_vals[k])

    solutions = []
    for sample, _qpu_e in zip(result.samples, result.energies):
        spins = sample_dict_to_vector(sample, nodes if nodes else sorted(sample))
        # Consensus energy: always recompute via golden-matched scorer
        e_milli = scoring.energy_milli(spins, h, scored_j, scored_edges)
        solutions.append(
            miner_pb2.Solution(
                spins_bytes=spins_to_bytes(spins),
                energy_milli=e_milli,
            )
        )

    meta = miner_pb2.SamplerMeta(
        reads=result.num_reads,
        sweeps=0,
        device_access_time_us=result.device_access_time_us,
        qpu_access_us=result.device_access_time_us,
        extra=result.extra,
    )
    return [
        miner_pb2.MinerMsg(
            result=miner_pb2.Result(
                job_id=job_id,
                solutions=solutions,
                meta=meta,
            )
        ),
        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
    ]
