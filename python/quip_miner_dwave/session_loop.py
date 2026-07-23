"""gRPC session loop: Hello → Welcome → Configure → credit/job cycle.

Mirrors ``rust/quip-mock-miner`` behavior using the ``quip_proto`` Python SDK.
Uses the synchronous gRPC client with a request queue (reliable over UDS).
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Iterator, Optional, Tuple

import grpc

from quip_proto import miner_pb2, miner_pb2_grpc, session as session_sdk

from quip_miner_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_INTERNAL_FATAL,
    EXIT_TOKEN_REJECTED,
)
from quip_miner_dwave.budget import QPUTimeManager, budget_from_backend_toml
from quip_miner_dwave.job import handle_job
from quip_miner_dwave.ocean import OceanSampler

logger = logging.getLogger(__name__)

_STOP = object()


def _status(miner_id: str, jobs_done: int = 0, abandoned: int = 0) -> miner_pb2.MinerMsg:
    return miner_pb2.MinerMsg(
        status=miner_pb2.Status(
            miner_id=miner_id,
            utilization=0.0,
            jobs_done=jobs_done,
            abandoned_generation=abandoned,
        )
    )


def _unix_target(uri: str) -> str:
    """Normalize ``unix:///path`` or bare path to a grpc UDS target."""
    if uri.startswith("unix://"):
        path = uri[len("unix://") :]
    elif uri.startswith("unix:"):
        path = uri[len("unix:") :]
    else:
        path = uri
    if not path.startswith("/"):
        path = "/" + path
    # grpc-python accepts both unix:/abs and unix:///abs; prefer the triple-slash
    # form which matches tonic's unix:// advertisement after strip.
    return f"unix://{path}"


def run_session(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[QPUTimeManager] = None,
) -> int:
    """Run one miner session; return a process exit code."""
    target = _unix_target(coordinator_uri)
    try:
        hello = session_sdk.build_hello(
            miner_id,
            BACKEND,
            ALGORITHM,
            [miner_pb2.ISING_SAMPLE],
        )
    except session_sdk.MissingToken:
        logger.error("QUIP_SESSION_TOKEN unset")
        return EXIT_TOKEN_REJECTED

    if sampler.native_topology_hash:
        hello.native_topology_hash = sampler.native_topology_hash

    out_q: queue.Queue = queue.Queue()
    # Pre-buffer Hello so the coordinator handshake has something immediately.
    out_q.put(miner_pb2.MinerMsg(hello=hello))

    def request_iter() -> Iterator[miner_pb2.MinerMsg]:
        while True:
            item = out_q.get()
            if item is _STOP:
                return
            yield item

    jobs_done = 0
    grace_ms = 5000
    config: Optional[session_sdk.SessionConfig] = None
    session_nodes: list[int] = []
    session_edges: list[Tuple[int, int]] = []
    session_hash: Optional[bytes] = None
    session_target: Optional[miner_pb2.SetTarget] = None
    pending_budget = budget
    exit_code = EXIT_CLEAN

    # tonic (Rust) rejects UDS streams whose :authority is the socket path;
    # pin a conventional authority so grpc-python ↔ tonic interop works.
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.default_authority", "localhost"),
            ("grpc.enable_http_proxy", 0),
        ],
    )
    try:
        # Wait briefly for the coordinator's listener (race with process spawn).
        deadline = time.time() + 10.0
        while True:
            try:
                grpc.channel_ready_future(channel).result(
                    timeout=max(0.05, deadline - time.time())
                )
                break
            except grpc.FutureTimeoutError:
                if time.time() >= deadline:
                    logger.error("timed out waiting for coordinator at %s", target)
                    return EXIT_INTERNAL_FATAL

        stub = miner_pb2_grpc.MinerServiceStub(channel)
        responses = stub.Session(request_iter())

        last_activity = time.monotonic()
        for cm in responses:
            last_activity = time.monotonic()
            which = cm.WhichOneof("msg")
            if which == "welcome":
                ver = cm.welcome.protocol_version
                if ver not in (0, 1):
                    logger.error("bad Welcome protocol_version=%s", ver)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            fatal=miner_pb2.Fatal(
                                exit_code=EXIT_INTERNAL_FATAL,
                                reason="bad welcome protocol_version",
                                restart_required=False,
                            )
                        )
                    )
                    exit_code = EXIT_INTERNAL_FATAL
                    break
            elif which == "configure":
                config = session_sdk.session_config_from_configure(
                    miner_id, cm.configure
                )
                if pending_budget is None and cm.configure.backend_toml:
                    pending_budget = budget_from_backend_toml(
                        cm.configure.backend_toml
                    )
                out_q.put(miner_pb2.MinerMsg(ready=miner_pb2.Ready()))
                if pending_budget is None or pending_budget.should_mine().should_mine:
                    depth = config.queue_depth if config else 3
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=depth)
                        )
                    )
            elif which == "topology":
                topo = cm.topology
                session_nodes = list(topo.nodes)
                session_hash = bytes(topo.hash)
                if topo.HasField("edges"):
                    session_edges = list(zip(topo.edges.u, topo.edges.v))
                else:
                    session_edges = []
                sampler.set_session_topology(session_nodes, session_edges)
            elif which == "set_target":
                session_target = cm.set_target
            elif which == "job":
                if (
                    pending_budget is not None
                    and not pending_budget.should_mine().should_mine
                ):
                    out_q.put(
                        miner_pb2.MinerMsg(
                            reject=miner_pb2.Reject(
                                job_id=cm.job.job_id,
                                reason=miner_pb2.OVERLOADED,
                            )
                        )
                    )
                    continue
                replies = handle_job(
                    cm.job,
                    sampler,
                    session_nodes=session_nodes,
                    session_edges=session_edges,
                    session_hash=session_hash,
                    session_target=session_target,
                )
                for reply in replies:
                    kind = reply.WhichOneof("msg")
                    if kind == "result":
                        jobs_done += 1
                        meta = reply.result.meta
                        if pending_budget is not None and meta is not None:
                            pending_budget.record_access_time(
                                meta.device_access_time_us
                            )
                    if kind == "job_request" and pending_budget is not None:
                        if not pending_budget.should_mine().should_mine:
                            pending_budget.end_burst()
                            continue
                    out_q.put(reply)
            elif which == "cancel":
                out_q.put(_status(miner_id, jobs_done, abandoned=1))
            elif which == "ping":
                out_q.put(_status(miner_id, jobs_done))
            elif which == "shutdown":
                grace_ms = cm.shutdown.grace_ms or 5000
                break

            # Soft idle timeout: if the coordinator stalls mid-session.
            idle_s = config.idle_timeout_s if config else 300
            if time.monotonic() - last_activity > idle_s:
                logger.info("idle timeout (%ss) — clean exit", idle_s)
                break

        # Signal end-of-outbound so the server can finish draining Results.
        out_q.put(_STOP)
        # Give the feeder thread a moment to flush (grace_ms).
        time.sleep(min(0.05, grace_ms / 1000.0))
        return exit_code
    except grpc.RpcError as exc:
        code = exc.code() if hasattr(exc, "code") else None
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return EXIT_TOKEN_REJECTED
        logger.exception("rpc error: %s", exc)
        return EXIT_INTERNAL_FATAL
    except Exception:
        logger.exception("session failed")
        return EXIT_INTERNAL_FATAL
    finally:
        try:
            out_q.put(_STOP)
        except Exception:  # noqa: BLE001
            pass
        channel.close()
        sampler.close()


def run_session_sync(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[QPUTimeManager] = None,
) -> int:
    """Sync entry (session is already synchronous)."""
    return run_session(coordinator_uri, miner_id, sampler, budget=budget)
