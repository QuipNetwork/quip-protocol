"""Per-validator child process + parent-side proxy.

`validator_main` is the child entry: it owns one `SubstrateClient` and
serves RPC requests from `req_q` on `resp_q`. No polling, no business
logic, no event emission — just request → response, including exceptions.

`ValidatorHandle` is the parent-side proxy: it spawns the child, tracks
in-flight Futures by `request_id`, and exposes an async send-receive API
that `ValidatorPool` consumes. Exception objects raised in the child are
serialized by `mp.Queue` and re-raised in the parent.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import multiprocessing.synchronize
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RpcRequest:
    """One RPC request enqueued from parent → child.

    Attributes:
        request_id: Monotonic id assigned by the parent so the child's
            response can be matched to the awaiting Future.
        op: Method name on the in-child `SubstrateClient`.
        args: Keyword arguments passed to the method.
    """

    request_id: int
    op: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RpcResponse:
    """One RPC response enqueued from child → parent.

    Exactly one of `result` or `exception` is non-None.
    """

    request_id: int
    result: Any = None
    exception: Optional[BaseException] = None


def validator_main(
    url: str,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    shutdown_event: mp.synchronize.Event,
    rpc_call_timeout_s: float,
    _client_factory: Optional[Callable[[str], Any]] = None,
) -> None:
    """Child process entry point.

    Args:
        url: Validator URL this child connects to.
        req_q: Inbound `RpcRequest` queue.
        resp_q: Outbound `RpcResponse` queue.
        shutdown_event: Set by the parent to request graceful exit.
        rpc_call_timeout_s: Reserved for per-call timeout enforcement
            inside the client (wired into `SubstrateClient`'s HTTP
            transport in a follow-up step).
        _client_factory: Test seam — injects a fake client. Production
            code passes None and the real `SubstrateClient` is used.
    """
    if _client_factory is None:
        from substrate.client import SubstrateClient
        client = SubstrateClient(url)
    else:
        client = _client_factory(url)

    logger.info("validator_main started: url=%s pid=%d", url, mp.current_process().pid)
    try:
        while not shutdown_event.is_set():
            try:
                req = req_q.get(timeout=0.1)
            except Exception:
                # Queue.get raises queue.Empty on timeout; loop and check shutdown.
                continue

            if not isinstance(req, RpcRequest):
                logger.warning("validator_main: ignoring non-RpcRequest on req_q: %r", req)
                continue

            try:
                method = getattr(client, req.op)
                result = method(**req.args)
                resp_q.put(RpcResponse(request_id=req.request_id, result=result))
            except BaseException as exc:  # noqa: BLE001
                # Send the exception back; parent decides whether to retry / fail.
                resp_q.put(RpcResponse(request_id=req.request_id, exception=exc))
    finally:
        try:
            client.close()
        except Exception:
            logger.exception("validator_main: client.close() raised during shutdown")
        logger.info("validator_main exiting: url=%s", url)


class ValidatorHandle:
    """Parent-side proxy for a validator child process.

    Spawns a child running `validator_main`, tracks in-flight Futures by
    `request_id`, and exposes an async send-receive API for `ValidatorPool`.
    """

    pass
