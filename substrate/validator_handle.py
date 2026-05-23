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
import multiprocessing.reduction
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _is_picklable(obj: object) -> bool:
    """Return True if *obj* can be serialised by the multiprocessing queue.

    mp.Queue uses ``multiprocessing.reduction.ForkingPickler`` internally, so
    we probe with the same pickler rather than the plain ``pickle`` module.
    """
    try:
        multiprocessing.reduction.ForkingPickler.dumps(obj)
        return True
    except Exception:
        return False


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
            except queue.Empty:
                # Loop back and check shutdown_event.
                continue

            if not isinstance(req, RpcRequest):
                logger.warning("validator_main: ignoring non-RpcRequest on req_q: %r", req)
                continue

            try:
                method = getattr(client, req.op)
                result = method(**req.args)
            except BaseException as exc:  # noqa: BLE001
                # Eagerly test picklability: mp.Queue serialises in a background
                # thread, so a try/except around put() won't catch the error.
                exc_to_send: BaseException = exc
                if not _is_picklable(exc):
                    exc_to_send = RuntimeError(f"{type(exc).__name__}: {exc}")
                resp_q.put(RpcResponse(request_id=req.request_id, exception=exc_to_send))
                continue
            # Guard against unpicklable results too.
            if not _is_picklable(result):
                safe = RuntimeError(
                    f"result for op={req.op!r} is not picklable"
                )
                resp_q.put(RpcResponse(request_id=req.request_id, exception=safe))
            else:
                resp_q.put(RpcResponse(request_id=req.request_id, result=result))
    finally:
        try:
            client.close()
        except Exception:
            logger.exception("validator_main: client.close() raised during shutdown")
        logger.info("validator_main exiting: url=%s", url)


class ValidatorSwapped(Exception):
    """In-flight RPC was cancelled because the handle is being shut down or swapped."""


class ValidatorHandle:
    """Parent-side proxy for one validator child process.

    Args:
        url: Validator URL the child will connect to.
        client_factory: Optional test seam; passed to `validator_main`.
        rpc_call_timeout_s: Per-RPC timeout (forwarded to child).
        req_q_size: Bound on the request queue. Default 64.
    """

    def __init__(
        self,
        url: str,
        *,
        client_factory: Optional[Callable[[str], Any]] = None,
        rpc_call_timeout_s: float = 10.0,
        req_q_size: int = 64,
    ) -> None:
        self.url = url
        self._client_factory = client_factory
        self._rpc_call_timeout_s = rpc_call_timeout_s
        self._req_q: mp.Queue = mp.Queue(maxsize=req_q_size)
        self._resp_q: mp.Queue = mp.Queue()
        self._shutdown_event = mp.Event()
        self._proc: Optional[mp.Process] = None
        self._next_request_id = 1
        self._inflight: dict[int, asyncio.Future] = {}
        self._drainer_task: Optional[asyncio.Task] = None
        self.is_shutdown = False

    def start(self) -> None:
        """Spawn the child process and start the response drainer."""
        self._proc = mp.Process(
            target=validator_main,
            args=(
                self.url,
                self._req_q,
                self._resp_q,
                self._shutdown_event,
                self._rpc_call_timeout_s,
                self._client_factory,
            ),
        )
        self._proc.start()
        loop = asyncio.get_running_loop()
        self._drainer_task = loop.create_task(self._drain_responses(loop))

    async def send(self, op: str, args: dict[str, Any]) -> Any:
        """Send an RPC request to the child and await the response.

        Raises:
            ValidatorSwapped: if `shutdown()` is called while this
                request is in flight.
            Anything the underlying client method raised.
        """
        if self.is_shutdown:
            raise ValidatorSwapped(f"handle for {self.url} is already shut down")
        request_id = self._next_request_id
        self._next_request_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._inflight[request_id] = fut
        await loop.run_in_executor(
            None,
            self._req_q.put,
            RpcRequest(request_id=request_id, op=op, args=args),
        )
        return await fut

    async def shutdown(self) -> None:
        """Signal shutdown, cancel in-flight Futures, await child exit."""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        self._shutdown_event.set()
        # Cancel every in-flight future with ValidatorSwapped.
        for fut in list(self._inflight.values()):
            if not fut.done():
                fut.set_exception(ValidatorSwapped(f"handle for {self.url} shutting down"))
        self._inflight.clear()
        # Wait for child exit (with timeout).
        if self._proc is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._proc.join, 5.0)
            if self._proc.is_alive():
                logger.warning("ValidatorHandle %s: child did not exit; terminating", self.url)
                self._proc.terminate()
                await loop.run_in_executor(None, self._proc.join, 2.0)
        if self._drainer_task is not None:
            self._drainer_task.cancel()
            try:
                await self._drainer_task
            except asyncio.CancelledError:
                pass

    async def _drain_responses(self, loop: asyncio.AbstractEventLoop) -> None:
        """Pull `RpcResponse`s off resp_q and resolve matching Futures."""
        while not self.is_shutdown:
            try:
                resp = await loop.run_in_executor(None, self._resp_q.get, True, 0.5)
            except Exception:
                # queue.Empty after timeout; loop.
                continue
            if not isinstance(resp, RpcResponse):
                logger.warning("ValidatorHandle %s: ignoring non-RpcResponse: %r", self.url, resp)
                continue
            fut = self._inflight.pop(resp.request_id, None)
            if fut is None:
                continue  # response for an already-cancelled request
            if fut.done():
                continue
            if resp.exception is not None:
                fut.set_exception(resp.exception)
            else:
                fut.set_result(resp.result)
