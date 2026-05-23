"""Tests for substrate.validator_handle.

The child process is a narrow RPC server: it reads requests off req_q,
invokes the named method on its SubstrateClient, and puts the result
(or exception) on resp_q. No polling logic in the child.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from substrate.validator_handle import (
    RpcRequest,
    RpcResponse,
    ValidatorHandle,
    validator_main,
)


class _FakeClient:
    """Minimal fake SubstrateClient for child-process tests."""

    def __init__(self, url):
        self.url = url
        self.calls = []

    def get_head(self):
        self.calls.append(("get_head", {}))
        return b"\xab" * 32

    def get_block_number(self, at=None):
        self.calls.append(("get_block_number", {"at": at}))
        return 42

    def query_difficulty(self):
        self.calls.append(("query_difficulty", {}))
        raise RuntimeError("simulated failure")

    def close(self):
        pass


def _run_validator_with_fake_client(req_q, resp_q, shutdown_event, url):
    """Spawn-target wrapper that injects _FakeClient instead of real one."""
    validator_main(
        url=url,
        req_q=req_q,
        resp_q=resp_q,
        shutdown_event=shutdown_event,
        rpc_call_timeout_s=10.0,
        _client_factory=_FakeClient,  # test seam
    )


def test_validator_handle_round_trip_normal_call():
    """A request put on req_q yields the method's result on resp_q."""
    req_q = mp.Queue()
    resp_q = mp.Queue()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=_run_validator_with_fake_client,
        args=(req_q, resp_q, shutdown_event, "http://test"),
    )
    proc.start()
    try:
        req_q.put(RpcRequest(request_id=1, op="get_head", args={}))
        resp = resp_q.get(timeout=5)
        assert isinstance(resp, RpcResponse)
        assert resp.request_id == 1
        assert resp.result == b"\xab" * 32
        assert resp.exception is None
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        assert not proc.is_alive()


def test_validator_handle_propagates_exceptions():
    """If the method raises, the exception is returned on resp_q (not the result)."""
    req_q = mp.Queue()
    resp_q = mp.Queue()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=_run_validator_with_fake_client,
        args=(req_q, resp_q, shutdown_event, "http://test"),
    )
    proc.start()
    try:
        req_q.put(RpcRequest(request_id=7, op="query_difficulty", args={}))
        resp = resp_q.get(timeout=5)
        assert resp.request_id == 7
        assert resp.result is None
        assert isinstance(resp.exception, RuntimeError)
        assert "simulated failure" in str(resp.exception)
    finally:
        shutdown_event.set()
        proc.join(timeout=5)


def test_validator_handle_shutdown_event_clean_exit():
    """Setting shutdown_event causes the child to exit cleanly."""
    req_q = mp.Queue()
    resp_q = mp.Queue()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=_run_validator_with_fake_client,
        args=(req_q, resp_q, shutdown_event, "http://test"),
    )
    proc.start()
    try:
        shutdown_event.set()
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_validator_handle_multiple_requests_ordered():
    """Requests are processed in order; responses come back in matching order."""
    req_q = mp.Queue()
    resp_q = mp.Queue()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=_run_validator_with_fake_client,
        args=(req_q, resp_q, shutdown_event, "http://test"),
    )
    proc.start()
    try:
        for i in range(1, 6):
            req_q.put(RpcRequest(request_id=i, op="get_block_number", args={}))
        ids_received = []
        for _ in range(5):
            resp = resp_q.get(timeout=5)
            ids_received.append(resp.request_id)
        assert ids_received == [1, 2, 3, 4, 5]
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
