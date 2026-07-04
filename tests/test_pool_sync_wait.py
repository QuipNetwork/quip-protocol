"""Tests for ValidatorPool sync-wait: probe-on-failure classification.

A connection-class error on a node whose get_sync_state says
is_syncing=True is 'alive but not ready', not 'down'. See
docs/superpowers/specs/2026-07-03-node-sync-progress-design.md.
"""
from __future__ import annotations

from typing import Any, Callable

import pytest

from substrate.pool import NodeSyncing, ValidatorPool, ValidatorSwapped
from substrate.url_failover import SubstrateUrlFailover


def _syncing(current: int = 100, highest: int = 1_000, peers: int = 3) -> dict:
    return {
        "is_syncing": True,
        "peers": peers,
        "current_block": current,
        "highest_block": highest,
        "starting_block": 0,
    }


def _synced(block: int = 1_000) -> dict:
    return {
        "is_syncing": False,
        "peers": 3,
        "current_block": block,
        "highest_block": block,
        "starting_block": 0,
    }


class _ScriptedHandle:
    """Fake ValidatorHandle whose responses come from a script callable.

    The script receives the op name and returns a value or raises.
    Fresh instances are created per spawn (unlike test_validator_pool's
    singleton fakes) because sync-wait respawns handles on the same URL.
    """

    def __init__(self, url: str, script: Callable[[str], Any]) -> None:
        self.url = url
        self.is_shutdown = False
        self._script = script

    def start(self) -> None:
        pass

    async def send(self, op: str, args: dict) -> Any:
        if self.is_shutdown:
            raise ValidatorSwapped(f"{self.url} already shut down")
        return self._script(op)

    async def shutdown(self) -> None:
        self.is_shutdown = True


def _make_pool(scripts: dict[str, Callable[[str], Any]]) -> ValidatorPool:
    """Pool whose handle_factory builds a fresh _ScriptedHandle per spawn."""
    urls = list(scripts)
    failover = SubstrateUrlFailover(urls, initial_backoff_s=0.01, max_backoff_s=0.05)
    return ValidatorPool(
        urls=urls,
        failover=failover,
        handle_factory=lambda url: _ScriptedHandle(url, scripts[url]),
        max_swap_retries=3,
        sync_poll_interval_s=0.01,
    )


@pytest.mark.asyncio
async def test_submit_on_syncing_node_raises_node_syncing_without_swap():
    """Non-idempotent op + syncing node → NodeSyncing immediately, no swap."""
    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            return _syncing()
        raise TimeoutError("runtime call timed out")

    pool = _make_pool({"http://a": script_a, "http://b": lambda op: "b-ok"})
    await pool.start()
    try:
        with pytest.raises(NodeSyncing):
            await pool.send("submit_signed_extrinsic", {"extrinsic_hex": "0xab"})
        # NodeSyncing subclasses ValidatorSwapped so existing callers work.
        assert issubclass(NodeSyncing, ValidatorSwapped)
        # No swap happened — A stays active for the sync-wait to come.
        assert pool.active_url() == "http://a"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_idempotent_op_prefers_healthy_url_over_syncing_one():
    """A syncing, B healthy → swap to B and answer from there ('try others first')."""
    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            return _syncing()
        raise TimeoutError("runtime call timed out")

    pool = _make_pool({"http://a": script_a, "http://b": lambda op: "b-ok"})
    await pool.start()
    try:
        assert await pool.send("get_head", {}) == "b-ok"
        assert pool.active_url() == "http://b"
        # Success clears the telemetry surface.
        assert pool.last_sync_state is None
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_probe_failure_keeps_existing_down_behavior():
    """Probe dies too → node is genuinely down → today's retry-then-raise."""
    def dead(op: str) -> Any:
        raise ConnectionError("dead")

    pool = _make_pool({"http://a": dead, "http://b": dead})
    await pool.start()
    try:
        with pytest.raises(ConnectionError):
            await pool.send("get_head", {})
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_non_connection_errors_still_pass_through_unprobed():
    """RuntimeError from the chain is not a reason to probe or swap."""
    calls: list[str] = []

    def script_a(op: str) -> Any:
        calls.append(op)
        raise RuntimeError("chain returned None")

    pool = _make_pool({"http://a": script_a})
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="chain returned None"):
            await pool.send("get_head", {})
        assert calls == ["get_head"]  # no get_sync_state probe issued
        assert pool.active_url() == "http://a"
    finally:
        await pool.shutdown()
