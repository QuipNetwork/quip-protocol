"""Tests for ValidatorPool sync-wait: probe-on-failure classification.

A connection-class error on a node whose get_sync_state says
is_syncing=True is 'alive but not ready', not 'down'. See
docs/superpowers/specs/2026-07-03-node-sync-progress-design.md.
"""
from __future__ import annotations

import asyncio
import logging
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


@pytest.mark.asyncio
async def test_single_syncing_url_blocks_until_synced_then_retries():
    """Single URL syncing: send() blocks through sync-wait, then succeeds."""
    sync_states = iter([_syncing(100), _syncing(600), _synced()])
    done = {"synced": False}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            state = next(sync_states)
            if not state["is_syncing"]:
                done["synced"] = True
            return state
        if op == "get_head":
            if not done["synced"]:
                raise TimeoutError("runtime call timed out")
            return b"head"
        raise AssertionError(f"unexpected op {op}")

    pool = _make_pool({"http://a": script_a})
    await pool.start()
    try:
        assert await pool.send("get_head", {}) == b"head"
        # Success clears both the record and the telemetry surface.
        assert pool.last_sync_state is None
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_sync_wait_publishes_progress_for_telemetry():
    """During the wait, last_sync_state carries the latest probe + url."""
    seen_states: list[dict] = []
    sync_states = iter([_syncing(100), _syncing(600), _synced()])
    done = {"synced": False}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            state = next(sync_states)
            if not state["is_syncing"]:
                done["synced"] = True
            return state
        if not done["synced"]:
            raise TimeoutError("runtime call timed out")
        return b"head"

    pool = _make_pool({"http://a": script_a})

    original_publish = pool._publish_sync_state

    def spy(url: str, state: dict) -> None:
        original_publish(url, state)
        seen_states.append(dict(pool.last_sync_state))

    pool._publish_sync_state = spy
    await pool.start()
    try:
        await pool.send("get_head", {})
        blocks = [s["current_block"] for s in seen_states]
        assert 100 in blocks and 600 in blocks
        assert all(s["url"] == "http://a" for s in seen_states)
        assert all("at" in s for s in seen_states)
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_sync_wait_picks_most_advanced_syncing_url():
    """Both URLs syncing → wait against the one with the higher block."""
    done = {"synced": False}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            return _syncing(current=100)
        raise TimeoutError("runtime call timed out")

    b_states = iter([_syncing(900), _synced()])

    def script_b(op: str) -> Any:
        if op == "get_sync_state":
            state = next(b_states)
            if not state["is_syncing"]:
                done["synced"] = True
            return state
        if not done["synced"]:
            raise TimeoutError("runtime call timed out")
        return b"head-b"

    pool = _make_pool({"http://a": script_a, "http://b": script_b})
    await pool.start()
    try:
        assert await pool.send("get_head", {}) == b"head-b"
        assert pool.active_url() == "http://b"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_sync_wait_aborts_when_probe_dies_mid_wait():
    """Node dies mid-sync: sync-wait returns to normal failure handling."""
    probes = {"count": 0}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            probes["count"] += 1
            if probes["count"] == 1:
                return _syncing(100)
            raise ConnectionError("node died mid-sync")
        raise TimeoutError("runtime call timed out")

    pool = _make_pool({"http://a": script_a})
    await pool.start()
    try:
        # all_down + failed sync-wait → the original TimeoutError surfaces,
        # exactly like today's exhausted-retries path.
        with pytest.raises(TimeoutError):
            await pool.send("get_head", {})
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_single_url_transient_blip_reconnects_without_all_down_backoff():
    """Single-URL deployment, healthy node, transient client-socket blip: the
    first spawned handle fails get_head AND the sync probe (looks 'down'); the
    pool must respawn its child on the same URL and retry seamlessly, NOT enter
    the 60s all-down backoff. Before the fix this hangs on the backoff sleep."""
    spawns = {"n": 0}

    def factory(url: str) -> _ScriptedHandle:
        idx = spawns["n"]
        spawns["n"] += 1

        def script(op: str) -> Any:
            if idx == 0:
                raise ConnectionError("client socket dropped")
            return b"head"

        return _ScriptedHandle(url, script)

    failover = SubstrateUrlFailover(
        ["http://a"], initial_backoff_s=60.0, max_backoff_s=60.0
    )
    pool = ValidatorPool(
        urls=["http://a"],
        failover=failover,
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    await pool.start()
    try:
        result = await asyncio.wait_for(pool.send("get_head", {}), timeout=2.0)
        assert result == b"head"
        assert spawns["n"] == 2  # original + one fast reconnect
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_single_url_submit_blip_raises_validator_swapped_fast():
    """Single-URL non-idempotent submit on a transient blip: the pool respawns
    the child fast and raises ValidatorSwapped so the caller re-signs — it must
    NOT stall on the 60s all-down backoff."""
    spawns = {"n": 0}

    def factory(url: str) -> _ScriptedHandle:
        idx = spawns["n"]
        spawns["n"] += 1

        def script(op: str) -> Any:
            if idx == 0:
                raise ConnectionError("client socket dropped")
            return "receipt"

        return _ScriptedHandle(url, script)

    failover = SubstrateUrlFailover(
        ["http://a"], initial_backoff_s=60.0, max_backoff_s=60.0
    )
    pool = ValidatorPool(
        urls=["http://a"],
        failover=failover,
        handle_factory=factory,
        max_swap_retries=3,
        reconnect_backoff_s=0.0,
    )
    await pool.start()
    try:
        with pytest.raises(ValidatorSwapped):
            await asyncio.wait_for(
                pool.send("submit_signed_extrinsic", {"extrinsic_hex": "0xab"}),
                timeout=2.0,
            )
        assert spawns["n"] == 2  # original + one fast reconnect
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_concurrent_senders_serialize_sync_wait(caplog):
    """Two concurrent send() calls on a syncing node: the sync-wait lock
    parks the second caller, and the post-lock short-circuit means only
    the first caller runs a wait loop — one WARNING, bounded probes.

    Traced sequence (single URL "http://a"):
      - Both callers fail get_head; each runs _record_sync_state:
        caller 1 consumes _syncing(100), caller 2 consumes _syncing(100)
        → 2 classification probes.
      - Caller 1 wins _swap_after_failure (identity guard: self._active is
        its captured handle) and swaps; caller 2's handle is now stale, so
        its _swap_after_failure short-circuits (self._active is not its
        handle) without a second swap. Both see _syncing_urls non-empty and
        enter _sync_wait.
      - Caller 1 acquires _sync_wait_lock, logs WARNING "mining paused",
        enters probe loop:
          probe 3: _syncing(600) → logs progress, sleeps poll_interval.
          probe 4: _synced()    → _clear_sync_state(), returns True.
        Lock released; _syncing_urls is now empty.
      - Caller 2 acquires lock, hits short-circuit (not self._syncing_urls)
        → returns True immediately, 0 probes, no WARNING logged.
      - Both retry get_head; done["synced"]=True → both return b"head".
      Total probes: 4. len(paused) == 1 (only caller 1 warned).
    Without the lock both callers enter the loop body and both warn.
    """
    probe_count = {"n": 0}
    sync_states = iter([_syncing(100), _syncing(100), _syncing(600), _synced()])
    done = {"synced": False}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            probe_count["n"] += 1
            try:
                state = next(sync_states)
            except StopIteration:
                state = _synced()
            if not state["is_syncing"]:
                done["synced"] = True
            return state
        if not done["synced"]:
            raise TimeoutError("runtime call timed out")
        return b"head"

    pool = _make_pool({"http://a": script_a})
    await pool.start()
    try:
        with caplog.at_level(logging.WARNING, logger="substrate.pool"):
            results = await asyncio.gather(
                pool.send("get_head", {}), pool.send("get_head", {})
            )
        assert results == [b"head", b"head"]
        paused = [r for r in caplog.records if "mining paused" in r.getMessage()]
        assert len(paused) == 1  # parked caller short-circuits, never warns
        # Expected: 2 classification probes + 2 winner wait-loop probes;
        # parked caller does zero probes (short-circuit). Without the lock
        # the second loop also probes and warns.
        assert probe_count["n"] <= 5
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_stale_syncing_record_cleared_when_node_goes_down(caplog):
    """A syncing record left by a failed submit must not steer a later
    failure into the quiet sync-wait branch once the node stops syncing.

    The down-phase probe answers with a NOT-syncing dict (node process up,
    chain stalled/ops failing) — the exact case where a stale record would
    otherwise take the quiet branch.

    Traced sequence:
      - submit_signed_extrinsic fails → probe returns _syncing(100) →
        _syncing_urls["http://a"] populated → NodeSyncing raised (no swap).
      - phase["down"] = True
      - get_head fails → probe returns _synced(100) (not syncing) →
        _record_sync_state pops the stale entry → returns False →
        normal swap → AllUrlsDown with empty _syncing_urls → single-URL
        reconnect branch → retries exhaust → ConnectionError propagates.
    With _syncing_urls.pop() reverted: stale record survives → quiet
    "no healthy validator" branch → caplog assertion fails.
    """
    phase = {"down": False}

    def script_a(op: str) -> Any:
        if op == "get_sync_state":
            if phase["down"]:
                return _synced(100)  # alive, not syncing — ops still failing
            return _syncing(100)
        if op == "submit_signed_extrinsic":
            raise TimeoutError("runtime call timed out")
        raise ConnectionError("node flapping")

    pool = _make_pool({"http://a": script_a})
    await pool.start()
    try:
        with pytest.raises(NodeSyncing):
            await pool.send("submit_signed_extrinsic", {"extrinsic_hex": "0xab"})
        assert pool._syncing_urls  # stale record from the no-swap path
        phase["down"] = True
        with caplog.at_level(logging.INFO, logger="substrate.pool"):
            with pytest.raises(ConnectionError):
                await pool.send("get_head", {})
        assert pool._syncing_urls == {}
        messages = [r.getMessage() for r in caplog.records]
        # Non-syncing reconnect branch taken (single URL); quiet sync-wait
        # "no healthy validator" branch NOT taken.
        assert any("reconnecting single validator" in m for m in messages)
        assert not any("no healthy validator" in m for m in messages)
    finally:
        await pool.shutdown()
