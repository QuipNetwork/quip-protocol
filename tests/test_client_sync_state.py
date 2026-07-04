"""Tests for SubstrateClient.get_sync_state — node-level sync probe.

The probe wraps two node RPCs that answer fast even during major sync:
system_health (isSyncing, peers) and system_syncState (starting/current/
highest block). The result crosses the validator-child mp.Queue, so it
must be a plain picklable dict.
"""
from __future__ import annotations

import multiprocessing.reduction
from unittest.mock import MagicMock

import pytest

from substrate.client import SubstrateClient


def _client_with_rpc(responses: dict) -> SubstrateClient:
    """Client whose iface answers rpc_request from a canned dict.

    A method mapped to an Exception instance is raised instead —
    mirrors substrate-interface raising on an unsupported RPC.
    """
    client = SubstrateClient("ws://test:9944")
    fake_iface = MagicMock()

    def rpc_request(method: str, params: list):
        value = responses[method]
        if isinstance(value, Exception):
            raise value
        return value

    fake_iface.rpc_request.side_effect = rpc_request
    client._iface = fake_iface
    return client


@pytest.mark.asyncio
async def test_get_sync_state_reports_syncing_node():
    client = _client_with_rpc({
        "system_health": {
            "result": {"peers": 5, "isSyncing": True, "shouldHavePeers": True}
        },
        "system_syncState": {
            "result": {"startingBlock": 100, "currentBlock": 5_000, "highestBlock": 90_000}
        },
    })
    state = await client.get_sync_state()
    assert state == {
        "is_syncing": True,
        "peers": 5,
        "current_block": 5_000,
        "highest_block": 90_000,
        "starting_block": 100,
    }


@pytest.mark.asyncio
async def test_get_sync_state_reports_synced_node():
    client = _client_with_rpc({
        "system_health": {
            "result": {"peers": 8, "isSyncing": False, "shouldHavePeers": True}
        },
        "system_syncState": {
            "result": {"startingBlock": 0, "currentBlock": 90_000, "highestBlock": 90_000}
        },
    })
    state = await client.get_sync_state()
    assert state["is_syncing"] is False
    assert state["current_block"] == 90_000


@pytest.mark.asyncio
async def test_get_sync_state_fails_open_without_syncstate_rpc():
    """A node without system_syncState reports not-syncing (spec: fail open)."""
    client = _client_with_rpc({
        "system_health": {
            "result": {"peers": 5, "isSyncing": True, "shouldHavePeers": True}
        },
        "system_syncState": RuntimeError("Method not found"),
    })
    state = await client.get_sync_state()
    assert state["is_syncing"] is False
    assert state["peers"] == 5
    assert state["current_block"] == 0


@pytest.mark.asyncio
async def test_get_sync_state_result_is_mp_picklable():
    """The dict rides the validator-child mp.Queue — must ForkingPickle."""
    client = _client_with_rpc({
        "system_health": {"result": {"peers": 1, "isSyncing": True}},
        "system_syncState": {
            "result": {"startingBlock": 0, "currentBlock": 1, "highestBlock": 2}
        },
    })
    state = await client.get_sync_state()
    multiprocessing.reduction.ForkingPickler.dumps(state)  # must not raise
