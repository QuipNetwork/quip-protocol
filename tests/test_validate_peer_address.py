"""Tests for NetworkNode._validate_peer_address JOIN-path address handling.

Covers the UDP probe gating logic at shared/network_node.py:3959 — most
importantly the ``claimed == fallback`` case, where the peer's claimed
public address matches its QUIC source IP.  In that case we trust QUIC's
source-address validation and return the claimed address without a
second probe; a false-negative probe must not reject the JOIN.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.network_node import NetworkNode


def _make_node() -> NetworkNode:
    """Build a minimal NetworkNode for exercising _validate_peer_address.

    Only the fields the validation path touches are populated.
    """
    node = object.__new__(NetworkNode)
    node.logger = MagicMock()
    return node


@pytest.mark.asyncio
async def test_claimed_reachable_returned_immediately():
    """Happy path: the first probe succeeds, no warning/info emitted."""
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=True)

    result = await node._validate_peer_address(
        claimed="91.245.109.47:20049",
        real_peer_addr="91.245.109.47:54321",
    )

    assert result == "91.245.109.47:20049"
    node._can_reach_address.assert_awaited_once()
    node.logger.warning.assert_not_called()
    node.logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_claim_matches_source_ip_returns_claim_when_probe_fails():
    """When claimed host == real QUIC source host and the probe fails,
    we must accept the claim (QUIC already validated the IP) rather than
    raising ValueError or logging the useless "falling back to X" warning.
    """
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=False)

    result = await node._validate_peer_address(
        claimed="47.85.107.87:20049",
        real_peer_addr="47.85.107.87:34251",
    )

    assert result == "47.85.107.87:20049"
    node._can_reach_address.assert_awaited_once_with(
        "47.85.107.87:20049", 2.0
    )
    node.logger.warning.assert_not_called()
    node.logger.info.assert_called_once()
    msg = node.logger.info.call_args.args[0]
    assert "47.85.107.87:20049" in msg
    assert "QUIC source verification" in msg


@pytest.mark.asyncio
async def test_claim_differs_from_source_falls_back_when_claim_unreachable():
    """NAT / misconfigured --public-host: claim unreachable but fallback
    to the real source IP works.  Warning is emitted once, fallback is
    probed, and the fallback address is returned.
    """
    node = _make_node()
    node._can_reach_address = AsyncMock(side_effect=[False, True])

    result = await node._validate_peer_address(
        claimed="8.8.8.8:20049",
        real_peer_addr="91.245.109.47:54321",
    )

    assert result == "91.245.109.47:20049"
    assert node._can_reach_address.await_count == 2
    node.logger.warning.assert_called_once()
    warn_msg = node.logger.warning.call_args.args[0]
    assert "8.8.8.8:20049" in warn_msg
    assert "91.245.109.47:20049" in warn_msg


@pytest.mark.asyncio
async def test_claim_and_different_fallback_both_unreachable_raises():
    """Both the claim and a genuinely different fallback fail: this is
    the only case where the JOIN should be rejected."""
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=False)

    with pytest.raises(ValueError, match="Cannot reach peer"):
        await node._validate_peer_address(
            claimed="8.8.8.8:20049",
            real_peer_addr="91.245.109.47:54321",
        )

    assert node._can_reach_address.await_count == 2
    node.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_private_claim_rewritten_to_real_ip_still_works():
    """The private-IP rewrite branch is unchanged: a peer claiming a
    private address has it replaced with the connecting IP and that is
    probed before being returned."""
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=True)

    result = await node._validate_peer_address(
        claimed="10.0.0.5:20049",
        real_peer_addr="91.245.109.47:54321",
    )

    assert result == "91.245.109.47:20049"
    node._can_reach_address.assert_awaited_once_with(
        "91.245.109.47:20049", 2.0
    )
    node.logger.info.assert_called_once()
    info_msg = node.logger.info.call_args.args[0]
    assert "private address" in info_msg


@pytest.mark.asyncio
async def test_private_claim_unreachable_real_ip_raises():
    """Private claim with unreachable connecting IP still rejects."""
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=False)

    with pytest.raises(ValueError, match="claimed private address"):
        await node._validate_peer_address(
            claimed="10.0.0.5:20049",
            real_peer_addr="91.245.109.47:54321",
        )


@pytest.mark.asyncio
async def test_ipv6_mapped_ipv4_claim_matches_plain_ipv4_source():
    """IPv6-mapped IPv4 addresses normalize to plain IPv4, so a claim
    written as ``::ffff:47.85.107.87`` and a source ``47.85.107.87``
    are still recognized as the same host (no spurious fallback warning).
    """
    node = _make_node()
    node._can_reach_address = AsyncMock(return_value=False)

    result = await node._validate_peer_address(
        claimed="[::ffff:47.85.107.87]:20049",
        real_peer_addr="47.85.107.87:34251",
    )

    assert result == "47.85.107.87:20049"
    node.logger.warning.assert_not_called()
    node.logger.info.assert_called_once()
