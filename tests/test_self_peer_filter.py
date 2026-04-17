"""Tests for the ``_filter_self_from_peers`` helper on NetworkNode.

A node that lists its own public address in ``initial_peers`` will
try to JOIN itself, fail validation, and pollute its local ban list
with its own loopback address. The filter strips self-references at
config load time and again after public-IP auto-detection.
"""

from unittest.mock import MagicMock

import pytest


def _make_shell(public_host: str):
    """Build a minimal NetworkNode with only the attributes the filter reads."""
    from shared.network_node import NetworkNode

    node = object.__new__(NetworkNode)
    node.public_host = public_host
    node.logger = MagicMock()
    return node


# ---------------------------------------------------------------------------
# _is_self_address
# ---------------------------------------------------------------------------


def test_is_self_address_exact_match_ipv4():
    node = _make_shell("127.0.0.1:20049")
    assert node._is_self_address("127.0.0.1:20049") is True


def test_is_self_address_different_port_is_not_self():
    node = _make_shell("127.0.0.1:20049")
    assert node._is_self_address("127.0.0.1:20050") is False


def test_is_self_address_different_host_is_not_self():
    node = _make_shell("127.0.0.1:20049")
    assert node._is_self_address("10.0.0.5:20049") is False


def test_is_self_address_ipv6_bracketed_match():
    node = _make_shell("[::1]:20049")
    assert node._is_self_address("[::1]:20049") is True


def test_is_self_address_ipv6_bare_no_port_matches_default():
    # ``parse_host_port`` fills in the default port when none is given,
    # so "::1" maps to ("::1", 20049) which equals the shell's self.
    node = _make_shell("[::1]:20049")
    assert node._is_self_address("::1") is True


def test_is_self_address_hostname_case_insensitive():
    node = _make_shell("QPU-1.Nodes.Quip.Network:20049")
    assert node._is_self_address("qpu-1.nodes.quip.network:20049") is True


def test_is_self_address_malformed_input_returns_false():
    node = _make_shell("127.0.0.1:20049")
    assert node._is_self_address("not-an-address-at-all") is False
    assert node._is_self_address("") is False


# ---------------------------------------------------------------------------
# _filter_self_from_peers
# ---------------------------------------------------------------------------


def test_filter_drops_self_and_keeps_others():
    node = _make_shell("127.0.0.1:20049")
    peers = [
        "127.0.0.1:20049",  # self — drop
        "127.0.0.1:20050",  # different port — keep
        "10.0.0.5:20049",   # different host — keep
    ]
    assert node._filter_self_from_peers(peers) == [
        "127.0.0.1:20050",
        "10.0.0.5:20049",
    ]


def test_filter_empty_input_returns_empty():
    node = _make_shell("127.0.0.1:20049")
    assert node._filter_self_from_peers([]) == []


def test_filter_no_self_returns_input_unchanged():
    node = _make_shell("127.0.0.1:20049")
    peers = ["10.0.0.5:20049", "10.0.0.6:20050"]
    assert node._filter_self_from_peers(peers) == peers


def test_filter_drops_multiple_self_entries():
    node = _make_shell("127.0.0.1:20049")
    peers = [
        "127.0.0.1:20049",
        "10.0.0.5:20049",
        "127.0.0.1:20049",  # listed twice
    ]
    assert node._filter_self_from_peers(peers) == ["10.0.0.5:20049"]


def test_filter_logs_when_entries_are_dropped():
    node = _make_shell("127.0.0.1:20049")
    node._filter_self_from_peers(["127.0.0.1:20049", "10.0.0.5:20049"])
    node.logger.info.assert_called_once()
    args, _ = node.logger.info.call_args
    assert "Filtered 1 self-reference" in args[0]
    assert "127.0.0.1:20049" in args[0]


def test_filter_is_silent_when_nothing_dropped():
    node = _make_shell("127.0.0.1:20049")
    node._filter_self_from_peers(["10.0.0.5:20049"])
    node.logger.info.assert_not_called()


def test_filter_preserves_malformed_peer_entries():
    """Bad entries shouldn't crash the filter or get silently eaten."""
    node = _make_shell("127.0.0.1:20049")
    peers = ["garbage", "127.0.0.1:20049", "10.0.0.5:20049"]
    result = node._filter_self_from_peers(peers)
    assert "garbage" in result
    assert "10.0.0.5:20049" in result
    assert "127.0.0.1:20049" not in result
