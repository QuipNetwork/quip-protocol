"""Unit tests for `shared.miner_bootstrap`.

These don't need a live chain — they stub `SubstrateClient` and exercise the
state machine directly. The highest-value coverage is the `_assert_dev_chain`
guard (without it, `--seed-chain` can sudo a production chain) and the
`_build_seed_topology` invariants the Rust pallet's
`validate_topology_consistency` relies on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.miner_bootstrap import (
    DEV_CHAIN_PREFIXES,
    _assert_dev_chain,
    _build_seed_topology,
)


# ---------------------------------------------------------------------------
# _assert_dev_chain
# ---------------------------------------------------------------------------


class _StubClient:
    """Minimal stand-in for SubstrateClient that just carries a chain name."""

    def __init__(self, chain_name: str) -> None:
        self._iface = SimpleNamespace(chain=chain_name)

    async def _run(self, fn):
        # Bootstrap's `_assert_dev_chain` calls `client._run(lambda: ...)`.
        # The real client offloads to an executor; the stub just calls
        # synchronously since the underlying op is property access.
        return fn()


@pytest.mark.parametrize("name", [
    "Development",
    "Local Testnet",
    "Local Testnet (3 Validators)",
    "quip-local",
    "quip-local-dev",
])
async def test_assert_dev_chain_accepts_dev_prefixes(name):
    # Must not raise — the docstring of bootstrap promises these prefixes.
    await _assert_dev_chain(_StubClient(name))


@pytest.mark.parametrize("name", [
    "Quip",
    "Quip Production",
    "Polkadot",
    "Kusama",
    "Westend",
    "",
])
async def test_assert_dev_chain_rejects_non_dev_chains(name):
    with pytest.raises(RuntimeError, match="non-dev chain"):
        await _assert_dev_chain(_StubClient(name))


def test_dev_chain_prefixes_constant_matches_faucet():
    # The bootstrap and the faucet bot intentionally keep duplicate copies
    # of this list (so the faucet stays standalone). Pin the values here so
    # a drift on either side gets flagged by tests on both sides.
    assert DEV_CHAIN_PREFIXES == ("Development", "Local Testnet", "quip-local")


# ---------------------------------------------------------------------------
# _build_seed_topology
# ---------------------------------------------------------------------------


def test_build_seed_topology_returns_unique_sorted_ids():
    # Whatever labeling the helper picks (dense 0..n-1 in Phase 2, raw
    # zephyr int-labels in Phase 4 after the relabel-bug fix), it must
    # emit unique ascending non-negative ints.
    nodes, edges = _build_seed_topology((2, 2))
    assert len(nodes) > 0
    assert nodes == sorted(nodes)
    assert len(set(nodes)) == len(nodes)
    assert all(isinstance(n, int) and n >= 0 for n in nodes)


def test_build_seed_topology_edges_are_canonical_and_sorted():
    nodes, edges = _build_seed_topology((2, 2))
    # u < v on every edge
    assert all(u < v for u, v in edges), "edges must satisfy u < v"
    # globally sorted
    assert edges == sorted(edges), "edges must be sorted"
    # no self-loops, no duplicates
    assert len({(u, v) for u, v in edges}) == len(edges)
    assert all(u != v for u, v in edges)


def test_build_seed_topology_is_deterministic():
    a_nodes, a_edges = _build_seed_topology((2, 2))
    b_nodes, b_edges = _build_seed_topology((2, 2))
    assert a_nodes == b_nodes
    assert a_edges == b_edges


def test_build_seed_topology_edges_reference_only_emitted_nodes():
    nodes, edges = _build_seed_topology((2, 2))
    node_set = set(nodes)
    for u, v in edges:
        assert u in node_set and v in node_set
