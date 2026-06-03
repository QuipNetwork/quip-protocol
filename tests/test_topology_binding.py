# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unit tests for `SubstrateClient.resolve_topology_binding`.

No live chain: ``get_head`` / ``get_mining_snapshot`` are stubbed. The key
assertion pins the canonical-hash recipe — ``binding.expected_hash`` must equal
an independent ``topology_hash(...)`` call over the same inputs — so the move of
the hash recipe out of the CLI and into the client can't silently diverge from
the chain's ``hash_topology``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.allowed_value_spec import AllowedValueSet
from shared.topology_hash import topology_hash
from substrate.client import (
    NoRegisteredTopology,
    SubstrateClient,
    TopologyBinding,
)
from substrate.types import SubstrateDifficulty, SubstrateMiningContext

_BIN_SPEC = AllowedValueSet((-1000, 1000))
_TER_SPEC = AllowedValueSet((-1000, 0, 1000))
_ACCOUNT = b"\x42" * 32


def _topology(nodes=(0, 1, 2, 3)):
    nodes = list(nodes)
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return SimpleNamespace(nodes=nodes, edges=edges)


def _snapshot(topology, *, chain_hash: bytes) -> SubstrateMiningContext:
    return SubstrateMiningContext(
        last_proof_block_hash=b"\xab" * 32,
        topology_hash=chain_hash,
        nodes=topology.nodes,
        edges=topology.edges,
        difficulty=SubstrateDifficulty(
            min_solutions=1, max_energy_milli=0, min_diversity_milli=0,
        ),
        miner_account_bytes=_ACCOUNT,
        allowed_h_values=_TER_SPEC,
        allowed_j_values=_BIN_SPEC,
        allowed_spin_values=_BIN_SPEC,
        block_hash=b"\x55" * 32,
        block_number=1,
    )


def _client_returning(snapshot):
    client = SubstrateClient(url="ws://stub:9944")
    client.get_head = AsyncMock(return_value=b"\x99" * 32)
    client.get_mining_snapshot = AsyncMock(return_value=snapshot)
    return client


def _expected_hash(topology) -> bytes:
    return topology_hash(
        topology.nodes, topology.edges, _TER_SPEC, _BIN_SPEC, _BIN_SPEC,
    )


async def test_expected_hash_matches_canonical_recipe():
    """binding.expected_hash equals an independent topology_hash() call."""
    topo = _topology()
    snap = _snapshot(topo, chain_hash=b"\x00" * 32)
    binding = await _client_returning(snap).resolve_topology_binding(
        topo, miner_account_bytes=_ACCOUNT,
    )
    assert binding.expected_hash == _expected_hash(topo)
    assert binding.chain_hash == b"\x00" * 32
    assert binding.snapshot is snap


async def test_matches_true_when_chain_hash_agrees():
    topo = _topology()
    snap = _snapshot(topo, chain_hash=_expected_hash(topo))
    binding = await _client_returning(snap).resolve_topology_binding(
        topo, miner_account_bytes=_ACCOUNT,
    )
    assert binding.matches


async def test_matches_false_on_topology_mismatch():
    topo = _topology()
    snap = _snapshot(topo, chain_hash=b"\x11" * 32)
    binding = await _client_returning(snap).resolve_topology_binding(
        topo, miner_account_bytes=_ACCOUNT,
    )
    assert not binding.matches
    assert binding.expected_hash != binding.chain_hash


async def test_raises_when_no_registered_topology():
    client = _client_returning(None)
    with pytest.raises(NoRegisteredTopology):
        await client.resolve_topology_binding(
            _topology(), miner_account_bytes=_ACCOUNT,
        )


def test_binding_matches_property_is_pure():
    """`matches` is a pure equality on the two hashes."""
    assert TopologyBinding(b"\x01" * 32, b"\x01" * 32, None).matches
    assert not TopologyBinding(b"\x01" * 32, b"\x02" * 32, None).matches
