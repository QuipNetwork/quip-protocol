"""Integration tests for `shared.substrate_client.SubstrateClient`.

These require a live docker-compose chain at `ws://localhost:9944`. The whole
module is skipped if the chain is unreachable, so the suite stays green for
contributors who don't run the chain locally.

Phase 1 only verifies the read paths (head queries, mining_snapshot,
storage). Extrinsic submission is covered by Phase 2's bootstrap tests
because it requires a funded signing account.
"""
from __future__ import annotations

import asyncio
import os
import socket

import pytest

from shared.signer import Sr25519Signer
from shared.substrate_client import SubstrateClient


DEFAULT_URL = os.environ.get("QUIP_SUBSTRATE_URL", "ws://localhost:9944")


def _chain_reachable(url: str) -> bool:
    # Parse `ws://host:port` enough to TCP-probe — the substrate websocket
    # handshake fails fast on a dead port, so this is just a cheap pre-check.
    bare = url.split("://", 1)[1]
    host, _, port_str = bare.partition(":")
    port = int(port_str) if port_str else 9944
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


pytestmark = pytest.mark.skipif(
    not _chain_reachable(DEFAULT_URL),
    reason=f"substrate chain not reachable at {DEFAULT_URL}",
)


@pytest.fixture
async def client():
    c = SubstrateClient(url=DEFAULT_URL)
    await c.connect()
    try:
        yield c
    finally:
        await c.close()


async def test_get_head_returns_32_bytes(client):
    head = await client.get_head()
    assert isinstance(head, bytes)
    assert len(head) == 32


async def test_get_head_advances(client):
    h1 = await client.get_head()
    n1 = await client.get_block_number(at=h1)
    # Wait one slot (6s for Aura on local dev chain) and confirm we see a
    # strictly later block.
    await asyncio.sleep(7)
    h2 = await client.get_head()
    n2 = await client.get_block_number(at=h2)
    assert n2 > n1, f"chain did not advance: n1={n1} n2={n2}"


async def test_get_finalized_head(client):
    finalized = await client.get_finalized_head()
    head = await client.get_head()
    nf = await client.get_block_number(at=finalized)
    nh = await client.get_block_number(at=head)
    assert nf <= nh, "finalized head should not exceed best head"


async def test_mining_snapshot_either_returns_or_none(client):
    head = await client.get_head()
    alice = Sr25519Signer.from_uri("//Alice")
    snapshot = await client.get_mining_snapshot(
        at=head, miner_account_bytes=alice.account_id_bytes()
    )
    # On a fresh dev chain `DefaultTopology` is unset and the runtime API
    # returns None. After Phase 2 bootstrap seeds the chain, this will return
    # a populated context. Both outcomes are correct in Phase 1.
    if snapshot is None:
        return
    assert len(snapshot.parent_hash) == 32
    assert len(snapshot.topology_hash) == 32
    assert snapshot.block_number >= 0
    assert len(snapshot.nodes) > 0
    assert len(snapshot.edges) > 0


async def test_query_balance_returns_int(client):
    """`query_balance` must return a non-negative int for any account
    regardless of whether it's funded. Earlier versions of this test
    asserted `//Alice` had > 10^18 plancks; that assumption broke when
    the hybrid-sig chain merge changed the genesis account endowments
    (no MultiSignature accounts are funded by default — funding happens
    via the hybrid-aware paymaster flow in Phase 7)."""
    alice = Sr25519Signer.from_uri("//Alice")
    balance = await client.query_balance(alice.account_id_bytes())
    assert isinstance(balance, int)
    assert balance >= 0


async def test_query_miner_unregistered_account(client):
    # A fresh account that's never been registered should return None.
    rando = Sr25519Signer.from_seed(b"\x42" * 32)
    miner_info = await client.query_miner(rando.account_id_bytes())
    assert miner_info is None


async def test_query_difficulty_either_returns_or_none(client):
    """`StorageValue<_, DifficultyConfig>` quirks: substrate-interface returns
    the `Default::default()` value (all zeros) when storage is empty rather
    than `None`. `query_difficulty` must honor `meta_info[result_found]`
    so the bootstrap idempotency check stays correct on a fresh chain."""
    difficulty = await client.query_difficulty()
    # On a freshly-built chain `Difficulty` is unset and we expect None. After
    # Phase 2 bootstrap (or any prior sudo set_difficulty) the storage is
    # populated and we expect a real SubstrateDifficulty. Either is correct,
    # but we must never see the all-zeros default-struct case.
    if difficulty is not None:
        # If we got a value, at least one field must be non-zero — otherwise
        # we're back to the "default returned for empty storage" bug.
        assert any([
            difficulty.min_solutions,
            difficulty.max_energy_milli,
            difficulty.min_diversity_milli,
            difficulty.min_quality_milli,
        ]), "query_difficulty returned all-zeros struct; storage is empty"
