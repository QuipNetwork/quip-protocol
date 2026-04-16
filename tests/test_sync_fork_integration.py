"""Handler-level integration test for fork-aware sync.

Wires a real ``BlockSynchronizer`` against two real NetworkNode shells
through an in-memory ``NodeClient`` that routes requests to the
shells' real ``_quic_handle_chain_manifest_request`` and
``_quic_handle_block_by_hash_request`` handlers. Payloads cross the
boundary through real ``sync_messages`` codecs and real
``Block.to_network`` / ``from_network``, so the serialization layer
is in the hot path — something the unit tests with fake-block
stand-ins cannot cover.

The test deliberately stops short of exercising
``Node.receive_block``'s full signature + quantum-proof validation;
that's the miner's job and requires a real PoW solution. Instead a
test-local consumer drains the commit queue, records the block
order, and completes each future with True — standing in for a node
that has already validated every block it commits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import pytest

from shared.block import Block, BlockHeader, MinerInfo, QuantumProof
from shared.block_requirements import BlockRequirements
from shared.block_synchronizer import BlockSynchronizer
from shared.node_client import QuicMessage, QuicMessageType
from shared.sync_messages import (
    decode_block_by_hash_response,
    decode_manifest_response,
    encode_block_by_hash_request,
    encode_manifest_request,
)


def _build_block(index: int, previous_hash: bytes, timestamp: int) -> Block:
    """Build a Block with computed raw/hash and a non-empty signature.

    The signature and quantum proof are placeholders — this test never
    flows through ``check_block``'s PoW/signature verification.
    """
    header = BlockHeader(
        previous_hash=previous_hash,
        index=index,
        timestamp=timestamp,
        data_hash=b"\x00" * 32,  # overwritten by finalize()
    )
    miner_info = MinerInfo(
        miner_id=f"test-miner-{index}",
        miner_type="CPU",
        reward_address=b"R" * 32,
        ecdsa_public_key=b"E" * 64,
        wots_public_key=b"W" * 64,
        next_wots_public_key=b"N" * 64,
    )
    quantum_proof = QuantumProof(
        nonce=index,
        salt=b"S",
        solutions=[[1]],
        mining_time=0.0,
        nodes=[0],
        edges=[],
    )
    req = BlockRequirements(
        difficulty_energy=0.0,
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=3600,
    )
    block = Block(
        header=header,
        miner_info=miner_info,
        quantum_proof=quantum_proof,
        next_block_requirements=req,
        data=b"",
    )
    block.finalize()
    block.signature = b"\x00" * 64
    return block


def _build_chain(
    base_hash: bytes,
    length: int,
    base_index: int,
    base_timestamp: int,
) -> List[Block]:
    """Build a linked chain of ``length`` blocks starting at ``base_index + 1``."""
    chain: List[Block] = []
    parent_hash = base_hash
    for i in range(length):
        blk = _build_block(
            index=base_index + i + 1,
            previous_hash=parent_hash,
            timestamp=base_timestamp + i + 1,
        )
        chain.append(blk)
        parent_hash = blk.hash
    return chain


def _make_peer_shell(chain: List[Block]):
    """A minimal NetworkNode whose handlers read from the given chain."""
    from shared.network_node import NetworkNode
    node = object.__new__(NetworkNode)
    node.chain = list(chain)
    node.chain_by_hash = {b.hash: b for b in chain}
    node.logger = logging.getLogger("test.peer")
    return node


class _InMemoryHandlerClient:
    """Route sync requests to real handler methods via real codecs.

    The serialized payload travels through ``encode_*``/``decode_*``
    just like it would on the wire, so this test catches Block
    round-trip bugs that pure-mock tests miss.
    """

    def __init__(self, peers: Dict[str, object]):
        self.peers = {host: {} for host in peers}
        self._shells = peers
        self.node_timeout = 5.0
        self.logger = logging.getLogger("test.client")

    async def get_peer_block(
        self, host: str, block_number: int = 0
    ) -> Optional[Block]:
        shell = self._shells.get(host)
        if shell is None:
            return None
        if block_number == 0:
            return shell.chain[-1]
        if block_number < len(shell.chain):
            return shell.chain[block_number]
        return None

    async def get_chain_manifest(self, host, locator, limit):
        shell = self._shells.get(host)
        if shell is None:
            return None
        req = QuicMessage(
            msg_type=QuicMessageType.CHAIN_MANIFEST_REQUEST,
            request_id=1,
            payload=encode_manifest_request(locator, limit),
        )
        resp = await shell._quic_handle_chain_manifest_request(req)
        if resp.msg_type != QuicMessageType.CHAIN_MANIFEST_RESPONSE:
            return None
        return decode_manifest_response(resp.payload)

    async def get_peer_block_by_hash(
        self, host: str, block_hash: bytes
    ) -> Optional[Block]:
        shell = self._shells.get(host)
        if shell is None:
            return None
        req = QuicMessage(
            msg_type=QuicMessageType.BLOCK_BY_HASH_REQUEST,
            request_id=1,
            payload=encode_block_by_hash_request(block_hash),
        )
        resp = await shell._quic_handle_block_by_hash_request(req)
        if resp.msg_type != QuicMessageType.BLOCK_BY_HASH_RESPONSE:
            return None
        return decode_block_by_hash_response(resp.payload)


async def _drain_commit_queue(
    queue: asyncio.Queue, committed: List[Block]
) -> None:
    """Test-local stand-in for the real block processor loop."""
    while True:
        block, future, force_reorg, source = await queue.get()
        committed.append(block)
        future.set_result(True)


@pytest.mark.asyncio
async def test_sync_picks_longest_fork_end_to_end():
    """Two peers on divergent forks; the longer chain wins and contaminates nothing."""
    # Shared genesis + 5-block shared prefix.
    genesis = _build_block(0, previous_hash=b"\x00" * 32, timestamp=1000)
    shared_prefix = _build_chain(
        base_hash=genesis.hash,
        length=5,
        base_index=0,
        base_timestamp=1000,
    )
    branch_point = shared_prefix[-1]

    fork_a = _build_chain(
        base_hash=branch_point.hash,
        length=15,  # heights 6..20
        base_index=5,
        base_timestamp=2000,
    )
    fork_b = _build_chain(
        base_hash=branch_point.hash,
        length=20,  # heights 6..25 — longer
        base_index=5,
        base_timestamp=3000,
    )

    peer_a_chain = [genesis] + shared_prefix + fork_a
    peer_b_chain = [genesis] + shared_prefix + fork_b

    peer_a = _make_peer_shell(peer_a_chain)
    peer_b = _make_peer_shell(peer_b_chain)

    # Syncing node knows only the shared prefix.
    local_chain = [genesis] + shared_prefix
    local_by_hash = {b.hash: b for b in local_chain}

    client = _InMemoryHandlerClient(peers={"A:1": peer_a, "B:1": peer_b})

    queue: asyncio.Queue = asyncio.Queue()
    committed: List[Block] = []

    sync = BlockSynchronizer(
        node_client=client,
        receive_block_queue=queue,
        local_tip=lambda: local_chain[-1],
        local_locator=lambda: [b.hash for b in reversed(local_chain)],
        local_get_block_by_hash=local_by_hash.get,
        max_in_flight=8,
    )

    processor = asyncio.create_task(_drain_commit_queue(queue, committed))
    try:
        result = await sync.sync_blocks()
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass

    assert result.success is True
    assert result.target_hash == fork_b[-1].hash
    assert result.target_height == 25
    assert result.committed == 20

    # Blocks arrived in ascending index order.
    commit_indices = [b.header.index for b in committed]
    assert commit_indices == list(range(6, 26))

    # Every committed block is from fork B.
    fork_b_by_index = {b.header.index: b for b in fork_b}
    for got in committed:
        expected = fork_b_by_index[got.header.index]
        assert got.hash == expected.hash
        assert got.header.previous_hash == expected.header.previous_hash

    # No block from fork A leaked in.
    fork_a_hashes = {b.hash for b in fork_a}
    for got in committed:
        assert got.hash not in fork_a_hashes
