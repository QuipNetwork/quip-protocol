"""Handler-level integration test for fork-aware sync.

Wires a real ``BlockSynchronizer`` to two NetworkNode shells through an
in-memory ``NodeClient`` that routes sync requests to the real handlers.
Payloads go through the real ``sync_messages`` codecs and
``Block.to_network`` / ``from_network`` — so this test fails if anything
along the serialization path regresses.

Full PoW and signature validation are out of scope: the commit consumer
grants every future, standing in for a node that has already validated
every block it receives.
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
    """Finalized ``Block`` with placeholder sig + PoW (never validated)."""
    block = Block(
        header=BlockHeader(
            previous_hash=previous_hash,
            index=index,
            timestamp=timestamp,
            data_hash=b"\x00" * 32,
        ),
        miner_info=MinerInfo(
            miner_id=f"m-{index}",
            miner_type="CPU",
            reward_address=b"R" * 32,
            ecdsa_public_key=b"E" * 64,
            wots_public_key=b"W" * 64,
            next_wots_public_key=b"N" * 64,
        ),
        quantum_proof=QuantumProof(
            nonce=index, salt=b"S",
            solutions=[[1]], mining_time=0.0,
            nodes=[0], edges=[],
        ),
        next_block_requirements=BlockRequirements(
            difficulty_energy=0.0,
            min_diversity=0.0,
            min_solutions=1,
            timeout_to_difficulty_adjustment_decay=3600,
        ),
        data=b"",
    )
    block.finalize()
    block.signature = b"\x00" * 64
    return block


def _build_chain(base_hash: bytes, length: int, base_index: int, base_timestamp: int) -> List[Block]:
    chain: List[Block] = []
    parent = base_hash
    for i in range(length):
        blk = _build_block(base_index + i + 1, parent, base_timestamp + i + 1)
        chain.append(blk)
        parent = blk.hash
    return chain


def _make_peer_shell(chain: List[Block]):
    from shared.network_node import NetworkNode
    node = object.__new__(NetworkNode)
    node.chain = list(chain)
    node.chain_by_hash = {b.hash: b for b in chain}
    node.logger = logging.getLogger("test.peer")
    return node


class _InMemoryHandlerClient:
    """Routes sync requests to real handler methods via real codecs."""

    def __init__(self, peers: Dict[str, object]):
        self.peers = {host: {} for host in peers}
        self._shells = peers
        self.node_timeout = 5.0
        self.logger = logging.getLogger("test.client")

    async def get_peer_block(self, host: str, block_number: int = 0) -> Optional[Block]:
        shell = self._shells.get(host)
        if shell is None:
            return None
        return shell.chain[-1] if block_number == 0 else shell.chain[block_number]

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
        return decode_manifest_response(resp.payload)

    async def get_peer_block_by_hash(self, host: str, block_hash: bytes) -> Optional[Block]:
        shell = self._shells.get(host)
        if shell is None:
            return None
        req = QuicMessage(
            msg_type=QuicMessageType.BLOCK_BY_HASH_REQUEST,
            request_id=1,
            payload=encode_block_by_hash_request(block_hash),
        )
        resp = await shell._quic_handle_block_by_hash_request(req)
        return decode_block_by_hash_response(resp.payload)


async def _drain_commit_queue(queue: asyncio.Queue, committed: List[Block]) -> None:
    while True:
        block, future, _force_reorg, _source = await queue.get()
        committed.append(block)
        future.set_result(True)


@pytest.mark.asyncio
async def test_sync_picks_longest_fork_end_to_end():
    """Two peers on divergent forks; longer chain wins with zero contamination."""
    genesis = _build_block(0, b"\x00" * 32, timestamp=1000)
    shared = _build_chain(genesis.hash, length=5, base_index=0, base_timestamp=1000)
    branch = shared[-1]
    fork_a = _build_chain(branch.hash, length=10, base_index=5, base_timestamp=2000)
    fork_b = _build_chain(branch.hash, length=15, base_index=5, base_timestamp=3000)  # longer

    peer_a = _make_peer_shell([genesis] + shared + fork_a)
    peer_b = _make_peer_shell([genesis] + shared + fork_b)

    local_chain = [genesis] + shared
    local_by_hash = {b.hash: b for b in local_chain}
    client = _InMemoryHandlerClient(peers={"A": peer_a, "B": peer_b})

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

    assert result.success
    assert result.target_hash == fork_b[-1].hash
    assert result.committed == 15
    assert [b.header.index for b in committed] == list(range(6, 21))
    fork_a_hashes = {b.hash for b in fork_a}
    assert not any(b.hash in fork_a_hashes for b in committed)
