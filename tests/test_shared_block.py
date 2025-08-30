from __future__ import annotations

from blake3 import blake3
import time

from shared.block import (
    QuantumProof,
    MinerInfo,
    BlockHeader,
    NextBlockRequirements,
    Block,
)


def sample_quantum_proof():
    return QuantumProof(
        nonce=123456789,
        solutions=[[1, -1, 0], [0, 1, -1], [-5], [3]],
        mining_time=1.234,
        node_list=[0, 1, 2, 3],
        edge_list=[(0, 1), (1, 2), (2, 3)],
    )


def sample_miner_info():
    return MinerInfo(
        miner_id="node1-CPU-1",
        miner_type="CPU",
        reward_address=b"R" * 32,
        ecdsa_public_key=b"E" * 64,
        wots_public_key=b"W" * 64,
        next_wots_public_key=b"N" * 64,
    )


def sample_requirements():
    return NextBlockRequirements(
        difficulty_energy=-20.0,
        min_diversity=0.5,
        min_solutions=2,
        timeout_to_difficulty_adjustment_decay=10,
    )


def sample_header(data_hash: bytes | None = None):
    # Block.from_network currently validates header.data_hash against an empty raw slice.
    # Set data_hash to blake3(b"").digest() for compatibility.
    if data_hash is None:
        data_hash = blake3(b"").digest()
    return BlockHeader(
        previous_hash=b"\x00" * 32,
        index=1,
        timestamp=int(time.time()),
        data_hash=data_hash,
    )


def test_quantum_proof_network_roundtrip():
    qp = sample_quantum_proof()
    data = qp.to_network()
    qp2 = QuantumProof.from_network(data)
    assert qp2.nonce == qp.nonce
    assert qp2.mining_time == qp.mining_time
    assert qp2.solutions == qp.solutions


def test_quantum_proof_compute_derived_fields():
    qp = sample_quantum_proof()
    req = sample_requirements()
    blk = make_sample_block()
    qp.compute_derived_fields(req, blk)
    assert isinstance(qp.energy, float)
    assert isinstance(qp.num_valid_solutions, int)
    assert isinstance(qp.diversity, float)


def test_miner_info_network_roundtrip():
    mi = sample_miner_info()
    data = mi.to_network()
    mi2 = MinerInfo.from_network(data)
    assert mi2.miner_id == mi.miner_id
    assert mi2.miner_type == mi.miner_type
    assert mi2.reward_address == mi.reward_address
    assert mi2.ecdsa_public_key == mi.ecdsa_public_key
    assert mi2.wots_public_key == mi.wots_public_key
    assert mi2.next_wots_public_key == mi.next_wots_public_key


def test_block_header_network_roundtrip():
    hdr = sample_header()
    data = hdr.to_network()
    hdr2 = BlockHeader.from_network(data)
    assert hdr2.previous_hash == hdr.previous_hash
    assert hdr2.index == hdr.index
    assert hdr2.timestamp == hdr.timestamp
    assert hdr2.data_hash == hdr.data_hash


def test_next_block_requirements_network_roundtrip():
    req = sample_requirements()
    data = req.to_network()
    req2 = NextBlockRequirements.from_network(data)
    assert req2.difficulty_energy == req.difficulty_energy
    assert req2.min_diversity == req.min_diversity
    assert req2.min_solutions == req.min_solutions
    assert req2.timeout_to_difficulty_adjustment_decay == req.timeout_to_difficulty_adjustment_decay


def make_sample_block():
    hdr = sample_header()
    mi = sample_miner_info()
    qp = sample_quantum_proof()
    req = sample_requirements()
    return Block(
        header=hdr,
        miner_info=mi,
        quantum_proof=qp,
        next_block_requirements=req,
        data="hello world",
        raw=b"",
        hash=b"",
        signature=b"SIG",
    )


def test_block_to_from_network_roundtrip():
    blk = make_sample_block()
    data = blk.to_network()
    blk2 = Block.from_network(data)

    assert blk2.header.index == blk.header.index
    assert blk2.header.previous_hash == blk.header.previous_hash
    assert blk2.header.timestamp == blk.header.timestamp
    assert blk2.header.data_hash == blk.header.data_hash

    assert blk2.miner_info.miner_id == blk.miner_info.miner_id
    assert blk2.miner_info.miner_type == blk.miner_info.miner_type
    assert blk2.miner_info.reward_address == blk.miner_info.reward_address

    assert blk2.quantum_proof.nonce == blk.quantum_proof.nonce
    assert blk2.quantum_proof.solutions == blk.quantum_proof.solutions

    assert blk2.next_block_requirements.min_solutions == blk.next_block_requirements.min_solutions
    assert blk2.data == blk.data
    assert blk2.signature == blk.signature

    # Derived fields are computed by .from_network
    assert isinstance(blk2.raw, (bytes, bytearray)) and len(blk2.raw) >= 0
    assert isinstance(blk2.hash, (bytes, bytearray)) and len(blk2.hash) == 32


def test_block_compute_derived_fields_sets_hash_and_raw():
    blk = make_sample_block()
    blk.finalize()
    assert blk.raw == blk.to_network()[:-len(blk.signature)]
    assert blk.hash == blake3(blk.raw).digest()


def test_block_validate_block_true_and_false():
    prev = make_sample_block()

    # Lenient requirements: very high energy threshold, low diversity, 1 solution
    prev.next_block_requirements = NextBlockRequirements(
        difficulty_energy=1e9,
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10,
    )

    blk = make_sample_block()
    assert blk.validate_block(prev) is True

    # Harsher requirements should fail
    harsh = NextBlockRequirements(
        difficulty_energy=-60.0,  # Harder threshold
        min_diversity=1.0,
        min_solutions=3,
        timeout_to_difficulty_adjustment_decay=10,
    )
    prev.next_block_requirements = harsh
    assert blk.validate_block(prev) is False

