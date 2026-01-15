from __future__ import annotations

from blake3 import blake3
import time
import json
import pytest

from shared.block import (
    QuantumProof,
    MinerInfo,
    BlockHeader,
    Block,
)
from shared.block_requirements import BlockRequirements, validate_block


def sample_quantum_proof():
    return QuantumProof(
        nonce=123456789,
        salt=b"S",
        solutions=[[1, -1, 0], [0, 1, -1], [-5], [3]],
        mining_time=1.234,
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3)],
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
    return BlockRequirements(
        difficulty_energy=-20.0,
        min_diversity=0.5,
        min_solutions=2,
        timeout_to_difficulty_adjustment_decay=10,
    )


def sample_header(data_hash: bytes | None = None):
    # Use the hash of "hello world" to match the data in make_sample_block
    if data_hash is None:
        data_hash = blake3(b"hello world").digest()
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
    qp.compute_derived_fields()  # No arguments - uses self.nonce, self.nodes, self.edges
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
    req2 = BlockRequirements.from_network(data)
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
        data=b"hello world",
        raw=b"",
        hash=b"",
        signature=b"SIG",
    )


def make_unsigned_sample_block():
    """Create a sample block without signature for tests that need to finalize."""
    hdr = sample_header()
    mi = sample_miner_info()
    qp = sample_quantum_proof()
    req = sample_requirements()
    return Block(
        header=hdr,
        miner_info=mi,
        quantum_proof=qp,
        next_block_requirements=req,
        data=b"hello world",
        raw=b"",
        hash=b"",
        signature=b"",  # No signature
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
    assert blk2.raw is not None and isinstance(blk2.raw, (bytes, bytearray)) and len(blk2.raw) >= 0
    assert blk2.hash is not None and isinstance(blk2.hash, (bytes, bytearray)) and len(blk2.hash) == 32


def test_block_compute_derived_fields_sets_hash_and_raw():
    blk = make_unsigned_sample_block()
    blk.finalize()
    assert blk.raw is not None
    assert blk.hash is not None
    # For unsigned blocks, raw should equal the network representation
    assert blk.raw == blk.to_network()
    assert blk.hash == blake3(blk.raw).digest()


def test_block_validate_block_true_and_false(monkeypatch):
    """Test validate_block returns True/False based on quantum proof validation."""
    from shared import block_requirements

    prev = make_sample_block()
    blk = make_sample_block()

    # Ensure block timestamp is after previous block timestamp
    prev.header.timestamp = int(time.time()) - 10
    blk.header.timestamp = int(time.time())

    # Lenient requirements
    prev.next_block_requirements = BlockRequirements(
        difficulty_energy=1e9,
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=10,
    )

    # Mock validate_quantum_proof to return True - block should be valid
    monkeypatch.setattr(block_requirements, "validate_quantum_proof", lambda *args, **kwargs: True)
    assert validate_block(blk, prev) is True

    # Mock validate_quantum_proof to return False - block should be invalid
    monkeypatch.setattr(block_requirements, "validate_quantum_proof", lambda *args, **kwargs: False)
    assert validate_block(blk, prev) is False


# JSON Serialization Tests

def test_block_header_json_roundtrip():
    """Test BlockHeader JSON serialization and deserialization."""
    hdr = sample_header()

    # Serialize to JSON dict
    json_dict = hdr.to_json()
    assert isinstance(json_dict, dict)
    assert 'previous_hash' in json_dict
    assert 'index' in json_dict
    assert 'timestamp' in json_dict
    assert 'data_hash' in json_dict

    # Verify hex encoding
    assert json_dict['previous_hash'] == hdr.previous_hash.hex()
    assert json_dict['data_hash'] == hdr.data_hash.hex()

    # Deserialize from JSON dict
    hdr2 = BlockHeader.from_json(json_dict)

    # Verify roundtrip
    assert hdr2.previous_hash == hdr.previous_hash
    assert hdr2.index == hdr.index
    assert hdr2.timestamp == hdr.timestamp
    assert hdr2.data_hash == hdr.data_hash


def test_quantum_proof_json_roundtrip():
    """Test QuantumProof JSON serialization and deserialization."""
    qp = sample_quantum_proof()

    # Serialize to JSON dict
    json_dict = qp.to_json()
    assert isinstance(json_dict, dict)
    assert 'proof_data' in json_dict  # Compressed binary data
    assert 'energy' in json_dict
    assert 'diversity' in json_dict
    assert 'num_valid_solutions' in json_dict

    # Verify proof_data is hex-encoded binary
    assert isinstance(json_dict['proof_data'], str)
    proof_data_bytes = bytes.fromhex(json_dict['proof_data'])
    assert len(proof_data_bytes) > 0

    # Deserialize from JSON dict
    qp2 = QuantumProof.from_json(json_dict)

    # Verify roundtrip
    assert qp2.nonce == qp.nonce
    assert qp2.nodes == qp.nodes
    assert qp2.edges == qp.edges
    assert qp2.solutions == qp.solutions
    assert qp2.mining_time == qp.mining_time
    assert qp2.energy == qp.energy
    assert qp2.diversity == qp.diversity
    assert qp2.num_valid_solutions == qp.num_valid_solutions


def test_next_block_requirements_json_roundtrip():
    """Test BlockRequirements JSON serialization and deserialization."""
    req = sample_requirements()

    # Serialize to JSON dict
    json_dict = req.to_json()
    assert isinstance(json_dict, dict)
    assert 'difficulty_energy' in json_dict
    assert 'min_diversity' in json_dict
    assert 'min_solutions' in json_dict
    assert 'timeout_to_difficulty_adjustment_decay' in json_dict

    # Deserialize from JSON dict
    req2 = BlockRequirements.from_json(json_dict)

    # Verify roundtrip
    assert req2.difficulty_energy == req.difficulty_energy
    assert req2.min_diversity == req.min_diversity
    assert req2.min_solutions == req.min_solutions
    assert req2.timeout_to_difficulty_adjustment_decay == req.timeout_to_difficulty_adjustment_decay


def test_miner_info_json_roundtrip():
    """Test MinerInfo JSON serialization and deserialization."""
    mi = sample_miner_info()

    # Serialize to JSON string
    json_str = mi.to_json()
    assert isinstance(json_str, str)

    # Parse back to verify it's valid JSON
    json_dict = json.loads(json_str)
    assert 'miner_id' in json_dict
    assert 'miner_type' in json_dict
    assert 'reward_address' in json_dict
    assert 'ecdsa_public_key' in json_dict
    assert 'wots_public_key' in json_dict
    assert 'next_wots_public_key' in json_dict

    # Deserialize from JSON string
    mi2 = MinerInfo.from_json(json_str)

    # Verify roundtrip
    assert mi2.miner_id == mi.miner_id
    assert mi2.miner_type == mi.miner_type
    assert mi2.reward_address == mi.reward_address
    assert mi2.ecdsa_public_key == mi.ecdsa_public_key
    assert mi2.wots_public_key == mi.wots_public_key
    assert mi2.next_wots_public_key == mi.next_wots_public_key


def test_block_json_roundtrip_with_raw_preservation():
    """Test Block JSON serialization with raw field preservation."""
    blk = make_sample_block()
    blk.signature = b""  # Clear signature so we can finalize
    blk.finalize()  # Generate raw bytes and hash

    # Verify we have raw bytes and hash
    assert blk.raw is not None and len(blk.raw) > 0
    assert blk.hash is not None and len(blk.hash) == 32

    # Serialize to JSON string
    json_str = blk.to_json()
    assert isinstance(json_str, str)

    # Parse back to verify it's valid JSON
    json_dict = json.loads(json_str)
    assert 'header' in json_dict
    assert 'miner_info' in json_dict
    assert 'quantum_proof' in json_dict
    assert 'next_block_requirements' in json_dict
    assert 'data' in json_dict
    assert 'raw' in json_dict  # Raw field should be present
    assert 'hash' in json_dict
    assert 'signature' in json_dict

    # Verify raw field is hex-encoded
    assert json_dict['raw'] == blk.raw.hex()

    # Deserialize from JSON string
    blk2 = Block.from_json(json_str)

    # Verify raw bytes are preserved exactly
    assert blk2.raw == blk.raw
    assert blk2.hash == blk.hash
    assert blk2.signature == blk.signature

    # Verify all components match
    assert blk2.header.previous_hash == blk.header.previous_hash
    assert blk2.header.index == blk.header.index
    assert blk2.miner_info.miner_id == blk.miner_info.miner_id
    assert blk2.quantum_proof.nonce == blk.quantum_proof.nonce
    assert blk2.data == blk.data


def test_block_json_backward_compatibility():
    """Test Block JSON deserialization from JSON without raw field."""
    blk = make_sample_block()
    blk.signature = b""  # Clear signature so we can finalize
    blk.finalize()  # Generate raw bytes and hash

    # Create JSON without raw field (simulate old format)
    json_dict = {
        'header': blk.header.to_json(),
        'miner_info': blk.miner_info.to_json(),
        'quantum_proof': blk.quantum_proof.to_json(),
        'next_block_requirements': blk.next_block_requirements.to_json(),
        'data': blk.data.hex(),
        'hash': blk.hash.hex() if blk.hash else None,
        'signature': blk.signature.hex() if blk.signature else None
    }
    json_str = json.dumps(json_dict)

    # Deserialize from JSON without raw field
    blk2 = Block.from_json(json_str)

    # Verify raw bytes are reconstructed correctly
    assert blk2.raw is not None and len(blk2.raw) > 0
    assert blk2.hash == blk.hash  # Hash should match original

    # Verify all components match
    assert blk2.header.previous_hash == blk.header.previous_hash
    assert blk2.header.index == blk.header.index
    assert blk2.miner_info.miner_id == blk.miner_info.miner_id
    assert blk2.quantum_proof.nonce == blk.quantum_proof.nonce
    assert blk2.data == blk.data


def test_block_json_raw_bytes_preservation():
    """Test that raw bytes are preserved through JSON roundtrip."""
    # Create a block and finalize it
    blk = make_sample_block()
    blk.signature = b""  # Clear signature so we can finalize
    blk.finalize()

    # Ensure we have raw bytes
    assert blk.raw is not None
    original_raw = blk.raw

    # Serialize to JSON and back
    json_str = blk.to_json()
    blk2 = Block.from_json(json_str)

    # Verify raw bytes are preserved exactly
    assert blk2.raw is not None
    assert blk2.raw == original_raw
    assert blk2.hash == blk.hash


def test_quantum_proof_compression_effectiveness():
    """Test that compression actually reduces data size significantly."""
    qp = sample_quantum_proof()

    # Calculate original size (old inefficient format)
    original_size = (
        8 +  # nonce (uint64)
        4 + len(qp.salt) +  # salt (length + bytes)
        8 +  # mining_time (float64)
        len(qp.nodes) * 4 +  # nodes (int32 each)
        len(qp.edges) * 8 +  # edges (2 × int32 each)
        len(qp.solutions) * 4 +  # solution count header
        sum(len(sol) * 4 for sol in qp.solutions)  # solutions (int32 per value)
    )

    # Get compressed size
    compressed_data = qp.to_network()
    compressed_size = len(compressed_data)

    # Verify compression works (ratio depends on data size; smaller datasets have more overhead)
    compression_ratio = original_size / compressed_size
    assert compression_ratio > 1.0, f"Expected some compression, got {compression_ratio:.1f}x"
    print(f"Compression ratio: {compression_ratio:.1f}x (smaller datasets have more relative overhead)")

    # Verify data integrity through roundtrip
    qp_restored = QuantumProof.from_network(compressed_data)
    assert qp_restored.nonce == qp.nonce
    assert qp_restored.salt == qp.salt
    assert qp_restored.nodes == qp.nodes
    assert qp_restored.edges == qp.edges
    assert qp_restored.solutions == qp.solutions
    assert qp_restored.mining_time == qp.mining_time


def test_all_components_json_serialization():
    """Test that all components can be serialized to JSON independently."""
    hdr = sample_header()
    mi = sample_miner_info()
    qp = sample_quantum_proof()
    req = sample_requirements()

    # Test all components can serialize to JSON
    assert isinstance(hdr.to_json(), dict)
    assert isinstance(mi.to_json(), str)  # MinerInfo returns string
    assert isinstance(qp.to_json(), dict)
    assert isinstance(req.to_json(), dict)

    # Test all components can deserialize from JSON
    hdr2 = BlockHeader.from_json(hdr.to_json())
    mi2 = MinerInfo.from_json(mi.to_json())
    qp2 = QuantumProof.from_json(qp.to_json())
    req2 = BlockRequirements.from_json(req.to_json())

    # Verify they match originals
    assert hdr2.previous_hash == hdr.previous_hash
    assert mi2.miner_id == mi.miner_id
    assert qp2.nonce == qp.nonce
    assert req2.difficulty_energy == req.difficulty_energy

