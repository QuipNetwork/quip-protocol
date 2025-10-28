"""Block data structures and parsing utilities for quantum blockchain."""

from blake3 import blake3
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from shared.time_utils import utc_timestamp

from shared.quantum_proof_of_work import (
    calculate_diversity, generate_ising_model_from_nonce, 
    energies_for_solutions
)
from shared.logging_config import get_logger

# Initialize logger
logger = get_logger('block')


def write_string(data: str) -> bytes:
    utf8_bytes = data.encode('utf-8')
    return struct.pack('!I', len(utf8_bytes)) + utf8_bytes

def read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack('!I', data[offset:offset+4])[0]
    offset += 4
    string = data[offset:offset+length].decode('utf-8')
    return string, offset + length

def write_bytes(data: bytes) -> bytes:
    return struct.pack('!I', len(data)) + data

def read_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    length = struct.unpack('!I', data[offset:offset+4])[0]
    offset += 4
    bytes_data = data[offset:offset+length]
    return bytes_data, offset + length

def write_varint(value: int) -> bytes:
    """Variable-length integer encoding with zigzag encoding for negatives (1-5 bytes per value)."""
    # Zigzag encoding: negative -> positive, positive -> negative * 2
    encoded = (value << 1) ^ (value >> 31) if value < 0 else (value << 1)
    result = b''
    while encoded >= 0x80:
        result += bytes([(encoded & 0x7F) | 0x80])
        encoded >>= 7
    result += bytes([encoded])
    return result

def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Variable-length integer decoding with zigzag decoding."""
    encoded = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Unexpected end of data")
        byte = data[offset]
        offset += 1
        encoded |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
        if shift >= 35:  # Prevent overflow
            raise ValueError("Varint too long")

    # Zigzag decoding
    value = (encoded >> 1) ^ (-(encoded & 1))
    return value, offset

def compress_nodes(nodes: List[int]) -> bytes:
    """Compress node list using delta encoding."""
    if not nodes:
        return b''
    # Sort nodes for better compression
    sorted_nodes = sorted(nodes)
    result = write_varint(sorted_nodes[0])
    for i in range(1, len(sorted_nodes)):
        delta = sorted_nodes[i] - sorted_nodes[i-1]
        result += write_varint(delta)
    return result

def decompress_nodes(data: bytes) -> tuple[List[int], int]:
    """Decompress node list from delta encoding. Returns (nodes, bytes_consumed)."""
    if not data:
        return [], 0
    offset = 0
    nodes = []
    first_node, offset = read_varint(data, offset)
    nodes.append(first_node)
    while offset < len(data):
        # Check if next byte indicates end of node data (would be edge data start)
        if offset >= len(data):
            break
        try:
            delta, new_offset = read_varint(data, offset)
            nodes.append(nodes[-1] + delta)
            offset = new_offset
        except ValueError:
            # If we can't read a valid varint, we've reached the end of node data
            break
    return nodes, offset

def compress_edges(edges: List[Tuple[int, int]]) -> bytes:
    """Compress edges using adjacency list with delta encoding."""
    if not edges:
        return b''

    # Build adjacency list
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    result = write_varint(len(adj))  # number of nodes with edges

    for node in sorted(adj.keys()):
        neighbors = sorted(adj[node])
        result += write_varint(node)  # node ID
        result += write_varint(len(neighbors))  # neighbor count

        if neighbors:
            result += write_varint(neighbors[0])  # first neighbor
            for i in range(1, len(neighbors)):
                delta = neighbors[i] - neighbors[i-1]
                result += write_varint(delta)  # delta-encoded neighbors

    return result

def decompress_edges(data: bytes) -> tuple[List[Tuple[int, int]], int]:
    """Decompress edges from adjacency list format. Returns (edges, bytes_consumed)."""
    if not data:
        return [], 0

    offset = 0
    edges = []
    num_nodes, offset = read_varint(data, offset)

    for _ in range(num_nodes):
        node, offset = read_varint(data, offset)
        num_neighbors, offset = read_varint(data, offset)

        neighbors = []
        if num_neighbors > 0:
            first_neighbor, offset = read_varint(data, offset)
            neighbors.append(first_neighbor)

            for _ in range(num_neighbors - 1):
                delta, offset = read_varint(data, offset)
                neighbors.append(neighbors[-1] + delta)

        # Reconstruct edges
        for neighbor in neighbors:
            if node < neighbor:  # Avoid duplicates in undirected graph
                edges.append((node, neighbor))

    return edges, offset

def compress_solutions(solutions: List[List[int]]) -> bytes:
    """Compress solutions by storing actual integer values."""
    if not solutions:
        return b''

    result = write_varint(len(solutions))  # solution count

    for solution in solutions:
        result += write_varint(len(solution))  # solution length
        for value in solution:
            result += write_varint(value)

    return result

def decompress_solutions(data: bytes) -> tuple[List[List[int]], int]:
    """Decompress solutions from varint-encoded format. Returns (solutions, bytes_consumed)."""
    if not data:
        return [], 0

    offset = 0
    num_solutions, offset = read_varint(data, offset)

    solutions = []

    for _ in range(num_solutions):
        solution_length, offset = read_varint(data, offset)
        solution = []
        for _ in range(solution_length):
            value, offset = read_varint(data, offset)
            solution.append(value)
        solutions.append(solution)

    return solutions, offset

@dataclass
class BlockRequirements:
    """Requirements that the next block must satisfy.

    Note: diversity_range and solutions_range are blockchain-level constraints
    that define the allowable MAX values. They are stored here for convenience
    but represent blockchain configuration rather than per-block requirements.
    """
    difficulty_energy: float
    min_diversity: float  # Set per block, constrained by diversity_range
    min_solutions: int    # Set per block, constrained by solutions_range
    timeout_to_difficulty_adjustment_decay: int
    h_values: Optional[List[float]] = None

    # Blockchain-level constraints (define allowable max values)
    # Currently fixed at (0.3, 0.3) and (20, 20) but infrastructure ready for chain consensus
    diversity_range: Optional[Tuple[float, float]] = None
    solutions_range: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.h_values is None:
            self.h_values = [-1.0, 0.0, 1.0]  # Default: ternary distribution

        # Set range defaults from blockchain configuration
        if self.diversity_range is None:
            from shared.energy_utils import DEFAULT_DIVERSITY_RANGE
            self.diversity_range = DEFAULT_DIVERSITY_RANGE

        if self.solutions_range is None:
            from shared.energy_utils import DEFAULT_SOLUTIONS_RANGE
            self.solutions_range = DEFAULT_SOLUTIONS_RANGE

        # Initialize min values from range if not already set
        # (For now, ranges are fixed so this effectively sets them to the fixed values)
        if self.min_diversity is None or self.min_diversity == 0:
            self.min_diversity = self.diversity_range[0]
        if self.min_solutions is None or self.min_solutions == 0:
            self.min_solutions = self.solutions_range[0]

    @property
    def effective_diversity(self) -> float:
        """Get the effective diversity requirement."""
        return self.min_diversity

    @property
    def effective_solutions(self) -> int:
        """Get the effective solutions requirement."""
        return self.min_solutions

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += struct.pack('!d', self.difficulty_energy)
        result += struct.pack('!d', self.min_diversity)
        result += struct.pack('!I', self.min_solutions)
        result += struct.pack('!i', self.timeout_to_difficulty_adjustment_decay)

        # Serialize h_values
        h_vals = self.h_values if self.h_values is not None else [-1.0, 0.0, 1.0]
        result += struct.pack('!I', len(h_vals))
        for h_val in h_vals:
            result += struct.pack('!d', h_val)

        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'BlockRequirements':
        """Deserialize from binary format."""
        offset = 0
        difficulty_energy = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_diversity = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_solutions = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        timeout_to_difficulty_adjustment_decay = struct.unpack('!i', data[offset:offset+4])[0]
        offset += 4

        # Deserialize h_values (backward compatible)
        h_values = None
        if offset < len(data):
            h_count = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            h_values = []
            for _ in range(h_count):
                h_val = struct.unpack('!d', data[offset:offset+8])[0]
                offset += 8
                h_values.append(h_val)

        return cls(
            difficulty_energy=difficulty_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=timeout_to_difficulty_adjustment_decay,
            h_values=h_values
        )

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            'difficulty_energy': self.difficulty_energy,
            'min_diversity': self.min_diversity,
            'min_solutions': self.min_solutions,
            'timeout_to_difficulty_adjustment_decay': self.timeout_to_difficulty_adjustment_decay,
            'h_values': self.h_values if self.h_values is not None else [-1.0, 0.0, 1.0]
        }

    @classmethod
    def from_json(cls, data: dict) -> 'BlockRequirements':
        """Deserialize from JSON-compatible dictionary."""
        return cls(
            difficulty_energy=float(data['difficulty_energy']),
            min_diversity=float(data['min_diversity']),
            min_solutions=int(data['min_solutions']),
            timeout_to_difficulty_adjustment_decay=int(data['timeout_to_difficulty_adjustment_decay']),
            h_values=data.get('h_values', None)  # Backward compatible: missing h_values → None → defaults to [-1, 0, 1]
        )

@dataclass
class QuantumProof:
    """Quantum mining proof containing the essential mining result data."""
    nonce: int
    salt: bytes
    nodes: List[int]  # List of nodes in the Ising model
    edges: List[Tuple[int, int]]  # List of edges in the Ising model
    solutions: List[List[int]]  # List of quantum solutions found
    mining_time: float

    # Computed fields (derived from validation, not stored in network format)
    energy: Optional[float] = None
    diversity: Optional[float] = None
    num_valid_solutions: Optional[int] = None

    def to_network(self) -> bytes:
        """Serialize to space-efficient binary format.
        Format:
        - nonce: uint64
        - salt: length-prefixed bytes
        - mining_time: float64
        - nodes: length-prefixed compressed delta-encoded varints
        - edges: length-prefixed compressed adjacency list
        - solutions: length-prefixed bit-packed binary values
        """
        result = struct.pack('!Q', self.nonce)
        result += write_bytes(self.salt)
        result += struct.pack('!d', self.mining_time)

        # Add length prefixes for each compressed section
        nodes_data = compress_nodes(self.nodes)
        result += struct.pack('!I', len(nodes_data)) + nodes_data

        edges_data = compress_edges(self.edges)
        result += struct.pack('!I', len(edges_data)) + edges_data

        solutions_data = compress_solutions(self.solutions)
        result += struct.pack('!I', len(solutions_data)) + solutions_data

        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'QuantumProof':
        """Deserialize from space-efficient binary format."""
        offset = 0

        nonce = struct.unpack('!Q', data[offset:offset+8])[0]
        offset += 8

        salt, offset = read_bytes(data, offset)

        mining_time = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8

        # Read length-prefixed compressed sections
        nodes_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        nodes_data = data[offset:offset+nodes_len]
        offset += nodes_len
        nodes, _ = decompress_nodes(nodes_data)

        edges_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        edges_data = data[offset:offset+edges_len]
        offset += edges_len
        edges, _ = decompress_edges(edges_data)

        solutions_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        solutions_data = data[offset:offset+solutions_len]
        solutions, _ = decompress_solutions(solutions_data)

        return cls(nonce=nonce, salt=salt, nodes=nodes, edges=edges,
                  solutions=solutions, mining_time=mining_time)

    def to_json(self) -> dict:
        """Serialize to space-efficient JSON-compatible dictionary."""
        # Use compressed binary format for proof_data
        proof_data = self.to_network()
        return {
            'proof_data': proof_data.hex(),
            'energy': self.energy,
            'diversity': self.diversity,
            'num_valid_solutions': self.num_valid_solutions
        }

    @classmethod
    def from_json(cls, data: dict) -> 'QuantumProof':
        """Deserialize from space-efficient JSON format."""
        proof_data = bytes.fromhex(data['proof_data'])
        quantum_proof = QuantumProof.from_network(proof_data)

        # Set derived fields
        quantum_proof.energy = data.get('energy')
        quantum_proof.diversity = data.get('diversity')
        quantum_proof.num_valid_solutions = data.get('num_valid_solutions')

        return quantum_proof

    def compute_derived_fields(self):
        """Calculate derived fields from solutions and requirements using Ising model.
        Requires the Block for deterministic model generation.
        """
        if not self.solutions:
            return

        # Generate model sized to the maximum solution length
        h, J = generate_ising_model_from_nonce(self.nonce,
                                              self.nodes,
                                              self.edges)

        energies = energies_for_solutions(self.solutions, h, J, self.nodes)

        # Set computed fields
        self.energy = min(energies) if energies else None
        self.num_valid_solutions = len(self.solutions)
        self.diversity = calculate_diversity(self.solutions)


@dataclass
class Transaction:
    """Transaction record containing a solve request and its result."""
    transaction_id: str  # Unique identifier
    timestamp: int  # Unix timestamp
    request_h: List[float]  # Linear bias coefficients
    request_J: List[Tuple[Tuple[int, int], float]]  # Sparse coupling matrix as list of ((i, j), value)
    num_samples: int  # Number of samples requested
    samples: List[List[int]]  # Solution bitstrings/spin configurations
    energies: List[float]  # Corresponding energies

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += write_string(self.transaction_id)
        result += struct.pack('!q', self.timestamp)

        # Serialize h array
        result += struct.pack('!I', len(self.request_h))
        for h_val in self.request_h:
            result += struct.pack('!d', h_val)

        # Serialize J array
        result += struct.pack('!I', len(self.request_J))
        for ((i, j), j_val) in self.request_J:
            result += struct.pack('!I', i)
            result += struct.pack('!I', j)
            result += struct.pack('!d', j_val)

        result += struct.pack('!I', self.num_samples)

        # Serialize samples
        solutions_data = compress_solutions(self.samples)
        result += struct.pack('!I', len(solutions_data)) + solutions_data

        # Serialize energies
        result += struct.pack('!I', len(self.energies))
        for energy in self.energies:
            result += struct.pack('!d', energy)

        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'Transaction':
        """Deserialize from binary format."""
        offset = 0

        transaction_id, offset = read_string(data, offset)
        timestamp = struct.unpack('!q', data[offset:offset+8])[0]
        offset += 8

        # Deserialize h
        h_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        request_h = []
        for _ in range(h_len):
            h_val = struct.unpack('!d', data[offset:offset+8])[0]
            offset += 8
            request_h.append(h_val)

        # Deserialize J
        j_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        request_J = []
        for _ in range(j_len):
            i = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            j = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            j_val = struct.unpack('!d', data[offset:offset+8])[0]
            offset += 8
            request_J.append(((i, j), j_val))

        num_samples = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4

        # Deserialize samples
        solutions_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        solutions_data = data[offset:offset+solutions_len]
        offset += solutions_len
        samples, _ = decompress_solutions(solutions_data)

        # Deserialize energies
        energies_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        energies = []
        for _ in range(energies_len):
            energy = struct.unpack('!d', data[offset:offset+8])[0]
            offset += 8
            energies.append(energy)

        return cls(
            transaction_id=transaction_id,
            timestamp=timestamp,
            request_h=request_h,
            request_J=request_J,
            num_samples=num_samples,
            samples=samples,
            energies=energies
        )

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp,
            'request_h': self.request_h,
            'request_J': [{'i': i, 'j': j, 'value': val} for ((i, j), val) in self.request_J],
            'num_samples': self.num_samples,
            'samples': self.samples,
            'energies': self.energies
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Transaction':
        """Deserialize from JSON-compatible dictionary."""
        request_J = [((entry['i'], entry['j']), entry['value']) for entry in data['request_J']]
        return cls(
            transaction_id=data['transaction_id'],
            timestamp=data['timestamp'],
            request_h=data['request_h'],
            request_J=request_J,
            num_samples=data['num_samples'],
            samples=data['samples'],
            energies=data['energies']
        )


@dataclass
class MinerInfo:
    """Information about the miner who created this block."""
    miner_id: str               # e.g., "node1-CPU-1"
    miner_type: str             # e.g., "CPU", "GPU", "QPU"
    reward_address: bytes         # ECDSA public key for rewards
    ecdsa_public_key: bytes       # Miner's ECDSA public key
    wots_public_key: bytes        # Current WOTS+ public key
    next_wots_public_key: bytes   # Next WOTS+ public key (for signature verification)

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += write_string(self.miner_id)
        result += write_string(self.miner_type)
        result += write_bytes(self.reward_address)
        result += write_bytes(self.ecdsa_public_key)
        result += write_bytes(self.wots_public_key)
        result += write_bytes(self.next_wots_public_key)
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'MinerInfo':
        """Deserialize from binary format."""
        offset = 0
        miner_id, offset = read_string(data, offset)
        miner_type, offset = read_string(data, offset)
        reward_address, offset = read_bytes(data, offset)
        ecdsa_public_key, offset = read_bytes(data, offset)
        wots_public_key, offset = read_bytes(data, offset)
        next_wots_public_key, offset = read_bytes(data, offset)

        return cls(
            miner_id=miner_id,
            miner_type=miner_type,
            reward_address=reward_address,
            ecdsa_public_key=ecdsa_public_key,
            wots_public_key=wots_public_key,
            next_wots_public_key=next_wots_public_key
        )

    def to_json(self) -> str:
        """Serialize to JSON string with hex-encoded bytes fields."""
        data = {
            'miner_id': self.miner_id,
            'miner_type': self.miner_type,
            'reward_address': self.reward_address.hex(),
            'ecdsa_public_key': self.ecdsa_public_key.hex(),
            'wots_public_key': self.wots_public_key.hex(),
            'next_wots_public_key': self.next_wots_public_key.hex(),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'MinerInfo':
        """Deserialize from JSON string with hex-encoded bytes fields."""
        data = json.loads(json_str)
        return cls(
            miner_id=data['miner_id'],
            miner_type=data['miner_type'],
            reward_address=bytes.fromhex(data['reward_address']),
            ecdsa_public_key=bytes.fromhex(data['ecdsa_public_key']),
            wots_public_key=bytes.fromhex(data['wots_public_key']),
            next_wots_public_key=bytes.fromhex(data['next_wots_public_key']),
        )


@dataclass
class BlockHeader:
    """Structured block header for better parsing and validation."""
    previous_hash: bytes
    index: int
    timestamp: int  # Unix timestamp
    data_hash: bytes # hash of all non-header data fields

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += struct.pack('!I', len(self.previous_hash)) + self.previous_hash
        result += struct.pack('!Q', self.index)
        result += struct.pack('!q', self.timestamp)  # signed 64-bit for timestamp
        result += struct.pack('!I', len(self.data_hash)) + self.data_hash
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'BlockHeader':
        """Deserialize from binary format."""
        offset = 0

        # Previous hash
        hash_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        previous_hash = data[offset:offset+hash_len]
        offset += hash_len

        # Index and timestamp
        index = struct.unpack('!Q', data[offset:offset+8])[0]
        offset += 8
        timestamp = struct.unpack('!q', data[offset:offset+8])[0]
        offset += 8

        # Data hash
        data_hash_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        data_hash = data[offset:offset+data_hash_len]

        return cls(
            previous_hash=previous_hash,
            index=index,
            timestamp=timestamp,
            data_hash=data_hash
        )

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            'previous_hash': self.previous_hash.hex(),
            'index': self.index,
            'timestamp': self.timestamp,
            'data_hash': self.data_hash.hex()
        }

    @classmethod
    def from_json(cls, data: dict) -> 'BlockHeader':
        """Deserialize from JSON-compatible dictionary."""
        return cls(
            previous_hash=bytes.fromhex(data['previous_hash']),
            index=data['index'],
            timestamp=int(data['timestamp']),
            data_hash=bytes.fromhex(data['data_hash'])
        )




@dataclass
class Block:
    header: BlockHeader

    miner_info: MinerInfo
    quantum_proof: QuantumProof
    next_block_requirements: BlockRequirements

    data: bytes  # Arbitrary data, eventually a merkle tree most likely.
    transactions: List[Transaction] = None  # Solve requests included in this block

    # NOTE: Maybe move this to a separate "NetworkBlock" class which returns
    #       Block, BlockHash, Signature on parse.
    #      For now, we keep this coupled, but it is usually a not great idea to
    #      couple signatures to the data they sign, network serialization, etc.

    # Computed Fields
    # Everything except the signature.
    raw: Optional[bytes] = None
    # Network hash (hash of the actual serialized network bytes)
    hash: Optional[bytes] = None
    # signature bytes
    signature: Optional[bytes] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.transactions is None:
            self.transactions = []

    def to_network(self) -> bytes:
        """Serialize to binary format, excluding derived fields (raw, hash).
        External to this class, yous should only call this after finalization
        (compute_hash) and signature is added.
        """
        result = b''
        result += self.header.to_network()
        result += self.miner_info.to_network()
        result += self.quantum_proof.to_network()
        result += self.next_block_requirements.to_network()
        result += write_bytes(self.data)

        # Serialize transactions
        result += struct.pack('!I', len(self.transactions))
        for tx in self.transactions:
            tx_data = tx.to_network()
            result += struct.pack('!I', len(tx_data)) + tx_data

        if self.signature:
            result += write_bytes(self.signature)
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'Block':
        """Deserialize from binary format."""


        offset = 0

        # Read header
        header_data = data[offset:]
        header = BlockHeader.from_network(header_data)
        header_size = len(header.to_network())
        offset += header_size

        # Read miner_info
        miner_data = data[offset:]
        block_data = miner_data
        miner_info = MinerInfo.from_network(miner_data)
        miner_size = len(miner_info.to_network())
        offset += miner_size

        # Read quantum_proof
        proof_data = data[offset:]
        block_data += proof_data
        quantum_proof = QuantumProof.from_network(proof_data)
        proof_size = len(quantum_proof.to_network())
        offset += proof_size

        # Read next_block_requirements
        req_data = data[offset:]
        next_block_requirements = BlockRequirements.from_network(req_data)
        req_size = len(next_block_requirements.to_network())
        offset += req_size

        # Read data
        block_data, offset = read_bytes(data, offset)

        # Read transactions
        num_transactions = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        transactions = []
        for _ in range(num_transactions):
            tx_len = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            tx_data = data[offset:offset+tx_len]
            offset += tx_len
            transactions.append(Transaction.from_network(tx_data))

        # Validate the block data is well formed.
        # TODO: Do this at the top of the function with a static signature size.
        header_block_data_hash = header.data_hash
        actual_data_hash = blake3(block_data).digest()
        if actual_data_hash != header_block_data_hash:
            raise ValueError("Data hash mismatch")

        raw = data[:offset]
        hash = blake3(raw).digest()

        # If not genesis, read signature
        if header.index > 0:
            signature, offset = read_bytes(data, offset)
        else:
            signature = b""

        # Create block with placeholder values for derived fields
        block = cls(
            header=header,
            miner_info=miner_info,
            quantum_proof=quantum_proof,
            next_block_requirements=next_block_requirements,
            data=block_data,
            transactions=transactions,
            raw=raw,
            hash=hash,
            signature=signature
        )

        return block

    def finalize(self):
        """Compute derived fields (raw, hash, etc) so the block can be signed."""
        if self.signature:
            raise ValueError("Block already signed")

        # Compute derived fields for quantum proof
        if self.quantum_proof:
            self.quantum_proof.compute_derived_fields()

        # Compute data hash (TBD merkle tree..)
        self.header.data_hash = blake3(self.data).digest()

        # Set raw to the network serialization
        self.raw = self.to_network()

        # Compute hash from raw bytes
        self.hash = blake3(self.raw).digest()


    def to_json(self) -> str:
        """Serialize block to JSON string."""
        data = {
            'header': self.header.to_json(),
            'miner_info': self.miner_info.to_json(),
            'quantum_proof': self.quantum_proof.to_json(),
            'next_block_requirements': self.next_block_requirements.to_json(),
            'data': self.data.hex(),
            'transactions': [tx.to_json() for tx in self.transactions],
            'raw': self.raw.hex() if self.raw else None,
            'hash': self.hash.hex() if self.hash else None,
            'signature': self.signature.hex() if self.signature else None
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Block':
        """Deserialize block from JSON string."""
        data = json.loads(json_str)

        if 'header' not in data:
            raise ValueError("Missing header in JSON data")
        
        if 'next_block_requirements' not in data:
            raise ValueError("Missing next_block_requirements in JSON data")

        # Parse components using their own from_json methods
        header = BlockHeader.from_json(data['header'])
        next_block_requirements = BlockRequirements.from_json(data['next_block_requirements'])

        if data['miner_info']:
            miner_info = MinerInfo.from_json(data['miner_info'])
        else:
            miner_info = MinerInfo(miner_id='',
                                   miner_type='', 
                                   reward_address=b'',
                                   ecdsa_public_key=b'',
                                   wots_public_key=b'',
                                  next_wots_public_key=b'')
        if data['quantum_proof']:
            quantum_proof = QuantumProof.from_json(data['quantum_proof'])
        else:
            quantum_proof = QuantumProof(nonce=0, 
                                         salt=b'',
                                         nodes=[],
                                         edges=[],
                                         solutions=[],
                                         mining_time=0.0)

        # Use preserved raw bytes if available, otherwise reconstruct
        raw_bytes = bytes.fromhex(data['raw']) if data.get('raw') else b''

        try:
            block_data = bytes.fromhex(data['data'])
        except ValueError:
            block_data = data['data'].encode()

        # Parse transactions
        transactions = []
        if 'transactions' in data and data['transactions']:
            transactions = [Transaction.from_json(tx_data) for tx_data in data['transactions']]

        # Create block
        block = cls(
            header=header,
            miner_info=miner_info,
            quantum_proof=quantum_proof,
            next_block_requirements=next_block_requirements,
            data=block_data,
            transactions=transactions,
            raw=raw_bytes,
            hash=bytes.fromhex(data['hash']) if data.get('hash') else b'',
            signature=bytes.fromhex(data['signature']) if data.get('signature') else b''
        )

        # If raw bytes weren't preserved, reconstruct them for signature verification
        if not raw_bytes:
            block.finalize()

        return block




def load_genesis_block(genesis_block_filepath: str) -> 'Block':
    """Load genesis block from a JSON file.

    Args:
        genesis_block_filepath: Path to genesis config file.
    Returns:
        Genesis Block object

    Raises:
        FileNotFoundError: If the specified config file is not found
        KeyError: If required configuration keys are missing
        json.JSONDecodeError: If JSON is malformed
    """
    config_path = Path(genesis_block_filepath)
    logger.info(f"Loading genesis block from: {config_path.name}")
    with open(config_path, 'r') as f:
        genesis_data = json.load(f)

    # Use create_genesis_block to parse and validate the data
    genesis_block = create_genesis_block(genesis_data)

    logger.info(f"Loaded genesis block from: {config_path.name}")
    # Note: adaptive_parameters not available in current BlockRequirements structure
    logger.info(f"Mining parameters: difficulty_energy={genesis_block.next_block_requirements.difficulty_energy}")

    return genesis_block


def create_genesis_block(genesis_data: Optional[dict] = None) -> Block:
    """Create the genesis block for the blockchain.

    Uses parse_block_json to create and validate the genesis block from the provided data.
    If no data is provided, creates a default genesis block.

    Args:
        genesis_data: Dictionary containing genesis block configuration. If None, creates default.

    Returns:
        Genesis Block object
    """
    if genesis_data is not None:
        # Use parse_block_json to create the block from the provided data
        return Block.from_json(json.dumps(genesis_data))

    # Create default genesis block data in the correct format
    default_genesis_data = {
        "header": {
            "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
            "index": 0,
            "timestamp": utc_timestamp(),
            "data_hash": "0000000000000000000000000000000000000000000000000000000000000000"
        },
        "data": "Genesis Block - Quip Protocol",
        "next_block_requirements": {
            "difficulty_energy": -1000.0,
            "min_diversity": 0.28,
            "min_solutions": 10,
            "timeout_to_difficulty_adjustment_decay": 600,
            "h_values": [-1.0, 0.0, 1.0]
        },
        "quantum_proof": None,
        "miner_info": None
    }

    # Use parse_block_json to create and validate the default genesis block
    return Block.from_json(json.dumps(default_genesis_data))


def calculate_adaptive_parameters(requirements: Dict[str, Any], miner_type: str) -> Dict[str, Any]:
    """Calculate adaptive mining parameters based on difficulty requirements.

    This implements the intelligent parameter selection based on current difficulty.

    Args:
        requirements: Mining requirements from get_mining_requirements()
        miner_type: Type of miner ('CPU', 'GPU', 'QPU')

    Returns:
        Dict with optimized parameters for the specific miner type
    """
    difficulty_energy = requirements['difficulty_energy']
    min_diversity = requirements['min_diversity']
    min_solutions = requirements['min_solutions']

    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    if miner_type == 'CPU' or miner_type.startswith('CPU'):
        # Simulated Annealing parameters
        base_sweeps = 512
        num_sweeps = int(base_sweeps * (difficulty_factor ** 1.5))  # Exponential scaling
        num_reads = max(min_solutions * 3, 100)  # At least 3x required solutions

        # Beta schedule adjustment for harder problems
        if difficulty_factor > 10:  # Very hard problems
            beta_range = [0.05, 15.0]  # Wider exploration
        else:
            beta_range = [0.1, 10.0]   # Standard range

        return {
            'num_sweeps': max(256, min(num_sweeps, 32768)),  # Reasonable bounds
            'num_reads': max(64, min(num_reads, 1000)),      # Reasonable bounds
            'beta_range': beta_range,
            'beta_schedule': 'geometric'
        }

    elif miner_type == 'QPU' or miner_type.startswith('QPU'):
        # Quantum Processing Unit parameters
        base_annealing_time = 20.0  # microseconds
        annealing_time = base_annealing_time * (difficulty_factor ** 0.8)  # Gentler scaling
        num_reads = max(min_solutions * 2, 64)  # QPU typically needs fewer reads

        return {
            'quantum_annealing_time': max(5.0, min(annealing_time, 200.0)),  # Reasonable bounds
            'num_reads': max(64, min(num_reads, 1000)),
            'chain_strength': 1.0,  # Standard for QPU
        }

    elif miner_type == 'GPU' or miner_type.startswith('GPU'):
        # GPU parameters (similar to CPU but optimized for parallel processing)
        base_sweeps = 256  # GPUs can do more parallel sweeps efficiently
        num_sweeps = int(base_sweeps * (difficulty_factor ** 1.2))  # Moderate scaling
        num_reads = max(min_solutions * 2, 100)

        return {
            'num_sweeps': max(128, min(num_sweeps, 8192)),   # GPU-optimized bounds
            'num_reads': max(64, min(num_reads, 1000)),
            'parallel_chains': 4,  # GPU can run multiple chains
        }

    else:
        # Default fallback
        return {
            'num_sweeps': 1024,
            'num_reads': 100,
            'beta_range': [0.1, 10.0]
        }