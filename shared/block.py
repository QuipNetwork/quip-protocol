"""Block data structures and parsing utilities for quantum blockchain."""

from blake3 import blake3
import json
import time
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from shared.quantum_proof_of_work import calculate_diversity, generate_ising_model_from_seed, ising_seed_from_block


@dataclass
class QuantumProof:
    """Quantum mining proof containing the essential mining result data."""
    nonce: int
    nodes: List[int]  # List of nodes in the Ising model
    edges: List[Tuple[int, int]]  # List of edges in the Ising model
    solutions: List[List[int]]  # List of quantum solutions found
    mining_time: float

    # Computed fields (derived from validation, not stored in network format)
    energy: Optional[float] = None
    diversity: Optional[float] = None
    num_valid_solutions: Optional[int] = None

    def to_network(self) -> bytes:
        """Serialize to binary format, excluding derived fields.
        Format:
        - nonce: uint64
        - mining_time: float64
        - nodes: uint32 count, then int32 each
        - edges: uint32 count, then pairs of int32 (u, v)
        - solutions: uint32 count, then for each solution: uint32 length, then int32 values
        """
        result = b''
        result += struct.pack('!Q', self.nonce)
        result += struct.pack('!d', self.mining_time)

        # Nodes
        nodes = self.nodes or []
        result += struct.pack('!I', len(nodes))
        for node in nodes:
            result += struct.pack('!i', node)

        # Edges
        edges = self.edges or []
        result += struct.pack('!I', len(edges))
        for u, v in edges:
            result += struct.pack('!i', u)
            result += struct.pack('!i', v)

        # Solutions array
        result += struct.pack('!I', len(self.solutions))
        for solution in self.solutions:
            result += struct.pack('!I', len(solution))
            for value in solution:
                result += struct.pack('!i', value)  # signed int
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'QuantumProof':
        """Deserialize from binary format."""
        offset = 0

        # Basic fields
        nonce = struct.unpack('!Q', data[offset:offset+8])[0]
        offset += 8
        mining_time = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8

        # Nodes
        num_nodes = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        nodes: List[int] = []
        for _ in range(num_nodes):
            node = struct.unpack('!i', data[offset:offset+4])[0]
            offset += 4
            nodes.append(node)

        # Edges
        num_edges = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        edges: List[Tuple[int, int]] = []
        for _ in range(num_edges):
            u = struct.unpack('!i', data[offset:offset+4])[0]
            offset += 4
            v = struct.unpack('!i', data[offset:offset+4])[0]
            offset += 4
            edges.append((u, v))

        # Solutions
        num_solutions = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        solutions: List[List[int]] = []
        for _ in range(num_solutions):
            solution_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            solution: List[int] = []
            for _ in range(solution_length):
                value = struct.unpack('!i', data[offset:offset+4])[0]
                offset += 4
                solution.append(value)
            solutions.append(solution)

        return cls(nonce=nonce, nodes=nodes, edges=edges, solutions=solutions, mining_time=mining_time)

    def compute_derived_fields(self, requirements: 'NextBlockRequirements', block: 'Block'):
        """Calculate derived fields from solutions and requirements using Ising model.
        Requires the Block for deterministic model generation.
        """
        if not self.solutions:
            return

        from shared.quantum_proof_of_work import generate_ising_model_from_seed, calculate_diversity

        # Generate model sized to the maximum solution length
        n_model = max(len(sol) for sol in self.solutions)
        seed = ising_seed_from_block(prev_hash=block.header.previous_hash,
                                     cur_timestamp=block.header.timestamp,
                                     cur_index=block.header.index,
                                     nonce=self.nonce)
        h, J = generate_ising_model_from_seed(seed,
                                              block.quantum_proof.nodes,
                                              block.quantum_proof.edges)

        def energy_of(solution: List[int]) -> float:
            # Map values to spins in {-1, +1}
            spins = [1 if v > 0 else -1 for v in solution]
            e = 0.0
            # Local fields up to this solution's length
            for i in range(len(spins)):
                e += h.get(i, 0.0) * spins[i]
            # Couplers only if both endpoints exist in this solution
            for (i, j), Jij in J.items():
                if i < len(spins) and j < len(spins):
                    e += Jij * spins[i] * spins[j]
            return float(e)

        energies = [energy_of(sol) for sol in self.solutions]

        # Solutions meeting difficulty threshold
        valid_indices = [i for i, e in enumerate(energies) if e < requirements.difficulty_energy]
        valid_solutions = [self.solutions[i] for i in valid_indices]

        # Set computed fields
        self.energy = min(energies) if energies else None
        self.num_valid_solutions = len(valid_solutions)
        self.diversity = calculate_diversity(valid_solutions) if len(valid_solutions) >= 2 else 0.0


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
        def write_string(data: str) -> bytes:
            utf8_bytes = data.encode('utf-8')
            return struct.pack('!I', len(utf8_bytes)) + utf8_bytes

        def write_bytes(data: bytes) -> bytes:
            return struct.pack('!I', len(data)) + data

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
        def read_string(data: bytes, offset: int) -> tuple[str, int]:
            length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            string = data[offset:offset+length].decode('utf-8')
            return string, offset + length

        def read_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
            length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            bytes_data = data[offset:offset+length]
            return bytes_data, offset + length

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


@dataclass
class NextBlockRequirements:
    """Requirements that the next block must satisfy."""
    difficulty_energy: float
    min_diversity: float
    min_solutions: int
    timeout_to_difficulty_adjustment_decay: int

    def to_network(self) -> bytes:
        """Serialize to binary format."""
        result = b''
        result += struct.pack('!d', self.difficulty_energy)
        result += struct.pack('!d', self.min_diversity)
        result += struct.pack('!I', self.min_solutions)
        result += struct.pack('!i', self.timeout_to_difficulty_adjustment_decay)
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'NextBlockRequirements':
        """Deserialize from binary format."""
        offset = 0
        difficulty_energy = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_diversity = struct.unpack('!d', data[offset:offset+8])[0]
        offset += 8
        min_solutions = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        timeout_to_difficulty_adjustment_decay = struct.unpack('!i', data[offset:offset+4])[0]

        return cls(
            difficulty_energy=difficulty_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=timeout_to_difficulty_adjustment_decay
        )

@dataclass
class Block:
    header: BlockHeader

    miner_info: MinerInfo
    quantum_proof: QuantumProof
    next_block_requirements: NextBlockRequirements

    data: bytes  # Arbitrary data, eventually a merkle tree most likely.

    # NOTE: Maybe move this to a separate "NetworkBlock" class which returns
    #       Block, BlockHash, Signature on parse. 
    #      For now, we keep this coupled, but it is usually a not great idea to 
    #      couple signatures to the data they sign, network serialization, etc.

    # Computed Fields
    # Everything except the signature.
    raw: Optional[bytes]
    # Network hash (hash of the actual serialized network bytes)
    hash: Optional[bytes]
    # signature bytes
    signature: Optional[bytes]

    def to_network(self) -> bytes:
        """Serialize to binary format, excluding derived fields (raw, hash).
        External to this class, yous should only call this after finalization
        (compute_hash) and signature is added.
        """
        def write_bytes(data: bytes) -> bytes:
            return struct.pack('!I', len(data)) + data

        result = b''
        result += self.header.to_network()
        result += self.miner_info.to_network()
        result += self.quantum_proof.to_network()
        result += self.next_block_requirements.to_network()
        result += write_bytes(self.data)
        if self.signature: 
            result += write_bytes(self.signature)
        return result

    @classmethod
    def from_network(cls, data: bytes) -> 'Block':
        """Deserialize from binary format."""
        def read_string(data: bytes, offset: int) -> tuple[str, int]:
            length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            string = data[offset:offset+length].decode('utf-8')
            return string, offset + length

        def read_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
            length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            bytes_data = data[offset:offset+length]
            return bytes_data, offset + length

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
        next_block_requirements = NextBlockRequirements.from_network(req_data)
        req_size = len(next_block_requirements.to_network())
        offset += req_size

        # Read data and signature
        block_data, offset = read_bytes(data, offset)

        # Validate the block data is well formed.
        # TODO: Do this at the top of the function with a static signature size.
        header_block_data_hash = header.data_hash
        actual_data_hash = blake3(block_data).digest()
        if actual_data_hash != header_block_data_hash:
            raise ValueError("Data hash mismatch")

        # Create block with placeholder values for derived fields
        block = cls(
            header=header,
            miner_info=miner_info,
            quantum_proof=quantum_proof,
            next_block_requirements=next_block_requirements,
            data=block_data,
            raw=None,  # Will be computed
            hash=None,  # Will be computed
            signature=None  # Will be computed
        )

        # Compute derived fields
        block.finalize()

        signature, offset = read_bytes(data, offset)
        block.signature = signature

        return block

    def finalize(self):
        """Compute derived fields (raw, hash, etc) so the block can be signed."""
        if self.signature:
            raise ValueError("Block already signed")

        # Compute derived fields for quantum proof
        if self.quantum_proof:
            self.quantum_proof.compute_derived_fields(self.next_block_requirements, self)

        # Compute data hash (TBD merkle tree..)
        self.header.data_hash = blake3(self.data).digest()

        # Set raw to the network serialization
        self.raw = self.to_network()

        # Compute hash from raw bytes
        self.hash = blake3(self.raw).digest()

    def validate_block(self, previous_block: 'Block') -> bool:
        """Validate this block against the previous block requirements.

        This method:
        1. Validates the signature against miner_info
        2. Validates quantum_proof against previous_block.next_block_requirements
        3. Computes and sets energy, diversity, num_valid_solutions
        4. Returns validation result

        Args:
            previous_block: The previous block containing requirements

        Returns:
            True if block is valid, False otherwise
        """
        if not self.quantum_proof or not self.miner_info:
            return False

        # Get requirements from previous block
        requirements = previous_block.next_block_requirements
        if not requirements:
            return False

        # TODO: Implement signature validation here
        # For now, assume signature is valid if present
        if not self.signature:
            return False

        # Validate quantum proof against requirements
        return self._validate_quantum_proof(requirements)

    def _validate_quantum_proof(self, requirements: NextBlockRequirements) -> bool:
        """Validate quantum proof against requirements and compute metrics."""
        if not self.quantum_proof:
            return False

        solutions = self.quantum_proof.solutions
        if not solutions:
            return False

        seed = ising_seed_from_block(self.header.previous_hash, self.header.timestamp, self.header.index, self.quantum_proof.nonce)
        h, J = generate_ising_model_from_seed(seed, self.quantum_proof.nodes, self.quantum_proof.edges)

        def energy_of(solution: List[int]) -> float:
            spins = [1 if v > 0 else -1 for v in solution]
            e = 0.0
            for i in range(len(spins)):
                e += h.get(i, 0.0) * spins[i]
            for (i, j), Jij in J.items():
                if i < len(spins) and j < len(spins):
                    e += Jij * spins[i] * spins[j]
            return float(e)

        energies = [energy_of(sol) for sol in solutions]

        # Find solutions meeting energy threshold
        valid_indices = [i for i, e in enumerate(energies) if e < requirements.difficulty_energy]
        valid_solutions = [solutions[i] for i in valid_indices]

        if len(valid_solutions) < requirements.min_solutions:
            return False

        # Calculate diversity using shared utility
        diversity = calculate_diversity(valid_solutions)
        if diversity < requirements.min_diversity:
            return False

        # Set computed fields
        self.energy = min(energies) if energies else None
        self.diversity = diversity
        self.num_valid_solutions = len(valid_solutions)

        return True

    def _calculate_diversity(self, solutions: List[List[int]]) -> float:
        """Calculate average normalized Hamming distance between solutions."""
        if len(solutions) < 2:
            return 0.0

        distances = []
        n = len(solutions[0]) if solutions else 0

        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                # Calculate Hamming distance
                distance = sum(1 for a, b in zip(solutions[i], solutions[j]) if a != b)
                normalized_distance = distance / n if n > 0 else 0
                distances.append(normalized_distance)

        return sum(distances) / len(distances) if distances else 0.0




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
    with open(config_path, 'r') as f:
        genesis_data = json.load(f)

    # Use create_genesis_block to parse and validate the data
    genesis_block = create_genesis_block(genesis_data)

    print(f"Loaded genesis block from: {config_path.name}")
    # Note: adaptive_parameters not available in current NextBlockRequirements structure
    print(f"Mining parameters: difficulty_energy={genesis_block.next_block_requirements.difficulty_energy}")

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
        return parse_block_json(genesis_data)

    # Create default genesis block data
    default_genesis_data = {
        "index": 0,
        "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        "timestamp": int(time.time()),
        "data": "Genesis Block - Quip Protocol",
        "next_block_requirements": {
            "difficulty_energy": -1000.0,
            "min_diversity": 0.28,
            "min_solutions": 10,
            "timeout_to_difficulty_adjustment_decay": 600
        },
        "quantum_proof": None,
        "miner_info": None,
        "signature": None
    }

    # Use parse_block_json to create and validate the default genesis block
    return parse_block_json(default_genesis_data)


def parse_block_json(block_data: dict) -> Block:
    """Parse a block from JSON/dictionary format (for JSON derivation).

    Args:
        block_data: Dictionary containing block data

    Returns:
        Block object with network hash computed from binary serialization

    Raises:
        KeyError: If required fields are missing
        ValueError: If field values are invalid
    """
    # Handle both new format (flat) and legacy format (with header)
    if 'header' in block_data:
        # Legacy format with header
        header_data = block_data['header']
        index = header_data['index']
        previous_hash = header_data['previous_hash']
        timestamp = header_data['timestamp']
    else:
        # New flat format
        index = block_data['index']
        previous_hash = block_data['previous_hash']
        timestamp = block_data['timestamp']

    # Parse next block requirements (required)
    if 'next_block_requirements' not in block_data:
        raise KeyError("Missing required 'next_block_requirements' field")

    req_data = block_data['next_block_requirements']
    requirements = NextBlockRequirements(
        difficulty_energy=req_data['difficulty_energy'],
        min_diversity=req_data['min_diversity'],
        min_solutions=req_data['min_solutions'],
        timeout_to_difficulty_adjustment_decay=req_data.get('timeout_to_difficulty_adjustment_decay', 10)
    )

    # Parse data (required)
    if 'data' not in block_data:
        raise KeyError("Missing required 'data' field")

    # Parse optional quantum proof
    quantum_proof = None
    if 'quantum_proof' in block_data and block_data['quantum_proof']:
        proof_data = block_data['quantum_proof']
        quantum_proof = QuantumProof(
            nonce=proof_data['nonce'],
            nodes=proof_data.get('nodes', []),
            edges=[tuple(edge) for edge in proof_data.get('edges', [])],
            solutions=proof_data['solutions'],
            mining_time=proof_data['mining_time']
        )

    # Parse optional miner info
    miner_info = None
    if 'miner_info' in block_data and block_data['miner_info']:
        miner_data = block_data['miner_info']
        miner_info = MinerInfo(
            miner_id=miner_data['miner_id'],
            miner_type=miner_data['miner_type'],
            reward_address=miner_data['reward_address'],
            ecdsa_public_key=miner_data['ecdsa_public_key'],
            wots_public_key=miner_data['wots_public_key'],
            next_wots_public_key=miner_data['next_wots_public_key']
        )

    # Create header from parsed values - for genesis, previous_hash and timestamp are strings, need to convert
    if isinstance(previous_hash, str):
        previous_hash = bytes.fromhex(previous_hash) if previous_hash != "0000000000000000000000000000000000000000000000000000000000000000" else b'\\x00' * 32
    if isinstance(timestamp, float):
        timestamp = int(timestamp)

    # Compute data hash placeholder for genesis
    data_hash = blake3(block_data['data'].encode()).digest()

    header = BlockHeader(
        previous_hash=previous_hash,
        index=index,
        timestamp=timestamp,
        data_hash=data_hash
    )

    # Create placeholder objects for required fields if None
    if quantum_proof is None:
        quantum_proof = QuantumProof(nonce=0, nodes=[], edges=[], solutions=[], mining_time=0.0)

    if miner_info is None:
        miner_info = MinerInfo(
            miner_id="genesis",
            miner_type="genesis",
            reward_address=b'\\x00' * 32,
            ecdsa_public_key=b'\\x00' * 64,
            wots_public_key=b'\\x00' * 64,
            next_wots_public_key=b'\\x00' * 64
        )

    # Handle signature conversion
    signature_bytes = b''
    if 'signature' in block_data and block_data['signature']:
        sig = block_data['signature']
        if isinstance(sig, str):
            signature_bytes = bytes.fromhex(sig) if sig else b''
        else:
            signature_bytes = sig or b''

    # Create block
    block = Block(
        header=header,
        miner_info=miner_info,
        quantum_proof=quantum_proof,
        next_block_requirements=requirements,
        data=block_data['data'].encode(),
        raw=b'',  # Will be computed
        hash=b'',  # Will be computed
        signature=signature_bytes
    )

    # Set computed fields if present (but these should be derived from validation)
    if 'energy' in block_data:
        block.energy = block_data['energy']
    if 'diversity' in block_data:
        block.diversity = block_data['diversity']
    if 'num_valid_solutions' in block_data:
        block.num_valid_solutions = block_data['num_valid_solutions']

    # Compute network hash from binary serialization
    block.finalize()

    return block








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