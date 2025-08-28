"""Block data structures and parsing utilities for quantum blockchain."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def load_genesis_block(config_file: Optional[str] = None) -> Dict:
    """Load genesis block and mining parameters from a JSON file.
    
    Args:
        config_file: Path to genesis config file. If None, defaults to genesis_block.json
    
    Returns:
        Dictionary with genesis block and mining parameters
        
    Raises:
        FileNotFoundError: If the specified config file is not found
        KeyError: If required configuration keys are missing
        json.JSONDecodeError: If JSON is malformed
    """
    if config_file is None:
        config_file = Path(__file__).parent.parent / "genesis_block.json"
    else:
        config_file = Path(config_file)
        if not config_file.is_absolute():
            config_file = Path(__file__).parent.parent / config_file
    
    if not config_file.exists():
        raise FileNotFoundError(f"Genesis configuration not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    if 'genesis_block' not in config:
        raise KeyError("Missing 'genesis_block' in genesis configuration")
    if 'mining_parameters' not in config:
        raise KeyError("Missing 'mining_parameters' in genesis configuration")
    
    result = {
        'genesis_block': config['genesis_block'].copy(),
        'mining_parameters': config['mining_parameters'].copy()
    }
    
    print(f"Loaded genesis configuration from: {config_file.name}")
    if 'description' in result['mining_parameters']:
        print(f"Mining parameters: {result['mining_parameters']['description']}")
    
    return result


def _parse_block_header(block_header: str) -> Dict:
    """Parse block header string back into block_data components.
    
    Args:
        block_header: Format is f"{previous_hash}{index}{timestamp}{data}"
        
    Returns:
        Dict with previous_hash, index, timestamp, data
        
    Raises:
        ValueError: If block_header cannot be parsed properly
    """
    # Format: f"{previous_hash}{index}{timestamp}{data}"
    # previous_hash is always 64 hex characters
    if len(block_header) < 64:
        raise ValueError(f"Block header too short: expected at least 64 chars, got {len(block_header)}")
        
    previous_hash = block_header[:64]
    remainder = block_header[64:]
    
    # Find where index ends and timestamp begins
    # Index is typically small, so look for the first decimal point (timestamp)
    timestamp_start = -1
    for i, char in enumerate(remainder):
        if char == '.' and i > 0:  # Found decimal point, likely timestamp
            timestamp_start = i
            break
    
    if timestamp_start <= 0:
        raise ValueError(f"Could not find timestamp in block header remainder: {remainder}")
        
    try:
        index = int(remainder[:timestamp_start])
    except ValueError:
        raise ValueError(f"Could not parse index from: {remainder[:timestamp_start]}")
        
    timestamp_and_data = remainder[timestamp_start:]
    
    # Find where timestamp ends - look for end of float
    timestamp_end = -1
    for i, char in enumerate(timestamp_and_data):
        if not (char.isdigit() or char == '.'):
            timestamp_end = i
            break
    
    if timestamp_end > 0:
        try:
            timestamp = float(timestamp_and_data[:timestamp_end])
        except ValueError:
            raise ValueError(f"Could not parse timestamp from: {timestamp_and_data[:timestamp_end]}")
        data = timestamp_and_data[timestamp_end:]
    else:
        # Timestamp goes to end
        try:
            timestamp = float(timestamp_and_data)
        except ValueError:
            raise ValueError(f"Could not parse timestamp from: {timestamp_and_data}")
        data = ""
    
    return {
        "previous_hash": previous_hash,
        "index": index, 
        "timestamp": timestamp,
        "data": data,
    }


@dataclass
class Block:
    index: int
    timestamp: float
    data: str
    previous_hash: str
    nonce: int
    quantum_proof: Optional[List[List[int]]] = None  # Multiple solutions
    energy: Optional[float] = None
    diversity: Optional[float] = None  # Average Hamming distance
    num_valid_solutions: Optional[int] = None
    miner_id: Optional[str] = None  # e.g., QPU-1, SA-2
    miner_type: Optional[str] = None  # QPU or SA
    mining_time: Optional[float] = None
    signature: Optional[str] = None  # WOTS+ signature signed by ECDSA
    reward_address: Optional[str] = None  # ECDSA public key for rewards
    miner_ecdsa_public_key: Optional[str] = None  # Miner's ECDSA public key
    miner_wots_plus_public_key: Optional[str] = None  # Miner's current WOTS+ public key
    hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute the hash of the block with new format."""
        # New format: f"{previous_hash}{index}{timestamp}{data}{signature}{reward_address}{miner_ecdsa_public_key}{miner_wots_plus_public_key}"
        block_string = f"{self.previous_hash}{self.index}{self.timestamp}{self.data}{self.signature or ''}{self.reward_address or ''}{self.miner_ecdsa_public_key or ''}{self.miner_wots_plus_public_key or ''}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __post_init__(self):
        """Compute hash after initialization."""
        if self.hash is None:
            self.hash = self.compute_hash()