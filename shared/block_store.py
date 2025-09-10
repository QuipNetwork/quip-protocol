"""
Simple block storage system for chain epochs.

Provides persistent storage for blockchain epochs with automatic naming and directory management.
"""

import json
import gzip
import pickle
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from shared.block import Block

logger = logging.getLogger(__name__)


class ChainStorageError(Exception):
    """Exception for chain storage operations."""
    pass


class BlockStore:
    """
    Storage manager for chain epochs.
    
    Handles saving and loading blockchain epochs with automatic file naming
    based on epoch data and timestamps.
    """
    
    def __init__(self, storage_dir: str = "epoch_storage", format_type: str = "pickle", compress: bool = True):
        """
        Initialize the block store.
        
        Args:
            storage_dir: Directory to store epoch files
            format_type: Default format ('json' or 'pickle')
            compress: Default compression setting
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.format_type = format_type
        self.compress = compress
        
        logger.info(f"BlockStore initialized: {self.storage_dir} (format: {format_type}, compress: {compress})")

    def save(self, epoch_chain: List[Block]) -> str:
        """
        Save an epoch chain with automatic filename generation.
        
        Args:
            epoch_chain: List of blocks to save
            
        Returns:
            Path to saved file
        """
        if not epoch_chain:
            raise ChainStorageError("Cannot save empty epoch chain")
        
        # Generate filename based on timestamp and epoch info
        timestamp = int(time.time())
        first_block = epoch_chain[0]
        last_block = epoch_chain[-1]
        
        # Create filename: epoch_<timestamp>_<first_index>_<last_index>
        filename = f"epoch_{timestamp}_{first_block.header.index}_{last_block.header.index}"
        
        if self.format_type == 'json':
            return save_epoch_json(epoch_chain, self.storage_dir / f"{filename}.json", self.compress)
        else:
            return save_epoch_pickle(epoch_chain, self.storage_dir / f"{filename}.pkl", self.compress)

    def load(self, filename: str) -> Tuple[List[Block], Dict[str, Any]]:
        """
        Load an epoch chain from file.
        
        Args:
            filename: Name of file to load (relative to storage_dir)
            
        Returns:
            Tuple of (chain, metadata)
        """
        file_path = self.storage_dir / filename
        
        # Auto-detect format from filename
        if '.json' in filename:
            return load_epoch_json(str(file_path))
        else:
            return load_epoch_pickle(str(file_path))

    def list_epochs(self) -> List[str]:
        """
        List all available epoch files.
        
        Returns:
            List of epoch filenames
        """
        epochs = []
        for file_path in self.storage_dir.glob("epoch_*"):
            if file_path.is_file():
                epochs.append(file_path.name)
        return sorted(epochs)


def save_epoch_json(chain: List[Block], file_path: Path, compress: bool = True) -> str:
    """
    Save epoch chain to JSON format.
    
    Args:
        chain: List of blocks to save
        file_path: Output file path
        compress: Whether to compress with gzip
        
    Returns:
        Path to saved file
    """
    try:
        # Convert blocks to dictionaries
        chain_data = []
        for block in chain:
            chain_data.append(block.to_json())
        
        # Create epoch data with metadata
        epoch_data = {
            'timestamp': time.time(),
            'chain_length': len(chain),
            'format': 'json',
            'compressed': compress,
            'first_block_index': chain[0].header.index if chain else None,
            'last_block_index': chain[-1].header.index if chain else None,
            'chain': chain_data
        }
        
        json_str = json.dumps(epoch_data, indent=2, default=str)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with atomic operation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', 
                                       dir=file_path.parent, delete=False) as tmp_file:
            if compress:
                final_path = file_path.with_suffix(file_path.suffix + '.gz')
                with gzip.open(tmp_file.name + '.gz', 'wt', encoding='utf-8') as gz_file:
                    gz_file.write(json_str)
                tmp_path = tmp_file.name + '.gz'
                Path(tmp_file.name).unlink(missing_ok=True)
            else:
                tmp_file.write(json_str)
                tmp_path = tmp_file.name
                final_path = file_path
        
        # Atomic move
        shutil.move(tmp_path, final_path)
        
        logger.info(f"Epoch saved to JSON: {final_path} ({len(chain)} blocks)")
        return str(final_path)
        
    except Exception as e:
        logger.error(f"Error saving epoch to JSON: {e}")
        raise ChainStorageError(f"Failed to save epoch: {e}")


def load_epoch_json(file_path: str) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Load epoch chain from JSON format.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (chain, metadata)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ChainStorageError(f"File not found: {file_path}")
        
        # Handle compressed files
        is_compressed = str(path).endswith('.gz')
        
        if is_compressed:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        metadata = {k: v for k, v in data.items() if k != 'chain'}
        chain_data = data.get('chain', [])
        
        # Reconstruct blocks
        chain = []
        for block_dict in chain_data:
            try:
                block = Block.from_json(block_dict)
                chain.append(block)
            except Exception as e:
                logger.warning(f"Failed to reconstruct block: {e}")
        
        logger.info(f"Epoch loaded from JSON: {file_path} ({len(chain)} blocks)")
        return chain, metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        raise ChainStorageError(f"Corrupted JSON file: {e}")
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise ChainStorageError(f"Failed to load epoch: {e}")


def save_epoch_pickle(chain: List[Block], file_path: Path, compress: bool = True) -> str:
    """
    Save epoch chain to pickle format.
    
    Args:
        chain: List of blocks to save
        file_path: Output file path
        compress: Whether to compress with gzip
        
    Returns:
        Path to saved file
    """
    try:
        # Create epoch data with metadata
        epoch_data = {
            'timestamp': time.time(),
            'chain_length': len(chain),
            'format': 'pickle',
            'compressed': compress,
            'first_block_index': chain[0].header.index if chain else None,
            'last_block_index': chain[-1].header.index if chain else None,
            'chain': chain
        }
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with atomic operation
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.tmp', 
                                       dir=file_path.parent, delete=False) as tmp_file:
            if compress:
                final_path = file_path.with_suffix(file_path.suffix + '.gz')
                with gzip.open(tmp_file.name + '.gz', 'wb') as gz_file:
                    pickle.dump(epoch_data, gz_file, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = tmp_file.name + '.gz'
                Path(tmp_file.name).unlink(missing_ok=True)
            else:
                pickle.dump(epoch_data, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = tmp_file.name
                final_path = file_path
        
        # Atomic move
        shutil.move(tmp_path, final_path)
        
        logger.info(f"Epoch saved to pickle: {final_path} ({len(chain)} blocks)")
        return str(final_path)
        
    except Exception as e:
        logger.error(f"Error saving epoch to pickle: {e}")
        raise ChainStorageError(f"Failed to save epoch: {e}")


def load_epoch_pickle(file_path: str) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Load epoch chain from pickle format.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Tuple of (chain, metadata)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ChainStorageError(f"File not found: {file_path}")
        
        # Handle compressed files
        is_compressed = str(path).endswith('.gz')
        
        if is_compressed:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        metadata = {k: v for k, v in data.items() if k != 'chain'}
        chain = data.get('chain', [])
        
        logger.info(f"Epoch loaded from pickle: {file_path} ({len(chain)} blocks)")
        return chain, metadata
        
    except pickle.UnpicklingError as e:
        logger.error(f"Invalid pickle: {e}")
        raise ChainStorageError(f"Corrupted pickle file: {e}")
    except Exception as e:
        logger.error(f"Error loading pickle: {e}")
        raise ChainStorageError(f"Failed to load epoch: {e}")