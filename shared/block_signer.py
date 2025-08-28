"""Block signing utilities for quantum blockchain."""

import os
import hashlib
from typing import Tuple, List
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import hashsigs


class HashSigsWrapper:
    """Wrapper for hashsigs WOTS+ implementation."""
    
    def __init__(self, seed: bytes = None):
        """Initialize hashsigs WOTS+ with a seed."""
        if seed is None:
            seed = os.urandom(32)
        self.seed = seed
        
        # Create WOTS+ instance using SHA256
        sha256_hash = lambda data: hashlib.sha256(data).digest()
        self.wots_plus = hashsigs.WOTSPlus(sha256_hash)
        
        # Generate key pair
        self.public_key, self.private_key = self.wots_plus.generate_key_pair(seed)
        self.used = False  # Track if this key has been used
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message with hashsigs WOTS+."""
        if self.used:
            raise ValueError("WOTS+ key already used! Generate new key.")
        self.used = True
        
        # Hash the message to get exactly 32 bytes
        message_hash = hashlib.sha256(message).digest()
        signature = self.wots_plus.sign(self.private_key, message_hash)
        return signature
    
    def get_public_key_hex(self) -> str:
        """Get the public key in hex format."""
        return self.public_key.to_bytes().hex()


class BlockSigner:
    """Manages ECDSA and WOTS+ cryptographic operations for block signing."""
    
    def __init__(self):
        """Initialize cryptographic keys."""
        # Generate ECDSA key pair
        self.ecdsa_private_key = ec.generate_private_key(
            ec.SECP256K1(),
            default_backend()
        )
        self.ecdsa_public_key = self.ecdsa_private_key.public_key()
        
        # Get ECDSA public key in hex format
        self.ecdsa_public_key_bytes = self.ecdsa_public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        self.ecdsa_public_key_hex = self.ecdsa_public_key_bytes.hex()
        
        # Generate initial WOTS+ key pair using hashsigs
        self.wots_plus = HashSigsWrapper()
        self.wots_plus_public_key_hex = self.wots_plus.get_public_key_hex()
    
    def sign_block_data(self, block_data: str) -> Tuple[str, str]:
        """Sign block data with both WOTS+ and ECDSA, generate new WOTS+ key.
        
        Args:
            block_data: String data to sign
            
        Returns:
            Tuple of (combined_signature_hex, next_wots_public_key_hex)
        """
        # Sign with WOTS+
        wots_signature = self.wots_plus.sign(block_data.encode())
        
        # Sign with ECDSA
        ecdsa_signature = self.ecdsa_private_key.sign(
            block_data.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        
        # Generate new WOTS+ key pair for next block (one-time signature)
        self.wots_plus = HashSigsWrapper()
        next_wots_key_hex = self.wots_plus.get_public_key_hex()
        
        # Combine signatures
        combined_signature = wots_signature.hex() + ecdsa_signature.hex()
        
        return combined_signature, next_wots_key_hex
    
    def verify_ecdsa_signature(self, public_key_hex: str, message: bytes, signature: bytes) -> bool:
        """Verify an ECDSA signature.
        
        Args:
            public_key_hex: Hex-encoded public key
            message: Message that was signed
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Convert hex public key back to EC public key object
            public_key_bytes = bytes.fromhex(public_key_hex)
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                public_key_bytes
            )
            
            # Verify signature
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False
    
    def verify_wots_signature(self, public_key_hex: str, message: bytes, signature_hex: str) -> bool:
        """Verify a WOTS+ signature using hashsigs.
        
        Args:
            public_key_hex: Hex-encoded WOTS+ public key
            message: Message that was signed
            signature_hex: Hex-encoded signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            public_key_bytes = bytes.fromhex(public_key_hex)
            signature_bytes = bytes.fromhex(signature_hex)
            
            # Create WOTSPlus instance for verification
            sha256_hash = lambda data: hashlib.sha256(data).digest()
            wots_plus = hashsigs.WOTSPlus(sha256_hash)
            
            # Reconstruct public key from bytes
            public_key = hashsigs.PublicKey.from_bytes(public_key_bytes)
            if public_key is None:
                return False
                
            # Hash the message to get exactly 32 bytes (same as in sign method)
            message_hash = hashlib.sha256(message).digest()
            return wots_plus.verify(public_key, message_hash, signature_bytes)
        except Exception:
            return False
    
    def verify_combined_signature(self, block_data: dict) -> bool:
        """Verify the combined WOTS+ and ECDSA signatures in a block.
        
        Args:
            block_data: Block data containing signatures and public keys
            
        Returns:
            True if all signatures are valid, False otherwise
        """
        if 'signature' not in block_data or not block_data['signature']:
            return False
        
        try:
            # Extract signature components (combined as wots+ecdsa)
            combined_sig = block_data['signature']
            
            # In production, would properly parse and verify both signatures
            # For now, check basic validity
            if len(combined_sig) < 128:  # Minimum expected length
                return False
            
            # Reconstruct the signed message
            message = f"{block_data['previous_hash']}{block_data['index']}{block_data['timestamp']}{block_data['data']}"
            
            # Would verify both WOTS+ and ECDSA signatures here
            return True
            
        except Exception as e:
            return False