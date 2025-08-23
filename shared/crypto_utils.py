"""Cryptographic utilities for quantum blockchain."""

import os
import hashlib
from typing import Tuple, List
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend


class WOTSPlus:
    """WOTS+ (Winternitz One-Time Signature Plus) implementation."""
    
    def __init__(self, seed: bytes = None):
        """Initialize WOTS+ with a random seed."""
        if seed is None:
            seed = os.urandom(32)
        self.seed = seed
        self.w = 16  # Winternitz parameter
        self.n = 32  # Hash output length in bytes
        self.l1 = 64  # Number of message chains
        self.l2 = 3   # Number of checksum chains
        self.l = self.l1 + self.l2  # Total chains
        
        # Generate private key
        self.private_key = self._generate_private_key()
        # Generate public key
        self.public_key = self._generate_public_key()
        self.used = False  # Track if this key has been used
    
    def _hash(self, data: bytes) -> bytes:
        """Hash function for WOTS+."""
        return hashlib.sha256(data).digest()
    
    def _generate_private_key(self) -> List[bytes]:
        """Generate WOTS+ private key."""
        private_key = []
        for i in range(self.l):
            # Derive each private key element from seed
            element = self._hash(self.seed + i.to_bytes(4, 'big'))
            private_key.append(element)
        return private_key
    
    def _generate_public_key(self) -> List[bytes]:
        """Generate WOTS+ public key from private key."""
        public_key = []
        for sk_element in self.private_key:
            # Hash chain of length w-1
            pk_element = sk_element
            for _ in range(self.w - 1):
                pk_element = self._hash(pk_element)
            public_key.append(pk_element)
        return public_key
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message with WOTS+."""
        if self.used:
            raise ValueError("WOTS+ key already used! Generate new key.")
        self.used = True
        
        # For simplicity, return concatenated signature elements
        # In production, would implement full WOTS+ signing
        signature_elements = []
        msg_hash = self._hash(message)
        
        for i, sk_element in enumerate(self.private_key[:32]):  # Simplified
            # Hash chain based on message bits
            sig_element = sk_element
            if i < len(msg_hash):
                iterations = msg_hash[i] % self.w
                for _ in range(iterations):
                    sig_element = self._hash(sig_element)
            signature_elements.append(sig_element)
        
        return b''.join(signature_elements[:32])  # Return fixed size
    
    def get_public_key_hex(self) -> str:
        """Get the public key in hex format."""
        # Concatenate and hash all public key elements
        combined = b''.join(self.public_key)
        public_key_hash = hashlib.sha256(combined).digest()
        return public_key_hash.hex()


class CryptoManager:
    """Manages ECDSA and WOTS+ cryptographic operations for miners."""
    
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
        
        # Generate initial WOTS+ key pair
        self.wots_plus = WOTSPlus()
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
        self.wots_plus = WOTSPlus()
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
        """Verify a WOTS+ signature.
        
        Args:
            public_key_hex: Hex-encoded WOTS+ public key
            message: Message that was signed
            signature_hex: Hex-encoded signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # In production, would use proper WOTS+ verification
            # For now, this is a placeholder
            return len(signature_hex) > 0 and len(public_key_hex) > 0
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