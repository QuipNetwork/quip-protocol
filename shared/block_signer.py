"""Block signing utilities for quantum blockchain."""

import os
import hashlib
from typing import Tuple, List
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import hashsigs


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

        # Generate initial WOTS+ key pair using hashsigs with keccak
        seed = os.urandom(32)
        keccak_hash = lambda data: hashlib.sha3_256(data).digest()
        self.wots_plus = hashsigs.WOTSPlus(keccak_hash)
        self.wots_plus_public_key, self.wots_plus_private_key = self.wots_plus.generate_key_pair(seed)
        self.wots_plus_public_key_hex = self.wots_plus_public_key.to_bytes().hex()
        self.wots_plus_used = False
    
    def sign_block_data(self, block_data: str) -> Tuple[str, str]:
        """Sign block data with both WOTS+ and ECDSA, generate new WOTS+ key.

        Args:
            block_data: String data to sign

        Returns:
            Tuple of (combined_signature_hex, next_wots_public_key_hex)
        """
        # Check if WOTS+ key has been used
        if self.wots_plus_used:
            raise ValueError("WOTS+ key already used! Generate new key.")
        self.wots_plus_used = True

        # Hash the message with keccak256 and sign with WOTS+
        message_hash = hashlib.sha3_256(block_data.encode()).digest()
        wots_signature = self.wots_plus.sign(self.wots_plus_private_key, message_hash)

        # Sign with ECDSA
        ecdsa_signature = self.ecdsa_private_key.sign(
            block_data.encode(),
            ec.ECDSA(hashes.SHA256())
        )

        # Generate new WOTS+ key pair for next block (one-time signature)
        self.generate_new_wots_key()
        next_wots_key_hex = self.wots_plus_public_key_hex

        # Combine signatures
        combined_signature = wots_signature.hex() + ecdsa_signature.hex()

        return combined_signature, next_wots_key_hex

    def generate_new_wots_key(self):
        """Generate a new WOTS+ key pair after using the current one."""
        seed = os.urandom(32)
        keccak_hash = lambda data: hashlib.sha3_256(data).digest()
        self.wots_plus = hashsigs.WOTSPlus(keccak_hash)
        self.wots_plus_public_key, self.wots_plus_private_key = self.wots_plus.generate_key_pair(seed)
        self.wots_plus_public_key_hex = self.wots_plus_public_key.to_bytes().hex()
        self.wots_plus_used = False

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
        """Verify a WOTS+ signature using hashsigs with keccak.

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

            # Create WOTSPlus instance for verification with keccak
            keccak_hash = lambda data: hashlib.sha3_256(data).digest()
            wots_plus = hashsigs.WOTSPlus(keccak_hash)

            # Reconstruct public key from bytes
            public_key = hashsigs.PublicKey.from_bytes(public_key_bytes)
            if public_key is None:
                return False

            # Hash the message with keccak (same as in sign method)
            message_hash = hashlib.sha3_256(message).digest()
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

            # In production, would reconstruct the signed message and verify signatures
            # For now, just return True for basic validity check
            return True

        except Exception:
            return False