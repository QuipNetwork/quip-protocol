"""Block signing utilities for quantum blockchain."""

import hashlib
import os
from blake3 import blake3
from typing import Tuple, List, Optional
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import hashsigs


class BlockSigner:
    """Manages ECDSA and WOTS+ cryptographic operations for block signing."""

    def __init__(self, seed: Optional[bytes] = None):
        """Initialize cryptographic keys.
        
        Args:
            seed: Optional seed for deterministic key generation (random if None)
        """
        # Use provided seed or generate random one
        if seed is None:
            seed = os.urandom(32)
        
        # Generate ECDSA key pair deterministically from seed if provided
        if len(seed) == 32:
            # Use first half of seed for ECDSA key generation
            ecdsa_seed = int.from_bytes(seed[:16], byteorder='big')
            # Use a deterministic approach for ECDSA key generation
            private_value = ecdsa_seed % ec.SECP256K1().key_size
            self.ecdsa_private_key = ec.derive_private_key(
                private_value,
                ec.SECP256K1(),
                default_backend()
            )
        else:
            # Fallback to random generation
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
        # WOTS+ requires a 32-byte private seed; if seed provided, reuse it; otherwise generate 32 bytes.
        self.wots_seed = seed if len(seed) == 32 else os.urandom(32)
        keccak_hash = lambda data: blake3(data).digest()
        self.wots_plus = hashsigs.WOTSPlus(keccak_hash)
        self.wots_plus_public_key, self.wots_plus_private_key = self.wots_plus.generate_key_pair(self.wots_seed)

        self.next_wots_seed = blake3(self.wots_seed).digest()
        self.next_wots_plus_public_key, self.next_wots_plus_private_key = self.wots_plus.generate_key_pair(self.next_wots_seed)
    
    # FIXME: Because we need to sign lots of candidate messages before iterating our key we need to switch
    #.       either to an XMSS system, SPHINCS, etc for this to be secure.
    def iterate_wots_key(self):
        """Iterate to the next WOTS+ key pair."""
        self.wots_seed = self.next_wots_seed
        self.wots_plus_public_key, self.wots_plus_private_key = self.next_wots_plus_public_key, self.next_wots_plus_private_key
        self.next_wots_seed = blake3(self.wots_seed).digest()
        self.next_wots_plus_public_key, self.next_wots_plus_private_key = self.wots_plus.generate_key_pair(self.next_wots_seed)
    
    def sign_block_data(self, block_data: bytes) -> bytes:
        """Sign block data with both WOTS+ and ECDSA, generate new WOTS+ key.

        Args:
            block_data: String data to sign

        Returns:
            Tuple of (combined_signature_hex, next_wots_public_key_hex)
        """
        # Hash the message with keccak256 and sign with WOTS+
        # FIXME: this is a hack as the hashsigs library doesn't do the hashing for us when it should.
        message_hash = blake3(block_data).digest()
        wots_signature = self.wots_plus.sign(self.wots_plus_private_key, message_hash)

        # Sign with ECDSA
        ecdsa_signature = self.ecdsa_private_key.sign(
            block_data,
            ec.ECDSA(hashes.SHA256())
        )
        # Combine signatures
        combined_signature = ecdsa_signature + wots_signature 

        return combined_signature

    def verify_ecdsa_signature(self, ecdsa_public_key: bytes, message: bytes, signature: bytes) -> bool:
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
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                ecdsa_public_key
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
    
    def verify_wots_signature(self, wots_public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a WOTS+ signature using hashsigs with keccak.

        Args:
            public_key_hex: Hex-encoded WOTS+ public key
            message: Message that was signed
            signature_hex: Hex-encoded signature

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            public_key = hashsigs.PublicKey.from_bytes(wots_public_key)
            if public_key is None:
                raise ValueError("Invalid WOTS+ public key")
            
            # Hash the message with keccak (same as in sign method)
            # FIXME: should not need to do this...
            message_hash = blake3(message).digest()
            return self.wots_plus.verify(public_key, message_hash, signature)
        except Exception:
            return False
    
    def verify_combined_signature(self, ecdsa_public_key: bytes, wots_public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify the combined WOTS+ and ECDSA signatures in a block.

        Args:
            block_data: Block data containing signatures and public keys

        Returns:
            True if all signatures are valid, False otherwise
        """
        ecdsa_signature = signature[:64]
        wots_signature = signature[64:]
        return self.verify_ecdsa_signature(ecdsa_public_key, message, ecdsa_signature) and self.verify_wots_signature(wots_public_key, message, wots_signature)