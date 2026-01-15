"""Tests for TOFU trust store."""

import asyncio
import os
import tempfile
import pytest

from shared.trust_store import TrustStore, TofuResult, compute_certificate_fingerprint


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_trust.db")


@pytest.fixture
async def trust_store(temp_db_path):
    """Create an initialized TrustStore for testing."""
    store = TrustStore(temp_db_path)
    await store.initialize()
    return store


class TestTrustStore:
    """Tests for TrustStore class."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db_path):
        """Test that trust store initializes correctly."""
        store = TrustStore(temp_db_path)
        await store.initialize()
        assert os.path.exists(temp_db_path)

    @pytest.mark.asyncio
    async def test_verify_fingerprint_new(self, trust_store):
        """Test TOFU on first connection returns NEW."""
        result = await trust_store.verify_fingerprint("peer1:8085", "abc123fingerprint")
        assert result == TofuResult.NEW

    @pytest.mark.asyncio
    async def test_verify_fingerprint_match(self, trust_store):
        """Test TOFU on subsequent connection with same fingerprint returns MATCH."""
        # First connection
        result1 = await trust_store.verify_fingerprint("peer1:8085", "abc123fingerprint")
        assert result1 == TofuResult.NEW

        # Second connection with same fingerprint
        result2 = await trust_store.verify_fingerprint("peer1:8085", "abc123fingerprint")
        assert result2 == TofuResult.MATCH

    @pytest.mark.asyncio
    async def test_verify_fingerprint_mismatch(self, trust_store):
        """Test TOFU on connection with different fingerprint returns MISMATCH."""
        # First connection
        result1 = await trust_store.verify_fingerprint("peer1:8085", "original_fingerprint")
        assert result1 == TofuResult.NEW

        # Second connection with different fingerprint
        result2 = await trust_store.verify_fingerprint("peer1:8085", "different_fingerprint")
        assert result2 == TofuResult.MISMATCH

    @pytest.mark.asyncio
    async def test_store_fingerprint(self, trust_store):
        """Test storing fingerprint directly."""
        # Store should succeed for new peer
        stored = await trust_store.store_fingerprint("newpeer:8085", "fingerprint123")
        assert stored is True

        # Store should fail for existing peer (no overwrite)
        stored_again = await trust_store.store_fingerprint("newpeer:8085", "different_fingerprint")
        assert stored_again is False

    @pytest.mark.asyncio
    async def test_get_fingerprint(self, trust_store):
        """Test retrieving stored fingerprint."""
        # Store a fingerprint
        await trust_store.store_fingerprint("peer:8085", "myfp123")

        # Retrieve it
        fp = await trust_store.get_fingerprint("peer:8085")
        assert fp == "myfp123"

        # Non-existent peer returns None
        fp_missing = await trust_store.get_fingerprint("unknown:8085")
        assert fp_missing is None

    @pytest.mark.asyncio
    async def test_remove_fingerprint(self, trust_store):
        """Test removing a fingerprint."""
        # Store and then remove
        await trust_store.store_fingerprint("peer:8085", "fp123")

        removed = await trust_store.remove_fingerprint("peer:8085")
        assert removed is True

        # Should be gone now
        fp = await trust_store.get_fingerprint("peer:8085")
        assert fp is None

        # Removing non-existent returns False
        removed_again = await trust_store.remove_fingerprint("peer:8085")
        assert removed_again is False

    @pytest.mark.asyncio
    async def test_list_trusted_peers(self, trust_store):
        """Test listing all trusted peers."""
        # Add multiple peers
        await trust_store.store_fingerprint("peer1:8085", "fp1")
        await trust_store.store_fingerprint("peer2:8086", "fp2")
        await trust_store.store_fingerprint("peer3:8087", "fp3")

        peers = await trust_store.list_trusted_peers()

        assert len(peers) == 3
        addresses = [p.peer_address for p in peers]
        assert "peer1:8085" in addresses
        assert "peer2:8086" in addresses
        assert "peer3:8087" in addresses

    @pytest.mark.asyncio
    async def test_get_peer_info(self, trust_store):
        """Test getting detailed peer info."""
        # Verify fingerprint to create entry with connection count
        await trust_store.verify_fingerprint("peer:8085", "fp123")
        await trust_store.verify_fingerprint("peer:8085", "fp123")

        info = await trust_store.get_peer_info("peer:8085")

        assert info is not None
        assert info.peer_address == "peer:8085"
        assert info.fingerprint == "fp123"
        assert info.connection_count == 2
        assert info.first_seen > 0
        assert info.last_seen >= info.first_seen

    @pytest.mark.asyncio
    async def test_clear_all(self, trust_store):
        """Test clearing all fingerprints."""
        # Add some peers
        await trust_store.store_fingerprint("peer1:8085", "fp1")
        await trust_store.store_fingerprint("peer2:8086", "fp2")

        # Clear all
        count = await trust_store.clear_all()
        assert count == 2

        # Should be empty now
        peers = await trust_store.list_trusted_peers()
        assert len(peers) == 0

    @pytest.mark.asyncio
    async def test_connection_count_increments(self, trust_store):
        """Test that connection count increments on successful matches."""
        # Multiple verifications
        for i in range(5):
            await trust_store.verify_fingerprint("peer:8085", "constant_fp")

        info = await trust_store.get_peer_info("peer:8085")
        assert info.connection_count == 5


class TestComputeCertificateFingerprint:
    """Tests for fingerprint computation."""

    def test_compute_fingerprint(self):
        """Test SHA-256 fingerprint computation."""
        # Known test vector
        cert_der = b"test certificate data"
        fp = compute_certificate_fingerprint(cert_der)

        # Should be 64 hex characters (SHA-256 = 256 bits = 64 hex chars)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic."""
        cert_der = b"same certificate"
        fp1 = compute_certificate_fingerprint(cert_der)
        fp2 = compute_certificate_fingerprint(cert_der)
        assert fp1 == fp2

    def test_fingerprint_different_for_different_certs(self):
        """Test that different certificates produce different fingerprints."""
        fp1 = compute_certificate_fingerprint(b"cert1")
        fp2 = compute_certificate_fingerprint(b"cert2")
        assert fp1 != fp2
