"""Tests for certificate manager."""

import os
import tempfile
import pytest

from shared.certificate_manager import CertificateManager


class TestCertificateManager:
    """Tests for CertificateManager class."""

    @pytest.mark.asyncio
    async def test_generate_self_signed(self):
        """Test self-signed certificate generation."""
        config = {}
        manager = CertificateManager(config)

        cert_path, key_path = await manager._generate_self_signed()

        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)

        # Verify cert is valid PEM
        with open(cert_path, "rb") as f:
            cert_data = f.read()
        assert b"BEGIN CERTIFICATE" in cert_data

        # Verify key is valid PEM
        with open(key_path, "rb") as f:
            key_data = f.read()
        assert b"BEGIN EC PRIVATE KEY" in key_data or b"BEGIN PRIVATE KEY" in key_data

    @pytest.mark.asyncio
    async def test_check_existing_cert_not_configured(self):
        """Test that missing cert config returns None."""
        config = {}
        manager = CertificateManager(config)

        result = await manager._check_existing_cert()
        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_cert_file_not_found(self):
        """Test that missing cert file returns None."""
        config = {
            "rest_tls_cert_file": "/nonexistent/cert.pem",
            "rest_tls_key_file": "/nonexistent/key.pem"
        }
        manager = CertificateManager(config)

        result = await manager._check_existing_cert()
        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_cert_valid(self):
        """Test that valid existing cert is returned."""
        # First generate a cert
        config = {}
        manager = CertificateManager(config)

        cert_path, key_path = await manager._generate_self_signed()

        # Now test with those paths configured
        config2 = {
            "rest_tls_cert_file": cert_path,
            "rest_tls_key_file": key_path
        }
        manager2 = CertificateManager(config2)

        result = await manager2._check_existing_cert()
        assert result is not None
        assert result[0] == cert_path
        assert result[1] == key_path

    @pytest.mark.asyncio
    async def test_get_certificate_fallback_to_self_signed(self):
        """Test that get_certificate falls back to self-signed."""
        config = {}
        manager = CertificateManager(config)

        cert_path, key_path = await manager.get_certificate()

        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)
        assert "rest_api_cert.pem" in cert_path

    @pytest.mark.asyncio
    async def test_check_certificate_validity(self):
        """Test certificate validity checking."""
        config = {}
        manager = CertificateManager(config)

        # Generate a valid cert
        cert_path, _ = await manager._generate_self_signed()

        # Should be valid
        is_valid = manager._check_certificate_validity(cert_path)
        assert is_valid is True

    def test_check_certificate_validity_invalid_path(self):
        """Test certificate validity check with invalid path."""
        config = {}
        manager = CertificateManager(config)

        is_valid = manager._check_certificate_validity("/nonexistent/cert.pem")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_key_file_permissions(self):
        """Test that generated key file has restrictive permissions."""
        config = {}
        manager = CertificateManager(config)

        _, key_path = await manager._generate_self_signed()

        # Check permissions (should be 0600)
        stat = os.stat(key_path)
        mode = stat.st_mode & 0o777
        assert mode == 0o600

    def test_tls_fallback_keys(self):
        """Test that shared TLS keys are used as fallback."""
        config = {
            "tls_cert_file": "/path/to/shared/cert.pem",
            "tls_key_file": "/path/to/shared/key.pem",
        }
        manager = CertificateManager(config)
        assert manager.cert_file == "/path/to/shared/cert.pem"
        assert manager.key_file == "/path/to/shared/key.pem"

    def test_rest_tls_overrides_shared(self):
        """Test that rest_tls_* keys override shared tls_* keys."""
        config = {
            "tls_cert_file": "/path/to/shared/cert.pem",
            "tls_key_file": "/path/to/shared/key.pem",
            "rest_tls_cert_file": "/path/to/rest/cert.pem",
            "rest_tls_key_file": "/path/to/rest/key.pem",
        }
        manager = CertificateManager(config)
        assert manager.cert_file == "/path/to/rest/cert.pem"
        assert manager.key_file == "/path/to/rest/key.pem"
