"""Tests for certificate manager."""

import os
import tempfile
import pytest

from shared.certificate_manager import CertificateManager


@pytest.fixture
def temp_cert_dir():
    """Create a temporary directory for certificates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCertificateManager:
    """Tests for CertificateManager class."""

    @pytest.mark.asyncio
    async def test_generate_self_signed(self, temp_cert_dir):
        """Test self-signed certificate generation."""
        config = {
            "cert_dir": temp_cert_dir
        }
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
    async def test_generate_self_signed_with_domain(self, temp_cert_dir):
        """Test self-signed cert generation with domain name."""
        config = {
            "cert_dir": temp_cert_dir,
            "rest_domain": "test.example.com"
        }
        manager = CertificateManager(config)

        cert_path, key_path = await manager._generate_self_signed()

        assert os.path.exists(cert_path)

        # Verify certificate contains the domain in SAN
        from cryptography import x509
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        dns_names = san.value.get_values_for_type(x509.DNSName)
        assert "test.example.com" in dns_names

    @pytest.mark.asyncio
    async def test_check_existing_cert_not_configured(self, temp_cert_dir):
        """Test that missing cert config returns None."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        result = await manager._check_existing_cert()
        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_cert_file_not_found(self, temp_cert_dir):
        """Test that missing cert file returns None."""
        config = {
            "cert_dir": temp_cert_dir,
            "rest_tls_cert_file": "/nonexistent/cert.pem",
            "rest_tls_key_file": "/nonexistent/key.pem"
        }
        manager = CertificateManager(config)

        result = await manager._check_existing_cert()
        assert result is None

    @pytest.mark.asyncio
    async def test_check_existing_cert_valid(self, temp_cert_dir):
        """Test that valid existing cert is returned."""
        # First generate a cert
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        cert_path, key_path = await manager._generate_self_signed()

        # Now test with those paths configured
        config2 = {
            "cert_dir": temp_cert_dir,
            "rest_tls_cert_file": cert_path,
            "rest_tls_key_file": key_path
        }
        manager2 = CertificateManager(config2)

        result = await manager2._check_existing_cert()
        assert result is not None
        assert result[0] == cert_path
        assert result[1] == key_path

    def test_port_availability_check(self, temp_cert_dir):
        """Test port availability checking."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        # Port 80 is typically not available without root
        # This test just verifies the method runs without error
        result = manager._is_port_available(80)
        # Result depends on system state, just verify it's a bool
        assert isinstance(result, bool)

        # High port should typically be available
        result_high = manager._is_port_available(59999)
        # May or may not be available depending on system

    @pytest.mark.asyncio
    async def test_get_certificate_fallback_to_self_signed(self, temp_cert_dir):
        """Test that get_certificate falls back to self-signed."""
        config = {
            "cert_dir": temp_cert_dir
            # No existing cert, no ACME config
        }
        manager = CertificateManager(config)

        cert_path, key_path = await manager.get_certificate()

        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)
        assert "rest_api_cert.pem" in cert_path

    @pytest.mark.asyncio
    async def test_check_certificate_validity(self, temp_cert_dir):
        """Test certificate validity checking."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        # Generate a valid cert
        cert_path, _ = await manager._generate_self_signed()

        # Should be valid
        is_valid = manager._check_certificate_validity(cert_path)
        assert is_valid is True

    def test_check_certificate_validity_invalid_path(self, temp_cert_dir):
        """Test certificate validity check with invalid path."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        is_valid = manager._check_certificate_validity("/nonexistent/cert.pem")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_key_file_permissions(self, temp_cert_dir):
        """Test that generated key file has restrictive permissions."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)

        _, key_path = await manager._generate_self_signed()

        # Check permissions (should be 0600)
        stat = os.stat(key_path)
        mode = stat.st_mode & 0o777
        assert mode == 0o600


class TestCertificateManagerConfig:
    """Tests for certificate manager configuration handling."""

    def test_default_cert_dir(self):
        """Test default certificate directory."""
        config = {}
        manager = CertificateManager(config)
        assert "~/.quip/certs" in manager.cert_dir or ".quip/certs" in manager.cert_dir

    def test_custom_cert_dir(self, temp_cert_dir):
        """Test custom certificate directory."""
        config = {"cert_dir": temp_cert_dir}
        manager = CertificateManager(config)
        assert manager.cert_dir == temp_cert_dir

    def test_acme_staging_config(self):
        """Test ACME staging configuration."""
        config = {
            "acme_staging": True,
            "rest_domain": "test.example.com",
            "acme_email": "test@example.com"
        }
        manager = CertificateManager(config)
        assert manager.acme_staging is True
        assert manager.domain == "test.example.com"
        assert manager.acme_email == "test@example.com"
