"""
Certificate manager for QuIP REST API.

Handles TLS certificates:
1. Use configured certificates if valid
2. Generate self-signed certificate as fallback
"""

import datetime
import ipaddress
import logging
import os
import socket
import tempfile
from typing import Optional, Tuple

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


class CertificateManager:
    """
    Manages TLS certificates for the REST API.

    Priority:
    1. Use configured cert/key files if they exist and are valid
    2. Generate self-signed certificate with warning
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Support fallback: rest_tls_cert_file -> tls_cert_file (shared with QUIC)
        self.cert_file = config.get("rest_tls_cert_file") or config.get("tls_cert_file")
        self.key_file = config.get("rest_tls_key_file") or config.get("tls_key_file")

    async def get_certificate(self) -> Tuple[str, str]:
        """Get certificate and key paths."""
        existing = await self._check_existing_cert()
        if existing:
            self.logger.info(f"Using configured certificate: {existing[0]}")
            return existing

        self.logger.warning(
            "No TLS certificate configured — generating self-signed certificate. "
            "Browser users will see certificate warnings."
        )
        return await self._generate_self_signed()

    async def _check_existing_cert(self) -> Optional[Tuple[str, str]]:
        """Check if configured certificate exists and is valid."""
        if not self.cert_file or not self.key_file:
            return None

        cert_path = os.path.expanduser(self.cert_file)
        key_path = os.path.expanduser(self.key_file)

        if not os.path.exists(cert_path):
            self.logger.debug(f"Certificate file not found: {cert_path}")
            return None

        if not os.path.exists(key_path):
            self.logger.debug(f"Key file not found: {key_path}")
            return None

        # Check certificate validity
        if not self._check_certificate_validity(cert_path):
            self.logger.warning(f"Certificate is expired or invalid: {cert_path}")
            return None

        return (cert_path, key_path)

    def _check_certificate_validity(self, cert_path: str, days_warning: int = 30) -> bool:
        """
        Check if certificate is valid and not expiring soon.

        Args:
            cert_path: Path to certificate file
            days_warning: Warn if certificate expires within this many days

        Returns:
            True if certificate is valid, False otherwise
        """
        try:
            with open(cert_path, "rb") as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data)
            now = datetime.datetime.now(datetime.UTC)

            if cert.not_valid_after_utc < now:
                self.logger.error(f"Certificate has expired: {cert_path}")
                return False

            if cert.not_valid_before_utc > now:
                self.logger.error(f"Certificate is not yet valid: {cert_path}")
                return False

            days_until_expiry = (cert.not_valid_after_utc - now).days
            if days_until_expiry < days_warning:
                self.logger.warning(
                    f"Certificate expires in {days_until_expiry} days: {cert_path}"
                )

            return True
        except Exception as e:
            self.logger.error(f"Error checking certificate validity: {e}")
            return False

    async def _generate_self_signed(self) -> Tuple[str, str]:
        """Generate a self-signed certificate."""
        cert_dir = tempfile.gettempdir()

        hostname = socket.getfqdn() or "localhost"

        key = ec.generate_private_key(ec.SECP256R1())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QuIP Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])

        san_list = [
            x509.DNSName(hostname),
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv6Address("::1")),
        ]

        now = datetime.datetime.now(datetime.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        cert_path = os.path.join(cert_dir, "rest_api_cert.pem")
        key_path = os.path.join(cert_dir, "rest_api_key.pem")

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        os.chmod(key_path, 0o600)
        self.logger.info(f"Generated self-signed certificate: {cert_path}")
        return (cert_path, key_path)
