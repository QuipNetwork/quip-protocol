"""
Certificate manager for QuIP REST API.

Handles certificate acquisition with priority-based fallback:
1. Use configured certificates if valid
2. Attempt ACME/Let's Encrypt if port 80 available
3. Generate self-signed certificate as fallback
"""

import asyncio
import datetime
import ipaddress
import logging
import os
import socket
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


class CertificateManager:
    """
    Manages TLS certificates for the REST API with priority-based acquisition.

    Priority:
    1. Use configured cert/key files if they exist and are valid
    2. Attempt ACME/Let's Encrypt if port 80 is available and domain is configured
    3. Generate self-signed certificate with warning
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the certificate manager.

        Config keys:
            rest_tls_cert_file: Path to existing certificate file
            rest_tls_key_file: Path to existing private key file
            rest_domain: Domain name for ACME certificate
            acme_email: Email for ACME account
            acme_staging: Use ACME staging server (for testing)
            cert_dir: Directory to store generated certificates
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.cert_file = config.get("rest_tls_cert_file")
        self.key_file = config.get("rest_tls_key_file")
        self.domain = config.get("rest_domain")
        self.acme_email = config.get("acme_email")
        self.acme_staging = config.get("acme_staging", False)
        self.cert_dir = os.path.expanduser(config.get("cert_dir", "~/.quip/certs"))

    async def get_certificate(self) -> Tuple[str, str]:
        """
        Get certificate and key paths using priority-based acquisition.

        Returns:
            Tuple of (cert_path, key_path)
        """
        # Priority 1: Use configured certificates if valid
        existing = await self._check_existing_cert()
        if existing:
            self.logger.info(f"Using configured certificate: {existing[0]}")
            return existing

        # Priority 2: Attempt ACME if configured
        if self.domain and self.acme_email:
            acme_cert = await self._attempt_acme()
            if acme_cert:
                self.logger.info(f"Obtained Let's Encrypt certificate for {self.domain}")
                return acme_cert

        # Priority 3: Generate self-signed certificate
        self.logger.warning(
            "\n" + "=" * 70 + "\n"
            "WARNING: Could not acquire publicly-trusted certificate.\n"
            "- Port 80 not available for ACME challenge, or\n"
            "- No domain configured for certificate issuance\n"
            "\n"
            "Browser users will see certificate warnings when connecting to this node's REST API.\n"
            "Node-to-node communication via QUIC is unaffected (uses TOFU model).\n"
            "Generated self-signed certificate for network participation.\n"
            "\n"
            "To enable browser-trusted certificates:\n"
            "1. Ensure port 80 is open and accessible from the internet\n"
            "2. Configure rest_domain and acme_email in your config file\n"
            + "=" * 70
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

    async def _attempt_acme(self) -> Optional[Tuple[str, str]]:
        """
        Attempt to obtain certificate via ACME/Let's Encrypt.

        Returns:
            Tuple of (cert_path, key_path) if successful, None otherwise
        """
        # Check if port 80 is available for HTTP-01 challenge
        if not self._is_port_available(80):
            self.logger.info("Port 80 not available, skipping ACME certificate acquisition")
            return None

        try:
            acme_client = AcmeClient(
                staging=self.acme_staging,
                logger=self.logger
            )
            return await acme_client.obtain_certificate(
                domain=self.domain,
                email=self.acme_email,
                cert_dir=self.cert_dir
            )
        except Exception as e:
            self.logger.warning(f"ACME certificate acquisition failed: {e}")
            return None

    async def _generate_self_signed(self) -> Tuple[str, str]:
        """
        Generate a self-signed certificate.

        Returns:
            Tuple of (cert_path, key_path)
        """
        os.makedirs(self.cert_dir, exist_ok=True)

        hostname = self.domain or socket.getfqdn() or "localhost"

        # Generate EC private key
        key = ec.generate_private_key(ec.SECP256R1())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QuIP Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])

        # Build Subject Alternative Names
        san_list = [
            x509.DNSName(hostname),
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]

        # Add domain if different from hostname
        if self.domain and self.domain != hostname:
            san_list.insert(0, x509.DNSName(self.domain))

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

        cert_path = os.path.join(self.cert_dir, "rest_api_cert.pem")
        key_path = os.path.join(self.cert_dir, "rest_api_key.pem")

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        # Set restrictive permissions on key file
        os.chmod(key_path, 0o600)

        self.logger.info(f"Generated self-signed certificate: {cert_path}")
        return (cert_path, key_path)

    def _is_port_available(self, port: int) -> bool:
        """
        Check if a port is available for binding.

        Args:
            port: Port number to check

        Returns:
            True if port can be bound, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False


class AcmeClient:
    """
    ACME client for Let's Encrypt certificate acquisition.

    Uses HTTP-01 challenge which requires port 80 to be available.
    """

    # Let's Encrypt directory URLs
    DIRECTORY_URL = "https://acme-v02.api.letsencrypt.org/directory"
    STAGING_DIRECTORY_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"

    def __init__(
        self,
        staging: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        self.staging = staging
        self.logger = logger or logging.getLogger(__name__)
        self.directory_url = self.STAGING_DIRECTORY_URL if staging else self.DIRECTORY_URL

    async def obtain_certificate(
        self,
        domain: str,
        email: str,
        cert_dir: str
    ) -> Optional[Tuple[str, str]]:
        """
        Obtain a certificate from Let's Encrypt using HTTP-01 challenge.

        Args:
            domain: Domain name for the certificate
            email: Email for ACME account registration
            cert_dir: Directory to store certificate and key

        Returns:
            Tuple of (cert_path, key_path) if successful, None otherwise
        """
        try:
            from acme import client, messages, challenges
            from acme.client import ClientNetwork, ClientV2
            import josepy as jose
        except ImportError:
            self.logger.warning(
                "ACME library not installed. Install with: pip install acme josepy"
            )
            return None

        os.makedirs(cert_dir, exist_ok=True)

        # Check for existing valid certificate
        cert_path = os.path.join(cert_dir, f"{domain}_cert.pem")
        key_path = os.path.join(cert_dir, f"{domain}_key.pem")
        account_key_path = os.path.join(cert_dir, "account_key.pem")

        if os.path.exists(cert_path) and os.path.exists(key_path):
            # Check if existing cert is still valid
            try:
                with open(cert_path, "rb") as f:
                    cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)
                now = datetime.datetime.now(datetime.UTC)
                days_until_expiry = (cert.not_valid_after_utc - now).days

                if days_until_expiry > 30:
                    self.logger.info(
                        f"Using existing Let's Encrypt certificate "
                        f"(expires in {days_until_expiry} days)"
                    )
                    return (cert_path, key_path)
                else:
                    self.logger.info(
                        f"Certificate expires in {days_until_expiry} days, renewing..."
                    )
            except Exception:
                pass

        self.logger.info(f"Requesting Let's Encrypt certificate for {domain}")

        try:
            # Generate or load account key
            if os.path.exists(account_key_path):
                with open(account_key_path, "rb") as f:
                    account_key = jose.JWKEC.load(f.read())
            else:
                account_key = jose.JWKEC(
                    key=ec.generate_private_key(ec.SECP256R1())
                )
                with open(account_key_path, "wb") as f:
                    f.write(account_key.key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    ))
                os.chmod(account_key_path, 0o600)

            # Create ACME client
            net = ClientNetwork(account_key, user_agent="QuIP-Network/1.0")
            directory = messages.Directory.from_json(
                net.get(self.directory_url).json()
            )
            acme_client = ClientV2(directory, net)

            # Register account
            registration = acme_client.new_account(
                messages.NewRegistration.from_data(
                    email=email,
                    terms_of_service_agreed=True
                )
            )

            # Generate certificate key
            cert_key = ec.generate_private_key(ec.SECP256R1())

            # Create CSR
            csr = (
                x509.CertificateSigningRequestBuilder()
                .subject_name(x509.Name([
                    x509.NameAttribute(NameOID.COMMON_NAME, domain),
                ]))
                .add_extension(
                    x509.SubjectAlternativeName([x509.DNSName(domain)]),
                    critical=False,
                )
                .sign(cert_key, hashes.SHA256())
            )

            # Request certificate
            order = acme_client.new_order(csr.public_bytes(serialization.Encoding.DER))

            # Complete HTTP-01 challenges
            for authz in order.authorizations:
                for challenge in authz.body.challenges:
                    if isinstance(challenge.chall, challenges.HTTP01):
                        response, validation = challenge.response_and_validation(account_key)

                        # Start simple HTTP server for challenge
                        challenge_server = await self._start_challenge_server(
                            challenge.chall.path,
                            validation
                        )

                        try:
                            # Notify ACME server we're ready
                            acme_client.answer_challenge(challenge, response)

                            # Wait for authorization
                            authz_resource = acme_client.poll_and_finalize(order)
                        finally:
                            # Stop challenge server
                            await self._stop_challenge_server(challenge_server)

            # Finalize order and get certificate
            order = acme_client.poll_and_finalize(order)

            # Save certificate
            with open(cert_path, "wb") as f:
                f.write(order.fullchain_pem.encode())

            with open(key_path, "wb") as f:
                f.write(cert_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ))
            os.chmod(key_path, 0o600)

            self.logger.info(f"Successfully obtained Let's Encrypt certificate for {domain}")
            return (cert_path, key_path)

        except Exception as e:
            self.logger.error(f"Failed to obtain ACME certificate: {e}")
            return None

    async def _start_challenge_server(
        self,
        challenge_path: str,
        validation: str
    ) -> asyncio.Server:
        """Start a simple HTTP server to respond to ACME challenge."""
        from aiohttp import web

        async def handle_challenge(request):
            if request.path == challenge_path:
                return web.Response(text=validation)
            return web.Response(status=404)

        app = web.Application()
        app.router.add_get('/.well-known/acme-challenge/{token}', handle_challenge)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 80)
        await site.start()

        return runner

    async def _stop_challenge_server(self, runner) -> None:
        """Stop the ACME challenge server."""
        await runner.cleanup()
