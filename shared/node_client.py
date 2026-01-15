"""QUIC client for QuIP P2P network peer communication."""

import asyncio
import datetime
import hashlib
import ipaddress
import json
import logging
import os
import ssl
import struct
import tempfile
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Optional, Any, Tuple

from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import (
    QuicEvent,
    DatagramFrameReceived,
    StreamDataReceived,
    ConnectionTerminated,
    HandshakeCompleted,
)

from shared.block import Block, BlockHeader, MinerInfo
from shared.version import get_version
from shared.time_utils import utc_timestamp_float

if TYPE_CHECKING:
    from shared.trust_store import TrustStore


# QUIC protocol constants
QUIP_ALPN_PROTOCOL = "quip-v1"
DEFAULT_QUIC_PORT = 20049
MAX_DATAGRAM_FRAME_SIZE = 65535
# Messages larger than this use QUIC streams instead of datagrams.
# RFC 9000 Section 14 mandates 1200 bytes as the minimum supported UDP payload
# to ensure compatibility across all network paths (IPv6 minimum MTU, broken PMTUD, etc.)
MAX_DATAGRAM_MESSAGE_SIZE = 1200


class QuicMessageType(IntEnum):
    """Message types for QUIC datagram protocol."""
    # Request types (0x00-0x7F)
    JOIN_REQUEST = 0x01
    HEARTBEAT = 0x02
    PEERS_REQUEST = 0x03
    GOSSIP = 0x04
    BLOCK_SUBMIT = 0x05
    STATUS_REQUEST = 0x06
    STATS_REQUEST = 0x07
    BLOCK_REQUEST = 0x08
    BLOCK_HEADER_REQUEST = 0x09
    SOLVE_REQUEST = 0x0A

    # Response types (0x80-0xFF)
    JOIN_RESPONSE = 0x81
    HEARTBEAT_RESPONSE = 0x82
    PEERS_RESPONSE = 0x83
    GOSSIP_RESPONSE = 0x84
    BLOCK_SUBMIT_RESPONSE = 0x85
    STATUS_RESPONSE = 0x86
    STATS_RESPONSE = 0x87
    BLOCK_RESPONSE = 0x88
    BLOCK_HEADER_RESPONSE = 0x89
    SOLVE_RESPONSE = 0x8A

    ERROR_RESPONSE = 0xFF

    @classmethod
    def response_for(cls, request_type: 'QuicMessageType') -> 'QuicMessageType':
        return cls(request_type | 0x80)

    @property
    def is_request(self) -> bool:
        return self.value < 0x80


@dataclass
class QuicMessage:
    """QUIC datagram message with framing.

    Wire format: [1B msg_type][4B request_id][4B payload_len][payload...]
    """
    msg_type: QuicMessageType
    request_id: int
    payload: bytes

    HEADER_SIZE = 9

    def to_bytes(self) -> bytes:
        header = struct.pack('!BII', self.msg_type, self.request_id, len(self.payload))
        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'QuicMessage':
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Datagram too short: {len(data)}")
        msg_type_raw, request_id, payload_len = struct.unpack('!BII', data[:cls.HEADER_SIZE])
        msg_type = QuicMessageType(msg_type_raw)
        payload = data[cls.HEADER_SIZE:cls.HEADER_SIZE + payload_len]
        return cls(msg_type=msg_type, request_id=request_id, payload=payload)

    def create_response(self, payload: bytes) -> 'QuicMessage':
        return QuicMessage(
            msg_type=QuicMessageType.response_for(self.msg_type),
            request_id=self.request_id,
            payload=payload
        )

    def create_error_response(self, error_message: str) -> 'QuicMessage':
        return QuicMessage(
            msg_type=QuicMessageType.ERROR_RESPONSE,
            request_id=self.request_id,
            payload=error_message.encode('utf-8')
        )


def generate_self_signed_cert(hostname: str = "localhost", cert_dir: Optional[str] = None) -> Tuple[str, str]:
    """Generate self-signed certificate for QUIC TLS 1.3."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QuIP Network"),
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
    ])

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
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    if cert_dir is None:
        cert_dir = tempfile.gettempdir()

    cert_path = os.path.join(cert_dir, "quip_quic_cert.pem")
    key_path = os.path.join(cert_dir, "quip_quic_key.pem")

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    return cert_path, key_path


class _QuicClientProtocol(QuicConnectionProtocol):
    """QUIC connection protocol handler for client."""

    def __init__(self, quic: QuicConnection, stream_handler: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(quic, stream_handler)
        self._logger = logger or logging.getLogger(__name__)
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._request_counter = 0
        self._connected = asyncio.Event()
        self._connection_closed = False  # Renamed to avoid conflict with parent's _closed Event
        self._peer_certificate_der: Optional[bytes] = None
        # Stream buffers for reassembling stream data
        self._stream_buffers: Dict[int, bytearray] = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            # Extract peer certificate for TOFU verification
            self._extract_peer_certificate()
            self._connected.set()
        elif isinstance(event, DatagramFrameReceived):
            self._logger.debug(f"DatagramFrameReceived: {len(event.data)} bytes")
            self._handle_response(event.data)
        elif isinstance(event, StreamDataReceived):
            self._handle_stream_data(event)
        elif isinstance(event, ConnectionTerminated):
            self._logger.info(f"ConnectionTerminated: code={event.error_code}, reason={event.reason_phrase}")
            self._connection_closed = True
            self._connected.clear()
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(ConnectionError("Connection terminated"))
            self._pending_requests.clear()
            self._stream_buffers.clear()

    def _handle_stream_data(self, event: StreamDataReceived) -> None:
        """Handle data received on a QUIC stream."""
        stream_id = event.stream_id
        if stream_id not in self._stream_buffers:
            self._stream_buffers[stream_id] = bytearray()
        self._stream_buffers[stream_id].extend(event.data)

        # Check if stream is complete (end_stream flag)
        if event.end_stream:
            data = bytes(self._stream_buffers.pop(stream_id))
            self._logger.debug(f"StreamDataReceived complete: {len(data)} bytes on stream {stream_id}")
            self._handle_response(data)

    def _extract_peer_certificate(self) -> None:
        """Extract peer certificate from TLS session for TOFU verification."""
        try:
            # Access the TLS session to get peer certificate
            # aioquic stores the peer certificate chain after handshake
            tls = getattr(self._quic, '_tls', None)
            if tls is not None:
                # Try to get peer certificate from TLS context
                peer_cert = getattr(tls, 'peer_certificate', None)
                if peer_cert is not None:
                    # Certificate is already in DER format from aioquic
                    self._peer_certificate_der = peer_cert
                    return

            # Alternative: try accessing via _peer_certificate attribute
            peer_cert = getattr(self._quic, '_peer_certificate', None)
            if peer_cert is not None:
                self._peer_certificate_der = peer_cert
        except Exception as e:
            self._logger.debug(f"Could not extract peer certificate: {e}")

    def _handle_response(self, data: bytes) -> None:
        try:
            msg = QuicMessage.from_bytes(data)
            self._logger.debug(
                f"Received response: type={msg.msg_type.name}, id={msg.request_id}"
            )
            if msg.request_id in self._pending_requests:
                future = self._pending_requests.pop(msg.request_id)
                if not future.done():
                    future.set_result(msg)
            else:
                self._logger.warning(
                    f"Received response for unknown request_id={msg.request_id}"
                )
        except Exception as e:
            self._logger.warning(f"Invalid response: {e}")

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send_request(self, msg_type: QuicMessageType, payload: bytes,
                           timeout: float = 10.0) -> Optional[QuicMessage]:
        if self._connection_closed:
            self._logger.warning(f"send_request: connection closed, not sending {msg_type.name}")
            return None

        self._request_counter += 1
        request_id = self._request_counter
        future: asyncio.Future[QuicMessage] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        msg = QuicMessage(msg_type=msg_type, request_id=request_id, payload=payload)
        msg_bytes = msg.to_bytes()

        try:
            # Use streams for large messages, datagrams for small ones
            if len(msg_bytes) > MAX_DATAGRAM_MESSAGE_SIZE:
                stream_id = self._quic.get_next_available_stream_id()
                self._quic.send_stream_data(stream_id, msg_bytes, end_stream=True)
                self._logger.debug(
                    f"Sending {msg_type.name} (id={request_id}) via stream {stream_id}, size={len(msg_bytes)}"
                )
            else:
                self._quic.send_datagram_frame(msg_bytes)
                self._logger.debug(
                    f"Sending {msg_type.name} (id={request_id}) via datagram, size={len(msg_bytes)}"
                )
            self.transmit()
        except Exception as e:
            self._logger.warning(f"Failed to send {msg_type.name}: {e}")
            self._pending_requests.pop(request_id, None)
            return None

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._logger.warning(f"Timeout waiting for response to {msg_type.name} (id={request_id})")
            self._pending_requests.pop(request_id, None)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._connection_closed

    @property
    def peer_certificate_fingerprint(self) -> Optional[str]:
        """Get SHA-256 fingerprint of peer certificate (hex encoded)."""
        if self._peer_certificate_der:
            return hashlib.sha256(self._peer_certificate_der).hexdigest()
        return None


class TofuVerificationError(Exception):
    """Raised when TOFU verification fails (fingerprint mismatch)."""
    pass


class NodeClient:
    """QUIC client for QuIP P2P networking with connection pooling and TOFU verification."""

    def __init__(self, node_timeout: float = 10.0, logger: Optional[logging.Logger] = None,
                 verify_ssl: bool = False, trust_store: Optional['TrustStore'] = None):
        self.node_timeout = node_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.verify_ssl = verify_ssl
        self.trust_store = trust_store
        self._connections: Dict[str, _QuicClientProtocol] = {}
        self._connection_contexts: Dict[str, Any] = {}  # Store context managers to keep connections alive
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self.peers: Dict[str, MinerInfo] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        for host in list(self._connections.keys()):
            await self._close_connection(host)
        self._connections.clear()
        self._connection_contexts.clear()

    async def _close_connection(self, host: str) -> None:
        """Close a connection and clean up its context manager."""
        if host in self._connections:
            self.logger.debug(f"Closing QUIC connection to {host}")
            try:
                self._connections[host].close()
            except Exception:
                pass
            del self._connections[host]
        if host in self._connection_contexts:
            try:
                ctx = self._connection_contexts[host]
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
            del self._connection_contexts[host]

    def update_peers(self, peers: Dict[str, MinerInfo]) -> None:
        self.peers = peers.copy()

    def add_peer(self, host: str, info: MinerInfo) -> None:
        self.peers[host] = info

    async def remove_peer(self, host: str) -> None:
        self.peers.pop(host, None)
        await self._close_connection(host)

    async def _get_connection(self, host: str) -> Optional[_QuicClientProtocol]:
        if host not in self._connection_locks:
            self._connection_locks[host] = asyncio.Lock()

        async with self._connection_locks[host]:
            if host in self._connections:
                conn = self._connections[host]
                if conn.is_connected:
                    return conn
                else:
                    self.logger.warning(
                        f"Connection to {host} exists but is_connected=False "
                        f"(_connected={conn._connected.is_set()}, _connection_closed={conn._connection_closed})"
                    )
            # Clean up old connection if exists
            await self._close_connection(host)

            addr, port = (host.rsplit(':', 1) if ':' in host else (host, DEFAULT_QUIC_PORT))
            port = int(port) if isinstance(port, str) else port

            configuration = QuicConfiguration(
                is_client=True,
                max_datagram_frame_size=MAX_DATAGRAM_FRAME_SIZE,
                alpn_protocols=[QUIP_ALPN_PROTOCOL],
                idle_timeout=300.0,
            )
            if not self.verify_ssl:
                configuration.verify_mode = ssl.CERT_NONE

            try:
                # Create connection context manager but don't use 'async with'
                # since that would close connection when block exits
                ctx = connect(
                    host=addr, port=port, configuration=configuration,
                    create_protocol=lambda *a, **k: _QuicClientProtocol(*a, logger=self.logger, **k),
                )
                # Manually enter the context to start the connection
                protocol = await ctx.__aenter__()

                if await protocol.wait_connected(timeout=5.0):
                    # TOFU verification if trust store is configured
                    if self.trust_store:
                        tofu_ok = await self._verify_tofu(host, protocol)
                        if not tofu_ok:
                            protocol.close()
                            await ctx.__aexit__(None, None, None)
                            return None

                    # Store both connection and context manager to keep connection alive
                    self._connections[host] = protocol
                    self._connection_contexts[host] = ctx
                    self.logger.info(f"QUIC connection established to {host}")
                    return protocol
                else:
                    # Connection failed, clean up
                    await ctx.__aexit__(None, None, None)
                    return None
            except TofuVerificationError as e:
                self.logger.error(f"TOFU verification failed for {host}: {e}")
                return None
            except Exception as e:
                self.logger.warning(f"Failed to connect to {host}: {e}")
                return None

    async def _verify_tofu(self, host: str, protocol: _QuicClientProtocol) -> bool:
        """
        Verify peer certificate using TOFU model.

        Returns True if verification passes (NEW or MATCH), False on MISMATCH.
        """
        from shared.trust_store import TofuResult

        fingerprint = protocol.peer_certificate_fingerprint
        if not fingerprint:
            # No certificate available - can't verify, but allow connection
            # This might happen with certain TLS configurations
            self.logger.debug(f"No peer certificate available for {host}, skipping TOFU")
            return True

        result = await self.trust_store.verify_fingerprint(host, fingerprint)

        if result == TofuResult.NEW:
            self.logger.info(f"TOFU: First connection to {host}, fingerprint stored")
            return True
        elif result == TofuResult.MATCH:
            self.logger.debug(f"TOFU: Fingerprint verified for {host}")
            return True
        else:  # MISMATCH
            self.logger.error(
                f"TOFU MISMATCH for {host}! Certificate fingerprint does not match "
                f"previously stored value. This could indicate a man-in-the-middle attack. "
                f"Connection refused. If the peer legitimately changed their certificate, "
                f"remove the old fingerprint from the trust store."
            )
            return False

    async def send_heartbeat(self, node_host: str, public_host: str, miner_info: MinerInfo) -> bool:
        protocol = await self._get_connection(node_host)
        if not protocol:
            return False
        payload = json.dumps({
            "sender": public_host, "version": get_version(), "timestamp": utc_timestamp_float()
        }).encode('utf-8')
        response = await protocol.send_request(QuicMessageType.HEARTBEAT, payload, timeout=5.0)
        return response is not None and response.msg_type == QuicMessageType.HEARTBEAT_RESPONSE

    async def get_peer_status(self, host: str) -> Optional[dict]:
        protocol = await self._get_connection(host)
        if not protocol:
            return None
        response = await protocol.send_request(QuicMessageType.STATUS_REQUEST, b'', timeout=self.node_timeout)
        if response and response.msg_type == QuicMessageType.STATUS_RESPONSE:
            try:
                return json.loads(response.payload.decode('utf-8'))
            except Exception:
                return None
        return None

    async def get_peer_block(self, host: str, block_number: int = 0) -> Optional[Block]:
        protocol = await self._get_connection(host)
        if not protocol:
            return None
        t0 = time.perf_counter()
        payload = struct.pack('!I', block_number)
        response = await protocol.send_request(QuicMessageType.BLOCK_REQUEST, payload, timeout=self.node_timeout)
        if response and response.msg_type == QuicMessageType.BLOCK_RESPONSE:
            try:
                block = Block.from_network(response.payload)
                self.logger.debug(f"Downloaded block {block.header.index} from {host} in {(time.perf_counter()-t0)*1000:.1f}ms")
                return block
            except Exception:
                return None
        return None

    async def get_peer_block_header(self, host: str, block_number: int = 0) -> Optional[BlockHeader]:
        protocol = await self._get_connection(host)
        if not protocol:
            return None
        payload = struct.pack('!I', block_number)
        response = await protocol.send_request(QuicMessageType.BLOCK_HEADER_REQUEST, payload, timeout=self.node_timeout)
        if response and response.msg_type == QuicMessageType.BLOCK_HEADER_RESPONSE:
            try:
                return BlockHeader.from_network(response.payload)
            except Exception:
                return None
        return None

    async def gossip_to(self, host: str, message: 'Message') -> bool:
        from shared.network_node import Message
        protocol = await self._get_connection(host)
        if not protocol:
            return False
        payload = message.to_network()
        response = await protocol.send_request(QuicMessageType.GOSSIP, payload, timeout=5.0)
        return response is not None and response.msg_type == QuicMessageType.GOSSIP_RESPONSE

    async def join_network_via_peer(self, peer_address: str, join_data: dict) -> Optional[dict]:
        protocol = await self._get_connection(peer_address)
        if not protocol:
            return None
        payload = json.dumps(join_data).encode('utf-8')
        response = await protocol.send_request(QuicMessageType.JOIN_REQUEST, payload, timeout=self.node_timeout)
        if response and response.msg_type == QuicMessageType.JOIN_RESPONSE:
            try:
                return json.loads(response.payload.decode('utf-8'))
            except Exception:
                return None
        return None

    async def connect_to_peer(self, peer_address: str) -> bool:
        protocol = await self._get_connection(peer_address)
        return protocol is not None and protocol.is_connected
