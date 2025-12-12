"""QUIC client for QuIP P2P network peer communication."""

import asyncio
import datetime
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
from typing import Dict, Optional, Any, Tuple

from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import (
    QuicEvent,
    DatagramFrameReceived,
    ConnectionTerminated,
    HandshakeCompleted,
)

from shared.block import Block, BlockHeader, MinerInfo
from shared.version import get_version
from shared.time_utils import utc_timestamp_float


# QUIC protocol constants
QUIP_ALPN_PROTOCOL = "quip-v1"
DEFAULT_QUIC_PORT = 20049
MAX_DATAGRAM_FRAME_SIZE = 65535


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

    now = datetime.datetime.utcnow()
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
        self._closed = False

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            self._connected.set()
        elif isinstance(event, DatagramFrameReceived):
            self._handle_response(event.data)
        elif isinstance(event, ConnectionTerminated):
            self._closed = True
            self._connected.clear()
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(ConnectionError("Connection terminated"))
            self._pending_requests.clear()

    def _handle_response(self, data: bytes) -> None:
        try:
            msg = QuicMessage.from_bytes(data)
            if msg.request_id in self._pending_requests:
                future = self._pending_requests.pop(msg.request_id)
                if not future.done():
                    future.set_result(msg)
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
        if self._closed:
            return None

        self._request_counter += 1
        request_id = self._request_counter
        future: asyncio.Future[QuicMessage] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        msg = QuicMessage(msg_type=msg_type, request_id=request_id, payload=payload)
        try:
            self._quic.send_datagram_frame(msg.to_bytes())
            self.transmit()
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            return None

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._closed


class NodeClient:
    """QUIC client for QuIP P2P networking with connection pooling."""

    def __init__(self, node_timeout: float = 10.0, logger: Optional[logging.Logger] = None,
                 verify_ssl: bool = False):
        self.node_timeout = node_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.verify_ssl = verify_ssl
        self._connections: Dict[str, _QuicClientProtocol] = {}
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self.peers: Dict[str, MinerInfo] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        for protocol in self._connections.values():
            try:
                protocol.close()
            except Exception:
                pass
        self._connections.clear()

    def update_peers(self, peers: Dict[str, MinerInfo]) -> None:
        self.peers = peers.copy()

    def add_peer(self, host: str, info: MinerInfo) -> None:
        self.peers[host] = info

    def remove_peer(self, host: str) -> None:
        self.peers.pop(host, None)
        if host in self._connections:
            try:
                self._connections[host].close()
            except Exception:
                pass
            del self._connections[host]

    async def _get_connection(self, host: str) -> Optional[_QuicClientProtocol]:
        if host not in self._connection_locks:
            self._connection_locks[host] = asyncio.Lock()

        async with self._connection_locks[host]:
            if host in self._connections and self._connections[host].is_connected:
                return self._connections[host]
            self._connections.pop(host, None)

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
                async with connect(
                    host=addr, port=port, configuration=configuration,
                    create_protocol=lambda *a, **k: _QuicClientProtocol(*a, logger=self.logger, **k),
                ) as protocol:
                    if await protocol.wait_connected(timeout=5.0):
                        self._connections[host] = protocol
                        self.logger.info(f"QUIC connection established to {host}")
                        return protocol
                    return None
            except Exception as e:
                self.logger.warning(f"Failed to connect to {host}: {e}")
                return None

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
