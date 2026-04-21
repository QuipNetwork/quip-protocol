import asyncio
import copy
import ipaddress
import json
import math
import random
import socket
import struct
import sys
import threading
import time

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any

from blake3 import blake3
import logging
from packaging import version

from shared.base_miner import MiningResult
from shared.block import Block, BlockHeader, MinerInfo
from shared.node import Node
from shared.logging_config import init_component_logger
from shared.system_info import override_public_address
from shared.version import (
    get_version, PROTOCOL_VERSION,
    is_version_compatible, MIN_COMPATIBLE_VERSION,
    select_compatible_peers,
)
from shared.node_client import (
    NodeClient, QuicMessage, QuicMessageType,
    generate_self_signed_cert, QUIP_ALPN_PROTOCOL, MAX_DATAGRAM_FRAME_SIZE,
    MAX_DATAGRAM_MESSAGE_SIZE,
)
from shared.sync_messages import (
    MAX_MANIFEST_ENTRIES,
    decode_block_by_hash_request,
    decode_manifest_request,
    encode_block_by_hash_response,
    encode_manifest_response,
)
from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import QuicEvent, DatagramFrameReceived, StreamDataReceived, ConnectionTerminated, HandshakeCompleted
from shared.block_inventory import BlockInventory
from shared.block_synchronizer import BlockSynchronizer
from shared.block_store import BlockStore
from shared.load_monitor import LoadMonitor, NodeLoad
from shared.peer_scorer import PeerScorer
from shared.process_pool import ProcessPool, ProcessPoolConfig
from shared.rate_limiter import PeerRateLimiter
from shared.swim_detector import SwimDetector, PeerState
from shared.telemetry import TelemetryManager
from shared.telemetry_cache import TelemetryCache
from shared.time_utils import (
    utc_timestamp_float, is_clock_synchronized, NETWORK_TIME_SYNC_INTERVAL,
    get_network_clock, network_timestamp
)


@dataclass(frozen=True)
class EpochInfo:
    """Information about a previous chain epoch to prevent block acceptance from old epochs."""
    first_hash: bytes      # Hash of block 1 from this epoch
    last_timestamp: int    # Timestamp of the last block before reset
    last_index: int        # Index of the last block before reset
    last_hash: bytes       # Hash of the last block before reset


@dataclass
class CandidatePeer:
    """A peer known by address but not yet in the active heartbeat set.

    Candidates are discovered via gossip or JOIN responses when the
    active peer set is full.  They are periodically probed and promoted
    to the active set when a slot opens up.
    """
    info: MinerInfo
    discovered_at: float       # time.monotonic()
    source: str                # "gossip", "join_response"
    probe_attempts: int = 0
    last_probe_at: float = 0.0
    descriptor: Optional[Dict[str, Any]] = None


# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

# Global logger for this module (set during NetworkNode initialization)
log = None


async def get_public_ip() -> Optional[str]:
    """
    Get the public IP address by querying external services.

    Returns:
        Public IP address as string, or None if unable to determine
    """
    import urllib.request
    import ssl

    # Use module-level logger
    logger = logging.getLogger(__name__)

    # List of reliable IP detection services (shuffled to spread load)
    services = [
        "https://check.quip.network",
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ipecho.net/plain",
        "https://checkip.amazonaws.com",
        "https://ident.me",
    ]
    random.shuffle(services)

    # Create SSL context that doesn't verify (for simplicity)
    ssl_context = ssl.create_default_context()

    for service in services:
        try:
            # Run blocking request in executor
            loop = asyncio.get_event_loop()

            def fetch_ip():
                req = urllib.request.Request(service, headers={'User-Agent': 'QuIP-Node/1.0'})
                with urllib.request.urlopen(req, timeout=5, context=ssl_context) as response:
                    return response.read().decode('utf-8').strip()

            ip = await loop.run_in_executor(None, fetch_ip)
            # Validate using ipaddress module (supports both IPv4 and IPv6)
            if ip:
                try:
                    addr = ipaddress.ip_address(ip)
                    # Normalize IPv6-mapped IPv4 (::ffff:1.2.3.4) to plain IPv4
                    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                        addr = addr.ipv4_mapped
                    ip = str(addr)
                    logger.info(f"Detected public IP: {ip}")
                    return ip
                except ValueError:
                    logger.debug(f"Invalid IP format from {service}: {ip}")
                    continue
        except Exception as e:
            logger.debug(f"Failed to get IP from {service}: {e}")
            continue

    logger.warning("Unable to determine public IP address")
    return None


def get_local_ip(prefer_ipv6: bool = False) -> str:
    """
    Get the local IP address (best guess).

    Args:
        prefer_ipv6: If True, try IPv6 first, then fall back to IPv4

    Returns:
        Local IP address as string
    """
    if prefer_ipv6:
        # Try IPv6 first
        ipv6 = get_local_ipv6()
        if ipv6:
            return ipv6

    try:
        # Connect to a remote address to determine which local interface would be used
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Use Google's DNS server - we don't actually send data
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"


def get_local_ipv6() -> Optional[str]:
    """
    Get the local IPv6 address (best guess).

    Returns:
        Local IPv6 address as string, or None if not available
    """
    try:
        # Connect to Google's IPv6 DNS server to determine which local interface would be used
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            # Use Google's IPv6 DNS server - we don't actually send data
            s.connect(("2001:4860:4860::8888", 80))
            local_ip = s.getsockname()[0]
            # Filter out link-local addresses (fe80::)
            if local_ip and not local_ip.startswith('fe80'):
                return local_ip
    except Exception:
        pass
    return None



@dataclass
class Message:
    """Base message structure for gossip communication."""
    type: str
    sender: str
    timestamp: float
    data: bytes
    id: Optional[str] = None

    def to_network(self) -> bytes:
        """Serialize message to binary: [u16 type][u16 sender][f64 ts][u16 id][u32 data_len][data]."""
        _st = struct
        def _u16(n: int) -> bytes: return _st.pack('!H', max(0, min(int(n), 0xFFFF)))
        def _u32(n: int) -> bytes: return _st.pack('!I', max(0, min(int(n), 0xFFFFFFFF)))
        def _f64(x: float) -> bytes: return _st.pack('!d', float(x))
        def _str(s: Optional[str]) -> bytes:
            if not s: return _u16(0)
            b = s.encode('utf-8'); return _u16(len(b)) + b
        payload = self.data or b''
        out = b''
        out += _str(self.type)
        out += _str(self.sender)
        out += _f64(self.timestamp)
        out += _str(self.id or '')
        out += _u32(len(payload)) + payload
        return out

    @classmethod
    def from_network(cls, data: bytes) -> 'Message':
        """Deserialize message from binary: [u16 type][u16 sender][f64 ts][u16 id][u32 data_len][data]."""
        _st = struct
        def _r_u16(buf: bytes, o: int): return _st.unpack('!H', buf[o:o+2])[0], o+2
        def _r_u32(buf: bytes, o: int): return _st.unpack('!I', buf[o:o+4])[0], o+4
        def _r_f64(buf: bytes, o: int): return _st.unpack('!d', buf[o:o+8])[0], o+8
        def _r_str(buf: bytes, o: int):
            ln, o = _r_u16(buf, o)
            if ln == 0: return '', o
            s = buf[o:o+ln].decode('utf-8'); return s, o+ln
        o = 0
        typ, o = _r_str(data, o)
        sender, o = _r_str(data, o)
        ts, o = _r_f64(data, o)
        mid, o = _r_str(data, o)
        dlen, o = _r_u32(data, o)
        payload = data[o:o+dlen] if dlen > 0 else b''
        return cls(type=typ or 'unknown', sender=sender or '', timestamp=ts, data=payload, id=(mid or None))


class NetworkNode(Node):
    """Peer-to-peer node for quantum blockchain network."""

    def __init__(self, config: dict, genesis_block: Block):
        self.config = config
        self.bind_address = config.get("listen", "127.0.0.1")
        self.port = config.get("port", 20049)

        self.node_name = config.get("node_name", socket.getfqdn())
        raw_public_host = config.get("public_host")
        self._public_host_explicit = raw_public_host is not None
        if raw_public_host is None:
            raw_public_host = get_local_ip()
        if ":" in str(raw_public_host):
            raise ValueError(
                f"public_host must be a hostname or IP without a port, got: '{raw_public_host}'. "
                "Use separate public_host and public_port settings."
            )
        self._public_port = config.get("public_port", self.port)
        self.public_host = f"{raw_public_host}:{self._public_port}"

        self.secret = config.get("secret", f"quip network node secret {random.randint(0, 1000000)}")
        self.auto_mine = config.get("auto_mine", False)
        
        # Chain storage configuration
        self.enable_epoch_storage = config.get("enable_epoch_storage", False)
        self.epoch_storage_dir = config.get("epoch_storage_dir", "epoch_storage")
        self.epoch_storage_format = config.get("epoch_storage_format", "pickle")  # 'json' or 'pickle'
        self.epoch_storage_compress = config.get("epoch_storage_compress", True)

        # TLS configuration for QUIC
        self.tls_cert_file = config.get("tls_cert_file")
        self.tls_key_file = config.get("tls_key_file")
        self.tls_enabled = bool(self.tls_cert_file and self.tls_key_file)
        self.verify_tls = config.get("verify_tls", False)

        # TOFU (Trust On First Use) configuration
        self.tofu_enabled = config.get("tofu", True)
        self.trust_db = config.get("trust_db", "~/.quip/trust.db")
        self.trust_store = None  # Initialized in start()

        # REST API configuration (enabled when rest_port > 0 or rest_insecure_port > 0)
        self.rest_host = config.get("rest_host", "127.0.0.1")
        self.rest_port = int(config.get("rest_port", -1))
        self.rest_insecure_port = int(config.get("rest_insecure_port", 20050))
        self.webroot = config.get("webroot")
        self.rest_api_enabled = self.rest_port > 0 or self.rest_insecure_port > 0
        self.rest_api_server = None  # Initialized in start()

        # Initialize logger with helper function
        self.logger = init_component_logger('network_node', self.node_name)

        # Durations as float seconds
        self.heartbeat_interval = float(config.get("heartbeat_interval", 15))
        self.heartbeat_timeout = float(config.get("heartbeat_timeout", 300))
        self.node_timeout = float(config.get("node_timeout", 60))

        self.initial_peers = config.get("peer", [
            "qpu-1.nodes.quip.network:20049",
            "cpu-1.quip.carback.us:20049",
            "gpu-1.quip.carback.us:20049",
            "gpu-2.quip.carback.us:20050",
            "nodes.quip.network:20049",
        ])
        # A node listing its own public address as a peer makes it
        # try to JOIN itself, fail validation, and backlist its own
        # loopback address. Strip at config load — and again after
        # public IP auto-detection below — so the ban list stays clean.
        self.initial_peers = self._filter_self_from_peers(self.initial_peers)
        self.fanout = int(config.get("fanout", 3))
        self.max_connections = int(config.get("max_connections", 50))

        # Two-tier peer management: active peers (heartbeated) vs
        # candidates (known but not yet promoted).
        self._max_active_peers = int(config.get("max_active_peers", 20))
        self._max_candidate_peers = int(config.get("max_candidate_peers", 100))
        self._candidate_peers: Dict[str, CandidatePeer] = {}

        self.net_lock = asyncio.Lock()
        self.running = False
        self.heartbeats = {}
        self.peer_versions: dict[str, str] = {}  # peer_host -> version string
        # Peer ban list is created here for pre-start checks (e.g. connect_to_peer);
        # once NodeClient is initialised in start(), we share its ban_list instead.
        from shared.peer_ban_list import PeerBanList
        self._own_ban_list = PeerBanList(logger=self.logger)
        self._ban_list: PeerBanList = self._own_ban_list

        # Connection worker (initialized in start())
        self._connection_worker: Optional['ConnectionWorkerHandle'] = None
        self._connection_request_pending = False

        # Process pool for per-connection peer isolation (initialized in start())
        self._process_pool: Optional[ProcessPool] = None

        # Per-peer rate limiter for incoming QUIC messages
        self._rate_limiter = PeerRateLimiter(
            tokens_per_second=float(config.get("rate_limit_tps", 10.0)),
            max_burst=int(config.get("rate_limit_burst", 20)),
        )

        # Load monitoring and SWIM failure detection
        self._load_monitor = LoadMonitor(
            max_connections=self.max_connections,
            high_watermark=float(config.get("load_high_watermark", 0.8)),
            low_watermark=float(config.get("load_low_watermark", 0.5)),
        )
        self._swim_detector = SwimDetector(
            k_probes=int(config.get("swim_k_probes", 3)),
            suspect_rounds=int(config.get("swim_suspect_rounds", 2)),
            probe_timeout=float(config.get("swim_probe_timeout", 10.0)),
        )
        # Load info received from peers: {peer_address: NodeLoad}
        self._peer_loads: Dict[str, NodeLoad] = {}

        # Block inventory for IHAVE/IWANT protocol
        self._block_inventory = BlockInventory()
        # Peer scoring for gossip target selection
        self._peer_scorer = PeerScorer(
            disconnect_threshold=float(
                config.get("peer_disconnect_threshold", -100.0)
            ),
        )

        self.gossip_lock = asyncio.Lock()
        # Bounded dedup: maps message_id -> timestamp, max 10k entries
        self.recent_messages: dict[str, float] = {}
        self._recent_messages_max = 10_000
        self._recent_messages_ttl = 300.0  # 5 minutes

        # Per-host announcement dedup: maps peer host -> last-seen announce
        # timestamp. Populated both when we originate a new_node gossip and
        # when we observe one from another sender (including JOIN-response
        # peer maps), so the gossip chain does not re-originate about peers
        # the network already knows.
        self._announced_nodes: dict[str, float] = {}
        self._announced_nodes_ttl = 300.0  # 5 minutes, matches recent_messages

        # Time synchronization tracking
        self.peer_timestamps = []  # Recent timestamps from peers
        self.last_time_sync_check = 0.0
        self.time_sync_warnings = 0

        # Callbacks
        self.on_new_node: Optional[Callable] = None
        self.on_node_lost: Optional[Callable] = None
        self.on_block_received: Optional[Callable] = None

        # QUIC server (replaces aiohttp web server)
        self.quic_server: Optional[QuicServer] = None

        # Background tasks
        self.heartbeat_task = None
        self.cleanup_task = None
        self.block_processor_task = None
        self.running = False

        # Background processing queues
        self.block_processing_queue = asyncio.Queue(maxsize=1000)
        self.gossip_processing_queue = asyncio.Queue(maxsize=1000)

        # Transaction queue for solve requests
        self.pending_transactions = []
        self.transactions_lock = asyncio.Lock()

        # Node client for HTTP communication
        self.node_client = None

        # Reset mechanism state variables
        self.reset_timer_task = None
        self.reset_scheduled = False  
        self.reset_start_time = None
        self.genesis_block = genesis_block
        
        # Epoch tracking to prevent accepting blocks from previous chain epochs
        # Changed to store only the most recent previous epoch instead of all epochs
        self.previous_epoch: Optional[EpochInfo] = None
        
        # Initialize block store for epoch storage if enabled
        if self.enable_epoch_storage:
            try:
                self.epoch_block_store = BlockStore(
                    storage_dir=self.epoch_storage_dir,
                    format_type=self.epoch_storage_format,
                    compress=self.epoch_storage_compress
                )
                self.logger.info(f"Epoch storage enabled: {self.epoch_storage_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize epoch storage: {e}")
                self.enable_epoch_storage = False
                self.epoch_block_store = None
        else:
            self.epoch_block_store = None
        
        # Telemetry
        self.telemetry = TelemetryManager(
            telemetry_dir=config.get("telemetry_dir", "telemetry"),
            enabled=config.get("telemetry_enabled", True),
            logger=self.logger,
        )
        self.telemetry.record_initial_peers(self.initial_peers)

        # Telemetry cache (read-only, for REST + QUIC telemetry endpoints)
        telem_api = config.get("telemetry_api", {})
        telem_api_enabled = telem_api.get(
            "enabled", config.get("telemetry_enabled", True),
        )
        if telem_api_enabled:
            self.telemetry_cache: Optional[TelemetryCache] = TelemetryCache(
                telemetry_dir=config.get("telemetry_dir", "telemetry"),
                refresh_interval=telem_api.get("cache_refresh_interval", 5.0),
                logger=self.logger,
            )
        else:
            self.telemetry_cache = None
        self._telemetry_api_config = telem_api

        # Maximum block index to synchronize with (prevents syncing with peers too far ahead)
        self.max_sync_block_index = 1024

        self.miners_config = config
        super().__init__(self.node_name, self.miners_config, genesis_block, secret=self.secret,
                         on_block_mined=self._on_block_received,
                         on_mining_started=self._network_on_mining_started,
                         on_mining_stopped=self._network_on_mining_stopped)

        # Stats caching infrastructure
        self._stats_cache = None
        self._stats_cache_lock = asyncio.Lock()

        self.logger.info(f"Network node {self.node_name} initialized with config {json.dumps(config)}")

    def _create_server_protocol(self, quic: QuicConnection, **kwargs) -> QuicConnectionProtocol:
        """Create a QUIC protocol handler for incoming connections."""
        # Note: aioquic passes stream_handler as kwarg, we accept but ignore it
        node = self

        class _ServerProtocol(QuicConnectionProtocol):
            def __init__(self, quic_conn: QuicConnection, stream_handler=None):
                super().__init__(quic_conn, stream_handler)
                self._peer_address: Optional[str] = None
                self._stream_buffers: Dict[int, bytearray] = {}

            def quic_event_received(self, event: QuicEvent) -> None:
                if isinstance(event, HandshakeCompleted):
                    peername = self._quic._network_paths[0].addr if self._quic._network_paths else None
                    if peername:
                        self._peer_address = f"{peername[0]}:{peername[1]}"
                    node.logger.debug(f"QUIC handshake completed with {self._peer_address}")
                elif isinstance(event, DatagramFrameReceived):
                    asyncio.create_task(self._handle_message(event.data))
                elif isinstance(event, StreamDataReceived):
                    self._handle_stream_data(event)
                elif isinstance(event, ConnectionTerminated):
                    node.logger.debug(
                        f"Connection terminated with {self._peer_address}: "
                        f"error_code={event.error_code}, reason={event.reason_phrase}"
                    )
                    self._stream_buffers.clear()

            def _handle_stream_data(self, event: StreamDataReceived) -> None:
                """Handle data received on a QUIC stream."""
                stream_id = event.stream_id
                if stream_id not in self._stream_buffers:
                    self._stream_buffers[stream_id] = bytearray()
                self._stream_buffers[stream_id].extend(event.data)

                # Check if stream is complete
                if event.end_stream:
                    data = bytes(self._stream_buffers.pop(stream_id))
                    node.logger.debug(f"Stream {stream_id} complete: {len(data)} bytes from {self._peer_address}")
                    asyncio.create_task(self._handle_message(data, response_stream_id=stream_id))

            async def _handle_message(self, data: bytes, response_stream_id: Optional[int] = None) -> None:
                """Handle incoming message (from datagram or stream)."""
                try:
                    msg = QuicMessage.from_bytes(data)
                    node.logger.debug(
                        f"Received {msg.msg_type.name} (id={msg.request_id}) "
                        f"from {self._peer_address}: {len(msg.payload)} bytes"
                    )
                    response = await node._handle_quic_message(msg, self)
                    if response is not None:
                        response_bytes = response.to_bytes()
                        try:
                            # Use streams for large responses, datagrams for small ones
                            if len(response_bytes) > MAX_DATAGRAM_MESSAGE_SIZE:
                                stream_id = self._quic.get_next_available_stream_id()
                                self._quic.send_stream_data(stream_id, response_bytes, end_stream=True)
                                node.logger.debug(
                                    f"Sent {response.msg_type.name} (id={response.request_id}) "
                                    f"via stream {stream_id}: {len(response_bytes)} bytes"
                                )
                            else:
                                self._quic.send_datagram_frame(response_bytes)
                                node.logger.debug(
                                    f"Sent {response.msg_type.name} (id={response.request_id}) "
                                    f"via datagram: {len(response_bytes)} bytes"
                                )
                            self.transmit()
                        except Exception as send_err:
                            node.logger.error(f"Failed to send response: {send_err}")
                except ValueError as e:
                    node.logger.warning(f"Invalid message from {self._peer_address}: {e}")
                except Exception as e:
                    node.logger.exception(f"Error handling message from {self._peer_address}: {e}")

        stream_handler = kwargs.get('stream_handler')
        return _ServerProtocol(quic, stream_handler)

    async def start(self):
        """Start the P2P node."""
        self.running = True

        # If public_host was not explicitly configured, detect the public IP.
        # get_local_ip() returns the LAN address which is unreachable from
        # remote peers behind NAT, causing JOIN rejections and ban escalation.
        if not self._public_host_explicit:
            public_ip = await get_public_ip()
            if public_ip:
                old = self.public_host
                self.public_host = f"{public_ip}:{self._public_port}"
                self.logger.info(
                    f"Auto-detected public IP: {old} -> {self.public_host}"
                )
                # Re-filter in case the auto-detected public address is
                # sitting in the initial peer list.
                self.initial_peers = self._filter_self_from_peers(self.initial_peers)
            else:
                self.logger.warning(
                    f"Could not detect public IP, using {self.public_host}"
                )

        # Initialize TOFU trust store if enabled
        if self.tofu_enabled:
            import os
            from shared.trust_store import TrustStore
            trust_db_path = os.path.expanduser(self.trust_db)
            self.trust_store = TrustStore(trust_db_path, logger=self.logger)
            await self.trust_store.initialize()
            self.logger.info(f"TOFU trust store initialized at {trust_db_path}")

        # Initialize QUIC client for peer communication (with TOFU support)
        self.node_client = NodeClient(
            node_timeout=self.node_timeout,
            logger=self.logger,
            verify_tls=self.verify_tls,
            trust_store=self.trust_store
        )
        # Share a single ban list between NetworkNode and NodeClient so that
        # protocol-level bans (version mismatch) also block QUIC connections.
        self.node_client.ban_list = self._own_ban_list
        self._ban_list = self._own_ban_list
        await self.node_client.start()

        # Start connection worker process for non-blocking peer discovery
        from shared.connection_worker import ConnectionWorkerHandle
        self._connection_worker = ConnectionWorkerHandle(
            node_timeout=self.node_timeout,
        )
        self._connection_worker.start()

        # Initialize process pool for per-connection peer isolation
        pool_cfg = ProcessPoolConfig(
            max_connections=self.max_connections,
            node_timeout=self.node_timeout,
            verify_tls=self.verify_tls,
        )
        self._process_pool = ProcessPool(config=pool_cfg, logger=self.logger)

        # Start QUIC server
        cert_file = self.tls_cert_file if self.tls_enabled else None
        key_file = self.tls_key_file if self.tls_enabled else None
        if not cert_file or not key_file:
            self.logger.warning("No TLS certificates provided, generating self-signed certificate")
            cert_file, key_file = generate_self_signed_cert()

        configuration = QuicConfiguration(
            is_client=False,
            max_datagram_frame_size=MAX_DATAGRAM_FRAME_SIZE,
            alpn_protocols=[QUIP_ALPN_PROTOCOL],
            idle_timeout=300.0,
        )
        configuration.load_cert_chain(cert_file, key_file)

        self._quic_server = await serve(
            host=self.bind_address,
            port=self.port,
            configuration=configuration,
            create_protocol=self._create_server_protocol,
        )

        # Suppress aioquic assertion errors from malformed packets
        # (e.g. v1 nodes sending non-INITIAL first packets).
        # Guard against re-installing the handler on repeated start()
        # calls — otherwise _default_handler would chain previous
        # installs and leak memory.
        loop = asyncio.get_running_loop()
        # Log any callback that blocks the event loop for >250ms. Bursts here
        # cluster around new-block arrivals when synchronous CPU work (SPHINCS+
        # verify, PoW validation) runs on the main loop and stalls gossip
        # handlers. Once that work is offloaded, these should become rare.
        loop.slow_callback_duration = 0.25
        if not getattr(loop, "_quip_quic_handler_installed", False):
            _default_handler = loop.get_exception_handler()

            def _quic_exception_handler(loop_arg, context):
                exc = context.get("exception")
                if (isinstance(exc, AssertionError)
                        and "first packet must be INITIAL" in str(exc)):
                    self.logger.debug(
                        "Dropped malformed QUIC packet "
                        "(non-INITIAL first packet)"
                    )
                    return
                if _default_handler:
                    _default_handler(loop_arg, context)
                else:
                    loop_arg.default_exception_handler(context)

            loop.set_exception_handler(_quic_exception_handler)
            loop._quip_quic_handler_installed = True

        # Suppress noisy aioquic internal warnings (CRYPTO frame errors
        # from incompatible peers)
        logging.getLogger("quic").setLevel(logging.ERROR)

        self.logger.info(
            f"Network node {self.node_name} ({self.crypto.ecdsa_public_key_hex[:8]}) "
            f"started at quic://{self.bind_address}:{self.port} with public address {self.public_host}"
        )

        # Start REST API server if enabled
        if self.rest_api_enabled:
            from shared.rest_api import RestApiServer
            from shared.certificate_manager import CertificateManager

            # Only set up cert manager when HTTPS port is enabled
            cert_manager = None
            if self.rest_port > 0:
                cert_config = {
                    "rest_tls_cert_file": (
                        self.config.get("rest_tls_cert_file")
                        or self.config.get("tls_cert_file")
                    ),
                    "rest_tls_key_file": (
                        self.config.get("rest_tls_key_file")
                        or self.config.get("tls_key_file")
                    ),
                }
                cert_manager = CertificateManager(cert_config, logger=self.logger)
            tapi = self._telemetry_api_config
            self.rest_api_server = RestApiServer(
                network_node=self,
                host=self.rest_host,
                port=self.rest_insecure_port,
                tls_port=self.rest_port,
                cert_manager=cert_manager,
                webroot=self.webroot,
                logger=self.logger,
                telemetry_cache=self.telemetry_cache,
                telemetry_access_token=tapi.get("access_token", ""),
                telemetry_rate_limit_rpm=tapi.get("rate_limit_rpm", 60),
                telemetry_max_sse=tapi.get("max_sse_connections", 20),
            )
            await self.rest_api_server.start()

        # Start telemetry cache
        if self.telemetry_cache is not None:
            await self.telemetry_cache.start()

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self.node_cleanup_loop())
        self.block_processor_task = asyncio.create_task(self.block_processor_loop())
        self.gossip_processor_task = asyncio.create_task(self.gossip_processor_loop())
        self.server_task = asyncio.create_task(self.server_loop())
        self.rebalance_task = asyncio.create_task(self.rebalance_loop())

        # have we fully synchronized with the network at least one time?
        self._synchronized = threading.Event()
        self._has_ever_had_peers = False  # distinguishes bootstrap (solo) from desync (lost peers)
        self.sync_block_cache = {}  # Regular dict is thread-safe for simple assignments in CPython

        # Sync failure tracking for consensus fallback
        self._sync_failure_count: int = 0
        self._last_sync_target: int = 0
        self._max_sync_failures: int = 3
        
        # Initialize stats cache
        asyncio.create_task(self._update_stats_cache())

        # Telemetry: periodic node table log and epoch sync for mid-epoch joins
        self.telemetry._periodic_task = asyncio.create_task(
            self.telemetry.start_periodic_log(interval=600.0)
        )
        self.telemetry.sync_epoch_from_chain(self.chain)

    @property
    def synchronized(self) -> bool:
        """Check if node is synchronized with the network."""
        return self._synchronized.is_set()

    def set_synchronized(self) -> None:
        """Mark node as synchronized with the network."""
        self._synchronized.set()

    async def stop(self):
        """Stop the P2P node."""
        async with self.net_lock:
            if not self.running:
                return
            self.running = False
        self.logger.info("Shutting down network node...")

        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.block_processor_task:
            self.block_processor_task.cancel()
        if self.gossip_processor_task:
            self.gossip_processor_task.cancel()
        if self.server_task:
            self.server_task.cancel()
        if hasattr(self, 'rebalance_task') and self.rebalance_task:
            self.rebalance_task.cancel()
        self.telemetry.stop()
        if self.reset_timer_task:
            self.reset_timer_task.cancel()

        # Stop telemetry cache
        if self.telemetry_cache is not None:
            await self.telemetry_cache.stop()

        # Stop REST API server if running
        if self.rest_api_server:
            self.logger.info("Stopping REST API server...")
            await self.rest_api_server.stop()

        # Stop connection worker before closing the QUIC client
        if self._connection_worker:
            self.logger.info("Stopping connection worker...")
            self._connection_worker.close()
            self._connection_worker = None

        # Shut down process pool (per-connection processes)
        if self._process_pool:
            self.logger.info("Stopping process pool...")
            self._process_pool.shutdown_all(timeout=5.0)
            self._process_pool = None

        self.logger.info("Cancelling QUIC client tasks...")
        # Close node client
        if self.node_client:
            await self.node_client.stop()

        self.logger.info("Cancelling QUIC server tasks...")
        # Stop QUIC server
        if self._quic_server:
            self._quic_server.close()
        

        all_tasks = asyncio.all_tasks()
        self.logger.info(f"Total active tasks: {len(all_tasks)}")
        for task in all_tasks:
            if not task.done():
                self.logger.info(f"Active task: {task.get_coro().__name__}") # type: ignore

        # Stop miner workers
        self.logger.info("Cancelling miner workers...")
        await self.stop_mining()
        self.logger.info("Stopping miner worker processes...")
        self.close()

        self.logger.info("Network node stopped")

    ##########################
    ## Server logic threads ##
    ##########################

    async def heartbeat_loop(self):
        """Send heartbeats to all known nodes with SWIM probing."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all nodes
                tasks = []
                async with self.net_lock:
                    peer_list = list(self.peers.keys())
                    for node_host in peer_list:
                        task = asyncio.create_task(
                            self._heartbeat_with_swim(node_host)
                        )
                        tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # SWIM: create and send indirect probes for suspects
                await self._run_swim_probes(peer_list)

                # SWIM: reap dead peers every heartbeat cycle (faster
                # than waiting for the 60s rebalance loop).
                self._swim_detector.expire_probes()
                for peer in self._swim_detector.get_dead_peers():
                    self.logger.warning(
                        f"SWIM declared {peer} dead, removing"
                    )
                    self._swim_detector.remove_peer(peer)
                    await self.remove_node(peer)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in heartbeat loop")

    async def _heartbeat_with_swim(self, node_host: str) -> None:
        """Send heartbeat and update SWIM/scoring based on result."""
        ok = await self.send_heartbeat(node_host)
        if ok:
            self._swim_detector.record_heartbeat_success(node_host)
            self._peer_scorer.record_heartbeat_ok(node_host)
        else:
            self._swim_detector.record_heartbeat_failure(node_host)
            self._peer_scorer.record_heartbeat_fail(node_host)
            state = self._swim_detector.get_state(node_host)
            if state == PeerState.SUSPECT:
                self.logger.debug(
                    f"SWIM: {node_host} now SUSPECT "
                    f"(direct heartbeat failed)"
                )

    async def _run_swim_probes(self, peer_list: list[str]) -> None:
        """Send SWIM indirect probe requests for all suspect peers."""
        probe_requests = self._swim_detector.create_probe_requests(
            peer_list
        )
        if not probe_requests:
            return

        self.logger.debug(
            f"SWIM: sending {len(probe_requests)} indirect probes"
        )
        tasks = []
        for req in probe_requests:
            tasks.append(asyncio.create_task(
                self._send_swim_probe(req)
            ))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_swim_probe(self, req) -> None:
        """Send a single SWIM probe request and record the result."""
        result = await self.node_client.send_probe_request(
            req.prober, req.target, req.probe_id
        )
        if result is not None:
            self._swim_detector.record_probe_result(
                req.target, req.prober, result
            )

    async def node_cleanup_loop(self):
        """Remove dead nodes from registry and prune stale state."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_timeout / 2)

                # Find dead nodes.  Peers that have never had a
                # successful heartbeat are evicted after node_timeout
                # (60s) instead of the full heartbeat_timeout (300s).
                current_time = utc_timestamp_float()
                now_mono = time.monotonic()
                dead_nodes = []

                async with self.net_lock:
                    for host in list(self.peers.keys()):
                        hb_time = self.heartbeats.get(host)
                        if hb_time is not None:
                            if current_time - hb_time > self.heartbeat_timeout:
                                dead_nodes.append(host)
                        else:
                            # Never got a heartbeat — use fast timeout.
                            joined_at = self._swim_detector.get_joined_at(host)
                            if joined_at is None:
                                # Peer isn't tracked by SWIM (race with
                                # remove_peer, or never registered). Assume
                                # it's past the fast-timeout window and log
                                # so an add_peer/register mismatch is
                                # discoverable.
                                self.logger.debug(
                                    f"Unproven peer {host} missing from "
                                    "SWIM detector; evicting via fast timeout"
                                )
                                dead_nodes.append(host)
                            elif now_mono - joined_at > self.node_timeout:
                                dead_nodes.append(host)

                # Prune stale gossip dedup entries and rate limiter buckets
                async with self.gossip_lock:
                    pruned = self._prune_recent_messages()
                    announced_pruned = self._prune_announced_nodes()
                    if pruned:
                        self.logger.debug(
                            f"Pruned {pruned} stale gossip dedup entries "
                            f"({len(self.recent_messages)} remaining)"
                        )
                    if announced_pruned:
                        self.logger.debug(
                            f"Pruned {announced_pruned} stale announced-node "
                            f"entries ({len(self._announced_nodes)} remaining)"
                        )
                self._rate_limiter.prune()

                # Reap dead connection processes
                if self._process_pool:
                    dead_procs = self._process_pool.reap_dead()
                    for peer in dead_procs:
                        self.logger.info(
                            f"Connection process for {peer} died, "
                            f"will reconnect on next cycle"
                        )

                # Remove dead nodes
                for host in dead_nodes:
                    await self.remove_node(host)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in cleanup loop")

    async def rebalance_loop(self):
        """Periodic load-based connection rebalancing.

        Runs every 60s. If the node is overloaded, sheds excess
        connections by sending MIGRATE to the least-active peers.
        Also broadcasts load info to peers and runs SWIM probing.
        """
        while self.running:
            try:
                await asyncio.sleep(60)

                # Update load monitor with current metrics
                pool_count = (
                    self._process_pool.connection_count
                    if self._process_pool else 0
                )
                self._load_monitor.update(
                    connection_count=pool_count,
                    block_queue=self.block_processing_queue.qsize()
                    if hasattr(self, 'block_processing_queue') else 0,
                    gossip_queue=self.gossip_processing_queue.qsize()
                    if hasattr(self, 'gossip_processing_queue') else 0,
                )

                # Broadcast load info to peers
                load_snapshot = self._load_monitor.snapshot()
                load_dict = load_snapshot.to_dict()
                load_dict["sender"] = self.public_host
                async with self.net_lock:
                    peer_list = list(self.peers.keys())
                for peer in peer_list:
                    try:
                        await self.node_client.send_load_info(
                            peer, load_dict
                        )
                    except (ConnectionError, asyncio.TimeoutError, OSError):
                        pass  # Expected network errors
                    except Exception as exc:
                        self.logger.debug(
                            f"Failed to send load info to {peer}: {exc}"
                        )

                # Check if we need to shed connections
                if self._load_monitor.is_overloaded():
                    to_shed = self._load_monitor.connections_to_shed()
                    if to_shed > 0 and self._process_pool:
                        await self._shed_connections(to_shed)

                # Peer scoring: decay and disconnect low scorers
                self._peer_scorer.decay_scores()
                low_scorers = self._peer_scorer.get_low_scoring_peers()
                for peer in low_scorers:
                    self.logger.info(
                        f"Disconnecting low-scoring peer {peer} "
                        f"(score={self._peer_scorer.get_score(peer):.1f})"
                    )
                    self._peer_scorer.remove_peer(peer)
                    await self.remove_node(peer)

                # Block inventory: expire stale IWANT requests
                expired_wants = self._block_inventory.expire_wants()
                for block_hash, peer in expired_wants:
                    self.logger.debug(
                        f"IWANT timeout for {block_hash.hex()[:16]}... "
                        f"from {peer}"
                    )

                # SWIM: expire stale probes
                self._swim_detector.expire_probes()

                # SWIM: handle dead peers
                dead_peers = self._swim_detector.get_dead_peers()
                for peer in dead_peers:
                    self.logger.warning(
                        f"SWIM declared {peer} dead, removing"
                    )
                    self._swim_detector.remove_peer(peer)
                    await self.remove_node(peer)

                # Promote candidates into freed active slots
                if self._candidate_peers:
                    await self._try_promote_candidates()

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Error in rebalance loop")

    async def _shed_connections(self, count: int) -> None:
        """Shed excess connections by sending MIGRATE to least-active peers."""
        if not self._process_pool:
            return

        targets = self._process_pool.get_least_active_peers(count)
        if not targets:
            return

        # Find least-loaded peers to suggest as alternatives
        async with self.net_lock:
            snapshot = list(self.peers.keys())
        alternatives = self._get_least_loaded_peers(
            exclude=set(targets), count=5, peer_list=snapshot
        )

        self.logger.info(
            f"Shedding {len(targets)} connections "
            f"(suggesting {len(alternatives)} alternatives)"
        )

        for peer in targets:
            try:
                await self.node_client.send_migrate(
                    peer, alternatives, reason="overloaded"
                )
            except Exception as exc:
                self.logger.warning(f"MIGRATE to {peer} failed: {exc}")
            # Kill the connection process regardless of MIGRATE success
            self._process_pool.kill(peer)

    def _get_least_loaded_peers(
        self, exclude: set, count: int,
        peer_list: Optional[list[str]] = None,
    ) -> list[str]:
        """Select the least-loaded known peers as alternatives.

        Uses peer load info if available, otherwise falls back to
        random selection from the peer list.

        Args:
            peer_list: Pre-copied peer list (caller should copy under
                net_lock). Falls back to self.peers if not provided.
        """
        # Sort peers by load (connection utilization)
        candidates = []
        async_peers = peer_list if peer_list is not None else list(self.peers.keys())
        for peer in async_peers:
            if peer in exclude:
                continue
            load = self._peer_loads.get(peer)
            if load:
                util = (
                    load.connection_count / load.max_connections
                    if load.max_connections > 0 else 0.0
                )
                candidates.append((peer, util))
            else:
                # Unknown load — assume moderate utilization
                candidates.append((peer, 0.5))

        candidates.sort(key=lambda x: x[1])
        return [p for p, _ in candidates[:count]]

    def _healthy_peers_snapshot(self) -> Dict[str, MinerInfo]:
        """Return active peers with a recent successful heartbeat.

        Must be called while holding net_lock.  Uses
        ``heartbeat_timeout / 2`` as the recency cutoff so we only
        share peers that are demonstrably alive.
        """
        now = utc_timestamp_float()
        cutoff = self.heartbeat_timeout / 2
        healthy = {
            host: info
            for host, info in self.peers.items()
            if host in self.heartbeats
            and (now - self.heartbeats[host]) < cutoff
        }
        # Surface the bootstrap case where the active set isn't empty
        # but nothing is fresh enough to share — otherwise new joiners
        # silently see an empty peer list and operators have no signal.
        if not healthy and self.peers:
            self.logger.info(
                f"Filtered JOIN/PEERS response empty: {len(self.peers)} "
                "active peers but none have a recent heartbeat"
            )
        return healthy

    async def server_loop(self):
        """Main server loop."""
        while self.running:
            try:
                # Check if we are connected to any active peers, if not,
                # submit connection requests to the worker process (non-blocking)
                # and poll for results.
                connected = await self.is_connected()
                if not connected:
                    self._request_peer_connections()
                    connected = await self._drain_connection_results()

                # If we are not connected and not in auto-mine mode,
                # poll for connection results while waiting.
                if not connected and not self.auto_mine:
                    self.logger.error(f"Not connected to network, retrying in {self.node_timeout} seconds...")
                    deadline = time.time() + self.node_timeout
                    while time.time() < deadline and self.running:
                        await asyncio.sleep(1)
                        connected = await self._drain_connection_results()
                        if connected:
                            break
                    continue
                elif not connected and self.auto_mine:
                    self.logger.info("No peers connected, automining...")

                # Wait for any outstanding blocks to be processed
                if self.block_processing_queue.qsize() > 0:
                    self.logger.info(f"Waiting for {self.block_processing_queue.qsize()} blocks to be processed before mining...")
                    await asyncio.sleep(0.1)
                    continue

                # Check if we are in synchronized state with peers
                # If not, stop mining and synchronize.
                latest_block = await self.check_synchronized()
                if latest_block != 0:
                    if self._is_mining:
                        await self.stop_mining()
                        self.logger.info("Stopped mining to synchronize with network...")

                    # Track repeated failures for the same sync target
                    if latest_block == self._last_sync_target:
                        self._sync_failure_count += 1
                    else:
                        self._sync_failure_count = 0
                        self._last_sync_target = latest_block

                    # After repeated failures, check if peers agree on our height
                    if self._sync_failure_count >= self._max_sync_failures:
                        consensus = await self._query_peer_consensus()
                        my_height = self.get_latest_block().header.index
                        if consensus is not None and my_height >= consensus:
                            self.logger.info(
                                f"Peer consensus height={consensus}, "
                                f"local height={my_height}. "
                                f"Declaring synchronized after "
                                f"{self._sync_failure_count} sync failures."
                            )
                            self.set_synchronized()
                            self._sync_failure_count = 0
                            self._last_sync_target = 0
                            continue

                        # Exponential backoff before retrying
                        backoff = min(30, 2 ** self._sync_failure_count)
                        self.logger.warning(
                            f"Sync failed {self._sync_failure_count} times "
                            f"for target {latest_block}, backing off {backoff}s"
                        )
                        await asyncio.sleep(backoff)

                    success = await self.synchronize_blockchain(latest_block)
                    if not success:
                        self._sync_failure_count += 1
                    else:
                        self._sync_failure_count = 0
                        self._last_sync_target = 0
                    continue

                # If we are synchronized, check if we are mining. If not, start mining on the next block.
                if not self._is_mining:
                    # Guard: Don't start mining unless we're actually synchronized
                    if not self.synchronized:
                        self.logger.debug("Not starting mining - not synchronized with network")
                        await asyncio.sleep(1)
                        continue
                    latest_block = self.get_latest_block()
                    # Create task with exception handler to crash on ValueError
                    task = asyncio.create_task(self.mine_block(latest_block))
                    task.add_done_callback(self._handle_mining_task_exception)
                    # Short sleep to allow mining task to start
                    await asyncio.sleep(0.1)
                else:
                    # If mining is active, sleep longer to avoid busy-waiting
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                if not self.running:
                    break
            except Exception as e:
                self.logger.exception(f"Error in server loop: {e}")
                # Sleep on error to avoid tight loop
                await asyncio.sleep(1)

    async def block_processor_loop(self):
        """Background loop to process blocks without blocking HTTP handlers."""
        while self.running:
            try:
                # Wait for blocks to process with timeout
                try:
                    block_data = await asyncio.wait_for(
                        self.block_processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the block in background
                # Unpack with optional force_reorg flag and peer_address (default False/None for backward compat)
                if len(block_data) >= 4:
                    block, response_future, force_reorg, peer_address = block_data
                elif len(block_data) == 3:
                    block, response_future, force_reorg = block_data
                    peer_address = None
                else:
                    block, response_future = block_data
                    force_reorg = False
                    peer_address = None
                try:
                    latest = self.get_latest_block()
                    # Cache out of order blocks for later processing
                    # NOTE: older blocks need processing to determine chain fork
                    if latest.header.index+1 < block.header.index:
                        self.sync_block_cache[block.header.index] = (block, force_reorg)
                        # WE return failure, but thats only to signal we didn't process it.
                        response_future.set_result(False)
                        continue
                    # Base case we can process the block
                    accepted, reason = await self.receive_block(block, force_reorg=force_reorg)
                    if not accepted:
                        miner_id = block.miner_info.miner_id if block.miner_info else "unknown"
                        source = f" (via {peer_address})" if peer_address else ""
                        reason_str = f": {reason}" if reason else ""
                        self.logger.warning(f"Block {block.header.index} mined by {miner_id}{source} was rejected{reason_str}")
                    response_future.set_result(accepted)
                except Exception as e:
                    self.logger.info(f"Error processing block: {e}")
                    if not response_future.done():
                        response_future.set_result(False)

                # Try to process any cached blocks
                await self._exhaust_block_cache()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in block processor loop")

    async def gossip_processor_loop(self):
        """Background loop to process gossip messages without blocking HTTP handlers.

        Emits two diagnostic signals:
          - INFO per message only when ``proc_ms > 500`` (a genuinely
            slow handler). Queue-wait time is excluded from the
            per-message trigger because a single backlog fans out to
            hundreds of entries with identical wait values.
          - INFO summary every ``_GOSSIP_SUMMARY_INTERVAL`` seconds
            capturing drained count, avg/max wait, and current qsize.
            This is the right place to observe backlog conditions.
        """
        _SLOW_PROC_MS = 500.0
        _SUMMARY_INTERVAL = 10.0

        window_count = 0
        window_wait_sum = 0.0
        window_wait_max = 0.0
        window_started = time.monotonic()

        while self.running:
            try:
                now = time.monotonic()
                if now - window_started >= _SUMMARY_INTERVAL:
                    if window_count > 0:
                        avg_wait = window_wait_sum / window_count
                        qsize_now = self.gossip_processing_queue.qsize()
                        self.logger.info(
                            f"🧩 Gossip summary: drained {window_count} msgs "
                            f"in last {_SUMMARY_INTERVAL:.0f}s, "
                            f"avg_wait={avg_wait:.1f} ms, "
                            f"max_wait={window_wait_max:.1f} ms, "
                            f"qsize_now={qsize_now}"
                        )
                    window_count = 0
                    window_wait_sum = 0.0
                    window_wait_max = 0.0
                    window_started = now

                # Wait for gossip messages to process with timeout
                try:
                    gossip_data = await asyncio.wait_for(
                        self.gossip_processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Expect instrumented tuple: (message, response_future, t_enq)
                message, response_future, t_enq = gossip_data

                t_deq = time.perf_counter()
                try:
                    t0 = time.perf_counter()
                    result = await self.handle_gossip(message)
                    t1 = time.perf_counter()
                    wait_ms = ((t_deq - t_enq) * 1000.0) if t_enq is not None else None
                    proc_ms = (t1 - t0) * 1000.0
                    qsize = self.gossip_processing_queue.qsize()

                    window_count += 1
                    if wait_ms is not None:
                        window_wait_sum += wait_ms
                        if wait_ms > window_wait_max:
                            window_wait_max = wait_ms

                    wait_str = f"{wait_ms:.1f} ms" if wait_ms is not None else "n/a"
                    if proc_ms > _SLOW_PROC_MS:
                        self.logger.info(
                            f"🧩 Slow gossip handler id={(message.id or '')[:8]} "
                            f"type={message.type}: wait={wait_str}, "
                            f"process={proc_ms:.1f} ms, qsize={qsize}"
                        )
                    else:
                        self.logger.debug(
                            f"🧩 Gossip handled id={(message.id or '')[:8]} "
                            f"type={message.type}: wait={wait_str}, "
                            f"process={proc_ms:.1f} ms, qsize={qsize}"
                        )

                    if response_future and not response_future.done():
                        try:
                            response_future.set_result(result)
                        except Exception as e:
                            self.logger.warning(f"Failed to set result on future: {e}")
                except Exception as e:
                    self.logger.exception(f"Error processing gossip: {e}")
                    if response_future and not response_future.done():
                        try:
                            response_future.set_result("error")
                        except Exception as e:
                            self.logger.warning(f"Failed to set error result on future: {e}")

            except asyncio.CancelledError:
                break

    def _handle_mining_task_exception(self, task: asyncio.Task):
        """Handle exceptions from mining tasks."""
        if task.done() and not task.cancelled():
            try:
                task.result()
            except ValueError as e:
                self.logger.error(f"ValueError in mining task: {e}")
            except Exception as e:
                self.logger.error(f"Exception in mining task: {e}")

    async def _exhaust_block_cache(self):
        """Exhaust the block cache by processing all blocks in order."""
        # Don't exhaust cache during sync - blocks will be processed after sync completes
        if not self.synchronized:
            return
        # Pause gossip and process the current block cache.
        async with self.gossip_lock:
            # Process cached blocks starting from end_index + 1
            next_block_index = self.get_latest_block().header.index + 1
            while next_block_index in self.sync_block_cache:
                cached_data = self.sync_block_cache.pop(next_block_index)
                # Handle both old format (block only) and new format (block, force_reorg)
                if isinstance(cached_data, tuple):
                    cached_block, force_reorg = cached_data
                else:
                    cached_block = cached_data
                    force_reorg = False
                self.logger.info(f"Processing cached block {next_block_index} received during sync")

                # Process the cached block
                success, reason = await self.receive_block(cached_block, force_reorg=force_reorg)
                if not success:
                    reason_str = f": {reason}" if reason else ""
                    self.logger.error(f"Failed to process cached block {next_block_index}{reason_str}")
                    break

                next_block_index += 1



    #############################
    ## Internal Event Handlers ##
    #############################

    def _on_new_node(self, host, info: MinerInfo):
        self.logger.info(f"New node joined: {host} {info.miner_id} ({info.ecdsa_public_key.hex()[:8]})")
        if self.on_new_node:
            try:
                self.on_new_node(host, info)
            except Exception as e:
                self.logger.error(f"Error in on_new_node callback: {e}")

    def _on_node_lost(self, host):
        self.logger.info(f"Node lost: {host}")
        if self.on_node_lost:
            try:
                self.on_node_lost(host)
            except Exception as e:
                self.logger.error(f"Error in on_node_lost callback: {e}")

    def _on_block_received(self, block: Block):       
        # Check if we've reached block 1024 and need to initiate reset
        if block.header.index == self.max_sync_block_index and not self.reset_scheduled:
            self.logger.info(f"Block limit ({self.max_sync_block_index}) reached - starting 5-minute reset coordination timer")
            self.reset_scheduled = True
            self.reset_start_time = utc_timestamp_float()
            
            # Start the 5-minute reset timer - schedule in event loop from sync context
            try:
                loop = asyncio.get_event_loop()
                self.reset_timer_task = loop.create_task(self._reset_coordination_timer())
            except RuntimeError:
                # No event loop running, skip reset timer
                self.logger.warning("No event loop running, cannot schedule reset timer")
        
        # Record block telemetry
        self.telemetry.record_block(block)

        # Trigger stats cache update in background
        try:
            loop = asyncio.get_event_loop()
            asyncio.create_task(self._update_stats_cache())
        except RuntimeError:
            # No event loop running, skip cache update
            pass
        
        # Call sync callback directly (no async task needed)
        if self.on_block_received:
            try:
                self.on_block_received(block)
            except Exception as e:
                self.logger.error(f"Error in on_block_received callback: {e}")

    def _network_on_mining_started(self, prev: Block):
        assert prev.hash
        self.logger.info(f"🏁 Mining started on block {prev.header.index + 1} with previous block hash {prev.hash.hex()[:8]}...")

    def _network_on_mining_stopped(self):
        self.logger.info(f"🛑 Mining stopped")

    #######################
    ## Reset Coordination ##
    #######################

    async def _reset_coordination_timer(self):
        """5-minute timer for reset coordination period."""
        try:
            # Execute the reset after timer expires
            self.logger.info("Reset coordination period complete - executing chain reset in 5 minutes")

            # Wait 5 minutes (300 seconds)
            await asyncio.sleep(300)
            
            await self._execute_chain_reset()
            
        except asyncio.CancelledError:
            self.logger.debug("Reset coordination timer cancelled")
        except Exception as e:
            self.logger.error(f"Error in reset coordination timer: {e}")

    async def _execute_chain_reset(self):
        """Execute the chain reset back to genesis block."""
        try:
            # Stop any active mining
            if self._is_mining:
                await self.stop_mining()
                self.logger.info("Stopped mining for chain reset")
            
            # Reset blockchain state
            async with self.chain_lock:
                # Save previous epoch to disk if storage is enabled and we have one
                if self.enable_epoch_storage and self.epoch_block_store and self.previous_epoch:
                    try:
                        # Save current chain as the previous epoch
                        saved_path = self.epoch_block_store.save(self.chain)
                        self.logger.info(f"Previous epoch chain saved to disk: {saved_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save previous epoch to disk: {e}")
                
                # Capture current epoch info before reset
                if len(self.chain) > 1:  # Only if we have blocks beyond genesis
                    head_block = self.get_latest_block()
                    block_1 = self.get_block(1) if len(self.chain) > 1 else None
                    
                    if head_block and block_1 and block_1.hash and head_block.hash:
                        # Store current epoch as the new previous epoch (replacing any existing one)
                        self.previous_epoch = EpochInfo(
                            first_hash=block_1.hash,
                            last_timestamp=head_block.header.timestamp,
                            last_index=head_block.header.index,
                            last_hash=head_block.hash
                        )
                        self.logger.info(f"Recorded previous epoch: block 1 hash {block_1.hash.hex()[:8]}..., "
                                       f"last block {head_block.header.index} hash {head_block.hash.hex()[:8]}...")
                
                # Reset telemetry epoch for next cycle
                self.telemetry.set_epoch_timestamp(None)

                # Reset to original genesis block (no more new genesis creation)
                self._index_truncate(0)
                self._index_append(self.genesis_block)
                self.logger.info("Chain reset to genesis block completed")
            
            # Reset synchronization state if it exists (only after start() is called)
            if hasattr(self, '_synchronized'):
                self._synchronized.clear()
            if hasattr(self, '_has_ever_had_peers'):
                self._has_ever_had_peers = False
            if hasattr(self, 'sync_block_cache'):
                self.sync_block_cache.clear()
            
            # Reset coordination state
            self.reset_scheduled = False
            self.reset_start_time = None
            self.reset_timer_task = None
            
            self.logger.info("📄 Chain reset to genesis completed - resuming normal operation")
            
        except Exception as e:
            self.logger.error(f"Error executing chain reset: {e}")

    #######################
    ## Control functions ##
    #######################

    async def _update_stats_cache(self):
        """Update stats cache in background without blocking."""
        try:
            # Get fresh stats (potentially expensive operation)
            fresh_stats = self.get_stats()

            # Include the whitelisted node descriptor for remote consumers.
            fresh_stats["descriptor"] = self.descriptor()

            # Add network-specific information
            clock = get_network_clock()
            fresh_stats.update({
                "network": {
                    "host": self.public_host,
                    "running": self.running,
                    "total_peers": len(self.peers),
                    "synchronized": self.synchronized,
                    "auto_mine": self.auto_mine,
                    "queue_sizes": {
                        "block_processing": self.block_processing_queue.qsize(),
                        "gossip_processing": self.gossip_processing_queue.qsize(),
                    }
                },
                "network_clock": {
                    "offset_seconds": round(clock.get_offset(), 1),
                    "is_trusted": clock.is_trusted(),
                }
            })
            
            # Atomically update the cache
            async with self._stats_cache_lock:
                self._stats_cache = fresh_stats
                
            self.logger.debug("Stats cache updated successfully")
            
        except Exception as e:
            self.logger.exception(f"Error updating stats cache: {e}")

    async def receive_block(self, block: Block, force_reorg: bool = False) -> tuple[bool, str | None]:
        """Receive a block from the network with epoch validation and max block limit.

        Args:
            block: The block to receive.
            force_reorg: If True, skip timestamp comparison to allow chain reorganization
                        during sync (longest chain wins). Default False.

        Returns:
            Tuple of (accepted: bool, rejection_reason: str | None).
        """
        # Reject blocks that are too far in the future (beyond max sync limit)
        if block.header.index > self.max_sync_block_index:
            reason = f"index > max_sync_block_index {self.max_sync_block_index}"
            self.logger.debug(f"Block {block.header.index} rejected: {reason}")
            return False, reason

        # Check against previous epoch to prevent accepting blocks from old chain epoch
        async with self.chain_lock:
            if self.previous_epoch:
                # Reject if this is block 1 from the previous epoch
                if block.header.index == 1 and block.hash and block.hash == self.previous_epoch.first_hash:
                    reason = "matches previous epoch first_hash"
                    self.logger.warning(f"Block 1 rejected: hash {block.hash.hex()[:8]}... {reason}")
                    return False, reason

                # Reject if this block hash matches the last block from the previous epoch
                if block.hash and block.hash == self.previous_epoch.last_hash:
                    reason = "matches previous epoch last_hash"
                    self.logger.warning(f"Block {block.header.index} rejected: hash {block.hash.hex()[:8]}... {reason}")
                    return False, reason

                # Reject if timestamp is older than the last timestamp from the previous epoch
                if block.header.timestamp <= self.previous_epoch.last_timestamp:
                    reason = f"timestamp <= previous epoch ({block.header.timestamp} <= {self.previous_epoch.last_timestamp})"
                    self.logger.warning(f"Block {block.header.index} rejected: {reason}")
                    return False, reason

        # If all validations pass, call parent receive_block
        return await super().receive_block(block, force_reorg=force_reorg)

    async def mine_block(self, previous_block: Block, transactions=None) -> Optional[MiningResult]:
        """Mine a block and broadcast if successful."""
        # Pull pending transactions if not provided
        if transactions is None:
            async with self.transactions_lock:
                transactions = self.pending_transactions.copy()
                self.pending_transactions.clear()
            if transactions:
                self.logger.info(f"Including {len(transactions)} transaction(s) in block")

        result = await super().mine_block(previous_block, transactions)
        if not result:
            return None

        data = f"{self.node_id} was here"
        wb = self.build_block(previous_block, result, data.encode(), transactions)
        if wb is None:
            self.logger.warning("Block validation failed, discarding mining result (see debug report)")
            return None
        wb = self.sign_block(wb)

        if not wb.hash:
            raise ValueError("Failed to finalize block")

        self.logger.info(f"Candidate Block {wb.header.index}-{wb.hash.hex()[:8]} mined on this node!")
    
        if wb.header.index == self.get_latest_block().header.index + 1:
            accepted, reason = await self.receive_block(wb)
            if accepted:
                self.logger.info(f"Accepted block {wb.header.index}-{wb.hash.hex()[:8]} from {wb.miner_info.miner_id}")
                asyncio.create_task(self.gossip_block(wb))
            else:
                self.logger.warning(f"Own block {wb.header.index}-{wb.hash.hex()[:8]} was rejected: {reason}")
        else:
            self.logger.info(f"Candidate Block {wb.header.index}-{wb.hash.hex()[:8]} sniped by another miner!")

        return result


    async def _query_peer_consensus(self) -> Optional[int]:
        """Query multiple peers for their latest block height.

        Returns the consensus height if a majority of sampled peers
        agree (within 1 block tolerance), or None if no consensus.
        """
        if not self.peers:
            return None

        from collections import Counter

        peer_list = list(self.peers.keys())
        sample_size = min(5, len(peer_list))
        sampled = random.sample(peer_list, sample_size)

        heights: list[int] = []
        for peer in sampled:
            header = await self.get_peer_block_header(peer)
            if header:
                heights.append(header.index)

        if not heights:
            return None

        counts = Counter(heights)
        most_common_height, _ = counts.most_common(1)[0]

        # Count heights within +/-1 of most common
        cluster_count = sum(
            c for h, c in counts.items()
            if abs(h - most_common_height) <= 1
        )

        if cluster_count > len(heights) / 2:
            return most_common_height
        return None

    async def check_synchronized(self) -> int:
        """Check if we are synchronized with the network using header-only fetch.

        Returns 0 if synchronized or the latest network block index if not.
        """
        my_latest_block = self.get_latest_block()
        if not self.peers:
            if self.auto_mine and not self._has_ever_had_peers:
                # True bootstrap: never had peers, solo mining OK
                self.logger.debug("Bootstrap mode: no peers ever seen, auto-mine enabled")
                self.set_synchronized()
                return 0
            elif self.auto_mine:
                # Had peers before but lost them all — do not auto-sync.
                # _synchronized was already cleared by remove_node(),
                # so the mining guard in server_loop will block mining.
                self.logger.debug("All peers lost, waiting for reconnection before mining")
                return 0
            else:
                raise RuntimeError("No peers to synchronize with")

        # Unified version filter with synchronize_blockchain. Peers whose
        # version is not yet observed (transitive peers awaiting their
        # first heartbeat) are excluded — we defer syncing from them until
        # they are positively version-gated.
        compatible_peers = list(
            select_compatible_peers(self.peers, self.peer_versions)
        )

        if not compatible_peers:
            self.logger.warning("No compatible-version peers available for sync")
            return 0

        # Query up to 3 peers and take the highest valid response
        net_latest: Optional[BlockHeader] = None
        sample_size = min(3, len(compatible_peers))
        sampled = random.sample(compatible_peers, sample_size)

        for peer in sampled:
            header = await self.get_peer_block_header(peer)
            if not header:
                continue
            if header.index > self.max_sync_block_index:
                self.logger.debug(f"Ignoring peer {peer} with block index {header.index} > max_sync_block_index {self.max_sync_block_index}")
                continue
            if net_latest is None or header.index > net_latest.index:
                net_latest = header

        if net_latest is None:
            self.logger.warning("Unable to get block headers from peers, assuming synchronized")
            return 0

        if my_latest_block.header.index > net_latest.index:
            self.set_synchronized()
            return 0

        if my_latest_block.header.index == net_latest.index:
            # FIXME: maybe put hash in header?
            # Our prev_hashes match and our timestamps match and my timestamp is older or equal, I am good.
            if my_latest_block.header.previous_hash == net_latest.previous_hash and my_latest_block.header.timestamp <= net_latest.timestamp:
                self.set_synchronized()
                return 0
            else:
                self.logger.warning(f"Fork detected at block {my_latest_block.header.index} - our prev_hash differs from network, triggering reorg sync")
                # Return non-zero to trigger synchronization - longest chain wins
                return net_latest.index

        return net_latest.index

    async def synchronize_blockchain(self, current_head: int = 0) -> bool:
        """Synchronize the blockchain with the network using BlockSynchronizer.

        Args:
            current_head: Target block index. If 0, queries peers.

        Returns:
            True if sync succeeded, False if it failed.
        """
        if self._is_mining:
            self.logger.error("Cannot synchronize while mining")
            return False

        if current_head == 0:
            current_head = await self.check_synchronized()
        if current_head == 0:
            return True

        my_latest_block = self.get_latest_block()

        # Enforce maximum sync block limit
        if current_head > self.max_sync_block_index:
            self.logger.warning(f"Refusing to sync beyond max_sync_block_index {self.max_sync_block_index}, requested: {current_head}")
            return False

        # Go back 6 blocks for reorg depth
        start_index = max(1, my_latest_block.header.index - 6)
        end_index = current_head
        if start_index > end_index:
            return True

        self.logger.info(f"Syncing with network from block {start_index} to {end_index}...")

        if not self.node_client:
            self.logger.error("NodeClient not initialized")
            return False

        # Update node client with only version-confirmed compatible peers
        compatible_peers = select_compatible_peers(
            self.peers, self.peer_versions,
        )
        self.node_client.update_peers(compatible_peers)

        synchronizer = BlockSynchronizer(
            node_client=self.node_client,
            receive_block_queue=self.block_processing_queue,
            local_tip=self.get_latest_block,
            local_locator=self.build_locator,
            local_get_block_by_hash=self.get_block_by_hash,
            logger=self.logger,
        )

        result = await synchronizer.sync_blocks(start_index, end_index)
        if result.success:
            self.logger.info(
                "Synced blocks %d–%d: %s", start_index, end_index,
                result.summary(),
            )
            self.set_synchronized()
            await self._exhaust_block_cache()
        else:
            self.logger.error(
                "Failed to sync blocks %d–%d: %s", start_index, end_index,
                result.summary(),
            )
        return result.success
            

    def _track_peer_timestamp(self, timestamp: float):
        """Track a peer timestamp for time synchronization."""
        current_time = utc_timestamp_float()

        # Feed the global network clock for offset calculation
        get_network_clock().record_peer_timestamp(timestamp)

        # Only track recent timestamps (within last 5 minutes)
        if abs(current_time - timestamp) < 300:
            self.peer_timestamps.append(int(timestamp))

            # Keep only recent timestamps (last 50)
            if len(self.peer_timestamps) > 50:
                self.peer_timestamps = self.peer_timestamps[-50:]

            # Check time synchronization periodically
            if current_time - self.last_time_sync_check > NETWORK_TIME_SYNC_INTERVAL:
                self._check_time_synchronization()
                self.last_time_sync_check = current_time

    def _check_time_synchronization(self):
        """Check if local clock is synchronized with network."""
        if len(self.peer_timestamps) < 3:
            return  # Not enough data

        clock = get_network_clock()
        if not is_clock_synchronized(self.peer_timestamps):
            offset = clock.get_offset()
            self.time_sync_warnings += 1

            if self.time_sync_warnings <= 3:  # Limit warnings
                self.logger.warning(
                    f"⚠️  Clock drift detected: local time is {abs(offset):.0f}s "
                    f"{'ahead' if offset > 0 else 'behind'} network time. "
                    f"Network clock is compensating (timestamps adjusted by {-offset:.0f}s)."
                )
            elif self.time_sync_warnings == 4:
                self.logger.info(f"Clock drift warnings suppressed (compensating by {-offset:.0f}s)")
        else:
            # Reset warning counter if synchronized
            if self.time_sync_warnings > 0:
                self.logger.info("✅ Clock synchronization restored")
                self.time_sync_warnings = 0

    async def is_connected(self) -> bool:
        """Ensure we are connected to the network."""
        for _, status in self.heartbeats.items():
            if utc_timestamp_float() - status < self.heartbeat_timeout:
                return True
        return False

    async def connect_to_network(self) -> bool:
        """Connect to the network."""
        for peer_address in self.initial_peers:
            if await self.connect_to_peer(peer_address):
                return True
        for host in self.heartbeats.keys():
            if await self.connect_to_peer(host):
                return True
        return False

    async def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a peer and join the network."""
        if not self.node_client:
            return False

        # Only enforce bans for non-initial peers. Initial/seed peers
        # must always be retryable so the node can recover from
        # transient network partitions.
        if peer_address not in self.initial_peers and self._is_backed_off(peer_address):
            return False

        try:
            join_data = {
                "host": self.public_host,
                "version": get_version(),
                "capabilities": ["mining", "relay"],
                # Serialize MinerInfo as JSON string for transport
                "info": self.info().to_json(),
                "descriptor": self.descriptor(),
            }

            # Use NodeClient's SSL-aware connection method
            is_initial = peer_address in self.initial_peers
            result = await self.node_client.join_network_via_peer(
                peer_address, join_data, bypass_ban=is_initial,
            )
            if not result:
                self.logger.warning(f"Failed to join via {peer_address}")
                return False

            # Update responder's descriptor in telemetry if present.
            # Override self-reported public_host/public_port with the
            # address we actually reached them on.
            responder_descriptor = result.get("descriptor") if isinstance(result, dict) else None
            if responder_descriptor and peer_address != self.public_host:
                self.telemetry.update_node(
                    peer_address, "active",
                    descriptor=override_public_address(
                        responder_descriptor, peer_address,
                    ),
                )
                # Record the responder's version so sync filters can
                # see this peer before the first heartbeat arrives.
                runtime = responder_descriptor.get("runtime") or {}
                responder_version = runtime.get("quip_version")
                if responder_version:
                    self.peer_versions[peer_address] = responder_version

            # Add all nodes from the peer's node list. Transitive peers
            # arrive without a descriptor — we will learn it via their
            # own heartbeat/join later.
            peers_found = 0
            peers_map = result.get("peers", {}) or {}
            # Snapshot the whole peer map as "already announced" so our
            # own gossip_new_node origination guard suppresses a
            # startup burst covering peers the network already knows.
            now = time.time()
            async with self.gossip_lock:
                for peer_host in peers_map:
                    if peer_host == self.public_host:
                        continue
                    self._announced_nodes[peer_host] = now
            for peer_host, peer_info_json in peers_map.items():
                # except ourselves
                if peer_host == self.public_host:
                    continue
                info = MinerInfo.from_json(peer_info_json)
                if await self.add_peer(peer_host, info, descriptor=None):
                    peers_found += 1

            if peers_found > 0:
                self.logger.info(f"Successfully joined network via {peer_address}")
                self.logger.info(f"Discovered {peers_found} peers")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to connect to peer {peer_address}: {e}")
            self.telemetry.update_node(peer_address, "failed")
            return False

    def _request_peer_connections(self) -> None:
        """Submit peer connection requests to the worker process.

        Non-blocking: puts work on the queue and returns immediately.
        Guards against duplicate submissions while a batch is in flight.
        """
        if self._connection_request_pending:
            return

        if not self._connection_worker or not self._connection_worker.is_alive():
            self.logger.warning("Connection worker not alive, restarting")
            from shared.connection_worker import ConnectionWorkerHandle
            self._connection_worker = ConnectionWorkerHandle(
                node_timeout=self.node_timeout,
            )
            self._connection_worker.start()

        initial_set = set(self.initial_peers)
        peers: list[str] = []

        # Always include initial/seed peers (bypass bans)
        peers.extend(self.initial_peers)

        # Add known heartbeat peers that aren't banned
        for host in self.heartbeats:
            if host not in initial_set and not self._is_backed_off(host):
                peers.append(host)

        if not peers:
            return

        join_data = {
            "host": self.public_host,
            "version": get_version(),
            "capabilities": ["mining", "relay"],
            "info": self.info().to_json(),
            "descriptor": self.descriptor(),
        }
        self._connection_worker.request_connections(
            peers, join_data, initial_peers=initial_set,
        )
        self._connection_request_pending = True

    async def _drain_connection_results(self) -> bool:
        """Poll for connection results from the worker process.

        Returns True if at least one peer was successfully joined.
        """
        if not self._connection_worker:
            return False

        results = self._connection_worker.poll_results()
        if not results:
            return False

        any_success = False
        for result in results:
            peer = result.get("peer", "?")
            if result.get("success") and result.get("peers_map"):
                peers_map = result["peers_map"]
                responder_desc = result.get("responder_descriptor")
                if responder_desc and peer != self.public_host:
                    self.telemetry.update_node(
                        peer, "active",
                        descriptor=override_public_address(responder_desc, peer),
                    )
                    # Record responder's version so sync filters can
                    # see this peer before the first heartbeat arrives.
                    runtime = responder_desc.get("runtime") or {}
                    responder_version = runtime.get("quip_version")
                    if responder_version:
                        self.peer_versions[peer] = responder_version
                peers_found = 0
                # Snapshot the whole peer map as "already announced" so
                # our gossip_new_node origination guard suppresses a
                # startup burst covering peers the network already knows.
                now = time.time()
                async with self.gossip_lock:
                    for peer_host in peers_map:
                        if peer_host == self.public_host:
                            continue
                        self._announced_nodes[peer_host] = now
                for peer_host, peer_info_json in peers_map.items():
                    if peer_host == self.public_host:
                        continue
                    info = MinerInfo.from_json(peer_info_json)
                    # Transitive peers have no descriptor; we'll learn
                    # it from their own heartbeat/join later.
                    if await self.add_peer(peer_host, info, descriptor=None):
                        peers_found += 1
                if peers_found > 0:
                    self.logger.info(
                        f"Joined network via {peer}, discovered {peers_found} peers"
                    )
                any_success = True
            else:
                self.logger.debug(f"Connection worker: failed to join {peer}")

        # Allow new requests once we've drained the batch
        self._connection_request_pending = False
        return any_success

    def _is_backed_off(self, peer_address: str) -> bool:
        """Check if peer is currently banned."""
        return self._ban_list.is_banned(peer_address)

    @staticmethod
    def _format_ban_remaining(seconds: float) -> str:
        """Format remaining ban time for error messages."""
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            return f"{int(seconds / 60)}m"
        return f"{seconds / 3600:.1f}h"

    async def _backoff_peer(self, peer_address: str, reason: str):
        """Ban peer with exponential backoff and disconnect."""
        self._ban_list.record_failure(peer_address, reason)
        await self.remove_node(peer_address)

    async def add_peer(
        self,
        host: str,
        info: MinerInfo,
        descriptor: Optional[Dict[str, Any]] = None,
        connected: bool = False,
    ) -> bool:
        """Add a node to the active peer set or candidate pool.

        When the active set is full, new peers are routed to the
        candidate pool unless *connected* is True (meaning the peer
        has an active QUIC connection to us).
        """
        from shared.address_utils import parse_host_port, format_host_port
        try:
            h, p = parse_host_port(host)
            host = format_host_port(h, p)
        except ValueError:
            self.logger.warning(f"Invalid peer address, skipping: {host}")
            return False

        if host == self.public_host:
            return False

        if self._is_backed_off(host):
            return False

        async with self.net_lock:
            # Already in the active set — update in place.
            if host in self.peers:
                self.add_or_update_peer(host, info)
                if self.node_client:
                    self.node_client.add_peer(host, info)
                self.telemetry.update_node(
                    host, "active", info, descriptor=descriptor,
                )
                return False  # not new

            # Active set has room, or the peer has a live connection.
            if connected or len(self.peers) < self._max_active_peers:
                # Remove from candidates if it was queued there.
                self._candidate_peers.pop(host, None)

                self.add_or_update_peer(host, info)
                if self.node_client:
                    self.node_client.add_peer(host, info)
                self.telemetry.update_node(
                    host, "active", info, descriptor=descriptor,
                )
                self._has_ever_had_peers = True
                self.logger.info(
                    f"New peer discovered: {host}: "
                    f"{info.miner_id} "
                    f"({info.ecdsa_public_key.hex()[:8]})"
                )
                self._on_new_node(host, info)
                if hasattr(self, '_swim_detector'):
                    self._swim_detector.add_peer(host)
                asyncio.create_task(self.gossip_new_node(host, info))
                return True

            # Active set full — route to candidate pool.
            self._add_candidate(host, info, descriptor=descriptor)
            return False

    def _add_candidate(
        self,
        host: str,
        info: MinerInfo,
        descriptor: Optional[Dict[str, Any]] = None,
        source: str = "gossip",
    ) -> None:
        """Add a peer to the candidate pool (must hold net_lock)."""
        if host in self._candidate_peers:
            # Update info but keep discovery timestamp.
            self._candidate_peers[host].info = info
            if descriptor:
                self._candidate_peers[host].descriptor = descriptor
            return

        # Evict oldest candidate if at capacity.
        if len(self._candidate_peers) >= self._max_candidate_peers:
            oldest = min(
                self._candidate_peers,
                key=lambda h: self._candidate_peers[h].discovered_at,
            )
            del self._candidate_peers[oldest]

        self._candidate_peers[host] = CandidatePeer(
            info=info,
            discovered_at=time.monotonic(),
            source=source,
            descriptor=descriptor,
        )
        self.logger.debug(
            f"Peer {host} added to candidate pool "
            f"({len(self._candidate_peers)}/{self._max_candidate_peers})"
        )

    async def _promote_candidate(self, host: str) -> bool:
        """Promote a candidate to the active peer set.

        Must be called while NOT holding net_lock (acquires it
        internally via add_peer).  Returns True on success. On
        failure (active set filled between probe and promotion,
        ban list hit, address parse error) the candidate is
        re-inserted so the next probe cycle can retry.
        """
        candidate = self._candidate_peers.pop(host, None)
        if candidate is None:
            return False
        promoted = await self.add_peer(
            host, candidate.info,
            descriptor=candidate.descriptor,
            connected=False,
        )
        if not promoted and host not in self.peers:
            # add_peer rejected the promotion; keep the candidate
            # around rather than losing it silently.
            async with self.net_lock:
                if host not in self._candidate_peers:
                    self._candidate_peers[host] = candidate
            self.logger.debug(
                f"Promotion of {host} rejected; returning to candidate pool"
            )
        return promoted

    async def _try_promote_candidates(self) -> None:
        """Probe top candidates and promote reachable ones.

        Called from rebalance_loop (every 60s).  Probes at most 5
        candidates per cycle to avoid blocking.
        """
        async with self.net_lock:
            slots = self._max_active_peers - len(self.peers)
            if slots <= 0:
                return
            # Pick oldest candidates that haven't been probed recently.
            now = time.monotonic()
            probe_list = []
            for host, cand in sorted(
                self._candidate_peers.items(),
                key=lambda kv: kv[1].discovered_at,
            ):
                if now - cand.last_probe_at < 30:
                    continue  # probed recently, skip
                probe_list.append(host)
                if len(probe_list) >= min(slots, 5):
                    break

        if self.node_client is None:
            # Probing a candidate requires the QUIC client; without it
            # (early startup / test paths) the rebalance cycle is a
            # no-op rather than an AttributeError.
            return

        for host in probe_list:
            cand = self._candidate_peers.get(host)
            if cand is None:
                continue
            cand.last_probe_at = time.monotonic()
            cand.probe_attempts += 1

            ok = await self.node_client.send_heartbeat(
                host, self.public_host, self.info()
            )
            if ok:
                self.logger.info(f"Candidate {host} reachable, promoting")
                await self._promote_candidate(host)
            else:
                self.logger.debug(
                    f"Candidate probe to {host} failed "
                    f"(attempt {cand.probe_attempts}/3)"
                )
                if cand.probe_attempts >= 3:
                    self.logger.info(
                        f"Candidate {host} unreachable after "
                        f"{cand.probe_attempts} probes, removing"
                    )
                    self._candidate_peers.pop(host, None)

    def remove_peer(self, peer_address: str) -> bool:
        """Remove a peer, pruning NetworkNode-specific per-peer state.

        Overrides Node.remove_peer so every deletion path — whether the
        async remove_node flow or a direct base-class call — also clears
        the parallel per-host maps (peer_versions, heartbeats). Pops
        those entries regardless of whether the peer was in self.peers,
        so stale state can't accumulate when peers churn in and out of
        the ban list.
        """
        was_present = super().remove_peer(peer_address)
        self.peer_versions.pop(peer_address, None)
        self.heartbeats.pop(peer_address, None)
        self._candidate_peers.pop(peer_address, None)
        # Clear the announcement dedup entry so a legitimate rejoin can
        # re-announce immediately rather than waiting out the TTL. Dict
        # .pop is atomic under the GIL, so we don't acquire gossip_lock.
        self._announced_nodes.pop(peer_address, None)
        return was_present

    async def remove_node(self, host: str):
        """Remove a node from our registry."""
        async with self.net_lock:
            if self.remove_peer(host):
                self.logger.info(f"Node removed: {host}")
                self.telemetry.remove_node(host)
                self._on_node_lost(host)

                # Clear synchronized state when all peers are lost
                if not self.peers and self._has_ever_had_peers:
                    self._synchronized.clear()
                    self.logger.warning(
                        "All peers lost — cleared synchronized state, "
                        "mining paused until re-sync"
                    )

                # Update node client to remove peer
                if self.node_client:
                    await self.node_client.remove_peer(host)
                if hasattr(self, '_swim_detector'):
                    self._swim_detector.remove_peer(host)
                if hasattr(self, '_peer_loads'):
                    self._peer_loads.pop(host, None)
                if hasattr(self, '_peer_scorer'):
                    self._peer_scorer.remove_peer(host)
                if hasattr(self, '_block_inventory'):
                    self._block_inventory.remove_peer(host)

    async def send_heartbeat(self, node_host: str) -> bool:
        """Send heartbeat to a specific node.

        Heartbeats are sent regardless of sync state so that desynced
        nodes can maintain peer connections for re-synchronization.
        """
        if not self.node_client:
            return False
        return await self.node_client.send_heartbeat(node_host, self.public_host, self.info())

    async def get_peer_status(self, host: str) -> Optional[dict]:
        """Get status information from a peer node."""
        if not self.node_client:
            return None
        return await self.node_client.get_peer_status(host)

    async def get_peer_block(self, host: str, block_number: int = 0) -> Optional[Block]:
        """Get a block from a peer node and log precise download timings."""
        if not self.node_client:
            return None
        return await self.node_client.get_peer_block(host, block_number)

    async def get_peer_block_header(self, host: str, block_number: int = 0) -> Optional[BlockHeader]:
        """Get only the block header from a peer node (lighter and faster)."""
        if not self.node_client:
            return None
        return await self.node_client.get_peer_block_header(host, block_number)

    async def refresh_peer_info(self, host: str) -> bool:
        """Refresh peer information by calling their status endpoint."""
        peer_status = await self.get_peer_status(host)
        if peer_status and 'info' in peer_status:
            try:
                info = MinerInfo.from_json(peer_status['info'])
                async with self.net_lock:
                    self.peers[host] = info
                descriptor = override_public_address(
                    peer_status.get('descriptor'), host,
                )
                self.telemetry.update_node(
                    host, "active", info, descriptor=descriptor,
                )

                self.logger.debug(f"Refreshed info for peer {host}")
                return True
            except Exception:
                self.logger.exception(f"Error parsing peer info from {host}")
                return False
        return False

    async def refresh_all_peer_info(self):
        """Refresh information for all known peers by calling their status endpoints."""
        async with self.net_lock:
            peer_hosts = list(self.peers.keys())

        for host in peer_hosts:
            await self.refresh_peer_info(host)

    #####################
    ## Gossip Protocol ##
    #####################

    async def gossip_to(self, host: str, message: Message) -> bool:
        """Send a message to a specific node and log precise timings."""
        if not self.node_client:
            return False
        return await self.node_client.gossip_to(host, message)

    async def gossip(self, message: Message):
        """Gossip a new message and log timings for the broadcast."""
        if message.id:
            raise ValueError("Message already has an ID, cannot originate a gossip message!")

        t0 = time.perf_counter()
        hasher = blake3()
        hasher.update(message.type.encode('utf-8'))
        hasher.update(b'\x00')
        hasher.update(message.sender.encode('utf-8'))
        hasher.update(struct.pack('!d', float(message.timestamp)))
        hasher.update(message.data or b'')
        message.id = hasher.hexdigest()
        target_count = min(self.fanout * 2, len(self.peers))
        await self.gossip_broadcast(message, target_count)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0
        assert message.id
        self.logger.debug(
            f"🗣️ Originated gossip type={message.type} id={message.id[:8]} to {target_count} peers: payload={len(message.data or b'')} bytes in {total_ms:.1f} ms"
        )

    async def gossip_broadcast(self, message: Message, fanout: int = 3):
        """Rebroadcast a message to random subset of peers"""
        if not message.id:
            raise ValueError("Message must have an ID to gossip!")

        # FIXME: some sort of validation the message has not been changed since origination
        # e.g., a signature. We could also at least check the id against the hash maybe?
        # Anyone can initiate a broadcast so it does not help to do this right now.

        async with self.gossip_lock:
            if message.id in self.recent_messages:
                return  # Already processed

            self._record_recent_message(message.id)

        # Select random peers
        peers = list(self.peers.keys())
        if len(peers) <= fanout:
            targets = peers
        else:
            targets = random.sample(peers, fanout)

        # Send to selected peers
        tasks = [self.gossip_to(peer, message) for peer in targets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def gossip_new_node(self, new_node_address: str, new_node_info: MinerInfo):
        """Broadcast a new node to all known nodes.
        Data encoding: [u16 host_len][host utf-8][u32 info_len][info json utf-8]

        Skips origination if an announcement for this host has been
        observed within ``_announced_nodes_ttl``. Without this guard,
        every node that learns about a peer re-originates a fresh
        new_node gossip, producing O(N²) traffic per join because each
        re-origination has a distinct (sender, timestamp) and therefore
        a distinct message ID that the recent_messages dedup cannot
        suppress.
        """
        now = time.time()
        async with self.gossip_lock:
            last = self._announced_nodes.get(new_node_address)
            if last is not None and now - last < self._announced_nodes_ttl:
                return
            self._announced_nodes[new_node_address] = now

        _st = struct
        host_b = new_node_address.encode('utf-8')
        info_json = new_node_info.to_json().encode('utf-8')
        payload = _st.pack('!H', len(host_b)) + host_b + _st.pack('!I', len(info_json)) + info_json
        message = Message(
            type="new_node",
            sender=self.public_host,
            timestamp=utc_timestamp_float(),
            data=payload
        )
        await self.gossip(message)

    async def gossip_block(self, block_data: Block):
        """Broadcast a new block to the network.

        Uses IHAVE/IWANT protocol: sends block hash via IHAVE to peers,
        peers that need the block respond with IWANT. Falls back to
        full-block gossip for the initial propagation to ensure the
        block reaches the network quickly.
        """
        block_hash = block_data.hash
        binary_data = block_data.to_network()

        # Record in our inventory
        self._block_inventory.record_have(block_hash)

        # Send IHAVE to peers via scored selection
        async with self.net_lock:
            peer_list = list(self.peers.keys())

        effective_fanout = self._adaptive_fanout(len(peer_list))
        targets = self._peer_scorer.select_gossip_targets(
            peer_list, effective_fanout
        )

        ihave_sent = 0
        for peer in targets:
            try:
                protocol = await self.node_client._get_connection(peer)
                if protocol is None:
                    continue
                payload = json.dumps({
                    "block_hash": block_hash.hex(),
                    "block_index": block_data.header.index,
                    "sender": self.public_host,
                }).encode('utf-8')
                await protocol.send_request(
                    QuicMessageType.IHAVE, payload, timeout=3.0
                )
                ihave_sent += 1
            except (ConnectionError, asyncio.TimeoutError, OSError):
                pass  # Expected network errors
            except Exception as exc:
                self.logger.debug(f"IHAVE to {peer} failed: {exc}")

        self.logger.debug(
            f"Sent IHAVE for block {block_data.header.index} "
            f"to {ihave_sent}/{len(targets)} peers "
            f"(hash={block_hash.hex()[:16]}...)"
        )

    def _adaptive_fanout(self, peer_count: int) -> int:
        """Calculate adaptive fanout based on network size.

        Scales as max(3, min(configured_fanout, sqrt(peer_count))).
        """
        if peer_count <= 0:
            return self.fanout
        sqrt_peers = int(math.sqrt(peer_count))
        return max(3, min(self.fanout, sqrt_peers))

    async def handle_gossip(self, message: Message) -> str:
        """Main gossip logic to handle a gossip message from another node and rebroadcast."""
        # NOTE: we don't try/except here as it's caught in the handle_put_gossip call.

        # Track peer timestamp for time synchronization
        self._track_peer_timestamp(message.timestamp)

        async with self.gossip_lock:
            if message.id in self.recent_messages:
                return "ok"  # Already processed
            # NOTE: We only check and do not add to the dedup dict,
            #       as that happens during gossip_broadcast.

        if message.type == "new_node":
            _st = struct
            o = 0
            if len(message.data) < 2:
                return "rejected"
            host_len = _st.unpack('!H', message.data[o:o+2])[0]
            o += 2
            if len(message.data) < o + host_len + 4:
                return "rejected"
            host = message.data[o:o+host_len].decode('utf-8')
            o += host_len
            info_len = _st.unpack('!I', message.data[o:o+4])[0]
            o += 4
            if len(message.data) < o + info_len:
                return "rejected"
            info_json = message.data[o:o+info_len].decode('utf-8')
            new_info = MinerInfo.from_json(info_json)
            if host:
                # Record the announcement before add_peer so the
                # gossip_new_node origination guard sees it and does
                # not re-originate with us as sender.
                async with self.gossip_lock:
                    self._announced_nodes[host] = time.time()
                await self.add_peer(host, new_info)

        elif message.type == "block":
            if not message.data:
                return "rejected, missing block data"
            block = Block.from_network(message.data)
            # Don't rebroadcast if we are not synchronized yet
            if not self.synchronized:
                self.sync_block_cache[block.header.index] = (block, False)  # force_reorg=False for gossip
                return "ok"
            # Don't rebroadcast if we likely already saw it and its a little old
            if self.get_latest_block().header.index-2 >= block.header.index:
                return "ok"
            # Queue for background processing to avoid blocking
            # Create a dummy future for gossip blocks (we don't need the result)
            # force_reorg=False because these are normal network propagation, not sync
            dummy_future = asyncio.Future()
            self.block_processing_queue.put_nowait((block, dummy_future, False, message.sender))

        asyncio.create_task(self.gossip_broadcast(message, self.fanout))
        return "ok"

    ############################
    ## QUIC Message Handlers ##
    ############################

    def _record_recent_message(self, message_id: str) -> None:
        """Record a message ID in the bounded dedup dict.

        Evicts oldest entries when the dict exceeds max size, and
        periodically prunes entries older than TTL.
        """
        now = time.time()
        self.recent_messages[message_id] = now

        # Evict if over capacity
        if len(self.recent_messages) > self._recent_messages_max:
            self._prune_recent_messages(now)

    def _prune_recent_messages(self, now: Optional[float] = None) -> int:
        """Remove messages older than TTL. Returns count removed."""
        if now is None:
            now = time.time()
        cutoff = now - self._recent_messages_ttl
        stale = [
            mid for mid, ts in self.recent_messages.items()
            if ts < cutoff
        ]
        for mid in stale:
            del self.recent_messages[mid]

        # If still over capacity after TTL pruning, drop oldest
        if len(self.recent_messages) > self._recent_messages_max:
            excess = len(self.recent_messages) - self._recent_messages_max
            oldest = sorted(
                self.recent_messages.items(), key=lambda x: x[1]
            )[:excess]
            for mid, _ in oldest:
                del self.recent_messages[mid]
            return len(stale) + excess
        return len(stale)

    def _prune_announced_nodes(self, now: Optional[float] = None) -> int:
        """Drop announcement dedup entries older than TTL.

        No size cap — the dict is bounded by the number of distinct
        peers that have been announced in the last TTL window, which
        grows with the network, not with message volume.
        """
        if now is None:
            now = time.time()
        cutoff = now - self._announced_nodes_ttl
        stale = [
            host for host, ts in self._announced_nodes.items()
            if ts < cutoff
        ]
        for host in stale:
            del self._announced_nodes[host]
        return len(stale)

    async def _handle_quic_message(
        self,
        msg: QuicMessage,
        protocol: Any
    ) -> Optional[QuicMessage]:
        """Dispatch incoming QUIC messages to appropriate handlers.

        This replaces all the HTTP route handlers with a single dispatch point.
        """
        try:
            # Per-peer rate limiting
            peer_addr = getattr(protocol, '_peer_address', None) or 'unknown'
            if not self._rate_limiter.allow(peer_addr):
                self.logger.debug(f"Rate limited {peer_addr}")
                return msg.create_error_response("Rate limited")

            # Validate protocol version - reject incompatible nodes.
            # We only log here; we don't add a telemetry entry because
            # the transport-level peer address (ip:ephemeral-port) is
            # not the peer's public identity and would pollute
            # nodes.json with ghost rows that never reconcile.
            if msg.protocol_version != PROTOCOL_VERSION:
                peer_addr = str(protocol._peer_address)
                self.logger.debug(
                    "Protocol version mismatch from %s: expected %d, got %d (%s)",
                    peer_addr, PROTOCOL_VERSION, msg.protocol_version,
                    msg.msg_type.name,
                )
                # Schedule connection close after response is sent
                asyncio.get_running_loop().call_later(0.1, protocol.close)
                return msg.create_error_response(
                    f"Protocol version mismatch: expected {PROTOCOL_VERSION}, got {msg.protocol_version}",
                )

            if msg.msg_type == QuicMessageType.JOIN_REQUEST:
                return await self._quic_handle_join(msg, protocol)
            elif msg.msg_type == QuicMessageType.HEARTBEAT:
                return await self._quic_handle_heartbeat(msg)
            elif msg.msg_type == QuicMessageType.PEERS_REQUEST:
                return await self._quic_handle_peers(msg)
            elif msg.msg_type == QuicMessageType.GOSSIP:
                return await self._quic_handle_gossip(msg)
            elif msg.msg_type == QuicMessageType.BLOCK_SUBMIT:
                return await self._quic_handle_block_submit(msg, protocol)
            elif msg.msg_type == QuicMessageType.STATUS_REQUEST:
                return await self._quic_handle_status(msg)
            elif msg.msg_type == QuicMessageType.STATS_REQUEST:
                return await self._quic_handle_stats(msg)
            elif msg.msg_type == QuicMessageType.BLOCK_REQUEST:
                return await self._quic_handle_block_request(msg)
            elif msg.msg_type == QuicMessageType.BLOCK_HEADER_REQUEST:
                return await self._quic_handle_block_header_request(msg)
            elif msg.msg_type == QuicMessageType.SOLVE_REQUEST:
                return await self._quic_handle_solve(msg)
            # Phase 2: rebalancing + SWIM message handlers
            elif msg.msg_type == QuicMessageType.MIGRATE:
                return await self._quic_handle_migrate(msg)
            elif msg.msg_type == QuicMessageType.PROBE_REQUEST:
                return await self._quic_handle_probe_request(msg)
            elif msg.msg_type == QuicMessageType.LOAD_INFO:
                return await self._quic_handle_load_info(msg)
            # Phase 3: IHAVE/IWANT block propagation
            elif msg.msg_type == QuicMessageType.IHAVE:
                return await self._quic_handle_ihave(msg)
            elif msg.msg_type == QuicMessageType.IWANT:
                return await self._quic_handle_iwant(msg)
            # Fork-aware sync: manifest + content-addressed block fetch
            elif msg.msg_type == QuicMessageType.CHAIN_MANIFEST_REQUEST:
                return await self._quic_handle_chain_manifest_request(msg)
            elif msg.msg_type == QuicMessageType.BLOCK_BY_HASH_REQUEST:
                return await self._quic_handle_block_by_hash_request(msg)
            # Telemetry stream API
            elif msg.msg_type == QuicMessageType.TELEMETRY_STATUS_REQUEST:
                return self._quic_handle_telemetry_status(msg)
            elif msg.msg_type == QuicMessageType.TELEMETRY_NODES_REQUEST:
                return self._quic_handle_telemetry_nodes(msg)
            elif msg.msg_type == QuicMessageType.TELEMETRY_EPOCHS_REQUEST:
                return self._quic_handle_telemetry_epochs(msg)
            elif msg.msg_type == QuicMessageType.TELEMETRY_BLOCK_REQUEST:
                return self._quic_handle_telemetry_block(msg)
            elif msg.msg_type == QuicMessageType.TELEMETRY_LATEST_REQUEST:
                return self._quic_handle_telemetry_latest(msg)
            else:
                self.logger.warning(f"Unknown message type: {msg.msg_type}")
                return msg.create_error_response(f"Unknown message type: {msg.msg_type}")

        except Exception as e:
            self.logger.exception(f"Error handling {msg.msg_type.name}: {e}")
            return msg.create_error_response(str(e))

    async def _quic_handle_migrate(self, msg: QuicMessage) -> QuicMessage:
        """Handle MIGRATE: remote node is asking us to disconnect and reconnect elsewhere."""
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            reason = data.get("reason", "unknown")
            alternatives = data.get("alternatives", [])
            self.logger.info(
                f"Received MIGRATE (reason={reason}), "
                f"{len(alternatives)} alternatives suggested"
            )
            # Try to connect to one of the suggested alternatives in
            # the background so we don't block the MIGRATE response.
            if alternatives:
                asyncio.create_task(
                    self._migrate_to_alternatives(alternatives)
                )
            return QuicMessage(
                msg_type=QuicMessageType.MIGRATE_ACK,
                request_id=msg.request_id,
                payload=json.dumps({"status": "ok"}).encode('utf-8'),
                protocol_version=msg.protocol_version,
            )
        except Exception as exc:
            return msg.create_error_response(str(exc))

    async def _migrate_to_alternatives(
        self, alternatives: list[str]
    ) -> None:
        """Try to connect to MIGRATE-suggested peers (background).

        A MIGRATE is a recovery hint from a peer that's shedding us;
        failures here matter for operators trying to diagnose why the
        recovery attempt didn't work, so every branch logs its reason.
        """
        for alt_host in alternatives[:5]:
            if alt_host == self.public_host:
                continue
            if self._is_backed_off(alt_host):
                self.logger.debug(
                    f"MIGRATE alternative {alt_host} skipped: backed off"
                )
                continue
            async with self.net_lock:
                if alt_host in self.peers:
                    continue
            try:
                success = await self.connect_to_peer(alt_host)
                if success:
                    self.logger.info(
                        f"MIGRATE: reconnected to {alt_host}"
                    )
                    return
                self.logger.warning(
                    f"MIGRATE alternative {alt_host}: connect_to_peer "
                    "returned False"
                )
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                self.logger.warning(
                    f"MIGRATE alternative {alt_host} unreachable: "
                    f"{type(exc).__name__}: {exc}"
                )
            except Exception:
                # Unexpected exception types are programming bugs we
                # want to surface loudly rather than mask as "peer
                # unreachable".
                self.logger.exception(
                    f"MIGRATE alternative {alt_host}: unexpected error"
                )
        self.logger.warning(
            "MIGRATE: none of the suggested alternatives were reachable"
        )

    async def _quic_handle_probe_request(self, msg: QuicMessage) -> QuicMessage:
        """Handle SWIM PROBE_REQUEST: check if a target peer is alive on behalf of requester."""
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            target = data.get("target", "")
            probe_id = data.get("probe_id", "")

            if not target:
                return msg.create_error_response("Missing target in probe request")

            # Try to reach the target via heartbeat
            alive = await self.node_client.send_heartbeat(
                target, self.public_host, self.info()
            )
            self.logger.debug(
                f"SWIM probe {probe_id}: {target} "
                f"{'alive' if alive else 'unreachable'}"
            )
            return QuicMessage(
                msg_type=QuicMessageType.PROBE_RESPONSE,
                request_id=msg.request_id,
                payload=json.dumps({
                    "target": target,
                    "probe_id": probe_id,
                    "alive": alive,
                }).encode('utf-8'),
                protocol_version=msg.protocol_version,
            )
        except Exception as exc:
            return msg.create_error_response(str(exc))

    async def _quic_handle_load_info(self, msg: QuicMessage) -> QuicMessage:
        """Handle LOAD_INFO: store peer's load metrics."""
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            peer_load = NodeLoad.from_dict(data)
            # Identify sender from the load data or protocol
            # We store by any identifying info available
            # The load info doesn't contain the sender address, so
            # we use a placeholder — the real caller should include it
            # For now, just store it if there's a sender field
            sender = data.get("sender", "")
            if sender:
                self._peer_loads[sender] = peer_load
            return QuicMessage(
                msg_type=QuicMessageType.LOAD_INFO_RESPONSE,
                request_id=msg.request_id,
                payload=b'ok',
                protocol_version=msg.protocol_version,
            )
        except Exception as exc:
            return msg.create_error_response(str(exc))

    async def _quic_handle_ihave(self, msg: QuicMessage) -> QuicMessage:
        """Handle IHAVE: peer announces they have a block.

        If we don't have it, we respond with IWANT to request it.
        """
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            block_hash_hex = data.get("block_hash", "")
            block_index = data.get("block_index", 0)
            sender = data.get("sender", "")

            if not block_hash_hex:
                return msg.create_error_response("Missing block_hash")

            block_hash = bytes.fromhex(block_hash_hex)

            # Check if we need this block
            should_request = self._block_inventory.record_ihave(
                sender, block_hash
            )

            if should_request:
                self.logger.debug(
                    f"IHAVE: need block {block_index} from {sender}, "
                    f"sending IWANT"
                )
                self._block_inventory.record_want(block_hash, sender)

                # Send IWANT back to the sender
                asyncio.create_task(
                    self._send_iwant(sender, block_hash, block_index)
                )

            return QuicMessage(
                msg_type=QuicMessageType.IHAVE_RESPONSE,
                request_id=msg.request_id,
                payload=json.dumps({
                    "need": should_request,
                }).encode('utf-8'),
                protocol_version=msg.protocol_version,
            )
        except Exception as exc:
            return msg.create_error_response(str(exc))

    async def _send_iwant(
        self, peer: str, block_hash: bytes, block_index: int
    ) -> None:
        """Send IWANT request to a peer for a specific block."""
        try:
            protocol = await self.node_client._get_connection(peer)
            if protocol is None:
                return
            payload = json.dumps({
                "block_hash": block_hash.hex(),
                "block_index": block_index,
            }).encode('utf-8')
            response = await protocol.send_request(
                QuicMessageType.IWANT, payload, timeout=10.0
            )
            if response and response.msg_type == QuicMessageType.IWANT_RESPONSE:
                # Response payload is the full block
                try:
                    block = Block.from_network(response.payload)
                    self._block_inventory.record_block_received(block_hash)
                    self._peer_scorer.record_valid_block(peer)
                    self.logger.debug(
                        f"IWANT: received block {block.header.index} "
                        f"from {peer}"
                    )
                    # Queue for processing
                    dummy_future = asyncio.Future()
                    self.block_processing_queue.put_nowait(
                        (block, dummy_future, False, peer)
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"IWANT: invalid block from {peer}: {exc}"
                    )
                    self._peer_scorer.record_invalid_block(peer)
        except Exception as exc:
            self.logger.debug(f"IWANT to {peer} failed: {exc}")

    async def _quic_handle_iwant(self, msg: QuicMessage) -> QuicMessage:
        """Handle IWANT: peer is requesting a block we announced via IHAVE."""
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            block_hash_hex = data.get("block_hash", "")
            block_index = data.get("block_index", 0)

            if not block_hash_hex:
                return msg.create_error_response("Missing block_hash")

            block_hash = bytes.fromhex(block_hash_hex)

            # Look up the block in our chain
            block = self.get_block(block_index)
            if block is None or block.header.block_hash != block_hash:
                return msg.create_error_response(
                    f"Block {block_index} not found or hash mismatch"
                )

            # Return the full block
            return QuicMessage(
                msg_type=QuicMessageType.IWANT_RESPONSE,
                request_id=msg.request_id,
                payload=block.to_network(),
                protocol_version=msg.protocol_version,
            )
        except Exception as exc:
            return msg.create_error_response(str(exc))

    # ------------------------------------------------------------------
    # Telemetry QUIC handlers
    # ------------------------------------------------------------------

    def _quic_telemetry_json_response(
        self, msg: 'QuicMessage', data: Any,
    ) -> 'QuicMessage':
        """Build a JSON telemetry response from cached data."""
        payload = json.dumps(data).encode("utf-8")
        return msg.create_response(payload)

    def _quic_handle_telemetry_status(
        self, msg: 'QuicMessage',
    ) -> 'QuicMessage':
        """Handle TELEMETRY_STATUS_REQUEST."""
        if self.telemetry_cache is None:
            return msg.create_error_response("Telemetry not enabled")
        return self._quic_telemetry_json_response(
            msg, self.telemetry_cache.get_status(),
        )

    def _quic_handle_telemetry_nodes(
        self, msg: 'QuicMessage',
    ) -> 'QuicMessage':
        """Handle TELEMETRY_NODES_REQUEST."""
        if self.telemetry_cache is None:
            return msg.create_error_response("Telemetry not enabled")
        data = self.telemetry_cache.get_nodes()
        if data is None:
            return msg.create_error_response("No node data available")
        return self._quic_telemetry_json_response(msg, data)

    def _quic_handle_telemetry_epochs(
        self, msg: 'QuicMessage',
    ) -> 'QuicMessage':
        """Handle TELEMETRY_EPOCHS_REQUEST."""
        if self.telemetry_cache is None:
            return msg.create_error_response("Telemetry not enabled")
        return self._quic_telemetry_json_response(
            msg, {"epochs": self.telemetry_cache.get_epochs()},
        )

    def _quic_handle_telemetry_block(
        self, msg: 'QuicMessage',
    ) -> 'QuicMessage':
        """Handle TELEMETRY_BLOCK_REQUEST."""
        if self.telemetry_cache is None:
            return msg.create_error_response("Telemetry not enabled")
        try:
            params = json.loads(msg.payload.decode("utf-8"))
            epoch = params["epoch"]
            block_index = int(params["block_index"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return msg.create_error_response("Invalid request payload")
        data = self.telemetry_cache.get_block(epoch, block_index)
        if data is None:
            return msg.create_error_response(
                f"Block {epoch}/{block_index} not found",
            )
        return self._quic_telemetry_json_response(msg, data)

    def _quic_handle_telemetry_latest(
        self, msg: 'QuicMessage',
    ) -> 'QuicMessage':
        """Handle TELEMETRY_LATEST_REQUEST."""
        if self.telemetry_cache is None:
            return msg.create_error_response("Telemetry not enabled")
        data = self.telemetry_cache.get_latest()
        if data is None:
            return msg.create_error_response("No blocks available")
        return self._quic_telemetry_json_response(msg, data)

    async def _can_reach_address(self, address: str, timeout: float) -> bool:
        """Check if we can reach a UDP address (for QUIC connectivity).

        Uses a lightweight UDP probe rather than full QUIC handshake.
        Sends a small packet and checks for ICMP unreachable errors.
        """
        import socket
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)

            # Resolve hostname to IP
            loop = asyncio.get_event_loop()
            try:
                infos = await asyncio.wait_for(
                    loop.getaddrinfo(host, port, type=socket.SOCK_DGRAM),
                    timeout=timeout
                )
                if not infos:
                    self.logger.debug(f"Cannot resolve {host}")
                    return False
                family, _, _, _, sockaddr = infos[0]
            except Exception as e:
                self.logger.debug(f"DNS resolution failed for {host}: {e}")
                return False

            # Create UDP socket and "connect" (sets default destination)
            sock = socket.socket(family, socket.SOCK_DGRAM)
            sock.setblocking(False)
            try:
                sock.connect(sockaddr)
                # Send a small probe packet (will be ignored by QUIC server as invalid)
                sock.send(b'\x00')
                # Brief wait to allow ICMP unreachable to come back
                await asyncio.sleep(0.1)
                # Try to send again - if port is unreachable, this may raise
                sock.send(b'\x00')
                return True
            except (ConnectionRefusedError, OSError) as e:
                self.logger.debug(f"UDP probe failed for {address}: {e}")
                return False
            finally:
                sock.close()
        except Exception as e:
            self.logger.debug(f"Cannot reach {address}: {e}")
            return False

    @staticmethod
    def _normalize_ip(raw_ip: str) -> str:
        """Normalize an IP: strip brackets, map IPv6-mapped IPv4 to plain IPv4."""
        host = raw_ip.strip('[]')
        try:
            addr = ipaddress.ip_address(host)
            if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                return str(addr.ipv4_mapped)
            return str(addr)
        except ValueError:
            return host

    @staticmethod
    def _is_private_ip(host: str) -> bool:
        """Check if an IP string (no port) is private/loopback/link-local."""
        try:
            addr = ipaddress.ip_address(host)
            if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
                addr = addr.ipv4_mapped
            return addr.is_private or addr.is_loopback or addr.is_link_local
        except ValueError:
            return False

    def _extract_peer_ip_port(self, address: str) -> tuple[str, int]:
        """Parse host:port using address_utils, normalize IPv6-mapped IPs."""
        from shared.address_utils import parse_host_port, DEFAULT_PORT
        host, port = parse_host_port(address, DEFAULT_PORT)
        host = self._normalize_ip(host)
        return host, port

    def _is_self_address(self, address: str) -> bool:
        """True when ``address`` refers to this node's own public host:port.

        Compares by parsed ``(host, port)`` tuples so that
        ``127.0.0.1:20049`` matches whether it's written with or
        without IPv6 brackets, and hostnames match case-insensitively.
        Returns False for malformed inputs rather than raising — the
        filter is a best-effort tidy-up, not a validation step.
        """
        try:
            addr_host, addr_port = self._extract_peer_ip_port(address)
            self_host, self_port = self._extract_peer_ip_port(self.public_host)
        except (ValueError, TypeError):
            return False
        if addr_port != self_port:
            return False
        return addr_host.lower() == self_host.lower()

    def _filter_self_from_peers(self, peers: list[str]) -> list[str]:
        """Drop entries from ``peers`` that refer to this node itself.

        Leaving ``public_host`` in the peer list drives a node to
        attempt JOINing itself, which fails validation and pollutes
        the local ban list with its own loopback address.
        """
        filtered = [p for p in peers if not self._is_self_address(p)]
        dropped = len(peers) - len(filtered)
        if dropped > 0:
            self.logger.info(
                f"Filtered {dropped} self-reference(s) from peer list "
                f"(local public address: {self.public_host})"
            )
        return filtered

    async def _validate_peer_address(
        self, claimed: str, real_peer_addr: str, timeout: float = 2.0
    ) -> str:
        """Validate claimed address is reachable, fallback to real IP if not.

        If the claimed address uses a private/loopback IP, it is immediately
        replaced with the real connecting IP since private addresses are not
        routable from remote peers.  IPv6-mapped IPv4 addresses are normalized
        to plain IPv4.

        Args:
            claimed: Address the peer claims (e.g., "your-public-ip:20049")
            real_peer_addr: Actual source address from QUIC (e.g., "1.2.3.4:54321")
            timeout: Connection timeout in seconds

        Returns:
            Validated address to use for this peer

        Raises:
            ValueError: If neither claimed nor fallback address is reachable
        """
        from shared.address_utils import format_host_port

        claimed_host, claimed_port = self._extract_peer_ip_port(claimed)
        real_host, _ = self._extract_peer_ip_port(real_peer_addr)
        claimed_addr = format_host_port(claimed_host, claimed_port)

        # Private/loopback IPs are never reachable from remote peers —
        # replace immediately with the connecting IP.
        if self._is_private_ip(claimed_host):
            fallback = format_host_port(real_host, claimed_port)
            self.logger.info(
                f"Peer claimed private address {claimed_addr}, "
                f"using connecting IP: {fallback}"
            )
            if await self._can_reach_address(fallback, timeout):
                return fallback
            raise ValueError(
                f"Peer claimed private address {claimed_addr} and "
                f"connecting IP {fallback} is unreachable"
            )

        # Try claimed address first
        if await self._can_reach_address(claimed_addr, timeout):
            return claimed_addr

        # Fallback to real connecting IP with claimed port
        fallback = format_host_port(real_host, claimed_port)
        self.logger.warning(
            f"Claimed address {claimed_addr} unreachable, "
            f"falling back to {fallback}"
        )

        if await self._can_reach_address(fallback, timeout):
            return fallback

        raise ValueError(
            f"Cannot reach peer at claimed {claimed_addr} "
            f"or fallback {fallback}"
        )

    async def _quic_handle_join(self, msg: QuicMessage, protocol: Any) -> QuicMessage:
        """Handle join request from a new node."""
        # Check connection capacity before processing
        if not self._load_monitor.should_accept_join():
            # Suggest least-loaded alternative peers when at capacity
            async with self.net_lock:
                healthy = self._healthy_peers_snapshot()
                peer_keys = list(healthy.keys())
            alt_peers = self._get_least_loaded_peers(
                exclude=set(), count=10, peer_list=peer_keys
            )
            self.logger.info(
                f"Rejecting JOIN (overloaded or at capacity), "
                f"suggesting {len(alt_peers)} alternative peers"
            )
            peers_snapshot = {
                h: healthy[h].to_json()
                for h in alt_peers if h in healthy
            }
            response_data = json.dumps({
                "status": "at_capacity",
                "peers": peers_snapshot,
            })
            return msg.create_response(response_data.encode('utf-8'))

        data = json.loads(msg.payload.decode('utf-8'))
        claimed_address = data.get("host")
        info_field = data.get("info")
        new_node_info = MinerInfo.from_json(info_field) if info_field else None
        new_node_descriptor = data.get("descriptor")

        if not claimed_address or not new_node_info:
            return msg.create_error_response("Missing host or info")

        # Validate the claimed address, fallback to real source IP if unreachable
        real_peer_address = protocol._peer_address
        try:
            new_node_address = await self._validate_peer_address(claimed_address, real_peer_address)
        except ValueError as e:
            return msg.create_error_response(str(e))

        if self._is_backed_off(new_node_address):
            remaining = self._ban_list.time_remaining(new_node_address)
            return msg.create_error_response(
                f"backed off ({self._format_ban_remaining(remaining)} remaining)"
            )

        join_version = data.get("version")
        # A missing version field is treated as incompatible — peers
        # cannot bypass the MIN_COMPATIBLE_VERSION gate by omitting it.
        if not is_version_compatible(join_version):
            await self._backoff_peer(
                new_node_address,
                f"incompatible version {join_version!r} "
                f"(min: {MIN_COMPATIBLE_VERSION}, "
                f"local: {get_version()})",
            )
            return msg.create_error_response(
                f"Version {join_version!r} incompatible "
                f"(minimum: {MIN_COMPATIBLE_VERSION})"
            )
        self.peer_versions[new_node_address] = join_version
        try:
            peer_ver = version.parse(join_version)
            local_ver = version.parse(get_version())
        except version.InvalidVersion:
            peer_ver = local_ver = None
        if peer_ver is not None and local_ver is not None and peer_ver > local_ver:
            self.logger.info(
                f"Peer {new_node_address} runs newer version "
                f"{join_version} (local: {get_version()})"
            )

        # Add the new node. Override the sender's self-reported
        # public_host/public_port with the address we actually validated.
        await self.add_peer(
            new_node_address, new_node_info,
            descriptor=override_public_address(
                new_node_descriptor, new_node_address,
            ),
            connected=True,
        )

        # Return only peers with a recent successful heartbeat so we
        # don't propagate stale/unreachable addresses to new joiners.
        async with self.net_lock:
            peers_snapshot = self._healthy_peers_snapshot()

        peers_payload: Dict[str, str] = {}
        for host, info in peers_snapshot.items():
            peers_payload[host] = info.to_json()
        peers_payload[self.public_host] = self.info().to_json()

        response_data = json.dumps({
            "status": "ok",
            "peers": peers_payload,
            "descriptor": self.descriptor(),
        })
        return msg.create_response(response_data.encode('utf-8'))

    async def _quic_handle_heartbeat(self, msg: QuicMessage) -> QuicMessage:
        """Handle heartbeat from another node."""
        data = json.loads(msg.payload.decode('utf-8'))
        sender = data.get("sender")
        net_version = data.get("version")
        timestamp = data.get("timestamp", utc_timestamp_float())

        if sender and self._is_backed_off(sender):
            return msg.create_response(
                json.dumps({"status": "backed_off"}).encode('utf-8')
            )

        # Heartbeats without a parseable version are rejected so a peer
        # can't bypass the version gate by omitting the field.
        if not is_version_compatible(net_version):
            if sender:
                await self._backoff_peer(
                    sender,
                    f"incompatible version {net_version!r} "
                    f"(min: {MIN_COMPATIBLE_VERSION}, "
                    f"local: {get_version()})",
                )
            return msg.create_response(
                json.dumps(
                    {"status": "incompatible_version"}
                ).encode('utf-8')
            )
        try:
            peer_ver = version.parse(net_version)
            local_ver = version.parse(get_version())
        except version.InvalidVersion:
            peer_ver = local_ver = None
        if peer_ver is not None and local_ver is not None and peer_ver > local_ver:
            self.logger.info(
                f"Peer {sender} runs newer version "
                f"{net_version} (local: {get_version()}). "
                f"Consider updating."
            )

        if sender and net_version:
            self.peer_versions[sender] = net_version

        if sender:
            async with self.net_lock:
                if sender in self.peers:
                    self.heartbeats[sender] = utc_timestamp_float()
                    self._track_peer_timestamp(timestamp)
                    self.telemetry.update_node(sender, "active", last_heartbeat=timestamp)
                else:
                    self.logger.info(f"New node discovered via heartbeat: {sender}")
                    asyncio.create_task(self.refresh_peer_info(sender))

        return msg.create_response(json.dumps({"status": "ok"}).encode('utf-8'))

    async def _quic_handle_peers(self, msg: QuicMessage) -> QuicMessage:
        """Return list of known healthy nodes."""
        async with self.net_lock:
            healthy = self._healthy_peers_snapshot()
            peers_data = {
                host: info.to_json() for host, info in healthy.items()
            }
        return msg.create_response(
            json.dumps({"peers": peers_data}).encode('utf-8')
        )

    async def _quic_handle_gossip(self, msg: QuicMessage) -> QuicMessage:
        """Handle a gossip message from another node.

        ACK-on-receipt semantics: respond as soon as the message is queued
        for background processing. The client (``gossip_to``) only checks
        that it got a GOSSIP_RESPONSE, so returning before ``handle_gossip``
        completes is safe. Backpressure is still visible via the
        ``server overloaded`` error when the queue is full.
        """
        gossip_message = Message.from_network(msg.payload)
        t_enq = time.perf_counter()

        try:
            self.gossip_processing_queue.put_nowait(
                (gossip_message, None, t_enq)
            )
        except asyncio.QueueFull:
            return msg.create_error_response("server overloaded")
        return msg.create_response(b'{"status":"accepted"}')

    async def _quic_handle_block_submit(self, msg: QuicMessage, protocol: Any) -> QuicMessage:
        """Handle new block submission (DEBUG purposes)."""
        data = json.loads(msg.payload.decode('utf-8'))

        block_bytes = bytes.fromhex(data['raw'])
        signature = bytes.fromhex(data['signature'])
        net_data = block_bytes + signature
        block = Block.from_network(net_data)

        # Get peer address for logging
        peer_address = getattr(protocol, '_peer_address', None)

        response_future: asyncio.Future[bool] = asyncio.Future()
        try:
            # force_reorg=False for submitted blocks (normal propagation rules)
            self.block_processing_queue.put_nowait((block, response_future, False, peer_address))
            result = await asyncio.wait_for(response_future, timeout=10.0)
            status = "ok" if result else "rejected"
            return msg.create_response(json.dumps({"status": status}).encode('utf-8'))
        except asyncio.QueueFull:
            return msg.create_error_response("server overloaded")
        except asyncio.TimeoutError:
            return msg.create_error_response("processing timeout")

    async def _quic_handle_status(self, msg: QuicMessage) -> QuicMessage:
        """Return node status."""
        status_data = {
            "host": self.public_host,
            "info": self.info().to_json(),
            "descriptor": self.descriptor(),
            "running": self.running,
            "total_peers": len(self.peers),
            "uptime": utc_timestamp_float() if self.running else 0
        }
        return msg.create_response(json.dumps(status_data).encode('utf-8'))

    async def _quic_handle_stats(self, msg: QuicMessage) -> QuicMessage:
        """Return node statistics from cache."""
        async with self._stats_cache_lock:
            if self._stats_cache is None:
                return msg.create_error_response("Stats cache not initialized")
            return msg.create_response(json.dumps(self._stats_cache).encode('utf-8'))

    async def _quic_handle_block_request(self, msg: QuicMessage) -> QuicMessage:
        """Return a specific block by number (binary format)."""
        # Payload is 4-byte big-endian block number
        if len(msg.payload) >= 4:
            block_number = struct.unpack('!I', msg.payload[:4])[0]
        else:
            block_number = 0  # Latest block

        if block_number == 0:
            block = self.get_latest_block()
        else:
            block = self.get_block(block_number)

        if block is None:
            return msg.create_error_response(f"Block {block_number} not found")

        # Return block in network binary format
        return msg.create_response(block.to_network())

    async def _quic_handle_block_header_request(self, msg: QuicMessage) -> QuicMessage:
        """Return a specific block header by number (binary format)."""
        # Payload is 4-byte big-endian block number
        if len(msg.payload) >= 4:
            block_number = struct.unpack('!I', msg.payload[:4])[0]
        else:
            block_number = 0  # Latest block

        if block_number == 0:
            block = self.get_latest_block()
        else:
            block = self.get_block(block_number)

        if block is None:
            return msg.create_error_response(f"Block {block_number} not found")

        # Return header in network binary format
        return msg.create_response(block.header.to_network())

    async def _quic_handle_chain_manifest_request(self, msg: QuicMessage) -> QuicMessage:
        """Return a slice of our canonical chain as ``(index, hash)`` tuples.

        The client sends a Bitcoin-style locator. We find the latest
        hash in that locator that is still on our canonical chain
        (O(1) per entry via ``chain_by_hash``), then return canonical
        entries starting from the next index, up to ``limit`` or our
        current tip — whichever is smaller.

        Returns an empty manifest when no locator hash matches our
        canonical chain (divergent genesis, or the client is a
        reorg'd-away fork we don't share). The client treats that as
        "no useful data here" and demotes the peer for the session.
        """
        try:
            locator, limit = decode_manifest_request(msg.payload)
        except ValueError as e:
            return msg.create_error_response(f"malformed manifest request: {e}")

        start_after = -1
        for h in locator:
            block = self.chain_by_hash.get(h)
            if block is not None:
                start_after = block.header.index
                break

        entries: list = []
        if start_after >= 0:
            first_idx = start_after + 1
            tip_idx = self.get_latest_block().header.index
            capped_limit = min(limit, MAX_MANIFEST_ENTRIES)
            last_idx = min(tip_idx, first_idx + capped_limit - 1)
            for i in range(first_idx, last_idx + 1):
                blk = self.chain[i]
                if blk.hash is None:
                    # Canonical-chain blocks are expected to be finalized;
                    # bail out rather than sending a partial manifest.
                    break
                entries.append((i, blk.hash))

        return msg.create_response(encode_manifest_response(entries))

    async def _quic_handle_block_by_hash_request(self, msg: QuicMessage) -> QuicMessage:
        """Return a canonical-chain block by hash, or empty for NOT_FOUND."""
        try:
            block_hash = decode_block_by_hash_request(msg.payload)
        except ValueError as e:
            return msg.create_error_response(
                f"malformed block-by-hash request: {e}"
            )

        block = self.get_block_by_hash(block_hash)
        return msg.create_response(encode_block_by_hash_response(block))

    async def _quic_handle_solve(self, msg: QuicMessage) -> QuicMessage:
        """Handle quantum annealing solve request."""
        data = json.loads(msg.payload.decode('utf-8'))

        # Validate request
        if 'h' not in data or 'J' not in data or 'num_samples' not in data:
            return msg.create_error_response("Missing required fields: h, J, num_samples")

        h = data['h']
        J_raw = data['J']
        num_samples = int(data['num_samples'])

        # Convert J to list of tuples format
        if isinstance(J_raw, dict):
            J = [((int(k.split(',')[0].strip('()')), int(k.split(',')[1].strip('()'))), v)
                 for k, v in J_raw.items()]
        elif isinstance(J_raw, list):
            J = [((entry[0], entry[1]), entry[2]) for entry in J_raw]
        else:
            return msg.create_error_response("Invalid J format. Must be dict or list.")

        # Generate transaction ID
        transaction_id = f"{self.public_host}-{time.time()}-{hash((tuple(h), tuple(J)))}"

        # Convert h and J to format needed by sampler
        h_dict = {i: val for i, val in enumerate(h)}
        J_dict = {(i, j): val for ((i, j), val) in J}

        # Use first available miner to solve
        if not hasattr(self, 'miner_handles') or not self.miner_handles:
            return msg.create_error_response("No miners available")

        miner_handle = self.miner_handles[0]
        self.logger.info(
            f"Solving BQM with {len(h)} variables, {len(J)} couplings, "
            f"{num_samples} samples using {miner_handle.miner_id}"
        )

        # Create appropriate sampler based on miner type
        sampler = None
        miner_kind = miner_handle.spec.get("kind", "").lower()

        qpu_sampler = None
        if miner_kind == "qpu":
            from dwave.system import DWaveSampler
            try:
                qpu_sampler = DWaveSampler()
                sampler = qpu_sampler
                self.logger.info(f"Using QPU sampler: {sampler.properties.get('chip_id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Failed to create QPU sampler: {e}")
                return msg.create_error_response(f"QPU not available: {e}")
        elif miner_kind in ["cpu", "metal", "cuda", "modal"]:
            from dwave.samplers import SimulatedAnnealingSampler
            sampler = SimulatedAnnealingSampler()
            self.logger.info("Using simulated annealing sampler")
        else:
            return msg.create_error_response(f"Unknown miner type: {miner_kind}")

        try:
            # Sample the Ising problem
            sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=num_samples)
        finally:
            if qpu_sampler is not None:
                qpu_sampler.close()

        # Extract samples and energies
        samples = []
        energies = []
        for sample, energy in sampleset.data(['sample', 'energy']):
            sample_list = [int(sample[i]) for i in sorted(sample.keys())]
            samples.append(sample_list)
            energies.append(float(energy))

        self.logger.info(
            f"Solve completed: {len(samples)} samples with energies "
            f"ranging from {min(energies):.2f} to {max(energies):.2f}"
        )

        # Create transaction record
        from shared.block import Transaction
        transaction = Transaction(
            transaction_id=transaction_id,
            timestamp=network_timestamp(),
            request_h=h,
            request_J=J,
            num_samples=num_samples,
            samples=samples[:num_samples],
            energies=energies[:num_samples]
        )

        # Add to pending transactions
        async with self.transactions_lock:
            self.pending_transactions.append(transaction)

        self.logger.info(f"Transaction {transaction_id} added to pending pool")

        response_data = {
            "samples": samples[:num_samples],
            "energies": energies[:num_samples],
            "transaction_id": transaction_id,
            "status": "completed"
        }
        return msg.create_response(json.dumps(response_data).encode('utf-8'))
