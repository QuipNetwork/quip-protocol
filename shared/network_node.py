import asyncio
import copy
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
from shared.version import get_version
from shared.node_client import (
    NodeClient, QuicMessage, QuicMessageType,
    generate_self_signed_cert, QUIP_ALPN_PROTOCOL, MAX_DATAGRAM_FRAME_SIZE,
    MAX_DATAGRAM_MESSAGE_SIZE,
)
from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import QuicEvent, DatagramFrameReceived, StreamDataReceived, ConnectionTerminated, HandshakeCompleted
from shared.block_synchronizer import BlockSynchronizer
from shared.block_store import BlockStore
from shared.time_utils import (
    utc_timestamp_float, utc_timestamp, get_network_time_offset,
    is_clock_synchronized, sync_time_with_network, NETWORK_TIME_SYNC_INTERVAL
)


@dataclass(frozen=True)
class EpochInfo:
    """Information about a previous chain epoch to prevent block acceptance from old epochs."""
    first_hash: bytes      # Hash of block 1 from this epoch
    last_timestamp: int    # Timestamp of the last block before reset
    last_index: int        # Index of the last block before reset
    last_hash: bytes       # Hash of the last block before reset

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

    # List of reliable IP detection services
    services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ipecho.net/plain",
        "https://checkip.amazonaws.com",
        "https://ident.me"
    ]

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
            # Basic validation - check if it looks like an IP
            if ip and '.' in ip and len(ip.split('.')) == 4:
                logger.info(f"Detected public IP: {ip}")
                return ip
        except Exception as e:
            logger.debug(f"Failed to get IP from {service}: {e}")
            continue

    logger.warning("Unable to determine public IP address")
    return None


def get_local_ip() -> str:
    """
    Get the local IP address (best guess).

    Returns:
        Local IP address as string
    """
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
        self.bind_address = config.get("listen", "127.0.0.1")
        self.port = config.get("port", 20049)

        self.node_name = config.get("node_name", socket.getfqdn())
        self.public_host = config.get("public_host", f"{get_local_ip()}:{self.port}")
        # Default to local_ip only when we are listening on a local host.
        # Note: We can't call asyncio.run() here since we're inside an async context.
        # Public IP will be determined asynchronously during node startup if needed.

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

        # TOFU (Trust On First Use) configuration
        self.tofu_config = config.get("tofu", {})
        self.tofu_enabled = self.tofu_config.get("enabled", True)
        self.trust_store = None  # Initialized in start()

        # REST API configuration
        self.rest_api_config = config.get("rest_api", {})
        self.rest_api_enabled = self.rest_api_config.get("enabled", False)
        self.rest_api_server = None  # Initialized in start()

        # Initialize logger with helper function
        self.logger = init_component_logger('network_node', self.node_name)

        # Durations as float seconds
        self.heartbeat_interval = float(config.get("heartbeat_interval", 15))
        self.heartbeat_timeout = float(config.get("heartbeat_timeout", 300))
        self.node_timeout = float(config.get("node_timeout", 10))

        self.initial_peers = config.get("peer", ["nodes.quip.network:20049"])
        self.fanout = int(config.get("fanout", 3))

        self.net_lock = asyncio.Lock()
        self.running = False
        self.heartbeats = {}

        self.gossip_lock = asyncio.Lock()
        self.recent_messages = set()

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
        
        # Maximum block index to synchronize with (prevents syncing with peers too far ahead)
        self.max_sync_block_index = 1024

        self.miners_config = {
            "cpu": config.get("cpu", None),
            "gpu": config.get("gpu", None),
            "qpu": config.get("qpu", None)
        }
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
                    node.logger.info(
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

        # Initialize TOFU trust store if enabled
        if self.tofu_enabled:
            import os
            from shared.trust_store import TrustStore
            trust_db_path = os.path.expanduser(
                self.tofu_config.get("trust_db", "~/.quip/trust.db")
            )
            self.trust_store = TrustStore(trust_db_path, logger=self.logger)
            await self.trust_store.initialize()
            self.logger.info(f"TOFU trust store initialized at {trust_db_path}")

        # Initialize QUIC client for peer communication (with TOFU support)
        self.node_client = NodeClient(
            node_timeout=self.node_timeout,
            logger=self.logger,
            trust_store=self.trust_store
        )
        await self.node_client.start()

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

        self.logger.info(
            f"Network node {self.node_name} ({self.crypto.ecdsa_public_key_hex[:8]}) "
            f"started at quic://{self.bind_address}:{self.port} with public address {self.public_host}"
        )

        # Start REST API server if enabled
        if self.rest_api_enabled:
            from shared.rest_api import RestApiServer
            from shared.certificate_manager import CertificateManager

            cert_manager = CertificateManager(self.rest_api_config, logger=self.logger)
            self.rest_api_server = RestApiServer(
                network_node=self,
                host=self.rest_api_config.get("host", "0.0.0.0"),
                port=self.rest_api_config.get("http_port", 8080),
                tls_port=self.rest_api_config.get("https_port", 443),
                cert_manager=cert_manager,
                logger=self.logger
            )
            await self.rest_api_server.start()

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self.node_cleanup_loop())
        self.block_processor_task = asyncio.create_task(self.block_processor_loop())
        self.gossip_processor_task = asyncio.create_task(self.gossip_processor_loop())
        self.server_task = asyncio.create_task(self.server_loop())

        # have we fully synchronized with the network at least one time?
        self._synchronized = threading.Event()
        self.sync_block_cache = {}  # Regular dict is thread-safe for simple assignments in CPython
        
        # Initialize stats cache
        asyncio.create_task(self._update_stats_cache())

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
        if self.reset_timer_task:
            self.reset_timer_task.cancel()

        # Stop REST API server if running
        if self.rest_api_server:
            self.logger.info("Stopping REST API server...")
            await self.rest_api_server.stop()

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
        sys.exit(0)

    ##########################
    ## Server logic threads ##
    ##########################

    async def heartbeat_loop(self):
        """Send heartbeats to all known nodes."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all nodes
                tasks = []
                async with self.net_lock:
                    for node_host in list(self.peers.keys()):
                        task = asyncio.create_task(self.send_heartbeat(node_host))
                        tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in heartbeat loop")

    async def node_cleanup_loop(self):
        """Remove dead nodes from registry."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_timeout / 2)

                # Find dead nodes
                current_time = utc_timestamp_float()
                dead_nodes = []

                async with self.net_lock:
                    for host, node_info in list(self.peers.items()):
                        if host not in self.heartbeats or current_time - self.heartbeats[host] > self.heartbeat_timeout:
                            dead_nodes.append(host)

                # Remove dead nodes
                for host in dead_nodes:
                    await self.remove_node(host)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in cleanup loop")

    async def server_loop(self):
        """Main server loop."""
        while self.running:
            try:
                # Check if we are connected to any active peers, if not, try to reconnect to known peers in our
                # heartbeats list or, if empty, the initial peers list.
                connected = await self.is_connected()
                if not connected:
                    connected = await self.connect_to_network()

                # If we are not connected and not in auto-mine mode, sleep and retry
                if not connected and not self.auto_mine:
                    self.logger.error(f"Not connected to network, retrying in {self.node_timeout} seconds...")
                    await asyncio.sleep(self.node_timeout)
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
                    await self.synchronize_blockchain(latest_block)
                    # NOTE: It's possible we can get triggered again if the sync takes too long, but that's OK
                    # as we will be closer to the goal.
                    continue

                # If we are synchronized, check if we are mining. If not, start mining on the next block.
                if not self._is_mining:
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
                block, response_future = block_data
                try:
                    latest = self.get_latest_block()
                    # Cache out of order blocks for later processing
                    # NOTE: older blocks need processing to determine chain fork
                    if latest.header.index+1 < block.header.index:
                        self.sync_block_cache[block.header.index] = block
                        # WE return failure, but thats only to signal we didn't process it.
                        response_future.set_result(False)
                        continue
                    # Base case we can process the block
                    result = await self.receive_block(block)
                    response_future.set_result(result)
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
        """Background loop to process gossip messages without blocking HTTP handlers."""
        while self.running:
            try:
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
                    wait_str = f"{wait_ms:.1f} ms" if wait_ms is not None else "n/a"
                    self.logger.debug(
                        f"🧩 Gossip handled id={(message.id or '')[:8]} type={message.type}: wait={wait_str}, process={proc_ms:.1f} ms, qsize={qsize}"
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
        """Handle exceptions from mining tasks - crash on ValueError."""
        if task.done() and not task.cancelled():
            try:
                task.result()  # This will raise the exception if one occurred
            except ValueError as e:
                # Shutdown miner workers and stop the event loop to crash the program
                self.logger.error(f"ValueError in mining task - shutting down: {e}")
                self.close()  # Shutdown miner workers first
                loop = asyncio.get_event_loop()
                loop.stop()
                sys.exit(-1)
            except Exception as e:
                # Log other exceptions but don't crash
                self.logger.error(f"Exception in mining task: {e}")

    async def _exhaust_block_cache(self):
        """Exhaust the block cache by processing all blocks in order."""
        # Pause gossip and process the current block cache. 
        async with self.gossip_lock:
            # Process cached blocks starting from end_index + 1
            next_block_index = self.get_latest_block().header.index + 1
            while next_block_index in self.sync_block_cache:
                cached_block = self.sync_block_cache.pop(next_block_index)
                self.logger.info(f"Processing cached block {next_block_index} received during sync")
                
                # Process the cached block
                success = await self.receive_block(cached_block)
                if not success:
                    self.logger.error(f"Failed to process cached block {next_block_index}")
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
                
                # Reset to original genesis block (no more new genesis creation)
                self.chain = [self.genesis_block]
                self.logger.info("Chain reset to genesis block completed")
            
            # Reset synchronization state if it exists (only after start() is called)
            if hasattr(self, '_synchronized'):
                self._synchronized.clear()
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
            
            # Add network-specific information
            fresh_stats.update({
                "network": {
                    "host": self.public_host,
                    "running": self.running,
                    "total_peers": len(self.peers),
                    "synchronized": self.synchronized,
                    "auto_mine": self.auto_mine,
                    "heartbeat_interval": self.heartbeat_interval,
                    "heartbeat_timeout": self.heartbeat_timeout,
                    "queue_sizes": {
                        "block_processing": self.block_processing_queue.qsize(),
                        "gossip_processing": self.gossip_processing_queue.qsize(),
                    }
                }
            })
            
            # Atomically update the cache
            async with self._stats_cache_lock:
                self._stats_cache = fresh_stats
                
            self.logger.debug("Stats cache updated successfully")
            
        except Exception as e:
            self.logger.exception(f"Error updating stats cache: {e}")

    async def receive_block(self, block: Block) -> bool:
        """Receive a block from the network with epoch validation and max block limit."""
        # Reject blocks that are too far in the future (beyond max sync limit)
        if block.header.index > self.max_sync_block_index:
            self.logger.debug(f"Block {block.header.index} rejected: index > max_sync_block_index {self.max_sync_block_index}")
            return False
        
        # Check against previous epoch to prevent accepting blocks from old chain epoch
        async with self.chain_lock:
            if self.previous_epoch:
                # Reject if this is block 1 from the previous epoch
                if block.header.index == 1 and block.hash and block.hash == self.previous_epoch.first_hash:
                    self.logger.warning(f"Block 1 rejected: hash {block.hash.hex()[:8]}... matches previous epoch first_hash")
                    return False
                
                # Reject if this block hash matches the last block from the previous epoch  
                if block.hash and block.hash == self.previous_epoch.last_hash:
                    self.logger.warning(f"Block {block.header.index} rejected: hash {block.hash.hex()[:8]}... matches previous epoch last_hash")
                    return False
                
                # Reject if timestamp is older than the last timestamp from the previous epoch
                if block.header.timestamp <= self.previous_epoch.last_timestamp:
                    self.logger.warning(f"Block {block.header.index} rejected: timestamp {block.header.timestamp} <= previous epoch last_timestamp {self.previous_epoch.last_timestamp}")
                    return False
        
        # If all validations pass, call parent receive_block
        return await super().receive_block(block)

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
        wb = self.sign_block(wb)

        if not wb.hash:
            raise ValueError("Failed to finalize block")

        self.logger.info(f"Candidate Block {wb.header.index}-{wb.hash.hex()[:8]} mined on this node!")
    
        if wb.header.index == self.get_latest_block().header.index + 1:
            await self.receive_block(wb)
            self.logger.info(f"Accepted block {wb.header.index}-{wb.hash.hex()[:8]} from {wb.miner_info.miner_id}")
            asyncio.create_task(self.gossip_block(wb))
        else:
            self.logger.info(f"Candidate Block {wb.header.index}-{wb.hash.hex()[:8]} sniped by another miner!")

        return result


    async def check_synchronized(self) -> int:
        """Check if we are synchronized with the network using header-only fetch.

        Returns 0 if synchronized or the latest network block index if not.
        """
        my_latest_block = self.get_latest_block()
        if not self.peers:
            if self.auto_mine:
                self.logger.debug("No connected peers, but auto-mine is enabled so we are synchronized by default")
                return 0
            else:
                raise RuntimeError("No peers to synchronize with")

        net_latest: Optional[BlockHeader] = None
        tries = 0
        peers = list(self.peers.keys())
        while net_latest is None:
            if not peers:
                self.logger.warning("No valid peers to synchronize with (all peers may have >max_sync_block_index blocks)")
                return 0
                
            random_peer = random.choice(peers)
            peers.remove(random_peer)
            header = await self.get_peer_block_header(random_peer)
            if header:
                # Ignore peers with blocks beyond our max sync limit
                if header.index > self.max_sync_block_index:
                    self.logger.debug(f"Ignoring peer {random_peer} with block index {header.index} > max_sync_block_index {self.max_sync_block_index}")
                    continue
                net_latest = header
                break
            tries += 1
            if tries > 3:
                self.logger.warning("Unable to get latest block header from peers, assuming we are synchronized")
                return 0
            await asyncio.sleep(1)

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
                self.logger.debug("Latest block prev_hash mismatch, may need to synchronize")
                return 0

        return net_latest.index

    async def  synchronize_blockchain(self, current_head: int = 0):
        """Synchronize the blockchain with the network using BlockSynchronizer."""
        if self._is_mining:
            raise RuntimeError("Cannot synchronize while mining")

        if current_head == 0:
            current_head = await self.check_synchronized()
        if current_head == 0:
            return

        my_latest_block = self.get_latest_block()
        
        # Enforce maximum sync block limit
        if current_head > self.max_sync_block_index:
            self.logger.warning(f"Refusing to synchronize beyond max_sync_block_index {self.max_sync_block_index}, requested: {current_head}")
            return

        # Always go back at least 2 blocks.
        start_index = max(1, my_latest_block.header.index-1)
        end_index = current_head
        if start_index > end_index:
            return

        self.logger.info(f"Syncing with network from block {start_index} to {end_index}...")

        # Use BlockSynchronizer for concurrent block downloads and sequential processing
        if not self.node_client:
            self.logger.error("NodeClient not initialized")
            return
            
        # Update node client with current peers
        self.node_client.update_peers(self.peers)
        
        # Create block synchronizer
        synchronizer = BlockSynchronizer(
            node_client=self.node_client,
            receive_block_queue=self.block_processing_queue,
            logger=self.logger
        )
        
        # Synchronize blocks using multiprocessing
        success = await synchronizer.sync_blocks(start_index, end_index)
        if not success:
            raise RuntimeError(f"Failed to synchronize blocks {start_index} to {end_index}")
        else:
            self.logger.info(f"Successfully synchronized blocks {start_index} to {end_index}")
            

    def _track_peer_timestamp(self, timestamp: float):
        """Track a peer timestamp for time synchronization."""
        current_time = utc_timestamp_float()

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

        if not is_clock_synchronized(self.peer_timestamps):
            offset = get_network_time_offset(self.peer_timestamps)
            self.time_sync_warnings += 1

            if self.time_sync_warnings <= 3:  # Limit warnings
                self.logger.warning(
                    f"⚠️  Clock synchronization issue detected! "
                    f"Local time is {offset} seconds {'ahead' if offset > 0 else 'behind'} network time. "
                    f"Consider synchronizing your system clock with NTP."
                )
            elif self.time_sync_warnings == 4:
                self.logger.warning("⚠️  Clock sync warnings suppressed (fix your system clock)")
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

        try:
            join_data = {
                "host": self.public_host,
                "version": get_version(),
                "capabilities": ["mining", "relay"],
                # Serialize MinerInfo as JSON string for transport
                "info": self.info().to_json()
            }

            # Use NodeClient's SSL-aware connection method
            result = await self.node_client.join_network_via_peer(peer_address, join_data)
            if not result:
                self.logger.warning(f"Failed to join via {peer_address}")
                return False

            # Add all nodes from the peer's node list
            peers_found = 0
            peers_map = result.get("peers", {}) or {}
            for peer_host, peer_info_json in peers_map.items():
                # except ourselves
                if peer_host == self.public_host:
                    continue
                info = MinerInfo.from_json(peer_info_json)
                if await self.add_peer(peer_host, info):
                    peers_found += 1

            if peers_found > 0:
                self.logger.info(f"Successfully joined network via {peer_address}")
                self.logger.info(f"Discovered {peers_found} peers")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to connect to peer {peer_address}: {e}")
            return False

    async def add_peer(self, host: str, info: MinerInfo) -> bool:
        """Add a node to our registry."""
        if host == self.public_host:
            return False

        async with self.net_lock:
            is_new = self.add_or_update_peer(host, info)

            # Always update node client with peer info (new or updated)
            if self.node_client:
                self.node_client.add_peer(host, info)

            if is_new:
                self.logger.info(f"New peer discovered: {host}: {info.miner_id} ({info.ecdsa_public_key.hex()[:8]})")
                self._on_new_node(host, info)

                # Broadcast new node to all other nodes
                asyncio.create_task(self.gossip_new_node(host, info))

            return is_new

    async def remove_node(self, host: str):
        """Remove a node from our registry."""
        async with self.net_lock:
            if host in self.peers:
                del self.peers[host]
                self.logger.info(f"Node removed: {host}")
                self._on_node_lost(host)

                # Update node client to remove peer
                if self.node_client:
                    await self.node_client.remove_peer(host)

    async def send_heartbeat(self, node_host: str) -> bool:
        """Send heartbeat to a specific node."""
        if not self.node_client or not self.synchronized:
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

            self.recent_messages.add(message.id)

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
        """
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
        Data encoding: raw block network bytes.
        """
        binary_data = block_data.to_network()
        bytes_sent = len(binary_data)

        message = Message(
            type="block",
            sender=self.public_host,
            timestamp=utc_timestamp_float(),
            data=binary_data
        )

        # Log byte count for gossiped block
        self.logger.debug(f"📤 Gossiped block {block_data.header.index}: {bytes_sent} bytes")

        await self.gossip(message)

    async def handle_gossip(self, message: Message) -> str:
        """Main gossip logic to handle a gossip message from another node and rebroadcast."""
        # NOTE: we don't try/except here as it's caught in the handle_put_gossip call.

        # Track peer timestamp for time synchronization
        self._track_peer_timestamp(message.timestamp)

        async with self.gossip_lock:
            if message.id in self.recent_messages:
                return "ok"  # Already processed
            # NOTE: We only check and do not add to the processing list,
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
                await self.add_peer(host, new_info)

        elif message.type == "block":
            if not message.data:
                return "rejected, missing block data"
            block = Block.from_network(message.data)
            # Don't rebroadcast if we are not synchronized yet
            if not self.synchronized:
                self.sync_block_cache[block.header.index] = block
                return "ok"
            # Don't rebroadcast if we likely already saw it and its a little old
            if self.get_latest_block().header.index-2 >= block.header.index:
                return "ok"
            # Queue for background processing to avoid blocking
            # Create a dummy future for gossip blocks (we don't need the result)
            dummy_future = asyncio.Future()
            self.block_processing_queue.put_nowait((block, dummy_future))

        asyncio.create_task(self.gossip_broadcast(message, self.fanout))
        return "ok"

    ############################
    ## QUIC Message Handlers ##
    ############################

    async def _handle_quic_message(
        self,
        msg: QuicMessage,
        protocol: Any
    ) -> Optional[QuicMessage]:
        """Dispatch incoming QUIC messages to appropriate handlers.

        This replaces all the HTTP route handlers with a single dispatch point.
        """
        try:
            if msg.msg_type == QuicMessageType.JOIN_REQUEST:
                return await self._quic_handle_join(msg)
            elif msg.msg_type == QuicMessageType.HEARTBEAT:
                return await self._quic_handle_heartbeat(msg)
            elif msg.msg_type == QuicMessageType.PEERS_REQUEST:
                return await self._quic_handle_peers(msg)
            elif msg.msg_type == QuicMessageType.GOSSIP:
                return await self._quic_handle_gossip(msg)
            elif msg.msg_type == QuicMessageType.BLOCK_SUBMIT:
                return await self._quic_handle_block_submit(msg)
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
            else:
                self.logger.warning(f"Unknown message type: {msg.msg_type}")
                return msg.create_error_response(f"Unknown message type: {msg.msg_type}")

        except Exception as e:
            self.logger.exception(f"Error handling {msg.msg_type.name}: {e}")
            return msg.create_error_response(str(e))

    async def _quic_handle_join(self, msg: QuicMessage) -> QuicMessage:
        """Handle join request from a new node."""
        data = json.loads(msg.payload.decode('utf-8'))
        new_node_address = data.get("host")
        info_field = data.get("info")
        new_node_info = MinerInfo.from_json(info_field) if info_field else None

        if not new_node_address or not new_node_info:
            return msg.create_error_response("Missing host or info")

        # Add the new node
        await self.add_peer(new_node_address, new_node_info)

        # Return our node list
        async with self.net_lock:
            peers_snapshot = copy.deepcopy(self.peers)

        peers_payload: Dict[str, str] = {}
        for host, info in peers_snapshot.items():
            peers_payload[host] = info.to_json()
        peers_payload[self.public_host] = self.info().to_json()

        response_data = json.dumps({"status": "ok", "peers": peers_payload})
        return msg.create_response(response_data.encode('utf-8'))

    async def _quic_handle_heartbeat(self, msg: QuicMessage) -> QuicMessage:
        """Handle heartbeat from another node."""
        data = json.loads(msg.payload.decode('utf-8'))
        sender = data.get("sender")
        net_version = data.get("version")
        timestamp = data.get("timestamp", utc_timestamp_float())

        if net_version:
            local_version = get_version()
            local_ver = version.parse(local_version)
            peer_ver = version.parse(net_version)

            if local_ver < peer_ver:
                self.logger.error(
                    f"Local version {local_version} is outdated compared to "
                    f"peer {sender} running version {net_version}"
                )
                self.logger.error("Please run 'pip install quip-network' to get the latest version")
                await self.stop()
            elif local_ver > peer_ver:
                self.logger.warning(
                    f"Peer {sender} is running older version {net_version} (local: {local_version})"
                )

        if sender:
            async with self.net_lock:
                if sender in self.peers:
                    self.heartbeats[sender] = utc_timestamp_float()
                    self._track_peer_timestamp(timestamp)
                else:
                    self.logger.info(f"New node discovered via heartbeat: {sender}")
                    asyncio.create_task(self.refresh_peer_info(sender))

        return msg.create_response(json.dumps({"status": "ok"}).encode('utf-8'))

    async def _quic_handle_peers(self, msg: QuicMessage) -> QuicMessage:
        """Return list of known nodes."""
        async with self.net_lock:
            peers_data = {host: info.to_json() for host, info in self.peers.items()}

        return msg.create_response(json.dumps({"peers": peers_data}).encode('utf-8'))

    async def _quic_handle_gossip(self, msg: QuicMessage) -> QuicMessage:
        """Handle a gossip message from another node."""
        # Parse the gossip Message from payload
        gossip_message = Message.from_network(msg.payload)

        # Queue for background processing
        response_future: asyncio.Future[str] = asyncio.Future()
        t_enq = time.perf_counter()

        try:
            self.gossip_processing_queue.put_nowait((gossip_message, response_future, t_enq))
            status = await asyncio.wait_for(response_future, timeout=5.0)
            return msg.create_response(json.dumps({"status": status}).encode('utf-8'))
        except asyncio.QueueFull:
            return msg.create_error_response("server overloaded")
        except asyncio.TimeoutError:
            return msg.create_error_response("processing timeout")

    async def _quic_handle_block_submit(self, msg: QuicMessage) -> QuicMessage:
        """Handle new block submission (DEBUG purposes)."""
        data = json.loads(msg.payload.decode('utf-8'))

        block_bytes = bytes.fromhex(data['raw'])
        signature = bytes.fromhex(data['signature'])
        net_data = block_bytes + signature
        block = Block.from_network(net_data)

        response_future: asyncio.Future[bool] = asyncio.Future()
        try:
            self.block_processing_queue.put_nowait((block, response_future))
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

        if miner_kind == "qpu":
            from dwave.system import DWaveSampler
            try:
                sampler = DWaveSampler()
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

        # Sample the Ising problem
        sampleset = sampler.sample_ising(h_dict, J_dict, num_reads=num_samples)

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
            timestamp=utc_timestamp(),
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
