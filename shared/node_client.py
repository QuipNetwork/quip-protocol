import asyncio
import random
import ssl
import time
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import aiohttp

from shared.block import Block, BlockHeader, MinerInfo

if TYPE_CHECKING:
    from shared.network_node import Message


class NodeClient:
    """HTTP client for peer communication with connection pooling and timeout handling."""
    
    def __init__(self, node_timeout: float = 10.0, logger: Optional[logging.Logger] = None, verify_ssl: bool = True):
        self.node_timeout = node_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.verify_ssl = verify_ssl
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.peers: Dict[str, MinerInfo] = {}
        # Connection string mapping: host -> full connection string (https://host:port or http://host:port)
        self.conn_str: Dict[str, str] = {}
        
    async def start(self):
        """Initialize HTTP session with connection pooling."""
        if self.http_session:
            return
        
        # Create SSL context
        ssl_context = None
        if self.verify_ssl:
            ssl_context = ssl.create_default_context()
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=10,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            ssl=ssl_context,
        )
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.node_timeout)
        )
        
    async def stop(self):
        """Close HTTP session and cleanup resources."""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
            
    def update_peers(self, peers: Dict[str, MinerInfo]):
        """Update peer list from NetworkNode."""
        self.peers = peers.copy()
        
    def add_peer(self, host: str, info: MinerInfo):
        """Add a peer to the client."""
        self.peers[host] = info
        
    def remove_peer(self, host: str):
        """Remove a peer from the client."""
        if host in self.peers:
            del self.peers[host]
        # Also remove connection string if it exists
        if host in self.conn_str:
            del self.conn_str[host]
    
    async def _establish_connection_string(self, peer_address: str) -> Optional[str]:
        """
        Establish connection string for a peer, trying HTTPS first, then HTTP.
        
        Args:
            peer_address: Host:port string
            
        Returns:
            Full connection string (https://host:port or http://host:port) or None if both fail
        """
        if not self.http_session:
            self.logger.error("HTTP session not initialized")
            return None
        
        # Try HTTPS first using /status endpoint (lightweight connection test)
        https_url = f"https://{peer_address}"
        try:
            async with self.http_session.get(
                f"{https_url}/status",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    self.logger.info(f"🔒 Established secure TLS connection to {peer_address}")
                    return https_url
        except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
            self.logger.debug(f"HTTPS connection to {peer_address} failed: {e}")
        
        # Fall back to HTTP
        http_url = f"http://{peer_address}"
        try:
            async with self.http_session.get(
                f"{http_url}/status",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    self.logger.warning(f"⚠️  Falling back to PLAINTEXT HTTP connection to {peer_address} (TLS unavailable)")
                    return http_url
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.logger.debug(f"HTTP connection to {peer_address} failed: {e}")
        
        self.logger.error(f"❌ Both HTTPS and HTTP connections failed to {peer_address}")
        return None
    
    async def _ensure_connection_string(self, host: str) -> Optional[str]:
        """
        Ensure we have a connection string for a host, establishing one if needed.
        
        Args:
            host: Host:port string
            
        Returns:
            Connection string or None if connection fails
        """
        if host in self.conn_str:
            return self.conn_str[host]
        
        # Try to establish connection string
        conn_str = await self._establish_connection_string(host)
        if conn_str:
            self.conn_str[host] = conn_str
        return conn_str
            
    async def connect_to_peer(self, peer_address: str) -> bool:
        """Connect to a peer and verify connection using TLS-first approach."""
        # Check if we already have a connection string for this peer
        if peer_address in self.conn_str:
            try:
                assert self.http_session
                # Test existing connection using /status endpoint
                async with self.http_session.get(
                    f"{self.conn_str[peer_address]}/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
            except Exception as e:
                self.logger.debug(f"Existing connection to {peer_address} failed: {e}")
                # Remove stale connection string and try to re-establish
                del self.conn_str[peer_address]
        
        # Establish new connection string
        conn_str = await self._establish_connection_string(peer_address)
        if conn_str:
            self.conn_str[peer_address] = conn_str
            return True
        
        return False
            
    async def send_heartbeat(self, node_host: str, public_host: str, miner_info: MinerInfo) -> bool:
        """Send heartbeat to a specific node."""
        if not self.http_session:
            return False
        
        # Ensure we have a connection string for this host
        conn_str = await self._ensure_connection_string(node_host)
        if not conn_str:
            return False
            
        try:
            # Import here to avoid circular imports
            from .version import get_version
            
            data = {"sender": public_host, "version": get_version()}
            async with self.http_session.post(
                f"{conn_str}/heartbeat",
                json=data,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception as e:
            self.logger.debug(f"Heartbeat to {node_host} failed: {e}")
            return False
            
    async def get_peer_status(self, host: str) -> Optional[dict]:
        """Get status from a peer node."""
        if not self.http_session:
            return None
        
        # Ensure we have a connection string for this host
        conn_str = await self._ensure_connection_string(host)
        if not conn_str:
            return None
            
        try:
            async with self.http_session.get(f"{conn_str}/status") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            self.logger.debug(f"Failed to get status from {host}: {e}")
            return None
            
    async def get_peer_block(self, host: str, block_number: int = 0) -> Optional[Block]:
        """Get a block from a peer node and log precise download timings."""
        if not self.http_session:
            return None
        
        # Ensure we have a connection string for this host
        conn_str = await self._ensure_connection_string(host)
        if not conn_str:
            return None
            
        import time  # Local import to avoid issues
        t0 = time.perf_counter()
        
        try:
            req = "/block/"
            if block_number > 0:
                req = f"/block/{block_number}"
            url = f"{conn_str}{req}?format=network"
            
            async with self.http_session.get(url) as resp:
                t_headers = time.perf_counter()  # time to response headers
                if resp.status == 200:
                    data = await resp.read()
                    t_done = time.perf_counter()
                    bytes_received = len(data)
                    block = Block.from_network(data)
                    block_index = getattr(getattr(block, 'header', None), 'index', block_number)
                    headers_ms = (t_headers - t0) * 1000.0
                    body_ms = (t_done - t_headers) * 1000.0
                    total_ms = (t_done - t0) * 1000.0
                    self.logger.debug(f"📥 Downloaded block {block_index} from {host}: {bytes_received} bytes in {total_ms:.1f} ms (headers {headers_ms:.1f} ms, body {body_ms:.1f} ms) url={url}")
                    return block
                else:
                    self.logger.debug(f"Failed to get block from {host}: HTTP {resp.status} url={url}")
                    return None
        except Exception as e:
            self.logger.debug(f"Failed to get block {block_number} from {host}: {e}")
            return None
            
    async def get_peer_block_header(self, host: str, block_number: int = 0) -> Optional[BlockHeader]:
        """Get only the block header from a peer node (lighter and faster)."""
        if not self.http_session:
            return None
        
        # Ensure we have a connection string for this host
        conn_str = await self._ensure_connection_string(host)
        if not conn_str:
            return None
            
        import time  # Local import to avoid issues
        t0 = time.perf_counter()
        
        try:
            req = "/block_header/"
            if block_number > 0:
                req = f"/block_header/{block_number}"
            url = f"{conn_str}{req}?format=network"
            
            async with self.http_session.get(url) as resp:
                t_headers = time.perf_counter()
                if resp.status == 200:
                    data = await resp.read()
                    t_done = time.perf_counter()
                    bytes_received = len(data)
                    header = BlockHeader.from_network(data)
                    headers_ms = (t_headers - t0) * 1000.0
                    body_ms = (t_done - t_headers) * 1000.0
                    total_ms = (t_done - t0) * 1000.0
                    self.logger.debug(f"📥 Downloaded block header {header.index} from {host}: {bytes_received} bytes in {total_ms:.1f} ms (headers {headers_ms:.1f} ms, body {body_ms:.1f} ms)")
                    return header
                else:
                    self.logger.debug(f"Failed to get block header from {host}: HTTP {resp.status}")
                    return None
        except Exception as e:
            self.logger.debug(f"Failed to get block header {block_number} from {host}: {e}")
            return None
            
    async def gossip_broadcast(self, message: 'Message', sender_host: str, fanout: int = 3):
        """Broadcast message to multiple peers using gossip protocol."""
        if not self.http_session or not self.peers:
            return
            
        # Select random subset of peers for gossip
        available_peers = [host for host in self.peers.keys() if host != sender_host]
        if not available_peers:
            return
            
        selected_peers = random.sample(
            available_peers, 
            min(fanout, len(available_peers))
        )
        
        # Broadcast to selected peers concurrently
        tasks = []
        for peer_host in selected_peers:
            task = self._send_gossip_message(peer_host, message)
            tasks.append(task)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def gossip_to(self, host: str, message: 'Message') -> bool:
        """Send a message to a specific node and log precise timings."""
        if not self.http_session:
            return False
        
        # Ensure we have a connection string for this host
        conn_str = await self._ensure_connection_string(host)
        if not conn_str:
            return False
            
        import time  # Local import to avoid issues
        t0 = time.perf_counter()
        url = f"{conn_str}/gossip"
        
        try:
            payload = message.to_network()
            bytes_sent = len(payload)
            headers = {'Content-Type': 'application/octet-stream'}
            
            async with self.http_session.post(url, data=payload, headers=headers) as resp:
                t_headers = time.perf_counter()
                # Read small JSON body to measure full roundtrip
                try:
                    await resp.read()
                except Exception:
                    pass
                t_done = time.perf_counter()
                headers_ms = (t_headers - t0) * 1000.0
                body_ms = (t_done - t_headers) * 1000.0
                total_ms = (t_done - t0) * 1000.0
                ok = resp.status == 200
                if ok:
                    self.logger.debug(
                        f"📨 Gossip to {host} type={message.type} id={(message.id or '')[:8]}: {bytes_sent} bytes in {total_ms:.1f} ms (headers {headers_ms:.1f} ms, body {body_ms:.1f} ms) url={url}"
                    )
                else:
                    self.logger.debug(f"Failed gossip to {host}: HTTP {resp.status} url={url}")
                return ok
        except Exception as e:
            t_err = time.perf_counter()
            self.logger.debug(f"Error gossiping to {host} after {(t_err - t0)*1000.0:.1f} ms: {e}")
            return False
            
    async def _send_gossip_message(self, host: str, message: 'Message'):
        """Send a single gossip message to a peer (legacy wrapper)."""
        return await self.gossip_to(host, message)