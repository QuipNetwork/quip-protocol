import asyncio
import json
import math
import random
import socket
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Callable, Tuple
from datetime import datetime
import aiohttp
from aiohttp import web
import logging

import copy

from blake3 import blake3

from shared.base_miner import MiningResult
from shared.block import Block, MinerInfo
from shared.node import Node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_public_ip() -> Optional[str]:
    """
    Get the public IP address by querying external services.

    Returns:
        Public IP address as string, or None if unable to determine
    """
    # List of reliable IP detection services
    services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ipecho.net/plain",
        "https://checkip.amazonaws.com",
        "https://ident.me"
    ]

    for service in services:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(service, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        ip = (await response.text()).strip()
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
    data: dict
    id: Optional[str] = None


class NetworkNode(Node):
    """Peer-to-peer node for quantum blockchain network."""
    
    def __init__(self, config: dict, genesis_block: Block):
        self.bind_address = config.get("listen", "127.0.0.1")
        self.port = config.get("port", 20049)

        self.node_name = config.get("node_name", socket.getfqdn())
        self.public_host = config.get("public_host", f"{get_local_ip()}:{self.port}")
        # Default to local_ip only when we are listening on a local host.
        if self.bind_address != "127.0.0.1":
            self.public_host = config.get("public_host", f"{get_public_ip()}:{self.port}")

        self.secret = config.get("secret", f"quip network node secret {random.randint(0, 1000000)}")
        self.auto_mine = config.get("auto_mine", False)

        # Durations as float seconds
        self.heartbeat_interval = float(config.get("heartbeat_interval", 15))
        self.heartbeat_timeout = float(config.get("heartbeat_timeout", 300))
        self.node_timeout = float(config.get("node_timeout", 3))

        self.initial_peers = config.get("peer", ["nodes.quip.network:20049"])
        self.fanout = int(config.get("fanout", 3))

        self.net_lock = asyncio.Lock()
        self.running = False
        self.heartbeats = {}

        self.gossip_lock = asyncio.Lock()
        self.recent_messages = set()

        # Callbacks
        self.on_new_node: Optional[Callable] = None
        self.on_node_lost: Optional[Callable] = None
        self.on_block_received: Optional[Callable] = None

        # Registered callback handlers from Node
        self.on_block_mined = self._on_block_received
        self.on_mining_started = self._network_on_mining_started
        self.on_mining_stopped = self._network_on_mining_stopped

        # Web server
        self.app = web.Application()
        self.setup_routes()
        self.runner = None

        # Background tasks
        self.heartbeat_task = None
        self.cleanup_task = None
        self.running = False

        self.miners_config = {
            "cpu": config.get("cpu", None),
            "gpu": config.get("gpu", None),
            "qpu": config.get("qpu", None)
        }
        super().__init__(self.node_name, self.miners_config, genesis_block, secret=self.secret)

    def setup_routes(self):
        """Setup HTTP routes for P2P communication."""
        self.app.router.add_post('/join', self.handle_put_join)
        self.app.router.add_post('/heartbeat', self.handle_put_heartbeat)
        self.app.router.add_post('/peers', self.handle_get_peers)
        self.app.router.add_post('/gossip', self.handle_put_gossip)
        self.app.router.add_post('/block', self.handle_put_block)
        self.app.router.add_get('/status', self.handle_get_status)
        self.app.router.add_get('/block/', self.handle_get_latest_block)
        self.app.router.add_get('/block/{number}', self.handle_get_block)
    
    async def start(self):
        """Start the P2P node."""
        self.running = True
        
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.bind_address, self.port)
        await site.start()
        
        logger.info(f"Network node {self.node_name} ({self.crypto.ecdsa_public_key_hex[:8]}) started at {self.bind_address}:{self.port} with public address {self.public_host}")
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self.node_cleanup_loop())
        self.server_task = asyncio.create_task(self.server_loop())
    
    async def stop(self):
        """Stop the P2P node."""
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Stop web server
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Network node stopped")

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
                logger.exception("Error in heartbeat loop")

    async def node_cleanup_loop(self):
        """Remove dead nodes from registry."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_timeout / 2)

                # Find dead nodes
                current_time = time.time()
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
                logger.exception("Error in cleanup loop")

    async def server_loop(self):
        """Main server loop."""
        while self.running:
            # Check if we are connected to any active peers, if not, try to reconnect to known peers in our 
            # heartbeats list or, if empty, the initial peers list. 
            connected = await self.is_connected()
            if not connected:
                connected = await self.connect_to_network()

            # If we are not connected and not in auto-mine mode, sleep and retry
            if not connected and not self.auto_mine:
                logger.error(f"Not connected to network, retrying in {self.node_timeout} seconds...")
                await asyncio.sleep(self.node_timeout)
                continue
            elif not connected and self.auto_mine:
                logger.info("No peers connected, automining...")

            # Check if we are in synchronized state with peers
            # If not, stop mining and synchronize. 
            synchronized = await self.check_synchronized()
            if not synchronized:
                if self._is_mining:
                    await self.stop_mining()
                    logger.info("Stopped mining to synchronize with network...")
                await self.synchronize_blockchain()
                # NOTE: It's possible we can get triggered again if the sync takes too long, but that's OK 
                # as we will be closer to the goal.
                continue

            # If we are synchronized, check if we are mining. If not, start mining on the next block.
            if not self._is_mining:
                latest_block = self.get_latest_block()
                # Create task with exception handler to crash on ValueError
                task = asyncio.create_task(self.mine_block(latest_block))
                task.add_done_callback(self._handle_mining_task_exception)

            await asyncio.sleep(5)

    def _handle_mining_task_exception(self, task: asyncio.Task):
        """Handle exceptions from mining tasks - crash on ValueError."""
        if task.done() and not task.cancelled():
            try:
                task.result()  # This will raise the exception if one occurred
            except ValueError as e:
                # Shutdown miner workers and stop the event loop to crash the program
                logger.error(f"ValueError in mining task - shutting down: {e}")
                self.close()  # Shutdown miner workers first
                loop = asyncio.get_event_loop()
                loop.stop()
                sys.exit(-1)
            except Exception as e:
                # Log other exceptions but don't crash
                logger.error(f"Exception in mining task: {e}")


    #############################
    ## Internal Event Handlers ##
    #############################

    async def _on_new_node(self, host, info: MinerInfo):
        logger.info(f"🌟 New node joined: {host} {info.miner_id} ({info.ecdsa_public_key.hex()[:8]})")
        if self.on_new_node:                
            asyncio.create_task(self.on_new_node(host, info))

    async def _on_node_lost(self, host):
        logger.info(f"💔 Node lost: {host}")
        if self.on_node_lost:
            asyncio.create_task(self.on_node_lost(host))
    
    async def _on_block_received(self, block: Block):
        logger.info(f"📦 New block mined: {block.header.index}")
        if self.on_block_received:
            asyncio.create_task(self.on_block_received(block))

    async def _network_on_mining_started(self, prev: Block):
        logger.info(f"⛏️ Mining started on block {prev.header.index + 1} with previous block hash {prev.hash}")

    async def _network_on_mining_stopped(self):
        logger.info(f"🛑 Mining stopped")


    #######################
    ## Control functions ##
    #######################

    async def mine_block(self, previous_block: Block) -> Optional[MiningResult]:
        """Mine a block and broadcast if successful."""
        result = await super().mine_block(previous_block)
        if not result:
            return None

        ## TODO: Pull current mempool and put it in here.
        data = f"{self.node_id} was here"
        wb = self.build_block(previous_block, result, data.encode())
        wb = self.sign_block(wb)

        if not wb.hash:
            raise ValueError("Failed to finalize block")

        logger.info(f"Block {wb.header.index}-{wb.hash.hex()[:8]} mined on this node!")
        accepted = await self.receive_block(wb)
        if not accepted:
            logger.warning(f"Block {wb.header.index}-{wb.hash.hex()[:8]} rejected by network!")
            return None
        
        asyncio.create_task(self.gossip_block(wb))
        return result


    async def check_synchronized(self) -> int: 
        """Check if we are synchronized with the network.
        
        Returns 0 if synchronized or the latest network block index if not.
        """
        my_latest_block = self.get_latest_block()
        if not self.peers:
            if self.auto_mine:
                logger.debug("No connected peers, but auto-mine is enabled so we are synchronized by default")
                return True
            else:
                raise RuntimeError("No peers to synchronize with")

        net_latest_block = None
        while not net_latest_block:
            random_peer = random.choice(list(self.peers.keys()))
            net_latest_block = await self.get_peer_block(random_peer)

        if my_latest_block.header.index >= net_latest_block.header.index:
            return 0
        return net_latest_block.header.index

    async def synchronize_blockchain(self, current_head: int = 0):
        """Synchronize the blockchain with the network."""
        if self._is_mining:
            raise RuntimeError("Cannot synchronize while mining")
        
        if current_head == 0:
            current_head = await self.check_synchronized()
        if current_head == 0:
            return

        my_latest_block = self.get_latest_block()
       
        logger.info(f"Syncing chain from {my_latest_block.header.index} to {current_head}...")
        for block_number in range(my_latest_block.header.index + 1, current_head + 1):
            block = None
            tries = 0
            backoff_sleep = 0.5
            while not block:
                if tries > 3:
                    raise RuntimeError(f"Failed to get block {block_number} from any peer")
                random_peer = random.choice(list(self.peers.keys()))
                block = await self.get_peer_block(random_peer, block_number)
                if not block:
                    await asyncio.sleep(backoff_sleep * (tries + 1))
                    continue
                status = await self.receive_block(block)
                if not status:
                    logger.warning(f"Failed to add block {block_number} from {random_peer}")
                    block = None
                    tries += 1
                    await asyncio.sleep(backoff_sleep * (tries + 1))
                    continue
                logger.info(f"Added block {block_number} from {random_peer}")

    async def is_connected(self) -> bool:
        """Ensure we are connected to the network."""
        for _, status in self.heartbeats.items():
            if time.time() - status < self.heartbeat_timeout:
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
        try:
            # Send join request
            async with aiohttp.ClientSession() as session:
                data = {
                    "host": self.public_host,
                    "version": "0.0.0",
                    "capabilities": ["mining", "relay"],
                    # Serialize MinerInfo as JSON string for transport
                    "info": self.info().to_json()
                }

                async with session.post(
                    f"http://{peer_address}/join",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.node_timeout)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        # Add all nodes from the peer's node list
                        peers_found = 0
                        peers_map = result.get("peers", {}) or {}
                        for peer_host, peer_info_json in peers_map.items():
                            # except ourselves
                            if peer_host == self.public_host:
                                continue
                            info = MinerInfo.from_json(peer_info_json)
                            await self.add_peer(peer_host, info)
                            peers_found += 1

                        logger.info(f"Successfully joined network via {peer_address}")
                        logger.info(f"Discovered {peers_found} peers")
                        return True
                    else:
                        logger.error(f"Failed to join via {peer_address}: {resp.status}")
                        return False

        except Exception as e:
            logger.warning(f"Failed to connect to peer {peer_address}: {e}")
            return False

    async def add_peer(self, host: str, info: MinerInfo) -> bool:
        """Add a node to our registry."""
        if host == self.public_host:
            logger.warning(f"Skipping adding ourselves as a peer: {host}")
            return False 
        
        async with self.net_lock:
            is_new = self.add_or_update_peer(host, info)
            
            if is_new:
                logger.info(f"New peer discovered: {host}: {info.miner_id} ({info.ecdsa_public_key.hex()[:8]})")
                await self._on_new_node(host, info)

                # Broadcast new node to all other nodes
                asyncio.create_task(self.gossip_new_node(host, info))
            
            return is_new
    
    async def remove_node(self, host: str):
        """Remove a node from our registry."""
        async with self.net_lock:
            if host in self.peers:
                del self.peers[host]
                logger.info(f"Node removed: {host}")
                await self._on_node_lost(host)    
    
    async def send_heartbeat(self, node_host: str) -> bool:
        """Send heartbeat to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                data = {"sender": self.public_host}
                async with session.post(
                    f"http://{node_host}/heartbeat",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.node_timeout)
                ) as resp:
                    return resp.status == 200
        except Exception:
            logger.exception(f"Error sending heartbeat to {node_host}")
            return False

    async def get_peer_status(self, host: str) -> Optional[dict]:
        """Get status information from a peer node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}/status",
                    timeout=aiohttp.ClientTimeout(total=self.node_timeout)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.debug(f"Failed to get status from {host}: HTTP {resp.status}")
                        return None
        except Exception:
            logger.exception(f"Error getting status from {host}")
            return None

    async def get_peer_block(self, host: str, block_number: int = 0) -> Optional[Block]:
        """Get a block from a peer node."""
        try:
            req = "/block"
            if block_number > 0:
                req = f"/block/{block_number}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}{req}",
                    timeout=aiohttp.ClientTimeout(total=self.node_timeout)
                ) as resp:
                    if resp.status == 200:
                        block_data = await resp.json()
                        block_bytes = bytes.fromhex(block_data['raw'])
                        signature = bytes.fromhex(block_data['signature'])
                        return Block.from_network(block_bytes + signature)
                    else:
                        logger.debug(f"Failed to get block from {host}: HTTP {resp.status}")
                        return None
        except Exception:
            logger.exception(f"Error getting block from {host}")
            return None

    async def refresh_peer_info(self, host: str) -> bool:
        """Refresh peer information by calling their status endpoint."""
        peer_status = await self.get_peer_status(host)
        if peer_status and 'info' in peer_status:
            try:
                info = MinerInfo.from_json(peer_status['info'])
                async with self.net_lock:
                    self.peers[host] = info

                logger.debug(f"Refreshed info for peer {host}")
                return True
            except Exception:
                logger.exception(f"Error parsing peer info from {host}")
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
        """Send a message to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{host}/gossip",
                    json=asdict(message),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            logger.exception(f"Failed to send message to {host}")
            return False

    async def gossip(self, message: Message):
        """Gossip a new message"""
        if message.id:
            raise ValueError("Message already has an ID, cannot originate a gossip message!")
        
        message.id = blake3(json.dumps(message.__dict__).encode()).hexdigest()
        await self.gossip_broadcast(message, min(self.fanout * 2, len(self.peers)))

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
        """Broadcast a new node to all known nodes."""
        message = Message(
            type="new_node",
            sender=self.public_host,
            timestamp=time.time(),
            data={
                "host": new_node_address,
                "info": new_node_info.to_json()
            }
        )
        
        await self.gossip(message)

    async def gossip_block(self, block_data: Block):
        """Broadcast a new block to the network."""
        message = Message(
            type="block",
            sender=self.public_host,
            timestamp=time.time(),
            data={"signed_block": block_data.to_network().hex()}
        )
        
        await self.gossip(message)

    async def handle_gossip(self, message: Message) -> str:
        """Main gossip logic to handle a gossip message from another node and rebroadcast."""
        # NOTE: we don't try/except here as it's caught in the handle_put_gossip call.

        async with self.gossip_lock:
            if message.id in self.recent_messages:
                return "ok"  # Already processed
            # NOTE: We only check and do not add to the processing list,
            #       as that happens during gossip_broadcast. 

        if message.type == "new_node":
            new_host = message.data.get("host")
            new_info = MinerInfo.from_json(message.data.get("info", "{}"))
            if new_host:
                await self.add_peer(new_host, new_info)
        
        elif message.type == "block":
            block_data = message.data.get("signed_block")
            # skip rebroadcast if invalid data field.
            if not block_data:
                return "rejected, missing signed_block field in data"
            
            block_bytes = bytes.fromhex(block_data)
            block = Block.from_network(block_bytes)
            # Don't rebroadcast if we reject
            if not await self.receive_block(block):
                return "rejected"

        asyncio.create_task(self.gossip_broadcast(message, self.fanout))
        return "ok"

    #######################
    ## HTTP PUT Handlers ##
    #######################
    
    async def handle_put_gossip(self, request: web.Request) -> web.Response:
        """Handle a gossip message from another node."""
        try:
            data = await request.json()
            message = Message(**data)

            status = await self.handle_gossip(message)

            return web.json_response({"status": status})
            
        except Exception as e:
            logger.exception("Error handling broadcast")
            return web.json_response({"error": str(e)}, status=500)


    async def handle_put_join(self, request: web.Request) -> web.Response:
        """Handle join request from a new node."""
        try:
            data = await request.json()
            new_node_address = data.get("host")
            info_field = data.get("info")
            # Expect MinerInfo as JSON string
            new_node_info = MinerInfo.from_json(info_field) if info_field else None

            if not new_node_address or not new_node_info:
                return web.json_response({"error": "Missing host or info"}, status=400)

            # Add the new node
            await self.add_peer(new_node_address, new_node_info)

            # Return our node list as JSON-serializable map host -> MinerInfo JSON string
            async with self.net_lock:
                peers_snapshot = copy.deepcopy(self.peers)

            peers_payload: Dict[str, str] = {}
            for host, info in peers_snapshot.items():
                peers_payload[host] = info.to_json()

            # Add ourself
            peers_payload[self.public_host] = self.info().to_json()

            return web.json_response({
                "status": "ok",
                "peers": peers_payload
            })

        except Exception as e:
            logger.exception("Error handling join")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_put_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from another node."""
        try:
            data = await request.json()
            sender = data.get("sender")
            
            if sender:
                async with self.net_lock:
                    if sender in self.peers:
                        self.heartbeats[sender] = time.time()
                    else:
                        # New node discovered via heartbeat - get their info in background
                        logger.info(f"New node discovered via heartbeat: {sender}")
                        asyncio.create_task(self.refresh_peer_info(sender))
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.exception("Error handling heartbeat")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_put_block(self, request: web.Request) -> web.Response:
        """Handle new block - this is largely open for DEBUG purposes as new blocks
        should be sent via gossip."""
        try:
            block_data = await request.json()

            block_bytes = bytes.fromhex(block_data['raw'])
            signature = bytes.fromhex(block_data['signature'])

            net_data = b''
            net_data += block_bytes
            net_data += signature

            block = Block.from_network(net_data)

            status = "rejected"
            if await self.receive_block(block):
                status = "ok"

            return web.json_response({"status": status})
            
        except Exception as e:
            logger.exception("Error handling new block")
            return web.json_response({"error": str(e)}, status=500)

    #######################
    ## HTTP GET Handlers ##
    #######################
    
    async def handle_get_status(self, request: web.Request) -> web.Response:
        """Return node status."""
        
        return web.json_response({
            "host": self.public_host,
            "info": self.info().to_json(),
            "running": self.running,
            "total_peers": len(self.peers),
            "uptime": time.time() if self.running else 0
        })

    async def handle_get_peers(self, request: web.Request) -> web.Response:
        """Return list of known nodes."""
        async with self.net_lock:
            peers_data = copy.deepcopy(self.peers)
        
        return web.json_response({"peers": peers_data})

    async def handle_get_latest_block(self, request: web.Request) -> web.Response:
        """Return the latest block."""
        try:
            block = self.get_latest_block()
            if block is None:
                return web.json_response({"error": "No blocks in chain"}, status=404)

            # Check response format
            format_param = request.query.get('format', 'json')
            if format_param == 'network':
                # Return network serialized binary data
                return web.Response(
                    body=block.to_network(),
                    content_type='application/octet-stream',
                    headers={'Content-Disposition': 'attachment; filename="latest_block.bin"'}
                )
            else:
                # Return JSON (default)
                return web.json_response(json.loads(block.to_json()))

        except Exception as e:
            logger.exception("Error getting latest block")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_block(self, request: web.Request) -> web.Response:
        """Return a specific block by number."""
        number_str = request.match_info.get('number', 'unknown')
        try:
            # Get block number from URL path parameter
            if number_str is None:
                return web.json_response({"error": "Block number required"}, status=400)

            try:
                block_number = int(number_str)
            except ValueError:
                return web.json_response({"error": "Invalid block number"}, status=400)

            block = self.get_block(block_number)
            if block is None:
                return web.json_response({"error": f"Block {block_number} not found"}, status=404)

            # Check response format
            format_param = request.query.get('format', 'json')
            if format_param == 'network':
                # Return network serialized binary data
                return web.Response(
                    body=block.to_network(),
                    content_type='application/octet-stream',
                    headers={'Content-Disposition': f'attachment; filename="block_{block_number}.bin"'}
                )
            else:
                # Return JSON (default)
                return web.json_response(json.loads(block.to_json()))

        except Exception as e:
            logger.exception(f"Error getting block {number_str}")
            return web.json_response({"error": str(e)}, status=500)
