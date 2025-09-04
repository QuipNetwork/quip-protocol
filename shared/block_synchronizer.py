import asyncio
import multiprocessing as mp
import queue
import random
import signal
import time
import logging
from typing import List, Optional, Callable

from shared.block import Block
from shared.node_client import NodeClient


class BlockSynchronizer:
    """Multiprocessing block synchronizer with producer/consumer pattern."""
    
    def __init__(self, 
                 node_client: NodeClient,
                 receive_block_callback: Callable,
                 logger: Optional[logging.Logger] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize BlockSynchronizer.
        
        Args:
            node_client: NodeClient for peer communication
            receive_block_callback: Callback to process/validate blocks
            logger: Logger instance
            max_workers: Max number of worker processes (default: min(30, cpu_count))
        """
        self.node_client = node_client
        self.receive_block_callback = receive_block_callback
        self.logger = logger or logging.getLogger(__name__)
        self.max_workers = max_workers or min(30, mp.cpu_count())
        
    async def sync_blocks(self, start_index: int, end_index: int) -> bool:
        """
        Synchronize blocks from start_index to end_index using multiprocessing.
        
        Returns:
            bool: True if all blocks were successfully synced
        """
        if start_index > end_index:
            return True
            
        total_blocks = end_index - start_index + 1
        self.logger.debug(f"Syncing chain from {start_index} to {end_index} ({total_blocks} blocks)...")
        
        # Use multiprocessing queues for communication between processes
        download_queue = mp.Queue()
        completed_queue = mp.Queue()
        
        # Initialize download queue with block numbers
        for block_number in range(start_index, end_index + 1):
            download_queue.put(block_number)
            
        # Calculate optimal number of producer processes
        num_producers = min(self.max_workers, len(self.node_client.peers), end_index - start_index + 1)
        num_producers = max(1, num_producers)  # At least 1 producer
        
        try:
            # Create shared data for worker processes
            peers_list = list(self.node_client.peers.keys())
            
            # Start producer processes
            producer_processes = []
            self.logger.debug(f"🚀 Starting {num_producers} producer processes for concurrent downloads")
            for i in range(num_producers):
                p = mp.Process(
                    target=self._producer_worker,
                    args=(download_queue, completed_queue, peers_list, self.node_client.node_timeout)
                )
                p.start()
                producer_processes.append(p)
                self.logger.debug(f"🔧 Started producer process {p.pid} ({i+1}/{num_producers})")
                
            # Run consumer in main process to handle block validation
            success = await self._consumer_async(completed_queue, start_index, end_index, download_queue)
            
            # Signal producers to stop and wait for them to finish
            self.logger.debug("🛑 Signaling producer processes to stop")
            for _ in range(num_producers):
                download_queue.put(None)  # Sentinel value to stop producers
                
            for p in producer_processes:
                p.join(timeout=30)  # Wait up to 30 seconds
                if p.is_alive():
                    self.logger.warning(f"⚠️ Force terminating producer process {p.pid}")
                    p.terminate()
                    p.join()
                else:
                    self.logger.debug(f"✅ Producer process {p.pid} stopped gracefully")
                    
            if success:
                self.logger.debug(f"🎉 Successfully synced {total_blocks} blocks from {start_index} to {end_index}")
            else:
                self.logger.error(f"❌ Failed to sync blocks from {start_index} to {end_index}")
                    
            return success
            
        except Exception as e:
            self.logger.error(f"Block sync failed: {e}")
            return False
            
    @staticmethod
    def _producer_worker(download_queue: mp.Queue, 
                        completed_queue: mp.Queue, 
                        peers: List[str], 
                        node_timeout: float):
        """
        Producer worker process that downloads blocks from peers.
        
        Args:
            download_queue: Queue of block numbers to download
            completed_queue: Queue to put completed downloads
            peers: List of peer host addresses
            node_timeout: Timeout for HTTP requests
        """
        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize NodeClient in this process
        client = NodeClient(node_timeout=node_timeout)
        
        async def download_worker():
            await client.start()
            
            try:
                while True:
                    block_number = None  # Initialize to handle potential exceptions
                    try:
                        # Get next block number to download (timeout to prevent hanging)
                        block_number = download_queue.get(timeout=1.0)
                        if block_number is None:  # Sentinel value to stop
                            break

                        # Log download start with timing
                        import os
                        import logging
                        import time
                        logger = logging.getLogger(f"BlockSync.Producer.{os.getpid()}")
                        download_start = time.perf_counter()
                        logger.debug(f"🔽 Starting download of block {block_number}")
                        
                        block = await _download_single_block(client, block_number, peers)
                        
                        # Log download completion with timing
                        download_time = time.perf_counter() - download_start
                        if block:
                            logger.info(f"✅ Completed download of block {block_number} in {download_time:.3f}s")
                        else:
                            logger.warning(f"❌ Failed download of block {block_number} after {download_time:.3f}s")
                        
                        completed_queue.put((block_number, block))

                    except queue.Empty:
                        continue
                    except Exception:
                        if block_number is None:
                            block_number = -1
                        # Put error result in completed queue
                        completed_queue.put((block_number, None))
                        
            finally:
                await client.stop()
                
        try:
            loop.run_until_complete(download_worker())
        finally:
            loop.close()
            
    async def _consumer_async(self, completed_queue: mp.Queue, 
                             start_index: int, end_index: int, 
                             download_queue: mp.Queue) -> bool:
        """
        Consumer running in main process that validates blocks sequentially.
        
        Args:
            completed_queue: Queue of completed block downloads
            start_index: Starting block index
            end_index: Ending block index  
            download_queue: Queue to put retry requests
            
        Returns:
            bool: True if all blocks processed successfully
        """
        next_expected = start_index
        pending_blocks = {}
        retry_count = {}
        max_retries = 3
        
        while next_expected <= end_index:
            try:
                # Get completed download (with timeout) - run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                try:
                    block_number, block = await loop.run_in_executor(
                        None, lambda: completed_queue.get(timeout=30.0)
                    )
                except Exception:
                    # Handle queue timeout or other errors
                    self.logger.error("Timeout waiting for block download completion")
                    return False
                
                if block is None:
                    # Download failed
                    retry_count[block_number] = retry_count.get(block_number, 0) + 1
                    if retry_count[block_number] <= max_retries:
                        self.logger.warning(f"🔄 Retrying download for block {block_number} ({retry_count[block_number]}/{max_retries})")
                        download_queue.put(block_number)  # Retry download
                        continue
                    else:
                        self.logger.error(f"❌ Failed to download block {block_number} after {max_retries} retries")
                        return False
                        
                # Log block received from producer
                self.logger.debug(f"📨 Received block {block_number} from producer, storing for sequential processing")
                        
                # Store block for sequential processing
                pending_blocks[block_number] = block
                
                # Process blocks in sequential order
                while next_expected in pending_blocks:
                    block_to_process = pending_blocks.pop(next_expected)
                    
                        
                    # Log block processing start
                    self.logger.debug(f"🔄 Processing block {next_expected} sequentially")
                    
                    # Validate and add block using async callback
                    status = await self.receive_block_callback(block_to_process)
                    
                    # Log block processing completion
                    if status:
                        self.logger.debug(f"✅ Successfully processed block {next_expected}")
                    else:
                        self.logger.warning(f"❌ Failed to process block {next_expected}")
                        
                    if not status:
                        retry_count[next_expected] = retry_count.get(next_expected, 0) + 1
                        if retry_count[next_expected] <= max_retries:
                            self.logger.warning(f"Block {next_expected} validation failed, retrying ({retry_count[next_expected]}/{max_retries})")
                            download_queue.put(next_expected)  # Retry download
                            continue
                        else:
                            self.logger.error(f"Block {next_expected} validation failed after {max_retries} retries")
                            return False
                    
                    next_expected += 1
                    
            except queue.Empty:
                self.logger.error("Timeout waiting for block download completion")
                return False
            except Exception as e:
                self.logger.error(f"Consumer error: {e}")
                return False
                
        return True


async def _download_single_block(client: NodeClient, 
                                block_number: int, 
                                peers: List[str]) -> Optional[Block]:
    """
    Download a single block from peers with retry logic.
    
    Args:
        client: NodeClient instance
        block_number: Block number to download
        peers: List of peer addresses
        
    Returns:
        Block if successful, None if failed
    """
    tries = 0
    max_tries = 3
    backoff_sleep = 0.5
    available_peers = peers.copy()
    
    while tries <= max_tries and available_peers:
        # Download from random peer
        random_peer = random.choice(available_peers)
        block = await client.get_peer_block(random_peer, block_number)
        
        if block:
            return block
            
        # Remove failed peer and retry
        tries += 1
        available_peers.remove(random_peer)
        if available_peers:  # Only sleep if we have more peers to try
            await asyncio.sleep(backoff_sleep * tries)
            
    return None