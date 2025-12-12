import asyncio
import multiprocessing as mp
import queue
import random
import signal
import time
import logging
from typing import List, Optional

from shared.block import Block
from shared.node_client import NodeClient
import os
import logging
import time


class BlockSynchronizer:
    """Multiprocessing block synchronizer with producer/consumer pattern."""

    def __init__(self,
                 node_client: NodeClient,
                 receive_block_queue: asyncio.Queue,
                 logger: Optional[logging.Logger] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize BlockSynchronizer.

        Args:
            node_client: NodeClient for peer communication
            receive_block_queue: Callback to process/validate blocks
            logger: Logger instance
            max_workers: Max number of worker processes (default: min(30, cpu_count))
        """
        self.node_client = node_client
        self.receive_block_queue = receive_block_queue
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
        
        producer_processes = []
        try:
            # Create shared data for worker processes
            peers_list = list(self.node_client.peers.keys())
            
            # Start producer processes
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
                    
            if success:
                self.logger.debug(f"🎉 Successfully synced {total_blocks} blocks from {start_index} to {end_index}")
            else:
                self.logger.error(f"❌ Failed to sync blocks from {start_index} to {end_index}")
                    
            return success
            
        except KeyboardInterrupt:
            self.logger.warning("🛑 Block synchronization interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Block sync failed: {e}")
            return False
        finally:
            # Always cleanup producer processes
            self.logger.debug("🛑 Signaling producer processes to stop")
            for _ in range(num_producers):
                try:
                    download_queue.put(None)  # Sentinel value to stop producers
                except:
                    pass  # Queue might be full or closed
                    
            for p in producer_processes:
                if p.is_alive():
                    p.join(timeout=2)  # Short timeout for graceful shutdown
                    if p.is_alive():
                        self.logger.warning(f"⚠️ Force terminating producer process {p.pid}")
                        p.terminate()
                        p.join(timeout=1)
                        if p.is_alive():
                            self.logger.warning(f"🔥 Sending SIGKILL to producer process {p.pid}")
                            try:
                                import os
                                os.kill(p.pid, 9)  # SIGKILL
                                p.join(timeout=1)
                                if p.is_alive():
                                    self.logger.error(f"❌ Failed to kill producer process {p.pid}")
                                else:
                                    self.logger.debug(f"✅ Producer process {p.pid} killed")
                            except (OSError, ProcessLookupError):
                                # Process already dead
                                self.logger.debug(f"✅ Producer process {p.pid} already terminated")
                        else:
                            self.logger.debug(f"✅ Producer process {p.pid} terminated gracefully")
                    else:
                        self.logger.debug(f"✅ Producer process {p.pid} stopped gracefully")
            
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
        shutdown_flag = False
        def signal_handler(_signum, _frame):
            nonlocal shutdown_flag
            shutdown_flag = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize NodeClient in this process
        client = NodeClient(node_timeout=node_timeout)
        
        async def download_worker():
            await client.start()
            
            try:
                while True:
                    # Check shutdown flag first
                    if shutdown_flag:
                        break
                        
                    block_number = None  # Initialize to handle potential exceptions
                    try:
                        # Get next block number to download (timeout to prevent hanging)
                        block_number = download_queue.get(timeout=1.0)
                        if block_number is None:  # Sentinel value to stop
                            break

                        # Log download start with timing
                        logger = logging.getLogger(f"BlockSync.Producer.{os.getpid()}")
                        download_start = time.perf_counter()
                        logger.debug(f"🔽 Starting download of block {block_number}")
                        
                        # Check shutdown flag before starting download
                        if shutdown_flag:
                            break
                            
                        block = await _download_single_block(client, block_number, peers, lambda: shutdown_flag)
                        
                        # Log download completion with timing
                        download_time = time.perf_counter() - download_start
                        if block:
                            logger.debug(f"✅ Completed download of block {block_number} in {download_time:.3f}s")
                        else:
                            logger.warning(f"❌ Failed download of block {block_number} after {download_time:.3f}s")
                        
                        try:
                            # Use timeout to avoid hanging on queue put
                            completed_queue.put((block_number, block), timeout=1.0)
                        except Exception:
                            # Queue might be full or closed, exit gracefully
                            break

                    except queue.Empty:
                        continue
                    except Exception:
                        if shutdown_flag:
                            break
                        if block_number is None:
                            block_number = -1
                        # Put error result in completed queue with timeout
                        try:
                            completed_queue.put((block_number, None), timeout=1.0)
                        except Exception:
                            # Queue might be closed, exit gracefully
                            break
                        
            finally:
                try:
                    await asyncio.wait_for(client.stop(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Force close if client.stop() hangs
                    pass
                except Exception:
                    # Ignore any other errors during cleanup
                    pass
                
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
        retry_count = {}
        max_retries = 3
        next_expected = start_index
        
        try:
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
                            
                    # Put block in queue for consumer task to process
                    dummy_future = asyncio.Future()
                    self.receive_block_queue.put_nowait((block, dummy_future))
                    next_expected += 1

                    # Give consumer task a chance to run
                    await asyncio.sleep(0.01)
                                                                        
                except queue.Empty:
                    self.logger.error("Timeout waiting for block download completion")
                    return False
                except Exception as e:
                    self.logger.error(f"Consumer error: {e}")
                    return False
        except KeyboardInterrupt:
            self.logger.warning("🛑 Block processing interrupted by user")
            return False
                    
        return True


async def _download_single_block(client: NodeClient,
                                block_number: int, 
                                peers: List[str],
                                is_shutdown_requested = None) -> Optional[Block]:
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
        # Check for shutdown request
        if is_shutdown_requested and is_shutdown_requested():
            return None
            
        # Download from random peer
        random_peer = random.choice(available_peers)
        try:
            # Use a shorter timeout for HTTP operations during shutdown
            block = await client.get_peer_block(random_peer, block_number)
            
            if block:
                return block
        except Exception:
            # Continue to next peer on any exception
            pass
            
        # Remove failed peer and retry
        tries += 1
        available_peers.remove(random_peer)
        if available_peers and (not is_shutdown_requested or not is_shutdown_requested()):
            # Only sleep if we have more peers to try and not shutting down
            await asyncio.sleep(backoff_sleep * tries)
            
    return None