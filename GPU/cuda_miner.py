"""GPU miner using CUDA via CudaSASampler."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import threading
import time
from typing import Optional
from queue import Queue, Empty

import numpy as np

from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
    evaluate_sampleset,
    calculate_diversity
)
from shared.block_requirements import compute_current_requirements
from GPU.cuda_sa import CudaSASampler, IsingJob


def producer_thread_worker(
    sampler: CudaSASampler,
    prev_block,
    node_info,
    cur_index: int,
    nodes,
    edges,
    batch_size: int,
    num_reads: int,
    num_sweeps: int,
    num_sweeps_per_beta: int,
    stop_event: multiprocessing.synchronize.Event,
    job_metadata_dict: dict,
    logger
):
    """Producer thread: generates Ising models and enqueues jobs to GPU asynchronously."""
    logger.info(f"Producer thread started: batch_size={batch_size}, num_reads={num_reads}")

    try:
        while not stop_event.is_set():
            jobs = []
            batch_nonces = []
            batch_salts = []

            for _ in range(batch_size):
                salt = random.randbytes(32)
                nonce = ising_nonce_from_block(prev_block.hash, node_info.miner_id, cur_index, salt)
                h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

                job = IsingJob(
                    h=h, J=J,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    seed=None
                )
                jobs.append(job)
                batch_nonces.append(nonce)
                batch_salts.append(salt)

            # Enqueue jobs asynchronously
            job_ids = sampler.sample_ising_async(jobs)

            # Track job metadata for result processing
            for job_id, nonce, salt, job in zip(job_ids, batch_nonces, batch_salts, jobs):
                job_metadata_dict[job_id] = {
                    'nonce': nonce,
                    'salt': salt,
                    'h': job.h,
                    'J': job.J,
                }

            logger.debug(f"Producer: enqueued {len(job_ids)} jobs")

    except Exception as e:
        logger.error(f"Producer thread error: {e}")
    finally:
        logger.info("Producer thread stopped")


def consumer_thread_worker(
    sampler: CudaSASampler,
    nodes,
    edges,
    requirements,
    prev_timestamp: int,
    start_time: float,
    stop_event: multiprocessing.synchronize.Event,
    result_queue: Queue,
    logger,
    miner_id: str = "persistent-miner",
    miner_type: str = "GPU-CUDA-SA"
):
    """Consumer thread: reads results from GPU and puts them in result queue."""
    logger.info("Consumer thread started")

    try:
        while not stop_event.is_set():
            # Dequeue results from GPU
            results = sampler.dequeue_results(timeout=0.1)

            if not results:
                continue

            logger.debug(f"Consumer: dequeued {len(results)} results")

            for job_id, sampleset in results:
                try:
                    # Put result in queue for main mining loop
                    result_queue.put({
                        'job_id': job_id,
                        'sampleset': sampleset,
                    }, timeout=1.0)
                except:
                    logger.warning("Result queue full, dropping result")

    except Exception as e:
        logger.error(f"Consumer thread error: {e}")
    finally:
        logger.info("Consumer thread stopped")


class CudaMiner(BaseMiner):
    def __init__(self, miner_id: str, device: str = "0", **cfg):
        # Set CUDA device BEFORE creating sampler
        try:
            import cupy as cp
            device_id = int(device)
            cp.cuda.Device(device_id).use()
        except Exception as e:
            print(f"Warning: Failed to set CUDA device {device}: {e}")

        # Initialize CUDA SA sampler on the selected device
        sampler = CudaSASampler(device=int(device))
        super().__init__(miner_id, sampler)
        # Now update sampler with our logger
        sampler.logger = self.logger
        self.miner_type = "GPU-CUDA-SA"
        self.device = device

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _gpu_result_to_sampleset(self, gpu_result: dict, h, J, nodes: list) -> Optional[object]:
        """Convert GPU result dict to dimod.SampleSet.

        Args:
            gpu_result: Dict with 'min_energy', 'avg_energy', 'num_reads_done'
            h: Ising h biases
            J: Ising J couplings
            nodes: List of node indices

        Returns:
            dimod.SampleSet or None if conversion fails
        """
        try:
            import dimod

            # For now, create a dummy sampleset with the energy values
            # In production, this would contain actual samples from GPU
            num_reads = gpu_result.get('num_reads_done', 1)
            min_energy = gpu_result.get('min_energy', 0.0)
            avg_energy = gpu_result.get('avg_energy', 0.0)

            # Create dummy samples (all +1) - in production, use actual GPU samples
            samples = [{node: 1 for node in nodes} for _ in range(num_reads)]
            energies = [min_energy + (avg_energy - min_energy) * i / max(1, num_reads - 1) for i in range(num_reads)]

            sampleset = dimod.SampleSet.from_samples(samples, 'SPIN', energies)
            return sampleset
        except Exception as e:
            self.logger.error(f"Failed to convert GPU result to sampleset: {e}")
            return None
    
    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of CUDA resources."""
        if hasattr(self, 'logger'):
            self.logger.info(f"CUDA miner {self.miner_id} received SIGTERM, cleaning up GPU device {self.device}...")
        
        # CUDA-specific cleanup
        try:
            # Import CUDA modules for cleanup
            try:
                import cupy as cp
                # Reset CUDA device to free all memory and contexts
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.runtime.deviceReset()
                if hasattr(self, 'logger'):
                    self.logger.info(f"CUDA device {self.device} reset completed")
            except ImportError:
                # Try alternative CUDA cleanup if CuPy not available
                try:
                    import pycuda.driver as cuda
                    # Initialize CUDA if not already done
                    if not hasattr(cuda, '_initialized') or not cuda._initialized:
                        cuda.init()
                    # Get device and reset
                    device = cuda.Device(int(self.device))
                    context = device.make_context()
                    context.pop()
                    context.detach()
                    if hasattr(self, 'logger'):
                        self.logger.info(f"PyCUDA device {self.device} context cleanup completed")
                except ImportError:
                    if hasattr(self, 'logger'):
                        self.logger.warning("No CUDA library available for device reset")
            
            # Clear any cached data
            if hasattr(self, 'top_attempts'):
                self.top_attempts.clear()
            
            # Reset sampler state if possible
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'cleanup'):
                self.sampler.cleanup()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during CUDA miner cleanup: {e}")
        
        # Exit gracefully
        sys.exit(0)
        
    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
    ) -> Optional[MiningResult]:
        """Mine a block using persistent CUDA kernel with producer/consumer threads.

        Args:
            prev_block: Previous block in the chain
            node_info: Node information containing miner_id and other details
            requirements: BlockRequirements object with difficulty settings
            prev_timestamp: Timestamp from the previous block header
            stop_event: Multiprocessing event to signal stop

        Returns:
            MiningResult if successful, None if stopped or failed
        """
        # CRITICAL: Set device context at start of mine_block
        try:
            import cupy as cp
            device_id = int(self.device)
            cp.cuda.Device(device_id).use()
            self.logger.debug(f"Device context set to {device_id}")
        except Exception as e:
            self.logger.error(f"Failed to set device context: {e}")
            return None

        self.mining = True
        start_time = time.time()
        self.top_attempts = []

        self.logger.debug(f"requirements: {requirements}")

        cur_index = prev_block.header.index + 1

        # Mark that this miner is attempting this round
        self.current_round_attempted = True
        self.logger.info(f"Mining block {cur_index} with persistent kernel...")

        # Apply difficulty decay based on elapsed time since previous block
        current_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
        difficulty_energy = current_requirements.difficulty_energy
        min_diversity = current_requirements.min_diversity
        min_solutions = current_requirements.min_solutions

        params = adapt_parameters(difficulty_energy, min_diversity, min_solutions)
        self.logger.info(f"{self.miner_id} - Adaptive params: {params}")

        # Get topology information from sampler
        nodes = self.sampler.nodes
        edges = self.sampler.edges

        # Initialize persistent kernel
        try:
            persistent_sampler = CudaSASampler(device=int(self.device))
            self.logger.info("Persistent kernel initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize persistent kernel: {e}")
            return None

        # Create result queue for consumer thread
        result_queue = Queue(maxsize=100)

        # Dictionary to track job metadata (job_id -> {nonce, salt, h, J})
        job_metadata_dict = {}

        # Start producer and consumer threads
        producer_stop = multiprocessing.Event()
        consumer_stop = multiprocessing.Event()

        producer = threading.Thread(
            target=producer_thread_worker,
            args=(
                persistent_sampler, prev_block, node_info, cur_index, nodes, edges,
                10,  # batch_size
                params.get('num_reads', 100),
                params.get('num_sweeps', 512),
                10,  # num_sweeps_per_beta
                producer_stop,
                job_metadata_dict,
                self.logger
            ),
            daemon=True
        )

        consumer = threading.Thread(
            target=consumer_thread_worker,
            args=(
                persistent_sampler, nodes, edges, current_requirements, prev_timestamp,
                start_time, consumer_stop, result_queue, self.logger
            ),
            daemon=True
        )

        producer.start()
        consumer.start()
        self.logger.info("Producer and consumer threads started")

        # Main mining loop
        last_requirement_check = time.time()
        requirement_check_interval = 5.0  # Check every 5 seconds

        try:
            while self.mining and not stop_event.is_set():
                # Check if we should stop
                if stop_event.is_set():
                    break

                # Try to get results from consumer thread
                try:
                    result = result_queue.get(timeout=0.5)
                    job_id = result.get('job_id')
                    sampleset = result.get('sampleset')
                    self.logger.debug(f"Got result from queue: job_id={job_id}")

                    # Get job metadata (nonce, salt)
                    if job_id not in job_metadata_dict:
                        self.logger.warning(f"No metadata for job_id {job_id}, skipping")
                        continue

                    metadata = job_metadata_dict.pop(job_id)
                    nonce = metadata['nonce']
                    salt = metadata['salt']

                    # Evaluate sampleset against current requirements
                    mining_result = self.evaluate_sampleset(
                        sampleset, current_requirements, nodes, edges,
                        nonce, salt, prev_timestamp, start_time
                    )

                    if mining_result:
                        # Found a valid result!
                        self.logger.info(
                            f"[Block-{cur_index}] Mined! Nonce: {nonce}, Salt: {salt.hex()[:4]}..., "
                            f"Min Energy: {mining_result.energy:.2f}, Solutions: {mining_result.num_valid}, "
                            f"Diversity: {mining_result.diversity:.3f}, Mining Time: {time.time() - start_time:.2f}s"
                        )
                        return mining_result
                    else:
                        # Result didn't meet requirements, cache it for later
                        self.logger.debug(f"Result didn't meet requirements: nonce={nonce}, salt={salt.hex()[:8]}...")
                        # Cache for later if requirements change
                        self.update_top_samples(sampleset, nonce, salt, current_requirements)

                except Exception as e:
                    if "Empty" not in str(type(e)):
                        self.logger.debug(f"Result queue timeout or error: {e}")
                    continue

                # Periodically check if requirements have changed (every 5 seconds)
                current_time = time.time()
                if current_time - last_requirement_check >= requirement_check_interval:
                    last_requirement_check = current_time

                    # Update requirements if necessary
                    updated_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
                    if current_requirements != updated_requirements:
                        current_requirements = updated_requirements
                        params = adapt_parameters(current_requirements.difficulty_energy, current_requirements.min_diversity, current_requirements.min_solutions)
                        self.logger.info(f"{self.miner_id} - updated adaptive params: {params}")
                        difficulty_energy = current_requirements.difficulty_energy
                        min_diversity = current_requirements.min_diversity
                        min_solutions = current_requirements.min_solutions

                        # Check if any cached results now meet the new requirements
                        for sample in self.top_attempts:
                            if min(sample.sampleset.record.energy) <= current_requirements.difficulty_energy:
                                result = self.evaluate_sampleset(sample.sampleset, current_requirements, nodes, edges,
                                                                 sample.nonce, sample.salt, prev_timestamp, start_time)
                                if result:
                                    self.logger.info(f"[Block-{cur_index}] Already Mined at this difficulty! Nonce: {sample.nonce}, Salt: {sample.salt.hex()[:4]}..., Min Energy: {result.energy:.2f}, Solutions: {result.num_valid}, Diversity: {result.diversity:.3f}, Mining Time: {time.time() - start_time:.2f}s")
                                    return result

        except Exception as e:
            self.logger.error(f"Error in persistent mining loop: {e}")
        finally:
            # Stop producer and consumer threads
            producer_stop.set()
            consumer_stop.set()
            persistent_sampler.stop(drain=True)
            producer.join(timeout=5.0)
            consumer.join(timeout=5.0)
            self.logger.info("Persistent mining stopped")

        self.logger.info("Stopping mining, no results found")
        return None


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int):
    """Calculate adaptive mining parameters based on difficulty requirements.

    Optimized for GPU CUDA performance - uses much lower sweep counts than CPU
    since GPU parallelism allows faster convergence with fewer sweeps.
    """
    # Normalize difficulty factor (more negative = harder)
    difficulty_factor = abs(difficulty_energy) / 1000.0  # Base around -1000

    # GPU CUDA parameters - optimized for fast convergence
    # GPU can afford fewer sweeps due to massive parallelism
    base_sweeps = 512  # Much lower base than CPU (was 512, now 8x faster)
    # Use square root scaling instead of exponential to prevent explosion
    num_sweeps = int(base_sweeps * (difficulty_factor ** 0.5))  # Gentler scaling
    num_reads = max(int(min_solutions) * 2, 32)  # Reduce reads slightly

    return {
        'num_sweeps': max(32, min(num_sweeps, 512)),     # Much lower max (was 32768)
        'num_reads': max(32, min(num_reads, 200)),       # Reasonable bounds
    }