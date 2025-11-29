"""GPU miner using CUDA persistent kernel via CudaSASamplerAsync."""
from __future__ import annotations

import math
import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Optional

import numpy as np
import dimod

from shared.base_miner import BaseMiner, MiningResult
from shared.quantum_proof_of_work import (
    ising_nonce_from_block,
    generate_ising_model_from_nonce,
    evaluate_sampleset,
)
from shared.block_requirements import compute_current_requirements
from shared.energy_utils import energy_to_difficulty, DEFAULT_NUM_NODES, DEFAULT_NUM_EDGES
from GPU.cuda_kernel import CudaKernelRealSA
from GPU.cuda_sa import CudaKernelAdapter, CudaSASamplerAsync
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


def adapt_parameters(
    difficulty_energy: float,
    min_diversity: float,
    min_solutions: int,
    num_nodes: int = DEFAULT_NUM_NODES,
    num_edges: int = DEFAULT_NUM_EDGES
) -> dict:
    """Adapt mining parameters based on difficulty.

    Uses GSE-based difficulty calculation with log-linear interpolation
    in sweep space, optimized for CUDA GPU performance.

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum solution diversity required (reserved)
        min_solutions: Minimum number of valid solutions required
        num_nodes: Number of nodes in topology (default: DEFAULT_TOPOLOGY)
        num_edges: Number of edges in topology (default: DEFAULT_TOPOLOGY)

    Returns:
        Dictionary with num_sweeps, num_reads, and num_sweeps_per_beta
    """
    # Get normalized difficulty [0, 1]
    difficulty = energy_to_difficulty(
        difficulty_energy,
        num_nodes=num_nodes,
        num_edges=num_edges
    )

    # CUDA GPU calibration ranges
    min_sweeps = 256     # Easiest difficulty (fast GPU convergence)
    max_sweeps = 2048    # Hardest difficulty

    # Direct linear scaling: difficulty × max_sweeps
    num_sweeps = max(min_sweeps, int(difficulty * max_sweeps))

    # Reads scale linearly with difficulty (capped at 256 for CUDA hardware limit)
    min_reads = 64
    max_reads = 256  # CUDA max_threads_per_job limit
    num_reads = max(min_reads, int(difficulty * max_reads))

    return {
        'num_sweeps': num_sweeps,
        'num_reads': max(num_reads, min_solutions),
        'num_sweeps_per_beta': 1
    }


class CudaMiner(BaseMiner):
    """CUDA GPU miner using persistent kernel for high-throughput mining.

    Uses CudaSASamplerAsync which wraps the persistent CUDA kernel.
    The kernel runs continuously on the GPU, processing jobs from a ring buffer.
    This eliminates kernel launch overhead and enables high job throughput.
    """

    def __init__(self, miner_id: str, device: str = "0", topology=None, **cfg):
        """Initialize CUDA miner.

        Args:
            miner_id: Unique identifier for this miner
            device: CUDA device ID (default "0")
            topology: Optional topology object (default: DEFAULT_TOPOLOGY)
            **cfg: Additional configuration parameters
        """
        # Set CUDA device BEFORE creating any CUDA objects
        try:
            if cp is None:
                raise ImportError("cupy not available")
            device_id = int(device)
            cp.cuda.Device(device_id).use()
        except Exception as e:
            print(f"Warning: Failed to set CUDA device {device}: {e}")

        # Get topology (use provided or default)
        topology_obj = topology if topology is not None else DEFAULT_TOPOLOGY
        self.nodes = list(topology_obj.graph.nodes)
        self.edges = list(topology_obj.graph.edges)
        # Precompute node indices as numpy array for fast filtering
        self._node_indices = np.array(self.nodes, dtype=np.int32)

        # Initialize persistent kernel with large ring buffer for batched mining
        self.kernel = CudaKernelRealSA(
            ring_size=256,  # Support up to 256 jobs in flight
            max_threads_per_job=256,
            max_N=5000,
            verbose=False  # Disable debug output for production
        )
        self.adapter = CudaKernelAdapter(self.kernel)
        self.async_sampler = CudaSASamplerAsync(self.adapter)

        # Create a minimal sampler interface for BaseMiner
        # BaseMiner expects a sampler with nodes/edges attributes
        class SamplerInterface:
            def __init__(self, nodes, edges, properties):
                self.nodes = nodes
                self.edges = edges
                self.nodelist = nodes
                self.edgelist = edges
                self.properties = properties
                self.sampler_type = "cuda-persistent"

            def sample_ising(self, h, J, **kwargs):
                """Dummy sample_ising - not used in CudaMiner."""
                raise NotImplementedError("CudaMiner handles sampling directly")

        minimal_sampler = SamplerInterface(self.nodes, self.edges, topology_obj.properties)

        # Initialize base miner (sets up logger, miner_id, etc.)
        super().__init__(miner_id, minimal_sampler)

        self.miner_type = "GPU-CUDA-Persistent"
        self.device = device

        # GPU utilization control (0-100, default 100)
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(f"gpu_utilization must be between 1-100, got {self.gpu_utilization}")

        # Register SIGTERM handler for graceful cleanup
        signal.signal(signal.SIGTERM, self._cleanup_handler)

        self.logger.info(f"CUDA miner initialized on device {device} (persistent kernel)")
        self.logger.info(f"Topology: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM signal for graceful cleanup of CUDA resources."""
        self.logger.info(f"CUDA miner {self.miner_id} received SIGTERM, cleaning up...")

        try:
            # Stop the sampler (stops persistent kernel)
            if hasattr(self, 'async_sampler'):
                self.async_sampler.stop_immediate()

            # Synchronize and free memory pools (deviceReset doesn't exist in CuPy)
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                self.logger.info(f"CUDA device {self.device} cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during CUDA miner cleanup: {e}")

        sys.exit(0)

    def _filter_samples_for_sparse_topology(self, sampleset: dimod.SampleSet) -> dimod.SampleSet:
        """Filter samples to extract only actual topology nodes.

        The sampler returns samples of length N=4800 (max node ID + 1) because
        the topology has sparse node IDs. But mining validation expects samples
        of length 4593 (actual number of nodes). This filters the samples to
        extract only the values at the actual node indices.

        Args:
            sampleset: SampleSet with samples of length 4800

        Returns:
            SampleSet with samples filtered to length 4593
        """
        # Use numpy advanced indexing for vectorized extraction
        # self._node_indices is precomputed in __init__ for speed
        samples = sampleset.record.sample
        filtered_samples = samples[:, self._node_indices].astype(np.int8)

        # Create new SampleSet with filtered samples
        return dimod.SampleSet.from_samples(
            filtered_samples,
            vartype='SPIN',
            energy=sampleset.record.energy,
            info=sampleset.info
        )

    def mine_block(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        drain: bool = False,
    ) -> Optional[MiningResult]:
        """Mine a block using persistent CUDA kernel.

        This simplified implementation uses the async persistent kernel API,
        eliminating the need for producer/consumer threads. The kernel handles
        job parallelism internally.

        Args:
            prev_block: Previous block in the chain
            node_info: Node information containing miner_id
            requirements: BlockRequirements object with difficulty settings
            prev_timestamp: Timestamp from previous block header
            stop_event: Multiprocessing event to signal stop
            drain: If False (default), return immediately on first valid result.
                If True, process entire batch and return best result.

        Returns:
            MiningResult if successful, None if stopped or failed
        """
        # Set device context
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(f"Failed to set device context: {e}")
            return None

        self.mining = True
        start_time = time.time()
        cur_index = prev_block.header.index + 1

        self.logger.info(f"Mining block {cur_index} with persistent kernel...")

        # Apply difficulty decay
        current_requirements = compute_current_requirements(requirements, prev_timestamp, self.logger)
        difficulty_energy = current_requirements.difficulty_energy
        min_diversity = current_requirements.min_diversity
        min_solutions = current_requirements.min_solutions

        # Adapt parameters based on difficulty
        params = adapt_parameters(
            difficulty_energy,
            min_diversity,
            min_solutions,
            num_nodes=len(self.nodes),
            num_edges=len(self.edges)
        )

        # Track current sweeps for incremental increase (reads stay constant)
        current_num_sweeps = params['num_sweeps']
        num_reads = params['num_reads']  # Constant - doesn't increment
        max_num_sweeps = params['num_sweeps']
        num_sweeps_per_beta = params['num_sweeps_per_beta']

        # Increment rate: increase by 5% every 30 seconds
        increment_interval = 30.0
        last_increment_time = start_time

        self.logger.info(f"Adaptive params: {current_num_sweeps} sweeps, {num_reads} reads")

        # Batch size: number of jobs to run in parallel
        # Scale batch size by utilization percentage to control GPU load
        max_batch_size = self.kernel.num_blocks  # Typically 48 SMs for most GPUs
        batch_size = max(1, int(max_batch_size * (self.gpu_utilization / 100.0)))
        self.logger.info(f"GPU batch size: {batch_size}/{max_batch_size} parallel jobs ({self.gpu_utilization}% utilization)")

        # Compute N for sparse topology
        N = max(max(self.nodes), max(max(i, j) for i, j in self.edges)) + 1

        attempts = 0

        while not stop_event.is_set():
            # Increment sweeps slowly over time (reads stay constant)
            current_time = time.time()
            if current_time - last_increment_time >= increment_interval:
                # Increase sweeps by 1% toward max
                current_num_sweeps = min(max_num_sweeps, int(current_num_sweeps * 1.05))
                last_increment_time = current_time

            # Generate batch of Ising problems
            h_list = []
            J_list = []
            h_dicts = []  # Keep dict versions for evaluate_sampleset
            J_dicts = []
            salts = []
            nonces = []

            for _ in range(batch_size):
                salt = random.randbytes(32)
                nonce = ising_nonce_from_block(
                    prev_block.hash, node_info.miner_id, cur_index, salt
                )

                h_dict, J_dict = generate_ising_model_from_nonce(nonce, self.nodes, self.edges)
                h_dicts.append(h_dict)
                J_dicts.append(J_dict)

                # Convert to arrays (using sparse indexing with N=4800)
                h = np.zeros(N, dtype=np.float32)
                for node, val in h_dict.items():
                    h[node] = val

                J = np.zeros(len(self.edges), dtype=np.float32)
                edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
                for (i, j), val in J_dict.items():
                    if (i, j) in edge_to_idx:
                        J[edge_to_idx[(i, j)]] = val
                    elif (j, i) in edge_to_idx:
                        J[edge_to_idx[(j, i)]] = val

                h_list.append(h)
                J_list.append(J)
                salts.append(salt)
                nonces.append(nonce)

            # Submit batch to GPU (non-blocking)
            try:
                job_ids = self.async_sampler.sample_ising_async(
                    h_list=h_list,
                    J_list=J_list,
                    num_reads=num_reads,
                    num_betas=current_num_sweeps,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    edges=self.edges,
                )
            except Exception as e:
                self.logger.error(f"Sampling error: {e}")
                continue

            # Process results as they arrive (incremental, not waiting for all)
            remaining_jobs = set(range(len(job_ids)))
            job_to_idx = {job_ids[i]: i for i in range(len(job_ids))}
            best_batch_result = None
            last_result_time = time.time()
            drain_timeout = 0.5  # 500ms timeout for drain mode

            while remaining_jobs:
                # Exit immediately if stop requested
                if stop_event.is_set():
                    self.logger.info("Mining stopped by stop_event")
                    return best_batch_result

                # Poll for next result (non-blocking)
                result = self.async_sampler.kernel.try_dequeue_result()
                if result is None:
                    # In drain mode with a valid result, exit if we've waited too long
                    if drain and best_batch_result and (time.time() - last_result_time) > drain_timeout:
                        self.logger.info(f"Drain timeout ({drain_timeout}s) - returning best result early")
                        return best_batch_result
                    time.sleep(0.001)  # 1ms backoff
                    continue

                # Reset timeout on each result received
                last_result_time = time.time()

                # Got a result - process it immediately
                job_id = result['job_id']

                if job_id not in job_to_idx:
                    continue  # Unexpected job (from previous batch?)

                idx = job_to_idx[job_id]
                if idx not in remaining_jobs:
                    continue  # Already processed

                remaining_jobs.remove(idx)
                attempts += 1

                # Convert result to SampleSet (with timing)
                t0 = time.time()
                samples = self.async_sampler.kernel.get_samples(result)
                energies = self.async_sampler.kernel.get_energies(result)
                t1 = time.time()
                sampleset = dimod.SampleSet.from_samples(
                    samples.astype(np.int8),
                    vartype='SPIN',
                    energy=energies,
                    info={'job_id': job_id, 'min_energy': result['min_energy']}
                )
                t2 = time.time()

                # Filter and evaluate (pass pre-computed h, J to avoid regeneration)
                filtered_sampleset = self._filter_samples_for_sparse_topology(sampleset)
                t3 = time.time()
                mining_result = evaluate_sampleset(
                    filtered_sampleset,
                    current_requirements,
                    self.nodes,
                    self.edges,
                    nonces[idx],
                    salts[idx],
                    prev_timestamp,
                    start_time,
                    self.miner_id,
                    self.miner_type,
                    h=h_dicts[idx],
                    J=J_dicts[idx]
                )
                t4 = time.time()

                # Log timing if any step took > 200ms (baseline is ~50ms for SampleSet overhead)
                get_time = (t1 - t0) * 1000
                sampleset_time = (t2 - t1) * 1000
                filter_time = (t3 - t2) * 1000
                eval_time = (t4 - t3) * 1000
                total_time = (t4 - t0) * 1000
                if total_time > 200:
                    self.logger.warning(
                        f"Slow processing ({total_time:.0f}ms): "
                        f"get={get_time:.0f}ms, sampleset={sampleset_time:.0f}ms, "
                        f"filter={filter_time:.0f}ms, eval={eval_time:.0f}ms"
                    )

                if mining_result:
                    self.logger.info(f"✅ Found valid block after {attempts} attempts!")
                    self.logger.info(f"   Energy: {mining_result.energy:.1f}")
                    self.logger.info(f"   Diversity: {mining_result.diversity:.3f}")
                    self.logger.info(f"   Solutions: {mining_result.num_valid}")
                    # Track best result in this batch (lowest energy)
                    if best_batch_result is None or mining_result.energy < best_batch_result.energy:
                        best_batch_result = mining_result

                # Batch complete - return best result if we found one
                if best_batch_result:
                    if not drain or attempts % batch_size == 0:
                        return best_batch_result

            # Log progress every 10 batches
            if attempts % (10 * batch_size) == 0:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"Attempts: {attempts} ({rate:.1f}/s), elapsed: {elapsed:.1f}s | "
                    f"Sweeps: {current_num_sweeps}/{max_num_sweeps}, Reads: {num_reads}"
                )

        self.logger.info("Mining stopped by stop_event")
        return None
