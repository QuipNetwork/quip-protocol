"""GPU miner using CUDA persistent kernel via CudaSASamplerAsync."""
from __future__ import annotations

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
from GPU.cuda_kernel import CudaKernelRealSA
from GPU.cuda_sa import CudaKernelAdapter, CudaSASamplerAsync
from dwave_topologies import DEFAULT_TOPOLOGY


def adapt_parameters(difficulty_energy: float, min_diversity: float, min_solutions: int) -> dict:
    """Adapt mining parameters based on difficulty.

    Args:
        difficulty_energy: Target energy threshold
        min_diversity: Minimum solution diversity required
        min_solutions: Minimum number of valid solutions required

    Returns:
        Dictionary with num_reads and num_sweeps parameters
    """
    # Scale parameters based on difficulty
    # Higher difficulty (lower energy) = more sweeps
    if difficulty_energy < -15500:
        num_sweeps = 2048
        num_reads = 150
    elif difficulty_energy < -15400:
        num_sweeps = 1024
        num_reads = 100
    elif difficulty_energy < -15000:
        num_sweeps = 512
        num_reads = 100
    else:
        num_sweeps = 256
        num_reads = 64

    return {
        'num_sweeps': num_sweeps,
        'num_reads': num_reads,
        'num_sweeps_per_beta': 1  # Use 1 sweep per beta (total sweeps = num_sweeps)
    }


class CudaMiner(BaseMiner):
    """CUDA GPU miner using persistent kernel for high-throughput mining.

    Uses CudaSASamplerAsync which wraps the persistent CUDA kernel.
    The kernel runs continuously on the GPU, processing jobs from a ring buffer.
    This eliminates kernel launch overhead and enables high job throughput.
    """

    def __init__(self, miner_id: str, device: str = "0", **cfg):
        """Initialize CUDA miner.

        Args:
            miner_id: Unique identifier for this miner
            device: CUDA device ID (default "0")
            **cfg: Additional configuration parameters
        """
        # Set CUDA device BEFORE creating any CUDA objects
        try:
            import cupy as cp
            device_id = int(device)
            cp.cuda.Device(device_id).use()
        except Exception as e:
            print(f"Warning: Failed to set CUDA device {device}: {e}")

        # Get topology from DEFAULT_TOPOLOGY first (needed for BaseMiner)
        self.nodes = list(DEFAULT_TOPOLOGY.graph.nodes)
        self.edges = list(DEFAULT_TOPOLOGY.graph.edges)

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
            def __init__(self, nodes, edges):
                self.nodes = nodes
                self.edges = edges
                self.nodelist = nodes
                self.edgelist = edges
                self.properties = {'topology': 'Advantage2'}
                self.sampler_type = "cuda-persistent"

            def sample_ising(self, h, J, **kwargs):
                """Dummy sample_ising - not used in CudaMiner."""
                raise NotImplementedError("CudaMiner handles sampling directly")

        minimal_sampler = SamplerInterface(self.nodes, self.edges)

        # Initialize base miner (sets up logger, miner_id, etc.)
        super().__init__(miner_id, minimal_sampler)

        self.miner_type = "GPU-CUDA-Persistent"
        self.device = device

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

            # Reset CUDA device
            import cupy as cp
            cp.cuda.Device(int(self.device)).use()
            cp.cuda.runtime.deviceReset()
            self.logger.info(f"CUDA device {self.device} reset completed")

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
        filtered_samples = []
        for sample in sampleset.record.sample:
            # Extract only values at node indices that exist in topology
            filtered_sample = np.array([sample[node] for node in self.nodes], dtype=np.int8)
            filtered_samples.append(filtered_sample)

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

        Returns:
            MiningResult if successful, None if stopped or failed
        """
        # Set device context
        try:
            import cupy as cp
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
        params = adapt_parameters(difficulty_energy, min_diversity, min_solutions)
        num_reads = params['num_reads']
        num_sweeps = params['num_sweeps']
        num_sweeps_per_beta = params['num_sweeps_per_beta']

        self.logger.info(f"Adaptive params: {num_sweeps} sweeps, {num_reads} reads")

        # Batch size: number of jobs to run in parallel
        # Use number of SMs (streaming multiprocessors) for optimal parallelism
        batch_size = self.kernel.num_blocks  # Typically 48 for most GPUs

        # Compute N for sparse topology
        N = max(max(self.nodes), max(max(i, j) for i, j in self.edges)) + 1

        attempts = 0
        while not stop_event.is_set():
            # Generate batch of Ising problems
            h_list = []
            J_list = []
            salts = []
            nonces = []

            for _ in range(batch_size):
                salt = random.randbytes(32)
                nonce = ising_nonce_from_block(
                    prev_block.hash, node_info.miner_id, cur_index, salt
                )

                h_dict, J_dict = generate_ising_model_from_nonce(nonce, self.nodes, self.edges)

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

            # Submit batch to GPU (async, returns immediately)
            try:
                samplesets = self.async_sampler.sample_ising(
                    h_list=h_list,
                    J_list=J_list,
                    num_reads=num_reads,
                    num_betas=num_sweeps,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    edges=self.edges,
                    timeout=300.0  # 5 minute timeout for production problems
                )
            except Exception as e:
                self.logger.error(f"Sampling error: {e}")
                continue

            # Evaluate each sampleset
            for i, sampleset in enumerate(samplesets):
                if stop_event.is_set():
                    break

                attempts += 1

                # Filter samples to match expected topology size
                filtered_sampleset = self._filter_samples_for_sparse_topology(sampleset)

                # Evaluate against requirements
                mining_result = evaluate_sampleset(
                    filtered_sampleset,
                    current_requirements,
                    self.nodes,
                    self.edges,
                    nonces[i],
                    salts[i],
                    prev_timestamp,
                    start_time,
                    self.miner_id,
                    self.miner_type
                )

                if mining_result:
                    self.logger.info(f"✅ Found valid block after {attempts} attempts!")
                    self.logger.info(f"   Energy: {mining_result.energy:.1f}")
                    self.logger.info(f"   Diversity: {mining_result.diversity:.3f}")
                    self.logger.info(f"   Solutions: {mining_result.num_valid}")
                    return mining_result

            # Log progress every 10 batches
            if attempts % (10 * batch_size) == 0:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                self.logger.info(f"Attempts: {attempts} ({rate:.1f}/s), elapsed: {elapsed:.1f}s")

        self.logger.info("Mining stopped by stop_event")
        return None
