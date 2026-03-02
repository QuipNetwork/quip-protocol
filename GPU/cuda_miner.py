"""GPU miner using CUDA persistent kernel via CudaSASamplerAsync."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from GPU.cuda_kernel import CudaKernelRealSA
from GPU.cuda_sa import CudaKernelAdapter, CudaSASamplerAsync
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaMiner(BaseMiner):
    """CUDA GPU miner using persistent kernel for high-throughput mining.

    Uses CudaSASamplerAsync which wraps the persistent CUDA kernel.
    The kernel runs continuously on the GPU, processing jobs from a ring buffer.
    This eliminates kernel launch overhead and enables high job throughput.
    """

    # CUDA GPU calibration ranges
    ADAPT_MIN_SWEEPS = 256
    ADAPT_MAX_SWEEPS = 2048
    ADAPT_MIN_READS = 64
    ADAPT_MAX_READS = 256  # CUDA max_threads_per_job limit
    ADAPT_EXTRA_PARAMS = {'num_sweeps_per_beta': 1}

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
        # Precompute edge-to-index mapping for array conversion
        self._edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}

        # Compute N for sparse topology
        self._N = max(max(self.nodes), max(max(i, j) for i, j in self.edges)) + 1

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

    def _pre_mine_setup(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> bool:
        """Set CUDA device context before mining."""
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(f"Failed to set device context: {e}")
            return False
        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> dimod.SampleSet:
        """Submit a single Ising problem to the persistent CUDA kernel."""
        # Convert dict h to array (sparse indexing with N=max_node_id+1)
        h_arr = np.zeros(self._N, dtype=np.float32)
        for node, val in h.items():
            h_arr[node] = val

        # Convert dict J to edge-indexed array
        J_arr = np.zeros(len(self.edges), dtype=np.float32)
        for (i, j), val in J.items():
            if (i, j) in self._edge_to_idx:
                J_arr[self._edge_to_idx[(i, j)]] = val
            elif (j, i) in self._edge_to_idx:
                J_arr[self._edge_to_idx[(j, i)]] = val

        # Submit single job to GPU
        job_ids = self.async_sampler.sample_ising_async(
            h_list=[h_arr],
            J_list=[J_arr],
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
            edges=self.edges,
        )

        # Wait for the result
        while True:
            result = self.async_sampler.kernel.try_dequeue_result()
            if result is not None and result['job_id'] == job_ids[0]:
                break
            time.sleep(0.001)

        # Convert to SampleSet
        samples = self.async_sampler.kernel.get_samples(result)
        energies = self.async_sampler.kernel.get_energies(result)
        return dimod.SampleSet.from_samples(
            samples.astype(np.int8),
            vartype='SPIN',
            energy=energies,
            info={'job_id': result['job_id'], 'min_energy': result['min_energy']},
        )

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology."""
        return self._filter_samples_for_sparse_topology(sampleset)
